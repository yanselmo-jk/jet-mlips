"""Shared LAMMPS ML-IAP graph path for every jet family (the pair_e3gnn scheme inside ML-IAP).

Once per step the model runs on static device buffers and is replayed from a CUDA graph captured per
(nlocal, node bucket, edge bucket). Three things keep it close to pair_e3gnn:
  - ghost fold: the LAMMPS exchange is probed with the node indices, which gives every ghost's owner, and each edge
    is pointed at the owner. The graph has local nodes only; the pair forces still go to LAMMPS' ghost j and its
    reverse comm. Several MPI ranks: the probe carries
    (rank, local index), ghosts owned by another rank become one "remote" row per owner atom after the local rows, and
    the stand-in's per-layer exchange fills them from their owners with NCCL inside the graph (backward: the gradient
    goes back to the owner), so every atom's features come from its owner's complete neighbourhood.
  - padding never computed: padded edges (vector (r_cut, 0, 0), ends spread over the local rows) are masked out of the
    conv CSR and sorted last (ops.EDGE_MASK -> ops.build_csr), so the kernels stop at the real edge count. Scatters
    outside the CSR (upstream convs, MACE density, ZBL) add W(r_cut) = 0, spread so no row serializes the atomics.
  - one capture, then replay: no per-layer exchange, no launches from Python inside the model.
A family wrapper subclasses its upstream ML-IAP class and this mixin, sets self.device, self.r_cut, self.vec_dtype,
self.index_dtype and implements _forward(vec) -> per-node energies (first nlocal rows used), reading self.s_ei
(2, ecap: row 0 = pair_i, row 1 = owner of pair_j), self.s_types, self.nlg (nlocal, ncap - nlocal) and
self.lmp_stand_in. JET_MLIAP_HEADROOM (1.1): capacity growth factor (every growth re-captures). JET_MLIAP_CHECK=N
(debug, off by default): every N-th replay and every replay right after nlocal changed (atoms migrated), the step is
recomputed eagerly and must match the replay, and the eager maps are checked against LAMMPS' own elements (every edge's
row carries the element of its LAMMPS atom j; on several ranks the exchange delivers each remote row's element), else
the run stops. The second check is what catches a stale owner map: it would make the eager step wrong the same way.
"""
import os
import sys

import torch

from . import ops

EDGE_BUCKET, NODE_BUCKET = 8192, 256
# per-peer exchange rows: every padded row crosses PCIe once per exchange (6 per step for 4 layers), so a tighter cap
# than the node buffers'; a migration that overflows it re-captures every rank
PEER_BUCKET, PEER_HEADROOM = 128, 1.05
HEADROOM = float(os.environ.get('JET_MLIAP_HEADROOM', '1.1'))


def _bucket(n, b):
    return max(1, (n + b - 1) // b) * b


def nranks():
    # ponytail: launcher env vars, not the LAMMPS communicator (one LAMMPS instance per MPI world)
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE', os.environ.get('PMI_SIZE', '1')))


def single_rank():
    return nranks() == 1


_CODE = float(2 ** 32)   # probe value of a row = rank * _CODE + local index (exact in fp64)


def _dist():
    """torch.distributed over the LAMMPS ranks (NCCL, this rank's current CUDA device = the Kokkos one)."""
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')   # ponytail: one node; set MASTER_ADDR / PORT for several
        os.environ.setdefault('MASTER_PORT', '29531')
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', os.environ.get('PMI_RANK', '0')))
        dist.init_process_group('nccl', rank=rank, world_size=nranks(),
                                device_id=torch.device('cuda', torch.cuda.current_device()))
    return dist


class LammpsStandIn:
    """lmp_data stand-in inside the graph. One rank: after the fold no ghost node is left, padding rows own themselves
    and the exchange is an identity gather (kept for models that call it per layer). Several ranks (xchg): the remote
    rows are refreshed from their owners (send_idx: rows each peer asked for, recv_pos: the rows they land in, pads go
    to a spare padding row; ok masks the pads in the backward)."""

    def __init__(self, owner, nlocal, ntotal, xchg=None, sel=None, splits=None):
        # sel / splits: the blocks of the other ranks only (no rank requests its own atoms; the self block would be an
        # all-padding copy)
        self.owner, self.nlocal, self.ntotal, self.xchg, self.sel, self.splits = owner, nlocal, ntotal, xchg, sel, splits

    def forward_exchange(self, x, out, vec_len):
        if self.xchg is None:
            out.copy_(x[self.owner])
            return
        send_idx, recv_pos, _ = (b.index_select(0, self.sel) for b in self.xchg)
        out.copy_(x)
        got = torch.empty((send_idx.numel(),) + x.shape[1:], dtype=x.dtype, device=x.device)
        _dist().all_to_all_single(got, x.index_select(0, send_idx).contiguous(), self.splits, self.splits)
        out.index_copy_(0, recv_pos, got)

    def reverse_exchange(self, g, gout, vec_len):
        if self.xchg is None:
            gout.zero_()
            gout.index_add_(0, self.owner, g)
            return
        send_idx, recv_pos, ok = (b.index_select(0, self.sel) for b in self.xchg)
        back = g.index_select(0, recv_pos) * ok.view((-1,) + (1,) * (g.dim() - 1)).to(g.dtype)
        gout.copy_(g)
        gout.index_fill_(0, recv_pos, 0)   # the remote rows' gradient belongs to their owners
        got = torch.empty_like(back)
        _dist().all_to_all_single(got, back.contiguous(), self.splits, self.splits)
        gout.index_add_(0, send_idx, got)


class GraphMLIAP:
    caps, graph, key = None, None, None
    n_captures = n_replays = n_checks = 0   # read by the family tests
    _last_nlocal = -1

    def _forward(self, vec):
        raise NotImplementedError

    def _cap(self, name, n, bucket, headroom=None):
        """Static size for n: grows (headroom) only when exceeded, never shrinks (a heating MD keeps adding pairs)."""
        if self.caps is None:
            self.caps = {}
        cap = self.caps.get(name, 0)
        if n > cap:
            cap = self.caps[name] = _bucket(int(n * (headroom or HEADROOM)) + 1, bucket)
        return cap

    def _alloc(self, key):
        nlocal, ncap, ecap, tcap, pcap = key
        dev = self.device
        self.graph, self.key = None, key
        self.s_types = torch.zeros(ncap, dtype=torch.int64, device=dev)
        self.s_vec = torch.zeros(ecap, 3, dtype=self.vec_dtype, device=dev)
        self.s_ei = torch.zeros(2, ecap, dtype=self.index_dtype, device=dev)
        self.s_mask = torch.zeros(ecap, dtype=torch.int32, device=dev)
        self.o_eatoms = torch.zeros(ncap, dtype=torch.float64, device=dev)
        self.o_pair_f = torch.zeros(ecap, 3, dtype=torch.float64, device=dev)
        # raw LAMMPS inputs, copied in eagerly (their device pointers change every step); _prep turns them into the
        # model buffers inside the graph
        self.r_elems = torch.zeros(nlocal, dtype=torch.int64, device=dev)
        self.r_lmask = torch.ones(nlocal, dtype=torch.float64, device=dev)   # several ranks: real local rows (nlocal is a capacity)
        self.r_rtypes = torch.zeros(ncap, dtype=torch.int64, device=dev)   # remote rows' elements (several ranks)
        self.r_rij = torch.zeros(ecap, 3, dtype=torch.float64, device=dev)
        self.r_pi = torch.zeros(ecap, dtype=torch.int64, device=dev)
        self.r_pj = torch.zeros(ecap, dtype=torch.int64, device=dev)
        self.r_n = torch.zeros((), dtype=torch.int64, device=dev)   # npairs
        self.r_owner = torch.arange(tcap, dtype=torch.int64, device=dev)
        xchg = sel = splits = None
        if pcap:   # several ranks: the static exchange maps (filled per step by _fill_xchg)
            w, r = _dist().get_world_size(), _dist().get_rank()
            sel = torch.cat([torch.arange(0, r * pcap), torch.arange((r + 1) * pcap, w * pcap)]).to(dev)
            splits = [0 if p == r else pcap for p in range(w)]
            xchg = (torch.zeros(w * pcap, dtype=torch.int64, device=dev),
                    torch.full((w * pcap,), ncap - 1, dtype=torch.int64, device=dev),
                    torch.zeros(w * pcap, dtype=torch.float32, device=dev))
        self.r_edge = torch.arange(ecap, device=dev)
        self.r_pad = torch.tensor([self.r_cut, 0.0, 0.0], dtype=torch.float64, device=dev)   # W = 0
        self.nlg = (nlocal, ncap - nlocal)
        self.lmp_stand_in = LammpsStandIn(torch.arange(ncap, device=dev), nlocal, ncap, xchg, sel, splits)
        self.n_remote = 0
        torch.cuda.empty_cache()   # a dead graph's private pool is only freed on an OOM retry otherwise

    def _prep(self):
        """Model buffers from the raw inputs (captured): padded edges get vector (r_cut, 0, 0), both ends on local row
        e % nlocal, and are masked out of the conv CSR."""
        nlocal, ncap = self.key[0], self.key[1]
        real = self.r_edge < self.r_n
        self.s_types[:nlocal] = self.r_elems
        # remote rows (several ranks): their elements; padding rows: a valid type
        self.s_types[nlocal:] = self.r_rtypes[nlocal:] if self.key[4] else self.r_elems[0]
        self.s_vec.copy_(torch.where(real.unsqueeze(1), self.r_rij, self.r_pad))
        pad = self.r_edge % nlocal   # spread over the local rows: unmasked scatters (upstream convs) do not pile onto one row
        self.s_ei[0].copy_(torch.where(real, self.r_pi, pad))
        self.s_ei[1].copy_(torch.where(real, self.r_owner[self.r_pj.clamp(0, self.r_owner.numel() - 1)], pad))   # ghost fold
        self.s_mask.copy_(real)

    def _step(self):
        nlocal = self.key[0]
        vec = self.s_vec.detach().requires_grad_(True)
        ops.EDGE_MASK = self.s_mask
        try:
            e = self._forward(vec).view(-1)[:nlocal]
            if self.key[4]:   # several ranks: the local rows past the real nlocal are dummies
                e = e * self.r_lmask.to(e.dtype)
        finally:
            ops.EDGE_MASK = None
        f = torch.autograd.grad([e.sum()], [vec])[0]   # f = dE/d rij: LAMMPS sign convention, no flip
        self.o_eatoms[:nlocal].copy_(e.detach())
        self.o_pair_f.copy_(f.detach())

    def _remote_map(self, codes, base, ref):
        """Several ranks, eager once per step: ghost codes (rank * _CODE + index) -> self.r_owner rows (own periodic
        images -> their local row, other ranks' atoms -> one remote row each), the remote rows' elements, and the
        exchange maps (all_to_all of the requested indices). Returns (n_remote, max requests per peer over all ranks)."""
        dist = _dist()
        rank, w = dist.get_rank(), dist.get_world_size()
        q = torch.div(codes, _CODE, rounding_mode='floor').long()
        i = (codes - q.double() * _CODE).long()
        mine = q == rank
        owner = torch.where(mine, i, 0)
        far = ~mine & ref   # other ranks' atoms some local edge reads (the skin shell beyond r_cut is not exchanged)
        uq, inv = torch.unique(codes[far], return_inverse=True)   # sorted: per-peer blocks in index order
        owner[far] = base + inv
        cnt = torch.bincount(torch.div(uq, _CODE, rounding_mode='floor').long(), minlength=w)
        need = cnt.max().clone()
        dist.all_reduce(need, op=dist.ReduceOp.MAX)
        return owner, uq, inv, far, cnt, int(need)

    def _fill_xchg(self, uq, cnt, pcap, base):
        """Static exchange maps for pcap requests per peer (pads: index 0 / the spare last row, ok = 0)."""
        dist, dev = _dist(), self.device
        send_idx, recv_pos, ok = self.lmp_stand_in.xchg
        w = dist.get_world_size()
        start = torch.cumsum(cnt, 0) - cnt
        q = torch.div(uq, _CODE, rounding_mode='floor').long()
        k = torch.arange(uq.numel(), device=dev) - start[q]   # position inside the peer's block
        slot = q * pcap + k
        req = torch.zeros(w * pcap, dtype=torch.int64, device=dev)
        req[slot] = (uq - q.double() * _CODE).long()
        recv_pos.fill_(self.key[1] - 1)
        recv_pos[slot] = base + torch.arange(uq.numel(), device=dev)
        ok.zero_()
        ok[slot] = 1.0
        dist.all_to_all_single(send_idx, req)   # block p: the local rows peer p asked for

    def _check_due(self, nlocal, multi):
        """JET_MLIAP_CHECK=N: every N-th replay and right after a migration (nlocal changed on any rank)."""
        n = int(os.environ.get('JET_MLIAP_CHECK', '0'))
        moved, self._last_nlocal = nlocal != self._last_nlocal, nlocal
        if not n:
            return False
        due = torch.tensor(int(moved or self.n_replays % n == 0), device=self.device)
        if multi:   # the eager step runs collectives: every rank checks or none
            _dist().all_reduce(due, op=_dist().ReduceOp.MAX)
        return bool(due.item())

    def _check(self, npairs, nlocal, lmp_data):
        """The replayed step against the same step run eagerly (same buffers, no graph): pair forces and energies; then
        the maps against LAMMPS' elements."""
        rank = _dist().get_rank() if not single_rank() else 0
        elems = torch.as_tensor(lmp_data.elems).to(self.device).long()
        pj = self.r_pj[:npairs]
        bad = (self.s_types[self.r_owner[pj]] != elems[pj]).sum().item()   # each edge's row = its LAMMPS atom j's element
        if self.key[4]:   # the exchange itself: send the local elements, the remote rows must receive their LAMMPS element
            x = self.r_rtypes.new_zeros(self.key[1], 1, dtype=torch.float64)
            x[:nlocal, 0] = elems[:nlocal].double()
            out = torch.empty_like(x)
            self.lmp_stand_in.forward_exchange(x, out, 1)
            rows = torch.arange(self.key[0], self.key[0] + self.n_remote, device=self.device)
            bad += (out[rows, 0].long() != self.r_rtypes[rows]).sum().item()
        f, e = self.o_pair_f[:npairs].clone(), self.o_eatoms[:nlocal].clone()
        self._prep()
        self._step()
        df = (self.o_pair_f[:npairs] - f).abs().max().item()
        de = (self.o_eatoms[:nlocal] - e).abs().max().item()
        self.n_checks += 1
        tol = 1e-4 * f.abs().max().item() + 1e-5
        fail = torch.tensor([bad, int(df > tol or de > 1e-4)], device=self.device)
        if not single_rank():   # every rank stops together (one rank raising alone leaves the others in a collective)
            _dist().all_reduce(fail, op=_dist().ReduceOp.MAX)
        if fail.any():
            msg = (f'{type(self).__name__}: JET_MLIAP_CHECK failed (rank {rank}, replay {self.n_replays}, nlocal {nlocal}): '
                   f'{bad} edge / remote rows with the wrong LAMMPS element here, graph vs eager max|dF| {df:.3e} eV/A, '
                   f'max|dE_atom| {de:.3e} eV (max over ranks: {fail.tolist()})')
            if single_rank():
                raise RuntimeError(msg)
            # several ranks: LAMMPS' error path and destroy_process_group both hang with the NCCL ops held by the CUDA
            # graphs, so every rank (all saw the same verdict) leaves at once and mpirun ends the job
            print(msg, file=sys.stderr, flush=True)
            os._exit(3)

    def graph_forces(self, lmp_data):
        """The whole step for a CUDA device; the caller checks npairs > 1. Several ranks: every rank takes the same
        capture decision (the warm-up steps and the replays both run collectives)."""
        nlocal, ntotal, npairs = lmp_data.nlocal, lmp_data.ntotal, lmp_data.npairs
        multi = not single_rank()
        torch.cuda.synchronize()   # Kokkos writes the LAMMPS arrays on another stream
        # several ranks: atoms migrate at every neighbour rebuild, so the local block is a capacity too (dummy rows
        # masked out of the energy) and the remote rows start after it; one rank: exactly nlocal
        base = self._cap('local', nlocal, NODE_BUCKET) if multi else nlocal
        nrows, pneed, remote = base, 0, None
        if multi or ntotal > nlocal:   # probe the LAMMPS exchange: ghost rows come back as their owner's code
            tcap = self._cap('total', ntotal, NODE_BUCKET)
            probe = torch.arange(tcap, dtype=torch.float64, device=self.device).unsqueeze(1)
            if multi:
                probe += _dist().get_rank() * _CODE
            out = torch.empty_like(probe)
            torch.cuda.synchronize()   # Kokkos exchanges on its own stream: probe written before, out read after
            lmp_data.forward_exchange(probe[:ntotal], out[:ntotal], 1)
            torch.cuda.synchronize()
            codes = out[nlocal:ntotal, 0]
            if multi:
                pj = torch.as_tensor(lmp_data.pair_j).to(self.device).long()
                ref = torch.zeros(ntotal - nlocal, dtype=torch.bool, device=self.device)
                ref[pj[pj >= nlocal] - nlocal] = True
                remote = self._remote_map(codes, base, ref)
                nrows, pneed = base + remote[1].numel() + 1, remote[5]   # + 1: the spare row the exchange pads hit
            else:
                remote = codes.long()
        key = (base, self._cap('atoms', nrows, NODE_BUCKET), self._cap('pairs', npairs, EDGE_BUCKET),
               self._cap('total', ntotal, NODE_BUCKET), self._cap('peer', pneed, PEER_BUCKET, PEER_HEADROOM) if multi else 0)
        changed = key != self.key
        if multi:   # one rank's re-capture forces all of them
            flag = torch.tensor(int(changed), device=self.device)
            _dist().all_reduce(flag, op=_dist().ReduceOp.MAX)
            if changed:
                self._alloc(key)
            elif flag.item():
                self.graph = None
        elif changed:
            self._alloc(key)
        elems = torch.as_tensor(lmp_data.elems)
        self.r_elems[:nlocal].copy_(elems[:nlocal])
        if multi:
            self.r_elems[nlocal:] = self.r_elems[0]
            self.r_lmask.fill_(1.0)
            self.r_lmask[nlocal:] = 0.0
        self.r_rij[:npairs].copy_(torch.as_tensor(lmp_data.rij))
        self.r_pi[:npairs].copy_(torch.as_tensor(lmp_data.pair_i))
        self.r_pj[:npairs].copy_(torch.as_tensor(lmp_data.pair_j))
        self.r_n.fill_(npairs)
        if multi:
            owner, uq, inv, far, cnt, _ = remote
            # nlocal moves when atoms migrate: rows that were ghosts last step may be local now, so own them again
            self.r_owner[:nlocal] = torch.arange(nlocal, device=self.device)
            self.r_owner[nlocal:ntotal] = owner
            self.r_rtypes.copy_(self.r_elems[0].expand_as(self.r_rtypes))
            self.r_rtypes[base + inv] = elems[nlocal:ntotal].to(self.device).long()[far]
            self.n_remote = uq.numel()
            self._fill_xchg(uq, cnt, key[4], base)
        elif remote is not None:
            self.r_owner[nlocal:ntotal] = remote
        if self.graph is None:
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):   # warm-up: lazy inits (cuBLAS, kernel attributes, compile) must not be captured
                        self._prep()
                        self._step()
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._prep()
                    self._step()
                self.graph = g
                self.n_captures += 1
                if os.environ.get('JET_MLIAP_LOG'):
                    print(f'{type(self).__name__}: capture {self.n_captures} key {self.key}', flush=True)
            except Exception as e:
                # no silent eager fallback: after a failed capture the eager step gave wrong energies (SevenNet, 2026-11-04)
                raise RuntimeError(f'{type(self).__name__}: CUDA graph capture failed; disable the graph path') from e
        else:
            self.graph.replay()
            self.n_replays += 1
            if self._check_due(nlocal, multi):
                self._check(npairs, nlocal, lmp_data)
        e_local = self.o_eatoms[:nlocal]
        if getattr(lmp_data, 'eflag', 1):   # LAMMPS reads it only then (pair_mliap_kokkos eflag_global): skip the sync (test stubs lack eflag)
            lmp_data.energy = e_local.sum().item()
        torch.as_tensor(lmp_data.eatoms).copy_(e_local)
        lmp_data.update_pair_forces_gpu(self.o_pair_f[:npairs])
