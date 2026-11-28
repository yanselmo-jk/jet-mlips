"""ASE side of the shared graph path (jet_mlip/mliap.py): a family's ASE calculator hands over its pair list and gets
the step's atomic energies, forces and virial from a CUDA graph, captured once per (atoms, pair capacity) and then
replayed: the ML-IAP scheme without LAMMPS. Every atom is local (pair k = (i, j), rij = x_j - x_i + shift @ cell), so
there is no ghost row to fold and the model runs on the same static buffers as under LAMMPS.

    g = AseGraph(forward, device, r_cut)          # once per calculator
    if g.due(len(atoms), i.numel()):              # else: the calculator's eager path
        e, f, w = g.compute(types, i, j, rij, pos)

forward(g, vec) -> per-atom energies (the first N rows are used) from the edge vectors vec (ecap, 3), the body of the
family's ML-IAP `_forward`. It reads g.s_types (N,) int64, g.s_ei (2, ecap: row 0 = i, row 1 = j), the convolutions'
CSR by row 0 (g.s_rowptr, g.s_perm: padded edges are masked out of it), g.nlg = (N, 0) and g.lmp_stand_in (an identity
exchange). Padded edges carry the vector (r_cut, 0, 0), where W = 0.
alloc(g), optional: the family's own static buffers, made whenever the capacities g.key = (N, N, ecap, ...) are new (a
tensor created inside forward would be captured as a host copy).
prep(g), optional: fills the buffers that belong to a pair list (an order of g.s_types, ...): captured in the pair-list
graph, which is replayed only when the list or the elements change, not with every step.
compile(), optional: -> the same forward with its glue under torch.compile (static shapes: one compile per capacity
key). A capture starts on the plain forward and is redone on the compiled one after compile_after seconds: a compile
takes longer than a short run and pays off in a long one. A run that ends right after the compile has lost compile
time / compile_after of its time (600 s: 1.4 % for an 8 s compile). JET_ASE_COMPILE overrides it (seconds; 0: at the
step after the capture; negative: never). If the compile fails, the plain forward stays.

The first call with an atom count stays eager: a scan over structures pays no capture, an MD or a relaxation captures
at its second step (two warm-up steps and the capture, then replays; the capture keeps the step's work memory).
JET_ASE_GRAPH=0: always eager (A/B). JET_ASE_SORT=0: the node rows in the caller's atom order. JET_MLIAP_LOG=1 prints
every capture, JET_MLIAP_CHECK=N checks every N-th replay against the eager step.
"""
import os
import time
import warnings

import torch

from . import ops
from .mliap import GraphMLIAP


class _Pairs:
    """GraphMLIAP.graph_forces' lmp_data for an ASE pair list: every atom local, so no ghost and no exchange probe."""

    eflag = 0   # the energies are read from the graph's buffer (no .item() sync and no copy here)

    def update_pair_forces_gpu(self, f):
        self.f = f


class AseGraph(GraphMLIAP):
    pad_nodes = False   # node rows = atoms (padded edges sit on real rows, masked out of the CSR)

    def __init__(self, forward, device, r_cut, vec_dtype=torch.float32, index_dtype=torch.int64, alloc=None, prep=None,
                 compile=None, compile_after=600.0):
        self.forward = self.plain = forward
        self.alloc, self.prep, self.compile, self.compile_after, self.fast = alloc, prep, compile, compile_after, None
        self.device, self.r_cut, self.vec_dtype, self.index_dtype = device, float(r_cut), vec_dtype, index_dtype
        self.data, self._n, self._t0, self._order = _Pairs(), -1, 0.0, None

    def _multi(self):
        return False

    def __reduce__(self):
        """A copied or pickled calculator starts without a graph and makes its own: the captures and the forward's
        closure belong to this one's model."""
        return type(None), ()

    def due(self, n, npairs):
        """True from the second call in a row with n atoms on (and more than one pair, and not JET_ASE_GRAPH=0)."""
        seen, self._n = self._n, n
        return seen == n and npairs > 1 and os.environ.get('JET_ASE_GRAPH', '1') != '0'

    def _alloc(self, key):
        super()._alloc(key)
        self.s_rowptr = torch.zeros(key[1] + 1, dtype=torch.int32, device=self.device)
        self.s_perm = torch.zeros(key[2], dtype=torch.int32, device=self.device)
        if self.alloc is not None:
            self.alloc(self)
        self.forward, self._t0 = self.plain, time.monotonic()   # new capacities: a new capture, uncompiled at first

    def _prep(self):
        """The convolutions' CSR (dst = row 0) belongs to the pair list: one per list, not one per layer and step."""
        super()._prep()
        ops.EDGE_MASK = self.s_mask
        try:
            rowptr, perm = ops.build_csr(self.s_ei[0], self.key[1])
        finally:
            ops.EDGE_MASK = None
        self.s_rowptr.copy_(rowptr)
        self.s_perm.copy_(perm)
        if self.prep is not None:
            self.prep(self)

    def _forward(self, vec):
        return self.forward(self, vec)

    def _rows(self, pos, i, j):
        """The node row of every atom, (N,) int32, in spatial order (bins of half the cutoff, as LAMMPS sorts its
        atoms), or None: the caller's numbering. Rows that a convolution gathers together then lie close: H200,
        Nequix, 45k atoms in random order 146.4 -> 128.7 ms per call, 5632 atoms 19.35 -> 19.13. Taken only where it
        halves the typical row distance of a pair: atoms numbered spatially already (a LAMMPS data file) keep their
        ascending pair list (sorting those: +0.9 % at 45k atoms). Renewed every 100 replays (the atoms move).
        JET_ASE_SORT=0: never.
        ponytail: bins of the positions as given; wrap them with the cell if atoms that left it by box lengths matter."""
        n = pos.shape[0]
        if os.environ.get('JET_ASE_SORT', '1') == '0':
            return None
        if self._order is None or self._order[0].numel() != n or self.n_replays % 100 == 0:
            b = ((pos - pos.min(0).values) / (self.r_cut / 2)).long()
            perm = torch.argsort((b[:, 2] * (b[:, 1].max() + 1) + b[:, 1]) * (b[:, 0].max() + 1) + b[:, 0])
            rows = torch.empty(n, dtype=torch.int32, device=pos.device)
            rows[perm] = torch.arange(n, dtype=torch.int32, device=pos.device)
            near = 2 * (rows[i] - rows[j]).abs().float().median() < (i - j).abs().float().median()
            self._order = perm, rows if near else None
        return self._order[1]

    def compute(self, types, i, j, rij, pos=None):
        """One step. types (N,) the model's type index of every atom; i, j (E,) int64 and rij (E, 3) = x_j - x_i +
        shift @ cell: the full pair list (both directions), grouped by i; pos (N, 3), optional: the positions (any
        periodic image), for the node rows' spatial order (_rows).
        -> (e (N,), F (N, 3), w (3, 3)) in fp64 on the device: atomic energies (read them before the next call: they
        may be the graph's buffer), forces, and w = sum_k rij_k (x) dE/drij_k, whose symmetric part over the volume is
        the stress."""
        with torch.cuda.device(self.device):   # the streams and the captures are the current device's
            return self._compute(types, i, j, rij, pos)

    def _compute(self, types, i, j, rij, pos):
        d, n = self.data, types.numel()
        if self.key is not None and self.key[0] != n:
            self.caps = None   # another structure: its own capacities (they only grow otherwise)
        rows = None if pos is None else self._rows(pos, i, j)
        if rows is None:
            d.pair_i, d.pair_j, d.elems = i.int(), j.int(), types.long()
        else:   # only the numbering: every atom's pairs stay together, in the caller's pair order (as rij and f)
            d.pair_i, d.pair_j, d.elems = rows[i], rows[j], types.long()[self._order[0]]
        if self.key is not None and not torch.equal(d.elems, self.r_elems):
            self.last = None   # other elements on the pair list of the step before: only a new list loads them
        d.nlocal = d.ntotal = n
        d.npairs = i.numel()
        d.rij = rij.double().contiguous()
        captures = self.n_captures
        try:
            self.graph_forces(d)
        except RuntimeError as e:   # torch.compile failed in the warm-up steps: the plain forward, for good
            if self.forward is self.plain or not type(e.__cause__).__module__.startswith('torch._dynamo'):
                raise
            warnings.warn(f'jet_mlip.ase_graph: torch.compile failed ({type(e.__cause__).__name__}), running uncompiled')
            self.forward, self.compile, self.graph = self.plain, None, None
            self.graph_forces(d)
        if self.n_captures != captures:
            torch.cuda.empty_cache()   # the warm-up steps' blocks (their stream's pool): the graph holds its own
        if self.compile is not None and self.forward is self.plain \
                and time.monotonic() - self._t0 >= float(os.environ.get('JET_ASE_COMPILE', self.compile_after)) >= 0:
            self.fast = self.fast or self.compile()
            self.forward, self.graph = self.fast, None   # the next step captures the compiled forward
        f = d.f
        forces = torch.zeros(n, 3, dtype=f.dtype, device=f.device).index_add_(0, i, f).index_add_(0, j, -f)
        e = self.o_eatoms[:n]
        return e if rows is None else e[rows], forces, d.rij.t() @ f
