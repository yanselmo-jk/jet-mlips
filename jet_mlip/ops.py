"""Python entry points. Every op falls back to the torch reference when the CUDA
library is not built (slow; for tests and CPU)."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from . import is_available, reference
from .tables import ConvConfig

_KEY_TO_ID: Optional[Dict[str, int]] = None


def has_op(name: str) -> bool:
    """True if the CUDA op `jet_mlip::<name>` is registered in the built library."""
    return is_available() and hasattr(torch.ops.jet_mlip, name)


def config_id(cfg: ConvConfig) -> int:
    """Id of a config compiled into jet_mlip._C; raises with instructions if missing."""
    global _KEY_TO_ID
    if _KEY_TO_ID is None:
        n = torch.ops.jet_mlip.num_configs()
        _KEY_TO_ID = {torch.ops.jet_mlip.config_key(i): i for i in range(n)}
    if cfg.key not in _KEY_TO_ID:
        raise KeyError(f'jet: config not compiled: {cfg.key}. Add it to jet_mlip/configs.py and rebuild.')
    return _KEY_TO_ID[cfg.key]


EDGE_MASK: Optional[torch.Tensor] = None   # int32 (E,) 1 = real edge; set by jet_mlip.mliap during a padded step


def build_csr(dst: torch.Tensor, n_nodes: int):
    """-> (rowptr int32 [n_nodes+1], perm int32 [E]); stable dst grouping. With EDGE_MASK set (jet_mlip.mliap: padded
    edges) rowptr counts the real edges only and the padding sorts last: it is never visited."""
    m = EDGE_MASK
    if m is not None and m.numel() == dst.numel():
        d = dst.int()
        counts = torch.zeros(n_nodes, dtype=torch.int32, device=d.device)
        counts.index_add_(0, d.clamp(max=n_nodes - 1), m * (d < n_nodes))   # padding dst may lie past n_nodes
        rowptr = torch.zeros(n_nodes + 1, dtype=torch.int32, device=d.device)
        rowptr[1:] = counts.cumsum(0)
        return rowptr, torch.sort(torch.where(m.bool(), d, n_nodes), stable=True).indices.int()   # padding last
    if is_available():
        return torch.ops.jet_mlip.build_csr(dst, n_nodes)
    d = dst.long()
    counts = torch.bincount(d, minlength=n_nodes)
    rowptr = torch.zeros(n_nodes + 1, dtype=torch.long, device=d.device)
    rowptr[1:] = counts.cumsum(0)
    perm = torch.sort(d, stable=True).indices
    return rowptr.int(), perm.int()


def edge_build(x, cand_ptr, cand_j, center, owner, cutoff_sq: float, rowptr, nbr, ctr, vec, total) -> None:
    """pair_e3gnn's edge list on the GPU, written in place (CUDA-graph capturable: no sync, no
    data-dependent allocation). x (nall, 3) f64; cand_ptr (nlocal+1) / cand_j int32 = candidate
    neighbors (rows of x) per node in neighbor-list order; center (nlocal) int32 = row of node ii;
    owner (nall) int32 = node owning row j or -1. Edge (ii, j) iff owner[j] >= 0 and
    |x[j]-x[i]|^2 < cutoff_sq (double, bitwise the host's). Outputs: rowptr (nlocal+1) int32 CSR
    over ii; nbr / ctr (cap) int32 = owner[j] / ii (model edge_index[1] / [0]); vec (cap, 3) f32
    = x[j]-x[i]; total (1) int32 = edge count. Entries >= cap are not written: re-run with a
    larger cap when total > cap. rowptr and nbr are conv_fused's jet_rowptr / jet_src."""
    torch.ops.jet_mlip.edge_build(x, cand_ptr, cand_j, center, owner, cutoff_sq, rowptr, nbr, ctr, vec, total)


def cand_build(x, ncenters: int, cut_sq: float):
    """pair_e3gnn's neighbor (candidate) list on the GPU (one rank; replaces LAMMPS' CPU full list).
    x (nall, 3) f64, rows [0, ncenters) the centers (LAMMPS local atoms), the rest ghosts. Returns cand_ptr
    (ncenters+1) int32 and cand_j int32: per center ii the rows j != ii with |x[j]-x[ii]|^2 < cut_sq (cell
    list, deterministic order), edge_build's candidate input. Allocates and syncs: call per rebuild only."""
    return torch.ops.jet_mlip.cand_build(x, ncenters, cut_sq)


def force_reduce(dEdr, vec, rowptr, nbr, out) -> None:
    """pair_e3gnn's force and virial in one kernel (CUDA-graph capturable). dEdr / vec (cap, 3) f32 per edge, rowptr
    (nrows+1) int32 = edge_build's CSR over centers, nbr (cap) int32. Writes out (3n+8) f64: force rows
    F_i = sum_{e: ctr=i} dEdr - sum_{e: nbr=i} dEdr in [0, 3n), virial -sum voigt(vec, dEdr) in [3n+1, 3n+7)
    (order xx yy zz xy yz zx); out[3n] and out[3n+7] are zeroed for the caller."""
    torch.ops.jet_mlip.force_reduce(dEdr, vec, rowptr, nbr, out)


def species_linear_weight(tp):
    """(weight (S, P), paths) of species_linear from an e3nn FullyConnectedTensorProduct(irreps_in, S x 0e,
    irreps_out) with shared weights (sevenn's 'nequip' self-connection): path (in irrep -> same out irrep) block
    [s, u*V + w] = path_weight / sqrt(2l+1) * w[u, s, w], e3nn's normalization folded in. None if the TP is not of
    that form (in2 not one S x 0e, a non-'uvw' path, an irrep in two paths, more than 16 paths)."""
    i1, i2, io = tp.irreps_in1, tp.irreps_in2, tp.irreps_out
    if len(i2) != 1 or str(i2[0].ir) != '0e' or not tp.shared_weights or len(tp.instructions) > 16 \
            or not all(i.connection_mode == 'uvw' and i.has_weight for i in tp.instructions):
        return None
    s1, so = i1.slices(), io.slices()
    paths, blocks, seen = [], [], set()
    for ins, w in zip(tp.instructions, tp.weight_views()):
        if ('i', ins.i_in1) in seen or ('o', ins.i_out) in seen:
            return None
        seen |= {('i', ins.i_in1), ('o', ins.i_out)}
        d = i1[ins.i_in1].ir.dim
        if w.numel() > 0:
            paths += [s1[ins.i_in1].start, so[ins.i_out].start, i1[ins.i_in1].mul, io[ins.i_out].mul, d]
            blocks.append((w.detach() * (ins.path_weight / d ** 0.5)).transpose(0, 1).reshape(i2[0].mul, -1))
    W = torch.cat(blocks, 1) if blocks else tp.weight.new_zeros(i2[0].mul, 0)
    return W.contiguous(), paths


def species_linear(x, species, weight, paths: List[int], dim_out: int, bias=None):
    """out = e3nn TP(x, one_hot(species)) as one (V, U) matrix per species and irrep path (fp32 CUDA, CUDA-graph
    capturable; grad x only). weight, paths from species_linear_weight; species (N,) int in [0, S). species None:
    an e3nn Linear (weight, paths from linear_weight), bias (dim_out,) added to the columns the paths write."""
    return torch.ops.jet_mlip.species_linear(x, species, weight, paths, dim_out, bias)


def linear_weight(lin, modal: int = 0):
    """(weight (1, P), paths, bias table (modal, dim_out) or None) of species_linear for an e3nn o3.Linear with shared
    internal weights and no bias: block [0, u*V + w] = path_weight * w[u, w]. modal > 0: the last input irrep is
    sevenn's modal one-hot (modal x 0e appended to x by IrrepsLinear); its paths become the table (bias =
    one_hot(modal) @ table). None if an input or output irrep is in two x paths, or a modal path has no x path."""
    if any(i.i_in == -1 for i in lin.instructions) or not (lin.shared_weights and lin.internal_weights) \
            or len(lin.instructions) > 16:
        return None
    ii, io = lin.irreps_in, lin.irreps_out
    s1, so = ii.slices(), io.slices()
    mi = len(ii) - 1 if modal else -2
    if modal and (ii[mi].mul != modal or str(ii[mi].ir) != '0e'):
        return None
    paths, blocks, seen, table, modal_out = [], [], set(), None, []
    if modal:
        table = lin.weight.new_zeros(modal, io.dim)
    for _, ins, w in lin.weight_views(yield_instruction=True):
        w = w.detach() * ins.path_weight
        if ins.i_in == mi:
            table[:, so[ins.i_out]] = w
            modal_out.append(ins.i_out)
            continue
        if ('i', ins.i_in) in seen or ('o', ins.i_out) in seen:
            return None
        seen |= {('i', ins.i_in), ('o', ins.i_out)}
        if w.numel() > 0:
            paths += [s1[ins.i_in].start, so[ins.i_out].start, ii[ins.i_in].mul, io[ins.i_out].mul, ii[ins.i_in].ir.dim]
            blocks.append(w.reshape(1, -1))
    if any(('o', o) not in seen for o in modal_out):
        return None   # the kernel adds the bias only to columns a path writes
    W = torch.cat(blocks, 1) if blocks else lin.weight.new_zeros(1, 0)
    return W.contiguous(), paths, table


_ACTS = {'silu': 1, 'tanh': 2, 'sigmoid': 3}


def _act(a):
    """(code, cst) of one e3nn Activation entry: None or normalize2mom(f)."""
    if a is None:
        return 0, 1.0
    code = _ACTS.get(getattr(a.f, '__name__', None))
    if code is None:
        raise ValueError(f'jet gate: unsupported activation {a.f}')
    return code, 1.0 if a._is_id else float(a.cst)


def gate_tables(gate):
    """Per-column tables of the gate op (fwd_i (Dout, 4) int32, fwd_f (Dout, 2), bwd_i (Din, 5) int32, bwd_f (Din, 2))
    for an e3nn nn.Gate: the Extract order (sorted input irreps), the activations with their normalize2mom constants
    and the ElementwiseTensorProduct coefficient (probed) folded in. None if an activation is not silu / tanh /
    sigmoid / identity. See csrc/gate.cu for the table layout."""
    from e3nn.nn._gate import _Sortcut   # gate.sc may already be jet's _Split view: rebuild e3nn's own column order
    try:
        acts_s = [_act(a) for (mul, _), a in zip(gate.act_scalars.irreps_in, gate.act_scalars.acts) for _ in range(mul)]
        acts_g = [_act(a) for (mul, _), a in zip(gate.act_gates.irreps_in, gate.act_gates.acts) for _ in range(mul)]
    except ValueError:
        return None
    cut = _Sortcut(gate.irreps_scalars, gate.irreps_gates, gate.irreps_gated).cut
    sl = cut.irreps_in.slices()
    s_cols, g_cols, x_cols = ([c for i in ins for c in range(sl[i].start, sl[i].stop)] for ins in cut.instructions)
    fi, ff = [], []
    bi, bf = [[0] * 5 for _ in range(cut.irreps_in.dim)], [[0.0, 0.0] for _ in range(cut.irreps_in.dim)]
    for j, (code, cst) in zip(s_cols, acts_s):   # one scalar channel each (Activation's own irreps, unsimplified)
        bi[j], bf[j] = [1, len(fi), 0, code, 0], [cst, 1.0]
        fi.append([j, -1, code, 0])
        ff.append([cst, 1.0])
    if gate.irreps_gated.dim:
        with torch.no_grad():
            coef = gate.mul(torch.ones(1, gate.irreps_gated.dim), torch.ones(1, gate.irreps_gates.dim))[0].tolist()
        u = col = 0
        for mul, ir in gate.irreps_gated:
            d = ir.dim
            for _ in range(mul):
                gj, (code, cst), c, o0, src0 = g_cols[u], acts_g[u], coef[col], len(fi), x_cols[col]
                assert all(abs(coef[col + m] - c) <= 1e-6 * abs(c) for m in range(d))
                for m in range(d):
                    assert x_cols[col + m] == src0 + m
                    bi[src0 + m], bf[src0 + m] = [2, o0 + m, gj, code, 0], [cst, c]
                    fi.append([src0 + m, gj, code, 0])
                    ff.append([cst, c])
                bi[gj], bf[gj] = [3, o0, src0, code, d], [cst, c]
                col += d
                u += 1
    assert len(fi) == gate.irreps_out.dim
    i32 = dict(dtype=torch.int32)
    return (torch.tensor(fi, **i32), torch.tensor(ff, dtype=torch.float32), torch.tensor(bi, **i32),
            torch.tensor(bf, dtype=torch.float32))


def gate(x, tables):
    """e3nn Gate as one CUDA kernel per direction (fp32, first-order autograd in x); tables from gate_tables."""
    return torch.ops.jet_mlip.gate(x, *tables)


DEFAULT_RADIAL_TABLE = 2048   # requested grid points (rounded down by table_points); at the fp32 floor for 7net-nano / 0 / omni


def table_points(n: int, r_on: float, r_cut: float) -> int:
    """Largest grid size <= n with a node at r_on, where the XPLOR cutoff starts and W'' jumps: a Hermite interval
    across that point is only O(h^2) (h^4 elsewhere) and set the table's force error (nano N=2048: max 1.2e-5 eV/A,
    aligned N=2047: 1.0e-6). n itself if no such size >= n/2 exists (r_on/r_cut not a short fraction)."""
    for m in range(n, n // 2, -1):
        x = (m - 1) * r_on / r_cut
        if abs(x - round(x)) < 1e-6:
            return m
    return n


def hermite_table(w_of_r, r_cut: float, n: int, device=None, dtype=torch.float32):
    """Table of any W(r) for the kernel's cubic Hermite: w_of_r maps (m,) fp64 radii to (m, NW) fp64 weights
    (differentiable: torch.func.jvp). (n, 4*NW) `dtype`, entry [k, 4c:4c+4] = [W(r_k), dW/dr(r_k),
    (W(r_k+1)-W(r_k))/h, dW/dr(r_k+1)] of column c, r_k = k*h, h = r_cut/(n-1): one (edge, column) is one 16 B load.
    The node difference is tabulated in fp64 rather than formed in fp32, where the 1/h of the Hermite derivative would
    amplify its cancellation. W must vanish at and beyond r_cut."""
    # r = 0 may be 0/0 (sin(c r)/r, Bessel bases); no edge is ever shorter than the first grid step (~3e-3 A)
    r = torch.linspace(0, r_cut, n, dtype=torch.float64, device=device).clamp_min(1e-6)
    val, dval = torch.func.jvp(w_of_r, (r,), (torch.ones_like(r),))
    diff, m1 = torch.zeros_like(val), torch.zeros_like(val)   # last row unused: the kernel clamps k <= n-2
    diff[:-1] = (val[1:] - val[:-1]) * ((n - 1) / r_cut)
    m1[:-1] = dval[1:]
    return torch.stack([val, dval, diff, m1], dim=2).reshape(n, -1).to(dtype).contiguous()


def make_radial_table(cfg: ConvConfig, w1, w2, w3, coeffs, r_on: float, r_cut: float, act_cst: float,
                      n: int = DEFAULT_RADIAL_TABLE):
    """Deploy-time table replacing SevenNet's per-edge radial MLP (hermite_table; w3 pruned, same column order as
    conv_fused's W). Rebuild if the weights change. n is rounded down by table_points (a node at r_on), so the
    table has <= n rows."""
    n = table_points(n, r_on, r_cut)
    if cfg is not None:
        assert w3.shape[1] == cfg.n_weight, 'w3 must be the pruned (64, NW) last-layer weight'
    dtype = w1.dtype   # fp32 for the kernel; fp64 weights give an fp64 table (tests)
    w1, w2, w3, coeffs = [t.detach().to(torch.float64) for t in (w1, w2, w3, coeffs)]
    return hermite_table(lambda r: reference.radial_mlp(reference.bessel_cutoff(r, coeffs, r_on, r_cut), w1, w2, w3,
                                                        act_cst), r_cut, n, w1.device, dtype)


def conv_fused(cfg: ConvConfig, x, edge_vec, src, dst, w1, w2, w3, coeffs, denominator,
               r_on: float, r_cut: float, act_cst: float, n_out: Optional[int] = None,
               csr=None, table=None, split_bwd: bool = True, table_idx=None):
    """Inference-path fused layer (fwd + first-order bwd wrt x and edge_vec only).
    x (N_all, IN_DIM); src/dst int (E) in original edge order; w3 already pruned
    (64, NPATH*MUL); returns out (n_out, OUT_DIM) with dead blocks zero.
    csr = (rowptr, perm) from build_csr(dst, n_out), built here if None.
    table = make_radial_table(...) to take the interpolated radial path (fp32 CUDA only;
    the torch fallback always takes the exact one). A 3-D table (ntables, n, 4*NW) needs
    table_idx (E,) int32 selecting the table of every edge (e.g. one per element pair).
    split_bwd (exact path only): True = two backward kernels (grad of the MLP hidden layer accumulated
    per edge in a transient (E, 64) buffer, the MLP backward once per edge); False = one kernel that
    redoes the MLP backward per path group (no buffer)."""
    n_out = x.shape[0] if n_out is None else n_out
    if not has_op('conv_fused') or not x.is_cuda:
        out = reference.conv(cfg, x, edge_vec, src.long(), dst.long(), w1, w2, w3,
                             coeffs, r_on, r_cut, act_cst, denominator)
        return out[:n_out]
    if csr is None:
        csr = build_csr(dst, n_out)
    rowptr, perm = csr
    return torch.ops.jet_mlip.conv_fused(
        x, edge_vec, src.int(), rowptr, perm, w1, w2, w3, coeffs,
        float(denominator), float(r_on), float(r_cut), float(act_cst), config_id(cfg), table, split_bwd,
        table_idx.int() if table_idx is not None else None)


def tp3(cfg: ConvConfig, x, sh, W, src, dst, n_out: Optional[int] = None):
    """Training-path tensor product + scatter: out[dst] += TP(x[src], sh, W) (not supported; built with JET_TRAIN=1).
    Differentiable to any order wrt x, sh, W. W is the pruned per-edge weight (E, NPATH*MUL)."""
    n_out = x.shape[0] if n_out is None else n_out
    if x.is_cuda and is_available() and not has_op('tp3'):   # never fall back silently to the slow reference on a GPU
        raise RuntimeError('jet_mlip: the train-mode tp3 op is not built (training is not supported; rebuild with '
                           'JET_TRAIN=1 to get the unsupported path)')
    if not has_op('tp3') or not x.is_cuda:
        msg = reference.tp3(cfg, x[src.long()], sh, W)
        out = x.new_zeros(n_out, cfg.out_dim)
        return out.index_add(0, dst.long(), msg)
    return torch.ops.jet_mlip.tp3(x, sh, W, src.int(), dst.int(), n_out, config_id(cfg))


def edge_embed(edge_vec, coeffs, r_on: float, r_cut: float, lmax: int):
    """sevenn EdgeEmbedding: -> (B (E, NB) = bessel(r)*xplor(r), sh (E, (lmax+1)^2) e3nn
    component-normalized spherical harmonics of the unit vector). One kernel; any-order autograd."""
    if not has_op('edge_embed') or not edge_vec.is_cuda:
        r = edge_vec.norm(dim=-1)
        return (reference.bessel_cutoff(r, coeffs, r_on, r_cut),
                reference.spherical_harmonics(list(range(lmax + 1)), edge_vec))
    return torch.ops.jet_mlip.edge_embed(edge_vec, coeffs, float(r_on), float(r_cut), int(lmax))


def silu_norm(z, cst: float):
    """cst * silu(z) (e3nn normalize2mom(silu)); fused fwd / bwd / bwd-bwd, no third order."""
    if not has_op('silu_norm') or not z.is_cuda:
        return torch.nn.functional.silu(z) * cst
    return torch.ops.jet_mlip.silu_norm(z, float(cst))
