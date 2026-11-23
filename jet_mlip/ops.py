"""Python entry points. Every op falls back to the torch reference when the CUDA
library is not built (slow; for tests and CPU)."""
from __future__ import annotations

from typing import Dict, Optional

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
