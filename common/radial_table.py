"""Deploy-time radial tables for jet_mlip's conv_fused TABLE path, for any W(r) (jet_mlip.ops.hermite_table: layout,
grid and why the node difference is tabulated). W must vanish at and beyond r_cut (the cutoff is inside w_of_r)."""
import torch

from jet_mlip.ops import hermite_table

DEFAULT_N = 2048   # table force error < 10 % of the official fp32 error for NequIP and Nequix (EquFlash: scan pending; MACE: jet_mace.convert)


def make_radial_table(w_of_r, r_cut: float, n: int = DEFAULT_N, device=None, dtype=torch.float32):
    """w_of_r: (m,) fp64 radii -> (m, NW) fp64 weights, differentiable. -> (n, 4*NW) `dtype` table."""
    return hermite_table(w_of_r, r_cut, n, device, dtype)


def make_radial_tables(w_of_r_list, r_cut: float, n: int = DEFAULT_N, device=None, dtype=torch.float32):
    """One table per entry (e.g. per element pair when W depends on the species) -> (len, n, 4*NW)."""
    return torch.stack([make_radial_table(w, r_cut, n, device, dtype) for w in w_of_r_list])


def interpolate(tab, r_max, r, idx=None):
    """Torch reference of the kernel's cubic-Hermite lookup: tab ([n_pairs,] n, 4*NW) on [0, r_max], r (E,),
    idx (E,) table index for 3-D tables -> (E, NW) fp64."""
    n = tab.shape[-2]
    h = r_max / (n - 1)
    xg = r.double() / h
    k = xg.floor().long().clamp(0, n - 2)
    t = (xg - k).clamp(0, 1).unsqueeze(-1)
    row0 = tab[idx.long(), k].double() if tab.dim() == 3 else tab[k].double()
    v0, m0, d, m1 = row0[:, 0::4], row0[:, 1::4], row0[:, 2::4], row0[:, 3::4]   # [W, dW/dr, diff/h, dW/dr(r_k+1)]
    return v0 + h * ((3 * t**2 - 2 * t**3) * d + (t**3 - 2 * t**2 + t) * m0 + (t**3 - t**2) * m1)
