"""Periodic neighbour list on the GPU for the ASE calculators: the pair set of
matscipy.neighbours.neighbour_list('ijS', ...) / ase.neighborlist.primitive_neighbor_list, any triclinic cell, mixed
pbc, cells shorter than the cutoff (self-images). The atoms are wrapped into the cell, their periodic images within
the cutoff of it are appended as ghost rows and ops.cand_build (cell list + fp64 distance cut) lists the pairs; a
ghost maps back to its atom and integer shift."""
import os

import numpy as np
import torch

from . import ops


def available(device) -> bool:
    """True if the family calculators take the GPU list: a CUDA device, cand_build in the build, JET_ASE_GPU_NL != 0."""
    return torch.device(device).type == 'cuda' and os.environ.get('JET_ASE_GPU_NL', '1') != '0' and ops.has_op('cand_build')


def neighbor_list(positions, cell, pbc, cutoff: float):
    """-> (i, j, S): int64 (E,), (E,), (E, 3) on positions' device. Every pair with
    |positions[j] - positions[i] + S @ cell| < cutoff (fp64), i == j only with S != 0; S is 0 along non-periodic
    directions. Grouped by i in ascending order, no order within an atom (matscipy's differs).
    positions (N, 3) tensor, need not lie inside the cell; cell 3x3 rows = lattice vectors (zero rows of non-periodic
    directions are completed as ase does); pbc 3 bools. Allocates and syncs."""
    from ase.geometry import complete_cell
    dev = positions.device
    x = positions.detach().to(torch.float64)
    n = x.shape[0]
    pbc = np.asarray(pbc.cpu() if torch.is_tensor(pbc) else pbc, dtype=bool).reshape(3)
    cell = complete_cell(np.asarray(cell.detach().cpu() if torch.is_tensor(cell) else cell, dtype=np.float64).reshape(3, 3))
    inv = np.linalg.inv(cell)   # fractional s = x @ inv
    # |ds_k| <= |dr| * |inv[:, k]|: a pair within the cutoff is closer than `reach` in fractional coordinate k
    reach = cutoff * np.linalg.norm(inv, axis=0)
    f64 = dict(dtype=torch.float64, device=dev)
    cell_t, per = torch.as_tensor(cell, **f64), torch.as_tensor(pbc, device=dev)
    s = x @ torch.as_tensor(inv, **f64)
    w = torch.where(per, s.floor(), torch.zeros_like(s))   # wrap the periodic directions: s - w in [0, 1]
    s, xw = s - w, x - w @ cell_t
    # ghosts: shifts n != 0 with |n_k| <= ceil(reach_k), of the atoms whose image lies within reach of [0, 1]
    shifts = torch.cartesian_prod(*[torch.arange(-m, m + 1, **f64) for m in np.where(pbc, np.ceil(reach), 0).astype(int)])
    shifts = shifts[(shifts != 0).any(1)]
    margin = torch.as_tensor(np.where(pbc, reach + 1e-6, np.inf), **f64)
    g = s[None] + shifts[:, None]   # (K, N, 3). ponytail: K*N*24 B transient; chunk over K if a huge system needs it
    k, a = ((g >= -margin) & (g <= 1 + margin)).all(-1).nonzero(as_tuple=True)
    xall = torch.cat([xw, xw[a] + shifts[k] @ cell_t])
    if x.is_cuda and ops.has_op('cand_build'):
        ptr, c = ops.cand_build(xall.contiguous(), n, cutoff * cutoff)
        i, c = torch.repeat_interleave(torch.arange(n, device=dev), (ptr[1:] - ptr[:-1]).long()), c.long()
    else:   # torch reference (tests, CPU). ponytail: O(N * (N + ghosts)) memory
        d2 = ((xall[None] - xw[:, None]) ** 2).sum(-1)
        d2[torch.arange(n), torch.arange(n)] = float('inf')
        i, c = (d2 < cutoff * cutoff).nonzero(as_tuple=True)
    j = torch.cat([torch.arange(n, device=dev), a])[c]
    S = torch.cat([shifts.new_zeros(n, 3), shifts[k]])[c] + w[i] - w[j]
    return i, j, S.long()
