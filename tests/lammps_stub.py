"""A stand-in for LAMMPS ML-IAP's MLIAPDataPy on one rank, built from ase Atoms: local atoms first, then one ghost per
(neighbour, periodic image) pair, as LAMMPS lays them out. Pair k: pair_i[k] local, pair_j[k] local or ghost,
rij[k] = x[pair_j] - x[pair_i]. forward_exchange copies each ghost row from its owner, reverse_exchange adds ghost rows
into their owners. `atom_forces()` turns the pair forces f = dE/drij the wrapper hands LAMMPS into per-atom forces
(F_i += f, F_owner(j) -= f) for comparison with an ASE / model reference.

    d = lammps_stub.make(atoms, cutoff, elems, cuda=True)   # cuda=True: module name holds 'kokkos' (device path)
"""
import numpy as np
import torch
from ase.neighborlist import neighbor_list


class LmpData:
    def __init__(self, atoms, cutoff, elems, device='cpu'):
        i, j, s = neighbor_list('ijS', atoms, cutoff)
        n = len(atoms)
        img = np.any(s != 0, axis=1)
        keys, inv = np.unique(np.c_[j[img], s[img]], axis=0, return_inverse=True)   # one ghost per (owner, image)
        pj = j.copy()
        pj[img] = n + inv.reshape(-1)
        pos = atoms.get_positions()
        cell = np.asarray(atoms.get_cell(complete=True))
        xg = np.concatenate([pos, pos[keys[:, 0]] + keys[:, 1:] @ cell])
        t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)   # noqa: E731
        self.nlocal, self.ntotal, self.npairs = n, n + len(keys), len(i)
        self.owner = t(np.r_[np.arange(n), keys[:, 0]], torch.int64)
        self.elems = t(np.asarray(elems)[self.owner.cpu().numpy()], torch.int32)
        self.pair_i, self.pair_j = t(i, torch.int32), t(pj, torch.int32)
        self.rij = t(xg[pj] - pos[i], torch.float64)
        self.eatoms = torch.zeros(n, dtype=torch.float64, device=device)
        self.energy = 0.0
        self.pair_forces = None

    def update_pair_forces_gpu(self, f):
        self.pair_forces = f.detach().double().clone()

    def forward_exchange(self, x, out, vec_len):
        out.copy_(x[self.owner])

    def reverse_exchange(self, g, gout, vec_len):
        gout.zero_()
        gout.index_add_(0, self.owner, g)

    def atom_forces(self):
        f = self.pair_forces
        F = torch.zeros(self.nlocal, 3, dtype=f.dtype, device=f.device)
        F.index_add_(0, self.pair_i.long(), f)
        F.index_add_(0, self.owner[self.pair_j.long()], -f)
        return F


class KokkosLmpData(LmpData):
    pass


KokkosLmpData.__module__ = 'kokkos_stub'   # wrappers pick the device from the module name of the data object


def make(atoms, cutoff, elems, cuda=False):
    return KokkosLmpData(atoms, cutoff, elems, 'cuda') if cuda else LmpData(atoms, cutoff, elems)
