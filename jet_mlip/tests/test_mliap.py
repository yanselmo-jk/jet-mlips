"""The shared graph path (jet_mlip/mliap.py) on tests/lammps_stub.py's MLIAPDataPy stand-in, CUDA only.

A toy family sums one W(r) per edge two ways -- an unmasked index_add over every padded row (what upstream
convolutions do) and a masked CSR walk over the real edges only (ops.EDGE_MASK -> ops.build_csr) -- so the tests
cover the ghost fold, the padding layout, capture/replay, capacity growth, and eflag without any family package.
The reference is the same energy built from the stub's own pair_i / rij in fp64 (no padding).
"""
import numpy as np
import pytest
import torch
from ase.build import bulk, make_supercell

from jet_mlip import mliap, ops
from jet_mlip.mliap import GraphMLIAP
from tests import lammps_stub

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')

R_CUT = 3.0


def w_of_r(r):
    """(1 - r/r_cut)^2 inside, exactly 0 at and beyond the cutoff (padding edges carry W(r_cut) = 0)."""
    return torch.where(r < R_CUT, (1 - r / R_CUT) ** 2, torch.zeros_like(r))


class Toy(GraphMLIAP):
    device = 'cuda'
    r_cut = R_CUT
    vec_dtype = torch.float64
    index_dtype = torch.int64

    def _forward(self, vec):
        ncap = self.key[1]
        # (a) every ecap row, padding included: the padding vectors are (r_cut, 0, 0), so they add W(r_cut) = 0
        ea = torch.zeros(ncap, dtype=vec.dtype, device=vec.device)
        ea.index_add_(0, self.s_ei[0], w_of_r(vec.norm(dim=-1)))
        # (b) the real edges only: rowptr/perm from the masked CSR, one segment sum per row (no Python edge loop)
        rowptr, perm = ops.build_csr(self.s_ei[0], ncap)
        cs = torch.cat([vec.new_zeros(1), w_of_r(vec[perm.long()].norm(dim=-1)).cumsum(0)])
        eb = cs[rowptr[1:].long()] - cs[rowptr[:-1].long()]
        return 0.5 * ea + 0.5 * eb


def structures():
    """Rattled periodic cells that produce ghost rows (the LiF cell's c axis is shorter than 2 r_cut)."""
    rng = np.random.default_rng(0)
    si = make_supercell(bulk('Si', 'diamond', a=5.43), [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    si.positions += rng.uniform(-0.05, 0.05, si.positions.shape)
    lif = make_supercell(bulk('LiF', 'rocksalt', a=4.02), [[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    lif.positions += rng.uniform(-0.05, 0.05, lif.positions.shape)
    return {'Si diamond': si, 'LiF rocksalt': lif}


def stub(atoms):
    return lammps_stub.make(atoms, R_CUT, [0] * len(atoms), cuda=True)


def reference(d):
    """Same energy and forces from the stub data directly, fp64, pair_i / rij, no padding."""
    rij = d.rij.clone().requires_grad_(True)
    e = torch.zeros(d.nlocal, dtype=torch.float64, device=rij.device)
    e.index_add_(0, d.pair_i.long(), w_of_r(rij.norm(dim=-1)))
    energy = e.sum()
    f = torch.autograd.grad(energy, rij)[0]   # dE/drij = LAMMPS' pair force
    F = torch.zeros(d.nlocal, 3, dtype=torch.float64, device=rij.device)
    F.index_add_(0, d.pair_i.long(), f)
    F.index_add_(0, d.owner[d.pair_j.long()], -f)
    return energy.item(), e.detach(), F


def check(wrapper, d, energy=True):
    e_ref, e_ref_atoms, f_ref = reference(d)
    wrapper.graph_forces(d)
    e, F = float(d.energy), d.atom_forces()
    if energy:
        assert abs(e - e_ref) <= 1e-10 * abs(e_ref), f'dE {e - e_ref:.3e} eV'
        torch.testing.assert_close(d.eatoms, e_ref_atoms, rtol=1e-10, atol=1e-12)   # LAMMPS reads eatoms with eflag only
    torch.testing.assert_close(F, f_ref, rtol=1e-10, atol=1e-12)
    return e_ref


@pytest.mark.parametrize('name', list(structures()))
def test_graph_forces_matches_reference(name):
    check(Toy(), stub(structures()[name]))


def test_padding_layout(monkeypatch):
    """After a step: padding rows are local, masked out, and sorted after the npairs real edges in the CSR."""
    w = Toy()
    d = stub(structures()['Si diamond'])
    w.graph_forces(d)
    nlocal, ncap, ecap = w.key[0], w.key[1], w.key[2]
    npairs = d.npairs
    assert (w.s_ei[:, npairs:] < nlocal).all()
    pad = torch.arange(npairs, ecap, device=w.s_ei.device) % nlocal   # spread over the local rows
    assert torch.equal(w.s_ei[:, npairs:], pad.expand(2, -1))
    assert (w.s_mask[:npairs] == 1).all() and (w.s_mask[npairs:] == 0).all()
    monkeypatch.setattr(ops, 'EDGE_MASK', w.s_mask)
    rowptr, perm = ops.build_csr(w.s_ei[0], ncap)
    assert rowptr[-1].item() == npairs
    assert torch.equal(torch.sort(perm[:npairs].long()).values, torch.arange(npairs, device=perm.device))
    assert torch.equal(perm[npairs:].long(), torch.arange(npairs, ecap, device=perm.device))   # padding last


@pytest.mark.parametrize('grouped', [True, False])
def test_masked_csr(monkeypatch, grouped):
    """build_csr with EDGE_MASK: rows hold exactly their real edges; a list grouped by dst with the padding last (a
    LAMMPS pair list) keeps its order (perm = arange), any other order is still a valid CSR."""
    g = torch.Generator().manual_seed(0)
    n, E, real = 37, 5000, 4321
    dst = torch.randint(0, n + 3, (E,), generator=g)   # some dst past n_nodes: counted as padding
    mask = (torch.arange(E) < real).int()
    if grouped:
        dst = torch.cat([dst[:real].sort().values, dst[real:]])
    else:
        mask = mask[torch.randperm(E, generator=g)]
    dst, mask = dst.cuda(), mask.cuda()
    monkeypatch.setattr(ops, 'EDGE_MASK', mask)
    rowptr, perm = ops.build_csr(dst.int(), n)
    live = mask.bool() & (dst < n)
    counts = torch.bincount(dst[live], minlength=n)
    assert rowptr.dtype == perm.dtype == torch.int32 and rowptr[0] == 0
    assert torch.equal(rowptr[1:].long(), counts.cumsum(0))
    assert torch.equal(perm.long().sort().values, torch.arange(E, device='cuda'))
    k = perm[:rowptr[-1]].long()
    assert live[k].all() and torch.equal(dst[k], torch.repeat_interleave(torch.arange(n, device='cuda'), counts))
    if grouped:
        assert torch.equal(perm.long(), torch.arange(E, device='cuda'))


def test_ghost_fold():
    """s_ei[1] is the owner of LAMMPS' pair_j for every real edge, never a ghost row."""
    w = Toy()
    d = stub(structures()['LiF rocksalt'])
    w.graph_forces(d)
    npairs = d.npairs
    assert (w.s_ei[1, :npairs] < d.nlocal).all()
    assert torch.equal(w.s_ei[1, :npairs].long(), d.owner[d.pair_j.long()])


def test_capture_once_then_replay():
    w = Toy()
    atoms = structures()['Si diamond']
    check(w, stub(atoms))
    moved = atoms.copy()
    moved.positions += np.random.default_rng(1).uniform(-0.01, 0.01, moved.positions.shape)
    d = stub(moved)
    assert d.npairs <= w.key[2]   # same edge bucket: replay, no re-capture
    check(w, d)
    assert w.n_captures == 1 and w.n_replays >= 1, (w.n_captures, w.n_replays)


def test_pair_list_reuse():
    """The pair list of the step before with other vectors: only the model graph replays. Another list of the same
    size (here the same pairs in another order) goes through the fold again."""
    w = Toy()
    atoms = structures()['LiF rocksalt']
    check(w, stub(atoms))
    d = stub(atoms)
    d.rij *= 1.001
    check(w, d)
    assert (w.n_lists, w.n_captures, w.n_replays) == (1, 1, 1)
    d = stub(atoms)
    d.pair_i, d.pair_j, d.rij = d.pair_i.flip(0), d.pair_j.flip(0), d.rij.flip(0)
    check(w, d)
    assert (w.n_lists, w.n_captures, w.n_replays) == (2, 1, 2)


def test_capacity_growth(monkeypatch):
    """More pairs than the edge cap re-captures; a small bucket keeps the second cell cheap."""
    monkeypatch.setattr(mliap, 'EDGE_BUCKET', 64)
    w = Toy()
    check(w, stub(structures()['LiF rocksalt']))
    assert w.n_captures == 1
    larger = make_supercell(structures()['LiF rocksalt'], [[1, 0, 0], [0, 1, 0], [0, 0, 2]])
    d = stub(larger)
    assert d.npairs > w.key[2]
    check(w, d)
    assert w.n_captures == 2, w.n_captures


@pytest.mark.parametrize('eflag', [0, 1])
def test_eflag(eflag):
    w = Toy()
    d = stub(structures()['Si diamond'])
    sentinel = 12345.0
    d.eflag = eflag
    d.energy = sentinel
    e_ref = check(w, d, energy=bool(eflag))   # check() also verifies the pair forces
    if eflag:
        assert abs(float(d.energy) - e_ref) <= 1e-10 * abs(e_ref)
    else:
        assert d.energy == sentinel


def test_same_build_skips_probe(monkeypatch):
    """nbuild (our LAMMPS patch) unchanged: another pair list of the same build goes in without the exchange probe."""
    w = Toy()
    d = stub(structures()['LiF rocksalt'])
    d.nbuild = 7
    check(w, d)
    probes = []
    monkeypatch.setattr(type(d), 'forward_exchange', lambda self, *a: probes.append(1))   # must not be called
    keep = d.npairs - 5   # pairs left r_cut: a shorter list, same atoms and ghosts
    d.npairs, d.pair_i, d.pair_j, d.rij = keep, d.pair_i[:keep], d.pair_j[:keep], d.rij[:keep]
    check(w, d)
    assert not probes and (w.n_lists, w.n_captures) == (2, 1)
    monkeypatch.undo()
    d.nbuild = 8   # rebuilt: probed again
    d.pair_i, d.pair_j, d.rij = d.pair_i.flip(0), d.pair_j.flip(0), d.rij.flip(0)
    check(w, d)
    assert w.n_lists == 3
