"""jet_mace.mliap.LAMMPS_MLIAP_MACE_Graph: the ML-IAP wrapper replays a CUDA graph over static, bucket-padded
buffers and gives the same energies / pair forces as the eager upstream wrapper, across edge-count changes
(same bucket), a new bucket and a changed ghost count. MACE-MPA-0: per-element-pair radial tables must use the
layer-0 species of the ghost rows LAMMPS no longer passes to the later interactions."""
import os

import mace  # noqa: F401  isort:skip
import pytest
import torch
from ase.build import bulk

pytest.importorskip('jet_mlip')
from jet_mace import convert, mliap  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')


class FakeData:   # module name must contain 'kokkos': upstream picks the GPU path from it
    __module__ = 'kokkos_fake'
    """The MLIAPDataPy fields the wrapper reads / writes (Kokkos path: device arrays)."""

    def __init__(self, batch, nlocal, nghost=0):
        ei = batch['edge_index']
        self.nlocal, self.ntotal, self.npairs = nlocal, nlocal + nghost, ei.shape[1]
        self.elems = batch['node_attrs'].argmax(1).int()
        self.elems = torch.cat([self.elems, self.elems[:nghost]])
        self.rij = (batch['positions'][ei[1]] - batch['positions'][ei[0]] + batch['shifts']).double()
        self.pair_i, self.pair_j = ei[1].int(), ei[0].int()
        # LAMMPS hands over a ghost j (>= nlocal) for a pair across the cell: the pairs whose j has a ghost read it
        self.pair_j[(batch['shifts'] != 0).any(1) & (self.pair_j < nghost)] += nlocal
        self.eatoms = torch.zeros(nlocal, dtype=torch.float64, device='cuda')
        self.energy = 0.0
        self.pair_forces = None

    def update_pair_forces_gpu(self, f):
        self.pair_forces = f.clone()

    def forward_exchange(self, x, out, vec_len):   # ghost k is the periodic image of local k
        out[:self.nlocal] = x[:self.nlocal]
        out[self.nlocal:self.ntotal] = x[:self.ntotal - self.nlocal]

    def reverse_exchange(self, g, gout, vec_len):
        gout.zero_()
        gout[:self.nlocal] = g[:self.nlocal]
        gout[:self.ntotal - self.nlocal] += g[self.nlocal:self.ntotal]


def structures(name):
    from mace.calculators import mace_mp
    calc = mace_mp(model=name, device='cuda', default_dtype='float32')
    out = []
    for rep, seed in (((2, 2, 1), 1), ((2, 2, 1), 2), ((3, 2, 1), 3)):
        atoms = bulk('LiF', 'rocksalt', a=4.0).repeat(rep)
        atoms.rattle(stdev=0.1, seed=seed)
        out.append((len(atoms), calc._atoms_to_batch(atoms).to('cuda').to_dict()))
    return calc, out


@pytest.mark.parametrize('name', ['medium', 'medium-mpa-0'])   # mpa-0: one radial table per element pair
def test_graph_wrapper_matches_eager(name, capsys):
    from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE
    calc, structs = structures(name)
    model = convert.patch_model(calc.models[0], radial_table=1024, cueq=True)
    model.lammps_mliap = True
    eager, graph = LAMMPS_MLIAP_MACE(model), mliap.LAMMPS_MLIAP_MACE_Graph(model)
    for k, (n, batch) in enumerate(structs + structs[:1]):
        nghost = 3 if k == 2 else 0
        d0, d1 = FakeData(batch, n, nghost), FakeData(batch, n, nghost)
        eager.compute_forces(d0)
        graph.compute_forces(d1)
        assert abs(d1.energy - d0.energy) < 1e-4 * n
        assert torch.allclose(d1.eatoms, d0.eatoms, atol=1e-4)
        assert torch.allclose(d1.pair_forces, d0.pair_forces, atol=2e-3)
        assert bool((d1.pair_j >= n).any()) == bool(nghost)   # ghost-source edges, folded onto their owners
        assert graph.key[:2] == (n, n)   # one rank: no ghost and no padding row
    assert graph.n_captures >= 2 and graph.n_replays >= 1
    # compiled when LAMMPS runs, not at deploy; not with JET_MACE_COMPILE=0, and not after the wrapper dropped a
    # compilation Dynamo failed on (torch >= 2.8 with e3nn 0.4.4)
    compiled = os.environ.get('JET_MACE_COMPILE', '1') != '0' and 'torch.compile failed' not in capsys.readouterr().out
    assert hasattr(graph.model.model, '_orig_mod') == compiled


def test_compile_switch_is_read_at_run_time(monkeypatch):
    """JET_MACE_COMPILE=0 in the LAMMPS run gives the uncompiled model, also from a pickle that holds a model compiled
    at deploy (up to v0.5.0)."""
    from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE
    calc, structs = structures('medium')
    model = convert.patch_model(calc.models[0], radial_table=1024, cueq=True)
    model.lammps_mliap = True
    eager, graph = LAMMPS_MLIAP_MACE(model), mliap.LAMMPS_MLIAP_MACE_Graph(model)
    assert not hasattr(graph.model.model, '_orig_mod')   # deploy does not compile
    graph.model.model = torch.compile(graph.model.model, dynamic=False)   # as a v0.5.0 pickle
    monkeypatch.setenv('JET_MACE_COMPILE', '0')
    n, batch = structs[0]
    d0, d1 = FakeData(batch, n), FakeData(batch, n)
    eager.compute_forces(d0)
    graph.compute_forces(d1)
    assert not hasattr(graph.model.model, '_orig_mod') and graph.n_captures == 1
    assert torch.allclose(d1.pair_forces, d0.pair_forces, atol=2e-3)
