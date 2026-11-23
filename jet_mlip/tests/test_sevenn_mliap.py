"""SevenNetGraphMLIAPWrapper on tests/lammps_stub.py's MLIAPDataPy stand-in (CUDA): the CUDA-graph path reproduces
the same wrapper's eager path and the upstream sevenn ML-IAP wrapper, for 7net-0 and (if loadable) 7net-omni with
its modal row collapsed, and an unloaded wrapper survives torch.save / torch.load.
"""
import numpy as np
import pytest
import torch
from ase.build import bulk, make_supercell

pytest.importorskip('sevenn')

from sevenn.mliap import SevenNetMLIAPWrapper  # noqa: E402

from jet_mlip.sevenn_mliap import SevenNetGraphMLIAPWrapper  # noqa: E402
from tests import lammps_stub  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')

CASES = [('7net-0', None), ('7net-omni', 'mpa')]


def structures():
    """Rattled cells with ghost rows (a short axis keeps 2 r_cut > the box)."""
    cells = {}
    for name, crystal, a in (('LiF rocksalt', 'rocksalt', 4.02), ('Si diamond', 'diamond', 5.43)):
        atoms = make_supercell(bulk(name.split()[0], crystal, a=a), [[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        atoms.positions += np.random.default_rng(len(cells)).uniform(-0.05, 0.05, atoms.positions.shape)
        cells[name] = atoms
    return cells


def make(cls, model, modal, **kwargs):
    try:
        return cls(model, modal=modal, use_jet=True, **kwargs)
    except Exception as e:
        pytest.skip(f'{model} not loadable: {e}')


def load(wrapper, model):
    """Build the lazy model now so an unavailable checkpoint skips instead of failing an assertion."""
    try:
        wrapper._ensure_model_initialized()
    except Exception as e:
        pytest.skip(f'{model} not loadable: {e}')
    return wrapper


def run(wrapper, atoms):
    z = atoms.get_atomic_numbers()
    if any(wrapper.element_types[int(i)] == 'X' for i in z):
        pytest.skip(f'{set(atoms.get_chemical_symbols())} not in the model elements')
    d = lammps_stub.make(atoms, wrapper.cutoff, z, cuda=True)
    wrapper.compute_forces(d)
    return float(torch.as_tensor(d.energy).detach()), d.atom_forces()


def report(what, e0, e1, f0, f1, n):
    de, df = abs(e1 - e0) / n, (f1 - f0).abs().max().item()
    print(f'{what}: dE/atom {de:.3e} eV max|dF| {df:.3e} eV/A')
    return de, df


@pytest.mark.parametrize('model,modal', CASES)
@pytest.mark.parametrize('name', list(structures()))
def test_graph_matches_eager_and_upstream(name, model, modal, monkeypatch):
    atoms = structures()[name]
    monkeypatch.delenv('JET_SEVENN_CUDA_GRAPH', raising=False)
    graph = load(make(SevenNetGraphMLIAPWrapper, model, modal), model)
    e1, f1 = run(graph, atoms)
    monkeypatch.setenv('JET_SEVENN_CUDA_GRAPH', '0')
    eager = load(make(SevenNetGraphMLIAPWrapper, model, modal), model)
    e0, f0 = run(eager, atoms)
    upstream = load(make(SevenNetMLIAPWrapper, model, modal), model)
    eu, fu = run(upstream, atoms)
    for what, e, f in (('eager', e0, f0), ('upstream', eu, fu)):
        de, df = report(f'{model} {name} {what}', e, e1, f, f1, len(atoms))
        assert de < 1e-5, f'{model} {name} {what}: dE/atom {de:.3e} eV'
        assert df < 1e-4, f'{model} {name} {what}: max|dF| {df:.3e} eV/A'
    assert graph.n_captures >= 1


@pytest.mark.parametrize('model,modal', CASES)
def test_save_load_round_trip(tmp_path, model, modal):
    atoms = structures()['LiF rocksalt']
    path = tmp_path / 'sevenn.mliap.pt'
    torch.save(make(SevenNetGraphMLIAPWrapper, model, modal), path)
    loaded = torch.load(path, weights_only=False)
    assert loaded.model is None   # lazy: rebuilt from the checkpoint on first use
    e0, f0 = run(load(make(SevenNetMLIAPWrapper, model, modal), model), atoms)
    e1, f1 = run(load(loaded, model), atoms)
    de, df = report(f'{model} round-trip', e0, e1, f0, f1, len(atoms))
    assert de < 1e-5 and df < 1e-4, (de, df)
