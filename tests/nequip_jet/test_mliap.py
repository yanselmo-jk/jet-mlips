"""jet_nequip.mliap on tests/lammps_stub.py's MLIAPDataPy stand-in (CUDA, fp32): the graph wrapper reproduces the
upstream NequIP ML-IAP wrapper (with the OAM ZBL pair term active on Li-F), the CUDA graph reproduces the same
wrapper's eager path, and an unloaded wrapper
survives torch.save / torch.load and then computes.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import bulk, make_supercell

# this test package shadows the installed `nequip` under pytest's prepend import mode (as test_convert.py): hand
# the name back before importing it
_tests = str(Path(__file__).resolve().parents[1])
_saved = sys.path[:]
sys.path[:] = [p for p in sys.path if str(Path(p or '.').resolve()) != _tests]
sys.modules.pop('nequip', None)
try:
    import nequip  # noqa: F401,E402
except ImportError as e:
    pytest.skip(f'nequip unavailable: {e}', allow_module_level=True)
finally:
    sys.path[:] = _saved
pytest.importorskip('lammps.mliap.mliap_unified_abc')

from nequip.integrations.lammps_mliap.lmp_mliap_wrapper import (  # noqa: E402
    NequIPLAMMPSMLIAPWrapper,
)
from nequip.model.saved_models.load_utils import _get_model_file_path  # noqa: E402
from nequip.train.lightning import _SOLE_MODEL_KEY  # noqa: E402

from jet_nequip.mliap import NequIPGraphMLIAPWrapper  # noqa: E402
from tests import lammps_stub  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')

MODEL = 'nequip.net:mir-group/NequIP-OAM-S:0.1'
ATTRS = ('r_cut', 'vec_dtype', 'index_dtype', 'use_graph', 'caps', 'graph', 'key')


@pytest.fixture(scope='module')
def model_path():
    try:
        with _get_model_file_path(MODEL) as path:
            yield str(path)   # keep the context open for the module: an uncached download may be temporary
    except Exception as e:
        pytest.skip(f'{MODEL} not loadable: {e}')


def make(cls, path, **kwargs):
    """As jet_nequip/deploy.py, but eager compile off: the tests compare un-compiled paths."""
    try:
        return cls(model_path=str(path), model_key=_SOLE_MODEL_KEY, modifiers=[], compile=False, tf32=False, **kwargs)
    except Exception as e:
        pytest.skip(f'{MODEL} not loadable: {e}')


def structures():
    """Rattled LiF rocksalt cells (Li-F switches on OAM's ZBL term; short axes give ghost rows)."""
    out = {}
    for name, reps, seed in (('LiF 2x2x1', [[2, 0, 0], [0, 2, 0], [0, 0, 1]], 0),
                             ('LiF 2x2x2', [[2, 0, 0], [0, 2, 0], [0, 0, 2]], 1)):
        atoms = make_supercell(bulk('LiF', 'rocksalt', a=4.02), reps)
        atoms.positions += np.random.default_rng(seed).uniform(-0.05, 0.05, atoms.positions.shape)
        out[name] = atoms
    return out


def run(wrapper, atoms):
    """LMP types are the model's type_names order (NequIP reads lmp_data.elems as ATOM_TYPE)."""
    try:
        elems = [wrapper.element_types.index(s) for s in atoms.get_chemical_symbols()]
    except ValueError:
        pytest.skip(f'{atoms.get_chemical_symbols()[0]} not in the model types {wrapper.element_types}')
    d = lammps_stub.make(atoms, 2.0 * float(wrapper.rcutfac), elems, cuda=True)
    wrapper.compute_forces(d)
    return float(torch.as_tensor(d.energy).detach()), d.atom_forces()


def report(what, e0, e1, f0, f1, n):
    de, df = abs(e1 - e0) / n, (f1 - f0).abs().max().item()
    print(f'{what}: dE/atom {de:.3e} eV max|dF| {df:.3e} eV/A')
    return de, df


@pytest.mark.parametrize('name', list(structures()))
def test_jet_matches_upstream(name, model_path):
    atoms = structures()[name]
    e0, f0 = run(make(NequIPLAMMPSMLIAPWrapper, model_path), atoms)
    e1, f1 = run(make(NequIPGraphMLIAPWrapper, model_path, jet=True), atoms)
    de, df = report(name, e0, e1, f0, f1, len(atoms))
    assert de < 1e-4, f'{name}: dE/atom {de:.3e} eV'
    assert df < 2e-3, f'{name}: max|dF| {df:.3e} eV/A'


@pytest.mark.parametrize('name', list(structures()))
def test_graph_matches_eager(name, model_path, monkeypatch):
    atoms = structures()[name]
    monkeypatch.setenv('JET_NEQUIP_CUDA_GRAPH', '0')
    e0, f0 = run(make(NequIPGraphMLIAPWrapper, model_path, jet=True), atoms)
    monkeypatch.delenv('JET_NEQUIP_CUDA_GRAPH', raising=False)
    graph = make(NequIPGraphMLIAPWrapper, model_path, jet=True)
    e1, f1 = run(graph, atoms)
    de, df = report(name, e0, e1, f0, f1, len(atoms))
    assert de < 1e-5, f'{name}: dE/atom {de:.3e} eV'
    assert df < 1e-4, f'{name}: max|dF| {df:.3e} eV/A'
    assert graph.n_captures >= 1


def test_save_load_round_trip(tmp_path, model_path):
    atoms = structures()['LiF 2x2x1']
    e0, f0 = run(make(NequIPLAMMPSMLIAPWrapper, model_path), atoms)
    path = tmp_path / 'oam.nequip.lmp.pt'
    torch.save(make(NequIPGraphMLIAPWrapper, model_path, jet=True), path)
    loaded = torch.load(path, weights_only=False)
    assert loaded.model is None   # bytes only: the model is rebuilt on first use
    e1, f1 = run(loaded, atoms)
    de, df = report('round-trip', e0, e1, f0, f1, len(atoms))
    assert de < 1e-4 and df < 2e-3, (de, df)
