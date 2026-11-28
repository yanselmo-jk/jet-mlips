"""JET-patched models, driven through their family's own ASE calculator, match the official calculator's energy and
forces (and still run BFGS). One test per family; each skips when CUDA, the family package or the model is unavailable.
The reference is always a second, unpatched calculator on the same structure (its own CPU neighbour list; the patched
calculator builds its list on the GPU, jet_mlip.neighbors). graph=True (the families on jet_mlip.ase_graph): a later call
with the same atom count comes from a CUDA graph and gives the eager call's energy, forces and stress."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.optimize import BFGS


@pytest.fixture(autouse=True)
def default_dtype():
    """nequip leaves torch's default dtype at float64 for the process; the next family's calculator must not inherit it
    (nequix then builds fp64 features for its fp32 model: RuntimeError in the first Linear)."""
    dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(dtype)


def structure():
    """64-atom rattled Si diamond supercell."""
    atoms = bulk('Si', 'diamond', a=5.43, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.05, seed=0)
    return atoms


def ef(calc, atoms):
    atoms = atoms.copy()
    return calc.get_potential_energy(atoms), calc.get_forces(atoms)


def results(calc, atoms):
    # a new step every time (no ASE result cache); all three arguments: not every calculator has defaults for them
    calc.calculate(atoms.copy(), ['energy', 'forces', 'stress'], all_changes)
    return calc.results['energy'], calc.results['forces'].copy(), np.array(calc.results['stress'])


def check(ref, jet, atoms, model, names, gpu_nl=True, graph=False):
    from unittest import mock

    from jet_mlip import neighbors
    e0, f0 = ef(ref, atoms)
    with mock.patch.object(neighbors, 'neighbor_list', wraps=neighbors.neighbor_list) as nl:
        try:
            e1, f1 = ef(jet, atoms)
        except KeyError as e:   # a JET_CONFIGS=<family> build in an env that also holds another family's package
            if 'config not compiled' not in str(e):
                raise
            pytest.skip(str(e))
    assert nl.called == (gpu_nl and neighbors.available('cuda'))   # the patched calculator lists its pairs on the GPU
    assert abs(e1 - e0) / len(atoms) < 1e-4, (e0, e1)
    assert np.abs(f1 - f0).max() < 2e-3
    assert any(type(m).__name__ in names for m in model.modules()), names
    atoms = atoms.copy()
    atoms.calc = jet
    BFGS(atoms, logfile=None).run(steps=2)
    if graph:
        from jet_mlip import ase_graph
        with mock.patch.dict(os.environ, JET_ASE_GRAPH='0'):
            e0, f0, s0 = results(jet, atoms)
        with mock.patch.object(ase_graph.AseGraph, 'compute', autospec=True, side_effect=ase_graph.AseGraph.compute) as step:
            e1, f1, s1 = results(jet, atoms)
        assert step.called
        assert abs(e1 - e0) / len(atoms) < 1e-5, (e0, e1)
        assert np.abs(f1 - f0).max() < 2e-5
        assert np.abs(s1 - s0).max() < 1e-4 * np.abs(s0).max() + 1e-8, (s0, s1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_sevennet():
    calculator = pytest.importorskip('sevenn.calculator')
    model = os.environ.get('JET_TEST_SEVENN_MODEL', '7net-0')
    try:
        ref = calculator.SevenNetCalculator(model, device='cuda')
    except Exception as e:
        pytest.skip(f'{model}: {e}')
    try:
        jet = calculator.SevenNetCalculator(model, device='cuda', enable_jet=True)
    except TypeError:
        pytest.skip('sevenn is not patched: enable_jet not accepted')
    check(ref, jet, structure(), jet.model, ('JetConvolution',), graph=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_mace():
    calculators = pytest.importorskip('mace.calculators')
    convert = pytest.importorskip('jet_mace.convert')
    model = os.environ.get('JET_TEST_MACE_MODEL', 'medium')
    kw = dict(model=model, device='cuda', default_dtype='float32')
    try:
        ref = calculators.mace_mp(**kw)
    except Exception as e:
        pytest.skip(f'{model}: {e}')
    jet = calculators.mace_mp(**kw)
    jet.models[0] = convert.patch_model(jet.models[0], cueq=True)
    check(ref, jet, structure(), jet.models[0], ('MaceTPConv',), graph=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_nequip():
    ase_integration = pytest.importorskip('nequip.integrations.ase')
    convert = pytest.importorskip('jet_nequip.convert')
    model = os.environ.get('JET_TEST_NEQUIP_MODEL', 'nequip.net:mir-group/NequIP-OAM-S:0.1')
    kw = dict(model_path=model, device='cuda', chemical_species_to_atom_type_map={'Si': 'Si'})
    try:
        ref = ase_integration.NequIPCalculator._from_saved_model(**kw)
    except Exception as e:
        pytest.skip(f'{model}: {e}')
    jet = ase_integration.NequIPCalculator._from_saved_model(**kw)
    jet.model = convert.patch_model(jet.model)
    check(ref, jet, structure(), jet.model, ('JetTPScatter',), graph=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_nequix():
    calculator = pytest.importorskip('nequix.calculator')
    convert = pytest.importorskip('jet_nequix.convert')
    model = os.environ.get('JET_TEST_NEQUIX_MODEL', 'nequix-mp-1')
    kw = dict(backend='torch', use_kernel=False)
    try:
        ref = calculator.NequixCalculator(model, **kw)
    except Exception as e:
        pytest.skip(f'{model}: {e}')
    jet = calculator.NequixCalculator(model, **kw)
    jet.model = convert.patch_model(jet.model)
    check(ref, jet, structure(), jet.model, ('JetConvolution', 'JetTPScatter'), graph=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_equflash():
    calculator = pytest.importorskip('GGNN.common.calculator')
    convert = pytest.importorskip('jet_equflash.convert')
    ckpt = os.environ.get('EQUFLASH_CKPT')
    if not ckpt or not Path(ckpt).exists():
        pytest.skip('EQUFLASH_CKPT not set or missing')
    try:
        ref = calculator.UCalculator(checkpoint_path=ckpt, cpu=False)   # UCalculator defaults to cpu=True
    except Exception as e:
        pytest.skip(f'{ckpt}: {e}')
    jet = calculator.UCalculator(checkpoint_path=ckpt, cpu=False)
    model = getattr(jet.trainer.model, 'module', jet.trainer.model)
    convert.patch_model(model)
    check(ref, jet, structure(), model, ('JetTPScatter',), gpu_nl=False, graph=True)   # GGNN lists its pairs on the GPU itself
