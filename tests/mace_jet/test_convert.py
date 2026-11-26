"""jet_mace.convert.patch_model: the patched MACE-MP-0 model (fused table convolution) reproduces the
upstream e3nn model's energy and forces. Runs the kernel on CUDA, the torch reference on CPU."""
import mace  # noqa: F401  isort:skip
import numpy as np
import pytest
import torch
from ase.build import bulk, molecule

pytest.importorskip('jet_mlip')
from jet_mace import convert  # noqa: E402


def calc(name, dtype, device, **kw):
    from mace.calculators import mace_mp, mace_off
    if name.startswith('off-'):
        return mace_off(model=name[4:], device=device, default_dtype=dtype, **kw)
    return mace_mp(model=name, device=device, default_dtype=dtype, **kw)


def rattled():
    atoms = bulk('LiF', 'rocksalt', a=4.0).repeat((3, 3, 2))
    atoms.rattle(stdev=0.15, seed=1)
    atoms.symbols[::7] = 'O'
    return atoms


def organic():
    """Two ethanol copies 3.5 A apart per axis (no pbc): a cluster of MACE-OFF23 elements."""
    a = molecule('CH3CH2OH')
    b = a.copy()
    b.positions += 3.5
    atoms = a + b
    atoms.rattle(stdev=0.05, seed=1)
    atoms.pbc = False
    atoms.center(vacuum=5.0)
    return atoms


@pytest.mark.parametrize('model', ['medium', 'medium-mpa-0', 'large', 'mace-matpes-pbe-0', 'off-medium', 'off-large'])
@pytest.mark.parametrize('device,dtype,tol_e,tol_f', [('cpu', 'float64', 1e-7, 1e-4),
                                                      pytest.param('cuda', 'float32', 1e-5, 2e-3, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA'))])
def test_patched_model_matches_upstream(model, device, dtype, tol_e, tol_f):
    atoms = organic() if model.startswith('off-') else rattled()
    ref = calc(model, dtype, device)
    e0, f0 = ref.get_potential_energy(atoms), ref.get_forces(atoms)
    acc = calc(model, dtype, device)
    acc.models[0] = convert.patch_model(acc.models[0], radial_table=1024)
    sh_calls = []
    acc.models[0].spherical_harmonics.register_forward_hook(lambda *a: sh_calls.append(1))
    e1, f1 = acc.get_potential_energy(atoms), acc.get_forces(atoms)
    assert all(b.macetp.n_eval_calls > 0 for b in acc.models[0].interactions)   # the fused path really ran
    # the edge embeddings are computed only for blocks that read them (the density blocks: MPA-0, MATPES)
    assert bool(sh_calls) != acc.models[0].macetp_all_fused
    assert acc.models[0].macetp_all_fused == (model in ('medium', 'large', 'off-medium', 'off-large'))
    assert all(b.macetp.lin is not None for b in acc.models[0].interactions)   # with the block's Linear
    assert all(b.macetp.n_split == {'large': 2, 'off-large': 2}.get(model, 1) for b in acc.models[0].interactions)
    assert abs(e1 - e0) / len(atoms) < tol_e, (e0, e1)
    assert np.abs(f1 - f0).max() < tol_f


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_cueq_hybrid_patch_matches_upstream():
    """cuEq for linear / symmetric contraction / skip_tp, jet_mace for the convolution."""
    atoms = rattled()
    ref = calc('medium', 'float32', 'cuda')
    e0, f0 = ref.get_potential_energy(atoms), ref.get_forces(atoms)
    acc = calc('medium', 'float32', 'cuda')
    acc.models[0] = convert.patch_model(acc.models[0], radial_table=1024, cueq=True)
    assert type(acc.models[0].interactions[0].linear).__module__.startswith('cuequivariance')
    e1, f1 = acc.get_potential_energy(atoms), acc.get_forces(atoms)
    assert abs(e1 - e0) / len(atoms) < 1e-5 and np.abs(f1 - f0).max() < 2e-3
