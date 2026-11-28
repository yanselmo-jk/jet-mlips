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
    sh_calls, re_calls, skip_calls = [], [], []
    acc.models[0].spherical_harmonics.register_forward_hook(lambda *a: sh_calls.append(1))
    acc.models[0].radial_embedding.register_forward_hook(lambda *a: re_calls.append(1))
    first = acc.models[0].interactions[0]
    first.skip_tp.register_forward_hook(lambda *a: skip_calls.append(1))
    e1, f1 = acc.get_potential_energy(atoms), acc.get_forces(atoms)
    assert all(b.macetp.n_eval_calls > 0 for b in acc.models[0].interactions)   # the fused path really ran
    # a non-residual first block (all but MP-0): its linear, 1/avg_num_neighbors and skip_tp are one map per element
    # in the convolution, and the block does not run skip_tp (the residual blocks' skip_tp is their sc: still called)
    assert first.macetp.skip == (model not in ('medium', 'large')) == (not type(first).__name__.endswith('ResidualInteractionBlock'))
    assert bool(skip_calls) != first.macetp.skip
    # no fused block reads the spherical harmonics; the radial embedding is computed only for the blocks that read it
    # (the density blocks: MPA-0, MATPES)
    assert acc.models[0].macetp_no_sh and not sh_calls
    assert bool(re_calls) != acc.models[0].macetp_all_fused
    # per-pair radial tables (the density models): layer 0 gathers each edge's table for the later layers too
    assert (first.macetp._z_cache.get('idx') is not None) == (model in ('medium-mpa-0', 'mace-matpes-pbe-0'))
    assert acc.models[0].macetp_all_fused == (model in ('medium', 'large', 'off-medium', 'off-large'))
    assert all(b.macetp.lin is not None for b in acc.models[0].interactions)   # with the block's Linear
    assert all(b.macetp.n_split == {'large': 2, 'off-large': 2}.get(model, 1) for b in acc.models[0].interactions)
    assert abs(e1 - e0) / len(atoms) < tol_e, (e0, e1)
    assert np.abs(f1 - f0).max() < tol_f


def test_reshape_irreps_view_matches_cat():
    """The patched reshape_irreps returns a view for cuEq ir_mul with one mul: same values as upstream's cat."""
    from e3nn import o3
    from mace.modules.irreps_tools import reshape_irreps
    from mace.modules.wrapper_ops import CuEquivarianceConfig
    irreps = o3.Irreps('8x0e+8x1o+8x2e')
    x = torch.randn(5, irreps.dim)
    out = reshape_irreps(irreps, CuEquivarianceConfig(enabled=True, layout='ir_mul', group='O3_e3nn'))(x)
    ref = torch.cat([x[:, s].reshape(5, ir.dim, mul) for s, (mul, ir) in zip(irreps.slices(), irreps)], dim=-2)
    assert torch.equal(out, ref) and out.data_ptr() == x.data_ptr()


def test_patched_blocks_assign_their_flags():
    """Every patched block forward that reads lin_done / skip_done also assigns it (v0.5.0's attention block did
    not: a NameError for those models)."""
    import ast
    import inspect
    import mace.modules.blocks as blocks
    for fn in ast.walk(ast.parse(inspect.getsource(blocks))):
        if isinstance(fn, ast.FunctionDef):
            names = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id in ('lin_done', 'skip_done')]
            stored = {n.id for n in names if isinstance(n.ctx, ast.Store)}
            assert all(n.id in stored for n in names), fn.lineno


def test_old_mace_patch_keeps_skip_tp(monkeypatch):
    """A mace patch without macetp_skip_done would run skip_tp after a convolution that applied it: no fold then."""
    monkeypatch.setattr(convert, '_SKIP_HOOK', None)
    model = convert.patch_model(calc('off-medium', 'float64', 'cpu').models[0], radial_table=64)
    assert not model.interactions[0].macetp.skip


def test_old_mace_patch_keeps_linear(monkeypatch):
    """A mace patch without macetp_lin_done (v0.4.5) runs the block's linear itself: the convolution must not fold it,
    and the model is still the upstream one."""
    monkeypatch.setattr(convert, '_LIN_HOOK', None)
    atoms, ref, acc = rattled(), calc('medium', 'float64', 'cpu'), calc('medium', 'float64', 'cpu')
    acc.models[0] = convert.patch_model(acc.models[0], radial_table=1024)
    assert all(b.macetp.lin is None and not b.macetp.skip for b in acc.models[0].interactions)
    assert abs(acc.get_potential_energy(atoms) - ref.get_potential_energy(atoms)) / len(atoms) < 1e-7
    assert np.abs(acc.get_forces(atoms) - ref.get_forces(atoms)).max() < 1e-4


def test_folded_linear_needs_the_hook(monkeypatch):
    """A model converted with the linear folded (a deployed pickle) under a mace patch without macetp_lin_done: the
    block would apply its linear again, so the convolution raises."""
    acc = calc('medium', 'float64', 'cpu')
    acc.models[0] = convert.patch_model(acc.models[0], radial_table=64)
    assert acc.models[0].interactions[0].macetp.lin is not None
    monkeypatch.setattr(convert, '_LIN_HOOK', None)
    with pytest.raises(RuntimeError, match='block linear in the convolution'):
        acc.get_potential_energy(rattled())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('model', ['medium', 'medium-mpa-0', 'off-medium'])   # the last two: skip_tp folded into layer 0
def test_cueq_hybrid_patch_matches_upstream(model):
    """cuEq for linear / symmetric contraction / skip_tp, jet_mace for the convolution."""
    atoms = organic() if model.startswith('off-') else rattled()
    ref = calc(model, 'float32', 'cuda')
    e0, f0 = ref.get_potential_energy(atoms), ref.get_forces(atoms)
    acc = calc(model, 'float32', 'cuda')
    acc.models[0] = convert.patch_model(acc.models[0], radial_table=1024, cueq=True)
    assert type(acc.models[0].interactions[0].linear).__module__.startswith('cuequivariance')
    assert acc.models[0].interactions[0].macetp.skip == (model != 'medium')
    e1, f1 = acc.get_potential_energy(atoms), acc.get_forces(atoms)
    assert abs(e1 - e0) / len(atoms) < 1e-5 and np.abs(f1 - f0).max() < 2e-3
