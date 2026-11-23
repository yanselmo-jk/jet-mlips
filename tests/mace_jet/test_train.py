"""Training path: in train mode the patched block runs jet_mlip's tp3 (per-edge radial MLP in torch, fused TP +
scatter with any-order autograd). Energy, forces and parameter gradients match the upstream e3nn model."""
import mace  # noqa: F401  isort:skip
import pytest
import torch
from ase.build import bulk

pytest.importorskip('jet_mlip')
from jet_mace import convert  # noqa: E402
from jet_mlip import ops  # noqa: E402


def batch_and_models(device, dtype):
    from mace.calculators import mace_mp
    atoms = bulk('LiF', 'rocksalt', a=4.0).repeat((2, 2, 1))
    atoms.rattle(stdev=0.1, seed=3)
    atoms.symbols[::5] = 'O'
    ref = mace_mp(model='medium', device=device, default_dtype=dtype)
    batch = ref._atoms_to_batch(atoms).to(device).to_dict()
    m0 = ref.models[0].train()
    m0.requires_grad_(True)   # the calculator froze the parameters
    acc = mace_mp(model='medium', device=device, default_dtype=dtype)
    m1 = convert.patch_model(acc.models[0], radial_table=256).train()
    m1.requires_grad_(True)
    return batch, m0, m1


def loss_and_grads(model, batch):
    out = model(dict(batch), training=True, compute_force=True)
    loss = out['energy'].sum() + (out['forces'] ** 2).sum()
    params = {n: p for n, p in model.named_parameters() if 'interactions.0' in n and ('conv_tp_weights' in n or 'linear_up' in n)}
    grads = torch.autograd.grad(loss, list(params.values()))
    return out['energy'].detach(), out['forces'].detach(), dict(zip(params, grads))


@pytest.mark.parametrize('device,dtype,tol', [('cpu', 'float64', 1e-7),
                                              pytest.param('cuda', 'float32', 1e-3, marks=pytest.mark.skipif(
                                                  not (torch.cuda.is_available() and ops.has_op('tp3')), reason='no CUDA or no JET_TRAIN=1 build'))])
def test_train_mode_uses_tp3_and_matches(device, dtype, tol):
    batch, m0, m1 = batch_and_models(device, dtype)
    e0, f0, g0 = loss_and_grads(m0, batch)
    e1, f1, g1 = loss_and_grads(m1, batch)
    assert m1.interactions[0].macetp.n_train_calls == 1
    assert torch.allclose(e1, e0, atol=tol * 10) and torch.allclose(f1, f0, atol=tol * 10)
    for n in g0:
        assert torch.allclose(g1[n], g0[n], atol=tol * g0[n].abs().max().clamp_min(1.0), rtol=tol), n
