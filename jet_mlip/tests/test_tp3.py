"""tp3 CUDA op vs jet_mlip.reference.tp3: forward (fp64 1e-10, fp32 1e-5), gradcheck,
gradgradcheck, and a force-style double-backward through e3nn spherical harmonics + MLP."""
import pytest
import torch
import jet_mlip  # isort:skip  (sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD before e3nn 0.4.4 loads)
from e3nn import o3

from jet_mlip import ops, reference
from jet_mlip.configs import selected
from jet_mlip.shapes import compiled_keys

pytestmark = pytest.mark.skipif(not (jet_mlip.is_available() and ops.has_op('tp3')), reason='jet_mlip._C not built')
torch.manual_seed(0)
# nondet_tol: scatter-add atomics reorder fp64 sums (~1e-16 rel), like CUDA index_add_
ALL, COMPILED = selected('all'), compiled_keys()
CFGS = [c for c in ALL if c.key in COMPILED]   # every family's shapes, or those of JET_CONFIGS=<family>
# three nano layers + 7net-0 layer 1 (mul 128/64/32); the ones this build lacks skip
NANO = [pytest.param(c, id=c.key[:40], marks=pytest.mark.skipif(c.key not in COMPILED, reason='config not in this build'))
        for c in ALL[:3] + [c for c in ALL if c.key.startswith('128x0e+64x1e+32x2e|1x0e+1x1e+1x2e|128x0e+64x0e')]]


def graph(cfg, n, e, dtype, device='cuda'):
    x = torch.randn(n, cfg.in_dim, dtype=dtype, device=device)
    sh = torch.randn(e, cfg.sh_dim, dtype=dtype, device=device)
    W = torch.randn(e, cfg.n_weight, dtype=dtype, device=device)
    src = torch.randint(0, n, (e,), device=device).int()
    dst = torch.randint(0, n, (e,), device=device).int()
    return x, sh, W, src, dst


def ref_tp3(cfg, x, sh, W, src, dst, n_out):
    msg = reference.tp3(cfg, x[src.long()], sh, W)
    return x.new_zeros(n_out, cfg.out_dim).index_add(0, dst.long(), msg)


def cuda_tp3(cfg, x, sh, W, src, dst, n_out):
    return torch.ops.jet_mlip.tp3(x, sh, W, src, dst, n_out, ops.config_id(cfg))


@pytest.mark.parametrize('cfg', CFGS, ids=[c.key[:40] for c in CFGS])
def test_forward_fp64(cfg):
    x, sh, W, src, dst = graph(cfg, 30, 300, torch.float64)
    out = cuda_tp3(cfg, x, sh, W, src, dst, 30)
    assert torch.allclose(out, ref_tp3(cfg, x, sh, W, src, dst, 30), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('cfg', CFGS, ids=[c.key[:40] for c in CFGS])
def test_forward_fp32(cfg):
    x, sh, W, src, dst = graph(cfg, 30, 300, torch.float64)
    ref = ref_tp3(cfg, x, sh, W, src, dst, 30)
    out = cuda_tp3(cfg, x.float(), sh.float(), W.float(), src, dst, 30)
    assert ((out.double() - ref).norm() / ref.norm()).item() < 1e-5


def test_n_out_and_empty():
    cfg = CFGS[0]
    x, sh, W, src, dst = graph(cfg, 6, 25, torch.float64)
    dst = dst.clamp(max=3)
    assert torch.allclose(cuda_tp3(cfg, x, sh, W, src, dst, 4), ref_tp3(cfg, x, sh, W, src, dst, 4), atol=1e-10)
    out = cuda_tp3(cfg, x, sh[:0], W[:0], src[:0], dst[:0], 6)
    assert out.shape == (6, cfg.out_dim) and (out == 0).all()


@pytest.mark.parametrize('sort', [False, True], ids=['unsorted_dst', 'sorted_dst'])
def test_isolated_node_and_n_out(sort):
    """OUT slot goes through the dst CSR: node 2 has no edges, n_out < N, dst sorted or not."""
    cfg = CFGS[1]
    x, sh, W, src, dst = graph(cfg, 8, 40, torch.float64)
    dst = dst.clamp(max=4)
    dst[dst == 2] = 3
    if sort:
        dst = dst.sort().values
    out = cuda_tp3(cfg, x, sh, W, src, dst, 5)
    assert (out[2] == 0).all()
    assert torch.allclose(out, ref_tp3(cfg, x, sh, W, src, dst, 5), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('cfg', NANO)
def test_first_order_vs_reference(cfg):
    x, sh, W, src, dst = graph(cfg, 6, 25, torch.float64)
    g = torch.randn(6, cfg.out_dim, dtype=torch.float64, device='cuda')
    ins = [t.clone().requires_grad_() for t in (x, sh, W)]
    ref = torch.autograd.grad((ref_tp3(cfg, *ins, src, dst, 6) * g).sum(), ins)
    got = torch.autograd.grad((cuda_tp3(cfg, *ins, src, dst, 6) * g).sum(), ins)
    for a, b in zip(ref, got):
        assert torch.allclose(a, b, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('cfg', NANO)
def test_gradcheck(cfg):
    x, sh, W, src, dst = graph(cfg, *((4, 10) if cfg.n_weight > 500 else (6, 25)), torch.float64)
    ins = [t.requires_grad_() for t in (x, sh, W)]
    f = lambda x, sh, W: cuda_tp3(cfg, x, sh, W, src, dst, x.shape[0])  # noqa: E731
    assert torch.autograd.gradcheck(f, ins, nondet_tol=1e-10)
    assert torch.autograd.gradgradcheck(f, ins, nondet_tol=1e-10)


@pytest.mark.parametrize('cfg', NANO)
def test_grad_ops_gradcheck(cfg):
    """The three partial ops are themselves differentiable to second order."""
    n, e = (4, 10) if cfg.n_weight > 500 else (6, 25)
    x, sh, W, src, dst = graph(cfg, n, e, torch.float64)
    g = torch.randn(n, cfg.out_dim, dtype=torch.float64, device='cuda')
    cid = ops.config_id(cfg)
    for f, ins in [
        (lambda g, sh, W: torch.ops.jet_mlip.tp3_x(g, sh, W, src, dst, n, cid), (g, sh, W)),
        (lambda g, x, sh: torch.ops.jet_mlip.tp3_w(g, x, sh, src, dst, cid), (g, x, sh)),
        (lambda g, x, W: torch.ops.jet_mlip.tp3_sh(g, x, W, src, dst, cid), (g, x, W)),
    ]:
        ins = [t.clone().requires_grad_() for t in ins]
        assert torch.autograd.gradcheck(f, ins, nondet_tol=1e-10)
        assert torch.autograd.gradgradcheck(f, ins, nondet_tol=1e-10)


def test_force_loss_double_backward():
    """E = sum(tp3(x, SH(vec), MLP(|vec|))); F = dE/dvec; (F^2).sum().backward() must
    give finite grads for x and the MLP, matching the pure-torch reference path."""
    cfg = CFGS[1]  # nano layer 1 (11 paths)
    torch.manual_seed(1)
    n, e = 6, 25
    dev, dt = 'cuda', torch.float64
    mlp = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.SiLU(), torch.nn.Linear(16, cfg.n_weight)).to(dev, dt)
    x0 = torch.randn(n, cfg.in_dim, dtype=dt, device=dev)
    vec0 = torch.randn(e, 3, dtype=dt, device=dev)
    src = torch.randint(0, n, (e,), device=dev).int()
    dst = torch.randint(0, n, (e,), device=dev).int()

    def run(tp):
        x = x0.clone().requires_grad_()
        vec = vec0.clone().requires_grad_()
        mlp.zero_grad()
        sh = o3.spherical_harmonics(cfg.irreps_sh, vec, normalize=True, normalization='component')
        W = mlp(vec.norm(dim=-1, keepdim=True))
        E = tp(cfg, x, sh, W, src, dst, n).sum()
        (F,) = torch.autograd.grad(E, vec, create_graph=True)
        (F ** 2).sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in mlp.parameters()) and torch.isfinite(x.grad).all()
        return [F.detach(), x.grad.clone()] + [p.grad.clone() for p in mlp.parameters()]

    ref, got = run(ref_tp3), run(cuda_tp3)
    for a, b in zip(ref, got):
        assert ((a - b).norm() / a.norm()).item() < 1e-6
