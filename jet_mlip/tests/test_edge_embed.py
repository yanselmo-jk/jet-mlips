"""edge_embed CUDA op vs sevenn EdgeEmbedding: fp64 1e-12 for lmax 0..3, gradcheck /
gradgradcheck wrt edge_vec and coeffs, and a force-style double backward vs the torch path."""
import pytest
import torch
from sevenn.nn.edge_embedding import BesselBasis, EdgeEmbedding, SphericalEncoding, XPLORCutoff

import jet_mlip
from jet_mlip import ops, reference

pytestmark = pytest.mark.skipif(not (jet_mlip.is_available() and ops.has_op('edge_embed')), reason='jet_mlip._C not built')
torch.manual_seed(0)
R_ON, R_CUT = 4.0, 5.0


def sevenn_embed(lmax, dtype):
    m = EdgeEmbedding(BesselBasis(R_CUT), XPLORCutoff(R_CUT, R_ON), SphericalEncoding(lmax)).to(dtype).cuda()
    return m


def vecs(e, dtype):
    v = torch.randn(e, 3, dtype=dtype, device='cuda')
    return v / v.norm(dim=-1, keepdim=True) * torch.rand(e, 1, dtype=dtype, device='cuda') * 5.5  # r in (0, 5.5): r_on / r_cut both sides


@pytest.mark.parametrize('lmax', [0, 1, 2, 3])
def test_forward_fp64(lmax):
    m = sevenn_embed(lmax, torch.float64)
    v = vecs(1000, torch.float64)
    B, sh = ops.edge_embed(v, m.basis_function.coeffs, R_ON, R_CUT, lmax)
    ref = m({'edge_vec': v})
    assert torch.allclose(B, ref['edge_embedding'], atol=1e-12, rtol=1e-12)
    assert torch.allclose(sh, ref['edge_attr'], atol=1e-12, rtol=1e-12)
    assert sh.shape == (1000, (lmax + 1) ** 2)


def test_forward_fp32():
    m = sevenn_embed(2, torch.float32)
    v = vecs(1000, torch.float32)
    B, sh = ops.edge_embed(v, m.basis_function.coeffs, R_ON, R_CUT, 2)
    ref = m({'edge_vec': v})
    assert torch.allclose(B, ref['edge_embedding'], atol=1e-6, rtol=1e-5)
    assert torch.allclose(sh, ref['edge_attr'], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('lmax', [1, 3])
def test_gradcheck(lmax):
    v = vecs(20, torch.float64).requires_grad_()
    c = sevenn_embed(lmax, torch.float64).basis_function.coeffs.detach().requires_grad_()
    f = lambda v, c: ops.edge_embed(v, c, R_ON, R_CUT, lmax)  # noqa: E731
    assert torch.autograd.gradcheck(f, (v, c))
    assert torch.autograd.gradgradcheck(f, (v, c))


def test_force_double_backward():
    """loss = |dE/dpos|^2 with E = sum(w_B * B + w_sh * sh): grads of w_B, w_sh, coeffs and pos
    must match the sevenn (e3nn) path through two backward passes."""
    lmax = 2
    m = sevenn_embed(lmax, torch.float64)
    v0 = vecs(300, torch.float64)
    wB = torch.randn(8, dtype=torch.float64, device='cuda')
    wsh = torch.randn(9, dtype=torch.float64, device='cuda')

    def run(use_op):
        v = v0.clone().requires_grad_()
        c = m.basis_function.coeffs.detach().clone().requires_grad_()
        a, b = wB.clone().requires_grad_(), wsh.clone().requires_grad_()
        if use_op:
            B, sh = ops.edge_embed(v, c, R_ON, R_CUT, lmax)
        else:
            B = reference.bessel_cutoff(v.norm(dim=-1), c, R_ON, R_CUT)  # == sevenn (test_reference.py)
            sh = m.spherical(v)
        E = (B * a).sum() + (sh * b).sum() + (B.pow(2) * sh[:, :1]).sum()
        (F,) = torch.autograd.grad(E, v, create_graph=True)
        loss = F.pow(2).sum() + E
        loss.backward()
        return [t.grad.clone() for t in (v, c, a, b)]

    for g0, g1 in zip(run(False), run(True)):
        assert torch.allclose(g0, g1, atol=1e-9, rtol=1e-9)


def test_reference_fallback_matches():
    m = sevenn_embed(3, torch.float64)
    v = vecs(100, torch.float64)
    c = m.basis_function.coeffs
    B, sh = ops.edge_embed(v, c, R_ON, R_CUT, 3)
    Bc, shc = ops.edge_embed(v.cpu(), c.cpu(), R_ON, R_CUT, 3)
    assert torch.allclose(B.cpu(), Bc, atol=1e-12) and torch.allclose(sh.cpu(), shc, atol=1e-12)


# ---- silu_norm: cst * silu(z), fused fwd / bwd / bwd-bwd ----
CST = 1.6791767923989418  # normalize2mom(silu).cst


@pytest.mark.skipif(not ops.has_op('silu_norm'), reason='silu_norm not built')
def test_silu_norm():
    z = torch.randn(64, 32, dtype=torch.float64, device='cuda', requires_grad=True)
    assert torch.allclose(ops.silu_norm(z, CST), torch.nn.functional.silu(z) * CST, atol=1e-14)
    f = lambda z: ops.silu_norm(z, CST)  # noqa: E731
    assert torch.autograd.gradcheck(f, (z,))
    assert torch.autograd.gradgradcheck(f, (z,))
    zf = z.detach().float().requires_grad_()  # fp32 double backward vs torch
    y0, y1 = [torch.autograd.grad(torch.autograd.grad(fn(zf).pow(2).sum(), zf, create_graph=True)[0].pow(3).sum(), zf)[0]
              for fn in (f, lambda z: torch.nn.functional.silu(z) * CST)]
    assert torch.allclose(y0, y1, rtol=1e-4, atol=1e-5)
    with pytest.raises(RuntimeError, match='third-order'):
        (g,) = torch.autograd.grad(f(z).sum(), z, create_graph=True)
        (gg,) = torch.autograd.grad(g.sum(), z, create_graph=True)
