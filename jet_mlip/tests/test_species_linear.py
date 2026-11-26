"""species_linear CUDA op vs e3nn FullyConnectedTensorProduct(x, one_hot(species)) in float64 (sevenn's 'nequip'
self-connection): forward and grad x for 7net-omni's irreps (first / middle / last layer shapes), ragged
multiplicities, absent species, empty input, and under CUDA-graph replay; each with both GEMM kernels."""
import pytest
import torch

import jet_mlip
from jet_mlip import ops

o3 = pytest.importorskip('e3nn.o3')
pytestmark = pytest.mark.skipif(not (jet_mlip.is_available() and ops.has_op('species_linear')),
                                reason='jet_mlip._C not built')

OMNI = '128x0e+64x1o+64x1e+32x2o+32x2e+32x3o+32x3e'
# the last: ragged row / k tiles (R % 64 in (0, 32] and (32, 64), K % 16 != 0), several row tiles, d = 1, 3, 5, 7
CASES = [(OMNI, OMNI, 89), ('128x0e', OMNI, 89), (OMNI, '128x0e', 9), ('37x0e+19x1o+70x2e', '20x0e+33x1o+5x2e+3x3o', 5),
         ('200x0e+100x1o+50x2e+40x3e', '150x0e+97x1o+17x2e+33x3e', 7)]


@pytest.fixture(autouse=True, params=['1', '0'], ids=['small', 'big'])
def kernel_variant(request, monkeypatch):
    """Both kernels on any GPU: JET_GEMM_SMALL forces the few-block one (1) or the 4-blocks-per-SM one (0)."""
    monkeypatch.setenv('JET_GEMM_SMALL', request.param)


def make(irreps_in, irreps_out, S, n, seed=0):
    torch.manual_seed(seed)
    tp = o3.FullyConnectedTensorProduct(irreps_in, f'{S}x0e', irreps_out).double()
    species = torch.randint(0, S, (n,))
    species[species == 1] = 2   # species 1 absent
    x = torch.randn(n, tp.irreps_in1.dim)
    return tp, x, species


def reference(tp, x, species, g):
    xd = x.double().requires_grad_()
    out = tp(xd, torch.nn.functional.one_hot(species, tp.irreps_in2.dim).double())
    gx, = torch.autograd.grad(out, xd, g.double())
    return out.detach(), gx


def run(tp, x, species, g):
    W, paths = ops.species_linear_weight(tp)
    xc = x.cuda().requires_grad_()
    out = ops.species_linear(xc, species.cuda(), W.float().cuda(), paths, tp.irreps_out.dim)
    gx, = torch.autograd.grad(out, xc, g.cuda())
    return out.detach().cpu().double(), gx.cpu().double()


@pytest.mark.parametrize('irreps_in,irreps_out,S', CASES)
def test_matches_e3nn(irreps_in, irreps_out, S):
    tp, x, species = make(irreps_in, irreps_out, S, 1000)
    g = torch.randn(len(x), tp.irreps_out.dim)
    ref, gref = reference(tp, x, species, g)
    out, gx = run(tp, x, species, g)
    assert (out - ref).abs().max() <= 1e-5 * ref.abs().max()
    assert (gx - gref).abs().max() <= 1e-5 * gref.abs().max()


@pytest.mark.parametrize('irreps_in,irreps_out,S', CASES)
def test_out_ir_mul(irreps_in, irreps_out, S):
    """out_ir_mul: the same numbers as the mul_ir output moved to ir_mul (reference.to_ir_mul), and the same grad x
    from the gradient handed over in ir_mul."""
    from jet_mlip.reference import to_ir_mul
    tp, x, species = make(irreps_in, irreps_out, S, 1000)
    W, paths = ops.species_linear_weight(tp)
    args = (species.cuda(), W.float().cuda(), paths, tp.irreps_out.dim)
    blocks = [(mul, ir.dim) for mul, ir in tp.irreps_out]
    g = torch.randn(len(x), tp.irreps_out.dim).cuda()
    xa, xb = x.cuda().requires_grad_(), x.cuda().requires_grad_()
    a = ops.species_linear(xa, *args)
    b = ops.species_linear(xb, *args, out_ir_mul=True)
    assert torch.equal(to_ir_mul(a.detach(), blocks), b.detach())
    ga, = torch.autograd.grad(a, xa, g)
    gb, = torch.autograd.grad(b, xb, to_ir_mul(g, blocks))
    assert torch.equal(ga, gb)


def test_many_node_tiles():
    """Few species, many nodes: several node tiles per species segment (the last one partial) in every tile shape."""
    tp, x, species = make(*CASES[-1][:2], 3, 2777, seed=3)
    g = torch.randn(len(x), tp.irreps_out.dim)
    ref, gref = reference(tp, x, species, g)
    out, gx = run(tp, x, species, g)
    assert (out - ref).abs().max() <= 1e-5 * ref.abs().max()
    assert (gx - gref).abs().max() <= 1e-5 * gref.abs().max()


@pytest.mark.parametrize('n', [16384, 16385])
def test_bucket_kernels(n):
    """Species bucketing: one block up to 16384 nodes, the count / scan / place kernels above."""
    tp, x, species = make('8x0e+4x1o', '4x0e+8x1o', 7, n, seed=4)
    g = torch.randn(len(x), tp.irreps_out.dim)
    ref, gref = reference(tp, x, species, g)
    out, gx = run(tp, x, species, g)
    assert (out - ref).abs().max() <= 1e-5 * ref.abs().max()
    assert (gx - gref).abs().max() <= 1e-5 * gref.abs().max()


def test_empty():
    tp, x, species = make(OMNI, OMNI, 3, 0)
    out, gx = run(tp, x, species, torch.zeros(0, tp.irreps_out.dim))
    assert out.shape == (0, tp.irreps_out.dim) and gx.shape == (0, tp.irreps_in1.dim)


def test_graph_replay():
    tp, x, species = make(OMNI, OMNI, 4, 700, seed=1)
    W, paths = ops.species_linear_weight(tp)
    xc, sc, Wc = x.cuda(), species.cuda(), W.float().cuda()
    g = torch.randn(len(x), tp.irreps_out.dim, device='cuda')
    xc.requires_grad_()

    def step():
        return torch.autograd.grad(ops.species_linear(xc, sc, Wc, paths, tp.irreps_out.dim), xc, g)[0]

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            gx = step()
    torch.cuda.current_stream().wait_stream(s)
    with torch.no_grad():   # replay sees new inputs, other species
        xc.mul_(-2)
        sc.copy_(torch.randint(0, 4, sc.shape))
    graph.replay()
    torch.cuda.synchronize()
    _, gref = reference(tp, xc.detach().cpu(), sc.cpu(), g.cpu())
    assert (gx.cpu().double() - gref).abs().max() <= 1e-5 * gref.abs().max()


@pytest.mark.parametrize('irreps_in,irreps_out,modal', [(OMNI, OMNI, 0), (OMNI, '128x0e', 3), ('89x0e', '128x0e', 0),
                                                         ('37x0e+19x1o+70x2e', '20x0e+33x1o+5x2e+3x3o', 2)])
def test_linear_matches_e3nn(irreps_in, irreps_out, modal):
    """species None: an e3nn Linear (sevenn IrrepsLinear), a modal one-hot input block folded into a bias."""
    torch.manual_seed(2)
    xin = o3.Irreps(irreps_in) + (o3.Irreps(f'{modal}x0e') if modal else o3.Irreps(''))
    lin = o3.Linear(xin, irreps_out).double()
    torch.nn.init.normal_(lin.weight)
    W, paths, table = ops.linear_weight(lin, modal)
    x = torch.randn(500, o3.Irreps(irreps_in).dim)
    g = torch.randn(len(x), lin.irreps_out.dim)
    oh = torch.zeros(len(x), modal, dtype=torch.float64)
    if modal:
        oh[:, 1] = 1
    xd = x.double().requires_grad_()
    ref = lin(torch.cat([xd, oh], 1))
    gref, = torch.autograd.grad(ref, xd, g.double())
    xc = x.cuda().requires_grad_()
    bias = (oh[:1] @ table).reshape(-1).float().cuda() if modal else None
    out = ops.species_linear(xc, None, W.float().cuda(), paths, lin.irreps_out.dim, bias)
    gx, = torch.autograd.grad(out, xc, g.cuda())
    assert (out.detach().cpu().double() - ref.detach()).abs().max() <= 1e-5 * ref.abs().max()
    assert (gx.cpu().double() - gref).abs().max() <= 1e-5 * gref.abs().max()
