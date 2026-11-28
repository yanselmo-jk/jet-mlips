"""gate CUDA op vs e3nn nn.Gate built like sevenn's EquivariantGate (silu / tanh scalars, silu / tanh gates; 7net-omni's
layer with odd scalars, a gated-free readout gate, an unsorted input): forward and grad x in float64, empty input,
CUDA-graph replay."""
import pytest
import torch

import jet_mlip
from jet_mlip import ops

o3 = pytest.importorskip('e3nn.o3')
from e3nn.nn import Gate  # noqa: E402

pytestmark = pytest.mark.skipif(not (jet_mlip.is_available() and ops.has_op('gate')), reason='jet_mlip._C not built')


def sevenn_gate(irreps_x):
    """sevenn EquivariantGate's construction (act_scalar = act_gate = {'e': silu, 'o': tanh})."""
    acts = {1: torch.nn.functional.silu, -1: torch.tanh}
    x = o3.Irreps(irreps_x)
    scalars = o3.Irreps([(m, ir) for m, ir in x if ir.l == 0])
    gated = o3.Irreps([(m, ir) for m, ir in x if ir.l > 0])
    p = 1 if '0e' in scalars else -1
    gates = o3.Irreps([(m, (0, p)) for m, _ in gated])
    return Gate(scalars, [acts[ir.p] for _, ir in scalars], gates, [acts[ir.p] for _, ir in gates], gated)


CASES = ['128x0e+64x1o+64x1e+32x2o+32x2e+32x3o+32x3e', '128x0o+128x0e+64x1o+64x1e+32x2o+32x2e+32x3o+32x3e', '128x0e',
         '32x1o+16x0e+8x2e+4x0o']


@pytest.mark.parametrize('irreps_x', CASES)
def test_matches_e3nn(irreps_x):
    gate = sevenn_gate(irreps_x)
    t = [v.cuda() for v in ops.gate_tables(gate)]
    torch.manual_seed(0)
    x = torch.randn(1500, gate.irreps_in.dim) * 2   # more rows than row strides of a block column (8 SMs worth)
    g = torch.randn(1500, gate.irreps_out.dim)
    xd = x.double().requires_grad_()
    ref = gate.double()(xd)
    gref, = torch.autograd.grad(ref, xd, g.double())
    xc = x.cuda().requires_grad_()
    out = ops.gate(xc, t)
    gx, = torch.autograd.grad(out, xc, g.cuda())
    assert (out.detach().cpu().double() - ref.detach()).abs().max() <= 1e-5 * ref.abs().max()
    assert (gx.cpu().double() - gref).abs().max() <= 1e-5 * gref.abs().max()


def test_empty_and_graph():
    gate = sevenn_gate(CASES[1])
    t = [v.cuda() for v in ops.gate_tables(gate)]
    assert ops.gate(torch.zeros(0, gate.irreps_in.dim, device='cuda'), t).shape == (0, gate.irreps_out.dim)
    x = torch.randn(200, gate.irreps_in.dim, device='cuda', requires_grad=True)
    g = torch.randn(200, gate.irreps_out.dim, device='cuda')
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.autograd.grad(ops.gate(x, t), x, g)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            gx, = torch.autograd.grad(ops.gate(x, t), x, g)
    torch.cuda.current_stream().wait_stream(s)
    with torch.no_grad():
        x.mul_(-1.5)
    graph.replay()
    torch.cuda.synchronize()
    xd = x.detach().cpu().double().requires_grad_()
    gref, = torch.autograd.grad(gate.double()(xd), xd, g.cpu().double())
    assert (gx.cpu().double() - gref).abs().max() <= 1e-5 * gref.abs().max()


def test_tables_on_another_device():
    gate = sevenn_gate(CASES[0])
    with pytest.raises(RuntimeError, match='tables must be on the device of x'):
        ops.gate(torch.zeros(4, gate.irreps_in.dim, device='cuda'), list(ops.gate_tables(gate)))

@pytest.mark.parametrize('irreps_x', CASES)
def test_ir_mul(irreps_x):
    """gate_tables(ir_mul=...): the same gate with x and the output in cuEq's ir_mul layout, values and grad x."""
    from jet_mlip import reference as ref
    gate = sevenn_gate(irreps_x)
    t = [v.cuda() for v in ops.gate_tables(gate, (gate.irreps_in, gate.irreps_out))]
    bi, bo = ([(mul, ir.dim) for mul, ir in irreps] for irreps in (gate.irreps_in, gate.irreps_out))
    torch.manual_seed(1)
    x, g = torch.randn(700, gate.irreps_in.dim).double() * 2, torch.randn(700, gate.irreps_out.dim).double()
    xd = x.clone().requires_grad_()
    out_ref = gate.double()(xd)
    gref, = torch.autograd.grad(out_ref, xd, g)
    xc = ref.to_ir_mul(x, bi).float().cuda().requires_grad_()
    out = ops.gate(xc, t)
    gx, = torch.autograd.grad(out, xc, ref.to_ir_mul(g, bo).float().cuda())
    assert (out.detach().cpu().double() - ref.to_ir_mul(out_ref.detach(), bo)).abs().max() <= 1e-5 * out_ref.abs().max()
    assert (gx.cpu().double() - ref.to_ir_mul(gref, bi)).abs().max() <= 1e-5 * gref.abs().max()
