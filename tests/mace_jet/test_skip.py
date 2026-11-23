"""jet_mace.skip.SpeciesLinear against the cuEq skip_tp it replaces: forward and dx on random x / one-hot node_attrs
(3 species + 2 all-zero padded rows), every interaction block of MACE-MP-0 'medium' and 'medium-mpa-0'.
CPU runs the Triton kernel through the interpreter (`TRITON_INTERPRET=1`); CUDA uses a real GPU."""
import mace  # noqa: F401  isort:skip
import pytest
import torch

from jet_mace.cueq import to_cueq_hybrid
from jet_mace.skip import SpeciesLinear

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def hybrid(name, device):
    from mace.calculators import mace_mp
    # cuEq builds its naive graph in the default dtype, so the model must be fp32 (the deploy dtype)
    return to_cueq_hybrid(mace_mp(model=name, return_raw_model=True, device=device).float(), device)


@pytest.mark.parametrize('name', ['medium', 'medium-mpa-0'])
@pytest.mark.parametrize('device', DEVICES)
def test_species_linear_matches_skip_tp(name, device):
    model = hybrid(name, device)
    torch.manual_seed(0)
    n = 300
    for blk, block in enumerate(model.interactions):
        orig = block.skip_tp
        new = SpeciesLinear.from_skip_tp(orig)
        z = torch.zeros(n, dtype=torch.long, device=device)
        z[100:200], z[200:] = 5, 27   # three species present
        na = torch.nn.functional.one_hot(z, orig.irreps_in2.num_irreps).float()
        na[-2:] = 0   # padded nodes: all-zero row
        x = torch.randn(n, orig.irreps_in1.dim, device=device)
        x0, x1 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        o0, o1 = orig(x0, na), new(x1, na)
        g = torch.randn_like(o0)
        (o0 * g).sum().backward()
        (o1 * g).sum().backward()
        dfwd, dgrad = (o0 - o1).abs().max().item(), (x0.grad - x1.grad).abs().max().item()
        print(f'{name} block {blk}: |dfwd| {dfwd:.2e} |dgrad| {dgrad:.2e}')
        assert dfwd < 1e-5 and dgrad < 1e-5, (dfwd, dgrad)


@pytest.mark.parametrize('device', DEVICES)
def test_mul_in_mul_out_mismatch(device):
    """A path with mul_in != mul_out, both below the 16-row tl.dot floor (the kernel pads): one synthetic skip_tp."""
    import cuequivariance as cue
    from cuequivariance_torch import FullyConnectedTensorProduct

    tp = FullyConnectedTensorProduct(
        cue.Irreps(cue.O3, '4x0e'), cue.Irreps(cue.O3, '16x0e'), cue.Irreps(cue.O3, '8x0e'),
        layout=cue.ir_mul, method='naive').to(device)
    with torch.no_grad():
        tp.weight.normal_(0, 0.2)
    new = SpeciesLinear.from_skip_tp(tp)
    z = torch.arange(12, device=device) % 3
    na = torch.nn.functional.one_hot(z, 16).float()
    x = torch.randn(12, 4, device=device)
    x0, x1 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    o0, o1 = tp(x0, na), new(x1, na)
    g = torch.randn_like(o0)
    (o0 * g).sum().backward()
    (o1 * g).sum().backward()
    assert (o0 - o1).abs().max() < 1e-5 and (x0.grad - x1.grad).abs().max() < 1e-5
