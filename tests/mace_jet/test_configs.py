"""jet_mace.configs: a MACE interaction block's conv_tp maps onto a jet_mlip ConvConfig whose fiber
tables reproduce e3nn's uvu tensor product exactly."""
import types

import mace  # noqa: F401  isort:skip  (e3nn 0.4.4 imports only after mace sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD)
import pytest
import torch
from e3nn import o3

pytest.importorskip('jet_mlip')
from jet_mace import configs  # noqa: E402


@pytest.fixture(scope='module')
def medium():
    from mace.calculators import mace_mp
    return mace_mp(model='medium', device='cpu', default_dtype='float64', return_raw_model=True)


def test_config_dims_match_block(medium):
    for block in medium.interactions:
        cfg = configs.config_from_block(block)
        tp = block.conv_tp
        assert cfg.in_dim == tp.irreps_in1.dim and cfg.sh_dim == tp.irreps_in2.dim and cfg.out_dim == tp.irreps_out.dim
        assert cfg.n_weight == tp.weight_numel and cfg.lmax == 3


def _block(irreps_in1, irreps_out=None):
    tp = types.SimpleNamespace(irreps_in1=o3.Irreps(irreps_in1), irreps_out=o3.Irreps(irreps_out or irreps_in1))
    return types.SimpleNamespace(conv_tp=tp)


def test_n_split():
    assert configs.n_split(_block('96x0e')) == 1
    assert configs.n_split(_block('128x0e')) == 1
    assert configs.n_split(_block('224x0e+224x1o', '224x0e')) == 2
    assert configs.n_split(_block('256x0e')) == 2
    assert configs.n_split(_block('1024x0e')) == 8
    with pytest.raises(ValueError):
        configs.n_split(_block('256x0e+128x1o'))   # mixed muls
    with pytest.raises(ValueError):
        configs.n_split(_block('132x0e'))   # no k gives a multiple of 8 (132/2..66 all leave 66, 44, 33, ...)


def test_fibers_reproduce_e3nn_tp(medium):
    from jet_mlip import reference
    torch.manual_seed(0)
    for block in medium.interactions:
        cfg = configs.config_from_block(block)
        tp = block.conv_tp
        E = 7
        x = torch.randn(E, tp.irreps_in1.dim, dtype=torch.float64)
        sh = o3.spherical_harmonics(tp.irreps_in2, torch.randn(E, 3, dtype=torch.float64), normalize=True, normalization='component')
        w = torch.randn(E, tp.weight_numel, dtype=torch.float64)
        ref = tp(x, sh, w)
        out = reference.tp3(cfg, x, sh, w[:, cfg.live_weight_cols()])
        assert torch.allclose(out, ref, atol=1e-10, rtol=1e-10)


def test_model_configs_are_unique_keys():
    cfgs = configs.model_configs(configs.load('medium'))
    assert len(cfgs) == 2 and len({c.key for c in cfgs}) == 2


def test_layout_transposes_match_permutation():
    """to_ir_mul / to_mul_ir (block transposes, cheap backward) equal the index permutation."""
    from e3nn import o3
    from jet_mace import convert
    irreps = o3.Irreps('128x0e+128x1o+64x2e')
    x = torch.randn(5, irreps.dim)
    # x_ir_mul = x_mul_ir[:, p] (e3nn mul_ir: off + u*(2l+1) + m; cuEq ir_mul: off + m*mul + u)
    p, off = [], 0
    for mul, ir in irreps:
        p += [off + u * (2 * ir.l + 1) + m for m in range(2 * ir.l + 1) for u in range(mul)]
        off += mul * (2 * ir.l + 1)
    assert torch.equal(convert.to_ir_mul(x, irreps), x[:, p])
    assert torch.equal(convert.to_mul_ir(x[:, p], irreps), x)
