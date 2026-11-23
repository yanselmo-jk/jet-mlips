"""Reference layer == sevenn e3nn modules (fp64, 1e-10) on the 7net-nano and 7net-0 configs."""
import math
import os
import re

import pytest
import torch
import jet_mlip  # noqa: F401  isort:skip  (sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD before e3nn 0.4.4 loads)
from e3nn import o3

from jet_mlip import reference, tables

torch.manual_seed(0)


def random_graph(n=20, e=200, in_dim=32, dtype=torch.float64):
    x = torch.randn(n, in_dim, dtype=dtype)
    dst = torch.randint(0, n, (e,))
    src = torch.randint(0, n, (e,))
    vec = torch.randn(e, 3, dtype=dtype) * 2.0
    vec[:5] *= 3  # some beyond cutoff
    return x, vec, src, dst


def load_fp64(name):
    """Pretrained model (name or checkpoint path) in fp64. The checkpoint is fp32, so e3nn's Wigner-3j buffers are
    rebuilt in fp64 (7net-0's old-format checkpoint cannot be loaded with a float64 default dtype)."""
    pytest.importorskip('sevenn')   # lazy: the MACE env has no sevenn, and test_conv_fused imports this module
    from sevenn.util import model_from_checkpoint, pretrained_name_to_path
    try:
        path = name if os.path.exists(name) else pretrained_name_to_path(name)
    except Exception as e:   # offline box (sevenn downloads 7net-nano on first use) or a sevenn without the name
        pytest.skip(f'{name} unavailable ({type(e).__name__}); copy its .pth to ~/.cache/sevennet/SevenNet_nano/')
    m, cfg = model_from_checkpoint(path)
    m = m.double().eval()
    for mod in m.modules():
        for bname, _ in list(mod.named_buffers(recurse=False)):
            mm = re.fullmatch(r'_w3j_(\d+)_(\d+)_(\d+)', bname)
            if mm:
                setattr(mod, bname, o3.wigner_3j(*map(int, mm.groups()), dtype=torch.float64))
    return m, cfg


@pytest.fixture(scope='module')
def nano():
    return load_fp64('7net-nano-5.5')


@pytest.fixture(scope='module')
def net0():
    return load_fp64('7net-0')


@pytest.mark.parametrize('model, layer', [('nano', 0), ('nano', 1), ('nano', 2)] + [('net0', i) for i in range(5)])
def test_conv_matches_sevenn(request, model, layer):
    KEY = pytest.importorskip('sevenn._keys')
    m, mcfg = request.getfixturevalue(model)
    conv = getattr(m, f'{layer}_convolution')
    lin = getattr(m, f'{layer}_self_interaction_2')
    emb = m.edge_embedding
    cfg = tables.config_from_sevenn_conv(conv, lin.linear.irreps_out)
    x, vec, src, dst = random_graph(in_dim=cfg.in_dim)

    # sevenn path
    data = {KEY.EDGE_VEC: vec, KEY.NODE_FEATURE: x, KEY.EDGE_IDX: torch.stack([dst, src])}
    data = emb(data)
    ref = conv(data)[KEY.NODE_FEATURE]

    w1, w2, w3 = (conv.weight_nn.layer0.weight, conv.weight_nn.layer1.weight, conv.weight_nn.layer2.weight)
    cst = conv.weight_nn.layer0.act.cst
    out = reference.conv(cfg, x, vec, src, dst, w1, w2, reference.prune_w3(cfg, w3),
                         emb.basis_function.coeffs, emb.cutoff_function.r_on,
                         emb.cutoff_function.r_cut, cst, conv.denominator)
    mask = cfg.out_mask()
    assert torch.allclose(out[:, mask], ref[:, mask], atol=1e-10, rtol=1e-10)
    assert (out[:, ~mask] == 0).all()
    # dead blocks really are ignored by the next linear
    assert torch.allclose(lin.linear(out), lin.linear(ref), atol=1e-10)


def test_nano_config_shapes(nano):
    m, _ = nano
    cfgs = tables.sevenn_nano_configs()
    for layer, cfg in enumerate(cfgs):
        conv = getattr(m, f'{layer}_convolution')
        lin = getattr(m, f'{layer}_self_interaction_2')
        c2 = tables.config_from_sevenn_conv(conv, lin.linear.irreps_out)
        assert cfg.key == c2.key
    assert [c.n_paths for c in cfgs] == [3, 11, 3]
    assert [len(c.fibers) for c in cfgs] == [9, 83, 9]
    assert cfgs[1].out_dim == 1632 and cfgs[1].n_weight == 352


def test_net0_config_shapes(net0):
    m, _ = net0
    h = '128x0e+64x1e+32x2e'
    cfgs = tables.sevenn_model_configs(['128x0e', h, h, h, h, '128x0e'], lmax=2, parity=False)
    for layer, cfg in enumerate(cfgs):
        conv = getattr(m, f'{layer}_convolution')
        lin = getattr(m, f'{layer}_self_interaction_2')
        assert cfg.key == tables.config_from_sevenn_conv(conv, lin.linear.irreps_out).key
    assert [c.n_weight for c in cfgs] == [384, 960, 960, 960, 224]
    assert [(p.mul, p.wcol) for p in cfgs[1].paths][:4] == [(128, 0), (64, 128), (32, 192), (128, 224)]
    assert cfgs[1].in_dim == 480 and cfgs[1].out_dim == 3136 and len(cfgs[1].fibers) == 137
    # e3nn flat weight layout: prefix sum of mul_x over all instructions (all live here)
    assert cfgs[1].live_weight_cols().tolist() == list(range(960))
    c = tables.sevenn_model_configs(['32x0e', '32x0e+16x1o', '32x0e'], lmax=1)[1]   # mul 16 path
    assert [(p.mul, p.wcol) for p in c.paths] == [(32, 0), (16, 32)] and c.n_weight == 48


def test_bad_mul():
    pytest.importorskip('sevenn')   # sevenn_model_configs builds the layers with sevenn
    with pytest.raises(ValueError):
        tables.sevenn_model_configs(['4x0e', '4x0e+4x1e', '4x0e'], lmax=1, parity=False)
    c = tables.sevenn_model_configs(['8x0e', '16x0e+8x1e', '8x0e'], lmax=1, parity=False)[1]   # mul 8 paths
    assert [(p.mul, p.wcol) for p in c.paths] == [(16, 0), (8, 16)] and c.n_weight == 24
    c = tables.sevenn_model_configs(['32x0e', '32x0e'], lmax=0, parity=False)[0]   # lmax 0: sh = 1x0e
    assert c.sh_dim == 1 and c.lmax == 0 and c.n_paths == 1 and len(c.fibers) == 1
    with pytest.raises(ValueError):
        tables.sevenn_model_configs(['256x0e', '256x0e', '256x0e'], lmax=1, parity=False)


def test_header_emits():
    pytest.importorskip('sevenn')
    h = tables.emit_header(tables.sevenn_nano_configs())
    assert 'JET_FOREACH_CONFIG' in h and 'NFIBER = 83' in h
