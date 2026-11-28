"""jet_equflash.convert: the patched EquFlash model (fused table convolution) reproduces the upstream torch model's
layer outputs, energies and forces on the CPU torch reference path, for v1 (IntegratedConv) and EquFlashV2
(FullConv, lmax 4).

The checkpoints' conv_type is cueq, whose kernel is CUDA-only (`uniform_1d`), so the CPU reference is the same layers
with e3nn TensorProducts built from the e3nn view of each layer (`convert.conv_kwargs`) -- the same instructions, and
the one-hot probe in convert's docstring shows cueq consumes weight_nn's columns in that same order.
GGNN is loaded, never copied (CC BY-NC-SA).
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list

if importlib.util.find_spec('GGNN') is None:
    pytest.skip('GGNN not installed', allow_module_level=True)
CKPTS = {
    'v1': os.environ.get('EQUFLASH_CKPT'),
    'v2': os.environ.get('EQUFLASH2_CKPT')
          or str(Path.home() / 'mlip-accel/models/equflash/equflashv2_oam.pt'),
}
CKPTS = {k: v for k, v in CKPTS.items() if v and Path(v).exists()}
if not CKPTS:
    pytest.skip('EQUFLASH_CKPT / EQUFLASH2_CKPT not set', allow_module_level=True)

from jet_equflash import convert  # noqa: E402

LAYERS = ('IntegratedConv', 'FullConv')


def load_ref_model(ckpt):
    """A fresh copy of the checkpoint in fp64 with e3nn TensorProducts instead of the CUDA-only cueq kernels."""
    from GGNN.common.calculator import UCalculator
    calc = UCalculator(checkpoint_path=ckpt, cpu=True)
    return convert.to_fp64(getattr(calc.trainer.model, 'module', calc.trainer.model).eval())


def structures():
    """A rattled primitive Si diamond cell (periodic) and a Li/C/H/O molecule in a box (periodic)."""
    rng = np.random.default_rng(0)
    si = bulk('Si', 'diamond', a=5.43)
    si.positions += rng.uniform(-0.05, 0.05, si.positions.shape)
    mol = Atoms('LiCHO', positions=rng.uniform(1.0, 8.0, size=(4, 3)), cell=[10.0] * 3, pbc=True)
    return {'Si diamond': si, 'LiCHO box': mol}


def graph(atoms, cutoff, dtype=torch.float64):
    """EquFlash's batch dict from ase Atoms. edge_vec is GGNN's: pos[src] + shift - pos[dst] with edge_index =
    (dst, src) (checked against GGNN.common.utils.generate_graph's edge_vec, which is CUDA-kernel-free to reproduce)."""
    i, j, s = neighbor_list('ijS', atoms, cutoff)
    t = torch.as_tensor
    pos = t(atoms.get_positions()).to(dtype)
    return dict(atomic_numbers=t(atoms.get_atomic_numbers()).long(), pos=pos,
                cell=t(atoms.get_cell(complete=True)).view(1, 3, 3).to(dtype),
                natoms=t([len(atoms)]), batch=torch.zeros(len(atoms), dtype=torch.long),
                edge_index=torch.stack([t(i), t(j)]).long(),
                edge_vec=(pos[t(j)] + t(s @ atoms.get_cell()).to(dtype) - pos[t(i)]).requires_grad_(True))


def forward(m, d):
    out = m(dict(d))
    return out['energy'].sum(), out['forces']


@pytest.fixture(scope='module', params=sorted(CKPTS))
def models(request):
    """One checkpoint -> (tag, the fp64 e3nn reference, a builder of patched copies). Each patched model needs its own
    load: the cueq modules hold a torch.jit ScriptFunction, so the model cannot be deep-copied."""
    tag = request.param
    cache = {}

    def build(radial_table):
        if radial_table not in cache:
            cache[radial_table] = convert.patch_model(load_ref_model(CKPTS[tag]), radial_table=radial_table)
        return cache[radial_table]
    return tag, load_ref_model(CKPTS[tag]), build


@pytest.mark.parametrize('name', list(structures()))
def test_layer_matches_upstream(models, name):
    """One patched layer (the middle shape, then layer 0's and the last layer's small shapes) equals the upstream e3nn
    layer: SH + W(r) + uvu TP + scatter + /denominator, in the same layout (V2: ir_mul)."""
    tag, m, _ = models
    d = graph(structures()[name], m.edge_embedding.cutoff_function.r_cut)
    vec, (dst, src) = d['edge_vec'].detach(), d['edge_index']
    ee = m.edge_embedding({'edge_vec': vec})
    layers = [l for l in m.modules() if type(l).__name__ in LAYERS]
    for layer in (layers[1], layers[0], layers[-1]):
        x = torch.randn(len(d['pos']), convert.conv_kwargs(layer)['irreps_in1'].dim, dtype=torch.float64)
        ref = layer(x, d['edge_index'], ee['edge_embedding'], ee['edge_attr'])
        out = convert.JetTPScatter(m, layer, radial_table=0)(x, vec, src, dst)
        assert out.shape == ref.shape
        err = (out - ref).abs().max()
        assert err < 1e-8 * ref.abs().max(), f'{tag} {name}: {err.item():.3e}'


@pytest.mark.parametrize('name', list(structures()))
@pytest.mark.parametrize('radial_table', [0, 2048])
def test_patched_model_matches_upstream(models, name, radial_table):
    tag, m, build = models
    d = graph(structures()[name], m.edge_embedding.cutoff_function.r_cut)
    e0, f0 = forward(m, d)
    patched = build(radial_table)
    assert all(type(layer).__name__ == 'JetConvolution' for layer in patched.modules() if hasattr(layer, 'jet'))
    e1, f1 = forward(patched, d)
    de, df = abs(e1.item() - e0.item()), (f1 - f0).abs().max().item()
    # exact W(r) (radial_table=0) matches to fp64 round-off; the fp32 table adds its interpolation error
    assert de < (1e-10 if radial_table == 0 else 1e-6) * abs(e0.item()), f'{tag} {name}: dE {de:.3e}'
    assert df < (1e-8 if radial_table == 0 else 1e-5), f'{tag} {name}: dF {df:.3e}'
    print(f'{tag} {name} radial_table={radial_table}: dE {de:.3e} eV  max|dF| {df:.3e} eV/A')


def test_configs_are_compiled(models):
    """Every layer's mul-SLICE slice shape is in jet_mlip.configs.equflash_configs() with the model's instruction order
    (the key does not encode it)."""
    from jet_mlip.configs import equflash_configs
    tag, m, _ = models
    compiled = {c.key: c for c in equflash_configs()}
    for layer in [l for l in m.modules() if type(l).__name__ in LAYERS]:
        kw = convert.conv_kwargs(layer)
        cfg = convert.config_from_layer(layer)
        assert cfg.n_weight == convert.SLICE * len(kw['instructions'])   # every path live, one slice
        assert cfg.irreps_sh == kw['irreps_in2']
        assert cfg.key in compiled, f'{tag}: jet_mlip.configs.equflash_configs() misses {cfg.key}'
        assert compiled[cfg.key].instructions == cfg.instructions, f'{tag}: compiled fiber order differs'


@pytest.mark.skipif(not torch.cuda.is_available() or 'v1' not in CKPTS, reason='needs CUDA and EQUFLASH_CKPT')
def test_glue_layouts_agree(monkeypatch):
    """v1 on fp32 CUDA: the blocks' Linears and Gate as jet ops, with the features in ir_mul (the default: cuBLAS
    Linears) or in e3nn's mul_ir, give the e3nn glue's energy and forces."""
    from GGNN.common.calculator import UCalculator
    out = {}
    for glue in ('0', 'mul_ir', 'ir_mul'):
        monkeypatch.setenv('JET_EQUFLASH_GLUE', glue)
        calc = UCalculator(checkpoint_path=CKPTS['v1'], cpu=False)   # a fresh load each: the model cannot be copied
        m = convert.patch_model(getattr(calc.trainer.model, 'module', calc.trainer.model).eval())
        jet = [x.ir_mul for x in m.modules() if isinstance(x, (convert.JetLinear, convert.JetGate))]
        assert (len(jet) > 0) == (glue != '0') and (bool(jet) and all(jet)) == (glue == 'ir_mul')
        d = graph(structures()['LiCHO box'], m.edge_embedding.cutoff_function.r_cut, torch.float32)
        e, f = forward(m, {k: v.detach().cuda() for k, v in d.items()})
        out[glue] = e.item(), f.detach().cpu()
    for glue in ('mul_ir', 'ir_mul'):
        assert abs(out[glue][0] - out['0'][0]) < 1e-5 * abs(out['0'][0]), (glue, out[glue][0], out['0'][0])
        assert (out[glue][1] - out['0'][1]).abs().max() < 2e-5, glue
