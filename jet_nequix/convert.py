"""Swap the per-edge radial MLP + uvu tensor product + scatter of Nequix convolutions for jet_mlip's fused
convolution (deploy-time radial table). `patch_model(model)` attaches a `JetTPScatter` to every
NequixTorchConvolution and makes it aggregate in one kernel; the layer's linear_1 / skip / layer_norm / gate are
untouched. Nequix is not isinstance-checked: the layer is found by type name, like jet_nequip.
"""
import copy
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from e3nn import o3
from jet_mlip import ops, reference, tables

from common.radial_table import DEFAULT_N, interpolate, make_radial_table


def _load_nequix_model():
    """`nequix.torch_impl.model` loaded from its files, without running nequix/__init__.py (it imports jax, whose
    virtual-address footprint does not fit the 4 GB ulimit the CPU checks run under; torch alone maps 3.5 GB) nor
    torch_impl/__init__.py (only torch.compile flags). Only torch / e3nn / scipy are needed."""
    if 'nequix.torch_impl.model' in sys.modules:
        return sys.modules['nequix.torch_impl.model']
    root = Path(importlib.util.find_spec('nequix').submodule_search_locations[0]) / 'torch_impl'
    for pkg, path in (('nequix', root.parent), ('nequix.torch_impl', root)):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(path)]
            sys.modules[pkg] = m
    for name in ('layer_norm', 'model'):   # model.py imports layer_norm at module level
        spec = importlib.util.spec_from_file_location(f'nequix.torch_impl.{name}', root / f'{name}.py')
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        setattr(sys.modules['nequix.torch_impl'], name, m)
    return sys.modules['nequix.torch_impl.model']


_NQ = _load_nequix_model()
NequixTorchConvolution = _NQ.NequixTorchConvolution
bessel_basis, polynomial_cutoff = _NQ.bessel_basis, _NQ.polynomial_cutoff
load_model = _NQ.load_model   # load with use_kernel=False: the /sqrt(avg_n_neighbors) reference path this module replaces


def _pad8(mul):
    return (mul + 7) // 8 * 8


def _padded(irreps):
    return o3.Irreps([(_pad8(mul), ir) for mul, ir in irreps])


def _pads(orig, padded):
    """Columns of `orig`'s entries inside the same irreps with every mul rounded up to 8 (None if unchanged)."""
    if all(a.mul == b.mul for a, b in zip(orig, padded)):
        return None
    return torch.cat([torch.arange(padded[:k].dim, padded[:k].dim + mir.dim) for k, mir in enumerate(orig)])


def config_from_layer(layer):
    """(ConvConfig, maps) of one layer. Nequix permutes its TP instructions with the inverse of tp_irreps.sort(),
    so instruction k writes sorted output block k: jet's i_out is the instruction index and the radial-MLP weight
    columns keep their order (`sort`'s layout is what linear_2 consumes). `maps` = (input columns, (radial columns,
    weight columns), output columns) of the mul -> pad8(mul) rounding jet's kernel requires, None where unchanged.
    The model's 89-species one-hot input (mul 89) is the only padded layer."""
    tp = layer.tp
    ins = [(i.i_in1, i.i_in2, k, 'uvu', True) for k, i in enumerate(tp.instructions)]
    cfg = tables.make_config(_padded(tp.irreps_in1), tp.irreps_in2, _padded(layer.sort.irreps_sorted), ins)
    wcols = None
    if cfg.n_weight != tp.weight_numel:
        src, dst, o0, o1 = [], [], 0, 0
        for p, i in enumerate(tp.instructions):
            mul = tp.irreps_in1[i.i_in1].mul
            src += list(range(o0, o0 + mul))
            dst += list(range(o1, o1 + mul))
            o0, o1 = o0 + mul, o1 + cfg.paths[p].mul
        wcols = (torch.tensor(src), torch.tensor(dst))
    return cfg, (_pads(tp.irreps_in1, cfg.irreps_x), wcols, _pads(layer.sort.irreps_sorted, cfg.irreps_out))


def w_of_r(model, layer, cfg, wcols):
    """W(r) of one layer: fp64 (m,) radii -> (m, cfg.n_weight) in cfg's weight layout. The upstream radial stack is
    copied and evaluated: the basis already carries the polynomial cutoff (p from the model config) and the MLP has
    no bias, so W = 0 at the cutoff."""
    mlp = copy.deepcopy(layer.radial_mlp).double().cpu()
    r_max, p, n_basis = float(model.cutoff), float(model.radial_polynomial_p), model.radial_basis_size

    def w(r):
        out = mlp(bessel_basis(r, n_basis, r_max) * polynomial_cutoff(r, r_max, p)[:, None])
        if wcols is None:
            return out
        ww = out.new_zeros(r.shape[0], cfg.n_weight)
        ww[:, wcols[1]] = out[:, wcols[0]]
        return ww
    return w


class JetTPScatter(torch.nn.Module):
    """out[dst] += TP(x[src], SH(vec), W(|vec|)) / sqrt(avg_n_neighbors) for one layer, in the layer's `sort`
    layout (Nequix's TP already writes that order). CUDA fp32: jet_mlip conv_fused (table path); else the torch
    reference. radial_table=0: no table, the exact fp64 W(r) in the reference path (CUDA takes it too)."""

    def __init__(self, model, layer, radial_table: int = DEFAULT_N):
        super().__init__()
        self.cfg, (xcols, self.wcols, outcols) = config_from_layer(layer)
        # buffers, not attributes: they move with .to(device) (a CPU index tensor breaks CUDA-graph capture)
        self.register_buffer('xcols', xcols, persistent=False)
        self.register_buffer('outcols', outcols, persistent=False)
        self.r_max = float(model.cutoff)
        self.denominator = float(np.sqrt(layer.avg_n_neighbors))   # upstream divides the summed messages by this
        w = w_of_r(model, layer, self.cfg, self.wcols)
        if radial_table > 0:
            self.register_buffer('table', make_radial_table(w, self.r_max, radial_table), persistent=False)
        else:
            self.table, self.exact_w = None, w
        nw = self.cfg.n_weight   # table mode ignores the MLP weights but the op checks their shapes
        for name, shape in (('w1', (8, 64)), ('w2', (64, 64)), ('w3', (64, nw)), ('coeffs', (8,))):
            self.register_buffer(name, torch.zeros(shape), persistent=False)

    @torch._dynamo.disable   # jet_mlip's op has no fake-tensor rule: torch.compile treats the call as an opaque step
    def forward(self, x, vec, src, dst):
        if self.xcols is not None:   # jet needs every mul a multiple of 8: the padded channels of x stay zero
            xp = x.new_zeros(x.shape[0], self.cfg.in_dim)
            xp[:, self.xcols] = x
            x = xp
        if x.is_cuda and x.dtype == torch.float32 and ops.has_op('conv_fused') and self.table is not None:
            rowptr, perm = ops.build_csr(dst.int(), x.shape[0])
            out = torch.ops.jet_mlip.conv_fused(x, vec.contiguous(), src.int(), rowptr, perm, self.w1, self.w2,
                                                  self.w3, self.coeffs, self.denominator, 0.0, self.r_max, 1.0,
                                                  ops.config_id(self.cfg), self.table, True, None)
        else:
            r = vec.norm(dim=-1)
            sh = reference.spherical_harmonics(self.cfg.irreps_sh, vec)
            # exact W(r) stays attached to r: forces need dW/dr (the table path interpolates differentiably too)
            w = (interpolate(self.table, self.r_max, r) if self.table is not None
                 else self.exact_w(r.double().cpu()).to(device=r.device, dtype=x.dtype))
            out = x.new_zeros(x.shape[0], self.cfg.out_dim)
            out = out.index_add_(0, dst, reference.tp3(self.cfg, x[src], sh, w)) / self.denominator
        return out if self.outcols is None else out[:, self.outcols]


class JetConvolution(NequixTorchConvolution):
    """The upstream layer with radial_mlp + tp + scatter replaced by `self.jet`. `sh` / `radial_basis` (computed
    once per model forward upstream) are unused: the fused convolution makes its own from the edge vectors, which
    `patch_model` stashes on the layer."""

    def forward(self, features, species, sh, radial_basis, senders, receivers):
        messages_agg = self.jet(self.linear_1(features), self.edge_vec, senders, receivers)
        skip = self.skip(species, features) if self.index_weights else self.skip(features)
        features = self.linear_2(messages_agg) + skip
        if self.layer_norm is not None:
            features = self.layer_norm(features)
        return self.gate(features)


def patch_model(model, radial_table: int = DEFAULT_N):
    """Attach fused convolutions to every NequixTorchConvolution of a NequixTorch model (in place; returns it).
    radial_table=0 keeps the exact W(r) on the torch reference path (fp64 checks; no parameter gradients)."""
    device = next(model.parameters()).device
    layers = [m for m in model.modules() if type(m).__name__ == 'NequixTorchConvolution']
    if not layers:
        raise ValueError('jet_nequix: no NequixTorchConvolution found in the model')
    for layer in layers:
        if layer.kernel:
            raise NotImplementedError('jet_nequix: load with use_kernel=False (openequivariance conv)')
        layer.jet = JetTPScatter(model, layer, radial_table).to(device)
        layer.__class__ = JetConvolution
    orig = model.node_energies

    def node_energies(displacements, species, senders, receivers):
        for layer in layers:   # the layers only get sh / radial basis, not the vectors the fused convolution needs
            layer.edge_vec = displacements
        return orig(displacements, species, senders, receivers)
    model.node_energies = node_energies
    return model
