"""jet_mlip fused convolution for EquFlash (GGNN) models.

`patch_model(model)` swaps every IntegratedConv's (v1) radial MLP + uvu tensor product + scatter + `/denominator` for
one jet module that takes (x, edge vectors, edge_index) and returns `out` in the same layout the layer's forward
returns. EquFlashV2's FullConv is patched the same way (same cue channelwise descriptor, no `transpose_in/out`). GGNN
is CC BY-NC-SA (this repo is MIT): it is not imported here, the layers are found by type name like jet_nequip.

Weight columns: the cuEq channelwise descriptor consumes weight_nn's output in the same segment order as e3nn's
instructions sorted by i_out, for v1 (4 and 23 paths) and for V2's simplified descriptor (5 and 42 paths): probed on
the GPU path by path, identity, so jet's weight layout (path prefix sums, in instruction order) needs no permutation.
`normalize_sph: null` / the V2 default leave e3nn's normalized ('component', normalize=True) spherical harmonics,
exactly jet's convention. A mul-256 v1 layer runs as 8 mul-32 channel slices (the uvu product is independent per
channel u; SLICE); a mul-128 V2 layer as 4.

V2's two other descriptor findings: (1) its FullConv output is the *simplified* mid irreps
(`640x0e+1024x1o+...`) in ir_mul layout -- the e3nn mid blocks sorted by ir, grouped, so jet's full-width mul_ir of
the sorted mid becomes V2's output with one ir_mul <-> mul_ir transpose per `layer.irreps_out` block; (2) V2 keeps
node features in ir_mul (m*mul+u: its own Linear and RMSNorm are written that way, and FullConv has no cue
transposes), so JetTPScatter transposes x into mul_ir and the output back.
"""
import copy

import torch
from e3nn import o3
from jet_mlip import ops, reference, tables

from common.radial_table import DEFAULT_N, interpolate, make_radial_table


SLICE = 32   # channels per kernel call: 8 x mul-32 is 2.8x faster than 2 x mul-128 per layer (H200, 264k edges;
# the kernel re-reads x for every path group and the group count grows with mul, so x traffic ~ mul^2)


def _is_full(layer):
    return type(layer).__name__ == 'FullConv'


def conv_kwargs(layer):
    """The layer's e3nn view (irreps_in1, irreps_in2, irreps_out, instructions). v1's IntegratedConv stores it in
    `convolution_kwargs`; a V2 FullConv only has irreps_in/filter/out, so its sorted mid irreps and uvu instruction
    list are rebuilt the IntegratedConv way: every (x, filter) product that contains an output irrep, Irreps.sort(),
    then sorted by i_out -- which the GPU probe shows is cue's weight column order."""
    if not _is_full(layer):
        return layer.convolution_kwargs
    x, f, out = (o3.Irreps(s) for s in (layer.irreps_in, layer.irreps_filter, layer.irreps_out))
    target = {ir for _, ir in out}
    ins, mid = [], []
    for i, (mul, ir1) in enumerate(x):
        for j, (_, ir2) in enumerate(f):
            for ir3 in ir1 * ir2:
                if ir3 in target:
                    ins.append((i, j, len(mid), 'uvu', True)); mid.append((mul, ir3))
    mid, p, _ = o3.Irreps(mid).sort()
    assert mid.simplify() == out, f'{mid.simplify()} != {out}'
    return dict(irreps_in1=x, irreps_in2=f, irreps_out=mid,
                instructions=sorted([(a, b, p[k], m, t) for a, b, k, m, t in ins], key=lambda q: q[2]))


def config_from_layer(layer):
    """ConvConfig of one layer, every mul cut to the SLICE channels of one kernel call."""
    kw = conv_kwargs(layer)
    x, out = (o3.Irreps([(SLICE, ir) for mul, ir in o3.Irreps(s)]) for s in (kw['irreps_in1'], kw['irreps_out']))
    return tables.make_config(x, kw['irreps_in2'], out, kw['instructions'])


def w_of_r(model, layer):
    """W(r) of one layer, fp64 (m,) radii -> (m, nW) in the upstream mul-256 weight layout. The upstream radial
    stack is copied and evaluated: Bessel * XPLOR cutoff then the FullyConnectedNet, so W = 0 at r_cut."""
    wn = copy.deepcopy(layer.weight_nn).double().cpu()
    basis = copy.deepcopy(model.edge_embedding.basis_function).double().cpu()
    cutoff = copy.deepcopy(model.edge_embedding.cutoff_function).double().cpu()

    def w(r):
        return wn(basis(r) * cutoff(r).unsqueeze(-1))
    return w


def _slices(tab, mul):
    """Full table (n, 4*NW) -> (mul/SLICE, n, 4*NW*SLICE/mul): each path's channel-contiguous columns (jet layout:
    column p*mul + u) cut into SLICE-channel pieces, one table per slice config."""
    t = tab.reshape(tab.shape[0], -1, mul, 4)
    return torch.stack([t[:, :, h:h + SLICE].reshape(t.shape[0], -1).contiguous() for h in range(0, mul, SLICE)])


class _Gather(torch.autograd.Function):
    """The slice glue as one node: index_select per slice forward, index_copy_ per slice backward. The maps partition
    the columns, so the backward needs no zero fill and no adds."""

    @staticmethod
    def forward(ctx, x, *idx):
        ctx.idx, ctx.shape = idx, x.shape
        return tuple(x.index_select(1, i) for i in idx)

    @staticmethod
    def backward(ctx, *g):
        gx = g[0].new_empty(ctx.shape)
        for gi, i in zip(g, ctx.idx):
            if gi is None:
                gx.index_fill_(1, i, 0)
            else:
                gx.index_copy_(1, i, gi)
        return (gx,) + (None,) * len(ctx.idx)


class _Scatter(torch.autograd.Function):
    """_Gather's inverse: the slice outputs into the full output columns (index_copy_ forward, index_select backward)."""

    @staticmethod
    def forward(ctx, D, idx, *outs):
        ctx.idx = idx
        out = outs[0].new_empty(outs[0].shape[0], D)
        for o, i in zip(outs, idx):
            out.index_copy_(1, i, o)
        return out

    @staticmethod
    def backward(ctx, g):
        return (None, None) + tuple(g.index_select(1, i) for i in ctx.idx)


class JetTPScatter(torch.nn.Module):
    """out[dst] += TP(x[src], SH(vec), W(|vec|)) / denominator for one layer, in mul_ir layout (a V2 FullConv gives
    and takes ir_mul x, mapped by the index maps below).
    CUDA fp32: jet_mlip conv_fused (table path, mul/SLICE channel slices); else the torch reference.
    radial_table=0: no table, the exact fp64 W(r) in the reference path (CUDA takes it too).
    in_idx / out_idx: the column maps that replace the slice/merge/transpose passes (built once here by pushing column
    ids through the old glue; they include the V2 ir_mul <-> mul_ir transpose)."""

    def __init__(self, model, layer, radial_table: int = DEFAULT_N):
        super().__init__()
        kw = conv_kwargs(layer)
        self.cfg = config_from_layer(layer)
        self.blocks_x = [(mul, 2 * ir.l + 1) for mul, ir in o3.Irreps(kw['irreps_in1'])]
        self.blocks_out = [(mul, 2 * ir.l + 1) for mul, ir in o3.Irreps(kw['irreps_out'])]
        self.ir_mul = _is_full(layer)   # V2 keeps node features ir_mul; its FullConv has no cue transposes
        if self.ir_mul:
            self.blocks_x_full = [(mul, ir.dim) for mul, ir in o3.Irreps(kw['irreps_in1'])]
            self.blocks_out_full = [(mul, ir.dim) for mul, ir in o3.Irreps(layer.irreps_out)]
        self.r_max = float(model.edge_embedding.cutoff_function.r_cut)
        self.denominator = float(layer.denominator)   # upstream divides the summed messages by this
        # per-slice column maps: the old glue applied to column ids once, at build (V2 transposed first, as before)
        ids = torch.arange(sum(m * d for m, d in self.blocks_x), dtype=torch.float64)[None]
        if self.ir_mul:
            ids = reference.to_mul_ir(ids, self.blocks_x_full)
        self.register_buffer('in_idx', torch.stack([t[0].long() for t in self._slices(ids)]), persistent=False)
        od, k = self.cfg.out_dim, self.in_idx.shape[0]
        tags = self._merge([(h * od + torch.arange(od, dtype=torch.float64))[None] for h in range(k)])
        if self.ir_mul:
            tags = reference.to_ir_mul(tags, self.blocks_out_full)
        tags = tags[0].long()
        assert torch.equal(tags.sort().values, torch.arange(k * od)), 'slice maps are not a column partition'
        out_idx = torch.empty(k * od, dtype=torch.long)
        out_idx[tags] = torch.arange(k * od)   # tag h*out_dim + p -> final column
        self.register_buffer('out_idx', out_idx.reshape(k, od), persistent=False)
        w = w_of_r(model, layer)
        if radial_table > 0:
            # grid with a node at the XPLOR onset r_on, where W'' jumps (SevenNet's ops.table_points)
            radial_table = ops.table_points(radial_table, float(model.edge_embedding.cutoff_function.r_on), self.r_max)
            self.register_buffer('table', _slices(make_radial_table(w, self.r_max, radial_table),
                                                  self.blocks_x[0][0]), persistent=False)
        else:
            self.table, self.exact_w = None, w
        nw = self.cfg.n_weight   # table mode ignores the MLP weights but the op checks their shapes
        for name, shape in (('w1', (8, 64)), ('w2', (64, 64)), ('w3', (64, nw)), ('coeffs', (8,))):
            self.register_buffer(name, torch.zeros(shape), persistent=False)

    @torch._dynamo.disable   # jet_mlip's op has no fake-tensor rule: torch.compile treats the call as an opaque step
    def forward(self, x, vec, src, dst):
        xs = _Gather.apply(x, *self.in_idx.unbind(0))
        vec = vec.contiguous()
        if x.is_cuda and x.dtype == torch.float32 and ops.has_op('conv_fused') and self.table is not None:
            rowptr, perm = ops.build_csr(dst.int(), x.shape[0])   # one CSR for all slices
            outs = [torch.ops.jet_mlip.conv_fused(xh, vec, src.int(), rowptr, perm, self.w1, self.w2, self.w3,
                                                  self.coeffs, self.denominator, 0.0, self.r_max, 1.0,
                                                  ops.config_id(self.cfg), tab, True, None)
                    for xh, tab in zip(xs, self.table)]
        else:
            r = vec.norm(dim=-1)
            sh = reference.spherical_harmonics(self.cfg.irreps_sh, vec)
            # exact W(r) stays attached to r: forces need dW/dr (the table path interpolates differentiably too)
            if self.table is not None:
                ws = [interpolate(tab, self.r_max, r) for tab in self.table]
            else:
                mul = self.blocks_x[0][0]
                wh = self.exact_w(r.double().cpu()).to(device=r.device, dtype=x.dtype)
                wh = wh.reshape(wh.shape[0], -1, mul)
                ws = [wh[:, :, h:h + SLICE].reshape(wh.shape[0], -1) for h in range(0, mul, SLICE)]
            outs = [x.new_zeros(x.shape[0], self.cfg.out_dim).index_add_(
                0, dst, reference.tp3(self.cfg, xh[src], sh, wi)) / self.denominator
                for xh, wi in zip(xs, ws)]
        out = _Scatter.apply(self.out_idx.numel(), self.out_idx.unbind(0), *outs)
        return out   # the kernel divides by `denominator` itself; the reference is divided above

    def _slices(self, x):
        """mul_ir (N, sum mul*(2l+1): u*(2l+1)+m) -> its SLICE-channel pieces, each in the slice config's layout."""
        n, mul = x.shape[0], self.blocks_x[0][0]
        parts = [b.reshape(n, m, d) for b, (m, d) in zip(torch.split(x, [m * d for m, d in self.blocks_x], dim=1), self.blocks_x)]
        return [torch.cat([b[:, h:h + SLICE].reshape(n, -1) for b in parts], dim=1) for h in range(0, mul, SLICE)]

    def _merge(self, outs):
        """The slice outputs, interleaved back into the full output mul_ir layout (block: u*(2l+1)+m)."""
        n = outs[0].shape[0]
        parts = [torch.split(o, [SLICE * d for _, d in self.blocks_out], dim=1) for o in outs]
        return torch.cat([torch.cat([p[j].reshape(n, SLICE, d) for p in parts], dim=1).reshape(n, m * d)
                          for j, (m, d) in enumerate(self.blocks_out)], dim=1)


class _E3nnFullConv(torch.nn.Module):
    """A V2 FullConv's `convolution`: e3nn TensorProduct + scatter, with the ir_mul <-> mul_ir transposes FullConv has
    no cue modules for (v1's IntegratedConv keeps them as transpose_in/out)."""

    def __init__(self, layer):
        super().__init__()
        kw = conv_kwargs(layer)
        self.tp = o3.TensorProduct(kw['irreps_in1'], kw['irreps_in2'], kw['irreps_out'],
                                   kw['instructions'], shared_weights=False, internal_weights=False)
        self.blocks_in = [(mul, ir.dim) for mul, ir in o3.Irreps(kw['irreps_in1'])]
        self.blocks_out = [(mul, ir.dim) for mul, ir in o3.Irreps(layer.irreps_out)]

    def forward(self, x, edge_attr, weight, edge_index):
        msg = self.tp(reference.to_mul_ir(x, self.blocks_in)[edge_index[1]], edge_attr, weight)
        out = x.new_zeros(x.shape[0], self.tp.irreps_out.dim).index_add_(0, edge_index[0], msg)
        return reference.to_ir_mul(out, self.blocks_out)


def to_e3nn(model):
    """Swap the cueq kernels of an EquFlash model for e3nn TensorProducts (v1: conv_type e3nn), in place: the CPU /
    fp64 reference used by the tests and the bench (the cueq `uniform_1d` kernel is CUDA-only). e3nn builds its
    Wigner constants in torch's default dtype, so call this with that dtype set to the target one. cueq consumes
    weight_nn's columns in this same e3nn instruction order, so the checkpoint's weights carry over unchanged.
    V2: FullConv gets an e3nn TP wrapped with its ir_mul transposes, and its BiLinearGate -- another CUDA-only cue
    kernel, with no e3nn mode for a channelwise uuu contraction -- is re-instantiated from the same descriptor with
    cue's torch `naive` method (matches the kernel to fp32 round-off, checked on the GPU)."""
    for layer in [l for l in model.modules() if type(l).__name__ in ('IntegratedConv', 'FullConv')]:
        if _is_full(layer):
            layer.convolution = _E3nnFullConv(layer)
            continue
        kw = conv_kwargs(layer)
        layer.convolution = o3.TensorProduct(kw['irreps_in1'], kw['irreps_in2'], kw['irreps_out'],
                                             kw['instructions'], shared_weights=False, internal_weights=False)
        layer.conv_type = 'e3nn'
        layer.transpose_in = layer.transpose_out = torch.nn.Identity()   # cueq's mul_ir <-> ir_mul transposes
    gates = [m for m in model.modules() if type(m).__name__ == 'BiLinearGate' and m.tp is not None]
    if gates:
        from cuequivariance_torch import SegmentedPolynomial
        for gate in gates:
            gate.tp = SegmentedPolynomial(gate.d.polynomial, method='naive', math_dtype=torch.get_default_dtype())
    return model


class _InputCast(torch.nn.Module):
    """GGNN's node embedding casts its one-hot input with `.float()`, which breaks an fp64 model: cast it back.
    V2 casts with `.to(input_linear.weight.dtype)` instead, hence the `weight` property."""

    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    @property
    def weight(self):
        return self.lin.weight

    def forward(self, x):
        return self.lin(x.to(self.lin.weight.dtype))


def to_fp64(model):
    """The model as an fp64 reference (tests / bench --fp64): the cueq kernels replaced by e3nn TensorProducts (they
    are CUDA-only fp32) with fp64 Wigner constants, and the one-hot node embedding input cast back to the model
    dtype. In place, like `to_e3nn`; the cueq modules hold a torch.jit ScriptFunction, so the model cannot be copied."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)   # e3nn builds its Wigner constants in the default dtype
    to_e3nn(model)
    torch.set_default_dtype(old)
    model.node_embedding.input_linear = _InputCast(model.node_embedding.input_linear)
    return model.double()


def _jet_forward(self, x, edge_index, edge_embedding, edge_attr, lmp_data=None):
    """The upstream layer minus its radial MLP / TP / scatter / denominator: one fused convolution on the layer's
    edge vectors (`patch_model` stashes them). edge_index is (dst, src), as upstream: out[dst] += TP(x[src], ...).
    LAMMPS ML-IAP: upstream's exchange first turns nlocal rows into ntotal (locals + ghosts); slice the locals back."""
    if self.use_lammps_mliap:
        x = self.exchange(x, lmp_data)
        return self.jet(x, self.edge_vec, edge_index[1], edge_index[0])[:lmp_data.nlocal]
    return self.jet(x, self.edge_vec, edge_index[1], edge_index[0])


def patch_model(model, radial_table: int = DEFAULT_N):
    """Attach fused convolutions to every IntegratedConv (v1) / FullConv (V2) of an EquFlash model (in place; returns
    it). radial_table=0 keeps the exact W(r) on the torch reference path (fp64 checks; no parameter gradients)."""
    device = next(model.parameters()).device
    layers = [m for m in model.modules() if type(m).__name__ in ('IntegratedConv', 'FullConv')]
    if not layers:
        raise ValueError('jet_equflash: no IntegratedConv / FullConv found in the model')
    classes = {}
    for layer in layers:
        layer.jet = JetTPScatter(model, layer, radial_table).to(device)
        layer.__class__ = classes.setdefault(type(layer),
                                             type('JetConvolution', (type(layer),), {'forward': _jet_forward}))
    orig = model.forward

    def forward(batch):
        for layer in layers:   # the layers only get edge_embedding / edge_attr, not the vectors we need
            layer.edge_vec = batch['edge_vec']
        return orig(batch)
    model.forward = forward
    return model
