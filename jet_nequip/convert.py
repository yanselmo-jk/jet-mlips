"""Swap the per-edge radial MLP + uvu tensor product + scatter of NequIP interaction blocks for jet_mlip's fused
convolution (deploy-time radial table). `patch_model(model)` attaches a `JetTPScatter` to every interaction block
and replaces the spherical-harmonic edge attrs by the raw edge vectors. Only submodules with a stable call contract
are touched: the packaged foundation models carry their own older nequip copy, so nothing is isinstance-checked
against the installed nequip. Blocks with a mul above the kernel's 128 (NequIP-OAM-XL's 320) run as 32-channel slices.
"""
import copy

import torch
from jet_mlip import ops, reference, tables
from nequip.nn.utils import with_edge_vectors_

from common.radial_table import DEFAULT_N, interpolate, make_radial_table, make_radial_tables

SLICE = 32   # channels per kernel call for a mul > 128 block: 32-channel slices are faster per layer than mul-128
# calls on H200 (measured on EquFlash)


def config_from_block(block):
    """(ConvConfig, cols) of one block: its tp instructions stably sorted by i_out, and `cols` the e3nn flat weight
    columns (original instruction order) that the sorted, live layout keeps."""
    tp = block.tp_scatter.tp
    ins = [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight) for i in tp.instructions]
    orig = [0]
    for i_in1, *_ in ins:
        orig.append(orig[-1] + tp.irreps_in1[i_in1].mul)
    order = sorted(range(len(ins)), key=lambda k: ins[k][2])   # stable: equal i_out keep e3nn's order
    cfg = tables.make_config(tp.irreps_in1, tp.irreps_in2, tp.irreps_out, [ins[k] for k in order])
    cols = torch.cat([torch.arange(orig[k], orig[k] + tp.irreps_in1[ins[k][0]].mul) for k in order])
    return cfg, cols[cfg.live_weight_cols()]


def slice_configs(block):
    """Channel slices of one block whose muls exceed the kernel's 128: [(cfg, w_cols, x_cols, out_cols)], slice h =
    channels [32h, 32h+32). The uvu product is independent per channel u, so a slice keeps the irreps_in1 entries with
    mul > 32h and their instructions, each as 32 channels. `w_cols` are the e3nn flat weight columns of the slice
    (pruned to the cfg's live paths), `x_cols` the mul_ir columns of the full x it reads, `out_cols` the full-out
    columns it fills."""
    tp = block.tp_scatter.tp
    ins = [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight) for i in tp.instructions]
    if any(tp.irreps_in1[k[0]].mul % SLICE for k in ins):
        raise NotImplementedError(f'jet_nequip: slicing needs every mul a multiple of {SLICE}, got {tp.irreps_in1}')
    if sorted(k[2] for k in ins) != list(range(len(tp.irreps_out))) \
            or any(tp.irreps_out[k[2]].mul != tp.irreps_in1[k[0]].mul for k in ins):
        raise NotImplementedError(f'jet_nequip: slicing needs one uvu instruction per output entry, got '
                                  f'{tp.irreps_out} from {tp.irreps_in1}')
    orig = [0]
    for i_in1, *_ in ins:
        orig.append(orig[-1] + tp.irreps_in1[i_in1].mul)
    order = sorted(range(len(ins)), key=lambda k: ins[k][2])   # stable: equal i_out keep e3nn's order
    slices = []
    for h in range(max(tp.irreps_in1[k[0]].mul for k in ins) // SLICE):
        lo = SLICE * h
        keep = [k for k in order if tp.irreps_in1[ins[k][0]].mul > lo]   # sorted by i_out
        x_ids = sorted({ins[k][0] for k in keep})
        x_of = {i: n for n, i in enumerate(x_ids)}
        cfg = tables.make_config(
            [(SLICE, tp.irreps_in1[i].ir) for i in x_ids], tp.irreps_in2,
            [(SLICE, tp.irreps_out[ins[k][2]].ir) for k in keep],
            [(x_of[ins[k][0]], ins[k][1], n, ins[k][3], ins[k][4]) for n, k in enumerate(keep)])
        w_cols = torch.cat([torch.arange(orig[k] + lo, orig[k] + lo + SLICE) for k in keep])
        # mul_ir columns: channel u of an irrep starts at off + u*(2l+1), so u in [lo, lo+32) is one contiguous range
        x_cols = torch.cat([torch.arange(tp.irreps_in1[:i].dim + lo * tp.irreps_in1[i].ir.dim,
                                         tp.irreps_in1[:i].dim + (lo + SLICE) * tp.irreps_in1[i].ir.dim) for i in x_ids])
        out_cols = torch.cat([torch.arange(tp.irreps_out[:j].dim + lo * tp.irreps_out[j].ir.dim,
                                           tp.irreps_out[:j].dim + (lo + SLICE) * tp.irreps_out[j].ir.dim)
                              for j in [ins[k][2] for k in keep]])
        slices.append((cfg, w_cols[cfg.live_weight_cols()], x_cols, out_cols))
    return slices


def _edge_norm(model):
    return next(m for m in model.modules() if type(m).__name__ == 'EdgeLengthNormalizer')


def _r_max(model):
    return float(_edge_norm(model).r_max)


def pair_cutoffs(model):
    """per_edge_type_cutoff models: (distinct 1/cutoff values (u,) fp64, (n_type^2,) int32 map from the row-major edge
    type t_center * n_type + t_neighbour to its value), else None. W depends on the pair only through its cutoff, so
    one table per distinct cutoff."""
    norm = _edge_norm(model)
    if not getattr(norm, '_per_edge_type', False):
        return None
    recip, inv = torch.unique(norm._rmax_recip.double().cpu().view(-1), return_inverse=True)
    return recip, inv.int()


def w_of_r(model, block, cols, rc_recip=None):
    """W(r) of one block: fp64 (m,) radii -> (m, len(cols)) in the given e3nn flat weight columns (config_from_block's
    sorted, pruned columns, or one slice's). The Bessel/cutoff module itself is copied and evaluated (the packaged
    versions may differ from the installed). rc_recip: 1/cutoff of a per-edge-type cutoff (x = r * rc_recip as upstream;
    W = 0 beyond the cutoff, as upstream's neighbour-list pruning; the bias-free edge MLP gives 0 there anyway)."""
    r_max = _r_max(model)
    enc = copy.deepcopy(next(m for m in model.modules() if type(m).__name__ == 'BesselEdgeLengthEncoding'))
    enc = enc.double().cpu()
    enc._output_dtype = torch.float64   # it casts its output to the model dtype otherwise
    mlp = copy.deepcopy(block.edge_mlp).double().cpu()
    factor = 1.0   # the ApplyFactor step between the encoding and the edge_mlp (nequip's 2 pi / r_max^2)
    for m in model.modules():
        if type(m).__name__ == 'ApplyFactor' and getattr(m, 'in_field', None) == enc.edge_invariant_field:
            factor = float(m.factor)

    def w(r):
        x = (r / r_max if rc_recip is None else r * rc_recip).unsqueeze(-1)
        out = mlp(factor * enc({enc.norm_length_field: x})[enc.edge_invariant_field])[:, cols]
        return out if rc_recip is None else out * (x < 1)
    return w


def _table(model, block, cols, n, pairs):
    if pairs is None:
        return make_radial_table(w_of_r(model, block, cols), _r_max(model), n)
    return make_radial_tables([w_of_r(model, block, cols, float(c)) for c in pairs[0]], _r_max(model), n)


class JetTPScatter(torch.nn.Module):
    """out[dst] += TP(x[src], SH(vec), W(|vec|)) for one block (the block normalizes by avg_num_neighbors before
    the scatter). vec = pos[src] - pos[dst], nequip's convention: same as jet_mlip, no sign flip. CUDA: jet_mlip
    conv_fused (table path); else the torch reference. A mul > 128 block runs as SLICE-channel slices: one table and
    one conv_fused call per slice, one CSR for all. per_edge_type_cutoff models: one table per distinct cutoff, indexed
    per edge through the edge type that EdgeVecAttrs appends as a 4th edge_attr column."""

    def __init__(self, model, block, radial_table: int = DEFAULT_N):
        super().__init__()
        tp = block.tp_scatter.tp
        self.wide = max(mul for mul, _ in tp.irreps_in1) > 128   # else one slice: the whole block, x unselected
        slices = slice_configs(block) if self.wide else [(*config_from_block(block), None, None)]
        self.cfgs = [cfg for cfg, *_ in slices]
        self.r_max = _r_max(model)
        pairs = pair_cutoffs(model)
        self.per_pair = pairs is not None
        if self.per_pair:
            self.register_buffer('pair_map', pairs[1], persistent=False)
        for i, (cfg, cols, x_cols, _) in enumerate(slices):
            if self.wide:
                self.register_buffer(f'x_idx{i}', x_cols, persistent=False)
            table = _table(model, block, cols, radial_table, pairs)
            self.register_buffer(f'table{i}', table, persistent=False)
            nw = cfg.n_weight   # table mode ignores the MLP weights but the op checks their shapes
            for name, shape in (('w1', (8, 64)), ('w2', (64, 64)), ('w3', (64, nw)), ('coeffs', (8,))):
                self.register_buffer(f'{name}{i}', torch.zeros(shape, dtype=table.dtype), persistent=False)   # nequip may leave fp64 as default
        if self.wide:
            # the slice outputs tile the full out: `cat(slice_outs, 1)[:, inv]` is the e3nn layout
            tags = torch.cat([out_cols for *_, out_cols in slices])
            assert torch.equal(tags.sort().values, torch.arange(tp.irreps_out.dim)), 'slices do not tile the block out'
            self.register_buffer('inv', torch.empty_like(tags).scatter_(0, tags, torch.arange(tags.numel())),
                                 persistent=False)

    def forward(self, x, edge_attr, edge_weight, edge_dst, edge_src):
        return self._conv(x, edge_attr, edge_src, edge_dst, x.shape[0])

    @torch._dynamo.disable   # jet_mlip's op has no fake-tensor rule: torch.compile treats the call as an opaque step
    def _conv(self, x, vec, src, dst, n_out):
        idx = None
        if self.per_pair:
            idx, vec = self.pair_map[vec[:, 3].long()], vec[:, :3]
        vec = vec.contiguous()
        xs = [x.index_select(1, self.get_buffer(f'x_idx{i}')) for i in range(len(self.cfgs))] if self.wide else [x]
        if x.is_cuda and x.dtype == torch.float32 and ops.has_op('conv_fused'):
            rowptr, perm = ops.build_csr(dst.int(), n_out)   # one CSR for all slices
            outs = [torch.ops.jet_mlip.conv_fused(xh, vec, src.int(), rowptr, perm, self.get_buffer(f'w1{i}'),
                                                  self.get_buffer(f'w2{i}'), self.get_buffer(f'w3{i}'),
                                                  self.get_buffer(f'coeffs{i}'), 1.0, 0.0, self.r_max, 1.0,
                                                  ops.config_id(self.cfgs[i]), self.get_buffer(f'table{i}'), True, idx)
                    for i, xh in enumerate(xs)]
        else:
            sh = reference.spherical_harmonics(self.cfgs[0].irreps_sh, vec)
            r = vec.norm(dim=-1)
            outs = [x.new_zeros(n_out, cfg.out_dim).index_add_(
                        0, dst, reference.tp3(cfg, xh[src], sh,
                                              interpolate(self.get_buffer(f'table{i}'), self.r_max, r, idx).to(x.dtype)))
                    for i, (cfg, xh) in enumerate(zip(self.cfgs, xs))]
        return torch.cat(outs, 1)[:, self.inv] if self.wide else outs[0]


class EdgeVecAttrs(torch.nn.Module):
    """Replaces SphericalHarmonicEdgeAttrs: the fused convolution takes the raw edge vector and makes its own SH.
    num_types > 0 (per_edge_type_cutoff models): a 4th column holds the row-major edge type (exact in fp32)."""

    def __init__(self, out_field='edge_attrs', dtype=torch.float32, num_types=0):
        super().__init__()
        self.out_field = out_field
        self.dtype = dtype
        self.num_types = num_types

    def forward(self, data):
        data = with_edge_vectors_(data, with_lengths=False)
        vec = data['edge_vectors'].to(self.dtype)
        if self.num_types:
            t, ei = data['atom_types'].view(-1), data['edge_index']
            vec = torch.cat([vec, (t[ei[0]] * self.num_types + t[ei[1]]).to(self.dtype).unsqueeze(-1)], 1)
        data[self.out_field] = vec
        return data


class _NoWeights(torch.nn.Module):
    """Empty replacement for the block's edge_mlp (its weights are in the radial table now)."""

    def forward(self, x):
        return x[:, :0]


def patch_model(model, radial_table: int = DEFAULT_N):
    """Attach fused convolutions to every InteractionBlock of a NequIP model (in place; returns it)."""
    device = next(model.parameters()).device
    n = 0
    for name, m in list(model.named_modules()):
        if type(m).__name__ == 'InteractionBlock':
            m.tp_scatter = JetTPScatter(model, m, radial_table).to(device)
            m.edge_mlp = _NoWeights()
            n += 1
        elif type(m).__name__ == 'SphericalHarmonicEdgeAttrs':
            if not (getattr(m.sh, 'normalize', True) and getattr(m.sh, 'normalization', 'component') == 'component'):
                raise NotImplementedError('jet_nequip: jet_mlip computes normalized, component-normalized SH only')
            parent = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
            # the model dtype (its first parameters can be fp64 buffers-turned-parameters, e.g. trainable Bessel weights)
            dtype = getattr(m, '_output_dtype', torch.get_default_dtype())
            nt = _edge_norm(model).num_types if pair_cutoffs(model) is not None else 0
            setattr(parent, name.rsplit('.', 1)[-1], EdgeVecAttrs(m.out_field, dtype, nt).to(device))
    if n == 0:
        raise ValueError('jet_nequip: no InteractionBlock found in the model')
    return model
