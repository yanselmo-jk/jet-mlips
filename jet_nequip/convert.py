"""Swap the per-edge radial MLP + uvu tensor product + scatter of NequIP interaction blocks for jet_mlip's fused
convolution (deploy-time radial table). `patch_model(model)` attaches a `JetTPScatter` to every interaction block,
replaces the spherical-harmonic edge attrs by the raw edge vectors and runs the e3nn glue around the convolution
(self-connection, linears, gate) as jet_mlip ops (`patch_glue`). Only submodules with a stable call contract
are touched: the packaged foundation models carry their own older nequip copy, so nothing is isinstance-checked
against the installed nequip. Blocks with a mul above the kernel's 128 (NequIP-OAM-XL's 320) run as 32-channel slices.
"""
import copy
import os
import sys
import types

import torch
from jet_mlip import ase_graph, neighbors, ops, reference, tables
from nequip.data import AtomicDataDict
from nequip.nn.utils import with_edge_vectors_

from common.radial_table import DEFAULT_N, interpolate, make_radial_table, make_radial_tables
from jet_equflash.convert import _Gather, _Scatter   # the slice column maps as one autograd node each way

CSR = None   # (rowptr, perm) of the step's edge list by dst, set by jet_nequip.mliap around a graph step (built once
# per pair list instead of once per layer); None or another size: every convolution builds its own

TYPES = None   # (N,) int64 atom type of every node row, set with CSR by jet_nequip.mliap on one rank: all rows are local

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
            # the slices' out columns tile the block out and their x columns are disjoint (_Scatter / _Gather copy each
            # way without a zero fill or an add); x columns no slice reads (an irrep without a path) are one more
            # gather output, unused: its gradient is the zero fill
            tags = torch.cat([out_cols for *_, out_cols in slices])
            assert torch.equal(tags.sort().values, torch.arange(tp.irreps_out.dim)), 'slices do not tile the block out'
            for i, (*_, out_cols) in enumerate(slices):
                self.register_buffer(f'o_idx{i}', out_cols, persistent=False)
            cols = torch.cat([x_cols for _, _, x_cols, _ in slices])
            read = torch.zeros(tp.irreps_in1.dim, dtype=torch.bool)
            read[cols] = True
            assert read.sum() == cols.numel(), 'slices share an x column'
            if not read.all():
                self.register_buffer(f'x_idx{len(slices)}', (~read).nonzero().view(-1), persistent=False)
            self.n_x, self.out_dim = len(slices) + int(not read.all()), tp.irreps_out.dim

    def forward(self, x, edge_attr, edge_weight, edge_dst, edge_src):
        return self._conv(x, edge_attr, edge_src, edge_dst, x.shape[0])

    @torch._dynamo.disable   # jet_mlip's op has no fake-tensor rule: torch.compile treats the call as an opaque step
    def _conv(self, x, vec, src, dst, n_out):
        idx = None
        if self.per_pair:
            idx, vec = self.pair_map[vec[:, 3].long()], vec[:, :3]
        vec = vec.contiguous()
        xs = _Gather.apply(x, *[self.get_buffer(f'x_idx{i}') for i in range(self.n_x)]) if self.wide else [x]
        if x.is_cuda and x.dtype == torch.float32 and ops.has_op('conv_fused'):
            csr = CSR
            if csr is None or csr[0].numel() != n_out + 1 or csr[1].numel() != dst.numel():
                csr = ops.build_csr(dst.int(), n_out)   # one CSR for all slices
            rowptr, perm = csr
            outs = [torch.ops.jet_mlip.conv_fused(xh, vec, src.int(), rowptr, perm, self.get_buffer(f'w1{i}'),
                                                  self.get_buffer(f'w2{i}'), self.get_buffer(f'w3{i}'),
                                                  self.get_buffer(f'coeffs{i}'), 1.0, 0.0, self.r_max, 1.0,
                                                  ops.config_id(cfg), self.get_buffer(f'table{i}'), True, idx)
                    for i, (cfg, xh) in enumerate(zip(self.cfgs, xs))]
        else:
            sh = reference.spherical_harmonics(self.cfgs[0].irreps_sh, vec)
            r = vec.norm(dim=-1)
            outs = [x.new_zeros(n_out, cfg.out_dim).index_add_(
                        0, dst, reference.tp3(cfg, xh[src], sh,
                                              interpolate(self.get_buffer(f'table{i}'), self.r_max, r, idx).to(x.dtype)))
                    for i, (cfg, xh) in enumerate(zip(self.cfgs, xs))]
        if not self.wide:
            return outs[0]
        # not cat(outs, 1)[:, inv]: the backward of advanced indexing is a sort-based index_put (half the OAM-XL step)
        return _Scatter.apply(self.out_dim, [self.get_buffer(f'o_idx{i}') for i in range(len(outs))], *outs)


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


class _CutoffOnly(torch.nn.Module):
    """BesselEdgeLengthEncoding without its Bessel basis: with every edge_mlp a _NoWeights nothing reads the edge
    embedding (an empty (E, 0) tensor here); the cutoff stays, the ZBL pair term reads it."""

    def __init__(self, enc, cutoff_field):
        super().__init__()
        self.enc, self.cutoff_field = enc, cutoff_field

    def forward(self, data):
        cutoff = self.enc.cutoff(data[self.enc.norm_length_field]).to(self.enc._output_dtype)
        data[self.cutoff_field] = cutoff
        data[self.enc.edge_invariant_field] = cutoff.new_zeros((cutoff.shape[0], 0))
        return data

    @classmethod
    def of(cls, enc):
        """The stub for `enc`, or `enc` itself if its forward is not 'cutoff and Bessel * cutoff' (the packaged models
        carry their own nequip): a probe must give the same cutoff, under the same key, and no other key."""
        try:
            x = torch.linspace(0.1, 0.9, 5, dtype=enc.bessel_weights.dtype, device=enc.bessel_weights.device).unsqueeze(-1)
            with torch.no_grad():
                ref = enc({enc.norm_length_field: x})
                extra = [k for k in ref if k not in (enc.norm_length_field, enc.edge_invariant_field)]
                stub = cls(enc, extra[0])
                out = stub({enc.norm_length_field: x})
            ok = len(extra) == 1 and out.keys() == ref.keys() and torch.equal(out[extra[0]], ref[extra[0]])
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError):   # another encoding contract
            ok = False
        return stub if ok else enc


class _JetGlue(torch.nn.Module):
    """An e3nn glue module of a block as one jet_mlip op on fp32 CUDA (first-order autograd in x); elsewhere (CPU,
    fp64) the e3nn module itself, kept as `self.e3nn`. The op's constant inputs are buffers folded from the weights
    at patch time: inference only."""

    def __init__(self, e3nn_module, **buffers):
        super().__init__()
        self.e3nn = e3nn_module
        for name, t in buffers.items():
            self.register_buffer(name, t, persistent=False)

    def forward(self, x, *args):
        if x.is_cuda and x.dtype == torch.float32:
            return self._jet(x, *args)
        return self.e3nn(x, *args)


class JetLinear(_JetGlue):
    """e3nn o3.Linear (linear_1 / linear_2): one GEMM per irrep in the mul_ir layout, no reshape / cat."""

    def __init__(self, lin, w, paths):
        super().__init__(lin, w=w)
        self.paths, self.dim_out = paths, lin.irreps_out.dim

    @torch._dynamo.disable   # as JetTPScatter._conv: no fake-tensor rule
    def _jet(self, x):
        return torch.ops.jet_mlip.species_linear(x, None, self.w, self.paths, self.dim_out, None)


class JetSelfConnection(_JetGlue):
    """The self-connection FullyConnectedTensorProduct(x, node_attrs): node_attrs is a function of the atom type
    (`attrs` (T, V): the type embedding rows), so the product is one linear map per type and irrep, w[t] = attrs[t] @
    (e3nn weight with its path normalization). The block hands over node_attrs, not the types: the type is the row of
    `attrs` nearest to it (equal rows have equal maps)."""

    def __init__(self, tp, w, paths, attrs):
        super().__init__(tp, w=(attrs @ w).contiguous(), key=attrs.t().contiguous(), norm=attrs.square().sum(1) / 2)
        self.paths, self.dim_out = paths, tp.irreps_out.dim

    @torch._dynamo.disable
    def _jet(self, x, node_attrs):
        species = (node_attrs @ self.key - self.norm).argmax(1)
        return torch.ops.jet_mlip.species_linear(x, species, self.w, self.paths, self.dim_out, None)


class JetGate(_JetGlue):
    """e3nn nn.Gate: one elementwise kernel per direction instead of the Extract / activation / product / cat chain."""

    def __init__(self, gate, tables):
        super().__init__(gate, **dict(zip(('fi', 'ff', 'bi', 'bf'), tables)))

    @torch._dynamo.disable
    def _jet(self, x):
        return torch.ops.jet_mlip.gate(x, self.fi, self.ff, self.bi, self.bf)


GLUE = 'sc,linear,gate'


def patch_glue(model, which=None):
    """Swap the e3nn glue around every convolution for jet_mlip ops (in place): `sc` the self-connection, `linear`
    linear_1 / linear_2, `gate` the Gate. which: comma list, default JET_NEQUIP_GLUE or all three ('' = none). A module
    the ops do not cover (ops.*_weight / gate_tables return None) and a jet_mlip built without the ops keep e3nn.
    Returns the number of modules swapped."""
    which = set((os.environ.get('JET_NEQUIP_GLUE', GLUE) if which is None else which).split(','))
    device = next(model.parameters()).device
    embed = [m for m in model.modules() if type(m).__name__ == 'NodeTypeEmbed'
             and not getattr(m, 'do_categorical_graph_field_embed', False)]
    n = 0

    def swap(parent, name, make, fold):
        nonlocal n
        folded = fold(copy.deepcopy(getattr(parent, name)).cpu())   # fold on the CPU, whatever device the model is on
        if folded is not None:
            setattr(parent, name, make(getattr(parent, name), *folded).to(device))
            n += 1

    def fold_linear(lin):
        r = ops.linear_weight(lin) if lin.weight.dtype == torch.float32 else None
        return r and r[:2]

    def fold_sc(tp):
        attrs = embed[0].embed_module.weight.detach().cpu()
        r = ops.species_linear_weight(tp) if tp.weight.dtype == torch.float32 else None
        return r and (*r, attrs)

    def fold_gate(gate):
        t = ops.gate_tables(gate)
        return t and (t,)

    for m in model.modules():
        if type(m).__name__ == 'InteractionBlock' and ops.has_op('species_linear'):
            for name in ('linear_1', 'linear_2') if 'linear' in which else ():
                if type(getattr(m, name)).__name__ == 'Linear':
                    swap(m, name, JetLinear, fold_linear)
            sc = getattr(m, 'sc', None)
            if 'sc' in which and type(sc).__name__ == 'FullyConnectedTensorProduct' and len(embed) == 1 \
                    and embed[0].embed_module.weight.shape[1] == sc.irreps_in2.dim:
                swap(m, 'sc', JetSelfConnection, fold_sc)
        elif type(m).__name__ == 'ConvNetLayer' and 'gate' in which and ops.has_op('gate') \
                and type(m.equivariant_nonlin).__name__ == 'Gate':
            swap(m, 'equivariant_nonlin', JetGate, fold_gate)
    return n


def _alpha(block):
    """The scalar a block multiplies x by before its convolution; None if it is not one scalar (per-type norm)."""
    if hasattr(block, 'scatter_norm_factor'):   # the packaged foundation models' nequip
        return 1.0 if block.scatter_norm_factor is None else float(block.scatter_norm_factor)
    norm = getattr(block, 'avg_num_neighbors_norm', None)
    return float(norm.norm_const) if norm is not None and norm.norm_const.numel() == 1 else None


@torch._dynamo.disable   # one opaque step per block, not one per op with compiled glue in between
def _local_forward(self, data):
    """InteractionBlock.forward on node rows that are all local (TYPES set: no ghost row, no exchange) as four jet ops:
    the self-connection by atom type (no node_attrs argmax), linear_1 with the scatter normalization folded in, the
    convolution, linear_2; no [:num_local] slice. Without TYPES: the upstream forward."""
    x, t = data[AtomicDataDict.NODE_FEATURES_KEY], TYPES
    if t is None or t.numel() != x.shape[0]:
        return super(type(self), self).forward(data)
    lin, ei = torch.ops.jet_mlip.species_linear, data[AtomicDataDict.EDGE_INDEX_KEY]
    sc = None if self.sc is None else lin(x, t, self.sc.w, self.sc.paths, self.sc.dim_out, None)
    x = lin(x, None, self.w1, self.linear_1.paths, self.linear_1.dim_out, None)
    x = self.tp_scatter._conv(x, data[AtomicDataDict.EDGE_ATTRS_KEY], ei[1], ei[0], x.shape[0])
    x = lin(x, None, self.linear_2.w, self.linear_2.paths, self.linear_2.dim_out, None)
    data[AtomicDataDict.NODE_FEATURES_KEY] = x if sc is None else x + sc
    return data


def patch_block_forward(model):
    """Give the InteractionBlocks of a patched CUDA model `_local_forward` (in place), for a caller that sets TYPES
    when every node row is local (jet_nequip.mliap on one rank). The packaged models carry their own nequip, so a
    block is swapped only if its glue is all jet ops and a random probe of its own forward gives the same features.
    Returns the number of blocks swapped."""
    global TYPES
    n, classes = 0, {}
    for block in [m for m in model.modules() if type(m).__name__ == 'InteractionBlock']:
        alpha, cls = _alpha(block), type(block)
        if alpha is None or not (isinstance(block.linear_1, JetLinear) and isinstance(block.linear_2, JetLinear)
                                 and isinstance(block.tp_scatter, JetTPScatter) and block.linear_1.w.is_cuda
                                 and (block.sc is None or isinstance(block.sc, JetSelfConnection))):
            continue
        block.register_buffer('w1', alpha * block.linear_1.w, persistent=False)
        N, E, kw = 6, 24, dict(device=block.w1.device, dtype=block.w1.dtype)   # (nequip leaves fp64 as torch's default)
        g = torch.Generator(kw['device']).manual_seed(0)
        t = torch.arange(N, device=kw['device']) % (1 if block.sc is None else block.sc.w.shape[0])
        vec = (torch.rand(E, 3, generator=g, **kw) - 0.5) * block.tp_scatter.r_max
        data = {AtomicDataDict.NODE_FEATURES_KEY: torch.randn(N, block.linear_1.e3nn.irreps_in.dim, generator=g, **kw),
                AtomicDataDict.NODE_ATTRS_KEY: None if block.sc is None else block.sc.key.t()[t],
                AtomicDataDict.EDGE_ATTRS_KEY: torch.cat([vec, vec[:, :1] * 0], 1) if block.tp_scatter.per_pair else vec,
                AtomicDataDict.EDGE_INDEX_KEY: torch.randint(0, N, (2, E), device=kw['device'], generator=g),
                AtomicDataDict.EDGE_EMBEDDING_KEY: vec[:, :0], AtomicDataDict.ATOM_TYPE_KEY: t,
                AtomicDataDict.LMP_MLIAP_DATA_KEY: types.SimpleNamespace(nlocal=N)}
        try:
            with torch.no_grad():
                ref = block(dict(data))[AtomicDataDict.NODE_FEATURES_KEY]
                # same class name: the blocks are found by type name
                block.__class__ = classes.setdefault(cls, type(cls.__name__, (cls,), {'forward': _local_forward}))
                TYPES = t
                out = block(dict(data))[AtomicDataDict.NODE_FEATURES_KEY]
            ok = out.shape == ref.shape and bool((out - ref).abs().max() <= 1e-4 * ref.abs().max())
        except (AttributeError, KeyError, RuntimeError, TypeError):   # another forward contract
            ok = False
        finally:
            TYPES = None
        if not ok:
            block.__class__ = cls
        n += ok
    return n


def _neighborlist(data, r_max):
    """nequip neighbour-list backend 'jet': one frame on a CUDA device through jet_mlip.neighbors, the edge_index /
    edge_cell_shift of the matscipy backend (edge order aside); anything else goes to that backend."""
    from nequip.data import _nl
    pos = data[AtomicDataDict.POSITIONS_KEY]
    if not pos.is_cuda or AtomicDataDict.num_frames(data) != 1:
        return _nl.NEIGHBORLIST_BACKEND_OPTIONS[_nl.NEIGHBORLIST_BACKEND_MATSCIPY].fn(data, r_max)
    cell, pbc = data.get(AtomicDataDict.CELL_KEY), data.get(AtomicDataDict.PBC_KEY)
    i, j, shift = neighbors.neighbor_list(pos, torch.zeros(3, 3) if cell is None else cell.view(3, 3),
                                          (False,) * 3 if pbc is None else pbc.view(3), r_max)
    data[AtomicDataDict.EDGE_INDEX_KEY] = torch.stack([i, j])
    if cell is not None:
        data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = shift.to(pos.dtype)
    return data


def _ase_graph(calc):
    """The calculator's graph step (jet_mlip.ase_graph), made at its first call: the energy model inside the model's
    ForceStressOutput (what the ML-IAP wrapper gets from disable_ForceStressOutput) on the graph's static buffers, as
    jet_nequip.mliap runs it under LAMMPS (the pair list's CSR, every block as four jet ops). False: a model without
    that wrapper, which stays eager."""
    if getattr(calc, 'jet_graph', None) is None:
        fso = getattr(calc.model, 'model', None)
        if type(fso).__name__ != 'ForceStressOutput' or not fso.do_derivatives:
            calc.jet_graph = False
            return False

        def step(energy):
            def forward(g, vec):
                global CSR, TYPES
                CSR, TYPES = (g.s_rowptr, g.s_perm), g.s_types
                try:
                    return energy({AtomicDataDict.EDGE_VECTORS_KEY: vec, AtomicDataDict.EDGE_INDEX_KEY: g.s_ei,
                                   AtomicDataDict.ATOM_TYPE_KEY: g.s_types,
                                   AtomicDataDict.LMP_MLIAP_DATA_KEY: g.lmp_stand_in,
                                   AtomicDataDict.NUM_LOCAL_GHOST_NODES_KEY: g.s_nlg})[AtomicDataDict.PER_ATOM_ENERGY_KEY]
                finally:
                    CSR = TYPES = None
            return forward

        def alloc(g):
            if not getattr(calc.model, 'jet_local', False):   # once per model, at the first capture (its probes
                # would add 30 ms to the eager first call). An eager call (TYPES unset) still takes upstream's forward
                patch_block_forward(calc.model)
                calc.model.jet_local = True
            g.s_nlg = torch.tensor(g.nlg, dtype=torch.int64, device=g.device)
        # compiled: inductor fuses the glue between the blocks (jet_nequip.mliap). H200, NequIP-OAM-M, 5632 atoms:
        # 10.57 -> 10.20 ms per call for 6.8 s of compile (cold inductor cache; 2.3 s warm), after the adapter's 600 s
        calc.jet_graph = ase_graph.AseGraph(step(fso.func), calc.device, _r_max(calc.model), vec_dtype=torch.float64,
                                            alloc=alloc, compile=lambda: step(torch.compile(fso.func, dynamic=False)))
    return calc.jet_graph


def _hook_ase():
    """ASE: the NequIPCalculator of a patched model builds its neighbour list on the GPU. Registers `_neighborlist`
    with nequip and wraps NequIPCalculator.atoms_to_data once: a calculator on a CPU-only backend whose model is
    patched switches to 'jet' and moves its transforms to the model device (nequip's own handling of a CUDA
    backend). NequIPCalculator.call_model is wrapped too: from the second call in a row with the same atom count on, a
    patched model on CUDA is replayed from a CUDA graph (_ase_graph; JET_ASE_GRAPH=0: off) on the eager path's fp64
    edge vectors, and hands calculate() the energies, forces and stress in the eager model's shapes and dtype
    (a subclass that reads other outputs in save_extra_outputs stays eager). No-op when nequip's ASE integration is
    not loaded (LAMMPS) or jet_mlip.neighbors is not available."""
    ase = sys.modules.get('nequip.integrations.ase')
    if ase is None or getattr(ase.NequIPCalculator.atoms_to_data, 'jet', False):
        return
    from nequip.data import _nl
    from nequip.data.transforms import NeighborListTransform
    _nl.register_neighborlist_backend('jet', _neighborlist, supports_cuda=True, overwrite=True)
    orig = ase.NequIPCalculator.atoms_to_data

    def atoms_to_data(self, atoms):
        if not self._move_to_device_before_transforms and getattr(self.model, 'jet_patched', False) \
                and neighbors.available(self.device):
            nl = [t for t in self.transforms if isinstance(t, NeighborListTransform)]
            if len(nl) == 1:
                nl[0].backend = 'jet'
                self.transforms = [t.to(self.device) for t in self.transforms]
                self._move_to_device_before_transforms = True
        return orig(self, atoms)
    atoms_to_data.jet = True
    ase.NequIPCalculator.atoms_to_data = atoms_to_data
    call, K = ase.NequIPCalculator.call_model, AtomicDataDict

    def call_model(self, data):
        ei = data.get(K.EDGE_INDEX_KEY)
        g = (self.device.type == 'cuda' and getattr(self.model, 'jet_patched', False) and ei is not None
             and K.CELL_KEY in data and type(self).save_extra_outputs is ase.NequIPCalculator.save_extra_outputs
             and _ase_graph(self))
        if not (g and g.due(data[K.ATOM_TYPE_KEY].numel(), ei.shape[1])):
            return call(self, data)
        pos, cell = data[K.POSITIONS_KEY], data[K.CELL_KEY].view(3, 3)
        # with_edge_vectors_'s fp64 operations, in its order
        rij = pos[ei[1]] - pos[ei[0]] + torch.sum(data[K.EDGE_CELL_SHIFT_KEY].view(-1, 3, 1) * cell, 1)
        e, f, w = g.compute(data[K.ATOM_TYPE_KEY].view(-1), ei[0], ei[1], rij, pos)
        vol = torch.linalg.det(cell).abs()   # no volume: NaN, as ForceStressOutput
        return {K.TOTAL_ENERGY_KEY: e.sum().view(1, 1), K.PER_ATOM_ENERGY_KEY: e.unsqueeze(-1), K.FORCE_KEY: f,
                K.STRESS_KEY: torch.where(vol > 0, (w + w.t()) / (2 * vol), float('nan'))}
    ase.NequIPCalculator.call_model = call_model


def patch_model(model, radial_table: int = DEFAULT_N):
    """Attach fused convolutions to every InteractionBlock of a NequIP model and jet ops to the glue around them
    (patch_glue), in place; returns it. Its ASE calculator then builds the neighbour list on the GPU (_hook_ase)."""
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
    for name, m in list(model.named_modules()):
        if type(m).__name__ == 'BesselEdgeLengthEncoding':
            parent = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
            setattr(parent, name.rsplit('.', 1)[-1], _CutoffOnly.of(m))
    patch_glue(model)
    model.jet_patched = True
    _hook_ase()
    return model
