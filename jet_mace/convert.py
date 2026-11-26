"""Swap the per-edge radial MLP + uvu tensor product + scatter of MACE interaction blocks for jet_mlip's fused
convolution (deploy-time radial table). `patch_model(model)` attaches a `MaceTPConv` to every interaction
block; the patched `mace.modules.blocks` call it in eval mode (mace/modules/macetp_hook.py). Parameters
are untouched: the state dict stays the upstream one.
"""

import torch
from e3nn import o3
from jet_mlip import ops, reference
from jet_mlip.reference import to_ir_mul, to_mul_ir

# MACE table N: the table force error (fp64 table model vs fp64, MP-0 medium, 348-atom box) is 180 % of the official fp32
# error at 2048, 23 % at 4096, 3 % at 8192; 8192 costs 3-4 % speed and +56 % memory on MPA-0 (per-pair tables) at 5632
# atoms, so 4096 (its forces match the official fp32 ones). 2026-11-04 N scan.
DEFAULT_N = 4096

CSR = None   # (rowptr, perm) of the receivers, set by jet_mace.mliap during a step: built once per pair list there

from .configs import config_from_block, n_split
from .table import TableBuilder, interpolate


def linear_paths(lin):
    """(weight (1, P), paths) of ops.species_linear for an e3nn o3.Linear, as ops.linear_weight, but consecutive input
    blocks of one irrep that feed the same output block are one path (older models keep irreps_mid unsimplified: one
    block per tensor-product path). None if the Linear is not of that form."""
    if not (lin.shared_weights and lin.internal_weights) or any(i.i_in < 0 for i in lin.instructions):
        return None
    ii, io = lin.irreps_in, lin.irreps_out
    groups = {}
    for _, ins, w in lin.weight_views(yield_instruction=True):
        groups.setdefault(ins.i_out, []).append((ins.i_in, w.detach() * ins.path_weight))
    paths, blocks, used = [], [], set()
    for o, g in groups.items():
        g.sort(key=lambda t: t[0])
        ins = [i for i, _ in g]
        if ins != list(range(ins[0], ins[0] + len(ins))) or used & set(ins):
            return None
        used |= set(ins)
        paths += [ii.slices()[ins[0]].start, io.slices()[o].start, sum(ii[i].mul for i in ins), io[o].mul, io[o].ir.dim]
        blocks.append(torch.cat([w for _, w in g], 0).reshape(1, -1))
    if not blocks or len(blocks) > 16:
        return None
    return torch.cat(blocks, 1).contiguous(), paths


class MaceTPConv(torch.nn.Module):
    """message[receiver] += TP(node_feats[sender], SH(-vectors), W(|vectors|)) for one block (no 1/avg_num_neighbors;
    the block divides after its Linear as upstream). CUDA: jet_mlip conv_fused (table path); else a slow torch reference.
    mul > 128 (MACE-MP-0 'large', 256; MACE-OFF23 'large', 224) runs as channel slices of the mul-128 shapes,
    the largest shape jet has."""

    lin = None   # (paths, dim_out) of the block's Linear when the convolution applies it (class default: older pickles)

    def __init__(self, model, i, radial_table: int = DEFAULT_N, layout: str = 'mul_ir', species=None):
        super().__init__()
        block = model.interactions[i]
        # the kernel works in e3nn's mul_ir layout; cuEq-converted blocks hand over / expect ir_mul
        self.ir_mul = layout == 'ir_mul'
        self.blocks_x = [(mul, 2 * ir.l + 1) for mul, ir in block.conv_tp.irreps_in1]   # plain ints: dynamo cannot iterate e3nn Irreps
        self.blocks_out = [(mul, 2 * ir.l + 1) for mul, ir in block.conv_tp.irreps_out]
        # the uvu product is independent per channel u, so a mul > 128 layer is k channel slices (see n_split);
        # the full blocks above still describe the tensors around it
        self.n_split = n_split(block)
        self.cfg = config_from_block(block, div=self.n_split)
        self.n_train_calls = self.n_eval_calls = 0   # bumped inside the dynamo-opaque steps (a counter in the traced code recompiles every call)
        self.n_table = int(radial_table)
        # ponytail: training for mul > 128 not supported (the fused slices are an inference shape)
        if self.n_split > 1 and self.n_table <= 0:
            raise NotImplementedError('jet_mace: training (radial_table=0) is not supported for mul > 128')
        # plain attribute (not a submodule): rebuilds the per-pair tables for a species set on demand
        self._make_table = TableBuilder(model, i, self.n_table) if self.n_table > 0 else None
        self.r_max = float(model.radial_embedding.cutoff_fn.r_max)
        self.register_buffer('atomic_numbers', model.atomic_numbers.clone(), persistent=False)
        self.register_buffer('table', torch.zeros(0), persistent=False)
        self.register_buffer('zz', torch.zeros(0, dtype=torch.int32), persistent=False)
        self.species = None   # Z set the per-pair table was built for (Agnesi models), else None
        if radial_table > 0 and (species is not None or not hasattr(model.radial_embedding, 'distance_transform')):
            self._build_table(species)
        # the block's Linear after the convolution, on the kernel's mul_ir output (its irreps_in, the simplified irreps_mid,
        # is the same layout): one CUDA op instead of a layout copy of the layer's largest tensor, a GEMM per path and a cat
        lw = linear_paths(block.linear) if isinstance(block.linear, o3.Linear) and self.n_table > 0 else None
        if lw is not None:
            self.register_buffer('lin_w', lw[0].clone(), persistent=False)
            self.lin = (lw[1], block.linear.irreps_out.dim)   # paths, dim_out
            self.blocks_lin = [(mul, 2 * ir.l + 1) for mul, ir in block.linear.irreps_out]
        self.register_buffer('live_cols', self.cfg.live_weight_cols(), persistent=False)   # identity for MACE (all paths live)
        nw = self.cfg.n_weight   # table mode ignores the MLP weights but the op checks their shapes
        for name, shape in (('w1', (8, 64)), ('w2', (64, 64)), ('w3', (64, nw)), ('coeffs', (8,))):
            self.register_buffer(name, torch.zeros(shape), persistent=False)

    def forward(self, node_feats: torch.Tensor, vectors: torch.Tensor, edge_index: torch.Tensor,
                node_attrs: torch.Tensor):
        if self.n_table <= 0:
            return None   # built for training only: the block takes its upstream path in eval
        sender, receiver = edge_index[0], edge_index[1]
        vec = -vectors   # jet_mlip's SH convention is pos[neighbour] - pos[centre]; MACE's vectors are the opposite
        n_out = node_feats.shape[0]
        if self.ir_mul:
            node_feats = to_mul_ir(node_feats, self.blocks_x)
        out = self._conv(node_feats, vec, sender, receiver, n_out, node_attrs)   # with the block's Linear if self.lin
        return to_ir_mul(out, self.blocks_out if self.lin is None else self.blocks_lin) if self.ir_mul else out

    def _build_table(self, species):
        rt = self._make_table(species)
        tab = rt.tab.to(self.atomic_numbers.device)
        if self.n_split > 1:
            # the jet W layout is channel-contiguous per live path (column p*mul + u): keep each slice
            # (p*mul + h*mul/k + [0, mul/k)) as its own table for the sliced configs
            lead, mul, k = tab.shape[:-1], self.blocks_x[0][0], self.n_split
            t = tab.reshape(*lead, -1, mul, 4)
            tab = torch.stack([t[..., h * mul // k:(h + 1) * mul // k, :].reshape(*lead, -1).contiguous()
                               for h in range(k)])
        self.table = tab
        if rt.per_pair:
            self.zz = rt.zz.to(self.atomic_numbers.device)
            self.species = frozenset(int(z) for z in species)

    def table_idx(self, node_attrs, sender, receiver, n_out):
        """Per-edge table index for per-pair tables (None for a single table); builds / rebuilds the tables for the
        species present on the first call and whenever a new species shows up."""
        if self.zz.numel() == 0 and self.species is None and self.table.numel() > 0:
            return None
        if node_attrs.shape[0] == n_out:   # layer 0 (or a non-LAMMPS forward) sees every node: cache their species
            self._z_cache['z'] = self.atomic_numbers[node_attrs.argmax(1)]
        z = self._z_cache['z']   # LAMMPS hands layer >0 only the local rows; the edges' ghosts keep layer-0 species
        # ponytail: the check syncs (unique + tolist), illegal during capture: the eager warm-up steps already built the
        # tables, and a species first seen after capture keeps the stale one until a re-capture (LAMMPS composition fixed)
        if not (z.is_cuda and torch.cuda.is_current_stream_capturing()):
            present = frozenset(int(v) for v in torch.unique(z).tolist())
            if self.species is None or not present <= self.species:
                self._build_table(sorted(present | (self.species or set())))
        return self.zz[z[sender], z[receiver]]

    def tp3(self, node_feats: torch.Tensor, edge_attrs: torch.Tensor, tp_weights: torch.Tensor, edge_index: torch.Tensor):
        """Training path: message[receiver] += TP(node_feats[sender], edge_attrs, tp_weights) with jet_mlip's tp3
        (torch reference off CUDA). edge_attrs are MACE's own spherical harmonics, so no sign convention applies."""
        if self.n_split > 1:
            raise NotImplementedError('jet_mace: training (tp3) is not supported for mul > 128')
        self.n_train_calls += 1
        sender, receiver = edge_index[0], edge_index[1]
        if self.ir_mul:
            node_feats = to_mul_ir(node_feats, self.blocks_x)
        out = ops.tp3(self.cfg, node_feats, edge_attrs, tp_weights[:, self.live_cols], sender, receiver, node_feats.shape[0])
        return to_ir_mul(out, self.blocks_out) if self.ir_mul else out

    @torch._dynamo.disable   # jet_mlip's op has no fake-tensor rule: torch.compile treats the call as an opaque step
    def _conv(self, node_feats, vec, sender, receiver, n_out, node_attrs):
        self.n_eval_calls += 1
        idx = self.table_idx(node_attrs, sender, receiver, n_out)   # per-pair tables: host-side species bookkeeping stays opaque too
        vec = vec.contiguous()
        # the kernel needs contiguous rows (a block may hand over a column slice)
        xs = [node_feats.contiguous()] if self.n_split == 1 else self._halves(node_feats)
        tabs = [self.table] if self.n_split == 1 else list(self.table)
        fast = node_feats.is_cuda and ops.has_op('conv_fused') and node_feats.dtype == torch.float32
        if fast:
            csr = CSR   # one CSR for every slice (and, from the ML-IAP graph path, for every layer)
            rowptr, perm = csr if csr is not None and csr[0].numel() == n_out + 1 else ops.build_csr(receiver.int(), n_out)
            outs = [torch.ops.jet_mlip.conv_fused(x, vec, sender.int(), rowptr, perm, self.w1, self.w2, self.w3,
                                                    self.coeffs, 1.0, 0.0, self.r_max, 1.0, ops.config_id(self.cfg), tab, True,
                                                    None if idx is None else idx.int())
                    for x, tab in zip(xs, tabs)]
        else:
            sh = reference.spherical_harmonics(self.cfg.irreps_sh, vec)
            w = [interpolate(tab, self.r_max, vec.norm(dim=-1), idx).to(node_feats.dtype) for tab in tabs]
            outs = [node_feats.new_zeros(n_out, self.cfg.out_dim).index_add_(
                0, receiver, reference.tp3(self.cfg, x[sender], sh, wi)) for x, wi in zip(xs, w)]
        out = outs[0] if self.n_split == 1 else self._merge(outs)
        if self.lin is None:
            return out
        if fast and ops.has_op('species_linear'):
            return ops.species_linear(out, None, self.lin_w, *self.lin)
        res, wo = out.new_zeros(out.shape[0], self.lin[1]), 0
        for io, oo, U, V, d in zip(*[iter(self.lin[0])] * 5):   # torch reference of the op
            w = self.lin_w[0, wo:wo + U * V].reshape(U, V).to(out.dtype)
            res[:, oo:oo + V * d] = torch.einsum('nud,uv->nvd', out[:, io:io + U * d].reshape(-1, U, d), w).reshape(-1, V * d)
            wo += U * V
        return res

    def _halves(self, x):
        """mul_ir (N, sum mul*(2l+1): u*(2l+1)+m) -> its k channel slices, each in the sliced layers' layout."""
        n, blocks, k = x.shape[0], self.blocks_x, self.n_split
        if len(blocks) == 1 and blocks[0][1] == 1:
            h = x.shape[1] // k
            return tuple(x[:, i * h:(i + 1) * h].contiguous() for i in range(k))   # the kernel takes contiguous rows only
        parts = torch.split(x, [mul * d for mul, d in blocks], dim=1)
        return tuple(torch.cat([b.reshape(n, mul, d)[:, h * mul // k:(h + 1) * mul // k].reshape(n, -1)
                                for b, (mul, d) in zip(parts, blocks)], dim=1) for h in range(k))

    def _merge(self, outs):
        """The k slice outputs, interleaved back into the full mul_ir layout (block: u*(2l+1)+m)."""
        n, blocks, k = outs[0].shape[0], self.blocks_out, self.n_split
        if len(blocks) == 1 and blocks[0][1] == 1:
            return torch.cat(outs, dim=1)
        parts = [torch.split(o, [mul * d // k for mul, d in blocks], dim=1) for o in outs]
        return torch.cat([torch.cat([a.reshape(n, mul // k, d) for a in ps], dim=1).reshape(n, mul * d)
                          for ps, (mul, d) in zip(zip(*parts), blocks)], dim=1)


def patch_model(model, radial_table: int = DEFAULT_N, cueq: bool = False, species=None):
    """Attach fused convolutions to every interaction block of an e3nn-backend MACE model (in place; returns it).
    cueq=True first converts the rest of the model (linear, symmetric contraction, skip_tp) to cuEq kernels.
    radial_table=0: training only (no table; eval takes the upstream path). species: atomic numbers to tabulate
    for Agnesi models (default: those present at the first call, extended on demand)."""
    device = next(model.parameters()).device
    source = model
    if cueq:
        from .cueq import to_cueq_hybrid
        model = to_cueq_hybrid(model, device)
    z_cache = {}   # plain dict (not a buffer / submodule: state dict and pickling unchanged) shared by the blocks
    for i, block in enumerate(model.interactions):
        block.macetp = MaceTPConv(source, i, radial_table, layout='ir_mul' if cueq else 'mul_ir', species=species).to(device)
        block.macetp._z_cache = z_cache
        if cueq:
            from .skip import SpeciesLinear
            tp = block.skip_tp   # the one-hot (N x 0e) element selector: a per-species linear map once probed
            # ponytail: training keeps needing the original skip_tp (weights frozen here); patched models are eval-only
            if type(tp).__module__.startswith('cuequivariance') and all(mi.ir.l == 0 for mi in tp.irreps_in2._mulirreps):
                block.skip_tp = SpeciesLinear.from_skip_tp(tp)
    # eval: no block reads the spherical harmonics / the radial embedding then, and the patched model skips both
    # (the density blocks read the radial embedding; the other block types are not patched)
    model.macetp_all_fused = radial_table > 0 and all(
        type(b).__name__ in ('RealAgnosticInteractionBlock', 'RealAgnosticResidualInteractionBlock')
        for b in model.interactions)
    return model.eval()   # the fused path is eval-only (mace_mp calculators hand over models in train mode); .train() for tp3
