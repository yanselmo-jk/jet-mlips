"""Instruction / Clebsch-Gordan fiber tables for the fused SevenNet convolution.

A "config" fully describes one `IrrepsConvolution` layer: node irreps (x), edge
spherical-harmonics irreps (sh), the e3nn tensor-product output irreps (out,
full width, as produced by `IrrepsConvolution.__init__`), the uvu instructions
sorted by i_out, and which output irreps are *live* (consumed by the following
e3nn Linear). Dead paths are dropped from the weight layout and the fiber list;
the output tensor keeps the full width with zeros in the dead blocks so the rest
of the SevenNet model is untouched.

Layouts (all e3nn "mul_ir"):
  x   [N, in_dim]   index = off1 + u*(2*l1+1) + m1
  sh  [E, sh_dim]   index = off2 + m2                (mul 1)
  W   [E, n_weight] index = wcol + u                (wcol = prefix sum of live path muls)
  out [N, out_dim]  index = off3 + u*(2*l3+1) + m3
u ranges over the path's multiplicity mul = irreps_x[i1].mul (uvu: out mul = in mul);
muls may differ between irreps (8 | mul, mul <= 128).
Every fiber is (p, m1, m2, m3, coef) with coef = path_weight * wigner_3j[m1,m2,m3]
(path_weight is e3nn's normalized instruction weight, sqrt(2*l3+1) for uvu with
component normalization); out[c] += coef * x[a] * sh[b] * W[w].
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')   # e3nn 0.4.4 (mace env) loads constants.pt with torch.load
from e3nn import o3  # noqa: E402

Instr = Tuple[int, int, int, str, bool]  # (i_in1, i_in2, i_out, mode, has_weight)


@dataclass
class Path:
    p: int          # live path index
    p_orig: int     # index in the original (sorted-by-i_out) instruction list
    mul: int        # channels (= irreps_x[i1].mul)
    wcol: int       # first column in the pruned W layout (prefix sum of live muls)
    i1: int
    i2: int
    i3: int
    l1: int
    l2: int
    l3: int
    off1: int
    off2: int
    off3: int
    path_weight: float


@dataclass
class ConvConfig:
    key: str
    irreps_x: o3.Irreps
    irreps_sh: o3.Irreps
    irreps_out: o3.Irreps                 # full e3nn irreps_mid (sorted)
    instructions: List[Instr]             # original, sorted by i_out
    live_out: List[bool]                  # per irreps_out entry
    paths: List[Path] = field(default_factory=list)
    fibers: List[Tuple[int, int, int, int, float]] = field(default_factory=list)

    @property
    def in_dim(self) -> int:
        return self.irreps_x.dim

    @property
    def sh_dim(self) -> int:
        return self.irreps_sh.dim

    @property
    def out_dim(self) -> int:
        return self.irreps_out.dim

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    @property
    def n_weight(self) -> int:
        return sum(pp.mul for pp in self.paths)

    @property
    def n_weight_full(self) -> int:
        return sum(self.irreps_x[i[0]].mul for i in self.instructions)

    @property
    def lmax(self) -> int:
        return max(ir.l for _, ir in self.irreps_x + self.irreps_sh + self.irreps_out)

    def live_weight_cols(self) -> torch.Tensor:
        """Original weight columns (e3nn flat layout) kept after pruning."""
        # e3nn flat layout: prefix sum of mul_x over ALL instructions in instruction order
        orig = [0]
        for i1, *_ in self.instructions:
            orig.append(orig[-1] + self.irreps_x[i1].mul)
        cols = [orig[pp.p_orig] + u for pp in self.paths for u in range(pp.mul)]
        return torch.tensor(cols, dtype=torch.long)

    def out_mask(self) -> torch.Tensor:
        m = torch.zeros(self.out_dim, dtype=torch.bool)
        for k, (mul_ir, live) in enumerate(zip(self.irreps_out, self.live_out)):
            if live:
                o = self.irreps_out[:k].dim
                m[o:o + mul_ir.dim] = True
        return m


def config_key(irreps_x, irreps_sh, irreps_out, live_out) -> str:
    live = ''.join('1' if v else '0' for v in live_out)
    return f'{irreps_x}|{irreps_sh}|{irreps_out}|{live}'.replace(' ', '')


def make_config(
    irreps_x,
    irreps_sh,
    irreps_out,
    instructions: Sequence[Instr],
    live_irreps=None,
) -> ConvConfig:
    """Build a config. `instructions` must be sorted by i_out (sevenn convention).
    `live_irreps`: irreps consumed downstream (e.g. the next Linear's irreps_out);
    an output entry is live iff its irrep appears there. None = all live."""
    irreps_x, irreps_sh, irreps_out = o3.Irreps(irreps_x), o3.Irreps(irreps_sh), o3.Irreps(irreps_out)
    if any(mul % 8 != 0 or mul > 128 for mul, _ in irreps_x):
        raise ValueError(f'jet needs every mul in irreps_x to be a multiple of 8 <= 128, got {irreps_x}')
    if any(m != 1 for m, _ in irreps_sh):
        raise ValueError('irreps_sh must have multiplicity 1')
    instructions = [tuple(i[:5]) for i in instructions]
    if [i[2] for i in instructions] != sorted(i[2] for i in instructions):
        raise ValueError('instructions must be sorted by i_out')
    if live_irreps is None:
        live_out = [True] * len(irreps_out)
    else:
        live_set = {ir for _, ir in o3.Irreps(live_irreps)}
        live_out = [ir in live_set for _, ir in irreps_out]

    # e3nn's normalized path weights
    tp = o3.TensorProduct(
        irreps_x, irreps_sh, irreps_out, instructions,
        shared_weights=False, internal_weights=False,
    )
    cfg = ConvConfig(config_key(irreps_x, irreps_sh, irreps_out, live_out),
                     irreps_x, irreps_sh, irreps_out, instructions, live_out)
    p, wcol = 0, 0
    for p_orig, (ins, eins) in enumerate(zip(instructions, tp.instructions)):
        i1, i2, i3, mode, has_w = ins
        assert mode == 'uvu' and has_w, ins
        if not live_out[i3]:
            continue
        l1, l2, l3 = irreps_x[i1].ir.l, irreps_sh[i2].ir.l, irreps_out[i3].ir.l
        mul = irreps_x[i1].mul
        assert irreps_out[i3].mul == mul, 'uvu: out mul == in mul'
        path = Path(p, p_orig, mul, wcol, i1, i2, i3, l1, l2, l3,
                    irreps_x[:i1].dim, irreps_sh[:i2].dim, irreps_out[:i3].dim,
                    float(eins.path_weight))
        cfg.paths.append(path)
        w3j = o3.wigner_3j(l1, l2, l3, dtype=torch.float64)
        for m1 in range(2 * l1 + 1):
            for m2 in range(2 * l2 + 1):
                for m3 in range(2 * l3 + 1):
                    c = w3j[m1, m2, m3].item()
                    if c != 0.0:
                        cfg.fibers.append((p, m1, m2, m3, path.path_weight * c))
        p += 1
        wcol += mul
    return cfg


def config_from_sevenn_conv(conv, live_irreps=None) -> ConvConfig:
    """From a sevenn `IrrepsConvolution` (or fused variant) instance."""
    kw = conv.convolution_kwargs
    return make_config(kw['irreps_in1'], kw['irreps_in2'], kw['irreps_out'], kw['instructions'], live_irreps)


def sevenn_model_configs(irreps_manual: Sequence, lmax: int, parity: bool = True) -> List[ConvConfig]:
    """Layer configs of a SevenNet nequip-style model given sevenn's `irreps_manual`
    (n_layers + 1 irreps: layer t maps irreps_manual[t] -> irreps_manual[t+1]; the first
    and last entries are scalars). lmax/parity: spherical harmonics of the edges."""
    from sevenn.nn.convolution import IrrepsConvolution

    sh = o3.Irreps.spherical_harmonics(lmax, -1 if parity else 1)
    irreps_manual = [o3.Irreps(i) for i in irreps_manual]
    cfgs = []
    for irreps_x, irreps_next in zip(irreps_manual[:-1], irreps_manual[1:]):
        lmax_out = irreps_next.lmax
        # sevenn's irreps_out_tp filter (parity 'full', 'even' for the scalar last layer); only its
        # irrep *types* matter to IrrepsConvolution, the live set is the next gate's input types
        if lmax_out == 0:
            irreps_out_filter = o3.Irreps('0e')
        else:
            irreps_out_filter = o3.Irreps([(1, (l, pp)) for l in range(lmax_out + 1) for pp in (1, -1)])
        conv = IrrepsConvolution(irreps_x, sh, irreps_out_filter, [8, 64, 64])
        cfgs.append(config_from_sevenn_conv(conv, irreps_next))
    return cfgs


def sevenn_nano_configs(lmax: int = 2, mul: int = 32, parity: bool = True) -> List[ConvConfig]:
    """The three layer configs of a SevenNet nequip-style model (nano: lmax 2, mul 32, O3)."""
    hidden = o3.Irreps([(mul, (l, (-1) ** l if parity else 1)) for l in range(lmax + 1)])
    scalar = o3.Irreps(f'{mul}x0e')
    return sevenn_model_configs([scalar, hidden, hidden, scalar], lmax, parity)


# ----------------------------------------------------------------------------
# C++ header emission (compile-time constexpr tables)
# ----------------------------------------------------------------------------

def _cname(key: str) -> str:
    import hashlib
    return 'Cfg_' + hashlib.md5(key.encode()).hexdigest()[:10]


def emit_header(configs: Sequence[ConvConfig]) -> str:
    out = ['// Generated by jet_mlip/tables.py. Do not edit.', '#pragma once',
           '#include "config.h"', '']
    names = []
    for cfg in configs:
        n = _cname(cfg.key)
        names.append(n)
        out.append(f'// key: {cfg.key}')
        out.append(f'struct {n} {{')
        out.append(f'  static constexpr int IN_DIM = {cfg.in_dim};')
        out.append(f'  static constexpr int SH_DIM = {cfg.sh_dim};')
        out.append(f'  static constexpr int OUT_DIM = {cfg.out_dim};')
        out.append(f'  static constexpr int NPATH = {cfg.n_paths};')
        out.append(f'  static constexpr int NFIBER = {len(cfg.fibers)};')
        out.append(f'  static constexpr int NW = {cfg.n_weight};')
        out.append(f'  static constexpr int LMAX = {cfg.lmax};')
        out.append('  static constexpr PathT paths[NPATH] = {')
        for pp in cfg.paths:
            out.append(f'    {{{pp.l1}, {pp.l2}, {pp.l3}, {pp.off1}, {pp.off2}, {pp.off3}, {pp.mul}, {pp.wcol}}},')
        out.append('  };')
        out.append('  static constexpr FiberT fibers[NFIBER] = {')
        for (p, m1, m2, m3, c) in cfg.fibers:
            out.append(f'    {{{p}, {m1}, {m2}, {m3}, {c!r}}},')
        out.append('  };')
        out.append('};')
        out.append('')
    out.append('// Dispatch: config id -> struct, in the order of jet_mlip/configs.py')
    out.append('#define JET_FOREACH_CONFIG(X) \\')
    for i, n in enumerate(names):
        out.append(f'  X({i}, {n}) \\')
    out.append('')
    out.append(f'static constexpr int JET_NUM_CONFIGS = {len(names)};')
    out.append('static const char* JET_CONFIG_KEYS[] = {')
    for cfg in configs:
        out.append(f'  "{cfg.key}",')
    out.append('};')
    return '\n'.join(out) + '\n'
