"""Pure-torch reference of the fused SevenNet convolution layer.

Numerically identical (up to summation order) to sevenn's
`EdgeEmbedding` (Bessel * XPLOR, spherical harmonics) + `IrrepsConvolution`
(e3nn FullyConnectedNet radial MLP + uvu TensorProduct + scatter + /denominator).
Slow (explicit fiber loops); used only for tests.
"""
from __future__ import annotations

import math

import torch
from e3nn import o3

from .tables import ConvConfig


def radial_mlp(B, w1, w2, w3, cst: float):
    """e3nn FullyConnectedNet([8,64,64,n], silu): weights are the raw parameters
    (layer{0,1,2}.weight, shape (h_in, h_out)); cst = normalize2mom(silu).cst."""
    h = B @ (w1 / math.sqrt(w1.shape[0]))
    h = torch.nn.functional.silu(h) * cst
    h = h @ (w2 / math.sqrt(w2.shape[0]))
    h = torch.nn.functional.silu(h) * cst
    return h @ (w3 / math.sqrt(w3.shape[0]))


def bessel_cutoff(r, coeffs, r_on: float, r_cut: float):
    """sevenn BesselBasis (prefactor 2/rc, sin(c r)/r) * XPLORCutoff(r_on, rc). r: (E,)"""
    ur = r.unsqueeze(-1)
    B = (2.0 / r_cut) * torch.sin(coeffs * ur) / ur
    r_sq = r * r
    ron2, rc2 = r_on * r_on, r_cut * r_cut
    cut = torch.where(r < r_on, torch.ones_like(r),
                      (rc2 - r_sq) ** 2 * (rc2 + 2 * r_sq - 3 * ron2) / (rc2 - ron2) ** 3)
    return B * cut.unsqueeze(-1)


def spherical_harmonics(irreps_sh, edge_vec):
    return o3.spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization='component')


def tp3(cfg: ConvConfig, x_e, sh, W):
    """Per-edge message. x_e: (E, in_dim) gathered node features, sh: (E, sh_dim),
    W: (E, n_weight) pruned per-edge weights -> (E, out_dim) full width."""
    E = x_e.shape[0]
    out = x_e.new_zeros(E, cfg.out_dim)
    for (p, m1, m2, m3, c) in cfg.fibers:
        pp = cfg.paths[p]
        u = torch.arange(pp.mul, device=x_e.device)
        a = pp.off1 + u * (2 * pp.l1 + 1) + m1
        b = pp.off2 + m2
        w = pp.wcol + u
        cc = pp.off3 + u * (2 * pp.l3 + 1) + m3
        out[:, cc] += c * x_e[:, a] * sh[:, b:b + 1] * W[:, w]
    return out


def conv(cfg: ConvConfig, x, edge_vec, src, dst, w1, w2, w3, coeffs, r_on, r_cut, cst, denominator):
    """Full fused layer: out[dst] += TP(x[src], SH(vec), MLP(bessel(|vec|))) ; out /= denominator.
    w3 is the *pruned* last-layer weight (64, n_weight)."""
    r = edge_vec.norm(dim=-1)
    B = bessel_cutoff(r, coeffs, r_on, r_cut)
    sh = spherical_harmonics(cfg.irreps_sh, edge_vec)
    W = radial_mlp(B, w1, w2, w3, cst)
    msg = tp3(cfg, x[src], sh, W)
    out = x.new_zeros(x.shape[0], cfg.out_dim)
    out.index_add_(0, dst, msg)
    return out / denominator


def prune_w3(cfg: ConvConfig, w3_full):
    """(64, n_weight_full) -> (64, n_weight) keeping live columns."""
    return w3_full[:, cfg.live_weight_cols().to(w3_full.device)].contiguous()


def _blocks(irreps):
    """[(mul, 2l+1)] from an e3nn Irreps (or pass such a list straight through)."""
    return [(mul, 2 * ir.l + 1) for mul, ir in irreps] if hasattr(irreps, 'lmax') else list(irreps)


def to_ir_mul(x, irreps):
    """e3nn mul_ir (N, sum mul*(2l+1): off + u*(2l+1) + m) -> cuEq ir_mul (off + m*mul + u), as block transposes
    (split + transpose: the backward is a cat, unlike an index permutation whose backward is an index_put)."""
    n, blocks = x.shape[0], _blocks(irreps)
    if len(blocks) == 1 and blocks[0][1] == 1:
        return x   # scalars only: both layouts coincide
    parts = torch.split(x, [mul * d for mul, d in blocks], dim=1)
    return torch.cat([b.reshape(n, mul, d).transpose(1, 2).reshape(n, -1) for b, (mul, d) in zip(parts, blocks)], dim=1)


def to_mul_ir(x, irreps):
    n, blocks = x.shape[0], _blocks(irreps)
    if len(blocks) == 1 and blocks[0][1] == 1:
        return x
    parts = torch.split(x, [mul * d for mul, d in blocks], dim=1)
    return torch.cat([b.reshape(n, d, mul).transpose(1, 2).reshape(n, -1) for b, (mul, d) in zip(parts, blocks)], dim=1)
