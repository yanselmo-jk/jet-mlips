"""conv_fused CUDA op vs the torch reference (fp64 1e-10, fp32 1e-5 / grads 1e-4, gradcheck)."""
import math

import pytest
import torch

import jet_mlip
from jet_mlip import ops, reference, tables
from jet_mlip.configs import selected
from jet_mlip.shapes import compiled_keys
from test_reference import load_fp64

if not jet_mlip.is_available() or not ops.has_op('conv_fused'):
    pytest.skip('jet_mlip._C with conv_fused not built', allow_module_level=True)

DEV = 'cuda'
R_ON, R_CUT, CST = 4.0, 5.0, 1.6791767923989418


def _by_key(prefix):
    """The config whose key starts with `prefix`, or None when this build lacks it (JET_CONFIGS=<family>)."""
    return next((c for c in ALL if c.key.startswith(prefix)), None)


def _param(cfg, id):
    """A parametrize entry that skips when this build lacks the config (JET_CONFIGS=all builds every one)."""
    return pytest.param(cfg, id=id, marks=pytest.mark.skipif(cfg is None or cfg.key not in COMPILED,
                                                             reason='config not in this build'))


ALL, COMPILED = selected('all'), compiled_keys()
CFGS = [c for c in ALL if c.key in COMPILED]   # every family's shapes, or those of JET_CONFIGS=<family>
IDS = [f'{i}:{c.n_paths}p:{c.irreps_x}'.replace(' ', '') for i, c in enumerate(CFGS)]
NET0_L1 = _by_key('128x0e+64x1e+32x2e|1x0e+1x1e+1x2e|128x0e+64x0e')    # 7net-0 L1 (mul 128/64/32)


def random_case(cfg, n=60, e=None, degrees=(0, 80), dtype=torch.float64, seed=0, sorted_dst=False, n_out=None,
                extremes=True):
    """Random graph with per-node degrees spanning `degrees` (some nodes isolated, some dense)."""
    g = torch.Generator().manual_seed(seed)
    n_out = n if n_out is None else n_out   # dst < n_out (ghost nodes are sources only)
    deg = torch.randint(degrees[0], degrees[1] + 1, (n_out,), generator=g)
    deg[0] = 0
    dst = torch.repeat_interleave(torch.arange(n_out), deg)
    if not sorted_dst:
        dst = dst[torch.randperm(dst.numel(), generator=g)]
    if e is not None:
        dst = dst[:e]
    E = dst.numel()
    src = torch.randint(0, n, (E,), generator=g)
    vec = torch.randn(E, 3, generator=g, dtype=dtype) * 2.0
    if extremes:
        vec[: E // 10] *= 4  # beyond cutoff (finite output, unclamped XPLOR polynomial)
        vec[E // 10: E // 10 + 3] *= 1e-8  # tiny r (SH normalize eps path)
    x = torch.randn(n, cfg.in_dim, generator=g, dtype=dtype)
    w1 = torch.randn(8, 64, generator=g, dtype=dtype)
    w2 = torch.randn(64, 64, generator=g, dtype=dtype)
    w3 = torch.randn(64, cfg.n_weight, generator=g, dtype=dtype)
    coeffs = torch.arange(1, 9, dtype=dtype) * math.pi / R_CUT
    t = [v.to(DEV) for v in (x, vec, src, dst, w1, w2, w3, coeffs)]
    return t, 7.0, n_out


def run_op(cfg, t, denom, n_out, dtype):
    x, vec, src, dst, w1, w2, w3, coeffs = [v.to(dtype) if v.is_floating_point() else v for v in t]
    return ops.conv_fused(cfg, x, vec, src, dst, w1, w2, w3, coeffs, denom, R_ON, R_CUT, CST, n_out=n_out)


def run_ref(cfg, t, denom, n_out, dtype=torch.float64):
    x, vec, src, dst, w1, w2, w3, coeffs = [v.to(dtype) if v.is_floating_point() else v for v in t]
    return reference.conv(cfg, x, vec, src.long(), dst.long(), w1, w2, w3, coeffs, R_ON, R_CUT, CST, denom)[:n_out]


@pytest.mark.parametrize('cfg', CFGS, ids=IDS)
@pytest.mark.parametrize('sorted_dst', [False, True])
def test_forward_fp64(cfg, sorted_dst):
    t, denom, n_out = random_case(cfg, sorted_dst=sorted_dst, n_out=50)
    out = run_op(cfg, t, denom, n_out, torch.float64)
    ref = run_ref(cfg, t, denom, n_out)
    assert out.shape == ref.shape and torch.isfinite(out).all()
    atol = 1e-10 if cfg.lmax < 4 else 1e-9   # lmax 4: outputs ~1e3, the l = 4 sum order leaves ~4e-10 of fp64 round-off
    assert torch.allclose(out, ref, atol=atol, rtol=1e-10), (out - ref).abs().max()
    assert (out[:, ~cfg.out_mask()] == 0).all()


@pytest.mark.parametrize('cfg', CFGS, ids=IDS)
def test_forward_fp32(cfg):
    t, denom, n_out = random_case(cfg, seed=1)
    out = run_op(cfg, t, denom, n_out, torch.float32).double()
    ref = run_ref(cfg, t, denom, n_out)
    assert (out - ref).norm() / ref.norm() < 1e-5


def test_edge_cases():
    cfg = CFGS[1]
    # E = 0
    t, denom, n_out = random_case(cfg, n=5, e=0)
    out = run_op(cfg, t, denom, n_out, torch.float32)
    assert out.shape == (5, cfg.out_dim) and (out == 0).all()
    # N_out = 0
    out = run_op(cfg, t, denom, 0, torch.float32)
    assert out.shape == (0, cfg.out_dim)


LMAX3 = [c for c in ALL if c.lmax == 3 and c.n_paths > 5][:1]   # 7net-omni middle layer, l = 3 sh + its Jacobian
LMAX4 = [c for c in ALL if c.lmax == 4 and c.n_paths > 5][:1]   # EquFlashV2 middle layer, l = 4 sh + its Jacobian


@pytest.mark.parametrize('cfg', [_param(c, i) for c, i in zip(
    [ALL[1], ALL[3]] + LMAX3 + LMAX4,
    ['nano-L1', 'net0-L0', 'lmax3-L1', 'lmax4-L1'])])
@pytest.mark.parametrize('split', [True, False], ids=['split', 'recompute'])
def test_gradcheck_fp64(cfg, split):
    # no extreme edges: the finite-difference probe (eps 1e-6) flips tiny-r directions, and the
    # unclamped XPLOR polynomial far beyond r_cut makes ~1e5 contributions whose roundoff exceeds atol
    t, denom, n_out = random_case(cfg, n=6, e=25, degrees=(3, 6), seed=2, extremes=False)
    x, vec, src, dst, w1, w2, w3, coeffs = t
    x = x.clone().requires_grad_(True)
    vec = vec.clone().requires_grad_(True)
    csr = ops.build_csr(dst, n_out)

    def f(x, vec):
        return ops.conv_fused(cfg, x, vec, src, dst, w1, w2, w3, coeffs, denom, R_ON, R_CUT, CST, n_out=n_out, csr=csr,
                              split_bwd=split)

    assert torch.autograd.gradcheck(f, (x, vec))


@pytest.mark.parametrize('cfg', CFGS, ids=IDS)
@pytest.mark.parametrize('split', [True, False], ids=['split', 'recompute'])
def test_backward_fp32(cfg, split):
    t, denom, n_out = random_case(cfg, n=40, seed=3, n_out=35)
    grads = {}
    for dtype in (torch.float64, torch.float32):
        x, vec, src, dst, w1, w2, w3, coeffs = [v.to(dtype) if v.is_floating_point() else v for v in t]
        x.requires_grad_(True)
        vec.requires_grad_(True)
        if dtype == torch.float64:
            out = reference.conv(cfg, x, vec, src.long(), dst.long(), w1, w2, w3, coeffs, R_ON, R_CUT, CST, denom)[:n_out]
        else:
            out = ops.conv_fused(cfg, x, vec, src, dst, w1, w2, w3, coeffs, denom, R_ON, R_CUT, CST, n_out=n_out,
                                 split_bwd=split)
        gout = torch.randn(out.shape, generator=torch.Generator().manual_seed(4), dtype=torch.float64).to(DEV, dtype)
        gx, gv = torch.autograd.grad(out, (x, vec), gout)
        grads[dtype] = (gx.double(), gv.double())
    (gx32, gv32), (gx64, gv64) = grads[torch.float32], grads[torch.float64]
    assert (gx32 - gx64).norm() / gx64.norm() < 1e-4, (gx32 - gx64).norm() / gx64.norm()
    # edge_vec grads: compared on the physical edges (1e-6 < r <= r_cut), the extreme ones only checked finite.
    # Beyond r_cut the unclamped XPLOR polynomial (~1e5) makes an edge's grad a cancelling sum of huge terms and at
    # r ~ 1e-8 dB/dr cancels catastrophically: fp32 cannot resolve either (visible on scalar-only-output configs,
    # where no angular term dominates the norm). A real neighbor list has neither kind of edge.
    r = t[1].norm(dim=1)
    phys = (r > 1e-6) & (r <= R_CUT)
    assert torch.isfinite(gv32).all()
    assert (gv32[phys] - gv64[phys]).norm() / gv64[phys].norm() < 1e-4, (gv32[phys] - gv64[phys]).norm() / gv64[phys].norm()


@pytest.mark.parametrize('cfg', [_param(CFGS[1], 'nano-L1'), _param(NET0_L1, 'net0-L1')])
def test_split_vs_recompute_bwd(cfg):
    """fp32: the two-kernel backward (grad_H2 summed per edge, MLP backward once) and the single recompute
    kernel differ only by fp32 summation order."""
    t, denom, n_out = random_case(cfg, n=40, seed=10, n_out=35)
    x, vec, src, dst, w1, w2, w3, coeffs = [v.float() if v.is_floating_point() else v for v in t]
    x.requires_grad_(True)
    vec.requires_grad_(True)
    csr = ops.build_csr(dst, n_out)
    gout = torch.randn(n_out, cfg.out_dim, generator=torch.Generator().manual_seed(11)).to(DEV)
    g = {}
    for split in (True, False):
        out = ops.conv_fused(cfg, x, vec, src, dst, w1, w2, w3, coeffs, denom, R_ON, R_CUT, CST, n_out=n_out, csr=csr,
                             split_bwd=split)
        g[split] = torch.autograd.grad(out, (x, vec), gout)
    for a, b in zip(g[True], g[False]):   # grad_x is an atomic sum (not bitwise); grad_vec sums grad_H2 per edge first
        assert (a - b).norm() / b.norm() < 1e-5


@pytest.mark.parametrize('split', [True, False], ids=['split', 'recompute'])
def test_capacity_tail_rows(split):
    """LAMMPS GPU edges: vec has capacity rows past rowptr[n_out] (zeros, identity perm). They are not edges:
    same grads as without them, and grad_vec stays finite there (r = 0 gave NaN in the split MLP kernel)."""
    cfg = NET0_L1
    if cfg is None or cfg.key not in COMPILED:
        pytest.skip('config not in this build')
    t, denom, n_out = random_case(cfg, n=40, seed=12, n_out=35)
    x, vec, src, dst, w1, w2, w3, coeffs = [v.float() if v.is_floating_point() else v for v in t]
    rowptr, perm = ops.build_csr(dst, n_out)
    E, k = vec.shape[0], 70
    gout = torch.randn(n_out, cfg.out_dim, generator=torch.Generator().manual_seed(13)).to(DEV)
    g = []
    for tail in (0, k):
        xx = x.detach().clone().requires_grad_(True)
        vv = torch.cat([vec, vec.new_zeros(tail, 3)]).requires_grad_(True)
        ss, dd = torch.cat([src, src.new_zeros(tail)]), torch.cat([dst, dst.new_zeros(tail)])
        pp = torch.cat([perm, torch.arange(E, E + tail, dtype=perm.dtype, device=perm.device)])
        out = ops.conv_fused(cfg, xx, vv, ss, dd, w1, w2, w3, coeffs, denom, R_ON, R_CUT, CST, n_out=n_out,
                             csr=(rowptr, pp), split_bwd=split)
        g.append(torch.autograd.grad(out, (xx, vv), gout))
    assert torch.isfinite(g[1][1]).all()
    assert (g[1][0] - g[0][0]).norm() / g[0][0].norm() < 1e-5
    assert (g[1][1][:E] - g[0][1]).norm() / g[0][1].norm() < 1e-5


@pytest.mark.parametrize('cfg', [_param(CFGS[1], 'nano-L1'), _param(NET0_L1, 'net0-L1')])
@pytest.mark.parametrize('table', [False, True])
def test_denominator_in_kernel(cfg, table):
    """fp32: denominator divided inside the kernel == op(denom=1) then torch out.div(denom): the forward bitwise,
    grad_vec to rounding (grad_x is an atomic sum, not compared)."""
    t, _, n_out = random_case(cfg, n=40, seed=6, n_out=35)
    x, vec, src, dst, w1, w2, w3, coeffs = [v.float() if v.is_floating_point() else v for v in t]
    vec.requires_grad_(True)
    d = torch.tensor([44.918491363525390625], device=DEV)   # 7net-nano's, as an fp32 CUDA parameter
    csr = ops.build_csr(dst, n_out)
    tab = ops.make_radial_table(cfg, w1, w2, w3, coeffs, R_ON, R_CUT, CST) if table else None

    def f(denom):
        return ops.conv_fused(cfg, x, vec, src, dst, w1, w2, w3, coeffs, denom, R_ON, R_CUT, CST, n_out=n_out,
                              csr=csr, table=tab)

    a, b = f(1.0).div(d), f(float(d))
    assert torch.equal(a, b)
    gout = torch.randn(a.shape, generator=torch.Generator().manual_seed(8)).to(DEV)
    # the path groups add into an edge's row in any order (atomics; with one block per SM they overlap in time)
    ga, gb = (torch.autograd.grad(o, vec, gout)[0] for o in (a, b))
    assert (ga - gb).abs().max() <= 1e-6 * ga.abs().max(), ((ga - gb).abs().max(), ga.abs().max())


def test_backward_without_grad_x():
    """x not requiring grad (layer 0 in LAMMPS): grad_x is neither allocated nor accumulated, grad_vec unchanged."""
    cfg = CFGS[1]
    t, denom, n_out = random_case(cfg, n=40, seed=7, n_out=35)
    x, vec, src, dst, w1, w2, w3, coeffs = [v.float() if v.is_floating_point() else v for v in t]
    vec.requires_grad_(True)
    rowptr, perm = ops.build_csr(dst, n_out)

    def f(xx):
        return ops.conv_fused(cfg, xx, vec, src, dst, w1, w2, w3, coeffs, denom, R_ON, R_CUT, CST, n_out=n_out,
                              csr=(rowptr, perm))

    gout = torch.randn(n_out, cfg.out_dim, generator=torch.Generator().manual_seed(9)).to(DEV)
    g_ref = torch.autograd.grad(f(x.clone().requires_grad_(True)), vec, gout)[0]
    # not bit-equal: a shape with several backward path groups adds into grad_vec with atomics, whose order (the last
    # bits) changes from run to run, also between two identical calls
    tol = dict(atol=1e-6 * g_ref.abs().max().item(), rtol=1e-5)
    assert torch.allclose(torch.autograd.grad(f(x), vec, gout)[0], g_ref, **tol)
    gx, gv = torch.ops.jet_mlip.conv_fused_bwd(gout, x, vec, src.int(), rowptr, perm, w1, w2, w3, coeffs, denom,
                                              R_ON, R_CUT, CST, ops.config_id(cfg), None, False)
    assert gx is None and torch.allclose(gv, g_ref, **tol)


# ---- 3-D (per element pair) tables: MACE MPA-0 / OMAT-0, NequIP per-edge-type tables ---------------
def _table_case(cfg, seed, k=3, n=40, n_out=35, **kw):
    """fp32 case with a (k, n, 4*NW) table of k distinct radial tables and a random per-edge table index."""
    t, denom, n_out = random_case(cfg, n=n, seed=seed, n_out=n_out, **kw)
    x, vec, src, dst, w1, w2, w3, coeffs = [v.float() if v.is_floating_point() else v for v in t]
    g = torch.Generator().manual_seed(seed)
    tabs = torch.stack([ops.make_radial_table(cfg, w1, w2, torch.randn(w3.shape, generator=g).to(DEV), coeffs,
                                              R_ON, R_CUT, CST, n=256) for _ in range(k)])
    ti = torch.randint(0, k, (vec.shape[0],), generator=g).to(DEV).int()
    return (x, vec, src, dst, w1, w2, w3, coeffs), tabs, ti, denom, n_out


@pytest.mark.parametrize('cfg', CFGS, ids=IDS)
def test_table_idx_3d(cfg):
    """3-D table + per-edge table_idx: output and grads == the 2-D table path run once per table on that
    table's masked edge set (a conv is a sum over edges, so the k runs add up)."""
    (x, vec, src, dst, w1, w2, w3, coeffs), tabs, ti, denom, n_out = _table_case(cfg, 14)
    x.requires_grad_(True)
    vec.requires_grad_(True)
    gout = torch.randn(n_out, cfg.out_dim, generator=torch.Generator().manual_seed(15)).to(DEV)
    out = ops.conv_fused(cfg, x, vec, src, dst, w1, w2, w3, coeffs, denom, R_ON, R_CUT, CST, n_out=n_out,
                         table=tabs, table_idx=ti)
    gx, gv = torch.autograd.grad(out, (x, vec), gout)
    xs, vs, outs = torch.zeros_like(x), torch.zeros_like(vec), torch.zeros_like(out)
    for k in range(tabs.shape[0]):
        keep = ti == k
        xk, vk = x.detach().clone().requires_grad_(True), vec[keep].detach().clone().requires_grad_(True)
        ok = ops.conv_fused(cfg, xk, vk, src[keep], dst[keep], w1, w2, w3, coeffs, denom, R_ON, R_CUT, CST,
                            n_out=n_out, table=tabs[k])
        gxk, gvk = torch.autograd.grad(ok, (xk, vk), gout)
        outs += ok.detach(); xs += gxk; vs[keep] = gvk
    assert (out - outs).norm() / outs.norm() < 1e-5
    assert (gx - xs).norm() / xs.norm() < 1e-5
    assert (gv - vs).norm() / vs.norm() < 1e-5


@pytest.mark.parametrize('cfg', CFGS, ids=IDS)
def test_table_idx_3d_identical(cfg):
    """3-D table whose tables are all identical == the 2-D table: same values, only the row offset differs."""
    (x, vec, src, dst, w1, w2, w3, coeffs), tabs, ti, denom, n_out = _table_case(cfg, 16)
    tabs = tabs[0].expand(3, -1, -1).contiguous()
    gout = torch.randn(n_out, cfg.out_dim, generator=torch.Generator().manual_seed(17)).to(DEV)
    res = []
    for tab, idx in ((tabs[0], None), (tabs, ti)):
        xx, vv = x.detach().clone().requires_grad_(True), vec.detach().clone().requires_grad_(True)
        out = ops.conv_fused(cfg, xx, vv, src, dst, w1, w2, w3, coeffs, denom, R_ON, R_CUT, CST, n_out=n_out,
                             table=tab, table_idx=idx)
        res.append((out,) + torch.autograd.grad(out, (xx, vv), gout))
    for a, b in zip(*res):
        # the path groups add into a row in any order (atomics; with one block per SM they overlap in time): the
        # rounding of that sum, relative to the largest term
        assert torch.allclose(a, b, atol=1e-6 * b.abs().max().item(), rtol=1e-5), (a - b).abs().max()


def test_table_rank_mismatch():
    """A 2-D table with table_idx, and a 3-D table without it, are rejected by the kernel's checks."""
    cfg = CFGS[0]
    (x, vec, src, dst, w1, w2, w3, coeffs), tabs, ti, denom, n_out = _table_case(cfg, 18)
    with pytest.raises(RuntimeError, match='table_idx must be int32'):
        ops.conv_fused(cfg, x, vec, src, dst, w1, w2, w3, coeffs, denom, R_ON, R_CUT, CST, n_out=n_out,
                       table=tabs[0], table_idx=ti)
    with pytest.raises(RuntimeError, match='3-D table needs table_idx'):
        ops.conv_fused(cfg, x, vec, src, dst, w1, w2, w3, coeffs, denom, R_ON, R_CUT, CST, n_out=n_out, table=tabs)


SCALAR_IN = [c for c in CFGS if len(c.irreps_x) == 1 and c.irreps_x[0].ir.l == 0 and c.irreps_x[0].mul <= 32
             and all(c.live_out)]   # the shapes conv_first.cu takes


@pytest.mark.parametrize('cfg', SCALAR_IN, ids=[str(c.irreps_x) + '|lmax' + str(c.lmax) for c in SCALAR_IN])
@pytest.mark.parametrize('ntab', [1, 3], ids=['2d', '3d'])
def test_first_layer_kernel(cfg, ntab, monkeypatch):
    """Scalar-only input of <= 32 channels on the radial table: the first-layer kernel (csrc/conv_first.cu) == the generic
    kernel (JET_FIRST_LAYER=0) in out, grad x and grad edge_vec; ghost sources (n_out < N), isolated nodes, unsorted dst,
    edges beyond the cutoff, with and without grad x. 4000 dst nodes: the kernel takes more than 1.5 x 8 warps x SMs
    (H200 1584), smaller graphs stay on the generic kernel."""
    (x, vec, src, dst, w1, w2, w3, coeffs), tabs, ti, denom, n_out = _table_case(cfg, 19, n=4100, n_out=4000,
                                                                                 degrees=(0, 12))
    tab, idx = (tabs, ti) if ntab == 3 else (tabs[0], None)
    gout = torch.randn(n_out, cfg.out_dim, generator=torch.Generator().manual_seed(20)).to(DEV)
    res = []
    for first in ('0', '1'):
        monkeypatch.setenv('JET_FIRST_LAYER', first)
        xx, vv = x.detach().clone().requires_grad_(True), vec.detach().clone().requires_grad_(True)
        out = ops.conv_fused(cfg, xx, vv, src, dst, w1, w2, w3, coeffs, denom, R_ON, R_CUT, CST, n_out=n_out,
                             table=tab, table_idx=idx)
        gv_only = torch.autograd.grad(out, vv, gout, retain_graph=True)   # the first layer of a model: no grad x
        res.append((out,) + torch.autograd.grad(out, (xx, vv), gout) + gv_only)
    assert not torch.equal(res[0][0], res[1][0]), 'bit-identical outputs: the first-layer kernel did not run'
    for a, b in zip(*res):
        assert torch.allclose(a, b, atol=1e-6 * a.abs().max().item(), rtol=1e-5), (a - b).abs().max()


KB2 = [c for c in CFGS if max(torch.ops.jet_mlip.config_groups(ops.config_id(c))[5:]) > 1]   # tab_kb: 2 blocks per SM


@pytest.mark.parametrize('cfg', KB2, ids=[IDS[CFGS.index(c)] for c in KB2])
def test_table_two_blocks_per_sm(cfg, monkeypatch):
    """Radial-table kernels built for 2 blocks per SM (tab_kb; taken above 1.5 x 8 warps x SMs dst nodes, H200 1584) ==
    the 1-block kernels (JET_TAB_KB=1) in out, grad x and grad edge_vec. JET_FIRST_LAYER=0: the generic kernel."""
    (x, vec, src, dst, w1, w2, w3, coeffs), tabs, _, denom, n_out = _table_case(cfg, 21, n=4100, n_out=4000,
                                                                                degrees=(0, 12))
    gout = torch.randn(n_out, cfg.out_dim, generator=torch.Generator().manual_seed(22)).to(DEV)
    monkeypatch.setenv('JET_FIRST_LAYER', '0')
    res = []
    for kb in ('1', '2'):
        monkeypatch.setenv('JET_TAB_KB', kb)
        xx, vv = x.detach().clone().requires_grad_(True), vec.detach().clone().requires_grad_(True)
        out = ops.conv_fused(cfg, xx, vv, src, dst, w1, w2, w3, coeffs, denom, R_ON, R_CUT, CST, n_out=n_out,
                             table=tabs[0])
        res.append((out,) + torch.autograd.grad(out, (xx, vv), gout))
    for a, b in zip(*res):
        assert torch.allclose(a, b, atol=1e-6 * a.abs().max().item(), rtol=1e-5), (a - b).abs().max()


# ---- full 7net-nano / 7net-0 models: per layer and whole-model forces -------------------
@pytest.fixture(scope='module')
def nano():
    m, _ = load_fp64('7net-nano-5.5')
    if any(_layer_args(m, i)[0].key not in COMPILED for i in range(3)):
        pytest.skip('nano: config not in this build (JET_CONFIGS=all)')
    m.set_is_batch_data(False)   # single structure, as the ASE calculator does
    return m.to(DEV)


@pytest.fixture(scope='module')
def net0():
    m, _ = load_fp64('7net-0')
    if any(_layer_args(m, i)[0].key not in COMPILED for i in range(5)):
        pytest.skip('net0: config not in this build (JET_CONFIGS=all)')
    m.set_is_batch_data(False)
    return m.to(DEV)


def _layer_args(m, layer):
    conv = getattr(m, f'{layer}_convolution')
    lin = getattr(m, f'{layer}_self_interaction_2')
    emb = m.edge_embedding
    cfg = tables.config_from_sevenn_conv(conv, lin.linear.irreps_out)
    w = conv.weight_nn
    return cfg, conv, emb, (w.layer0.weight, w.layer1.weight, reference.prune_w3(cfg, w.layer2.weight),
                           emb.basis_function.coeffs, float(conv.denominator), float(emb.cutoff_function.r_on),
                           float(emb.cutoff_function.r_cut), float(w.layer0.act.cst))


@pytest.mark.parametrize('model, layer', [('nano', 0), ('nano', 1), ('nano', 2)] + [('net0', i) for i in range(5)])
def test_model_layer(request, model, layer):
    KEY = pytest.importorskip('sevenn._keys')
    m = request.getfixturevalue(model)
    cfg, conv, emb, (w1, w2, w3, coeffs, denom, r_on, r_cut, cst) = _layer_args(m, layer)
    assert ops.config_id(cfg) >= 0
    g = torch.Generator().manual_seed(5)
    n, E = 30, 900
    x = torch.randn(n, cfg.in_dim, generator=g, dtype=torch.float64).to(DEV)
    vec = (torch.randn(E, 3, generator=g, dtype=torch.float64) * 2.5).to(DEV)
    src, dst = torch.randint(0, n, (2, E), generator=g).to(DEV)
    data = {KEY.EDGE_VEC: vec, KEY.NODE_FEATURE: x, KEY.EDGE_IDX: torch.stack([dst, src])}
    ref = conv(emb(data))[KEY.NODE_FEATURE]
    out = ops.conv_fused(cfg, x, vec, src, dst, w1, w2, w3, coeffs, denom, r_on, r_cut, cst)
    mask = cfg.out_mask().to(DEV)
    assert torch.allclose(out[:, mask], ref[:, mask], atol=1e-10, rtol=1e-10)
    assert (out[:, ~mask] == 0).all()
    out32 = ops.conv_fused(cfg, x.float(), vec.float(), src, dst, w1.float(), w2.float(), w3.float(),
                           coeffs.float(), denom, r_on, r_cut, cst).double()
    assert (out32[:, mask] - ref[:, mask]).norm() / ref[:, mask].norm() < 1e-5


@pytest.mark.parametrize('model', ['nano', 'net0'])
def test_model_forces(request, model):
    """Whole model: energy/forces with every conv replaced by conv_fused vs the original."""
    KEY = pytest.importorskip('sevenn._keys')
    from ase.build import bulk
    from sevenn.train.dataload import unlabeled_atoms_to_graph
    from sevenn.atom_graph_data import AtomGraphData

    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True).repeat((2, 2, 2))   # 32 atoms
    atoms.rattle(0.1, seed=1)
    m = request.getfixturevalue(model)
    r_cut = float(m.edge_embedding.cutoff_function.r_cut)

    def run(patched):
        torch.set_default_dtype(torch.float64)   # graph tensors and the onehot are built in the default dtype
        def to_double(d):   # the graph builder and the onehot module hardcode float32
            for k in list(d.keys()):
                if torch.is_tensor(d[k]) and d[k].dtype == torch.float32:
                    d[k] = d[k].double()
            return d

        data = to_double(AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms, r_cut)).to(DEV))
        data = m._preprocess(data)   # edge_vec.requires_grad (forces = scatter of dE/d edge_vec)
        for name, mod in m._modules.items():   # the model's own sequence, conv swapped
            if patched and name.endswith('_convolution'):
                layer = int(name.split('_')[0])
                cfg, conv, emb, (w1, w2, w3, coeffs, denom, r_on, rc, cst) = _layer_args(m, layer)
                dst, src = data[KEY.EDGE_IDX]
                data[KEY.NODE_FEATURE] = ops.conv_fused(cfg, data[KEY.NODE_FEATURE], data[KEY.EDGE_VEC], src, dst,
                                                        w1, w2, w3, coeffs, denom, r_on, rc, cst)
            else:
                data = to_double(mod(data))
        torch.set_default_dtype(torch.float32)
        return data[KEY.PRED_TOTAL_ENERGY].detach(), data[KEY.PRED_FORCE].detach()

    e0, f0 = run(False)
    e1, f1 = run(True)
    assert torch.allclose(e0, e1, atol=1e-9, rtol=1e-9)
    assert torch.allclose(f0, f1, atol=1e-9, rtol=1e-9), (f0 - f1).abs().max()
