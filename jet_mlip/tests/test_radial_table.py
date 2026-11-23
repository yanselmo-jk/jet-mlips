"""Radial table mode: the deploy-time replacement of the per-edge radial MLP by a
1-D table of W(r) + dW/dr with cubic Hermite interpolation.

Three levels: the interpolation itself (fp64, h^4), one conv_fused layer (fp32 vs the
exact kernel) and whole models (forces vs the e3nn path on the 5568-atom box).
The model sweep also prints the numbers behind docs/bench_rtx4070.md "Radial table";
it needs the jet-aware sevenn (PYTHONPATH=<SevenNet>:<worktree>).
"""
import math
import os
import statistics
import time

import pytest
import torch

import jet_mlip
from jet_mlip import ops, reference
from test_conv_fused import CFGS, CST, R_CUT, R_ON, random_case

if not jet_mlip.is_available() or not ops.has_op('conv_fused'):
    pytest.skip('jet_mlip._C with conv_fused not built', allow_module_level=True)

DEV = 'cuda'
NANO_L1 = CFGS[1]
BOX = os.environ.get('JET_BOX5K', '')   # box5k: 5568-atom electrolyte, LAMMPS data file (atomic style)
T32C = '/home/yanselmo/0_projects-material/3_mlip_distill/work/smoke/distill/T32c/train_mm/checkpoint_best.pth'
MODELS = {'nano': ('7net-nano-5.5', None), '7net-0': ('7net-0', None), 'T32c': (T32C, 'dft')}
NS = [1024, 2048, 4096, 8192]
Z = {1: 3, 2: 6, 3: 1, 4: 8, 5: 7, 6: 9, 7: 16, 8: 15, 9: 5}   # box5k type -> Z


def hermite(table, r, r_cut):
    """The interpolation the kernel does, in torch: W(r) and dW/dr(r) from the table."""
    n = table.shape[0]
    h = r_cut / (n - 1)
    k = (r / h).floor().long().clamp(0, n - 2)
    t = (r / h - k).clamp(0, 1).unsqueeze(-1)
    t2, t3 = t * t, t * t * t
    v0, m0, dn, m1 = table.view(n, -1, 4)[k].unbind(-1)
    w = v0 + h * ((t3 - 2 * t2 + t) * m0 + (3 * t2 - 2 * t3) * dn + (t3 - t2) * m1)
    d = (3 * t2 - 4 * t + 1) * m0 + (6 * t - 6 * t2) * dn + (3 * t2 - 2 * t) * m1
    return w, d


def _mlp_weights(nw, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(8, 64, generator=g, dtype=dtype), torch.randn(64, 64, generator=g, dtype=dtype),
            torch.randn(64, nw, generator=g, dtype=dtype),
            torch.arange(1, 9, dtype=dtype) * math.pi / R_CUT)


def test_hermite_converges():
    """fp64 table vs the exact MLP: h^4 where W is smooth. The one interval straddling
    r_on is only h^2 (XPLOR is C1 there, so W'' jumps) and is checked separately."""
    w1, w2, w3, coeffs = _mlp_weights(64)
    r = torch.linspace(0.3, R_CUT, 20001, dtype=torch.float64)
    exact = reference.radial_mlp(reference.bessel_cutoff(r, coeffs, R_ON, R_CUT), w1, w2, w3, CST)
    ns = (256, 512, 1024, 2048)
    smooth, kink = [], []
    for n in ns:
        table = ops.make_radial_table(None, w1, w2, w3, coeffs, R_ON, R_CUT, CST, n)
        assert table.shape == (ops.table_points(n, R_ON, R_CUT), 256) and table.dtype == torch.float64
        w, _ = hermite(table, r, R_CUT)
        e = (w - exact).abs().max(1).values
        smooth.append(e[(r - R_ON).abs() > 3 * R_CUT / (table.shape[0] - 1)].max().item())
        kink.append(e.max().item())
    print('\nhermite max|dW|: smooth', [f'{n}:{e:.1e}' for n, e in zip(ns, smooth)],
          ' incl. r_on', [f'{n}:{e:.1e}' for n, e in zip(ns, kink)])
    for a, b in zip(smooth, smooth[1:]):
        assert a / b > 8, smooth    # h^4 would be 16; 8 leaves room for the fp64 floor
    assert max(kink) < 1e-6


def test_table_points():
    """Grid sizes put a node at r_on (7net-nano 5.0/5.5, 7net-0 4.5/5.0, 7net-omni 5.5/6.0, tests 4.0/5.0)."""
    assert [ops.table_points(2048, a, b) for a, b in ((5.0, 5.5), (4.5, 5.0), (5.5, 6.0), (4.0, 5.0))] == [2047, 2041, 2041, 2046]
    assert ops.table_points(2048, 5.0, 2 ** 0.5 * 4) == 2048   # irrational ratio: n unchanged


def test_hermite_derivative():
    """dW/dr from the same Hermite polynomial matches autograd of the interpolant."""
    w1, w2, w3, coeffs = _mlp_weights(64, seed=2)
    table = ops.make_radial_table(None, w1, w2, w3, coeffs, R_ON, R_CUT, CST, 2048)
    r = torch.linspace(0.4, R_CUT - 0.01, 97, dtype=torch.float64)
    _, d = hermite(table, r, R_CUT)
    fd = (hermite(table, r + 1e-6, R_CUT)[0] - hermite(table, r - 1e-6, R_CUT)[0]) / 2e-6
    assert (d - fd).abs().max() < 1e-5 * d.abs().max()
    # beyond the cutoff both vanish (XPLOR is 0 at r_cut); W only to rounding, since the
    # last interval reconstructs it as W(r_k) + h * (W(r_k+1) - W(r_k)) / h
    w_end, d_end = hermite(table, torch.tensor([R_CUT, 2 * R_CUT], dtype=torch.float64), R_CUT)
    assert w_end.abs().max() < 1e-12 * table[:, ::4].abs().max() and d_end.abs().max() == 0


@pytest.mark.parametrize('n', NS)
def test_conv_fused_table_vs_exact(n):
    """nano L1 fp32: table mode vs the exact kernel, forward and both gradients."""
    cfg = NANO_L1
    t, denom, n_out = random_case(cfg, n=40, seed=3, n_out=35, extremes=False)
    x, vec, src, dst, w1, w2, w3, coeffs = t
    vec = vec / vec.norm(dim=1, keepdim=True) * (torch.rand(vec.shape[0], 1, device=DEV, dtype=vec.dtype)
                                                 * (R_CUT - 0.5) + 0.5)
    t = (x, vec, src, dst, w1, w2, w3, coeffs)
    table = ops.make_radial_table(cfg, *[v.float() for v in (w1, w2, w3, coeffs)], R_ON, R_CUT, CST, n)
    assert table.shape == (ops.table_points(n, R_ON, R_CUT), 4 * cfg.n_weight)

    outs = {}
    for tab in (None, table):
        xf, vf = x.float().requires_grad_(True), vec.float().requires_grad_(True)
        o = ops.conv_fused(cfg, xf, vf, src, dst, *[v.float() for v in (w1, w2, w3, coeffs)],
                           denom, R_ON, R_CUT, CST, n_out=n_out, table=tab)
        gout = torch.randn(o.shape, generator=torch.Generator().manual_seed(4)).to(DEV)
        gx, gv = torch.autograd.grad(o, (xf, vf), gout)
        outs[tab is None] = (o.detach().double(), gx.double(), gv.double())
    (o0, gx0, gv0), (o1, gx1, gv1) = outs[True], outs[False]
    rel = [(a - b).norm().item() / b.norm().item() for a, b in ((o1, o0), (gx1, gx0), (gv1, gv0))]
    print(f'\nN={n:5d} nano L1 rel diff vs exact fp32: out {rel[0]:.2e} grad_x {rel[1]:.2e} grad_vec {rel[2]:.2e}')
    assert rel[0] < 1e-5 and rel[1] < 1e-4 and rel[2] < 1e-4


# ---- whole models on the 5568-atom box ---------------------------------------------------
@pytest.fixture(scope='module')
def box():
    from ase.io import read
    if not os.path.exists(BOX):
        pytest.skip('set JET_BOX5K to the box5k LAMMPS data file')
    return read(BOX, format='lammps-data', atom_style='atomic', Z_of_type=Z)


def _energy_force_ms(model, atoms, radial_table, modal=None, reps=5):
    """bench_model.py protocol: graph built once on GPU, median ms of forward + force."""
    import sevenn._keys as KEY
    from sevenn.atom_graph_data import AtomGraphData
    from sevenn.calculator import SevenNetCalculator
    from sevenn.train.dataload import unlabeled_atoms_to_graph

    kw = {} if radial_table is None else {'enable_jet': True, 'radial_table': radial_table}
    if modal:
        kw['modal'] = modal
    calc = SevenNetCalculator(model, **kw)
    m = calc.model
    data = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms, calc.cutoff, with_shift=False)).to(DEV)
    if modal:
        data[KEY.DATA_MODALITY] = modal

    def run():
        out = m(data.clone())
        return out[KEY.PRED_TOTAL_ENERGY], out[KEY.PRED_FORCE]

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    e, f = run()
    out = e.detach().cpu(), f.detach().cpu(), statistics.median(ts)
    del calc, m, data, e, f   # 5568 atoms x e3nn 7net-0 already needs more than the 12 GB card
    torch.cuda.empty_cache()
    return out


@pytest.mark.parametrize('name', list(MODELS))
def test_model_forces_table(name, box):
    """The table path must not move the forces more than fp32 reordering already does:
    max|F_table - F_exact| <= max|F_exact - F_e3nn| (both fp32; docs/bench_rtx4070.md
    "Radial table" has the full N sweep). Set JET_TABLE_NS to re-measure that sweep."""
    model, modal = MODELS[name]
    if not (model.startswith('7net') or os.path.exists(model)):
        pytest.skip(f'{model} missing')
    try:
        _, f_e3nn, ms0 = _energy_force_ms(model, box, None, modal)
    except RuntimeError:   # e3nn 7net-0 does not fit in 12 GB on the 5568-atom box
        torch.cuda.empty_cache()
        pytest.skip('e3nn reference out of memory')
    try:
        e0, f_exact, ms_exact = _energy_force_ms(model, box, 0, modal)
    except KeyError as e:   # 'config not compiled': a student shape outside the default build (JET_CONFIGS=all)
        pytest.skip(str(e))
    floor = (f_exact - f_e3nn).abs().max().item()
    print(f'\n{name}: e3nn {ms0:.1f} ms, jet exact {ms_exact:.1f} ms, fp32 floor {floor:.2e} eV/A')
    for n in [int(v) for v in os.environ.get('JET_TABLE_NS', str(ops.DEFAULT_RADIAL_TABLE)).split(',')]:
        e, f, ms = _energy_force_ms(model, box, n, modal)
        de = abs(e.item() - e0.item()) / len(box)
        df = (f - f_exact).abs().max().item()
        print(f'{name:7s} N={n:5d} max|dF| vs exact={df:.2e} eV/A (vs e3nn {(f - f_e3nn).abs().max().item():.2e})'
              f'  |dE|/atom={de:.2e} eV  {ms:.1f} ms')
        # fp32 atomics make both sides run-to-run noisy (the exact-vs-e3nn floor itself moves ~2x between
        # runs), so allow 2x: the table must not be clearly worse than the exact path, not luckier than it.
        assert df <= max(2 * floor, 1e-8), (name, n, df, floor)
