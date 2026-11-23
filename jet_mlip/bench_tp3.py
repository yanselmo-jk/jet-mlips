"""Benchmark tp3 (layer 1, fp32) vs e3nn TensorProduct + index_add, cuEquivariance
uniform_1d (sevenn's channelwise path) and FlashTP on the 5568-node box5k graph:
nano (cutoff 5.5): E = 216704 edges; 7net-0 (cutoff 5.0): E = 174208.
Note: jet computes only the live paths (pruned W); the others take the full e3nn weight
(nano L1: 11 of 15 instructions, (E, 480)). Usage: python jet_mlip/bench_tp3.py [--model 7net-0] [--sorted-dst]"""
import argparse
import sys

import torch
from e3nn import o3

from jet_mlip import ops
from jet_mlip.configs import selected

torch.manual_seed(0)
dev = 'cuda'
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='7net-nano-5.5', choices=['7net-nano-5.5', '7net-0'])
ap.add_argument('--sorted-dst', action='store_true', help='dst-grouped edge list, as sevenn produces')
args = ap.parse_args()
ALL = selected('all')
if args.model == '7net-0':
    cfg = next(c for c in ALL if c.key.startswith('128x0e+64x1e+32x2e|1x0e+1x1e+1x2e|128x0e+64x0e'))
    N, E = 5568, 174208
else:
    cfg = ALL[1]   # 7net-nano layer 1
    N, E = 5568, 216704
group = 'O3' if any(ir.p == -1 for _, ir in cfg.irreps_sh) else 'SO3'
print(f'{args.model} L1: {cfg.n_paths} paths, nw {cfg.n_weight} (full {cfg.n_weight_full}), N={N} E={E}')
x = torch.randn(N, cfg.in_dim, device=dev)
sh = torch.randn(E, cfg.sh_dim, device=dev)
Wp = torch.randn(E, cfg.n_weight, device=dev)          # pruned (jet)
Wf = torch.randn(E, cfg.n_weight_full, device=dev)     # full e3nn layout
src = torch.randint(0, N, (E,), device=dev)
dst = torch.randint(0, N, (E,), device=dev)
if args.sorted_dst:
    dst = dst.sort().values
cid = ops.config_id(cfg)


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[n // 2]


def bench(name, fwd, x_, sh_, W_):
    ins = [t.clone().requires_grad_() for t in (x_, sh_, W_)]
    t_f = timeit(lambda: fwd(*(t.detach() for t in ins)))
    t_fb = timeit(lambda: torch.autograd.grad(fwd(*ins).sum(), ins))
    print(f'{name:<28s} fwd {t_f:8.3f} ms   fwd+bwd {t_fb:8.3f} ms')


bench('jet tp3', lambda x, sh, W: torch.ops.jet_mlip.tp3(x, sh, W, src.int(), dst.int(), N, cid), x, sh, Wp)

tp = o3.TensorProduct(cfg.irreps_x, cfg.irreps_sh, cfg.irreps_out, cfg.instructions,
                      shared_weights=False, internal_weights=False).to(dev)
bench('e3nn TP + index_add',
      lambda x, sh, W: x.new_zeros(N, cfg.out_dim).index_add(0, dst, tp(x[src], sh, W)), x, sh, Wf)

try:
    from sevenn.nn.cue_helper import as_cue_irreps, cueq_fused_scatter_channelwise_conv
    conv = cueq_fused_scatter_channelwise_conv(*(as_cue_irreps(i, group) for i in (cfg.irreps_x, cfg.irreps_sh, cfg.irreps_out)),
                                               tp_method='uniform_1d').to(dev)
    bench('cuEq uniform_1d (sevenn)', lambda x, sh, W: conv(x, sh, W, src, dst), x, sh, Wf)
except Exception as ex:  # noqa: BLE001
    print('cuEquivariance skipped:', type(ex).__name__, str(ex)[:120], file=sys.stderr)

try:
    from flashTP_e3nn import uvu_TP
    ftp = uvu_TP(cfg.irreps_x, cfg.irreps_sh, cfg.irreps_out, cfg.instructions, device=dev)
    bench('FlashTP uvu_TP', lambda x, sh, W: ftp(x, sh, W, src.int(), dst.int()), x, sh, Wf)
except Exception as ex:  # noqa: BLE001
    print('FlashTP skipped:', type(ex).__name__, str(ex)[:120], file=sys.stderr)
