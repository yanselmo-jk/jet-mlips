"""edge_build CUDA op vs pair_e3gnn's host loop (numpy float64, no FMA): bitwise rowptr / nbr / ctr / vec / total
on random periodic boxes with ghosts (owner -1 rows, empty rows, an exact r = cutoff pair, self-image ghosts),
cap < total, nlocal = 0, and under CUDA-graph replay after moving the atoms.
`python test_edge_build.py <lammps.data> [cutoff] [skin]` times the op on a real box."""
import sys

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

import jet_mlip
from jet_mlip import ops

pytestmark = pytest.mark.skipif(not (jet_mlip.is_available() and ops.has_op('edge_build')), reason='jet_mlip._C not built')
RC, SKIN = 5.5, 2.0


def host_edges(x, cand_ptr, cand_j, center, owner, cutoff_sq):
    """pair_e3gnn::compute's loop: numpy evaluates each op separately (no FMA), same order as the host."""
    nlocal = len(center)
    ii = np.repeat(np.arange(nlocal), np.diff(cand_ptr))
    d = x[cand_j] - x[center[ii]]
    r = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]
    keep = (owner[cand_j] >= 0) & (r < cutoff_sq)
    rowptr = np.zeros(nlocal + 1, np.int32)
    rowptr[1:] = np.cumsum(np.bincount(ii[keep], minlength=nlocal))
    return rowptr, owner[cand_j[keep]].astype(np.int32), ii[keep].astype(np.int32), d[keep].astype(np.float32)


def make_box(pos, L, rng, foreign=0.1, empty=()):
    """LAMMPS-like arrays for nodes at pos in a periodic cube: local rows in random order (center[ii] = row of
    node ii), ghost images within RC+SKIN of a face (owner = the original's node; a `foreign` fraction gets -1
    = other-rank atoms), candidates = every row j != i within RC+SKIN of row i, shuffled per node; nodes in
    `empty` get no candidates."""
    nlocal = len(pos)
    center = rng.permutation(nlocal).astype(np.int32)
    x = np.empty_like(pos)
    x[center] = pos
    owner = np.empty(nlocal, np.int32)
    owner[center] = np.arange(nlocal)
    shifts = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)], float) * L
    g = (x[:, None, :] + shifts[None]).reshape(-1, 3)
    gowner = np.repeat(owner, len(shifts))
    m = np.all((g > -(RC + SKIN)) & (g < L + RC + SKIN), axis=1)
    g, gowner = g[m], gowner[m]
    gowner[rng.random(len(gowner)) < foreign] = -1
    x, owner = np.concatenate([x, g]), np.concatenate([owner, gowner])
    cands = cKDTree(x).query_ball_point(x[center], RC + SKIN)
    cands = [[] if ii in empty else [j for j in rng.permutation(c) if j != center[ii]] for ii, c in enumerate(cands)]
    cand_ptr = np.zeros(nlocal + 1, np.int32)
    cand_ptr[1:] = np.cumsum([len(c) for c in cands])
    cand_j = np.concatenate([np.asarray(c, np.int32) for c in cands] or [np.zeros(0, np.int32)])
    return x, cand_ptr, cand_j, center, owner


def run_op(x, cand_ptr, cand_j, center, owner, cap, cutoff_sq=RC * RC):
    dev = [torch.as_tensor(a).cuda() for a in (x, cand_ptr, cand_j, center, owner)]
    out = [torch.full((len(center) + 1,), -7, dtype=torch.int32, device='cuda'),
           torch.full((cap,), -7, dtype=torch.int32, device='cuda'), torch.full((cap,), -7, dtype=torch.int32, device='cuda'),
           torch.full((cap, 3), 123.0, device='cuda'), torch.full((1,), -7, dtype=torch.int32, device='cuda')]
    ops.edge_build(*dev, cutoff_sq, *out)
    return dev, out


def check(out, ref, cap):
    rowptr, nbr, ctr, vec, total = [t.cpu() for t in out]
    r_rowptr, r_nbr, r_ctr, r_vec = ref
    n = min(cap, len(r_nbr))
    assert total.item() == len(r_nbr)
    assert torch.equal(rowptr, torch.as_tensor(r_rowptr))
    assert torch.equal(nbr[:n], torch.as_tensor(r_nbr[:n]))
    assert torch.equal(ctr[:n], torch.as_tensor(r_ctr[:n]))
    assert torch.equal(vec[:n].view(torch.int32), torch.as_tensor(r_vec[:n]).view(torch.int32))   # bitwise, -0.0 too
    assert (nbr[n:] == -7).all() and (ctr[n:] == -7).all() and (vec[n:] == 123.0).all()   # tail untouched


@pytest.mark.parametrize('nlocal,L', [(300, 20.0), (24, 5.0)])   # L = 5 < RC: self-image ghosts are edges (nbr == ctr)
def test_bitwise(nlocal, L):
    rng = np.random.default_rng(nlocal)
    pos = rng.random((nlocal, 3)) * L
    pos[0], pos[1] = (1.0, 1.0, 1.0), (1.0 + RC, 1.0, 1.0)   # exact r = RC: 6.5 - 1.0 = 5.5 exactly, excluded by <
    x, cand_ptr, cand_j, center, owner = make_box(pos, L, rng, empty=(2, 3))
    ref = host_edges(x, cand_ptr, cand_j, center, owner, RC * RC)
    if L > 2 * RC:   # in the small box an image of atom 1 is inside the cutoff
        assert center[1] in cand_j[cand_ptr[0]:cand_ptr[1]] and 1 not in ref[1][ref[0][0]:ref[0][1]]
    assert ref[0][3] == ref[0][2] == ref[0][4]   # empty rows
    assert (owner[cand_j] < 0).any() and (L > RC or (ref[1] == ref[2]).any())
    dev, out = run_op(x, cand_ptr, cand_j, center, owner, cap=len(ref[1]) + 5)
    check(out, ref, cap=len(ref[1]) + 5)
    _, out = run_op(x, cand_ptr, cand_j, center, owner, cap=len(ref[1]) // 2)
    check(out, ref, cap=len(ref[1]) // 2)


def test_nlocal_zero():
    e = np.zeros(0, np.int32)
    _, out = run_op(np.zeros((0, 3)), np.zeros(1, np.int32), e, e, e, cap=4)
    assert out[0].tolist() == [0] and out[4].item() == 0


def test_cuda_graph():
    rng = np.random.default_rng(7)
    x, cand_ptr, cand_j, center, owner = make_box(rng.random((200, 3)) * 15.0, 15.0, rng)
    cap = len(cand_j)
    dev, out = run_op(x, cand_ptr, cand_j, center, owner, cap)   # eager warm-up (cub temp storage)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ops.edge_build(*dev, RC * RC, *out)
    for _ in range(3):
        x = x + rng.normal(scale=0.1, size=x.shape)
        dev[0].copy_(torch.as_tensor(x))
        g.replay()
        check(out, host_edges(x, cand_ptr, cand_j, center, owner, RC * RC), cap)


if __name__ == '__main__':   # timing on a LAMMPS data file: python test_edge_build.py box.data [cutoff] [skin]
    lines = open(sys.argv[1]).read().split('\n')
    RC = float(sys.argv[2]) if len(sys.argv) > 2 else RC
    SKIN = float(sys.argv[3]) if len(sys.argv) > 3 else SKIN
    L = float(next(l for l in lines if 'xlo' in l).split()[1])
    k = lines.index(next(l for l in lines if l.startswith('Atoms'))) + 2
    pos = np.array([list(map(float, l.split()[2:5])) for l in lines[k:k + int(lines[2].split()[0])]]) % L
    x, cand_ptr, cand_j, center, owner = make_box(pos, L, np.random.default_rng(0), foreign=0.0)
    dev, out = run_op(x, cand_ptr, cand_j, center, owner, cap=len(cand_j))
    check(out, host_edges(x, cand_ptr, cand_j, center, owner, RC * RC), len(cand_j))
    print(f'nlocal {len(center)} nall {len(x)} cand/node {len(cand_j) / len(center):.1f} edges/node {out[4].item() / len(center):.1f}')
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ops.edge_build(*dev, RC * RC, *out)
    for name, fn in [('eager', lambda: ops.edge_build(*dev, RC * RC, *out)), ('graph', g.replay)]:
        ts = []
        for _ in range(50):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); e.synchronize()
            ts.append(s.elapsed_time(e))
        print(f'{name}: median {np.median(ts):.3f} ms, min {min(ts):.3f} ms')
