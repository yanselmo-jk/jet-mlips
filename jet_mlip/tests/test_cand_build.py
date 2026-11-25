"""cand_build CUDA op vs a KD-tree: per center the same candidate set (rows j != i within cut), on a periodic box
with ghosts, a sparse box (bin enlargement), a cubic lattice with pairs at exactly cut, and ncenters = 0."""
import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

import jet_mlip
from jet_mlip import ops

pytestmark = pytest.mark.skipif(not (jet_mlip.is_available() and ops.has_op('cand_build')), reason='jet_mlip._C not built')
CUT = 7.5


def ref_sets(x, ncenters, cut):
    return [set(j for j in c if j != i) for i, c in enumerate(cKDTree(x).query_ball_point(x[:ncenters], cut - 1e-9))]


def check(x, ncenters, cut=CUT):
    ptr, cand = ops.cand_build(torch.as_tensor(x).cuda(), ncenters, cut * cut)
    ptr, cand = ptr.cpu().numpy(), cand.cpu().numpy()
    assert ptr[0] == 0 and len(cand) == ptr[-1]
    for i, s in enumerate(ref_sets(x, ncenters, cut)):
        got = cand[ptr[i]:ptr[i + 1]]
        assert len(got) == len(set(got)) and set(got) == s, i
    ptr2, cand2 = ops.cand_build(torch.as_tensor(x).cuda(), ncenters, cut * cut)   # deterministic order
    assert np.array_equal(ptr2.cpu().numpy(), ptr) and np.array_equal(cand2.cpu().numpy(), cand)


def test_periodic_with_ghosts():
    rng = np.random.default_rng(0)
    L, n = 24.0, 1500
    pos = rng.random((n, 3)) * L
    shifts = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)], float) * L
    g = (pos[:, None, :] + shifts[None]).reshape(-1, 3)
    g = g[np.all((g > -CUT) & (g < L + CUT), axis=1)]
    check(np.concatenate([pos, g]), n)


def test_sparse_and_edges():
    rng = np.random.default_rng(1)
    check(np.concatenate([rng.random((40, 3)) * 5, rng.random((10, 3)) * 5 + 500.0]), 50)   # two clusters, empty space
    a = 2.5   # lattice spacing = cut / 3: pairs at exactly cut are excluded (strict <)
    x = np.array([(i, j, k) for i in range(6) for j in range(6) for k in range(6)], float) * a
    check(x, len(x), cut=3 * a)
    ptr, cand = ops.cand_build(torch.as_tensor(x).cuda(), 0, CUT * CUT)
    assert ptr.tolist() == [0] and cand.numel() == 0
