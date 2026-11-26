"""force_reduce CUDA op vs the torch reduction pair_e3gnn used before (index_add, voigt rows, sum) in float64: random
CSR with empty rows, padding rows past the edge count, more force rows than CSR rows (halo), and under CUDA-graph replay."""
import pytest
import torch

import jet_mlip
from jet_mlip import ops

pytestmark = pytest.mark.skipif(not (jet_mlip.is_available() and ops.has_op('force_reduce')), reason='jet_mlip._C not built')


def case(nrows=700, nforce=900, cap_extra=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, 60, (nrows,), generator=g)
    counts[::17] = 0
    rowptr = torch.zeros(nrows + 1, dtype=torch.int32)
    rowptr[1:] = counts.cumsum(0)
    E = int(rowptr[-1])
    cap = E + cap_extra
    nbr = torch.randint(0, nforce, (cap,), generator=g, dtype=torch.int32)
    dEdr = torch.randn(cap, 3, generator=g)
    dEdr[E:] = 0   # rows past the count: dE/dr is exactly 0 there
    vec = torch.randn(cap, 3, generator=g) * 3
    return rowptr, nbr, dEdr, vec, nforce, E


def reference(rowptr, nbr, dEdr, vec, nforce, E):
    ctr = torch.repeat_interleave(torch.arange(len(rowptr) - 1), (rowptr[1:] - rowptr[:-1]).long())
    d, r, j = dEdr[:E].double(), vec[:E].double(), nbr[:E].long()
    f = torch.zeros(nforce, 3, dtype=torch.float64)
    f.index_add_(0, ctr, d)
    f.index_add_(0, j, -d)
    voigt = torch.cat([r * d, (r[:, 0] * d[:, 1])[:, None], (r[:, 1] * d[:, 2])[:, None], (r[:, 2] * d[:, 0])[:, None]], 1)
    return f.flatten(), -voigt.sum(0)


def run(rowptr, nbr, dEdr, vec, nforce):
    out = torch.full((3 * nforce + 8,), 7.0, dtype=torch.float64, device='cuda')
    ops.force_reduce(dEdr.cuda(), vec.cuda(), rowptr.cuda(), nbr.cuda(), out)
    return out.cpu()


@pytest.mark.parametrize('seed', [0, 1])
def test_matches_reference(seed):
    rowptr, nbr, dEdr, vec, nforce, E = case(seed=seed)
    f, vir = reference(rowptr, nbr, dEdr, vec, nforce, E)
    out = run(rowptr, nbr, dEdr, vec, nforce)
    assert torch.allclose(out[:3 * nforce], f, atol=1e-9)
    assert torch.allclose(out[3 * nforce + 1:3 * nforce + 7], vir, atol=1e-9)
    assert out[3 * nforce] == 0 and out[3 * nforce + 7] == 0


def test_empty_and_graph():
    out = torch.full((3 * 5 + 8,), 7.0, dtype=torch.float64, device='cuda')
    z = torch.zeros(0, 3, device='cuda')
    ops.force_reduce(z, z, torch.zeros(1, dtype=torch.int32, device='cuda'), torch.zeros(0, dtype=torch.int32, device='cuda'), out)
    assert out.abs().sum() == 0
    rowptr, nbr, dEdr, vec, nforce, E = case(seed=2)
    dd, vv, rr, nn = dEdr.cuda(), vec.cuda(), rowptr.cuda(), nbr.cuda()
    out = torch.zeros(3 * nforce + 8, dtype=torch.float64, device='cuda')
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        ops.force_reduce(dd, vv, rr, nn, out)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            ops.force_reduce(dd, vv, rr, nn, out)
    dd.mul_(2)   # replay sees the new dE/dr
    g.replay()
    torch.cuda.synchronize()
    f, vir = reference(rowptr, nbr, dEdr * 2, vec, nforce, E)
    assert torch.allclose(out.cpu()[:3 * nforce], f, atol=1e-9)
    assert torch.allclose(out.cpu()[3 * nforce + 1:3 * nforce + 7], vir, atol=1e-8)
