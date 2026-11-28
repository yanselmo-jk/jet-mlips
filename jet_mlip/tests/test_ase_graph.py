"""jet_mlip.ase_graph.AseGraph on a toy pair energy: the replayed CUDA graph's energies, forces and virial against
autograd through the positions and a strain; the eager first call, a pair list that moves with the atoms, other
elements on the same pair list (weights filled per list: alloc / prep), the node rows in spatial and in the caller's
order, the off switch and another atom count."""
import copy
import pickle

import pytest
import torch

import jet_mlip
from jet_mlip import ase_graph, neighbors

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and jet_mlip.is_available()),
                                reason='needs CUDA and jet_mlip._C')

R_CUT = 3.0


def energies(vec, i, j, types, n):
    """E_i = sum over the pairs (i, j) of (1 + t_i + 2 t_j) (r_cut - r)^2: 0 for a padded edge, whose vector is
    (r_cut, 0, 0)."""
    phi = (1 + types[i] + 2 * types[j]) * (R_CUT - vec.norm(dim=1)).square()
    return torch.zeros(n, dtype=vec.dtype, device=vec.device).index_add_(0, i, phi)


def alloc(g):
    g.s_w = torch.zeros(g.key[2], dtype=torch.float64, device=g.device)


def prep(g):
    g.s_w.copy_(1 + g.s_types[g.s_ei[0]] + 2 * g.s_types[g.s_ei[1]])   # per pair list, not per step


def forward(g, vec):
    phi = g.s_w * (R_CUT - vec.norm(dim=1)).square()
    return torch.zeros(g.key[1], dtype=vec.dtype, device=vec.device).index_add_(0, g.s_ei[0], phi)


def graph(device='cuda', **kw):
    return ase_graph.AseGraph(forward, device, R_CUT, vec_dtype=torch.float64, alloc=alloc, prep=prep, **kw)


def system(n, seed, size=1):
    cell = size * torch.tensor([[7.0, 0, 0], [1.0, 6.5, 0], [0.5, -0.8, 8.0]], dtype=torch.float64)
    frac = torch.rand(n, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return (frac @ cell).cuda(), cell.cuda()


def kinds(n, seed=0):
    return torch.randint(3, (n,), generator=torch.Generator().manual_seed(seed)).cuda()


def step(g, pos, cell, seed=0):
    """-> (due, the graph's (e, F, w) or None, the reference (e, F, dE/dstrain)); seed: the atoms' types."""
    i, j, S = neighbors.neighbor_list(pos, cell, (True,) * 3, R_CUT)
    due = g.due(len(pos), i.numel())
    types = kinds(len(pos), seed)
    out = g.compute(types, i, j, pos[j] - pos[i] + S.double() @ cell, pos) if due else None
    x = pos.clone().requires_grad_(True)
    eps = torch.zeros(3, 3, dtype=torch.float64, device=pos.device, requires_grad=True)
    vec = (x[j] - x[i] + S.double() @ cell) @ (torch.eye(3, dtype=torch.float64, device=pos.device) + (eps + eps.t()) / 2)
    e = energies(vec, i, j, types, len(pos))
    gx, ge = torch.autograd.grad(e.sum(), [x, eps])
    return due, out, (e.detach(), -gx, ge)


def close(out, ref):
    (e, f, w), (e0, f0, w0) = out, ref
    return bool((e - e0).abs().max() < 1e-9 and (f - f0).abs().max() < 1e-9 and ((w + w.t()) / 2 - w0).abs().max() < 1e-8)


def test_graph_matches_autograd(monkeypatch):
    monkeypatch.delenv('JET_ASE_GRAPH', raising=False)
    g = graph()
    pos, cell = system(60, 0)
    assert not step(g, pos, cell)[0]   # the first call with 60 atoms stays eager
    rng = torch.Generator().manual_seed(1)
    for _ in range(4):   # then the graph: one capture, replays on a pair list that changes with the atoms
        pos = pos + 0.05 * torch.randn(pos.shape, generator=rng, dtype=torch.float64).cuda()
        due, out, ref = step(g, pos, cell)
        assert due and close(out, ref)
    assert g.n_captures == 1 and g.n_replays == 3 and g.n_lists >= 2
    due, out, ref = step(g, pos, cell, seed=1)   # other elements, the same pair list
    assert due and close(out, ref)
    monkeypatch.setenv('JET_ASE_GRAPH', '0')
    assert not step(g, pos, cell)[0]
    monkeypatch.delenv('JET_ASE_GRAPH')
    pos, cell = system(35, 2)   # another structure: eager once, then a capture with its own capacities
    assert not step(g, pos, cell)[0]
    due, out, ref = step(g, pos, cell)
    assert due and close(out, ref) and g.key[0] == 35 and g.n_captures == 2
    assert copy.deepcopy(g) is None and pickle.loads(pickle.dumps(g)) is None   # a calculator's copy makes its own


def test_rows_in_spatial_order(monkeypatch):
    """Atoms in random order in a box of many bins run on node rows in spatial order; atoms given in that order keep
    their numbering. The results come back in the caller's order either way."""
    monkeypatch.delenv('JET_ASE_GRAPH', raising=False)
    monkeypatch.delenv('JET_ASE_SORT', raising=False)
    g = graph()
    pos, cell = system(480, 3, 3)
    step(g, pos, cell)   # eager
    due, out, ref = step(g, pos, cell)
    assert due and close(out, ref) and g._order[1] is not None and g.s_types.tolist() != kinds(480).tolist()
    monkeypatch.setenv('JET_ASE_SORT', '0')   # the same step on the caller's rows: another list, no new capture
    due, out, ref = step(g, pos, cell)
    assert close(out, ref) and g.s_types.tolist() == kinds(480).tolist() and g.n_captures == 1
    monkeypatch.delenv('JET_ASE_SORT')
    pos, g._order = pos[g._order[0]], None   # the atoms in bin order: nothing to gain
    due, out, ref = step(g, pos, cell)
    assert close(out, ref) and g._order[1] is None


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='needs two CUDA devices')
def test_not_the_current_device(monkeypatch):
    """A calculator on cuda:1, called while cuda:0 is the current device: the capture must be cuda:1's."""
    monkeypatch.delenv('JET_ASE_GRAPH', raising=False)
    g = graph('cuda:1')
    compute = g.compute

    def from_device_0(*args):
        with torch.cuda.device(0):
            return compute(*args)
    g.compute = from_device_0
    with torch.cuda.device(1):   # the test's own tensors and its neighbour list
        pos, cell = system(60, 0)
        step(g, pos, cell)   # eager
        for k in range(3):
            due, out, ref = step(g, pos + 0.02 * k, cell)
            assert due and close(out, ref)
    assert g.n_captures == 1 and g.n_replays == 2


def test_compiled_forward(monkeypatch):
    """compile(): its forward takes over JET_ASE_COMPILE seconds after a capture, with a capture of its own; a compile
    that fails in the warm-up steps leaves the plain forward."""
    monkeypatch.delenv('JET_ASE_GRAPH', raising=False)
    monkeypatch.setenv('JET_ASE_COMPILE', '0')
    calls = []

    def fast(g, vec):
        calls.append(1)
        return forward(g, vec)

    def broken(g, vec):
        raise torch._dynamo.exc.TorchDynamoException('no compile')

    for compiled in (fast, broken):
        g = graph(compile=lambda: compiled)
        pos, cell = system(60, 0)
        step(g, pos, cell)   # eager
        assert close(*step(g, pos + 0.01, cell)[1:]) and g.forward is compiled   # captured plain; compiled is next
        if compiled is fast:
            assert close(*step(g, pos + 0.02, cell)[1:]) and calls and g.forward is fast
        else:
            with pytest.warns(UserWarning, match='torch.compile failed'):
                assert close(*step(g, pos + 0.02, cell)[1:])
            assert g.compile is None and g.forward is g.plain
        assert g.n_captures == 2
