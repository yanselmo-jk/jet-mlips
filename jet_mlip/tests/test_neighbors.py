"""neighbors.neighbor_list vs ase (and matscipy when installed): the same set of (i, j, S) on a random triclinic cell
with unwrapped atoms, a cell shorter than the cutoff (self-images), a slab, a wire and a molecule. Runs the torch
reference on the CPU and, when the library is built, the cand_build path on the GPU."""
import numpy as np
import pytest
import torch
from ase.geometry import complete_cell
from ase.neighborlist import primitive_neighbor_list

import jet_mlip
from jet_mlip import neighbors, ops

DEVICES = ['cpu'] + (['cuda'] if jet_mlip.is_available() and ops.has_op('cand_build') else [])
REFS = [lambda pos, cell, pbc, rc: primitive_neighbor_list('ijS', pbc, cell, pos, rc)]
try:
    from matscipy.neighbours import neighbour_list
    REFS.append(lambda pos, cell, pbc, rc: neighbour_list('ijS', positions=pos, cell=cell, pbc=pbc, cutoff=rc))
except ImportError:
    pass


def cases():
    rng = np.random.default_rng(0)
    tri = np.array([[11.0, 0.5, -0.8], [2.5, 12.0, 1.0], [-1.5, 3.0, 13.0]])
    yield 'triclinic', (rng.random((300, 3)) * 3 - 1) @ tri, tri, (True, True, True), 5.0   # atoms up to a cell away
    small = np.array([[2.6, 0.0, 0.0], [0.9, 2.8, 0.0], [0.4, -0.7, 3.1]])
    yield 'small', rng.random((2, 3)) @ small, small, (True, True, True), 6.0
    slab = np.array([[9.0, 0.0, 0.0], [3.0, 8.0, 0.0], [1.0, 2.0, 30.0]])
    yield 'slab', rng.random((120, 3)) * (1, 1, 0.3) @ slab, slab, (True, True, False), 5.0
    yield 'wire', rng.random((20, 3)) * (4.0, 6.0, 6.0), np.diag([4.0, 0.0, 0.0]), (True, False, False), 6.0
    yield 'molecule', rng.random((40, 3)) * 9.0 - 30.0, np.zeros((3, 3)), (False, False, False), 5.0


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('case', list(cases()), ids=lambda c: c[0])
def test_same_pairs(case, device):
    _, pos, cell, pbc, rc = case
    i, j, S = neighbors.neighbor_list(torch.as_tensor(pos, device=device), cell, pbc, rc)
    assert i.dtype == j.dtype == S.dtype == torch.int64 and (i[1:] >= i[:-1]).all()
    got = [tuple(r) for r in torch.cat([i[:, None], j[:, None], S], 1).tolist()]
    assert len(got) == len(set(got)) > 0
    for ref in REFS:
        ri, rj, rS = ref(pos, complete_cell(cell), pbc, rc)
        assert set(got) == set(zip(ri.tolist(), rj.tolist(), *rS.T.tolist()))
    # the pairs are the ones within the cutoff, with the shift convention D = pos[j] - pos[i] + S @ cell
    d = torch.as_tensor(pos)[j.cpu()] - torch.as_tensor(pos)[i.cpu()] + S.cpu().double() @ torch.as_tensor(cell)
    assert d.norm(dim=1).max() < rc
