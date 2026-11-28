"""jet_mace.ase_graph: the ASE calculator's CUDA-graph step of a model with one radial table per element pair
(MACE-MPA-0) when an element appears that the tables lack, at the same atom count: an eager step rebuilds the tables and
a new graph reads them (the old graph would keep the old tables). The plain graph-vs-eager check of the calculator is
tests/test_ase_calculators.py::test_mace."""
import os
from unittest import mock

import mace  # noqa: F401  isort:skip
import numpy as np
import pytest
import torch
from ase.build import bulk

pytest.importorskip('jet_mlip')
from jet_mace import convert  # noqa: E402
from jet_mlip import ase_graph  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')


def results(calc, atoms, graph=True):
    """(energy, forces, the step came from the graph) of a new step."""
    with mock.patch.dict(os.environ, JET_ASE_GRAPH='1' if graph else '0'), mock.patch.object(
            ase_graph.AseGraph, 'compute', autospec=True, side_effect=ase_graph.AseGraph.compute) as step:
        calc.calculate(atoms.copy(), ['energy'])
    return calc.results['energy'], calc.results['forces'].copy(), step.called


def test_new_element_gets_new_tables():
    from mace.calculators import mace_mp
    calc = mace_mp(model='medium-mpa-0', device='cuda', default_dtype='float32')
    calc.models[0] = convert.patch_model(calc.models[0], cueq=True)
    lif = bulk('LiF', 'rocksalt', a=4.0, cubic=True).repeat((2, 2, 2))
    lif.rattle(0.05, seed=0)
    licl = lif.copy()
    licl.numbers[licl.numbers == 9] = 17
    assert not results(calc, lif)[2]    # the first call is eager
    assert results(calc, lif)[2]
    assert not results(calc, licl)[2]   # Cl is not in the tables: eager, which adds it
    results(calc, licl)                 # the new graph's first call
    for atoms in (licl, lif):           # the new graph, then the first element set on it
        e1, f1, used = results(calc, atoms)
        e0, f0, _ = results(calc, atoms, graph=False)
        assert used
        assert abs(e1 - e0) / len(atoms) < 1e-5, (e0, e1)
        assert np.abs(f1 - f0).max() < 2e-5
