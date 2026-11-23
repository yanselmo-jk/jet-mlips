"""jet_mace.table: the deploy-time radial table reproduces a MACE block's conv_tp weights W(r) (MP-0),
and refuses models whose W depends on the element pair (Agnesi transform: mpa-0 / omat-0)."""
import mace  # noqa: F401  isort:skip
import pytest
import torch

from common import radial_table
from jet_mace import table


def test_interpolate_reproduces_analytic_w():
    """CPU: the 4-column cubic-Hermite lookup of a 1024-row table reproduces W(r) between the nodes."""
    r_cut, n = 4.0, 1024

    def w(r):
        return torch.stack([torch.sin(r), torch.exp(-r)], dim=1)

    tab = radial_table.make_radial_table(w, r_cut, n)
    r = torch.rand(1000, dtype=torch.float64) * r_cut
    assert torch.allclose(table.interpolate(tab, r_cut, r), w(r), atol=1e-6)


def load(name):
    from mace.calculators import mace_mp
    return mace_mp(model=name, device='cpu', default_dtype='float64', return_raw_model=True)


def exact_w(model, i, r):
    """W the way ScaleShiftMACE.forward feeds interaction i (Li-O edges; MP-0 ignores the species)."""
    z = model.atomic_numbers
    node_attrs = torch.nn.functional.one_hot(torch.tensor([list(z).index(3), list(z).index(8)]), len(z)).double()
    edge_index = torch.tensor([[0] * len(r), [1] * len(r)])
    feats, cutoff = model.radial_embedding(r.unsqueeze(-1), node_attrs, edge_index, z)
    assert cutoff is None
    return model.interactions[i].conv_tp_weights(feats)


def test_table_matches_exact_weights():
    model = load('medium')
    r = torch.tensor([0.7, 1.3, 2.9, 4.4, 5.8, 6.5], dtype=torch.float64)
    for i in range(len(model.interactions)):
        rt = table.TableBuilder(model, i, 4097)()
        assert rt.tab.shape[1] == 4 * model.interactions[i].conv_tp.weight_numel
        w = table.interpolate(rt.tab, rt.r_max, r)
        assert torch.allclose(w, exact_w(model, i, r), atol=2e-4, rtol=1e-3), i
        assert w[-1].abs().max() < 1e-12   # beyond r_max the polynomial cutoff zeroes everything


def exact_w_pair(model, i, r, zi, zj):
    z = model.atomic_numbers
    node_attrs = torch.nn.functional.one_hot(torch.tensor([list(z).index(zi), list(z).index(zj)]), len(z)).double()
    edge_index = torch.tensor([[0] * len(r), [1] * len(r)])
    feats, _ = model.radial_embedding(r.unsqueeze(-1), node_attrs, edge_index, z)
    return model.interactions[i].conv_tp_weights(feats)


def test_agnesi_models_get_one_table_per_element_pair():
    """mpa-0: W depends on (r, Z_i, Z_j) through the Agnesi r_0 -> one table per unordered pair of the given species."""
    model = load('medium-mpa-0')
    rt = table.TableBuilder(model, 0, 4097)([3, 8, 6])   # Li O C
    assert rt.tab.shape[0] == 6 and rt.tab.dim() == 3
    r = torch.tensor([0.9, 1.7, 2.9, 4.4, 5.8], dtype=torch.float64)
    for zi, zj in ((3, 8), (8, 3), (6, 6), (3, 6)):
        idx = rt.zz[zi, zj].expand(len(r))
        assert torch.allclose(table.interpolate(rt.tab, rt.r_max, r, idx), exact_w_pair(model, 0, r, zi, zj), atol=2e-4, rtol=1e-3), (zi, zj)
    assert rt.zz[3, 8] == rt.zz[8, 3]


def test_agnesi_model_needs_species():
    with pytest.raises(ValueError, match='species'):
        table.TableBuilder(load("medium-mpa-0"), 0, 65)()
