"""common.radial_table: tabulate any W(r) with its derivative, node difference and next-node derivative on the
jet_mlip grid."""
import torch

from common import radial_table


def w_of_r(r):   # (n,) -> (n, 2): analytic, so dW/dr is known
    return torch.stack([torch.sin(r), r * r], dim=1)


def test_table_layout_and_derivative():
    n, r_cut = 65, 4.0
    tab = radial_table.make_radial_table(w_of_r, r_cut, n)
    assert tab.shape == (n, 8) and tab.dtype == torch.float32   # (n, 4*NW): [W, dW/dr, node diff/h, dW/dr(r_k+1)] per column
    r = torch.linspace(0, r_cut, n, dtype=torch.float64).clamp_min(1e-6)
    t = tab.view(n, 2, 4).double()   # per column: [W, dW/dr, node diff/h, dW/dr(r_k+1)]
    v0, m0, d, m1 = t[:, :, 0], t[:, :, 1], t[:, :, 2], t[:, :, 3]
    val, dval = w_of_r(r), torch.stack([torch.cos(r), 2 * r], 1)
    assert torch.allclose(v0, val, atol=1e-6)
    assert torch.allclose(m0, dval, atol=1e-5)
    h = r_cut / (n - 1)
    assert torch.allclose(d[:-1], (val[1:] - val[:-1]) / h, atol=1e-4)
    assert torch.equal(m1[:-1], m0[1:])   # the next row's slope, tabulated so the kernel never gathers row k+1
    assert d[-1].abs().max() == 0 and m1[-1].abs().max() == 0


def test_stacked_tables_per_type():
    tabs = radial_table.make_radial_tables([w_of_r, lambda r: 2 * w_of_r(r)], 4.0, 33)
    assert tabs.shape == (2, 33, 8)
    assert torch.allclose(tabs[1, :, :2], 2 * tabs[0, :, :2])
