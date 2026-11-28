"""jet_nequix.convert: the patched Nequix model (fused table convolution) reproduces the upstream torch model's
messages, energies and forces on the CPU torch reference path.

No jax: `nequix/__init__.py` imports it, so the torch implementation is loaded through
`jet_nequix.convert.load_model` and `preprocess_graph` is copied from nequix/data.py (see below)."""
import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk

if importlib.util.find_spec('nequix') is None:   # find_spec does not execute nequix/__init__.py (imports jax)
    pytest.skip('nequix not installed', allow_module_level=True)
from e3nn import o3  # noqa: E402

from jet_nequix import convert  # noqa: E402
from jet_nequix.convert import bessel_basis, load_model, polynomial_cutoff  # noqa: E402

MODELS = os.environ.get('NEQUIX_DIR')   # folder holding nequix-mp-1.pt


def structures():
    """A rattled primitive Si diamond cell (periodic) and a Li/C/H/O molecule in a box (no cell)."""
    # Nequix's TP output is ~5.5k wide, so the fp64 reference autograd graph grows with the edge count: the
    # primitive 2-atom cell keeps the run small (a rattled 2x2x2 supercell peaks at ~3.2 GB RSS on CPU).
    rng = np.random.default_rng(0)
    si = bulk('Si', 'diamond', a=5.43)
    si.positions += rng.uniform(-0.05, 0.05, si.positions.shape)
    mol = Atoms('LiCHO', positions=rng.uniform(1.0, 8.0, size=(4, 3)))
    return {'Si diamond': si, 'LiCHO box': mol}


@pytest.fixture(scope='module', autouse=True)
def fp64_default():
    """e3nn's TensorProduct builds its Clebsch-Gordan constants in torch's default dtype: fp32 by default, which
    would cap the upstream fp64 model at ~1e-7. Build fp64 ones for the comparison and restore afterwards."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.fixture(scope='module')
def model(fp64_default):
    if not MODELS:
        pytest.skip('NEQUIX_DIR not set')
    path = Path(MODELS) / 'nequix-mp-1.pt'
    if not path.exists():
        pytest.skip(f'{path} not present')
    m, config = load_model(str(path), False)
    return m.double(), config


def preprocess_graph(atoms, atom_indices, cutoff):
    """nequix/data.py:preprocess_graph without its `targets` branch (that module imports jax at top level)."""
    from ase.geometry import complete_cell
    from matscipy.neighbours import neighbour_list

    cell = complete_cell(atoms.cell)
    src, dst, shift = neighbour_list('ijS', positions=atoms.positions, cell=cell, pbc=atoms.pbc, cutoff=cutoff)
    return {
        'n_node': np.array([len(atoms)]).astype(np.int32), 'n_edge': np.array([len(src)]).astype(np.int32),
        'senders': dst.astype(np.int32), 'receivers': src.astype(np.int32),
        'species': np.array([atom_indices[n] for n in atoms.get_atomic_numbers()]).astype(np.int32),
        'positions': atoms.positions.astype(np.float32), 'shifts': shift.astype(np.float32),
        'cell': atoms.cell.astype(np.float32) if atoms.pbc.all() else None,
    }


def inputs(atoms, config, dtype=torch.float64):
    """The model's own data pipeline -> forward arguments."""
    g = preprocess_graph(atoms, {n: i for i, n in enumerate(sorted(config['atomic_numbers']))}, config['cutoff'])
    t = torch.from_numpy
    return {
        'species': t(g['species']).long(), 'positions': t(g['positions']).to(dtype),
        'edge_attr': t(g['shifts']).to(dtype),
        'edge_index': torch.stack([t(g['senders']), t(g['receivers'])]).long(),
        'cell': t(g['cell'])[None].to(dtype) if g['cell'] is not None else None,
        'n_node': t(g['n_node']), 'n_edge': t(g['n_edge']),
        'n_graph': torch.zeros(len(atoms), dtype=torch.int64),
    }


def forward(m, d):
    e, f, _ = m(d['species'], d['positions'], d['edge_attr'], d['edge_index'], d['cell'], d['n_node'], d['n_edge'],
                d['n_graph'])
    return e.sum(), f


def edge_vectors(d):
    r = d['positions'][d['edge_index'][0]] - d['positions'][d['edge_index'][1]]
    return r + d['edge_attr'] @ d['cell'][0] if d['cell'] is not None else r


@pytest.mark.parametrize('name', list(structures()))
def test_layer_matches_upstream(model, name):
    m, config = model
    d = inputs(structures()[name], config)
    r, (send, recv) = edge_vectors(d), d['edge_index']
    rn = r.norm(dim=-1)
    basis = bessel_basis(rn, config['radial_basis_size'], config['cutoff']) \
        * polynomial_cutoff(rn, config['cutoff'], config['radial_polynomial_p'])[:, None]
    sh = o3.spherical_harmonics(o3.Irreps.spherical_harmonics(config['lmax']), r, normalize=True,
                                normalization='component')
    for layer in m.layers:
        x = torch.randn(len(d['species']), layer.tp.irreps_in1.dim, dtype=torch.float64)
        msg = layer.tp(layer.linear_1(x)[send], sh, layer.radial_mlp(basis))
        ref = layer.sort(torch.zeros(len(x), msg.shape[1], dtype=torch.float64).index_add_(0, recv, msg))
        ref = ref / np.sqrt(layer.avg_n_neighbors)
        out = convert.JetTPScatter(m, layer, radial_table=0)(layer.linear_1(x), r, send, recv)
        assert out.shape == ref.shape
        err = (out - ref).abs().max()
        assert err < 1e-8 * ref.abs().max(), f'{name}: {err.item():.3e}'


@pytest.mark.parametrize('name', list(structures()))
@pytest.mark.parametrize('radial_table', [0, 2048])
def test_patched_model_matches_upstream(model, name, radial_table):
    m, config = model
    d = inputs(structures()[name], config)
    e0, f0 = forward(m, d)
    patched = convert.patch_model(copy.deepcopy(m), radial_table=radial_table)
    assert all(type(layer).__name__ == 'JetConvolution' for layer in patched.layers)
    e1, f1 = forward(patched, d)
    de, df = abs(e1.item() - e0.item()), (f1 - f0).abs().max().item()
    # exact W(r) (radial_table=0) matches to fp64 round-off; the table adds its interpolation error on forces
    assert de < (1e-10 if radial_table == 0 else 1e-5) * abs(e0.item()), f'{name}: dE {de:.3e}'
    assert df < (1e-8 if radial_table == 0 else 1e-5), f'{name}: dF {df:.3e}'
    print(f'{name} radial_table={radial_table}: dE {de:.3e} eV  max|dF| {df:.3e} eV/A')


def test_configs_are_a_weight_permutation(model):
    """Weight layout: the padded weight map is a permutation of the MLP columns (identity when no mul is padded),
    and the model's instruction order agrees with the compiled config of the same key (the key does not encode it)."""
    m, _ = model
    from jet_mlip.configs import nequix_configs
    compiled = {c.key: c for c in nequix_configs()}
    for layer in m.layers:
        cfg, (xcols, wcols, outcols) = convert.config_from_layer(layer)
        assert cfg.n_weight == (layer.tp.weight_numel if wcols is None else cfg.n_weight)
        assert cfg.irreps_sh == layer.tp.irreps_in2
        if wcols is not None:   # the 89-species one-hot padded to 96: MLP columns land in the padded layout
            assert sorted(wcols[0].tolist()) == list(range(layer.tp.weight_numel))
            assert int(wcols[1].max()) < cfg.n_weight
        if xcols is not None:
            assert xcols.numel() == layer.tp.irreps_in1.dim
        if outcols is not None:
            assert outcols.numel() == layer.sort.irreps_sorted.dim
        assert cfg.key in compiled, f'jet_mlip.configs.nequix_configs() misses {cfg.key}'
        assert compiled[cfg.key].instructions == cfg.instructions, 'compiled fiber order differs from the model'
