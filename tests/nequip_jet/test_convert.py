"""jet_nequip.convert: the patched NequIP model (fused table convolution) reproduces the upstream e3nn model's
energy and forces on the CPU torch reference path."""
import copy
import sys
from pathlib import Path

# this test package shadows the installed `nequip` under pytest's prepend import mode: hand the name back
_tests = str(Path(__file__).resolve().parents[1])
_saved = sys.path[:]
sys.path[:] = [p for p in sys.path if str(Path(p or '.').resolve()) != _tests]
sys.modules.pop('nequip', None)
import nequip  # noqa: F401,E402
sys.path[:] = _saved

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from ase import Atoms  # noqa: E402

pytest.importorskip('nequip')
from nequip.data import AtomicDataDict, from_ase  # noqa: E402
from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform  # noqa: E402
from nequip.model import NequIPGNNModel  # noqa: E402
from nequip.utils.global_state import set_global_state  # noqa: E402

from jet_mlip import ops  # noqa: E402
from jet_nequip import convert  # noqa: E402

TYPE_NAMES = ['H', 'C', 'O']


@pytest.fixture(scope='module')
def model():
    set_global_state()
    return NequIPGNNModel(
        num_layers=3, l_max=2, parity=False, num_features=[32, 16, 8], r_max=4.0,
        type_names=TYPE_NAMES, avg_num_neighbors=10.0,
        per_type_energy_shifts={'H': 0.0, 'C': 0.0, 'O': 0.0}, seed=123, model_dtype='float32',
    )


@pytest.fixture(scope='module')
def model_wide():
    """NequIP-OAM-XL-shaped muls (160x0e = 5 slices): every block is wider than the kernel's 128."""
    set_global_state()
    return NequIPGNNModel(
        num_layers=3, l_max=2, parity=False, num_features=[160, 64, 32], r_max=4.0,
        type_names=TYPE_NAMES, avg_num_neighbors=10.0,
        per_type_energy_shifts={'H': 0.0, 'C': 0.0, 'O': 0.0}, seed=123, model_dtype='float32',
        type_embed_num_features=32,
    )


@pytest.fixture(scope='module')
def model_wide_parity():
    """Sliced, with parity: the last block has x irreps no tensor-product path reads (their gradient is zero)."""
    set_global_state()
    return NequIPGNNModel(
        num_layers=3, l_max=2, parity=True, num_features=[160, 64, 32], r_max=4.0,
        type_names=TYPE_NAMES, avg_num_neighbors=10.0,
        per_type_energy_shifts={'H': 0.0, 'C': 0.0, 'O': 0.0}, seed=123, model_dtype='float32',
        type_embed_num_features=32,
    )


PEC = {'H': 2.5, 'C': {'H': 3.0, 'C': 3.5, 'O': 3.2}, 'O': 3.8}   # per_edge_type_cutoff, all < r_max 4.0


@pytest.fixture(scope='module')
def model_pec():
    """per_edge_type_cutoff model (one radial table per distinct cutoff); NequIP-OAM-M conv shapes, compiled into
    jet_mlip, so the CUDA case runs the kernel."""
    set_global_state()
    return NequIPGNNModel(
        num_layers=3, l_max=2, parity=False, num_features=[128, 64, 32], r_max=4.0,
        type_names=TYPE_NAMES, avg_num_neighbors=10.0, per_edge_type_cutoff=PEC,
        per_type_energy_shifts={'H': 0.0, 'C': 0.0, 'O': 0.0}, seed=123, model_dtype='float32',
        type_embed_num_features=32,
    )


@pytest.fixture(scope='module')
def model_wide_pec():
    """Sliced (160x0e) and per_edge_type_cutoff together."""
    set_global_state()
    return NequIPGNNModel(
        num_layers=3, l_max=2, parity=False, num_features=[160, 64, 32], r_max=4.0,
        type_names=TYPE_NAMES, avg_num_neighbors=10.0, per_edge_type_cutoff=PEC,
        per_type_energy_shifts={'H': 0.0, 'C': 0.0, 'O': 0.0}, seed=123, model_dtype='float32',
        type_embed_num_features=32,
    )


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    atoms = Atoms(positions=rng.uniform(0.0, 5.0, size=(16, 3)), numbers=rng.choice([1, 6, 8], size=16))
    atoms.pbc = False
    mapper = ChemicalSpeciesToAtomTypeMapper(
        model_type_names=TYPE_NAMES, chemical_species_to_atom_type_map={t: t for t in TYPE_NAMES},
    )
    return NeighborListTransform(r_max=4.0)(mapper(from_ase(atoms)))


def blocks(model):
    return [m for m in model.modules() if type(m).__name__ == 'InteractionBlock']


@pytest.mark.parametrize('model_name', ['model', 'model_wide'])
def test_block_matches_upstream(request, data, model_name):
    model = request.getfixturevalue(model_name)
    func = model.model.func
    d = func.spharm(data)
    d = func.edge_norm(d)
    d = func.bessel_encode(d)
    d = func.factor(d)
    dst, src = d[AtomicDataDict.EDGE_INDEX_KEY]
    for block in blocks(model):
        x = torch.randn(16, block.tp_scatter.tp.irreps_in1.dim, dtype=torch.float32)
        w = block.edge_mlp(d[AtomicDataDict.EDGE_EMBEDDING_KEY])
        ref = block.tp_scatter(x, d[AtomicDataDict.EDGE_ATTRS_KEY], w, dst, src)
        out = convert.JetTPScatter(model, block)(x, d[AtomicDataDict.EDGE_VECTORS_KEY], None, dst, src)
        assert out.shape == ref.shape
        assert (out - ref).abs().max() < 1e-4 * ref.abs().max() + 1e-5


def _run(m, data):
    d = dict(data)
    d.pop(AtomicDataDict.EDGE_VECTORS_KEY, None)   # else ForceStressOutput takes the ML-IAP edge-force branch
    d[AtomicDataDict.POSITIONS_KEY] = d[AtomicDataDict.POSITIONS_KEY].clone().requires_grad_(True)
    return m(d)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='no CUDA'))])
def test_per_edge_type_needs_no_pruning(model_pec, data, device):
    """Upstream on the pruned neighbour list (what its ML-IAP wrapper does) == the patched model on the full list
    (what the CUDA-graph wrapper hands it). CUDA: the kernel's per-edge table index."""
    from nequip.data.transforms import NeighborListPruneTransform
    pruned = NeighborListPruneTransform(r_max=4.0, per_edge_type_cutoff=PEC, type_names=TYPE_NAMES)(dict(data))
    assert pruned[AtomicDataDict.EDGE_INDEX_KEY].shape[1] < data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
    model_pec = copy.deepcopy(model_pec).to(device)
    data, pruned = (AtomicDataDict.to_(dict(d), device) for d in (data, pruned))
    ref = _run(model_pec, pruned)
    patched = convert.patch_model(copy.deepcopy(model_pec), radial_table=2048)
    assert all(b.tp_scatter.per_pair for b in blocks(patched))
    out = _run(patched, data)
    assert abs(out[AtomicDataDict.TOTAL_ENERGY_KEY].item() - ref[AtomicDataDict.TOTAL_ENERGY_KEY].item()) \
        < 1e-5 * abs(ref[AtomicDataDict.TOTAL_ENERGY_KEY].item())
    f0, f1 = ref[AtomicDataDict.FORCE_KEY], out[AtomicDataDict.FORCE_KEY]
    assert (f1 - f0).abs().max() < 1e-4 * f0.abs().max() + 1e-6


@pytest.mark.parametrize('model_name', ['model', 'model_wide', 'model_wide_parity', 'model_pec', 'model_wide_pec'])
def test_patched_model_matches_upstream(request, data, model_name):
    model = request.getfixturevalue(model_name)

    def run(m):
        d = dict(data)
        d.pop(AtomicDataDict.EDGE_VECTORS_KEY, None)   # else ForceStressOutput takes the ML-IAP edge-force branch
        d[AtomicDataDict.POSITIONS_KEY] = d[AtomicDataDict.POSITIONS_KEY].clone().requires_grad_(True)
        return m(d)

    ref = run(model)
    patched = convert.patch_model(copy.deepcopy(model), radial_table=2048)
    assert all(type(b.tp_scatter).__name__ == 'JetTPScatter' for b in blocks(patched))
    last = blocks(patched)[-1].tp_scatter
    assert model_name != 'model_wide_parity' or last.n_x == len(last.cfgs) + 1   # the unread x columns
    # no block reads the Bessel edge embedding any more: the encoding computes the cutoff only
    assert type(patched.model.func.bessel_encode).__name__ == '_CutoffOnly'
    out = run(patched)
    assert out[AtomicDataDict.EDGE_EMBEDDING_KEY].shape[1] == 0 and torch.equal(
        out[AtomicDataDict.EDGE_CUTOFF_KEY], ref[AtomicDataDict.EDGE_CUTOFF_KEY])
    assert abs(out[AtomicDataDict.TOTAL_ENERGY_KEY].item() - ref[AtomicDataDict.TOTAL_ENERGY_KEY].item()) \
        < 1e-5 * abs(ref[AtomicDataDict.TOTAL_ENERGY_KEY].item())
    f0, f1 = ref[AtomicDataDict.FORCE_KEY], out[AtomicDataDict.FORCE_KEY]
    assert (f1 - f0).abs().max() < 1e-4 * f0.abs().max() + 1e-6


@pytest.mark.parametrize('model_name', ['model', 'model_wide'])
def test_cols_are_a_weight_permutation(request, model_name):
    for block in blocks(request.getfixturevalue(model_name)):
        tp = block.tp_scatter.tp
        if max(mul for mul, _ in tp.irreps_in1) > 128:   # sliced: the union of the slice columns must still be all
            cols = torch.cat([w_cols for _, w_cols, *_ in convert.slice_configs(block)])
        else:
            cols = convert.config_from_block(block)[1]
        assert sorted(cols.tolist()) == list(range(tp.weight_numel))


def test_wide_model_is_sliced(model_wide):
    assert all(len(convert.slice_configs(b)) > 1 for b in blocks(model_wide)[1:])


@pytest.mark.skipif(not (torch.cuda.is_available() and ops.has_op('species_linear') and ops.has_op('gate')),
                    reason='no CUDA jet_mlip glue ops')
@pytest.mark.parametrize('model_name', ['model', 'model_pec'])
def test_glue_matches_e3nn(request, model_name):
    """Every jet glue op against the e3nn module it replaces: forward and grad x, fp32 CUDA."""
    model = copy.deepcopy(request.getfixturevalue(model_name)).cuda()
    assert convert.patch_glue(model, 'sc,linear,gate') == 11   # 3 layers: 6 linears, 2 self-connections, 3 gates
    assert convert.patch_glue(model, 'sc,linear,gate') == 0    # already swapped
    attrs = next(m for m in model.modules() if type(m).__name__ == 'NodeTypeEmbed').embed_module.weight.detach()
    torch.manual_seed(0)
    seen = set()
    for m in model.modules():
        if not isinstance(m, convert._JetGlue):
            continue
        seen.add(type(m).__name__)
        sc = isinstance(m, convert.JetSelfConnection)
        x = torch.randn(300, (m.e3nn.irreps_in1 if sc else m.e3nn.irreps_in).dim, device='cuda', dtype=torch.float32, requires_grad=True)
        args = (attrs[torch.randint(0, len(TYPE_NAMES), (300,), device='cuda')],) if sc else ()
        out, ref = m(x, *args), m.e3nn(x, *args)
        g = torch.randn_like(ref)
        gx, gref = (torch.autograd.grad(o, x, g)[0] for o in (out, ref))
        assert (out - ref).abs().max() <= 1e-5 * ref.abs().max(), type(m).__name__
        assert (gx - gref).abs().max() <= 1e-5 * gref.abs().max(), type(m).__name__
    assert seen == {'JetLinear', 'JetSelfConnection', 'JetGate'}


@pytest.mark.skipif(not (torch.cuda.is_available() and ops.has_op('species_linear') and ops.has_op('gate')),
                    reason='no CUDA jet_mlip glue ops')
def test_local_block_forward(model_pec, data, monkeypatch):
    """The one-rank graph path's block forward (four jet ops, the atom types handed over) against the block's own
    forward with the jet glue (model_pec: its conv shapes are compiled into jet_mlip); a block whose own forward the
    probe does not reproduce keeps it."""
    model = convert.patch_model(copy.deepcopy(model_pec).cuda(), radial_table=2048)
    d = AtomicDataDict.to_(dict(data), 'cuda')
    ref = _run(model, d)
    assert convert.patch_block_forward(model) == 3
    calls = []   # the glue modules' own forwards: the local forward calls the ops itself
    for b in blocks(model):
        b.linear_1.register_forward_hook(lambda *a: calls.append(1))
    for types in (d[AtomicDataDict.ATOM_TYPE_KEY].view(-1), None):   # None: no caller vouches for local rows -> upstream forward
        monkeypatch.setattr(convert, 'TYPES', types)
        del calls[:]
        out = _run(model, d)
        assert bool(calls) == (types is None)
        e0, e1 = ref[AtomicDataDict.TOTAL_ENERGY_KEY].item(), out[AtomicDataDict.TOTAL_ENERGY_KEY].item()
        f0, f1 = ref[AtomicDataDict.FORCE_KEY], out[AtomicDataDict.FORCE_KEY]
        assert abs(e1 - e0) < 1e-5 * abs(e0) and (f1 - f0).abs().max() < 1e-4 * f0.abs().max() + 1e-6
    monkeypatch.setattr(convert, 'TYPES', None)
    other = convert.patch_model(copy.deepcopy(model_pec).cuda(), radial_table=2048)
    cls = type(blocks(other)[0])
    upstream = cls.forward

    def doubled(self, data):   # a forward contract the local forward does not know
        data = upstream(self, data)
        data[AtomicDataDict.NODE_FEATURES_KEY] = 2 * data[AtomicDataDict.NODE_FEATURES_KEY]
        return data
    monkeypatch.setattr(cls, 'forward', doubled)
    assert convert.patch_block_forward(other) == 0 and all(type(b) is cls for b in blocks(other))
