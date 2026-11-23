"""jet_mace.deploy: writes a LAMMPS ML-IAP model (pickled LAMMPS_MLIAP_MACE around the patched model) whose
forward on a LAMMPS-style batch gives the upstream total energy."""
import mace  # noqa: F401  isort:skip
import pytest
import torch
from ase.build import bulk

pytest.importorskip('jet_mlip')
from jet_mace import deploy  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_deploy_roundtrip(tmp_path):
    from mace.calculators import mace_mp
    atoms = bulk('LiF', 'rocksalt', a=4.0).repeat((2, 2, 2))
    atoms.rattle(stdev=0.1, seed=2)
    ref = mace_mp(model='medium', device='cuda', default_dtype='float32')
    e0 = ref.get_potential_energy(atoms)
    out = tmp_path / 'medium_jet_mace-mliap_lammps.pt'
    deploy.main(['medium', str(out), '--radial_table', '1024'])
    lmp_model = torch.load(out, map_location='cuda', weights_only=False)
    assert lmp_model.element_types == ['Li', 'F'] or 'Li' in lmp_model.element_types
    batch = ref._atoms_to_batch(atoms).to('cuda').to_dict()
    data = {'vectors': batch['positions'][batch['edge_index'][1]] - batch['positions'][batch['edge_index'][0]] + batch['shifts'],
            'node_attrs': batch['node_attrs'], 'edge_index': batch['edge_index'], 'batch': batch['batch'],
            'lammps_class': None, 'natoms': (len(atoms), 0)}
    e, node_e, pair_f = lmp_model.model(data)
    assert abs(float(e) - e0) / len(atoms) < 1e-5 and pair_f.shape == data['vectors'].shape
