"""ASE: the MACECalculator of a jet-patched model replays its step from a CUDA graph (jet_mlip.ase_graph). The step is
the ML-IAP wrapper's (jet_mace.mliap `_forward`: the inner model on edge vectors, with the pair list's CSR and species
order), on the pair list of the calculator's own GPU neighbour list. `hook()`, called by convert.patch_model, wraps
MACECalculator.calculate once. From the second call in a row with the same atom count on, energy, node energies, forces
and stress come from the graph, with upstream's keys, dtypes and units. The energy is summed as upstream sums it: the
atomic E0 and the interaction energies each in fp32. Upstream's fp32 sum of E0 is biased with thousands of atoms (5632
atoms, MP-0 medium: -32165.02 eV, 1.40 eV = 2.5e-4 eV/atom below the fp64 sum of the same values), so an fp64 sum here
would shift the energy between the first call and the later ones. JET_ASE_GRAPH=0: off.
Upstream's calculate stays for everything else: the first call, CPU, JET_ASE_GPU_NL=0, use_compile, a committee, the
dipole / polar model types, atomic stresses, the charge / spin embedding models, fp64 models, a model in train mode or
patched without radial tables, and a step with an element the per-pair radial tables do not hold yet.
"""
import sys
from types import SimpleNamespace

import numpy as np
import torch
from jet_mlip.ase_graph import AseGraph


def _alloc(g):
    """The node buffers of the skip connections (jet_mace.skip.NODES), as the ML-IAP wrapper's."""
    i32 = dict(dtype=torch.int32, device=g.device)
    g.s_z, g.s_zperm = torch.zeros(g.key[1], **i32), torch.zeros(g.key[1], **i32)
    g.s_one = torch.ones(g.key[1], dtype=g.dtype, device=g.device)


def _prep(g):
    """Their species order: per pair list, not per step."""
    g.s_z.copy_(g.s_types)
    g.s_zperm.copy_(torch.argsort(g.s_z, stable=True))   # int32 keys: half the radix passes of int64


def _graph(calc, nl_device):
    """The calculator's graph step, made at its first call; None where the calculator stays eager."""
    model = calc.models[0]
    dtype = next(model.parameters()).dtype
    # ponytail: one model; a committee would take one graph per model
    if not (calc.num_models == 1 and calc.model_type == 'MACE' and not calc.compute_atomic_stresses
            and getattr(model, 'macetp_no_sh', False) and not model.training and dtype == torch.float32
            and not hasattr(model, 'joint_embedding') and nl_device(calc) is not None):
        return None
    g = getattr(calc, 'jet_graph', None)
    if g is None or g.model.model is not model:
        from .mliap import LAMMPS_MLIAP_MACE_Graph   # here, not at import: jet_mace.skip needs triton (CUDA only)
        g = calc.jet_graph = AseGraph(LAMMPS_MLIAP_MACE_Graph._forward, calc.device, calc.r_max, vec_dtype=dtype,
                                      alloc=_alloc, prep=_prep)
        heads = calc.available_heads
        head = torch.tensor([heads.index(calc.head) if calc.head in heads else len(heads) - 1], device=calc.device)
        # what the wrapper's _forward reads besides the graph's buffers (no embedding model here: no charge, no spin)
        g.model = SimpleNamespace(model=model, head=head, total_charge=None, total_spin=None)
        g.dtype, g.num_species, g.jet_z = dtype, len(calc.z_table), None
    return g


def hook():
    """Wrap MACECalculator.calculate once. No-op when mace.calculators is not loaded (LAMMPS) or the installed mace
    patch has no GPU neighbour list."""
    mod = sys.modules.get('mace.calculators.mace')
    if mod is None or getattr(mod.MACECalculator.calculate, 'jet', False):
        return
    try:
        from mace.modules.macetp_hook import macetp_nl_device
    except ImportError:
        return
    from ase.calculators.calculator import Calculator, all_changes
    from ase.stress import full_3x3_to_voigt_6_stress
    from mace.data import get_neighborhood
    from mace.tools.scatter import scatter_sum
    from mace.tools.utils import atomic_numbers_to_indices
    orig = mod.MACECalculator.calculate

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        g = _graph(self, macetp_nl_device)
        # the pair count is known only after the list: 2 here, checked below
        if g is None or not g.due(len(atoms), 2):
            return orig(self, atoms, properties, system_changes)
        dev, model = self.device, self.models[0]
        if g.jet_z is None or not np.array_equal(g.jet_z[0], atoms.numbers):
            present = set(atoms.numbers.tolist())
            if any(b.macetp.species is not None and not present <= b.macetp.species for b in model.interactions):
                self.jet_graph = None   # the eager step extends the per-pair tables; a new graph reads them
                return orig(self, atoms, properties, system_changes)
            types = torch.as_tensor(atomic_numbers_to_indices(atoms.numbers, self.z_table), device=dev)
            one_hot = torch.nn.functional.one_hot(types, g.num_species).to(g.dtype)
            e0 = model.atomic_energies_fn(one_hot).index_select(1, g.model.head)[:, 0].detach()   # upstream's node_e0
            g.jet_z = atoms.numbers.copy(), types, e0, scatter_sum(e0, torch.zeros_like(types), dim=0, dim_size=1)
        # upstream's list (AtomicData.from_config): the cell completed along the non-periodic directions
        ei, _, unit_shifts, cell = get_neighborhood(atoms.positions, self.r_max, tuple(atoms.pbc), np.array(atoms.cell),
                                                    device=dev)
        if ei.shape[1] <= 1:
            return orig(self, atoms, properties, system_changes)
        Calculator.calculate(self, atoms)
        pos = torch.as_tensor(atoms.positions, device=dev)
        p = pos.to(g.dtype)
        # upstream's edge vectors at zero strain, in the model's dtype (get_symmetric_displacement)
        vec = p[ei[1]] - p[ei[0]] + unit_shifts.to(g.dtype) @ torch.as_tensor(cell, dtype=g.dtype, device=dev)
        _, types, e0, e0_sum = g.jet_z
        e, forces, w = g.compute(types, ei[0], ei[1], vec, pos)
        # e = e0 + interaction energy, each fp32, added in fp64 (ScaleShiftMACE): the difference is the latter again
        inter = scatter_sum((e - e0).to(g.dtype), torch.zeros_like(types), dim=-1, dim_size=1)
        eu, lu = self.energy_units_to_eV, self.length_units_to_A
        energies = e.cpu().numpy() * eu
        stress = ((w + w.t()) / (2 * abs(np.linalg.det(cell)))).to(g.dtype).cpu().numpy() * (eu / lu ** 3)
        energy = (e0_sum + inter).item() * eu
        self.results = {'energy': energy, 'free_energy': energy, 'energies': energies,
                        'node_energy': energies - e0.cpu().numpy(),
                        'forces': forces.to(g.dtype).cpu().numpy() * (eu / lu),
                        'stress': full_3x3_to_voigt_6_stress(stress)}
    calculate.jet = True
    mod.MACECalculator.calculate = calculate
