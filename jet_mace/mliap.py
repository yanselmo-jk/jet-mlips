"""LAMMPS ML-IAP wrapper for jet_mace-patched MACE models on the shared graph path (jet_mlip/mliap.py: ghost fold,
padding masked out of the conv CSR, CUDA-graph replay). Single MPI rank; JET_MACE_CUDA_GRAPH=0 or CPU -> the upstream
eager path. JET_MACE_COMPILE=0 -> no torch.compile. A failed capture raises.
"""
import os

import torch
import torch.nn.functional as F

os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', os.path.expanduser('~/.cache/torchinductor'))   # /tmp is a small tmpfs here
from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE

from jet_mlip.mliap import GraphMLIAP


class LAMMPS_MLIAP_MACE_Graph(GraphMLIAP, LAMMPS_MLIAP_MACE):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        if os.environ.get('JET_MACE_COMPILE', '1') != '0':   # inductor fuses the elementwise / layout glue between the cuEq ops
            torch._dynamo.config.recompile_limit = 64      # (static shapes: one compile per capture key, then replayed)
            self.model.model = torch.compile(self.model.model, dynamic=False)
        self.r_max = float(model.r_max)

    def _forward(self, vec):
        nlocal = self.key[0]
        batch = {
            'vectors': vec,
            'node_attrs': F.one_hot(self.s_types, self.num_species).to(self.dtype),
            'edge_index': torch.stack([self.s_ei[1], self.s_ei[0]]),
            'batch': torch.zeros(nlocal, dtype=torch.int64, device=self.device),
            'lammps_class': self.lmp_stand_in,
            'natoms': self.nlg,
            'head': self.model.head,
            'total_charge': self.model.total_charge,
            'total_spin': self.model.total_spin,
        }
        # call the inner model: the wrapper's forward adds compute_edge_forces=True, whose grad() consumes the graph
        # the shared step still needs for dE/d vec
        return self.model.model(batch, training=False, compute_force=False, compute_virials=False,
                                compute_stress=False, compute_displacement=False,
                                compute_edge_forces=False, lammps_mliap=True)['node_energy']

    def graph_forces(self, lmp_data):
        try:
            return super().graph_forces(lmp_data)
        except RuntimeError as e:
            # torch >= 2.8 cannot compile e3nn 0.4.4 code (dynamo NotImplementedError in e3nn/o3/_irreps.py): drop
            # torch.compile once and capture the plain model
            orig = getattr(self.model.model, '_orig_mod', None)
            if orig is None or not type(e.__cause__).__module__.startswith('torch._dynamo'):
                raise
            print(f'{type(self).__name__}: torch.compile failed ({type(e.__cause__).__name__}), running uncompiled')
            self.model.model, self.graph = orig, None
            return super().graph_forces(lmp_data)

    def compute_forces(self, data):
        if not self.initialized:
            self._initialize_device(data)
            # at run time (the wrapper is pickled at deploy; v0.2.1 pickles lack the shared-path attributes)
            self.r_cut, self.vec_dtype, self.index_dtype = self.r_max, self.dtype, torch.int64
            self.use_graph = os.environ.get('JET_MACE_CUDA_GRAPH', '1') != '0'
        if self.device.type != 'cuda' or not self.use_graph:
            return super().compute_forces(data)
        self.step += 1
        if data.nlocal == 0 or data.npairs <= 1:
            return
        self.graph_forces(data)
