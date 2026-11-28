"""LAMMPS ML-IAP wrapper for jet_mace-patched MACE models on the shared graph path (jet_mlip/mliap.py: ghost fold,
padding masked out of the conv CSR, CUDA-graph replay). JET_MACE_CUDA_GRAPH=0 or CPU -> the upstream eager path.
JET_MACE_COMPILE=0 -> no torch.compile; both are read when LAMMPS runs. A failed capture raises.
One rank: the fold leaves no ghost row, so the model runs without a LAMMPS class on exactly nlocal rows: no per-layer
exchange copy, and no row truncation, whose backward zero-fills and copies every message tensor (0.4 of 10.6 ms per
step, MP-0 medium, 5632 atoms).
"""
import os

import torch
import torch.nn.functional as F

os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', os.path.expanduser('~/.cache/torchinductor'))   # /tmp is a small tmpfs here
from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE

from jet_mlip import ops
from jet_mlip.mliap import GraphMLIAP

from . import convert, skip


class LAMMPS_MLIAP_MACE_Graph(GraphMLIAP, LAMMPS_MLIAP_MACE):
    pad_nodes = False

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.r_max = float(model.r_max)

    def _alloc(self, key):
        super()._alloc(key)
        i32 = dict(dtype=torch.int32, device=self.device)
        self.s_rowptr, self.s_perm = torch.zeros(key[1] + 1, **i32), torch.zeros(key[2], **i32)
        self.s_z, self.s_zperm = torch.zeros(key[1], **i32), torch.zeros(key[1], **i32)
        self.s_one = torch.ones(key[1], dtype=self.dtype, device=self.device)

    def _prep(self):
        """Per pair list, not per step and layer: the conv CSR."""
        super()._prep()
        ops.EDGE_MASK = self.s_mask
        try:
            rowptr, perm = ops.build_csr(self.s_ei[0], self.key[1])
        finally:
            ops.EDGE_MASK = None
        self.s_rowptr.copy_(rowptr)
        self.s_perm.copy_(perm)

    def _types(self):
        """The species order of the skip connections follows the elements: every step, with them."""
        super()._types()
        self.s_z.copy_(self.s_types)
        self.s_zperm.copy_(torch.argsort(self.s_z, stable=True))   # int32 keys: half the radix passes of int64

    def _forward(self, vec):
        nlocal = self.key[0]
        batch = {
            'vectors': vec,
            'node_attrs': F.one_hot(self.s_types, self.num_species).to(self.dtype),
            'edge_index': torch.stack([self.s_ei[1], self.s_ei[0]]),
            'batch': torch.zeros(nlocal, dtype=torch.int64, device=self.device),
            'lammps_class': self.lmp_stand_in if self.key[4] else None,   # several ranks: remote rows, exchanged per layer
            'natoms': self.nlg,
            'head': self.model.head,
            'total_charge': self.model.total_charge,
            'total_spin': self.model.total_spin,
        }
        convert.CSR, convert.SPECIES = (self.s_rowptr, self.s_perm), self.s_types
        skip.NODES = (self.s_z, self.s_one, self.s_zperm)
        try:
            # call the inner model: the wrapper's forward adds compute_edge_forces=True, whose grad() consumes the graph
            # the shared step still needs for dE/d vec
            return self.model.model(batch, training=False, compute_force=False, compute_virials=False,
                                    compute_stress=False, compute_displacement=False,
                                    compute_edge_forces=False, lammps_mliap=True)['node_energy']
        finally:
            convert.CSR = convert.SPECIES = skip.NODES = None

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
            inner = getattr(self.model.model, '_orig_mod', self.model.model)   # pickles up to v0.5.0: compiled at deploy
            if os.environ.get('JET_MACE_COMPILE', '1') != '0':   # inductor fuses the elementwise / layout glue between the cuEq ops
                torch._dynamo.config.recompile_limit = 64      # (static shapes: one compile per capture key, then replayed)
                inner = torch.compile(inner, dynamic=False)
            self.model.model = inner
        if self.device.type != 'cuda' or not self.use_graph:
            return super().compute_forces(data)
        self.step += 1
        if data.nlocal == 0 or data.npairs <= 1:
            return
        self.graph_forces(data)
