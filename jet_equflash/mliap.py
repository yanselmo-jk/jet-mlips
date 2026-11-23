"""LAMMPS ML-IAP (unified) wrapper for EquFlash with jet convolutions: upstream's GGNN wrapper -- lazy model build,
per-layer LAMMPS ghost exchange, normalizers / element refs -- with `convert.patch_model` applied once the model
exists. GGNN is CC BY-NC-SA, so it is only loaded here, never copied.

Single rank (every ghost is a periodic image of a local atom, so its owner is local): the ghosts' owners are probed
once per call with the node indices through LAMMPS, and every layer's exchange then becomes a gather of owner rows on
the GPU instead of a Kokkos exchange staged through the host. Several MPI ranks keep the LAMMPS exchange
(a ghost's owner may live on another rank).

Single rank on CUDA runs the shared graph path (`jet_mlip/mliap.py`: ghost fold, padding masked out of the conv CSR,
CUDA-graph replay); `JET_EQUFLASH_CUDA_GRAPH=0` keeps the owner-gather eager path (A/B), and checkpoints with
normalizers or element refs stay eager (they act on the total energy outside the model).

    python -m jet_equflash.deploy <ckpt.pt> <out.pt> [--radial_table N] [--no-jet]
"""
import os

import torch
from torch_geometric.data import Batch

from common.radial_table import DEFAULT_N
from jet_mlip.mliap import GraphMLIAP, single_rank

try:
    import GGNN.model.SevenNet.sevennet  # noqa: F401
except ModuleNotFoundError as e:   # upstream imports two unused names from a subpackage this GGNN install lacks
    if e.name != 'GGNN.model.SevenNet':
        raise
    import sys
    import types
    for name in ('GGNN.model.SevenNet', 'GGNN.model.SevenNet.sevennet'):
        mod = types.ModuleType(name)
        mod.SevenNet = mod.convert_compiled_to_original_sevennet = None
        sys.modules[name] = mod
from GGNN.common.lammps_mliap.lmp_mliap_wrapper import GGNNLAMMPSMLIAPWrapper

from . import convert


class _OwnerExchange:
    """lmp_data stand-in: ghost rows <- owner rows, backward index_add to the owners; everything else reaches the
    real object (attribute reads through __getattr__, writes like `lmp_data.energy = ...` delegated)."""

    def __init__(self, lmp_data, owner):
        object.__setattr__(self, '_lmp', lmp_data)
        object.__setattr__(self, '_owner', owner)   # (ntotal,) int64; owner[i] = i for local rows

    def __getattr__(self, name):
        return getattr(self._lmp, name)

    def __setattr__(self, name, value):
        setattr(self._lmp, name, value)

    def forward_exchange(self, x, out, vec_len):
        out.copy_(x[self._owner])

    def reverse_exchange(self, g, gout, vec_len):
        gout.zero_()
        gout.index_add_(0, self._owner, g)


class EquFlashMLIAPWrapper(GGNNLAMMPSMLIAPWrapper):
    """The plain upstream wrapper (--no-jet A/B), pickled under this module so that loading it installs the stub above."""


class EquFlashJetMLIAPWrapper(GraphMLIAP, GGNNLAMMPSMLIAPWrapper):
    def __init__(self, ckpt_path: str, radial_table: int = DEFAULT_N):
        super().__init__(ckpt_path)
        self.radial_table = radial_table

    def _alloc(self, key):
        super()._alloc(key)
        nlocal, ncap = key[0], key[1]
        self.s_num_atoms = torch.tensor([nlocal, ncap - nlocal], dtype=torch.int64, device=self.device)
        self.s_natoms = torch.tensor(nlocal, dtype=torch.int64, device=self.device)

    def _forward(self, vec):
        nlocal = self.key[0]
        batch = Batch(edge_vec=vec, atomic_numbers=self.s_types[:nlocal], edge_index=self.s_ei,
                      num_atoms=self.s_num_atoms, natoms=self.s_natoms, lmp_data=self.lmp_stand_in)
        return self.model(batch)['atomic_energy']

    def compute_forces(self, lmp_data):
        if self.model is None:
            # the parent only reads __module__ (device) and, with nlocal = 0, returns right after loading the model
            probe = type('Probe', (), dict(nlocal=0, npairs=0, __module__=type(lmp_data).__module__))()
            super().compute_forces(probe)
            convert.patch_model(self.model, self.radial_table)   # lazy: the patched forward is a closure (unpicklable)
            for m in self.model.modules():   # node embedding's type map is a plain CPU tensor copied per call: not capturable
                if torch.is_tensor(getattr(m, 'type_map', None)):
                    m.type_map = m.type_map.to(self.device)
            self.r_cut = float(self.model.edge_embedding.cutoff_function.r_cut)
            self.vec_dtype, self.index_dtype = torch.float32, torch.int64
            # normalizers / element refs act on the total energy outside the model: those checkpoints stay eager
            # ponytail: V2 invalidates the CUDA graph capture (reported at a cuequivariance segmented_polynomial launch,
            # 0.6.0 and 0.9.1 alike; the invalidating op is not isolated): eager until it is
            self.use_graph = (self.device != 'cpu' and not self.normalizers and not self.elementrefs
                              and type(self.model).__name__ != 'EquFlashV2'
                              and os.environ.get('JET_EQUFLASH_CUDA_GRAPH', '1') != '0')
        if lmp_data.nlocal == 0 or lmp_data.npairs <= 1:
            return
        if self.use_graph:
            return self.graph_forces(lmp_data)
        ntotal = lmp_data.ntotal
        # ponytail: launcher env vars, not the LAMMPS communicator (>1 rank needs the per-layer LAMMPS exchange)
        if not single_rank() or ntotal <= lmp_data.nlocal:
            return super().compute_forces(lmp_data)
        dev = self.device
        if dev != 'cpu':
            torch.cuda.synchronize()   # Kokkos writes the LAMMPS arrays on another stream
        owner = torch.arange(ntotal, dtype=torch.int64, device=dev)
        probe = torch.arange(ntotal, dtype=torch.float64, device=dev).unsqueeze(1)
        out = torch.empty_like(probe)
        lmp_data.forward_exchange(probe, out, 1)   # node indices: each ghost row comes back as its owner
        if dev != 'cpu':
            torch.cuda.synchronize()
        owner[lmp_data.nlocal:] = out[lmp_data.nlocal:, 0].long()
        super().compute_forces(_OwnerExchange(lmp_data, owner))
