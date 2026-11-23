"""LAMMPS ML-IAP wrapper for jet_nequip-patched NequIP models on the shared graph path (jet_mlip/mliap.py: ghost
fold, padding masked out of the conv CSR, CUDA-graph replay). Single MPI rank; JET_NEQUIP_CUDA_GRAPH=0, CPU or
per-edge-type pruning -> the upstream eager/compiled path. JET_NEQUIP_COMPILE=0 -> no torch.compile. A failed
capture raises.
"""
import os

import torch

os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', os.path.expanduser('~/.cache/torchinductor'))   # /tmp is a small tmpfs here
from nequip.data import AtomicDataDict
from nequip.integrations.lammps_mliap.lmp_mliap_wrapper import NequIPLAMMPSMLIAPWrapper
from nequip.nn import graph_model

from common.radial_table import DEFAULT_N
from jet_mlip.mliap import GraphMLIAP

from .convert import patch_model


class NequIPJetMLIAPWrapper(NequIPLAMMPSMLIAPWrapper):
    """Jet convolution, upstream compile (dynamic=True) and ghost exchange: deploy --no-graph."""

    def __init__(self, *args, radial_table: int = DEFAULT_N, **kwargs):
        self.radial_table = radial_table   # the parent's __init__ already calls _load_model_from_bytes
        super().__init__(*args, **kwargs)

    def _load_model_from_bytes(self):
        return patch_model(super()._load_model_from_bytes(), self.radial_table)


class NequIPGraphMLIAPWrapper(GraphMLIAP, NequIPLAMMPSMLIAPWrapper):
    """CUDA-graph ML-IAP wrapper. jet=False runs official modifiers (e.g. enable_OpenEquivariance) unpatched, for A/B."""

    def __init__(self, *args, jet=True, radial_table=DEFAULT_N, **kwargs):
        self.jet = jet
        self.radial_table = radial_table   # the parent's __init__ already calls _load_model_from_bytes
        super().__init__(*args, **kwargs)

    def _load_model_from_bytes(self):
        model = super()._load_model_from_bytes()
        self.r_max = self.r_cut = float(model.metadata[graph_model.R_MAX_KEY])
        return patch_model(model, self.radial_table) if self.jet else model

    def _initialize_model(self, lmp_data) -> None:
        compile = self.compile
        self.compile = False   # the parent would compile with dynamic=True
        super()._initialize_model(lmp_data)
        self.compile = compile
        # read at run time (the wrapper is pickled at deploy; v0.2.1 pickles lack the shared-path attributes)
        self.use_graph = os.environ.get('JET_NEQUIP_CUDA_GRAPH', '1') != '0'
        self.r_cut, self.vec_dtype, self.index_dtype = float(self.r_max), torch.float64, torch.int64
        if self.compile and os.environ.get('JET_NEQUIP_COMPILE', '1') != '0':   # static shapes: one compile per key
            torch._inductor.config.freezing = 1
            torch._dynamo.config.recompile_limit = 64
            self.model = torch.compile(self.model, dynamic=False)

    def _alloc(self, key):
        super()._alloc(key)
        self.s_nlg = torch.tensor(self.nlg, dtype=torch.int64, device=self.device)

    def _forward(self, vec):
        data = {
            AtomicDataDict.EDGE_VECTORS_KEY: vec,
            AtomicDataDict.EDGE_INDEX_KEY: self.s_ei,
            AtomicDataDict.ATOM_TYPE_KEY: self.s_types,
            AtomicDataDict.LMP_MLIAP_DATA_KEY: self.lmp_stand_in,
            AtomicDataDict.NUM_LOCAL_GHOST_NODES_KEY: self.s_nlg,
        }
        return self.model(data)[AtomicDataDict.PER_ATOM_ENERGY_KEY]

    def compute_forces(self, lmp_data):
        if self.model is None:
            self._initialize_model(lmp_data)
        # per-edge-type pruning reshapes the edge list per type (no static graph): jet's per-cutoff tables are 0 beyond
        # each pair's cutoff, so the patched model needs no pruning; official modifiers keep the eager pruned path
        if self.device == 'cpu' or not self.use_graph or (self.nl_pruner is not None and not self.jet):
            return super().compute_forces(lmp_data)
        if lmp_data.nlocal == 0 or lmp_data.npairs <= 1:
            return
        self.graph_forces(lmp_data)
