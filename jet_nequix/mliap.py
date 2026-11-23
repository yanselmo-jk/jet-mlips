"""LAMMPS ML-IAP (unified) wrapper for Nequix torch models, jet-patched by default.

CUDA runs the shared graph path (jet_mlip/mliap.py: ghost fold, padding masked out of the conv CSR, CUDA-graph replay;
several ranks: every layer's remote rows refreshed by the graph's NCCL exchange). CPU or JET_NEQUIX_CUDA_GRAPH=0 ->
eager on the local graph, one rank only. Sign: rij is Nequix's receiver -> sender displacement, so dE/drij is LAMMPS'
pair force without a flip.
"""
import json
import os
import tempfile

import torch
from ase.data import chemical_symbols
from lammps.mliap.mliap_unified_abc import MLIAPUnified

from common.radial_table import DEFAULT_N
from jet_mlip.mliap import GraphMLIAP, LammpsStandIn, single_rank

from .convert import load_model, patch_model


class _StandInExchange(torch.autograd.Function):
    """A layer's remote rows through the graph's stand-in (NCCL, no host sync: capturable); backward: the gradients
    go back to their owners."""

    @staticmethod
    def forward(ctx, x, stand_in):
        ctx.stand_in = stand_in
        out = torch.empty_like(x)
        stand_in.forward_exchange(x, out, x.shape[1])
        return out

    @staticmethod
    def backward(ctx, g):
        gout = torch.empty_like(g)
        ctx.stand_in.reverse_exchange(g, gout, g.shape[1])
        return gout, None


class NequixMLIAPWrapper(GraphMLIAP, MLIAPUnified):
    """MLIAPUnified over a Nequix torch model; jet=True swaps in jet_nequix's fused convolution."""

    def __init__(self, model_path, radial_table: int = DEFAULT_N, jet: bool = True):
        with open(model_path, 'rb') as f:
            self.model_bytes = f.read()   # the deployed .pt must be self-contained
        self.jet, self.radial_table = jet, radial_table
        config = json.loads(self.model_bytes.split(b'\n', 1)[0])
        super().__init__(
            element_types=[chemical_symbols[z] for z in config['atomic_numbers']],
            ndescriptors=1, nparams=1, rcutfac=0.5 * float(config['cutoff']))
        # loaded lazily: patch_model attaches a closure to model.node_energies, which torch.save cannot pickle
        self.model, self.device = None, None

    def _initialize_model(self, lmp_data):
        self.device = 'cuda' if 'kokkos' in type(lmp_data).__module__.lower() else 'cpu'
        with tempfile.TemporaryDirectory() as tmp:   # load_model takes a path
            path = os.path.join(tmp, 'model.pt')
            with open(path, 'wb') as f:
                f.write(self.model_bytes)
            self.model = load_model(path, False)[0]
        self.model = self.model.to(self.device).eval()
        if self.jet:
            patch_model(self.model, self.radial_table)
        self.r_max = self.r_cut = float(self.model.cutoff)
        self.vec_dtype, self.index_dtype = torch.float32, torch.int64
        self.use_graph = self.device != 'cpu' and os.environ.get('JET_NEQUIX_CUDA_GRAPH', '1') != '0'
        if not single_rank():
            if not self.use_graph:
                raise RuntimeError('jet_nequix: several MPI ranks need the CUDA graph path (CUDA, JET_NEQUIX_CUDA_GRAPH=1)')
            for layer in self.model.layers:   # refresh every convolution input's remote rows, once per layer
                layer.register_forward_pre_hook(self._refresh_ghost_rows)

    def _refresh_ghost_rows(self, module, args):
        stand_in = getattr(self, 'lmp_stand_in', None)
        if not isinstance(stand_in, LammpsStandIn) or stand_in.xchg is None:
            return None
        return (_StandInExchange.apply(args[0], stand_in),) + args[1:]

    def compute_forces(self, lmp_data):
        if self.model is None:
            self._initialize_model(lmp_data)
        if lmp_data.nlocal == 0 or lmp_data.npairs <= 1:
            return
        if self.use_graph:
            self.graph_forces(lmp_data)
        else:
            self._local_forces(lmp_data)

    def _store(self, lmp_data, e, f):
        lmp_data.energy = e.sum().item()
        torch.as_tensor(lmp_data.eatoms).copy_(e)
        lmp_data.update_pair_forces_gpu(f.double())

    def _local_forces(self, lmp_data):
        dev = self.device
        if dev != 'cpu':
            torch.cuda.synchronize()   # Kokkos writes the LAMMPS arrays on another stream
        owner = torch.arange(lmp_data.ntotal, dtype=torch.int64, device=dev)
        if lmp_data.ntotal > lmp_data.nlocal:   # probe the exchange with the node indices: ghost rows come back as owners
            probe = torch.arange(lmp_data.ntotal, dtype=torch.float64, device=dev).unsqueeze(1)
            out = torch.empty_like(probe)
            if dev != 'cpu':
                torch.cuda.synchronize()
            lmp_data.forward_exchange(probe, out, 1)
            if dev != 'cpu':
                torch.cuda.synchronize()
            owner[lmp_data.nlocal:] = out[lmp_data.nlocal:, 0].long()
        pair_i = torch.as_tensor(lmp_data.pair_i, dtype=torch.int64, device=dev)
        pair_j = torch.as_tensor(lmp_data.pair_j, dtype=torch.int64, device=dev)
        vec = torch.as_tensor(lmp_data.rij, dtype=torch.float32, device=dev).requires_grad_(True)
        species = torch.as_tensor(lmp_data.elems, dtype=torch.int64, device=dev)[:lmp_data.nlocal]
        e = self.model.node_energies(vec, species, owner[pair_j], pair_i).view(-1)
        f = torch.autograd.grad(e.sum(), vec)[0]   # dE/drij: LAMMPS' sign convention, no flip
        self._store(lmp_data, e.detach(), f)

    def _forward(self, vec):
        # all ncap rows: padding edges point at the last (padding) row, and the model indexes it
        return self.model.node_energies(vec, self.s_types, self.s_ei[1], self.s_ei[0])

    def compute_descriptors(self, lmp_data):
        pass

    def compute_gradients(self, lmp_data):
        pass
