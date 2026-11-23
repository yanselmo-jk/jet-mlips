"""LAMMPS ML-IAP wrapper for SevenNet models with jet support on the shared graph path (jet_mlip/mliap.py: ghost
fold, padding masked out of the conv CSR, CUDA-graph replay). The pair_e3gnn pair style stays the other SevenNet
route (sevenn deploy). Single MPI rank for the graph path; JET_SEVENN_CUDA_GRAPH=0, CPU or several ranks -> the
upstream eager path. A failed capture raises.

    python -m jet_mlip.sevenn_mliap <checkpoint path or pretrained name, e.g. 7net-0> <out.pt>
        [--modal NAME] [--radial_table N (default: omit = SevenNet default grid; 0 = exact)]
        [--exact_backward split|recompute] [--no-jet] [--no-graph]
"""
import argparse
import os

import torch

import sevenn._keys as KEY
from sevenn.mliap import SevenNetMLIAPWrapper

from jet_mlip import ops
from jet_mlip.mliap import GraphMLIAP


class SevenNetGraphMLIAPWrapper(GraphMLIAP, SevenNetMLIAPWrapper):
    """CUDA-graph ML-IAP wrapper. The parent's eager path is the fallback (CPU, several ranks, --no-graph)."""

    def _ensure_model_initialized(self):
        if self.model is not None:
            return
        super()._ensure_model_initialized()
        # the one-hot lookup table is a plain CPU attribute, copied to the GPU on every call: not capturable
        if torch.is_tensor(getattr(self.model, 'z_to_onehot_tensor', None)):
            self.model.z_to_onehot_tensor = self.model.z_to_onehot_tensor.to(self.device)
        # modal-wise rescale indexes shift[modal, atom] with a 0-dim modal tensor: eager Python turns it into .item(),
        # a sync capture refuses. The deployed modal is fixed, so keep only its row.
        from sevenn.nn.scale import ModalWiseRescale
        from sevenn.nn.sequential import _ModalInputPrepare
        mi = next((m.modal_idx for m in self.model.modules() if isinstance(m, _ModalInputPrepare)), None)
        for m in self.model.modules():
            if mi is not None and isinstance(m, ModalWiseRescale):
                for name in ('shift', 'scale'):
                    if getattr(m, 'use_modal_wise_' + name):
                        setattr(m, name, torch.nn.Parameter(getattr(m, name)[mi].clone(), requires_grad=False))
                        setattr(m, 'use_modal_wise_' + name, False)
        # read at run time (the wrapper is pickled at deploy)
        self.use_graph = os.environ.get('JET_SEVENN_CUDA_GRAPH', '1') != '0'
        self.r_cut, self.vec_dtype = float(self.cutoff), torch.float32
        self.index_dtype = torch.int32 if self.use_jet else torch.int64

    def _forward(self, vec):
        ncap = self.key[1]
        # SevenNet's convs take a CSR from the data dict (row 0 = dst) instead of building one
        rowptr, perm = ops.build_csr(self.s_ei[0], ncap)
        data = {
            KEY.JET_ROWPTR: rowptr, KEY.JET_PERM: perm, KEY.JET_SRC: self.s_ei[1].int(),
            KEY.EDGE_VEC: vec,
            KEY.EDGE_IDX: self.s_ei,
            KEY.ATOMIC_NUMBERS: self.s_types,
            KEY.MLIAP_NUM_LOCAL_GHOST: self.nlg,
            KEY.LAMMPS_DATA: self.lmp_stand_in,
        }
        return self.model(data)[KEY.ATOMIC_ENERGY]

    def compute_forces(self, lmp_data):
        self._ensure_model_initialized()
        if lmp_data.nlocal == 0:
            return
        if self.device.type == 'cpu' or not self.use_graph or lmp_data.npairs <= 1:
            return super().compute_forces(lmp_data)
        self.graph_forces(lmp_data)


def main(args=None):
    parser = argparse.ArgumentParser(description='Create a SevenNet LAMMPS ML-IAP file (jet + CUDA-graph wrapper).')
    parser.add_argument('model_path', help='checkpoint path or pretrained name (e.g. 7net-0)')
    parser.add_argument('output_path', help='path to write the ML-IAP file')
    parser.add_argument('--modal', default=None, help='modal name (e.g. oc20, mp, matpes, oom)')
    parser.add_argument('--radial_table', type=int, default=None,
                        help='jet radial grid points; 0 = exact MLP (default: SevenNet default grid)')
    parser.add_argument('--exact_backward', choices=['split', 'recompute'], default='split')
    parser.add_argument('--no-jet', action='store_true', help='e3nn convolutions instead of jet (A/B)')
    parser.add_argument('--no-graph', action='store_true', help='plain upstream wrapper instead of CUDA-graph replay')
    args = parser.parse_args(args=args)

    kw = dict(modal=args.modal, use_jet=not args.no_jet, exact_backward=args.exact_backward)
    if args.radial_table is not None:
        kw['radial_table'] = args.radial_table
    from jet_mlip import sevenn_mliap   # pickle the class by its module path, not __main__ (python -m)
    cls = SevenNetMLIAPWrapper if args.no_graph else sevenn_mliap.SevenNetGraphMLIAPWrapper
    torch.save(cls(model_path=args.model_path, **kw), args.output_path)


if __name__ == '__main__':
    main()
