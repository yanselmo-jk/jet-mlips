"""Deploy a MACE model for LAMMPS ML-IAP (`pair_style jet/mliap <out>`) with the jet_mace fused convolution.

    python -m jet_mace.deploy medium out.pt [--radial_table 4096] [--no-cueq] [--head HEAD]

<model> is a mace_mp foundation name (medium, small, medium-0b3, mace-matpes-pbe-0, ...), off-small / off-medium /
off-large (MACE-OFF23), or a .model file. The output is the pickled
LAMMPS_MLIAP_MACE of upstream `mace_create_lammps_model --format mliap`, around the patched fp32 CUDA model
(cuEq for the non-convolution parts unless --no-cueq). LAMMPS' python must import `jet_mace` and `jet_mlip`.
"""
import argparse
import os
import sys

import torch

from . import configs, convert


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model')
    ap.add_argument('output')
    ap.add_argument('--radial_table', type=int, default=convert.DEFAULT_N)
    ap.add_argument('--no-cueq', action='store_true')
    ap.add_argument('--no-graph', action='store_true', help='upstream eager ML-IAP wrapper instead of the CUDA-graph one')
    ap.add_argument('--head', default=None)
    a = ap.parse_args(argv)
    from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE
    from .mliap import LAMMPS_MLIAP_MACE_Graph
    if os.path.isfile(a.model):
        model = torch.load(a.model, map_location='cuda', weights_only=False)
    else:
        model = configs.load(a.model, 'cuda')
    model = model.float().to('cuda').eval()
    model = convert.patch_model(model, radial_table=a.radial_table, cueq=not a.no_cueq)
    model.lammps_mliap = True
    head = a.head if a.head is not None else model.heads[-1]
    cls = LAMMPS_MLIAP_MACE if a.no_graph else LAMMPS_MLIAP_MACE_Graph
    torch.save(cls(model, head=head), a.output)
    print('wrote', a.output, 'head', head, 'elements', [int(z) for z in model.atomic_numbers])
    return 0


if __name__ == '__main__':
    sys.exit(main())
