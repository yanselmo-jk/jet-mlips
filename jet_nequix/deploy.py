"""Deploy a Nequix torch model for LAMMPS ML-IAP, jet-patched unless `--no-jet`.

    python -m jet_nequix.deploy <model.pt> <out.pt> [--radial_table N] [--no-jet]

LAMMPS:

    pair_style jet/mliap out.pt      # = mliap unified out.pt 0, KOKKOS only
    pair_coeff * * <symbols in LAMMPS type order>
"""
import argparse
import pathlib

import torch

from common.radial_table import DEFAULT_N

from .mliap import NequixMLIAPWrapper


def main(args=None):
    p = argparse.ArgumentParser(description='Create a Nequix LAMMPS ML-IAP file from a saved model.')
    p.add_argument('model_path', type=pathlib.Path, help='Nequix .pt (config line + state dict)')
    p.add_argument('output_path', type=pathlib.Path, help='path to write the ML-IAP file')
    p.add_argument('--radial_table', type=int, default=DEFAULT_N)
    p.add_argument('--no-jet', action='store_true', help='upstream torch model, no jet_mlip patch (A/B)')
    a = p.parse_args(args=args)
    torch.save(NequixMLIAPWrapper(a.model_path, radial_table=a.radial_table, jet=not a.no_jet), a.output_path)


if __name__ == '__main__':
    main()
