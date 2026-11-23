"""Write a LAMMPS ML-IAP unified model file from an EquFlash checkpoint: the GGNN wrapper with the layers patched to
jet_equflash's fused convolution (`--no-jet`: the plain upstream wrapper, for A/B).

    python -m jet_equflash.deploy <ckpt.pt> <out.pt> [--radial_table N] [--no-jet]

    pair_style mliap unified out.pt 0
    pair_coeff * * Si O H        # LAMMPS type order; element_types is every chemical symbol, so elems = Z
"""
import argparse

from common.radial_table import DEFAULT_N


def main(argv=None):
    ap = argparse.ArgumentParser(description='Deploy an EquFlash ML-IAP model (jet convolution unless --no-jet).')
    ap.add_argument('ckpt')
    ap.add_argument('out')
    ap.add_argument('--radial_table', type=int, default=DEFAULT_N)
    ap.add_argument('--no-jet', action='store_true', help='save the plain upstream wrapper (A/B)')
    a = ap.parse_args(argv)

    import torch
    if a.no_jet:
        from .mliap import EquFlashMLIAPWrapper
        wrapper = EquFlashMLIAPWrapper(a.ckpt)
    else:
        from .mliap import EquFlashJetMLIAPWrapper
        wrapper = EquFlashJetMLIAPWrapper(a.ckpt, a.radial_table)
    torch.save(wrapper, a.out)


if __name__ == '__main__':
    main()
