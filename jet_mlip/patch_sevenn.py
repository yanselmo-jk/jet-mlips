"""Apply the jet patch to an installed SevenNet 0.13.0: `python -m jet_mlip.patch_sevenn [--check | --reverse]`.

Same as `python -m common.patch_tool sevenn` (patches/sevenn-jet*.patch, picked by dry run; see patches/BASE).
D3 in LAMMPS is jet-mlips' own `pair_style jet/d3`, so SevenNet's D3 code is left as shipped.
"""
import sys

from common.patch_tool import main

if __name__ == '__main__':
    sys.exit(main(['sevenn'] + sys.argv[1:]))
