"""Deploy a LAMMPS ML-IAP model for any supported family in this repo through one command.

    python -m jet_mlip.deploy <family> <model> [<output>] [family options]

Without <output> the default is jet_deployed_<model stem>.pt (.nequip.lmp.pt for nequip).

families:

    sevenn     python -m jet_mlip.deploy sevenn 7net-0
    mace       python -m jet_mlip.deploy mace medium
    nequip     python -m jet_mlip.deploy nequip nequip.net:mir-group/NequIP-OAM-L:0.1
    nequix     python -m jet_mlip.deploy nequix nequix-mp-1.pt
    equflash   python -m jet_mlip.deploy equflash equflash_oam_30M.pt

`python -m jet_mlip.deploy <family> --help` shows the family options. The per-family CLIs still work
(e.g. `python -m jet_mace.deploy medium out.pt`); this only forwards to them.
"""
import importlib
import sys

FAMILIES = {
    'sevenn': 'jet_mlip.sevenn_mliap',
    'mace': 'jet_mace.deploy',
    'nequip': 'jet_nequip.deploy',
    'nequix': 'jet_nequix.deploy',
    'equflash': 'jet_equflash.deploy',
}


def default_output(family, model):
    """jet_deployed_<last path component, ':' -> '-', model extension stripped><.nequip.lmp.pt | .pt>."""
    stem = model.rsplit('/', 1)[-1].replace(':', '-')
    for ext in ('.pt', '.pth', '.model', '.ckpt', '.nequip.zip', '.zip'):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
            break
    return f'jet_deployed_{stem}' + ('.nequip.lmp.pt' if family == 'nequip' else '.pt')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0 if argv else 2

    family, rest = argv[0], argv[1:]
    if family not in FAMILIES:
        print(__doc__, file=sys.stderr)
        print(f'unknown family {family!r}; choose from {", ".join(FAMILIES)}', file=sys.stderr)
        return 2
    if rest and not rest[0].startswith('-') and (len(rest) < 2 or rest[1].startswith('-')):
        rest = [rest[0], default_output(family, rest[0]), *rest[1:]]

    module = importlib.import_module(FAMILIES[family])
    result = module.main(rest)
    if len(rest) > 1:
        print('output:', rest[1])
    return 0 if result is None else result


if __name__ == '__main__':
    sys.exit(main())
