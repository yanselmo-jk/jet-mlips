"""Apply a shipped patch to an installed upstream package (GNU `patch -p1` on its site-packages root).

    python -m common.patch_tool mace             # mace-torch: picks patches/mace-torch-<ver>.patch by dry run
    python -m common.patch_tool mace --check
    python -m common.patch_tool mace --reverse
    python -m common.patch_tool sevenn           # = python -m jet_mlip.patch_sevenn

One patch file per supported upstream version (see patches/BASE); the first that dry-runs cleanly is used.
Options: --root DIR (directory containing the package; default: found by import), --check (dry run),
--reverse (undo, using the recorded file), --patch FILE (explicit file, no auto-pick). Refuses to apply twice.
"""
import argparse
import glob
import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

PATCHES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'patches')


@dataclass
class Pkg:
    module: str   # import name; the patch root is the directory containing its top-level package
    marker: str   # file (relative to root) that only exists after patching
    stem: str     # patches/<stem>-<version>.patch


PKGS = {
    'mace': Pkg('mace', 'mace/modules/macetp_hook.py', 'mace-torch'),
    'sevenn': Pkg('sevenn', 'sevenn/nn/jet_helper.py', 'sevenn-jet'),   # sevenn-jet.patch (git 8d9905c), -v0.13.0
}


def dry_run(root, p, reverse=False):
    cmd = ['patch', '-p1', '-N', '--dry-run', '-d', root, '-i', p] + (['-R'] if reverse else [])
    return subprocess.run(cmd, capture_output=True, text=True)


def candidates(stem):
    return sorted(glob.glob(os.path.join(PATCHES, f'{stem}*.patch')), reverse=True)


def pick(root, stem):
    """First shipped variant that dry-runs cleanly on `root`; None if none does."""
    for p in candidates(stem):
        if dry_run(root, p).returncode == 0:
            return p
    return None


def refusing_hunks(root, stem):
    for p in candidates(stem):
        r = dry_run(root, p)
        keep, cur = [], None
        for l in (r.stdout + r.stderr).splitlines():
            if l.startswith('checking file'):
                cur = l
                continue
            if 'FAILED' in l or 'already exists' in l or 'ignored' in l or 'malformed' in l:
                if cur:
                    keep.append(cur)
                    cur = None
                keep.append('  ' + l)
        print(f'--- {os.path.basename(p)}: rc={r.returncode}\n' + ('\n'.join(keep) if keep else '  (no refusing hunk)'))


def find_root(pkg, explicit=None):
    if explicit:
        return os.path.abspath(explicit)
    path = os.path.abspath(importlib.import_module(pkg.module).__file__)
    for _ in range(pkg.module.count('.') + 2):   # <root>/a/b/__init__.py -> <root>
        path = os.path.dirname(path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pkg', choices=sorted(PKGS))
    ap.add_argument('--root', help='dir containing the package (default: from import)')
    ap.add_argument('--check', action='store_true', help='dry run')
    ap.add_argument('--reverse', action='store_true', help='remove the patch')
    ap.add_argument('--patch', help='apply exactly this patch file (default: auto-pick from patches/)')
    a = ap.parse_args(argv)
    pkg = PKGS[a.pkg]
    root = find_root(pkg, a.root)
    marker = os.path.join(root, pkg.marker)
    record = os.path.join(root, pkg.module.split('.')[0], '.mlip_accel_patch')   # name of the applied patch file
    applied = record + '.diff'   # its content at apply time: --reverse uses this copy, so regenerating patches/ is safe
    if not a.reverse and os.path.exists(marker):
        # up to date only if a shipped patch is fully in the files (it reverses cleanly): an older patch, or files put
        # back by a reinstall of the package while the marker and record stayed, both fail here
        current = next((c for c in candidates(pkg.stem) if dry_run(root, c, reverse=True).returncode == 0), None)
        if current:
            print(f'already patched by {os.path.basename(current)}: {marker} exists; use --reverse to remove it')
            return 0
        which = open(record).readline().strip() if os.path.exists(record) else 'an unrecorded patch'
        print(f'{marker} exists, but no shipped patch is fully applied (record: {which}; an older patch, or files '
              'reinstalled under it): run with --reverse first (add --patch <old file> if it does not reverse), '
              'or reinstall the package cleanly and remove the marker, then patch again')
        return 1
    if a.patch:
        p = os.path.abspath(a.patch)
    elif a.reverse and os.path.exists(applied):
        p = applied
    elif a.reverse:   # patched before the record existed (old jet_mlip.patch_sevenn): the variant that reverses cleanly
        p = next((c for c in candidates(pkg.stem) if dry_run(root, c, reverse=True).returncode == 0), None)
    else:
        p = pick(root, pkg.stem)
    if p is None:
        print(f'no shipped patch applies cleanly to {root} (supported upstream versions: patches/BASE). Refusing hunks:')
        refusing_hunks(root, pkg.stem)
        return 1
    print('using', os.path.basename(p))
    cmd = ['patch', '-p1', '-N', '--no-backup-if-mismatch', '-d', root, '-i', p]
    cmd += ['--dry-run'] if a.check else []
    cmd += ['-R'] if a.reverse else []
    print(' '.join(cmd))
    if subprocess.run(cmd).returncode != 0:
        print('patch failed (supported upstream versions: patches/BASE)')
        return 1
    if a.check:
        print('patch reverses cleanly' if a.reverse else 'patch applies cleanly')
    elif a.reverse:
        for f in (record, applied):
            os.path.exists(f) and os.remove(f)
    else:
        open(record, 'w').write(os.path.basename(p) + '\n')
        shutil.copy(p, applied)
    return 0


if __name__ == '__main__':
    sys.exit(main())
