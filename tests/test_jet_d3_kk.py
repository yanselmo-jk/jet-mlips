"""pair_style jet/d3/kk (what jet/d3 becomes under -sf kk; named in full here) against the plain jet/d3 of the same binary: energy, pressure
and forces of a rattled 216-atom Si cell at run 0. Needs a GPU and `lmp` built by lammps/build_lammps.sh (or LMP=<path>);
skips otherwise."""
import os
import shutil
import subprocess

import numpy as np
import pytest

INPUT = """units metal
atom_style atomic
boundary p p p
newton on
lattice diamond 5.43
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 28.0855
displace_atoms all random 0.1 0.1 0.1 4711
pair_style {style} 9000 1600 damp_bj pbe
pair_coeff * * Si
thermo_style custom step pe press pxy
thermo_modify format float %.12g
run 0
write_dump all custom {out} id fx fy fz modify sort id format float %.12g
"""


def run(lmp, tmp_path, name, style, flags):
    out = tmp_path / f'{name}.dump'
    (tmp_path / f'{name}.in').write_text(INPUT.format(out=out, style=style))
    r = subprocess.run([lmp, *flags, '-in', f'{name}.in', '-log', f'{name}.log'], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    lines = r.stdout.splitlines()
    thermo = np.array(lines[next(i for i, s in enumerate(lines) if s.split()[:2] == ['Step', 'PotEng']) + 1].split()[1:], float)
    return thermo, np.loadtxt(out, skiprows=9)[:, 1:]


def test_kk_matches_plain(tmp_path):
    lmp = os.environ.get('LMP') or shutil.which('lmp')
    if not lmp or 'jet/d3/kk' not in subprocess.run([lmp, '-h'], capture_output=True, text=True).stdout:
        pytest.skip('needs lmp with jet/d3/kk (lammps/build_lammps.sh in this env, or LMP=)')
    t0, f0 = run(lmp, tmp_path, 'plain', 'jet/d3', [])
    t1, f1 = run(lmp, tmp_path, 'kk', 'jet/d3/kk', ['-k', 'on', 'g', '1', '-sf', 'kk', '-pk', 'kokkos', 'newton', 'on', 'neigh', 'half'])
    assert t0[0] < 0 and np.abs(f0).max() > 1e-3
    assert abs(t1[0] - t0[0]) < 1e-9 * len(f0) and np.abs(t1[1:] - t0[1:]).max() < 1e-6 * np.abs(t0[1:]).max()
    assert np.abs(f1 - f0).max() < 1e-9
