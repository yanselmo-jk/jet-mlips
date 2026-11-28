"""The jet_d3_ase calculator against finite differences, rotations and a molecule; needs CUDA."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
if not torch.cuda.is_available():
    pytest.skip('no CUDA', allow_module_level=True)
pytest.importorskip('jet_d3_ase._C')

from ase.build import bulk, molecule  # noqa: E402
from ase.calculators.calculator import PropertyNotImplementedError  # noqa: E402
from jet_d3_ase import JetD3Calculator  # noqa: E402

A = 5.43


def cubic(n=2, rattle=0.0):
    atoms = bulk('Si', 'diamond', a=A, cubic=True).repeat((n, n, n))
    if rattle:
        atoms.rattle(rattle, seed=0)
    return atoms


def energy(calc, atoms, i, k, d=0.0):
    a = atoms.copy()
    a.positions[i, k] += d
    a.calc = calc
    return a.get_potential_energy()


@pytest.mark.parametrize('damping', ['bj', 'zero'])
def test_forces_stress(damping):
    atoms = cubic(rattle=0.05)
    calc = JetD3Calculator(damping=damping)
    atoms.calc = calc
    f = atoms.get_forces()
    stress = atoms.get_stress()
    assert np.abs(f.sum(axis=0)).max() < 1e-8

    h = 1e-2   # the energy is summed in fp32 (noise ~1e-6 eV here): a smaller step resolves the noise, not the slope
    for i in range(3):
        for k in range(3):
            ep = energy(calc, atoms, i, k, +h)
            em = energy(calc, atoms, i, k, -h)
            assert abs(f[i, k] + (ep - em) / (2 * h)) < 1e-4, (i, k)

    eps = 1e-3
    vol = atoms.get_volume()
    for row, col, voigt in ((0, 0, 0), (0, 1, 5)):
        es = []
        for d in (eps, -eps):
            deform = np.eye(3)
            deform[row, col] += d
            a = atoms.copy()
            a.calc = calc
            a.set_cell(atoms.cell.array @ deform.T, scale_atoms=True)
            es.append(a.get_potential_energy())
        assert abs((es[0] - es[1]) / (2 * eps * vol) - stress[voigt]) < 1e-5, voigt


def test_prelaunch_without_sums():
    """pair_e3gnn launches D3 before compute() knows eflag / vflag; a launch that left the energy and virial sums out
    (here: no step marked) is repeated by compute(), so the result is the one of a plain compute()."""
    from jet_d3_ase._C import D3
    atoms = cubic(rattle=0.05)
    d3 = D3(['9000', '1600', 'damp_bj', 'pbe'], ['Si'])
    args = (atoms.positions, np.ones(len(atoms), dtype=np.int32), [*atoms.cell.lengths(), 0, 0, 0, 0, 0, 0], [1, 1, 1])
    e0, f0, v0 = d3.compute(*args)
    e1, f1, v1 = d3.compute(*args, prelaunch=True)
    assert e0 < 0 and abs(e1 - e0) < 1e-12
    assert np.abs(f1 - f0).max() < 1e-12 and np.abs(v1 - v0).max() < 1e-10


def rotation(seed=1):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_rotation():
    calc = JetD3Calculator()
    atoms = cubic(rattle=0.05)
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()

    r = rotation()
    a = atoms.copy()
    a.calc = calc
    a.set_cell(atoms.cell.array @ r.T, scale_atoms=False)
    a.positions = atoms.positions @ r.T
    assert abs(a.get_potential_energy() - e0) < 1e-8
    assert np.abs(a.get_forces() - f0 @ r.T).max() < 1e-6

    # stress of a rotated, sheared (triclinic) cell rotates as R s R^T; a left-handed copy has the same energy
    from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress
    t = atoms.copy()
    t.set_cell(atoms.cell.array @ np.array([[1, 0.07, 0.03], [0, 1, -0.05], [0, 0, 1]]).T, scale_atoms=True)
    t.calc = calc
    s0 = voigt_6_to_full_3x3_stress(t.get_stress())
    tr = t.copy()
    tr.set_cell(t.cell.array @ r.T, scale_atoms=False)
    tr.positions = t.positions @ r.T
    tr.calc = calc
    assert np.abs(tr.get_stress() - full_3x3_to_voigt_6_stress(r @ s0 @ r.T)).max() < 1e-7
    left = t.copy()
    left.set_cell(t.cell.array[[1, 0, 2]], scale_atoms=False)
    left.calc = calc
    assert abs(left.get_potential_energy() - t.get_potential_energy()) < 1e-5   # other LAMMPS frame: fp32 rounding

    # triclinic cell: the same per-atom energy as the cubic one; cells well above the 21 A CN cut-off (in small cells
    # the fp32 sum over many lattice images moves the energy by up to ~1e-5 eV/atom)
    prim = bulk('Si', 'diamond', a=A).repeat((6, 6, 6))
    prim.calc = calc
    cubic_atoms = cubic(4)
    cubic_atoms.calc = calc
    assert abs(prim.get_potential_energy() / len(prim)
               - cubic_atoms.get_potential_energy() / len(cubic_atoms)) < 1e-6
    # absolute: fp64 s-dftd3 1.7.0 (BJ, PBE, no ATM, cut-offs sqrt(9000) / sqrt(1600) bohr) gives -0.3142749603 eV/atom
    assert abs(cubic_atoms.get_potential_energy() / len(cubic_atoms) + 0.3142749603) < 1e-7


def test_molecule():
    atoms = molecule('CH3CH2OH')
    atoms.calc = JetD3Calculator()
    e = atoms.get_potential_energy()
    assert np.isfinite(e)
    assert e < 0
    assert np.abs(atoms.get_forces().sum(axis=0)).max() < 1e-8
    assert 'stress' not in atoms.calc.results
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stress()


def test_functionals():
    atoms = cubic()
    e = {}
    for name, kw in [('pbe', {}),
                     ('r2scan', {'xc': 'r2scan'}),
                     ('custom', {'params': {'s6': 1.0, 's8': 0.7898, 'a1': 0.4948, 'a2': 5.7308}})]:
        a = atoms.copy()
        a.calc = JetD3Calculator(**kw)
        e[name] = a.get_potential_energy()
    assert abs(e['r2scan'] - e['pbe']) > 1e-6
    assert abs(e['custom'] - e['r2scan']) < 1e-10

    # set() changes the D3 parameters of an existing calculator; a molecule after a crystal has no stale stress
    calc = JetD3Calculator()
    a = atoms.copy()
    a.calc = calc
    a.get_potential_energy()
    calc.set(xc='r2scan')
    assert abs(a.get_potential_energy() - e['r2scan']) < 1e-10
    calc.calculate(atoms, ['energy', 'stress'])
    calc.calculate(molecule('CH3CH2OH'), ['energy'])
    assert 'stress' not in calc.results
    with pytest.raises(ValueError):
        JetD3Calculator(cutoff=float('inf'))
    with pytest.raises(ValueError):
        JetD3Calculator(params={'s6': 1.0, 's8': 0.8, 'a1': 0.4, 'a2': 5.0}, damping='zero')


def test_prelaunch_forces_only():
    """An MD step that needs forces only (eflag = vflag = 0): the prelaunch leaves the energy and virial sums out and
    compute() does not launch again; the forces are those of a full compute, with or without the prelaunch."""
    from jet_d3_ase._C import D3
    atoms = cubic(rattle=0.05)
    d3 = D3(['9000', '1600', 'damp_bj', 'pbe'], ['Si'])
    args = (atoms.positions, np.ones(len(atoms), dtype=np.int32), [*atoms.cell.lengths(), 0, 0, 0, 0, 0, 0], [1, 1, 1])
    e0, f0, v0 = d3.compute(*args)
    assert e0 < 0 and np.abs(f0).max() > 1e-3 and np.abs(v0).max() > 1e-3
    for prelaunch in (True, False):
        e1, f1, v1 = d3.compute(*args, prelaunch=prelaunch, ev=False)
        assert e1 == 0 and not v1.any()   # not asked for
        assert np.abs(f1 - f0).max() < 1e-12, prelaunch


def test_float_float_sums(monkeypatch):
    """JET_D3_FP64=0 (float-float pair sums, for GPUs with slow fp64) against the default plain fp64 sums; the switch is
    read when the pair style is set up."""
    res = []
    for flag in ('1', '0'):
        monkeypatch.setenv('JET_D3_FP64', flag)
        atoms = cubic(rattle=0.05)
        atoms.calc = JetD3Calculator()
        res.append((atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()))
    (e0, f0, s0), (e1, f1, s1) = res
    assert not np.array_equal(f0, f1), 'bit-identical forces: JET_D3_FP64=0 did not change the sums'
    assert abs(e1 - e0) < 1e-9 * len(atoms) and np.abs(f1 - f0).max() < 1e-9 and np.abs(s1 - s0).max() < 1e-10


@pytest.mark.parametrize('shape', ['cubic', 'triclinic', 'slab'])
def test_tile_lists(monkeypatch, shape):
    """A box large enough for the tile lists (every periodic direction beyond 1.5 vdW cut-offs, 75.3 A) against
    JET_D3_SCAN=1, the scan over all block pairs that small boxes use: the same pair terms, so the results differ only
    by the order of the fp64 sums. The switch is read when the pair style is set up."""
    atoms = cubic(15, rattle=0.1)   # 81.45 A, 27,000 atoms
    atoms.symbols[::3] = 'C'
    atoms.symbols[::7] = 'H'
    if shape == 'triclinic':
        atoms.set_cell(atoms.cell.array + [[0, 0, 0], [9.0, 0, 0], [-7.0, 5.0, 0]], scale_atoms=True)
    if shape == 'slab':   # vacuum above, not periodic along z (no stress)
        del atoms[atoms.positions[:, 2] > 40.0]
        atoms.pbc = (True, True, False)
    res = []
    for scan in ('0', '1'):
        monkeypatch.setenv('JET_D3_SCAN', scan)
        atoms.calc = JetD3Calculator()
        res.append((atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress() if all(atoms.pbc) else np.zeros(6)))
    (e0, f0, s0), (e1, f1, s1) = res
    assert e0 < 0 and np.abs(f0).max() > 1e-3
    assert abs(e1 - e0) < 1e-11 * abs(e0)
    assert np.abs(f1 - f0).max() < 1e-12
    assert np.abs(s1 - s0).max() <= 1e-10 * np.abs(s0).max()
