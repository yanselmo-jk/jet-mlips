# SPDX-License-Identifier: GPL-2.0 (like jet_d3/)
"""ASE calculator for the GPU D3 dispersion of the LAMMPS pair style jet/d3."""
import math

import numpy as np
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes

from ._C import D3

_BOHR = 0.52917726  # Angstrom; the LAMMPS D3 cut-offs are in bohr^2


class JetD3Calculator(Calculator):
    """D3 dispersion on the GPU (jet/d3). xc: functional name ('pbe', 'r2scan', ...); damping: 'bj' or 'zero';
    params: {'s6', 's8', 'a1', 'a2'} (BJ only) instead of xc; cutoff / cn_cutoff in Angstrom (default: the LAMMPS
    defaults, 9000 and 1600 bohr^2)."""
    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']
    default_parameters = dict(xc='pbe', damping='bj', params=None, cutoff=None, cn_cutoff=None)

    def __init__(self, **kwargs):
        self._d3 = {}  # elements tuple -> D3, for the current parameters
        super().__init__(**kwargs)
        self._settings()   # validate now

    def set(self, **kwargs):
        changed = super().set(**kwargs)
        if changed:   # new D3 parameters: drop the D3 objects and the cached results (ASE keeps them otherwise)
            self._d3.clear()
            self.reset()
        return changed

    def _settings(self):
        """The pair_style jet/d3 arguments of the current parameters."""
        p = self.parameters
        if p.damping not in ('bj', 'zero'):
            raise ValueError("damping must be 'bj' or 'zero'")
        if p.params is not None and p.damping != 'bj':
            raise ValueError("params need damping='bj'")
        s = []
        for r, default in ((p.cutoff, 9000.0), (p.cn_cutoff, 1600.0)):
            r2 = default if r is None else (float(r) / _BOHR) ** 2
            if not (math.isfinite(r2) and r2 > 0):
                raise ValueError('cut-offs must be finite and positive')
            s.append(repr(r2))
        s.append('damp_' + p.damping)
        if p.params is None:
            return s + [str(p.xc).lower()]
        return s + ['custom'] + [repr(float(p.params[k])) for k in ('s6', 's8', 'a1', 'a2')]

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        pos = atoms.positions
        cell = atoms.cell.array
        rank = np.linalg.matrix_rank(cell)
        if rank == 3:
            # cell = A @ Q with A the LAMMPS (lower-triangular, positive diagonal) lattice rows
            q, r = np.linalg.qr(cell.T)
            sign = np.sign(np.diag(r))
            a = r.T * sign
            q = sign[:, None] * q.T
            x = pos @ q.T
            box = [a[0, 0], a[1, 1], a[2, 2], a[1, 0], a[2, 0], a[2, 1], 0.0, 0.0, 0.0]
        elif not any(atoms.pbc):
            lo = pos.min(axis=0) - 1.0
            hi = pos.max(axis=0) + 1.0
            q = np.eye(3)
            x = pos - lo
            box = [*(hi - lo), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            raise ValueError('periodic directions need a cell of rank 3')

        symbols = atoms.get_chemical_symbols()
        elements = tuple(sorted(set(symbols)))
        index = {s: i + 1 for i, s in enumerate(elements)}
        if elements not in self._d3:
            self._d3[elements] = D3(self._settings(), list(elements))
        e, f, virial = self._d3[elements].compute(
            x, np.array([index[s] for s in symbols], dtype=np.int32), box, [int(p) for p in atoms.pbc])

        results = {'energy': e, 'free_energy': e, 'forces': f @ q}
        if all(atoms.pbc):
            vm = np.array([[virial[0], virial[3], virial[4]],
                           [virial[3], virial[1], virial[5]],
                           [virial[4], virial[5], virial[2]]])
            m = -(q.T @ vm @ q) / atoms.get_volume()
            results['stress'] = np.array([m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1]])
        self.results = results
        if 'stress' in properties and 'stress' not in results:
            raise PropertyNotImplementedError('stress needs a fully periodic cell')
