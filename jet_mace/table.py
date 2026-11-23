"""Deploy-time radial tables of a MACE interaction block (jet_mlip conv_fused TABLE path).

MACE's per-edge TP weights are W = MLP(Bessel(T(r)) * cutoff(r)), T = optional AgnesiTransform(r; Z_i, Z_j).
Without the transform (MACE-MP-0 medium / small) W is a function of r alone: one table (n, 4*NW). With it
(mpa-0, omat-0) r_0 = (covalent radius_i + radius_j)/2 enters the MLP input, so W is tabulated once per unordered
pair of the species given at build time: (n_pairs, n, 4*NW) plus a Z x Z -> pair index lookup (`zz`), fed to
the kernel as the per-edge `table_idx`.
"""
import copy
import itertools

import torch

from common.radial_table import interpolate, make_radial_table, make_radial_tables  # noqa: F401  (interpolate re-exported)


class RadialTable:
    def __init__(self, tab, r_max, zz=None):
        self.tab = tab            # (n, 4*NW) or (n_pairs, n, 4*NW) fp32 [W, dW/dr, node diff / h, dW/dr(r_k+1)] per column
        self.r_max = float(r_max)
        self.zz = zz              # (Zmax+1, Zmax+1) int32 pair index (-1 = species not tabulated), or None

    @property
    def per_pair(self):
        return self.zz is not None


class TableBuilder:
    """Holds fp64 CPU copies of one block's radial stack and builds its table(s); picklable (deployed models carry it
    so per-pair tables can be extended when a new species shows up)."""

    def __init__(self, model, i, n):
        block, re = model.interactions[i], model.radial_embedding
        if not getattr(re, 'apply_cutoff', True):
            raise ValueError('jet_mace: apply_cutoff=False models are not supported yet')
        self.n = n
        self.mlp = copy.deepcopy(block.conv_tp_weights).double().cpu()   # tables are built on the CPU in fp64
        self.bessel = copy.deepcopy(re.bessel_fn).double().cpu()
        self.cutoff = copy.deepcopy(re.cutoff_fn).double().cpu()
        self.transform = copy.deepcopy(re.distance_transform).double().cpu() if hasattr(re, 'distance_transform') else None
        self.r_max = float(re.cutoff_fn.r_max)

    def w_of_r(self, zi=None, zj=None):
        """W(r) of one element pair (AgnesiTransform.forward for the pair, then the upstream radial stack)."""
        if self.transform is None:
            return lambda r: self.mlp(self.bessel(r.unsqueeze(-1)) * self.cutoff(r.unsqueeze(-1)))
        tr = self.transform
        r_0 = 0.5 * (tr.covalent_radii[zi] + tr.covalent_radii[zj])

        def w(r):
            x = r / r_0
            s = (1 + tr.a * x ** tr.q / (1 + x ** (tr.q - tr.p))).reciprocal()
            return self.mlp(self.bessel(s.unsqueeze(-1)) * self.cutoff(r.unsqueeze(-1)))
        return w

    def __call__(self, species=None) -> 'RadialTable':
        if self.transform is None:
            with torch.no_grad():
                return RadialTable(make_radial_table(self.w_of_r(), self.r_max, self.n), self.r_max)
        if species is None:
            raise ValueError('jet_mace: this model has a distance transform (Agnesi): pass species=[Z, ...] present in the system')
        species = sorted(set(int(z) for z in species))
        pairs = list(itertools.combinations_with_replacement(species, 2))
        with torch.no_grad():
            tab = make_radial_tables([self.w_of_r(zi, zj) for zi, zj in pairs], self.r_max, self.n)
        zz = torch.full((max(species) + 1,) * 2, -1, dtype=torch.int32)
        for k, (zi, zj) in enumerate(pairs):
            zz[zi, zj] = zz[zj, zi] = k
        return RadialTable(tab, self.r_max, zz)

