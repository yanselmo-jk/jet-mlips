"""GPU D3 dispersion (the LAMMPS pair style jet/d3) for ASE. GPL-2.0, like jet_d3/."""
import torch  # noqa: F401  (loads the libtorch / CUDA runtime the extension links)

from ._C import D3  # noqa: F401
from .calculator import JetD3Calculator  # noqa: F401
