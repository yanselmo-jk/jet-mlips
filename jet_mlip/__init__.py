"""JET-MLIP (Joint-kernel Equivariant convolution with Tabulated radial weights): fused CUDA convolution for
equivariant MLIPs (Bessel + cutoff + radial MLP + spherical harmonics + uvu tensor product + scatter in one kernel)."""
import os

import torch

from .tables import ConvConfig, config_from_sevenn_conv, make_config  # noqa: F401

_C_PATH = None
for _f in os.listdir(os.path.dirname(__file__)):
    if _f.startswith('_C') and _f.endswith('.so'):
        _C_PATH = os.path.join(os.path.dirname(__file__), _f)
        break


def is_available() -> bool:
    return _C_PATH is not None and torch.cuda.is_available()


def library_path() -> str:
    """Path of the shared object (also used to link LAMMPS)."""
    if _C_PATH is None:
        raise ImportError('jet_mlip._C not built; run `pip install -e jet_mlip`')
    return _C_PATH


# A LAMMPS binary built with `sevenn patch_lammps --jet` links its own copy of
# this library and registers the ops at startup; its ML-IAP path then imports this
# package in-process. Loading the copy again would fail with a duplicate operator
# registration, so only load when the ops are not there yet.
if _C_PATH is not None and not hasattr(torch.ops.jet_mlip, 'config_groups'):
    torch.ops.load_library(_C_PATH)
