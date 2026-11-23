import glob
import os
import subprocess
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.environ.get('JET_TRAIN', '0') == '1'   # tp3, the train-mode path of the SevenNet / MACE patches (not supported)

# regenerate the constexpr tables from jet_mlip/configs.py
subprocess.check_call([sys.executable, os.path.join(here, 'configs.py')])

# Target GPUs. Empty = the GPU this runs on (torch's default). One or more sm numbers, e.g.
# JET_CUDA_ARCH=89 (RTX 40), 90 (H100/H200), 100 (B200/GB200), 120 (RTX 50), or a fat binary
# "89,90,100,120" (SASS for each + PTX of the last; sm_100/120 need CUDA >= 12.8).
arch = os.environ.get('JET_CUDA_ARCH', '').replace('sm_', '').replace('.', '')
archs = [a.strip() for a in arch.replace(';', ',').replace(' ', ',').split(',') if a.strip()]

# per-block opt-in shared memory (bytes) by sm
OPTIN_SMEM = {'70': 98304, '72': 98304, '75': 65536, '80': 166912, '86': 101376, '87': 166912,
              '89': 101376, '90': 232448, '100': 232448, '103': 232448, '120': 101376}


def smem_max():
    """Per-block opt-in shared memory the kernels may use. The path group count (Groups<>) is a
    constexpr derived from it, so it is a build-time constant. Order: JET_SMEM_MAX, else the
    smallest limit over the requested archs (a fat binary must fit its smallest target), else the
    GPU this runs on, else sm_89's."""
    env = os.environ.get('JET_SMEM_MAX')
    if env:
        return int(env)
    if archs:
        return min(OPTIN_SMEM.get(a, 101376) for a in archs)
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    return 101376


nvcc = ['-O3', '-lineinfo', '--expt-relaxed-constexpr', '-std=c++17', f'-DJET_SMEM_MAX={smem_max()}']
for a in archs:
    nvcc += [f'-gencode=arch=compute_{a},code=sm_{a}']
if archs:   # PTX of the newest target, so a GPU newer than every listed sm can still JIT it
    nvcc += [f'-gencode=arch=compute_{archs[-1]},code=compute_{archs[-1]}']
print(f'jet_mlip build: archs {archs or ["(local GPU)"]}, SMEM_MAX {smem_max()}')

setup(
    name='jet-mlip',
    version='0.4.3',
    license='CC-BY-NC-SA-4.0 (LAMMPS-derived files: GPL-2.0)',
    packages=['jet_mlip'],
    package_dir={'jet_mlip': '.'},
    ext_modules=[
        CUDAExtension(
            'jet_mlip._C',
            sources=sorted(s for s in glob.glob(os.path.join(here, 'csrc', '*.cpp')) + glob.glob(os.path.join(here, 'csrc', '*.cu'))
                           + glob.glob(os.path.join(here, 'csrc', 'inst', '*.cu'))
                           if TRAIN or os.path.basename(s) != 'tp3.cu'),   # JET_TRAIN=1: also the tp3 training op
            include_dirs=[os.path.join(here, 'csrc')],
            extra_compile_args={'cxx': ['-O3', '-std=c++17'], 'nvcc': nvcc},
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)},
)
