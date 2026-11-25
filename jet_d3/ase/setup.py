"""jet_d3_ase: the GPU D3 pair style jet/d3 as an ASE calculator (GPL-2.0, like jet_d3/).

    JET_CUDA_ARCH=90 pip install -e jet_d3/ase --no-build-isolation   # in an env with torch + CUDA toolkit, like jet_mlip

jet_d3/pair_jet_d3.cu is compiled unchanged, against the small LAMMPS stand-in in shim/ (no LAMMPS needed), with the
flags of the LAMMPS build (-O3 --use_fast_math). GPU architecture: JET_CUDA_ARCH as for jet_mlip (90 -> 9.0; or
TORCH_CUDA_ARCH_LIST; default: the GPU present).
"""
import os

if os.environ.get('JET_CUDA_ARCH') and not os.environ.get('TORCH_CUDA_ARCH_LIST'):
    a = os.environ['JET_CUDA_ARCH']
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{a[:-1]}.{a[-1]}'

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))
d3 = os.path.dirname(here)   # jet_d3/: pair_jet_d3.{h,cu}, pair_jet_d3_pars.h

setup(
    name='jet_d3_ase',
    version='0.4.5',
    description='JET-MLIPs GPU D3 (jet/d3) for ASE',
    license='GPL-2.0',
    packages=['jet_d3_ase'],
    ext_modules=[CUDAExtension(
        'jet_d3_ase._C',
        sources=[os.path.join(here, 'd3_ext.cpp'), os.path.join(d3, 'pair_jet_d3.cu')],
        include_dirs=[os.path.join(here, 'shim'), d3],   # shim first: its pair.h / mpi.h stand in for LAMMPS'
        extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math']},
    )],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['ase', 'numpy'],
)
