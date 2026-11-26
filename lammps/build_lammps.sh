#!/usr/bin/env bash
# Build LAMMPS with ML-IAP (unified python interface) + KOKKOS (CUDA) + PYTHON + KSPACE + MOLECULE + EXTRA-FIX
# against the python of the active conda env, and installs `lmp` into that env:
#   conda activate <env>; bash lammps/build_lammps.sh   -> `lmp` in <env> (build dir $SRC/lammps-accel/build_<env>)
#   SEVENN=1 bash lammps/build_lammps.sh                 -> also pair_style e3gnn (tree $SRC/lammps-accel-sevenn)
# `lmp` is a launcher in $CONDA_PREFIX/bin: it puts this env's libraries (torch, python, the CUDA runtime of the build)
# on the library path and runs the compiled binary, so it works whenever the env is active (mpirun included).
# Knobs: SRC ($HOME/src), LAMMPS_DIR (lammps-accel; lammps-accel-sevenn with SEVENN=1; or the path of an extracted
# LAMMPS source folder, the one holding src/: offline), LAMMPS_TAG (stable_22Jul2025_update6;
# ML-IAP forward_exchange needs >= 22Jul2025), JOBS (4), TORCH_CUDA_ARCH_LIST ("8.9", "9.0", ...;
# default: the GPU present), CUDA_ENV (conda env whose nvcc/g++ compile Kokkos when the active env has none; default mace),
# OFFLINE=1 (skip pip), FRESH=1 (wipe the build dir), SEVENN=1 (also pair_style e3gnn /
# e3gnn/parallel: sevenn patch_lammps --jet; env needs sevenn + jet_mlip, e.g. sevenn-jet; own source tree
# lammps-accel-sevenn, because the libtorch CMake block would break the other builds of a shared tree).
set -eo pipefail
[ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-base}" != base ] || {
  echo "activate the conda env that holds the official package and jet-mlips first: conda activate <env>"; exit 1; }
ENV_NAME=${CONDA_DEFAULT_ENV##*/}
TAG=${TAG:-$ENV_NAME}
SRC=${SRC:-$HOME/src}
JOBS=${JOBS:-4}
LAMMPS_TAG=${LAMMPS_TAG:-stable_22Jul2025_update6}
if [ "${SEVENN:-0}" = 1 ]; then LAMMPS_DIR=${LAMMPS_DIR:-lammps-accel-sevenn}; else LAMMPS_DIR=${LAMMPS_DIR:-lammps-accel}; fi
HERE=$(cd "$(dirname "$0")" && pwd)
# sevenn's pair_e3gnn needs both packages in the activated env (sevenn-jet is the packed one)
[ "${SEVENN:-0}" = 1 ] && { python -c 'import sevenn, jet_mlip' || { echo "$ENV_NAME lacks sevenn / jet_mlip (needed for SEVENN=1)"; exit 1; }; }
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$(python -c 'import torch;print("%d.%d" % torch.cuda.get_device_capability() if torch.cuda.is_available() else "8.9")')}
CUDA_ARCHS=$(echo "$TORCH_CUDA_ARCH_LIST" | tr ' ,' ';;' | tr -d '.')
ARCH1=${CUDA_ARCHS%%;*}
case $ARCH1 in 80) KOKKOS_ARCH=Kokkos_ARCH_AMPERE80;; 86) KOKKOS_ARCH=Kokkos_ARCH_AMPERE86;; 89) KOKKOS_ARCH=Kokkos_ARCH_ADA89;;
  90) KOKKOS_ARCH=Kokkos_ARCH_HOPPER90;; 100) KOKKOS_ARCH=Kokkos_ARCH_BLACKWELL100;; 120) KOKKOS_ARCH=Kokkos_ARCH_BLACKWELL120;;
  *) KOKKOS_ARCH=${KOKKOS_ARCH:-};; esac
[ -n "$KOKKOS_ARCH" ] || { echo "set KOKKOS_ARCH for sm_$ARCH1"; exit 1; }
# CUDA toolkit + host compiler: this env's, else CUDA_ENV's (an env without nvcc, e.g. torch cu130 wheels), else nvcc on PATH
CUDA_ENV_PREFIX=$(conda info --base 2>/dev/null || true)/envs/${CUDA_ENV:-mace}   # no conda on the offline box: CUDA_HOME is exported there
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -x "$CONDA_PREFIX/bin/nvcc" ]; then CUDA_HOME=$CONDA_PREFIX; elif [ -x "$CUDA_ENV_PREFIX/bin/nvcc" ]; then CUDA_HOME=$CUDA_ENV_PREFIX;
  else CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")"); fi
fi
export CUDA_HOME CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH   # cmake / make / mpicxx of the toolchain env when this env lacks them
CUDA_INC=$CUDA_HOME/targets/x86_64-linux/include; [ -d "$CUDA_INC" ] || CUDA_INC=$CUDA_HOME/include
CUDA_LIB=$CUDA_HOME/targets/x86_64-linux/lib;     [ -d "$CUDA_LIB" ] || CUDA_LIB=$CUDA_HOME/lib64
export CPATH=$CUDA_INC:${CPATH:-} LIBRARY_PATH=$CUDA_LIB:${LIBRARY_PATH:-}
GXX=$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++; [ -x "$GXX" ] || GXX=$(command -v g++)   # nvcc 12.x rejects gcc > 13
GCC=${GXX%g++}gcc
export NVCC_WRAPPER_DEFAULT_COMPILER=$GXX CC=$GCC CXX=$GXX
echo "env $ENV_NAME ($CONDA_PREFIX)  GPU sm_${CUDA_ARCHS//;/, sm_}  CUDA_HOME=$CUDA_HOME  nvcc=$($CUDACXX --version | tail -1)  g++=$GXX"
[ -n "${OFFLINE:-}" ] || pip install -q "cython==3.0.11" cupy-cuda12x   # ML-IAP Kokkos python module needs both

cd "$SRC"
[ -d "$LAMMPS_DIR" ] || git clone --depth=1 --branch "$LAMMPS_TAG" https://github.com/lammps/lammps.git "$LAMMPS_DIR"
cd "$LAMMPS_DIR"
[ -d src ] || { echo "LAMMPS_DIR=$LAMMPS_DIR has no src/ (give the extracted LAMMPS source folder)"; exit 1; }
grep -q 'forward_comm(class Pair \*, int size' src/comm.h || { echo "needs LAMMPS >= 22 Jul 2025 (tree: $PWD)"; exit 1; }
# Kokkos ML-IAP host-comm fallback (MPI not GPU-aware, e.g. conda OpenMPI on WSL2) reads device memory from the host and
# segfaults with mpirun -np > 1: stage through DualViews. Kokkos 4.x (22Jul2025 line) / 5.x (feature releases) variants.
HC=lammps_mliap_kokkos_hostcomm.patch; grep -q tdual_xfloat_1d src/KOKKOS/kokkos_type.h || HC=lammps_mliap_kokkos_hostcomm_kokkos5.patch
grep -q "mliap:fill" src/KOKKOS/pair_mliap_kokkos.cpp || git apply -C1 "$HERE/$HC"
# ML-IAP neighdata kernels: one team per atom instead of one thread looping over ~1000 neighbours (~10 ms/step saved
# on 5568 atoms with a 15 A list)
grep -q "mliap-accel: neighdata" src/KOKKOS/mliap_data_kokkos.cpp || git apply -C1 "$HERE/lammps_mliap_kokkos_neighdata.patch"
# ML-IAP global virial by reduction instead of 6 atomics contended by every pair (3.07M pairs, H200: 27 -> 0.32 ms per call)
grep -q "mliap-accel: virial reduce" src/KOKKOS/mliap_unified_kokkos.cpp || git apply -C1 "$HERE/lammps_mliap_kokkos_virial.patch"
# `pair_style jet/mliap <model.pt>` (= `mliap unified <model.pt> 0`), KOKKOS only: the JET ML-IAP models need Kokkos device data
grep -q "PairJetMLIAPKokkos" src/KOKKOS/pair_mliap_kokkos.h || git apply -C1 "$HERE/lammps_mliap_kokkos_jet_style.patch"

# GPU D3 pair style jet/d3 (jet_d3/, custom BJ parameters): D3=1 copies the sources and appends
# one CMake block (once). `pair_style hybrid/overlay jet/mliap m.pt jet/d3 9000 1600 damp_bj custom s6 s8 a1 a2`.
if [ "${D3:-1}" = 1 ]; then
  cp "$HERE/../jet_d3/pair_jet_d3.cu" "$HERE/../jet_d3/pair_jet_d3.h" "$HERE/../jet_d3/pair_jet_d3_pars.h" src/
  grep -q 'mlip-accel: pair_jet_d3' cmake/CMakeLists.txt || cat >> cmake/CMakeLists.txt <<'CM'

# --- mlip-accel: pair_jet_d3 (CUDA) ---
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
target_sources(lammps PRIVATE ${LAMMPS_SOURCE_DIR}/pair_jet_d3.cu)
set_source_files_properties(${LAMMPS_SOURCE_DIR}/pair_jet_d3.cu PROPERTIES COMPILE_OPTIONS "-O3;--use_fast_math")
target_link_libraries(lammps PUBLIC CUDA::cudart CUDA::cuda_driver)
CM
fi

# pair_e3gnn(+parallel) + the jet .so link from the installed sevenn; no --d3: jet_d3 (pair_style jet/d3) is ours and has the
# `custom s6 s8 a1 a2` functional (sevenn's own d3 is not needed).
# After the D3 block: torch's cuda.cmake (find_package(Torch)) resets CMAKE_CUDA_ARCHITECTURES for later CUDA sources
if [ "${SEVENN:-0}" = 1 ] && [ ! -f src/pair_e3gnn.cpp ]; then sevenn patch_lammps . --jet; fi
# re-point the jet .so link at this env's build (patch_lammps runs once per tree; the env may use another checkout now)
[ "${SEVENN:-0}" = 1 ] && ln -sf "$(python -c 'import jet_mlip;print(jet_mlip.library_path())')" cmake/jet_mlip/libjet_mlip.so

BUILD=build_$TAG
[ -n "${FRESH:-}" ] && rm -rf "$BUILD"
mkdir -p "$BUILD" && cd "$BUILD"
# sevenn's CMake block does find_package(Torch) + links libtorch and the jet .so; MKL_INCLUDE_DIR=/tmp
EXTRA=()
if [ "${SEVENN:-0}" = 1 ]; then EXTRA=(-D CMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" -D MKL_INCLUDE_DIR=/tmp); fi
cmake ../cmake -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER=$PWD/../lib/kokkos/bin/nvcc_wrapper -D CMAKE_CUDA_HOST_COMPILER="$GXX" -D CMAKE_CUDA_COMPILER="$CUDACXX" \
  -D CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" -D CUDAToolkit_INCLUDE_DIR="$CUDA_INC" -D CUDA_TOOLKIT_ROOT_DIR="$(dirname "$CUDA_INC")" \
  -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D $KOKKOS_ARCH=ON \
  -D PKG_ML-IAP=ON -D PKG_ML-SNAP=ON -D MLIAP_ENABLE_PYTHON=ON -D PKG_PYTHON=ON -D BUILD_SHARED_LIBS=ON \
  -D PKG_KSPACE=ON -D FFT=KISS -D PKG_MOLECULE=ON -D PKG_EXTRA-FIX=ON -D PKG_EXTRA-PAIR=ON -D BUILD_MPI=ON \
  -D Python_EXECUTABLE="$CONDA_PREFIX/bin/python" "${EXTRA[@]}"
make -j"$JOBS"
# python module into this env (= make install-python minus its build venv, which pip-installs build tools)
rm -rf build-python && cp -r ../python build-python && cp liblammps.so build-python/lammps/
(V=$PWD/../src/version.h; cd build-python && LAMMPS_SHARED_LIB=liblammps.so LAMMPS_VERSION_FILE=$V \
  "$CONDA_PREFIX/bin/pip" install -q --no-build-isolation --no-deps --no-index --force-reinstall . && cd .. && rm -rf build-python)   # this env's pip: PATH starts with the toolchain env
# runtime: this env's libs (torch, cupy, python) and the CUDA runtime that Kokkos was built with
JET_DIR= TORCH_LIB=
if [ "${SEVENN:-0}" = 1 ]; then   # the e3gnn .so lives in the jet_mlip package dir and links torch/lib
  JET_DIR=$(python -c 'import jet_mlip,os;print(os.path.dirname(jet_mlip.__file__))'):
  TORCH_LIB=$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):
fi
cat > lmp_accel_$TAG <<WRAP
#!/usr/bin/env bash
export LD_LIBRARY_PATH=$PWD:$JET_DIR$TORCH_LIB$CONDA_PREFIX/lib:$CUDA_LIB:\${LD_LIBRARY_PATH:-}
export PATH=$CONDA_PREFIX/bin:\$PATH
exec $PWD/lmp "\$@"
WRAP
chmod +x lmp_accel_$TAG
PAT="mliap|pppm"; [ "${SEVENN:-0}" = 1 ] && PAT="e3gnn|mliap|jet/d3"
./lmp_accel_$TAG -h | grep -E "$PAT" | head -3
# `lmp` in this env; an lmp that is not ours (e.g. conda-forge's lammps) is left alone
if [ -e "$CONDA_PREFIX/bin/lmp" ] && ! grep -qs "exec .*/lmp " "$CONDA_PREFIX/bin/lmp"; then
  echo "BUILD OK: $PWD/lmp_accel_$TAG ($CONDA_PREFIX/bin/lmp exists and is not ours: not replaced)"
else
  cp lmp_accel_$TAG "$CONDA_PREFIX/bin/lmp" && echo "BUILD OK: lmp installed in $ENV_NAME ($CONDA_PREFIX/bin/lmp)"
fi
