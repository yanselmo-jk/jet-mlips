# Design and layout

Kernels: `jet_mlip/` (fused equivariant convolution with tabulated radial weights, torch ops `jet_mlip::*`):
per-edge `table_idx` for per-element-pair tables, the SevenNet, MACE, NequIP-OAM and Nequix shapes, and
`JET_CONFIGS=all | <family list>` (one shape list per family in `jet_mlip/configs.py`, each complete on its own;
the SevenNet list is literal, so envs without sevenn can build it; `python -m jet_mlip.shapes` checks a model). `jet_d3/`: LAMMPS `pair_style jet/d3`.
`patches/sevenn-jet*.patch`: the SevenNet 0.13.0 jet patch (pointed at `jet_mlip`), applied by
`python -m jet_mlip.patch_sevenn`. The sevenn D3 patch was dropped in v0.4.0: D3 is `jet/d3` for every family.

Repo layout:
- `common/`: `patch_tool.py` (apply / check / reverse the shipped patches), `radial_table.py`.
- `patches/`: one patch per upstream version (`patches/BASE`).
- `lammps/`: `build_lammps.sh` (ML-IAP + Kokkos CUDA + PYTHON + KSPACE + D3, one build dir per env), example inputs (`examples/`).

## MACE (jet_mace)

Fused JET convolution with a deploy-time radial table (per element pair for the Agnesi models MPA-0 /
OMAT-0), the one-hot element `skip_tp` as a species-grouped Triton GEMM (`jet_mace/skip.py`), cuEq for the rest,
`torch.compile` over the non-convolution glue, one CUDA graph per MD step. The 256-channel `large` model runs as two
128-channel halves. `JET_MLIAP_HEADROOM` is the capacity growth factor: every growth re-captures and recompiles.
Both wrappers point torchinductor's cache at `~/.cache/torchinductor`.

## NequIP (jet_nequip)

The packaged model keeps its own nequip code; only the convolution (`tp_scatter` + `edge_mlp`) and the SH edge attrs are
swapped (fused JET convolution with a radial table). Static buckets + CUDA graph (shared path, any rank count).


## LAMMPS details (moved from the README, 2026-11-05)

| option | patch added | pair styles |
|---|---|---|
| (always) | ML-IAP Kokkos: host comm fix, neighbour data, virial reduction (`lammps/*.patch`) | `mliap unified` (MACE, NequIP, Nequix, EquFlash) |
| `D3=1` (default) | `jet_d3/` (GPU D3, also `custom s6 s8 a1 a2` BJ parameters) | `jet/d3` |
| `SEVENN=1` | `sevenn patch_lammps --jet` (needs sevenn + jet_mlip in the env; own source tree) | `e3gnn`, `e3gnn/parallel` |

A binary runs the models of its own env: MACE (e3nn 0.4.4) needs its own env and build; SevenNet and NequIP share one
(`SEVENN=1` build in a NequIP env with sevenn installed). An env with its own CUDA toolkit builds with it (e.g. EquFlash's
env: CUDA 13.4 and conda-forge openmpi; `conda activate equflash; bash lammps/build_lammps.sh`, no `CUDA_HOME` from another env).

`mliap unified`: run with `-k on g 1 -sf kk -pk kokkos newton on neigh half`. On one GPU every family (MACE, NequIP,
Nequix, EquFlash v1, SevenNet ML-IAP) runs one shared CUDA-graph path (`jet_mlip/mliap.py`: ghosts folded onto their
owners, padded edges masked out of the convolution, one capture per size bucket, then replay). EquFlash V2 runs eager
(its graph capture is invalidated). Print a thermo quantity that needs the model (`thermo_style custom step pe temp`).

Several GPUs (`mpirun -np N lmp -k on g N ...`, add `gpu/aware off` to `-pk kokkos` with a non-CUDA-aware MPI, v0.4.0):
every `mliap unified` family stays on the shared graph path; atoms owned by another rank are remote rows refreshed per
layer by NCCL inside the graph (`jet_mlip/mliap.py`). With the graph off (`JET_<FAMILY>_CUDA_GRAPH=0`) MACE, NequIP,
SevenNet and EquFlash take their official per-layer LAMMPS exchange; Nequix eager is one rank only.

Radial table size (`--radial_table N`, grid points). The defaults keep the table's force error well below the official
fp32 error (fp64 table model vs the fp64 model, max |dF| on a 348-atom box, as a share of the official fp32 error):
MACE 4096 (23 %; 2048 gives 180 %, 8192 gives 3 % but is 3-4 % slower with +56 % memory on MPA-0), NequIP 2048
(1.5-4 %), Nequix 2048 (1-2 %), SevenNet and EquFlash 2048, rounded down so that a grid node sits at the XPLOR cutoff
onset r_on (SevenNet 7net-nano 2047: ~7 %; EquFlash 2041: JET fp32 forces match the official fp32 ones).

Deploy options: `--radial_table N` (see above; SevenNet also `--radial_table 0` = exact radial MLP, with
`--exact_backward recompute` for no per-edge buffer: table-level memory, up to 8 % slower than the default `split`; only
SevenNet has a JET exact kernel, the other families take their table-off path in reference or upstream code), `--no-graph` (official ML-IAP wrapper instead of the
CUDA-graph one); MACE `--no-cueq`, `--head`; NequIP `--no-jet --modifiers enable_OpenEquivariance` (official kernels in the
same wrapper). Runtime knobs: `JET_<FAMILY>_CUDA_GRAPH=0` (FAMILY = MACE, NEQUIP, NEQUIX, EQUFLASH, SEVENN: eager
path), `JET_MLIAP_HEADROOM` (size-bucket growth, default 1.1), `JET_MACE_COMPILE=0`, `JET_NEQUIP_COMPILE=0`.

## Tests

`python -m pytest tests/mace_jet tests/test_*.py` (MACE env), `python -m pytest tests/nequip_jet` (NequIP env),
`python -m pytest tests/nequix_jet` (Nequix env), `python -m pytest tests/equflash_jet` (EquFlash env; the LAMMPS tests need a GPU),
`python -m pytest jet_mlip/tests` (kernel; the sevenn comparisons need a SevenNet env).
The LAMMPS ML-IAP path has its own tests (GPU): `jet_mlip/tests/test_mliap.py` (shared CUDA-graph wrapper: padding,
ghost fold, capture / replay, capacity growth), `jet_mlip/tests/test_sevenn_mliap.py`, and `test_mliap.py` in each
family's test folder (graph vs eager, vs upstream, files deployed with v0.2.1).

## Version history in detail

| version | changes |
|---|---|
| v0.3.1 | shared ML-IAP wrapper: padded edges spread over the local rows (sorted last in the masked CSR), so scatters outside the JET convolution (upstream convolutions, MACE density, NequIP ZBL) no longer serialize on one row (Nequix `--no-jet` 348 atoms: 50.8 -> 32.9 ms/step); kernels as v0.3.0; one deploy command `python -m jet_mlip.deploy <family>` |
| v0.3.0 | one LAMMPS ML-IAP CUDA-graph wrapper for every family (`jet_mlip/mliap.py`: ghost fold, padded edges masked out of the convolution, input preparation inside the graph, no energy sync when LAMMPS does not need the energy); SevenNet ML-IAP route `jet_mlip.sevenn_mliap` (e3gnn stays the default); EquFlash v1 on the CUDA graph; radial-table kernel: 1/h as a hi/lo pair (removes a systematic table-index rounding bias); EquFlash table grid has a node at the XPLOR onset r_on (re-deploy EquFlash models); files deployed with v0.2.1 still load |
| v0.2.1 | SevenNet deploy options on one axis: `--radial_table N` (0 or `--no-radial_table` = exact MLP), `--exact_backward split` or `recompute` (exact only; replaces `--jet_memory`, `fastest` removed); pair_e3gnn frees the warm-up cache before the CUDA-graph capture (startup peak 7net-omni 6.1 -> 4.3 GB). Kernels as v0.2.0 |
| v0.2.0 | own kernel library `jet_mlip` (fiber fold, register-model path groups for the radial-table kernels, lmax 4); SevenNet patch; Nequix and EquFlash v1/V2 (ASE + LAMMPS ML-IAP, multi-GPU); GPU D3 `jet/d3`; MIT license |
| v0.1.0 | MACE and NequIP on the first JET kernel |
