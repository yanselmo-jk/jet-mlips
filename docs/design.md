# Design and layout

Kernels: `jet_mlip/` (fused equivariant convolution with tabulated radial weights, torch ops `jet_mlip::*`):
per-edge `table_idx` for per-element-pair tables, the SevenNet, MACE, NequIP-OAM and Nequix shapes, and
`JET_CONFIGS=all | <family list>` (one shape list per family in `jet_mlip/configs.py`, each complete on its own;
the SevenNet list is literal, so envs without sevenn can build it; `python -m jet_mlip.shapes` checks a model). `jet_d3/`: LAMMPS `pair_style jet/d3`.
`patches/sevenn-jet*.patch`: the SevenNet 0.13.0 jet patch (pointed at `jet_mlip`), applied by
`python -m jet_mlip.patch_sevenn`. The sevenn D3 patch was dropped in v0.4.0: D3 is `jet/d3` for every family.

Radial-table convolution kernels (v0.5.0 additions):
- First layer (`conv_first.cu`): a scalar-only input of at most 32 channels, on graphs with more destination nodes than
  1.5 x 8 warps x SMs, runs a dedicated kernel; `JET_FIRST_LAYER=0` takes the generic kernel.
- Blocks per SM: each config gets 2 blocks/SM per direction when its register model fits (forward <= 141, backward <=
  154 registers); the 2-block kernel is used only on graphs above the same node threshold (small graphs are faster with
  1); `JET_TAB_KB=1` forces 1.
- Pair-factored fiber loop: the backward combines the fibers of one (m1, m2) pair (`q = sum coef * gout`) before
  multiplying by sh, x and W once per pair and unit; the forward computes `coef * (x * sh)`. Only the summation order
  changes.

`jet/d3` (GPU D3): C6 interpolated once per pair, BJ or zero damping; pair sums in fp64 by default (`JET_D3_FP64=0`:
  float-float, for GPUs with slow fp64); energy and virial sums skipped on steps that need neither; atoms packed in
  cell order. Cut-offs are the physical ones of the D3 parametrization. Under `-sf kk` it runs as `jet/d3/kk`
  (`pair_jet_d3_kokkos.*`): reads positions from and adds forces to the KOKKOS device views, so a `hybrid/overlay` with
  `jet/mliap` does no per-step host/device copies; packing and the MPI gather stay on the host (one local-x copy).
  With `jet/e3gnn`, pair_e3gnn calls `jet/d3`'s `prelaunch()` so D3 runs on its own stream during the model.

Repo layout:
- `common/`: `patch_tool.py` (apply / check / reverse the shipped patches), `radial_table.py`.
- `patches/`: one patch per upstream version (`patches/BASE`).
- `lammps/`: `build_lammps.sh` (ML-IAP + Kokkos CUDA + PYTHON + KSPACE + D3, one build dir per env), and the examples moved to `examples/lammps/` (notebook + inputs; ASE: `examples/ase/`).

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
| (always) | ML-IAP Kokkos: host comm fix, neighbour data, virial reduction (`lammps/*.patch`) | `jet/mliap` (= `mliap unified <file> 0`, KOKKOS only; MACE, NequIP, Nequix, EquFlash, SevenNet ML-IAP) |
| `D3=1` (default) | `jet_d3/` (GPU D3, also `custom s6 s8 a1 a2` BJ parameters) | `jet/d3`, `jet/d3/kk` (under `-sf kk`) |
| `SEVENN=1` | `sevenn patch_lammps --jet` (needs sevenn + jet_mlip in the env; own source tree) | `jet/e3gnn`, `jet/e3gnn/parallel` (= `e3gnn`, `e3gnn/parallel`) |

A binary runs the models of its own env: MACE (e3nn 0.4.4) needs its own env and build; SevenNet and NequIP share one
(`SEVENN=1` build in a NequIP env with sevenn installed). An env with its own CUDA toolkit builds with it (e.g. EquFlash's
env: CUDA 13.4 and conda-forge openmpi; `conda activate equflash; bash lammps/build_lammps.sh`, no `CUDA_HOME` from another env).

`jet/mliap`: run with `-k on g 1 -sf kk -pk kokkos newton on neigh half`. On one GPU every family (MACE, NequIP,
Nequix, EquFlash v1, SevenNet ML-IAP) runs one shared CUDA-graph path (`jet_mlip/mliap.py`: ghosts folded onto their
owners, padded edges masked out of the convolution, one capture per size bucket, then replay). EquFlash V2 runs eager
(its graph capture is invalidated). Print a thermo quantity that needs the model (`thermo_style custom step pe temp`).

Several GPUs (`mpirun -np N lmp -k on g N ...`, add `gpu/aware off` to `-pk kokkos` with a non-CUDA-aware MPI, v0.4.0):
every `jet/mliap` family stays on the shared graph path; atoms owned by another rank are remote rows refreshed per
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
| v0.5.0 | skip connections (self-connection, per-element linears) as single CUDA ops and the layers around the convolution optimized; first-layer kernel, 2 blocks/SM for small radial-table kernels on large graphs, pair-factored fiber loop (7net-omni electrolyte 66.9 -> 62.0 ms/step, EquFlash v1 -10 %); `jet/d3` about 2x faster and `jet/d3/kk`; pair styles `jet/mliap`, `jet/e3gnn`; ASE neighbour list on the GPU; examples in `examples/` (notebooks) |
| v0.3.1 | shared ML-IAP wrapper: padded edges spread over the local rows (sorted last in the masked CSR), so scatters outside the JET convolution (upstream convolutions, MACE density, NequIP ZBL) no longer serialize on one row (Nequix `--no-jet` 348 atoms: 50.8 -> 32.9 ms/step); kernels as v0.3.0; one deploy command `python -m jet_mlip.deploy <family>` |
| v0.3.0 | one LAMMPS ML-IAP CUDA-graph wrapper for every family (`jet_mlip/mliap.py`: ghost fold, padded edges masked out of the convolution, input preparation inside the graph, no energy sync when LAMMPS does not need the energy); SevenNet ML-IAP route `jet_mlip.sevenn_mliap` (e3gnn stays the default); EquFlash v1 on the CUDA graph; radial-table kernel: 1/h as a hi/lo pair (removes a systematic table-index rounding bias); EquFlash table grid has a node at the XPLOR onset r_on (re-deploy EquFlash models); files deployed with v0.2.1 still load |
| v0.2.1 | SevenNet deploy options on one axis: `--radial_table N` (0 or `--no-radial_table` = exact MLP), `--exact_backward split` or `recompute` (exact only; replaces `--jet_memory`, `fastest` removed); pair_e3gnn frees the warm-up cache before the CUDA-graph capture (startup peak 7net-omni 6.1 -> 4.3 GB). Kernels as v0.2.0 |
| v0.2.0 | own kernel library `jet_mlip` (fiber fold, register-model path groups for the radial-table kernels, lmax 4); SevenNet patch; Nequix and EquFlash v1/V2 (ASE + LAMMPS ML-IAP, multi-GPU); GPU D3 `jet/d3`; MIT license |
| v0.1.0 | MACE and NequIP on the first JET kernel |

## Debugging and A/B (moved from the README, 2026-11-27)

Deploy options (add them to the `python -m jet_mlip.deploy` line):

- `--no-graph`: the official ML-IAP wrapper instead of the CUDA-graph one (slower; for comparison or debugging).
- `--no-jet`: deploy the official model unchanged, for A/B comparison. MACE: `--no-jet` is not supported; use the
official `mace_create_lammps_model <model> --format mliap` instead.

Environment variables, set in front of `lmp` (e.g. `JET_MACE_CUDA_GRAPH=0 lmp ...`):

- `JET_<FAMILY>_CUDA_GRAPH=0`: run without the CUDA graph (`<FAMILY>` = `SEVENN`, `MACE`, `NEQUIP`, `NEQUIX`, `EQUFLASH`).
- `JET_MACE_COMPILE=0`, `JET_NEQUIP_COMPILE=0`, `JET_NEQUIX_COMPILE=0`: no torch.compile (read when LAMMPS runs).
- `JET_MLIAP_CHECK=N` (debug): every N-th step and after atoms migrate, compare the step with an eager rerun and check
LAMMPS' element types; stops the run on a mismatch.
- `JET_ASE_GPU_NL=0`: the ASE calculators use their package's CPU neighbour list instead of `jet_mlip.neighbors`.
- `JET_FIRST_LAYER=0`: the generic convolution kernel for the first layer instead of the dedicated one.
- `JET_TAB_KB=1`: one block per SM for every radial-table kernel (no 2-blocks/SM variant on large graphs).
