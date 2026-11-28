# Design and layout

Kernels: `jet_mlip/` (torch ops `jet_mlip::*`):
- Fused equivariant convolution with tabulated radial weights (`conv_fused`, per-edge `table_idx` for per-element-pair tables), the SevenNet, MACE, NequIP, Nequix and EquFlash shapes, and `JET_CONFIGS=all | <family list>` (one shape list per family in [jet_mlip/configs.py](../jet_mlip/configs.py), each complete on its own; the SevenNet list is literal, so envs without `sevenn` can build it; `python -m jet_mlip.shapes` checks a model).
- Non-convolution ops (v0.5.0): `species_linear` (one-hot element `skip_tp` / self-connection and shared-weight e3nn `Linear`, optional direct `ir_mul` output and modal bias table), `linear_ir_mul` (copy-free cuBLAS GEMM per `(irrep, m)` block for e3nn `Linear` with both sides in `ir_mul`), `gate` (e3nn `Gate` in one kernel per direction, `mul_ir` or `ir_mul`), `cand_build` (cell-list GPU neighbour list for `jet/e3gnn` and ASE calculators in [jet_mlip/neighbors.py](../jet_mlip/neighbors.py)), `build_csr` / `build_csr_masked`, `edge_build`, `force_reduce`.
- Op contracts: `species_linear` gives NaN output for a species outside `[0, S)` (no host check, the op stays capturable); `force_reduce` needs `0 <= nbr < n` (a violation gives that centre NaN forces); `gate` tables must be on the device of `x`; every CUDA entry sets a device guard.
- `jet_d3/`: LAMMPS `pair_style jet/d3` (`jet/d3/kk` under `-sf kk`) and ASE `JetD3Calculator` (`jet_d3/ase/`).
- `patches/sevenn-jet*.patch`: the SevenNet 0.13.0 jet patch (pointed at `jet_mlip`), applied by `python -m jet_mlip.patch_sevenn`. The sevenn D3 patch was dropped in v0.4.0: D3 is `jet/d3` for every family.

Radial-table convolution kernels (v0.5.0 additions):
- First layer (`conv_first.cu`): a scalar-only input of at most 32 channels, on graphs with more destination nodes than
  1.5 x 8 warps x SMs, runs a dedicated kernel; `JET_FIRST_LAYER=0` takes the generic kernel.
- Blocks per SM: each config gets 2 blocks/SM per direction when its register model fits (forward <= 141, backward <=
  154 registers); the 2-block kernel is used only on graphs above the same node threshold (small graphs are faster with
  1); `JET_TAB_KB=1` forces 1.
- Pair-factored fiber loop: the backward combines the fibers of one `(m1, m2)` pair (`q = sum coef * gout`) before
  multiplying by `sh`, `x` and `W` once per pair and unit; the forward computes `coef * (x * sh)`. Only the summation order
  changes.

`jet/d3` (GPU D3): C6 interpolated once per pair, BJ or zero damping; pair sums in fp64 by default (`JET_D3_FP64=0`:
float-float, for GPUs with slow fp64); energy and virial sums skipped on steps that need neither; atoms packed in
cell order. Cut-offs are the physical ones of the D3 parametrization. Under `-sf kk` it runs as `jet/d3/kk`
(`pair_jet_d3_kokkos.*`): reads positions from and adds forces to the KOKKOS device views, so a `hybrid/overlay` with
`jet/mliap` does no per-step host/device copies; packing and the MPI gather stay on the host (one local-x copy).
With `jet/e3gnn`, `pair_e3gnn` calls `jet/d3`'s `prelaunch()` so D3 runs on its own stream during the model.
Large boxes list the near tiles (32x32 atom blocks within the cut-off) on the GPU each step instead of scanning every
block pair, so the cost stays O(N) (0.81 us/atom from 343k atoms on H200); this adds one stream sync per step and up
to 8 bytes of GPU memory per tile. Small boxes keep the block-pair scan.

ASE CUDA graph ([jet_mlip/ase_graph.py](../jet_mlip/ase_graph.py), every family): `AseGraph` reuses the ML-IAP wrapper's
static buffers and capture / replay for ASE calls. The first call (and the first call with a new atom count) runs
eagerly, the next one captures; a new pair list or new elements replay the list graph, a new capacity bucket
re-captures. Rows are renumbered in spatial order when the input is not (`JET_ASE_SORT`). The total energy is summed
like the family's eager path, so eager and graph calls agree (MACE: upstream sums E0 and the interaction energy in
fp32, 1.4 eV = 2.5e-4 eV/atom below the fp64 sum on 5632 atoms; LAMMPS sums in fp64). `torch.compile` of the captured
step (Nequix, NequIP) starts 600 s after the capture (`JET_ASE_COMPILE`). Stays eager: CPU, a call that asks for
atomic virials, EquFlash V2, committee / dipole MACE models, models the ML-IAP wrapper keeps eager.

Repo layout:
- `common/`: [patch_tool.py](../common/patch_tool.py) (apply / check / reverse the shipped patches), [radial_table.py](../common/radial_table.py).
- `patches/`: one patch per upstream version ([patches/BASE](../patches/BASE)).
- `lammps/`: [build_lammps.sh](../lammps/build_lammps.sh) (ML-IAP + Kokkos CUDA + PYTHON + KSPACE + D3, one build dir per env).
- `examples/`: LAMMPS notebook and inputs in [examples/lammps/](../examples/lammps/), ASE notebook in [examples/ase/](../examples/ase/).

## SevenNet (`patches/sevenn-jet*.patch`, `jet_mlip/sevenn_mliap.py`)

Fused JET convolution with a deploy-time radial table (or `--radial_table 0` for the exact radial MLP), self-connections and `IrrepsLinear` (including multi-fidelity modal bias tables) via `jet_mlip::species_linear`, gates via `jet_mlip::gate`. In LAMMPS, `jet/e3gnn` builds its candidate and edge lists and reduces forces/virial on the GPU (`cand_build`, `edge_build`, `force_reduce`); `jet_mlip.sevenn_mliap` runs SevenNet on the shared ML-IAP CUDA-graph wrapper. In ASE, `SevenNetCalculator(..., enable_jet=True)` builds its periodic neighbour list on the GPU (`jet_mlip.neighbors`) and replays steps from `AseGraph`.

## MACE (`jet_mace`)

Fused JET convolution with a deploy-time radial table (per element pair for the Agnesi models MPA-0 / OMAT-0). Each interaction block's `Linear` (and, on non-residual first blocks, `1/avg_num_neighbors` and `skip_tp`) is folded into `jet_mlip::species_linear` writing directly in cuEq's `ir_mul` layout (no layout copy); residual `skip_tp` runs as a species-grouped Triton GEMM ([jet_mace/skip.py](../jet_mace/skip.py)); cuEq runs the symmetric contraction, with `torch.compile` over the remaining glue and one CUDA graph per MD step. Blocks with `mul > 128` (`MP-0 large` 256, `OFF23 large` 224) run as two channel slices. In ASE, patched calculators build their neighbour list on the GPU (`jet_mlip.neighbors`) and replay steps from `AseGraph` ([jet_mace/ase_graph.py](../jet_mace/ase_graph.py)). Both wrappers point torchinductor's cache at `~/.cache/torchinductor`.

## NequIP (`jet_nequip`)

The packaged model keeps its own `nequip` code; `patch_model` swaps the convolution (`tp_scatter` + `edge_mlp`) and spherical-harmonic edge attrs for the fused JET convolution with a radial table, and replaces the e3nn glue around it (self-connection, linears, gate) with `jet_mlip` ops (`species_linear`, `gate`). Blocks with `mul > 128` (`NequIP-OAM-XL` 320) run as 32-channel slices. GPU neighbour list (`jet_mlip.neighbors`) + `AseGraph` in ASE; static buckets + CUDA graph in LAMMPS (shared path, any rank count).

## Nequix (`jet_nequix`)

Swaps every `NequixTorchConvolution` for the fused JET convolution with a radial table (`mul` padded up to a multiple of 8 for the 89-species input), runs `linear_1` / `linear_2` / `skip` through `jet_mlip::species_linear`, and replaces `RMSLayerNorm`'s `Irreps` loop with precomputed slices for `torch.compile`. GPU neighbour list (`jet_mlip.neighbors`) + `AseGraph` in ASE; static buckets + CUDA graph in LAMMPS.

## EquFlash (`jet_equflash`)

Swaps `IntegratedConv` (v1) and `FullConv` (V2, with `ir_mul <-> mul_ir` block transposes) for the fused JET convolution with a radial table in 32-channel slices. In EquFlash v1, block `Linear`s and `Gate` run as `linear_ir_mul` / `gate` on features kept in `ir_mul` so each `(irrep, m)` block of a `Linear` is one copy-free cuBLAS GEMM (`JET_EQUFLASH_GLUE=0 | mul_ir | ir_mul`). GGNN is CC BY-NC-SA 4.0 and is not imported at module level (layers are matched by type name). EquFlash v1 runs on the shared CUDA-graph path in both LAMMPS and ASE (`AseGraph`, with GGNN's neighbour list); EquFlash V2 runs eager (its graph capture is invalidated).

## LAMMPS details

| option | patch added | pair styles |
|---|---|---|
| (always) | ML-IAP Kokkos: host comm fix, neighbour data, virial reduction (`lammps/*.patch`) | `jet/mliap` (= `mliap unified <file> 0`, KOKKOS only; MACE, NequIP, Nequix, EquFlash, SevenNet ML-IAP) |
| `D3=1` (default) | `jet_d3/` (GPU D3, also `custom s6 s8 a1 a2` BJ parameters) | `jet/d3`, `jet/d3/kk` (under `-sf kk`) |
| `SEVENN=1` | `sevenn patch_lammps --jet` (needs `sevenn` + `jet_mlip` in the env; own source tree) | `jet/e3gnn`, `jet/e3gnn/parallel` (= `e3gnn`, `e3gnn/parallel`) |

A binary runs the models of its own env: MACE (`e3nn==0.4.4`) needs its own env and build; SevenNet and NequIP share one
(`SEVENN=1` build in a NequIP env with `sevenn` installed). An env with its own CUDA toolkit builds with it (e.g. EquFlash's
env: conda-forge `openmpi`; `conda activate equflash; bash lammps/build_lammps.sh`, no `CUDA_HOME` from another env).

`jet/mliap`: run with `-k on g 1 -sf kk -pk kokkos newton on neigh half`. On one GPU every family (MACE, NequIP,
Nequix, EquFlash v1, SevenNet ML-IAP) runs one shared CUDA-graph path ([jet_mlip/mliap.py](../jet_mlip/mliap.py): ghosts folded onto their
owners, padded edges masked out of the convolution, one capture per size bucket, then replay; the pair-list graph is replayed only when LAMMPS hands over another pair list, another neighbour build `nbuild`, or other local elements such as under `fix atom/swap`). EquFlash V2 runs eager.
Print a thermo quantity that needs the model (`thermo_style custom step pe temp`).

Several GPUs (`mpirun -np N lmp -k on g N ...`, add `gpu/aware off` to `-pk kokkos` with a non-CUDA-aware MPI):
every `jet/mliap` family stays on the shared graph path; atoms owned by another rank are remote rows refreshed per
layer by NCCL inside the graph ([jet_mlip/mliap.py](../jet_mlip/mliap.py)). With the graph off (`JET_<FAMILY>_CUDA_GRAPH=0`), MACE, NequIP,
SevenNet and EquFlash take their official per-layer LAMMPS exchange; Nequix eager is one rank only.

Radial table size (`--radial_table N`, grid points). The defaults keep the table's force error well below the official
fp32 error (fp64 table model vs the fp64 model, max `|dF|` on a 348-atom box, as a share of the official fp32 error):
MACE 4096 (23 %; 2048 gives 180 %, 8192 gives 3 % but is 3-4 % slower with +56 % memory on MPA-0), NequIP 2048
(1.5-4 %), Nequix 2048 (1-2 %), SevenNet and EquFlash 2048, rounded down so that a grid node sits at the XPLOR cutoff
onset `r_on` (SevenNet 7net-nano 2047: ~7 %; EquFlash 2041: JET fp32 forces match the official fp32 ones).

## Deploy options, debugging and A/B

Deploy options (add them to `python -m jet_mlip.deploy <family> ...`):

- `--radial_table N`: grid points (see above). SevenNet also supports `--radial_table 0` (exact radial MLP), with
  `--exact_backward recompute` for no per-edge buffer (table-level memory, up to 8 % slower than the default `split`;
  only SevenNet has a JET exact kernel, the other families take their table-off path in reference or upstream code).
- `--modal NAME` (SevenNet): modal of a multi-fidelity model (`7net-mf-0`, `7net-omni`).
- `--head NAME` (MACE): output head of a multi-head model; `--no-cueq` (MACE): e3nn rest instead of cuEq.
- `--no-graph`: official ML-IAP wrapper instead of the CUDA-graph one (slower; for comparison or debugging).
- `--no-jet`: deploy the official model unchanged, for A/B comparison (NequIP also takes `--modifiers enable_OpenEquivariance`
  to run official kernels in the same wrapper). MACE does not support `--no-jet`; use `mace_create_lammps_model <model> --format mliap`.

Environment variables, set in front of `lmp` or Python (e.g. `JET_MACE_CUDA_GRAPH=0 lmp ...`):

- `JET_<FAMILY>_CUDA_GRAPH=0`: run LAMMPS ML-IAP without the CUDA graph (`<FAMILY>` = `SEVENN`, `MACE`, `NEQUIP`, `NEQUIX`, `EQUFLASH`).
- `JET_MLIAP_HEADROOM`: size-bucket capacity growth factor (default `1.1`; every growth re-captures and recompiles).
- `JET_MLIAP_LOG=1`: log every CUDA-graph capture (LAMMPS ML-IAP and ASE).
- `JET_MLIAP_CHECK=N` (debug): every N-th step and after atoms migrate, compare the step with an eager rerun and check
  LAMMPS' element types; stops the run on a mismatch.
- `JET_MACE_COMPILE=0`, `JET_NEQUIP_COMPILE=0`, `JET_NEQUIX_COMPILE=0`: disable `torch.compile` in LAMMPS ML-IAP (read when LAMMPS runs).
- `JET_D3_FP64=0`: float-float pair sums in `jet/d3` and `JetD3Calculator` (default `1` = fp64 sums).
- `JET_ASE_GPU_NL=0`: ASE calculators use their package's CPU neighbour list instead of `jet_mlip.neighbors`.
- `JET_ASE_GRAPH=0`: run every ASE call eagerly (no CUDA graph).
- `JET_ASE_SORT=0`: disable spatial renumbering of ASE node rows.
- `JET_ASE_COMPILE=<seconds>`: `torch.compile` the captured ASE step that long after capture (default `600`; `0` = immediately after capture; negative = never).
- `JET_EQUFLASH_GLUE=0 | mul_ir | ir_mul`: EquFlash v1 block glue in e3nn (`0`), jet ops in `mul_ir`, or jet ops in `ir_mul` (default `ir_mul`).
- `JET_FIRST_LAYER=0`: generic convolution kernel for the first layer instead of `conv_first.cu`.
- `JET_TAB_KB=1`: one block per SM for every radial-table kernel (no 2-blocks/SM variant on large graphs).
- `JET_D3_SCAN=1`: every box takes the block-pair scan (O(N^2); reference for the tile lists, debugging only).

## Tests

- `python -m pytest tests/mace_jet tests/test_*.py` (MACE env; `tests/test_jet_d3_ase.py` when `jet_d3/ase` is built, `tests/test_ase_calculators.py` for ASE calculators),
- `python -m pytest tests/nequip_jet` (NequIP env),
- `python -m pytest tests/nequix_jet` (Nequix env),
- `python -m pytest tests/equflash_jet` (EquFlash env; the LAMMPS tests need a GPU),
- `python -m pytest jet_mlip/tests` (kernel, `AseGraph`, and shared ML-IAP wrapper; the `sevenn` comparisons need a SevenNet env).

The LAMMPS ML-IAP path has its own GPU tests: `jet_mlip/tests/test_mliap.py` (shared CUDA-graph wrapper: padding,
ghost fold, capture / replay, capacity growth, element-swap rebuild), `jet_mlip/tests/test_sevenn_mliap.py`, and `test_mliap.py` in each
family's test folder (graph vs eager, vs upstream, files deployed with v0.2.1).

## Version history in detail

| version | changes |
|---|---|
| v0.5.0 | skip connections (self-connection, per-element linears) as single CUDA ops (`species_linear`) and the layers around the convolution optimized (`gate`, folded MACE linear + `skip_tp`, EquFlash v1 block linears and gates in `ir_mul` via `linear_ir_mul`, -17 %); first-layer kernel, 2 blocks/SM for small radial-table kernels on large graphs, pair-factored fiber loop (7net-omni electrolyte 66.9 -> 62.0 ms/step, EquFlash v1 -10 %), small-kernel cleanup and op device guards; `jet/d3` about 2x faster, `jet/d3/kk`, and ASE `JetD3Calculator`; pair styles `jet/mliap`, `jet/e3gnn`, and ML-IAP pair-list element check (`fix atom/swap`); ASE neighbour list on the GPU and ASE CUDA-graph replay (`jet_mlip/ase_graph.py`) for every family; examples in `examples/` (notebooks); license CC BY-NC-SA 4.0 (LAMMPS-derived files GPL-2.0) |
| v0.3.1 | shared ML-IAP wrapper: padded edges spread over the local rows (sorted last in the masked CSR), so scatters outside the JET convolution (upstream convolutions, MACE density, NequIP ZBL) no longer serialize on one row (Nequix `--no-jet` 348 atoms: 50.8 -> 32.9 ms/step); kernels as v0.3.0; one deploy command `python -m jet_mlip.deploy <family>` |
| v0.3.0 | one LAMMPS ML-IAP CUDA-graph wrapper for every family (`jet_mlip/mliap.py`: ghost fold, padded edges masked out of the convolution, input preparation inside the graph, no energy sync when LAMMPS does not need the energy); SevenNet ML-IAP route `jet_mlip.sevenn_mliap` (`e3gnn` stays the default); EquFlash v1 on the CUDA graph; radial-table kernel: `1/h` as a hi/lo pair (removes a systematic table-index rounding bias); EquFlash table grid has a node at the XPLOR onset `r_on` (re-deploy EquFlash models); files deployed with v0.2.1 still load |
| v0.2.1 | SevenNet deploy options on one axis: `--radial_table N` (`0` or `--no-radial_table` = exact MLP), `--exact_backward split` or `recompute` (exact only; replaces `--jet_memory`, `fastest` removed); `pair_e3gnn` frees the warm-up cache before the CUDA-graph capture (startup peak 7net-omni 6.1 -> 4.3 GB). Kernels as v0.2.0 |
| v0.2.0 | own kernel library `jet_mlip` (fiber fold, register-model path groups for the radial-table kernels, lmax 4); SevenNet patch; Nequix and EquFlash v1/V2 (ASE + LAMMPS ML-IAP, multi-GPU); GPU D3 `jet/d3`; MIT license |
| v0.1.0 | MACE and NequIP on the first JET kernel |
