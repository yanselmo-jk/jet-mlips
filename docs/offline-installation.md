# Offline installation

JET itself needs no network at run time. On a machine without internet, bring everything over from a connected
machine with the same OS and CUDA driver generation, then build there (the kernels are compiled for the local GPU).

## 1. On a connected machine

1. Get the sources: this repository (`git clone`, or a release archive) and the LAMMPS source
   (`stable_22Jul2025_update6` or newer, e.g. `lammps-stable_22Jul2025_update6.tar.gz` from [github.com/lammps/lammps](https://github.com/lammps/lammps)).
2. Create and fill the env as in [README.md](../README.md) (official package, its version pins, `ninja`, `cython==3.0.11`, and `cupy-cuda12x` — or `cupy-cuda13x` on CUDA 13).
   Only for SevenNet or MACE: with the env activated, patch that package from the repository folder (they call JET
   from inside the package; NequIP, Nequix and EquFlash need no patch). Then pack the env:

   ```bash
   conda activate <env> && cd jet-mlips
   python -m jet_mlip.patch_sevenn      # only if you use SevenNet
   python -m common.patch_tool mace     # only if you use MACE
   conda deactivate
   conda install -n <env> -c conda-forge conda-pack
   conda pack -n <env> -o <env>.tar.gz
   ```

3. Get the checkpoint file of every model you will deploy and copy it over; offline, `<model>` is its path:
   - **SevenNet**: `7net-0`, `7net-mf-0` and `7net-omni` ship inside the `sevenn` package (their names work offline); `7net-nano` needs its checkpoint file (`checkpoint_7net_nano_6.0.pth`).
   - **MACE**: `.model` files cached under `~/.cache/mace/` after one online load/deploy (or downloaded from the MACE releases).
   - **NequIP**: a nequip.net id is a package file `<name>.nequip.zip` — download it from the model's nequip.net page, or take it from `~/.nequip/model_cache/` after one online deploy (the `.metadata.json` next to it names the model).
   - **Nequix**: `nequix-mp-1.pt` / `nequix-oam-1.pt` from `~/.cache/nequix/models/` (downloaded by the `nequix` package).
   - **EquFlash**: `equflash_oam_30M.pt` / `equflashv2_oam.pt` from the GGNN release.

## 2. On the offline machine

1. Unpack and activate the env:

   ```bash
   mkdir -p ~/envs/<env> && tar -xzf <env>.tar.gz -C ~/envs/<env>
   source ~/envs/<env>/bin/activate && conda-unpack
   ```

2. Build JET ([README.md](../README.md) section 2, without `git clone` or re-patching):

   ```bash
   cd jet-mlips
   JET_CONFIGS=all JET_CUDA_ARCH=90 MAX_JOBS=4 pip install -e jet_mlip --no-build-isolation --no-deps
   pip install -e . --no-deps --no-build-isolation
   JET_CUDA_ARCH=90 pip install -e jet_d3/ase --no-build-isolation --no-deps   # only for D3 in ASE
   ```

3. Build LAMMPS (KOKKOS with CUDA, as `jet/mliap` needs) from the extracted source folder (the one holding `src/`;
   use a separate copy for `SEVENN=1`).
   `OFFLINE=1` skips the pip step (`cython` and `cupy` are already in the env):

   ```bash
   tar -xzf lammps-stable_22Jul2025_update6.tar.gz
   LAMMPS_DIR=$PWD/lammps-stable_22Jul2025_update6 OFFLINE=1 bash lammps/build_lammps.sh
   ```

   The ML-IAP patches it applies are `lammps/*.patch`; one of them adds `pair_style jet/mliap` (KOKKOS only). It also
   builds `pair_style jet/d3`, which runs as `jet/d3/kk` under `-sf kk`.

4. Deploy with the model file instead of a name:

   ```bash
   python -m jet_mlip.deploy mace   /path/to/mace_checkpoint.model          medium_jet.pt
   python -m jet_mlip.deploy nequip /path/to/NequIP-OAM-L-0.1.nequip.zip    oam-l.nequip.lmp.pt
   ```

   Every supported model with its deploy line: [examples/lammps/lammps_md.ipynb](../examples/lammps/lammps_md.ipynb) (replace names with local files; for SevenNet `jet/e3gnn`, use `sevenn get_model <model> --enable_jet -o model.pt`).

5. Run ([README.md](../README.md) section 4):

   ```lammps
   pair_style jet/mliap medium_jet.pt     # = mliap unified medium_jet.pt 0, KOKKOS only
   pair_coeff * * Li C H O N F S P B
   ```

   ```bash
   lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lammps
   ```

   Example inputs and notebooks: [examples/lammps/](../examples/lammps/), [examples/ase/ase_md.ipynb](../examples/ase/ase_md.ipynb). D3 on a consumer GeForce card: `JET_D3_FP64=0 lmp ...`
   ([README.md](../README.md), D3 dispersion).

## 3. Updating an offline install

1. Bring the new repository over. With the env active, undo the old package patches (the recorded files) and apply
   the new ones from the new repository folder (v0.5.0 changes both):

   ```bash
   python -m jet_mlip.patch_sevenn --reverse && python -m jet_mlip.patch_sevenn          # SevenNet env only
   python -m common.patch_tool mace --reverse && python -m common.patch_tool mace         # MACE env only
   ```

2. Rebuild JET (section 2, step 2, including `jet_d3/ase` if used).
3. Rebuild LAMMPS (section 2, step 3). The ML-IAP tree takes the new patches in place. For `SEVENN=1` the script copies
   the installed sevenn's `pair_e3gnn` sources again when they changed; a tree it cannot update stops with a message
   to start from a freshly extracted LAMMPS folder.
4. Deploy the models again.
