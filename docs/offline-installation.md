# Offline installation

JET itself needs no network at run time. On a machine without internet, bring everything over from a connected
machine with the same OS and CUDA driver generation, then build there (the kernels are compiled for the local GPU).

## 1. On a connected machine

1. Get the sources: this repository (`git clone`, or a release archive) and the LAMMPS source
   (stable_22Jul2025_update6 or newer, e.g. `lammps-stable_22Jul2025_update6.tar.gz` from github.com/lammps/lammps).
2. Create and fill the env as in the README (official package, its version pins, `ninja`, `cython==3.0.11`, `cupy`).
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

3. Get the checkpoint file of every model you will deploy and copy it over; offline, `<model>` is its path.
   (SevenNet's pretrained checkpoints ship inside the `sevenn` package: their names work offline. A nequip.net id is
   a package file `<name>.nequip.zip`: download it from the model's nequip.net page, or take it from
   `~/.nequip/model_cache/` after one online deploy; the `.metadata.json` next to it names the model.)

## 2. On the offline machine

1. Unpack and activate the env:

   ```bash
   mkdir -p ~/envs/<env> && tar -xzf <env>.tar.gz -C ~/envs/<env>
   source ~/envs/<env>/bin/activate && conda-unpack
   ```

2. Build JET (README section 2, without the `git clone`):

   ```bash
   cd jet-mlips
   JET_CONFIGS=all JET_CUDA_ARCH=90 MAX_JOBS=4 pip install -e jet_mlip --no-build-isolation --no-deps
   pip install -e . --no-deps --no-build-isolation
   ```

3. Build LAMMPS (KOKKOS with CUDA, as `jet/mliap` needs) from the extracted source folder (the one holding `src/`;
   a separate copy for `SEVENN=1`).
   `OFFLINE=1` skips the pip step (cython and cupy are already in the env):

   ```bash
   tar -xzf lammps-stable_22Jul2025_update6.tar.gz
   LAMMPS_DIR=$PWD/lammps-stable_22Jul2025_update6 OFFLINE=1 bash lammps/build_lammps.sh
   ```

   The ML-IAP patches it applies are `lammps/*.patch`; one of them adds `pair_style jet/mliap` (KOKKOS only).

4. Deploy with the model file instead of a name:

   ```bash
   python -m jet_mlip.deploy mace   /path/to/mace_checkpoint.model          medium_jet.pt
   python -m jet_mlip.deploy nequip /path/to/NequIP-OAM-L-0.1.nequip.zip    oam-l.nequip.lmp.pt
   ```

5. Run (README "Run"):

   ```
   pair_style jet/mliap medium_jet.pt     # = mliap unified medium_jet.pt 0, KOKKOS only
   pair_coeff * * Li C H O N F S P B
   ```

   ```bash
   lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lammps
   ```

## 3. Updating an offline install

1. Bring the new repository over. With the env active, undo the old package patches (the recorded files) and apply
   the new ones (0.5.0 changes both):

   ```bash
   python -m jet_mlip.patch_sevenn --reverse && python -m jet_mlip.patch_sevenn          # SevenNet
   python -m common.patch_tool mace --reverse && python -m common.patch_tool mace         # MACE
   ```

2. Rebuild JET (section 2, step 2).
3. Rebuild LAMMPS (step 3). The ML-IAP tree takes the new patches in place. For `SEVENN=1` start from a freshly
   extracted LAMMPS folder: the script copies SevenNet's `pair_e3gnn` only once.
4. Deploy the models again.
