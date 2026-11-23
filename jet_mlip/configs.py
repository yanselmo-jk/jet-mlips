"""Configs compiled into jet_mlip._C: one list per family (sevenn, mace, nequip, nequix, equflash) plus the shapes in
user_configs(). JET_CONFIGS=all (default) builds every family, JET_CONFIGS=mace,nequip only those families; both add the
user shapes. Run this file to regenerate csrc/configs_gen.h (`python -m jet_mlip.shapes <family> <model> --apply` adds shapes)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jet_mlip import tables  # noqa: E402


def mace_configs():
    """JET_CONFIGS=mace: the conv shapes of the supported MACE models (MP medium / small / 0b3, MATPES, OFF23 small /
    medium / large). Every family list is complete on its
    own: shapes shared with another family are listed again (all-builds drop the duplicates)."""
    cfgs = []
    cfgs += [tables.make_config('128x0e', '1x0e+1x1o+1x2e+1x3o', '128x0e+128x1o+128x2e+128x3o', [(0, 0, 0, 'uvu', True), (0, 1, 1, 'uvu', True), (0, 2, 2, 'uvu', True), (0, 3, 3, 'uvu', True)])]   # mace medium L1
    cfgs += [tables.make_config('128x0e+128x1o', '1x0e+1x1o+1x2e+1x3o', '128x0e+128x0e+128x1o+128x1o+128x1o+128x2e+128x2e+128x2e+128x3o+128x3o', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (0, 1, 2, 'uvu', True), (1, 0, 3, 'uvu', True), (1, 2, 4, 'uvu', True), (0, 2, 5, 'uvu', True), (1, 1, 6, 'uvu', True), (1, 3, 7, 'uvu', True), (0, 3, 8, 'uvu', True), (1, 2, 9, 'uvu', True)])]   # mace medium L2
    cfgs += [tables.make_config('112x0e', '1x0e+1x1o+1x2e+1x3o', '112x0e+112x1o+112x2e+112x3o', [(0, 0, 0, 'uvu', True), (0, 1, 1, 'uvu', True), (0, 2, 2, 'uvu', True), (0, 3, 3, 'uvu', True)])]   # MACE-OFF23 large L1 as 2 x mul-112 slices
    cfgs += [tables.make_config('112x0e+112x1o+112x2e', '1x0e+1x1o+1x2e+1x3o', '112x0e+112x0e+112x0e+112x1o+112x1o+112x1o+112x1o+112x1o+112x2e+112x2e+112x2e+112x2e+112x2e+112x3o+112x3o+112x3o+112x3o', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (0, 1, 3, 'uvu', True), (1, 0, 4, 'uvu', True), (1, 2, 5, 'uvu', True), (2, 1, 6, 'uvu', True), (2, 3, 7, 'uvu', True), (0, 2, 8, 'uvu', True), (1, 1, 9, 'uvu', True), (1, 3, 10, 'uvu', True), (2, 0, 11, 'uvu', True), (2, 2, 12, 'uvu', True), (0, 3, 13, 'uvu', True), (1, 2, 14, 'uvu', True), (2, 1, 15, 'uvu', True), (2, 3, 16, 'uvu', True)])]   # MACE-OFF23 large L2 as 2 x mul-112 slices
    cfgs += [tables.make_config('96x0e', '1x0e+1x1o+1x2e+1x3o', '96x0e+96x1o+96x2e+96x3o', [(0, 0, 0, 'uvu', True), (0, 1, 1, 'uvu', True), (0, 2, 2, 'uvu', True), (0, 3, 3, 'uvu', True)])]   # off-small L0
    return _unique(cfgs)


def nequip_configs():
    """JET_CONFIGS=nequip: the conv shapes of NequIP-OAM-S / M / L / XL 0.1 and MP-L (nequip.net; XL as 32-channel slices),
    instructions sorted by i_out (jet_nequip.convert permutes the weight columns to match)."""
    cfgs = []
    cfgs += [tables.make_config('32x0e', '1x0e+1x1e', '32x0e+32x1e', [(0, 0, 0, 'uvu', True), (0, 1, 1, 'uvu', True)])]   # NequIP-OAM-S layer0 NW 64
    cfgs += [tables.make_config('128x0e+64x1e', '1x0e+1x1e', '128x0e+64x0e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True)])]   # NequIP-OAM-S layer1 NW 192
    cfgs += [tables.make_config('32x0e', '1x0e+1x1o+1x2e', '32x0e+32x1o+32x2e', [(0, 0, 0, 'uvu', True), (0, 1, 1, 'uvu', True), (0, 2, 2, 'uvu', True)])]   # NequIP-OAM-M layer0 NW 96
    cfgs += [tables.make_config('128x0e+64x1o+32x2e', '1x0e+1x1o+1x2e', '128x0e+64x0e+32x0e+128x1o+64x1o+64x1o+32x1o+128x2e+64x2e+32x2e+32x2e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (0, 1, 3, 'uvu', True), (1, 0, 4, 'uvu', True), (1, 2, 5, 'uvu', True), (2, 1, 6, 'uvu', True), (0, 2, 7, 'uvu', True), (1, 1, 8, 'uvu', True), (2, 0, 9, 'uvu', True), (2, 2, 10, 'uvu', True)])]   # NequIP-OAM-M layer1 NW 768
    cfgs += [tables.make_config('128x0e+64x1o+32x2e', '1x0e+1x1o+1x2e', '128x0e+64x0e+32x0e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True)])]   # NequIP-OAM-M layer3 NW 224
    cfgs += [tables.make_config('48x0e', '1x0e+1x1o+1x2e+1x3o', '48x0e+48x1o+48x2e+48x3o', [(0, 0, 0, 'uvu', True), (0, 1, 1, 'uvu', True), (0, 2, 2, 'uvu', True), (0, 3, 3, 'uvu', True)])]   # NequIP-OAM-L layer0 NW 192
    cfgs += [tables.make_config('128x0e+64x1o+32x2e+32x3o', '1x0e+1x1o+1x2e+1x3o', '128x0e+64x0e+32x0e+32x0e+128x1o+64x1o+64x1o+32x1o+32x1o+32x1o+128x2e+64x2e+64x2e+32x2e+32x2e+32x2e+32x2e+128x3o+64x3o+32x3o+32x3o+32x3o+32x3o', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (3, 3, 3, 'uvu', True), (0, 1, 4, 'uvu', True), (1, 0, 5, 'uvu', True), (1, 2, 6, 'uvu', True), (2, 1, 7, 'uvu', True), (2, 3, 8, 'uvu', True), (3, 2, 9, 'uvu', True), (0, 2, 10, 'uvu', True), (1, 1, 11, 'uvu', True), (1, 3, 12, 'uvu', True), (2, 0, 13, 'uvu', True), (2, 2, 14, 'uvu', True), (3, 1, 15, 'uvu', True), (3, 3, 16, 'uvu', True), (0, 3, 17, 'uvu', True), (1, 2, 18, 'uvu', True), (2, 1, 19, 'uvu', True), (2, 3, 20, 'uvu', True), (3, 0, 21, 'uvu', True), (3, 2, 22, 'uvu', True)])]   # NequIP-OAM-L layer1 NW 1312
    cfgs += [tables.make_config('128x0e+64x1o+32x2e+32x3o', '1x0e+1x1o+1x2e+1x3o', '128x0e+64x0e+32x0e+32x0e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (3, 3, 3, 'uvu', True)])]   # NequIP-OAM-L layer5 NW 256
    cfgs += [tables.make_config('32x0e+32x1o+32x2e', '1x0e+1x1o+1x2e+1x3o+1x4e', '32x0e+32x0e+32x0e+32x1o+32x1o+32x1o+32x1o+32x1o+32x2e+32x2e+32x2e+32x2e+32x2e+32x2e+32x3o+32x3o+32x3o+32x3o+32x3o+32x4e+32x4e+32x4e+32x4e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (0, 1, 3, 'uvu', True), (1, 0, 4, 'uvu', True), (1, 2, 5, 'uvu', True), (2, 1, 6, 'uvu', True), (2, 3, 7, 'uvu', True), (0, 2, 8, 'uvu', True), (1, 1, 9, 'uvu', True), (1, 3, 10, 'uvu', True), (2, 0, 11, 'uvu', True), (2, 2, 12, 'uvu', True), (2, 4, 13, 'uvu', True), (0, 3, 14, 'uvu', True), (1, 2, 15, 'uvu', True), (1, 4, 16, 'uvu', True), (2, 1, 17, 'uvu', True), (2, 3, 18, 'uvu', True), (0, 4, 19, 'uvu', True), (1, 3, 20, 'uvu', True), (2, 2, 21, 'uvu', True), (2, 4, 22, 'uvu', True)])]   # NequIP-OAM-XL layer1 slice1 NW 736
    cfgs += [tables.make_config('32x0e+32x1o', '1x0e+1x1o+1x2e+1x3o+1x4e', '32x0e+32x0e+32x1o+32x1o+32x1o+32x2e+32x2e+32x2e+32x3o+32x3o+32x3o+32x4e+32x4e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (0, 1, 2, 'uvu', True), (1, 0, 3, 'uvu', True), (1, 2, 4, 'uvu', True), (0, 2, 5, 'uvu', True), (1, 1, 6, 'uvu', True), (1, 3, 7, 'uvu', True), (0, 3, 8, 'uvu', True), (1, 2, 9, 'uvu', True), (1, 4, 10, 'uvu', True), (0, 4, 11, 'uvu', True), (1, 3, 12, 'uvu', True)])]   # NequIP-OAM-XL layer1 slice2 NW 416
    cfgs += [tables.make_config('32x0e+32x1o+32x2e', '1x0e+1x1o+1x2e+1x3o+1x4e', '32x0e+32x0e+32x0e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True)])]   # NequIP-OAM-XL layer5 slice1 NW 96
    cfgs += [tables.make_config('32x0e+32x1o', '1x0e+1x1o+1x2e+1x3o+1x4e', '32x0e+32x0e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True)])]   # NequIP-OAM-XL layer5 slice2 NW 64
    cfgs += [tables.make_config('32x0e', '1x0e+1x1o+1x2e+1x3o+1x4e', '32x0e', [(0, 0, 0, 'uvu', True)])]   # NequIP-OAM-XL layer5 slice3 NW 32
    cfgs += [tables.make_config('32x0e', '1x0e+1x1o+1x2e+1x3o+1x4e', '32x0e+32x1o+32x2e+32x3o+32x4e', [(0, 0, 0, 'uvu', True), (0, 1, 1, 'uvu', True), (0, 2, 2, 'uvu', True), (0, 3, 3, 'uvu', True), (0, 4, 4, 'uvu', True)])]   # NequIP-OAM-XL:0.1 L0
    cfgs += [tables.make_config('32x0e+32x1o+32x2e+32x3o+32x4e', '1x0e+1x1o+1x2e+1x3o+1x4e', '32x0e+32x0e+32x0e+32x0e+32x0e+32x1o+32x1o+32x1o+32x1o+32x1o+32x1o+32x1o+32x1o+32x2e+32x2e+32x2e+32x2e+32x2e+32x2e+32x2e+32x2e+32x2e+32x2e+32x3o+32x3o+32x3o+32x3o+32x3o+32x3o+32x3o+32x3o+32x3o+32x3o+32x4e+32x4e+32x4e+32x4e+32x4e+32x4e+32x4e+32x4e+32x4e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (3, 3, 3, 'uvu', True), (4, 4, 4, 'uvu', True), (0, 1, 5, 'uvu', True), (1, 0, 6, 'uvu', True), (1, 2, 7, 'uvu', True), (2, 1, 8, 'uvu', True), (2, 3, 9, 'uvu', True), (3, 2, 10, 'uvu', True), (3, 4, 11, 'uvu', True), (4, 3, 12, 'uvu', True), (0, 2, 13, 'uvu', True), (1, 1, 14, 'uvu', True), (1, 3, 15, 'uvu', True), (2, 0, 16, 'uvu', True), (2, 2, 17, 'uvu', True), (2, 4, 18, 'uvu', True), (3, 1, 19, 'uvu', True), (3, 3, 20, 'uvu', True), (4, 2, 21, 'uvu', True), (4, 4, 22, 'uvu', True), (0, 3, 23, 'uvu', True), (1, 2, 24, 'uvu', True), (1, 4, 25, 'uvu', True), (2, 1, 26, 'uvu', True), (2, 3, 27, 'uvu', True), (3, 0, 28, 'uvu', True), (3, 2, 29, 'uvu', True), (3, 4, 30, 'uvu', True), (4, 1, 31, 'uvu', True), (4, 3, 32, 'uvu', True), (0, 4, 33, 'uvu', True), (1, 3, 34, 'uvu', True), (2, 2, 35, 'uvu', True), (2, 4, 36, 'uvu', True), (3, 1, 37, 'uvu', True), (3, 3, 38, 'uvu', True), (4, 0, 39, 'uvu', True), (4, 2, 40, 'uvu', True), (4, 4, 41, 'uvu', True)])]   # NequIP-OAM-XL:0.1 L1
    cfgs += [tables.make_config('32x0e+32x1o+32x2e+32x3o+32x4e', '1x0e+1x1o+1x2e+1x3o+1x4e', '32x0e+32x0e+32x0e+32x0e+32x0e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (3, 3, 3, 'uvu', True), (4, 4, 4, 'uvu', True)])]   # NequIP-OAM-XL:0.1 L41
    return _unique(cfgs)


def nequix_configs():
    """JET_CONFIGS=nequix: the conv shapes of nequix-mp-1 / nequix-oam-1 (torch implementation; both are
    128x0e+64x1o+32x2e+32x3o, lmax 3, 4 layers). Instructions already in sorted-i_out order (Nequix permutes its TP
    instructions that way), so jet_nequix.convert keeps the weight columns as they are."""
    cfgs = []
    cfgs += [tables.make_config('96x0e', '1x0e+1x1o+1x2e+1x3o', '96x0e+96x1o+96x2e+96x3o', [(0, 0, 0, 'uvu', True), (0, 1, 1, 'uvu', True), (0, 2, 2, 'uvu', True), (0, 3, 3, 'uvu', True)])]   # nequix-mp-1 / oam-1 layer0 (89 species one-hot padded to 96) NW 384
    cfgs += [tables.make_config('128x0e+64x1o+32x2e+32x3o', '1x0e+1x1o+1x2e+1x3o', '128x0e+64x0e+32x0e+32x0e+128x1o+64x1o+64x1o+32x1o+32x1o+32x1o+128x2e+64x2e+64x2e+32x2e+32x2e+32x2e+32x2e+128x3o+64x3o+32x3o+32x3o+32x3o+32x3o', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (3, 3, 3, 'uvu', True), (0, 1, 4, 'uvu', True), (1, 0, 5, 'uvu', True), (1, 2, 6, 'uvu', True), (2, 1, 7, 'uvu', True), (2, 3, 8, 'uvu', True), (3, 2, 9, 'uvu', True), (0, 2, 10, 'uvu', True), (1, 1, 11, 'uvu', True), (1, 3, 12, 'uvu', True), (2, 0, 13, 'uvu', True), (2, 2, 14, 'uvu', True), (3, 1, 15, 'uvu', True), (3, 3, 16, 'uvu', True), (0, 3, 17, 'uvu', True), (1, 2, 18, 'uvu', True), (2, 1, 19, 'uvu', True), (2, 3, 20, 'uvu', True), (3, 0, 21, 'uvu', True), (3, 2, 22, 'uvu', True)])]   # nequix-mp-1 / oam-1 layer1 NW 1312
    cfgs += [tables.make_config('128x0e+64x1o+32x2e+32x3o', '1x0e+1x1o+1x2e+1x3o', '128x0e+64x0e+32x0e+32x0e+128x1o+64x1o+64x1o+32x1o+32x1o+32x1o+128x2e+64x2e+64x2e+32x2e+32x2e+32x2e+32x2e+128x3o+64x3o+32x3o+32x3o+32x3o+32x3o', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (3, 3, 3, 'uvu', True), (0, 1, 4, 'uvu', True), (1, 0, 5, 'uvu', True), (1, 2, 6, 'uvu', True), (2, 1, 7, 'uvu', True), (2, 3, 8, 'uvu', True), (3, 2, 9, 'uvu', True), (0, 2, 10, 'uvu', True), (1, 1, 11, 'uvu', True), (1, 3, 12, 'uvu', True), (2, 0, 13, 'uvu', True), (2, 2, 14, 'uvu', True), (3, 1, 15, 'uvu', True), (3, 3, 16, 'uvu', True), (0, 3, 17, 'uvu', True), (1, 2, 18, 'uvu', True), (2, 1, 19, 'uvu', True), (2, 3, 20, 'uvu', True), (3, 0, 21, 'uvu', True), (3, 2, 22, 'uvu', True)])]   # nequix-mp-1 / oam-1 layer2 NW 1312
    cfgs += [tables.make_config('128x0e+64x1o+32x2e+32x3o', '1x0e+1x1o+1x2e+1x3o', '128x0e+64x0e+32x0e+32x0e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (3, 3, 3, 'uvu', True)])]   # nequix-mp-1 / oam-1 layer3 NW 256
    return _unique(cfgs)


def equflash_configs():
    """JET_CONFIGS=equflash: the conv shapes of EquFlash-OAM-30M (channel 256, lmax 3) and EquFlashV2-OAM (channel 128,
    lmax 4), 10 layers each, as mul-32 channel slices (jet_equflash.convert.SLICE: 8 x mul 32 is 2.8x faster than
    2 x mul 128 per layer for v1). Instructions sorted
    by i_out; the cueq channelwise descriptor uses that same weight order (probed on the GPU), so no permutation."""
    shapes = [
        ('128x0e', '1x0e+1x1o+1x2e+1x3o', '128x0e+128x1o+128x2e+128x3o', [(0, 0, 0, 'uvu', True), (0, 1, 1, 'uvu', True), (0, 2, 2, 'uvu', True), (0, 3, 3, 'uvu', True)]),   # layer0 (4 paths)
        ('128x0e+128x1o+128x2e+128x3o', '1x0e+1x1o+1x2e+1x3o', '128x0e+128x0e+128x0e+128x0e+128x1o+128x1o+128x1o+128x1o+128x1o+128x1o+128x2e+128x2e+128x2e+128x2e+128x2e+128x2e+128x2e+128x3o+128x3o+128x3o+128x3o+128x3o+128x3o', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (3, 3, 3, 'uvu', True), (0, 1, 4, 'uvu', True), (1, 0, 5, 'uvu', True), (1, 2, 6, 'uvu', True), (2, 1, 7, 'uvu', True), (2, 3, 8, 'uvu', True), (3, 2, 9, 'uvu', True), (0, 2, 10, 'uvu', True), (1, 1, 11, 'uvu', True), (1, 3, 12, 'uvu', True), (2, 0, 13, 'uvu', True), (2, 2, 14, 'uvu', True), (3, 1, 15, 'uvu', True), (3, 3, 16, 'uvu', True), (0, 3, 17, 'uvu', True), (1, 2, 18, 'uvu', True), (2, 1, 19, 'uvu', True), (2, 3, 20, 'uvu', True), (3, 0, 21, 'uvu', True), (3, 2, 22, 'uvu', True)]),   # layers 1-8 (23 paths)
        ('128x0e+128x1o+128x2e+128x3o', '1x0e+1x1o+1x2e+1x3o', '128x0e+128x0e+128x0e+128x0e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (3, 3, 3, 'uvu', True)]),   # layer9 (4 paths, scalars)
        ('128x0e', '1x0e+1x1o+1x2e+1x3o+1x4e', '128x0e+128x1o+128x2e+128x3o+128x4e', [(0, 0, 0, 'uvu', True), (0, 1, 1, 'uvu', True), (0, 2, 2, 'uvu', True), (0, 3, 3, 'uvu', True), (0, 4, 4, 'uvu', True)]),   # EquFlashV2 layer 0 (5 paths)
        ('128x0e+128x1o+128x2e+128x3o+128x4e', '1x0e+1x1o+1x2e+1x3o+1x4e', '128x0e+128x0e+128x0e+128x0e+128x0e+128x1o+128x1o+128x1o+128x1o+128x1o+128x1o+128x1o+128x1o+128x2e+128x2e+128x2e+128x2e+128x2e+128x2e+128x2e+128x2e+128x2e+128x2e+128x3o+128x3o+128x3o+128x3o+128x3o+128x3o+128x3o+128x3o+128x3o+128x3o+128x4e+128x4e+128x4e+128x4e+128x4e+128x4e+128x4e+128x4e+128x4e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (3, 3, 3, 'uvu', True), (4, 4, 4, 'uvu', True), (0, 1, 5, 'uvu', True), (1, 0, 6, 'uvu', True), (1, 2, 7, 'uvu', True), (2, 1, 8, 'uvu', True), (2, 3, 9, 'uvu', True), (3, 2, 10, 'uvu', True), (3, 4, 11, 'uvu', True), (4, 3, 12, 'uvu', True), (0, 2, 13, 'uvu', True), (1, 1, 14, 'uvu', True), (1, 3, 15, 'uvu', True), (2, 0, 16, 'uvu', True), (2, 2, 17, 'uvu', True), (2, 4, 18, 'uvu', True), (3, 1, 19, 'uvu', True), (3, 3, 20, 'uvu', True), (4, 2, 21, 'uvu', True), (4, 4, 22, 'uvu', True), (0, 3, 23, 'uvu', True), (1, 2, 24, 'uvu', True), (1, 4, 25, 'uvu', True), (2, 1, 26, 'uvu', True), (2, 3, 27, 'uvu', True), (3, 0, 28, 'uvu', True), (3, 2, 29, 'uvu', True), (3, 4, 30, 'uvu', True), (4, 1, 31, 'uvu', True), (4, 3, 32, 'uvu', True), (0, 4, 33, 'uvu', True), (1, 3, 34, 'uvu', True), (2, 2, 35, 'uvu', True), (2, 4, 36, 'uvu', True), (3, 1, 37, 'uvu', True), (3, 3, 38, 'uvu', True), (4, 0, 39, 'uvu', True), (4, 2, 40, 'uvu', True), (4, 4, 41, 'uvu', True)]),   # EquFlashV2 layer 1 (42 paths)
        ('128x0e+128x1o+128x2e+128x3o+128x4e', '1x0e+1x1o+1x2e+1x3o+1x4e', '128x0e+128x0e+128x0e+128x0e+128x0e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (3, 3, 3, 'uvu', True), (4, 4, 4, 'uvu', True)]),   # EquFlashV2 layer 9 (5 paths)
    ]
    return _unique([tables.make_config(*[s.replace('128x', '32x') for s in a[:3]], a[3]) for a in shapes])


def sevenn_configs():
    """The official SevenNet conv shapes (7net-nano, 7net-0 / mf-0, 7net-omni) as literal lines dumped with sevenn
    0.13.0 (keys checked equal), so an env without sevenn can build them."""
    cfgs = []
    cfgs += [tables.make_config('32x0e', '1x0e+1x1o+1x2e', '32x0e+32x1o+32x2e', [(0, 0, 0, 'uvu', True), (0, 1, 1, 'uvu', True), (0, 2, 2, 'uvu', True)])]
    cfgs += [tables.make_config('32x0e+32x1o+32x2e', '1x0e+1x1o+1x2e', '32x0e+32x0e+32x0e+32x1o+32x1o+32x1o+32x1o+32x1e+32x1e+32x2o+32x2o+32x2e+32x2e+32x2e+32x2e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (0, 1, 3, 'uvu', True), (1, 0, 4, 'uvu', True), (1, 2, 5, 'uvu', True), (2, 1, 6, 'uvu', True), (1, 1, 7, 'uvu', True), (2, 2, 8, 'uvu', True), (1, 2, 9, 'uvu', True), (2, 1, 10, 'uvu', True), (0, 2, 11, 'uvu', True), (1, 1, 12, 'uvu', True), (2, 0, 13, 'uvu', True), (2, 2, 14, 'uvu', True)], '1x0e+1x0e+1x0e+1x1o+1x1o+1x1o+1x1o+1x2e+1x2e+1x2e+1x2e')]
    cfgs += [tables.make_config('32x0e+32x1o+32x2e', '1x0e+1x1o+1x2e', '32x0e+32x0e+32x0e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True)])]
    cfgs += [tables.make_config('128x0e', '1x0e+1x1e+1x2e', '128x0e+128x1e+128x2e', [(0, 0, 0, 'uvu', True), (0, 1, 1, 'uvu', True), (0, 2, 2, 'uvu', True)])]
    cfgs += [tables.make_config('128x0e+64x1e+32x2e', '1x0e+1x1e+1x2e', '128x0e+64x0e+32x0e+128x1e+64x1e+64x1e+64x1e+32x1e+32x1e+128x2e+64x2e+64x2e+32x2e+32x2e+32x2e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (0, 1, 3, 'uvu', True), (1, 0, 4, 'uvu', True), (1, 1, 5, 'uvu', True), (1, 2, 6, 'uvu', True), (2, 1, 7, 'uvu', True), (2, 2, 8, 'uvu', True), (0, 2, 9, 'uvu', True), (1, 1, 10, 'uvu', True), (1, 2, 11, 'uvu', True), (2, 0, 12, 'uvu', True), (2, 1, 13, 'uvu', True), (2, 2, 14, 'uvu', True)])]
    cfgs += [tables.make_config('128x0e+64x1e+32x2e', '1x0e+1x1e+1x2e', '128x0e+64x0e+32x0e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True)])]
    cfgs += [tables.make_config('128x0e', '1x0e+1x1o+1x2e+1x3o', '128x0e+128x1o+128x2e+128x3o', [(0, 0, 0, 'uvu', True), (0, 1, 1, 'uvu', True), (0, 2, 2, 'uvu', True), (0, 3, 3, 'uvu', True)])]
    cfgs += [tables.make_config('128x0e+64x1o+32x2e+32x3o', '1x0e+1x1o+1x2e+1x3o', '128x0e+64x0e+32x0e+32x0e+128x1o+64x1o+64x1o+32x1o+32x1o+32x1o+64x1e+32x1e+32x1e+64x2o+32x2o+32x2o+32x2o+128x2e+64x2e+64x2e+32x2e+32x2e+32x2e+32x2e+128x3o+64x3o+32x3o+32x3o+32x3o+32x3o+64x3e+32x3e+32x3e+32x3e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (3, 3, 3, 'uvu', True), (0, 1, 4, 'uvu', True), (1, 0, 5, 'uvu', True), (1, 2, 6, 'uvu', True), (2, 1, 7, 'uvu', True), (2, 3, 8, 'uvu', True), (3, 2, 9, 'uvu', True), (1, 1, 10, 'uvu', True), (2, 2, 11, 'uvu', True), (3, 3, 12, 'uvu', True), (1, 2, 13, 'uvu', True), (2, 1, 14, 'uvu', True), (2, 3, 15, 'uvu', True), (3, 2, 16, 'uvu', True), (0, 2, 17, 'uvu', True), (1, 1, 18, 'uvu', True), (1, 3, 19, 'uvu', True), (2, 0, 20, 'uvu', True), (2, 2, 21, 'uvu', True), (3, 1, 22, 'uvu', True), (3, 3, 23, 'uvu', True), (0, 3, 24, 'uvu', True), (1, 2, 25, 'uvu', True), (2, 1, 26, 'uvu', True), (2, 3, 27, 'uvu', True), (3, 0, 28, 'uvu', True), (3, 2, 29, 'uvu', True), (1, 3, 30, 'uvu', True), (2, 2, 31, 'uvu', True), (3, 1, 32, 'uvu', True), (3, 3, 33, 'uvu', True)])]
    cfgs += [tables.make_config('128x0e+64x1o+64x1e+32x2o+32x2e+32x3o+32x3e', '1x0e+1x1o+1x2e+1x3o', '64x0o+32x0o+32x0o+128x0e+64x0e+32x0e+32x0e+128x1o+64x1o+64x1o+64x1o+32x1o+32x1o+32x1o+32x1o+32x1o+64x1e+64x1e+64x1e+32x1e+32x1e+32x1e+32x1e+32x1e+64x2o+64x2o+64x2o+32x2o+32x2o+32x2o+32x2o+32x2o+32x2o+32x2o+128x2e+64x2e+64x2e+64x2e+32x2e+32x2e+32x2e+32x2e+32x2e+32x2e+32x2e+128x3o+64x3o+64x3o+32x3o+32x3o+32x3o+32x3o+32x3o+32x3o+32x3o+64x3e+64x3e+32x3e+32x3e+32x3e+32x3e+32x3e+32x3e+32x3e', [(2, 1, 0, 'uvu', True), (3, 2, 1, 'uvu', True), (6, 3, 2, 'uvu', True), (0, 0, 3, 'uvu', True), (1, 1, 4, 'uvu', True), (4, 2, 5, 'uvu', True), (5, 3, 6, 'uvu', True), (0, 1, 7, 'uvu', True), (1, 0, 8, 'uvu', True), (1, 2, 9, 'uvu', True), (2, 1, 10, 'uvu', True), (3, 2, 11, 'uvu', True), (4, 1, 12, 'uvu', True), (4, 3, 13, 'uvu', True), (5, 2, 14, 'uvu', True), (6, 3, 15, 'uvu', True), (1, 1, 16, 'uvu', True), (2, 0, 17, 'uvu', True), (2, 2, 18, 'uvu', True), (3, 1, 19, 'uvu', True), (3, 3, 20, 'uvu', True), (4, 2, 21, 'uvu', True), (5, 3, 22, 'uvu', True), (6, 2, 23, 'uvu', True), (1, 2, 24, 'uvu', True), (2, 1, 25, 'uvu', True), (2, 3, 26, 'uvu', True), (3, 0, 27, 'uvu', True), (3, 2, 28, 'uvu', True), (4, 1, 29, 'uvu', True), (4, 3, 30, 'uvu', True), (5, 2, 31, 'uvu', True), (6, 1, 32, 'uvu', True), (6, 3, 33, 'uvu', True), (0, 2, 34, 'uvu', True), (1, 1, 35, 'uvu', True), (1, 3, 36, 'uvu', True), (2, 2, 37, 'uvu', True), (3, 1, 38, 'uvu', True), (3, 3, 39, 'uvu', True), (4, 0, 40, 'uvu', True), (4, 2, 41, 'uvu', True), (5, 1, 42, 'uvu', True), (5, 3, 43, 'uvu', True), (6, 2, 44, 'uvu', True), (0, 3, 45, 'uvu', True), (1, 2, 46, 'uvu', True), (2, 3, 47, 'uvu', True), (3, 2, 48, 'uvu', True), (4, 1, 49, 'uvu', True), (4, 3, 50, 'uvu', True), (5, 0, 51, 'uvu', True), (5, 2, 52, 'uvu', True), (6, 1, 53, 'uvu', True), (6, 3, 54, 'uvu', True), (1, 3, 55, 'uvu', True), (2, 2, 56, 'uvu', True), (3, 1, 57, 'uvu', True), (3, 3, 58, 'uvu', True), (4, 2, 59, 'uvu', True), (5, 1, 60, 'uvu', True), (5, 3, 61, 'uvu', True), (6, 0, 62, 'uvu', True), (6, 2, 63, 'uvu', True)])]
    cfgs += [tables.make_config('128x0o+128x0e+64x1o+64x1e+32x2o+32x2e+32x3o+32x3e', '1x0e+1x1o+1x2e+1x3o', '128x0o+64x0o+32x0o+32x0o+128x0e+64x0e+32x0e+32x0e+128x1o+64x1o+64x1o+64x1o+32x1o+32x1o+32x1o+32x1o+32x1o+128x1e+64x1e+64x1e+64x1e+32x1e+32x1e+32x1e+32x1e+32x1e+128x2o+64x2o+64x2o+64x2o+32x2o+32x2o+32x2o+32x2o+32x2o+32x2o+32x2o+128x2e+64x2e+64x2e+64x2e+32x2e+32x2e+32x2e+32x2e+32x2e+32x2e+32x2e+128x3o+64x3o+64x3o+32x3o+32x3o+32x3o+32x3o+32x3o+32x3o+32x3o+128x3e+64x3e+64x3e+32x3e+32x3e+32x3e+32x3e+32x3e+32x3e+32x3e', [(0, 0, 0, 'uvu', True), (3, 1, 1, 'uvu', True), (4, 2, 2, 'uvu', True), (7, 3, 3, 'uvu', True), (1, 0, 4, 'uvu', True), (2, 1, 5, 'uvu', True), (5, 2, 6, 'uvu', True), (6, 3, 7, 'uvu', True), (1, 1, 8, 'uvu', True), (2, 0, 9, 'uvu', True), (2, 2, 10, 'uvu', True), (3, 1, 11, 'uvu', True), (4, 2, 12, 'uvu', True), (5, 1, 13, 'uvu', True), (5, 3, 14, 'uvu', True), (6, 2, 15, 'uvu', True), (7, 3, 16, 'uvu', True), (0, 1, 17, 'uvu', True), (2, 1, 18, 'uvu', True), (3, 0, 19, 'uvu', True), (3, 2, 20, 'uvu', True), (4, 1, 21, 'uvu', True), (4, 3, 22, 'uvu', True), (5, 2, 23, 'uvu', True), (6, 3, 24, 'uvu', True), (7, 2, 25, 'uvu', True), (0, 2, 26, 'uvu', True), (2, 2, 27, 'uvu', True), (3, 1, 28, 'uvu', True), (3, 3, 29, 'uvu', True), (4, 0, 30, 'uvu', True), (4, 2, 31, 'uvu', True), (5, 1, 32, 'uvu', True), (5, 3, 33, 'uvu', True), (6, 2, 34, 'uvu', True), (7, 1, 35, 'uvu', True), (7, 3, 36, 'uvu', True), (1, 2, 37, 'uvu', True), (2, 1, 38, 'uvu', True), (2, 3, 39, 'uvu', True), (3, 2, 40, 'uvu', True), (4, 1, 41, 'uvu', True), (4, 3, 42, 'uvu', True), (5, 0, 43, 'uvu', True), (5, 2, 44, 'uvu', True), (6, 1, 45, 'uvu', True), (6, 3, 46, 'uvu', True), (7, 2, 47, 'uvu', True), (1, 3, 48, 'uvu', True), (2, 2, 49, 'uvu', True), (3, 3, 50, 'uvu', True), (4, 2, 51, 'uvu', True), (5, 1, 52, 'uvu', True), (5, 3, 53, 'uvu', True), (6, 0, 54, 'uvu', True), (6, 2, 55, 'uvu', True), (7, 1, 56, 'uvu', True), (7, 3, 57, 'uvu', True), (0, 3, 58, 'uvu', True), (2, 3, 59, 'uvu', True), (3, 2, 60, 'uvu', True), (4, 1, 61, 'uvu', True), (4, 3, 62, 'uvu', True), (5, 2, 63, 'uvu', True), (6, 1, 64, 'uvu', True), (6, 3, 65, 'uvu', True), (7, 0, 66, 'uvu', True), (7, 2, 67, 'uvu', True)], '1x0e+1x0e+1x0e+1x0e+1x1o+1x1o+1x1o+1x1o+1x1o+1x1o+1x1o+1x1o+1x1o+1x2e+1x2e+1x2e+1x2e+1x2e+1x2e+1x2e+1x2e+1x2e+1x2e+1x2e+1x3o+1x3o+1x3o+1x3o+1x3o+1x3o+1x3o+1x3o+1x3o+1x3o')]
    cfgs += [tables.make_config('128x0e+64x1o+32x2e+32x3o', '1x0e+1x1o+1x2e+1x3o', '128x0e+64x0e+32x0e+32x0e', [(0, 0, 0, 'uvu', True), (1, 1, 1, 'uvu', True), (2, 2, 2, 'uvu', True), (3, 3, 3, 'uvu', True)])]
    return _unique(cfgs)


def user_configs():
    """Shapes added by `python -m jet_mlip.shapes <family> <model> --apply`."""
    cfgs = []
    return _unique(cfgs)


FAMILIES = {'sevenn': sevenn_configs, 'mace': mace_configs, 'nequip': nequip_configs,
            'nequix': nequix_configs, 'equflash': equflash_configs}


def selected(which='all'):
    """The configs of JET_CONFIGS=<which>: 'all', or a comma list of families (e.g. 'mace,nequip'), plus user_configs()."""
    names = list(FAMILIES) if which == 'all' else which.split(',')
    bad = [n for n in names if n not in FAMILIES]
    if bad:
        raise SystemExit(f'JET_CONFIGS={which}: unknown {", ".join(bad)}; '
                         f'use all or a comma list of {", ".join(FAMILIES)}')
    return _unique([c for n in names for c in FAMILIES[n]()] + user_configs())


def _unique(cfgs):
    """Dedupe by key, keep order."""
    seen, out = set(), []
    for c in cfgs:
        if c.key not in seen:
            seen.add(c.key)
            out.append(c)
    return out


def _write_if_changed(path, text):
    """Keep mtimes of unchanged generated files so ninja does not rebuild everything."""
    if os.path.exists(path) and open(path).read() == text:
        return
    with open(path, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    cfgs = selected(os.environ.get('JET_CONFIGS', 'all'))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc', 'configs_gen.h')
    _write_if_changed(path, tables.emit_header(cfgs))
    print(f'wrote {path}: {len(cfgs)} configs (JET_CONFIGS={os.environ.get("JET_CONFIGS", "all")})')
    # one instantiation TU per config for conv_fused and tp3 (keeps nvcc memory per process small); JET_LOW_RAM=1: one
    # per config, dtype and variant (a 7net-omni layer took > 12.5 GB of cicc in one TU, too much for a 16 GB machine)
    inst = os.path.join(os.path.dirname(path), 'inst')
    os.makedirs(inst, exist_ok=True)
    split = os.environ.get('JET_LOW_RAM', '0') == '1'
    train = os.environ.get('JET_TRAIN', '0') == '1'   # tp3 (train-mode path of the SevenNet / MACE patches): off by default
    tus = {}
    for i, c in enumerate(cfgs):
        n = tables._cname(c.key)
        for t in ('float', 'double'):
            for b in ('false', 'true'):
                tus.setdefault(f'conv_fused_{i}_{t}_{b}.cu' if split else f'conv_fused_{i}.cu', []).append(
                    f'template void launch<{t}, {n}, EPL_OF<{t}, {n}, {b}>, {b}>(const Args<{t}>&);')
            for slot in range(4 if train else 0):   # tp3.cuh: the decoupled tensor product, same split
                tus.setdefault(f'tp3_{i}_{t}_{slot}.cu' if split else f'tp3_{i}.cu', []).append(
                    f'template void launch_cfg<{t}, {n}, {slot}>(const {t}*, const {t}*, const {t}*, const {t}*, const int*, '
                    f'const int*, const int*, const int*, {t}*, int, cudaStream_t);')
    for f in os.listdir(inst):
        if f not in tus:
            os.remove(os.path.join(inst, f))
    for f, insts in tus.items():
        hdr, ns = ('tp3.cuh', 'tp3k') if f.startswith('tp3') else ('conv_fused.cuh', 'cf')
        _write_if_changed(
            os.path.join(inst, f),
            f'// Generated by jet_mlip/configs.py. Do not edit.\n#include "../{hdr}"\n'
            f'namespace jet_mlip {{ namespace {ns} {{\n' + ''.join(x + '\n' for x in insts) + '}}\n')
