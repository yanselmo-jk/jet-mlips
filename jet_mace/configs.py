"""MACE interaction blocks -> jet_mlip ConvConfig; the CLI is `python -m jet_mlip.shapes mace <model>`."""
from e3nn import o3
from jet_mlip import tables


def instructions(tp):
    return [(i.i_in1, i.i_in2, i.i_out, i.connection_mode, i.has_weight) for i in tp.instructions]


def load(name, device='cpu'):
    """A mace_mp foundation model by name, or MACE-OFF23 for an 'off-' prefix ('off-small' ...)."""
    from mace.calculators import mace_mp, mace_off
    if name.startswith('off-'):
        return mace_off(model=name[4:], device=device, default_dtype='float32', return_raw_model=True)
    return mace_mp(model=name, device=device, default_dtype='float32', return_raw_model=True)


def n_split(block) -> int:
    """Number of channel slices its conv_tp needs: jet takes mul <= 128, so one uniform mul > 128 is sliced."""
    tp = block.conv_tp
    muls = {mul for irs in (tp.irreps_in1, tp.irreps_out) for mul, _ in irs}
    if max(muls) <= 128:
        return 1
    if len(muls) == 1:
        m = muls.pop()
        for k in range(2, m + 1):
            if m % k == 0 and m // k <= 128 and (m // k) % 8 == 0:
                return k
    raise ValueError(f'jet needs every mul a multiple of 8 <= 128, or one uniform mul that can be sliced: '
                     f'got in1 {tp.irreps_in1}, out {tp.irreps_out}')


def config_from_block(block, div: int = 1) -> tables.ConvConfig:
    """From a MACE InteractionBlock (e3nn backend): its conv_tp is a uvu TensorProduct whose output
    is fully consumed by the following Linear, so every path is live. div windows every mul (a mul-256
    block becomes the mul-128 shapes compiled for 'medium')."""
    tp = block.conv_tp
    if div == 1:
        return tables.make_config(tp.irreps_in1, tp.irreps_in2, tp.irreps_out, instructions(tp))
    x, out = (o3.Irreps([(mul // div, ir) for mul, ir in o3.Irreps(irs)]) for irs in (tp.irreps_in1, tp.irreps_out))
    return tables.make_config(x, tp.irreps_in2, out, instructions(tp))


def model_configs(model):
    return [config_from_block(b, div=n_split(b)) for b in model.interactions]
