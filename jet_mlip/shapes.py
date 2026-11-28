"""The conv shapes of a model vs what is compiled into jet_mlip._C: `python -m jet_mlip.shapes <family> <model> [--apply]`

<model> is what `python -m jet_mlip.deploy <family> <model>` accepts:
    sevenn    7net-0 (pretrained name) or a checkpoint .pth
    mace      medium (mace_mp foundation name), off-medium, or a .model file
    nequip    nequip.net:mir-group/NequIP-OAM-L:0.1, a .ckpt or a .nequip.zip package
    nequix    nequix-mp-1.pt
    equflash  equflash_oam_30M.pt

Missing shapes are printed as `user_configs()` lines; --apply appends them (then rebuild:
`pip install -e jet_mlip --no-build-isolation` with the JET_CONFIGS of the first build). Exit code 0 = nothing missing.
"""
import os
import sys

from e3nn import o3

from jet_mlip import configs, tables


def compiled_keys():
    import torch
    try:
        return {torch.ops.jet_mlip.config_key(i) for i in range(torch.ops.jet_mlip.num_configs())}
    except Exception:
        return set()


def _mace(model):
    from jet_mace import configs as mace
    return mace.model_configs(mace.load(model))


def _nequip(model):
    from nequip.model.saved_models.load_utils import _get_model_file_path, load_saved_model
    from nequip.model.utils import _EAGER_MODEL_KEY
    from nequip.utils.global_state import set_global_state
    from jet_nequip import convert
    set_global_state()
    with _get_model_file_path(model) as path:
        m = load_saved_model(str(path), compile_mode=_EAGER_MODEL_KEY)
    cfgs = []
    for b in m.modules():
        if type(b).__name__ == 'InteractionBlock':
            if max(mul for mul, _ in b.tp_scatter.tp.irreps_in1) > 128:
                cfgs += [cfg for cfg, *_ in convert.slice_configs(b)]
            else:
                cfgs.append(convert.config_from_block(b)[0])
    return cfgs


def _nequix(model):
    from jet_nequix import convert
    return [convert.config_from_layer(layer)[0] for layer in convert.load_model(model, False)[0].layers]


def _equflash(model):
    import contextlib
    import io
    from jet_equflash import convert
    from jet_equflash.mliap import EquFlashMLIAPWrapper
    w = EquFlashMLIAPWrapper(model)
    # the GGNN wrapper only reads __module__ (device) and returns right after loading with nlocal 0
    with contextlib.redirect_stdout(io.StringIO()):   # it prints the whole model on rank 0
        w.compute_forces(type('Probe', (), dict(nlocal=0, npairs=0, __module__='cpu'))())
    return [convert.config_from_layer(layer) for layer in w.model.modules()
            if type(layer).__name__ in ('IntegratedConv', 'FullConv')]


def _sevenn(model):
    from sevenn.nn.convolution import IrrepsConvolution
    from sevenn.util import model_from_checkpoint, pretrained_name_to_path
    path = model if os.path.isfile(model) else pretrained_name_to_path(model)
    m, c = model_from_checkpoint(path)
    layers = dict(m.named_children())
    cfgs = []
    for k, conv in layers.items():
        if isinstance(conv, IrrepsConvolution):
            live = layers[f"{k.split('_')[0]}_self_interaction_2"].irreps_out   # the next gate's input types
            cfgs.append(tables.config_from_sevenn_conv(conv, live))
    cut = c.get('cutoff_function'); cut = cut.get('cutoff_function_name') if isinstance(cut, dict) else cut
    if cut != 'XPLOR' or list(c.get('weight_nn_hidden_neurons', [])) != [64, 64] or c.get('act_radial', 'silu') != 'silu':
        print('!! not usable with jet (needs XPLOR cutoff, MLP [64, 64], silu)', file=sys.stderr)
    return cfgs


LOADERS = {'sevenn': _sevenn, 'mace': _mace, 'nequip': _nequip, 'nequix': _nequix, 'equflash': _equflash}


def shape_state(key, compiled, user_keys, family_of):
    """'compiled', or why a rebuild is needed; family_of(key) is the family list holding it, if any."""
    if key in compiled:
        return 'compiled'
    if key in user_keys:
        return 'in user_configs(), rebuild'
    family = family_of(key)
    if family:
        return f'MISSING (in the {family} list: JET_CONFIGS including {family} also builds it)'
    return 'MISSING'


def emit_line(cfg, tag):
    """The `user_configs()` line that compiles `cfg` (self-contained: no family import at build time)."""
    live = ''
    if not all(cfg.live_out):
        live = ', live_irreps=' + repr(str(o3.Irreps([mir for mir, ok in zip(cfg.irreps_out, cfg.live_out) if ok])))
    return (f"    cfgs += [tables.make_config({str(cfg.irreps_x)!r}, {str(cfg.irreps_sh)!r}, "
            f"{str(cfg.irreps_out)!r}, {[tuple(i) for i in cfg.instructions]!r}{live})]   # {tag}\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    apply = '--apply' in argv
    args = [a for a in argv if a != '--apply']
    if len(args) != 2 or args[0] not in LOADERS:
        print(__doc__ if len(args) != 2 else
              f'unknown family {args[0]!r}; choose from {", ".join(LOADERS)}', file=sys.stderr)
        return 2

    family, model = args
    compiled = compiled_keys()
    if not compiled:
        print('(jet_mlip._C not built or not importable: every shape shows as MISSING)')
    user_keys = {c.key for c in configs.user_configs()}
    families = {c.key: name for name, cfgs_of in configs.FAMILIES.items() for c in cfgs_of()}
    lines = []
    for i, cfg in enumerate(LOADERS[family](model)):
        state = shape_state(cfg.key, compiled, user_keys, families.get)
        print(f'L{i}: {cfg.key} ({cfg.n_paths} paths, NW {cfg.n_weight}): {state}')
        line = emit_line(cfg, f'{family} {os.path.basename(model.rstrip("/"))} L{i}')
        if state.startswith('MISSING') and line.split('   #')[0] not in [x.split('   #')[0] for x in lines]:
            lines.append(line)
    if not lines:
        print('all shapes compiled or in user_configs()')
        return 0

    print('\n# add to user_configs() in jet_mlip/configs.py, then rebuild:')
    print(''.join(lines), end='')
    if apply:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs.py')
        text = open(p).read()
        head = 'def user_configs('
        assert head in text, 'jet_mlip/configs.py has no user_configs()'
        anchor = '    cfgs = []\n'
        i = text.index(anchor, text.index(head)) + len(anchor)
        new = [x for x in lines if x.split('   #')[0] not in text[text.index(head):]]
        if new:
            open(p, 'w').write(text[:i] + ''.join(new) + text[i:])
        print(f'appended {len(new)} line(s) to user_configs() in {p}')
        print('rebuild: pip install -e jet_mlip --no-build-isolation (same JET_CONFIGS as the first build, default all)')
    return 1


if __name__ == '__main__':
    sys.exit(main())
