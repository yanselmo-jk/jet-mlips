"""cuEquivariance for everything but the convolution: linear layers, symmetric contractions and the skip
FullyConnectedTensorProduct run on cuEq kernels (ir_mul layout); the channelwise conv_tp stays e3nn and is
bypassed by the jet_mace fused convolution in eval mode. Mirrors mace/cli/convert_e3nn_hybrid.py without OEQ."""
import torch


def to_cueq_hybrid(model, device='cuda'):
    from mace.cli.convert_e3nn_cueq import transfer_weights
    from mace.modules.wrapper_ops import CuEquivarianceConfig
    from mace.tools.scripts_utils import extract_config_mace_model

    config = extract_config_mace_model(model)
    config['cueq_config'] = CuEquivarianceConfig(
        enabled=True, layout='ir_mul', group='O3_e3nn', optimize_all=False, optimize_linear=True,
        optimize_symmetric=True, optimize_fctp=True, optimize_channelwise=False, conv_fusion=False)
    dtype = next(model.parameters()).dtype
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        target = model.__class__(**config).to(device)
    finally:
        torch.set_default_dtype(prev)
    transfer_weights(model, target, len(config['hidden_irreps'].slices()) - 1, config['correlation'],
                     config['num_interactions'], config.get('use_reduced_cg', True), keep_last_layer_irreps=False)
    return target.to(device).eval()   # a freshly built model is in train mode; the fused path is eval-only
