"""Deploy a NequIP model for LAMMPS ML-IAP (`pair_style jet/mliap <out>`) with the jet_nequip fused
convolution and CUDA-graph replay. The upstream `create_lmp_mliap_file.py` body, plus `--radial_table`, `--no-jet`
(official modifiers only, unpatched, for A/B) and `--no-graph` (upstream compile path: dynamic=True, no graph).

    python -m jet_nequip.deploy <model_path or nequip.net id> out.nequip.lmp.pt [--radial_table 2048]
"""
import argparse
import pathlib

import torch

from common.radial_table import DEFAULT_N
from nequip.integrations.lammps_mliap.create_lmp_mliap_file import logger
from nequip.integrations.lammps_mliap.lmp_mliap_wrapper import NequIPLAMMPSMLIAPWrapper
from nequip.model.saved_models.load_utils import _get_model_file_path
from nequip.train.lightning import _SOLE_MODEL_KEY

from .mliap import NequIPGraphMLIAPWrapper, NequIPJetMLIAPWrapper


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Create NequIP LAMMPS ML-IAP file from saved models."
    )
    parser.add_argument(
        "model_path", help="path to a checkpoint model or packaged model file", type=pathlib.Path,
    )
    parser.add_argument(
        "output_path", help="path to write NequIP LAMMPS ML-IAP interface file (must end with `.nequip.lmp.pt`)",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--model",
        help=f"name of model to use (default: {_SOLE_MODEL_KEY}, meant to work for the conventional single model case)",
        type=str, default=_SOLE_MODEL_KEY,
    )
    parser.add_argument("--modifiers", help="modifiers to apply to the model", nargs="+", type=str, default=[])
    parser.add_argument(
        "--no-compile", help="disable torch.compile for the model (default: False, compile is enabled)",
        action="store_true", default=False,
    )
    parser.add_argument(
        "--tf32", help="whether to use TF32 or not (default: False)",
        action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument('--radial_table', type=int, default=DEFAULT_N)
    parser.add_argument('--no-jet', action='store_true', help='no jet_mlip patch: official modifiers only (A/B)')
    parser.add_argument('--no-graph', action='store_true', help='upstream compile path instead of the CUDA-graph wrapper')
    args = parser.parse_args(args=args)

    if not str(args.output_path).endswith(".nequip.lmp.pt"):
        raise ValueError(f"Output path must end with `.nequip.lmp.pt`, got: {args.output_path}")

    logger.info(f"Creating LAMMPS ML-IAP artefact from {args.model_path} ...")
    with _get_model_file_path(str(args.model_path)) as model_file_path:
        if args.no_graph:
            cls = NequIPLAMMPSMLIAPWrapper if args.no_jet else NequIPJetMLIAPWrapper
            kw = {} if args.no_jet else {"radial_table": args.radial_table}
        else:
            cls, kw = NequIPGraphMLIAPWrapper, {"jet": not args.no_jet, "radial_table": args.radial_table}
        mliap_module = cls(
            model_path=str(model_file_path),
            model_key=args.model,
            modifiers=args.modifiers,
            compile=not args.no_compile,
            tf32=args.tf32,
            **kw,
        )
    torch.save(mliap_module, args.output_path)
    logger.info(f"LAMMPS ML-IAP artefact saved to {args.output_path}")


if __name__ == "__main__":
    main()
