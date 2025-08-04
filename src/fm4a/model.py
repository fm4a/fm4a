"""
fm4a.model
==========

Provides helper functions instantiate the Prithvi-WxC model and load pre-trained weights.
"""
from pathlib import Path
from typing import Optional

from PrithviWxC.dataloaders.merra2 import (
    input_scalers,
    output_scalers,
    static_input_scalers,
)
from PrithviWxC.model import PrithviWxC
import torch

from .definitions import (
    SURFACE_VARS,
    STATIC_SURFACE_VARS,
    VERTICAL_VARS,
    LEVELS
)


MODEL_PARAMS = {
    "small": {
        "in_channels": 160,
        "input_size_time": 2,
        "in_channels_static": 8,
        "input_scalers_epsilon": 0.0,
        "static_input_scalers_epsilon": 0.0,
        "n_lats_px": 360,
        "n_lons_px": 576,
        "patch_size_px": [2, 2],
        "mask_unit_size_px": [30, 32],
        "embed_dim": 1024,
        "n_blocks_encoder": 8,
        "n_blocks_decoder": 2,
        "mlp_multiplier": 4,
        "n_heads": 16,
        "dropout": 0.0,
        "drop_path": 0.0,
        "residual": "climate",
        "masking_mode": "both",
        "encoder_shifting": False,
        "decoder_shifting": True,
        "parameter_dropout": 0.0,
        "positional_encoding": "fourier",
        "checkpoint_encoder": [],
        "checkpoint_decoder": []
    },
    "large": {
        "in_channels": 160,
        "input_size_time": 2,
        "in_channels_static": 8,
        "input_scalers_epsilon": 0.0,
        "static_input_scalers_epsilon": 0.0,
        "n_lats_px": 360,
        "n_lons_px": 576,
        "patch_size_px": [2, 2],
        "mask_unit_size_px": [30, 32],
        "embed_dim": 2560,
        "n_blocks_encoder": 12,
        "n_blocks_decoder": 2,
        "mlp_multiplier": 4,
        "n_heads": 16,
        "dropout": 0.0,
        "drop_path": 0.0,
        "residual": "climate",
        "masking_mode": "both",
        "encoder_shifting": True,
        "parameter_dropout": 0.0,
        "positional_encoding": "fourier",
        "checkpoint_encoder": [],
        "checkpoint_decoder": []
    },
}

def load_model(
        config: str,
        scaling_factor_dir: Path,
        weights: Optional[Path] = None
) -> PrithviWxC:
    """
    Load Prithvi-WxC model.

    Args:
        config: Name of the model configuration "large" or "small".
        scaling_factor_dir: The path containing the scaling factors.
    """
    surf_in_scal_path = scaling_factor_dir / "climatology" / "musigma_surface.nc"
    vert_in_scal_path = scaling_factor_dir / "climatology" / "musigma_vertical.nc"
    surf_out_scal_path = scaling_factor_dir / "climatology" / "anomaly_variance_surface.nc"
    vert_out_scal_path = scaling_factor_dir / "climatology" / "anomaly_variance_vertical.nc"

    in_mu, in_sig = input_scalers(
        SURFACE_VARS,
        VERTICAL_VARS,
        LEVELS,
        surf_in_scal_path,
        vert_in_scal_path,
    )
    output_sig = output_scalers(
        SURFACE_VARS,
        VERTICAL_VARS,
        LEVELS,
        surf_out_scal_path,
        vert_out_scal_path,
    )
    static_mu, static_sig = static_input_scalers(
        surf_in_scal_path,
        STATIC_SURFACE_VARS,
    )
    params = MODEL_PARAMS[config]
    params["input_scalers_mu"] = in_mu
    params["input_scalers_sigma"] = in_sig
    params["static_input_scalers_mu"] = static_mu
    params["static_input_scalers_sigma"] = static_sig
    params["static_input_scalers_epsilon"] = 0.0
    params["output_scalers"] = output_sig ** 0.5
    params["mask_ratio_inputs"] = 0.0
    params["mask_ratio_targets"] = 0.0

    model = PrithviWxC(**params)

    if weights is not None:
        state_dict = torch.load(weights, weights_only=False)
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        model.load_state_dict(state_dict, strict=True)

    return model
