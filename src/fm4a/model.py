"""
fm4a.model
==========

Provides helper functions instantiate the Prithvi-WxC model and load pre-trained weights.
"""
from pathlib import Path

from PrithviWxC.dataloaders.merra2 import (
    input_scalers,
    output_scalers,
    static_input_scalers,
)

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
        "encoder_shifting": True,
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
        scaling_factor_dir: Path
):
    """
    Load Prithvi-WxC model.

    Args:
        config: Name of the model configuration "large" or "small".
        scaling_factor_dir: The path containing the scaling factors.
    """
    surf_in_scal_path = scaling_factor_dir / "musigma_surface.nc"
    vert_in_scal_path = scaling_factor_dir / "musigma_vertical.nc"
    surf_out_scal_path = scaling_factor_dir / "anomaly_variance_surface.nc"
    vert_out_scal_path = scaling_factor_dir / "anomaly_variance_vertical.nc"



    in_mu, in_sig = input_scalers(
        surface_vars,
        vertical_vars,
        levels,
        surf_in_scal_path,
        vert_in_scal_path,
    )
    output_sig = output_scalers(
        surface_vars,
        vertical_vars,
        levels,
        surf_out_scal_path,
        vert_out_scal_path,
    )
    static_mu, static_sig = static_input_scalers(
        surf_in_scal_path,
        static_surface_vars,
    )

    params = MODEL_PARAMS[config]
    params["input_scalers_mu"] = in_mu
    params["input_scalers_sigma"] = in_sig
    params["static_input_scalers_mu"] = static_mu
    params["static_input_scalers_epsilon"] = static_sig
    params["output_scalers"] = output_sig ** 0.5
    params["masking_ratios"] = 0.0
    params["mask_ratio_targets"] = 0.0

    model = PrithviWxC(
        in_channels=config["params"]["in_channels"],
        input_size_time=config["params"]["input_size_time"],
        in_channels_static=config["params"]["in_channels_static"],
        input_scalers_mu=in_mu,
        input_scalers_sigma=in_sig,
        input_scalers_epsilon=config["params"]["input_scalers_epsilon"],
        static_input_scalers_mu=static_mu,
        static_input_scalers_sigma=static_sig,
        static_input_scalers_epsilon=config["params"][
                "static_input_scalers_epsilon"
        ],
        output_scalers=output_sig**0.5,
        n_lats_px=config["params"]["n_lats_px"],
        n_lons_px=config["params"]["n_lons_px"],
        patch_size_px=config["params"]["patch_size_px"],
        mask_unit_size_px=config["params"]["mask_unit_size_px"],
        mask_ratio_inputs=masking_ratio,
        mask_ratio_targets=0.0,
        embed_dim=config["params"]["embed_dim"],
        n_blocks_encoder=config["params"]["n_blocks_encoder"],
        n_blocks_decoder=config["params"]["n_blocks_decoder"],
        mlp_multiplier=config["params"]["mlp_multiplier"],
        n_heads=config["params"]["n_heads"],
        dropout=config["params"]["dropout"],
        drop_path=config["params"]["drop_path"],
        parameter_dropout=config["params"]["parameter_dropout"],
        residual=residual,
        masking_mode=masking_mode,
        encoder_shifting=encoder_shifting,
        decoder_shifting=decoder_shifting,
        positional_encoding=positional_encoding,
        checkpoint_encoder=[],
        checkpoint_decoder=[],
        )
