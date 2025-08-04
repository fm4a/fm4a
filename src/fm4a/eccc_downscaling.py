"""
fm4a.eccc_downscaling
=====================

Functionality to load the input data for the ECCC downscaling.
"""
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr


SURFACE_VARS = [
    "TT", "UUWE", "VVSN"
]

STATIC_SURFACE_VARS = ["ME", "MG", "Z0"]
VERTICAL_PRESSURE_VARS = [
    "GZ_pressure_levels",
    "HU_pressure_levels",
    "TT_pressure_levels",
    "UUWE_pressure_levels",
    "VVSN_pressure_levels",
    "WW_pressure_levels"
]
VERTICAL_LEVEL1_VARS = [
    "GZ_model_levels",
    "HU_model_levels",
    "TT_model_levels",
    "WW_model_levels"
]

VERTICAL_LEVEL2_VARS = [
    "UUWE_model_levels",
    "VVSN_model_levels"
]

OTHER_VARS = ["NT", "P0", "PC", "PR", "TM", "H"]


def downsample_tensor(
        tensor: torch.Tensor,
        factor:int
) -> torch.Tensor:
    return  F.interpolate(
        tensor.unsqueeze(0),
        scale_factor=(1/factor, 1/factor),
        mode='nearest-exact'
    ).squeeze(0)


def load_input_data(
        dynamic_input_file: Union[str, Path],
        static_input_file: Union[str, Path],
) -> Dict[str, torch.Tensor]:

    dynamic_input_file = Path(dynamic_input_file)
    static_input_file = Path(static_input_file)

    with xr.open_dataset(dynamic_input_file) as data:
        data = data[{"time": 0}]
        all_vars = SURFACE_VARS + VERTICAL_PRESSURE_VARS + VERTICAL_LEVEL1_VARS + VERTICAL_LEVEL2_VARS + OTHER_VARS
        arrays = [data[var].compute().data for var in all_vars]
        time = data.time.data
        lats = data.lat.data
        coords = data[["lat", "lon"]].compute()
    data.close()
    arrays = [array[None] if array.ndim == 2 else array for array in arrays]
    dynamic_data = np.concatenate(arrays, axis=0)


    # Static input
    with xr.open_dataset(static_input_file) as data:
        all_vars = STATIC_SURFACE_VARS
        data = data[all_vars].compute().interp_like(coords)
        arrays = [data[var].compute().data for var in all_vars]
    data.close()

    lats_rad = np.deg2rad(coords.lat)
    lons_rad = np.deg2rad(coords.lon)

    # Position signal
    arrays += [
        np.sin(lats_rad),
        np.cos(lons_rad),
        np.sin(lons_rad)
    ]

    # Time signal
    doy = (time - time.astype("datetime64[Y]").astype("datetime64[s]")).astype("timedelta64[D]").astype("float32") + 1
    doy = min(doy, 365)
    hour = (time - time.astype("datetime64[D]").astype("datetime64[s]")).astype("timedelta64[h]").astype("float32")
    arrays += [
        np.cos(doy / 366 * 2 * np.pi) * np.ones_like(lats),
        np.sin(doy / 366 * 2 * np.pi) * np.ones_like(lats),
        np.cos(hour / 366 * 2 * np.pi) * np.ones_like(lats),
        np.sin(hour / 366 * 2 * np.pi) * np.ones_like(lats)
    ]
    print([arr.shape for arr in arrays])
    static_data = np.stack(arrays)[..., :1280,:2528]

    dynamic_data = downsample_tensor(torch.tensor(dynamic_data[:1280,:2528]), 8)
    static_data = torch.tensor(static_data)


    return {
        "x": dynamic_data,
        "static": static_data
    }
