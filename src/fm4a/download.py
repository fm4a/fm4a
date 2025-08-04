"""
fm4a.download
=============

Provides download functionality for MERRA input data for the Prithvi-WxC model.
"""
from datetime import datetime, timedelta
from functools import cache
import getpass
import os
from pathlib import Path
import re
from typing import List, Optional, Tuple, Union

import requests
from huggingface_hub import hf_hub_download, snapshot_download
import numpy as np
from tqdm import tqdm
import xarray as xr


from .definitions import (
    SURFACE_VARS,
    STATIC_SURFACE_VARS,
    VERTICAL_VARS,
    LEVELS,
    NAN_VALS
)

def find_file_url(
        base_url: str,
        product_name: str,
        time: np.datetime64,
):
    """
    Find URL of MERRA-2 file accounting for changes in production stream.

    Args:
        base_urls: The stem of the URL where the files are located.
        product_name: The product name as used in the file name.
        time: A numpy datetime64 object specifying the time for which to download the file.

    Return:
        A string containing the URL of the desired MERRA-2 file.
    """
    if time is None:
        url = f"{base_url}/1980/"
        fname = f"MERRA2_\d\d\d\.{product_name}\.00000000.nc4"
    else:
        date = time.astype("datetime64[s]").item()
        url = date.strftime(
            f"{base_url}/%Y/%m/"
        )
        fname = date.strftime(f"MERRA2_\d\d\d\.{product_name}\.%Y%m%d\.nc4")
    regexp = re.compile(rf'href="({fname})"')
    response = requests.get(url)
    response.raise_for_status()
    matches = regexp.findall(response.text)
    if len(matches) == 0:
        raise ValueError(
            "Found no matching file in %s.",
            url
        )
    return url + matches[1]


MERRA2_PRODUCTS = {
    "M2I3NXASM": ("https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVASM.5.12.4", "inst3_3d_asm_Nv"),
    "M2I1NXASM": ("https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I1NXASM.5.12.4", "inst1_2d_asm_Nx"),
    "M2T1NXLND": ("https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXLND.5.12.4", "tavg1_2d_lnd_Nx"),
    "M2T1NXFLX": ("https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXFLX.5.12.4", "tavg1_2d_flx_Nx"),
    "M2T1NXRAD": ("https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXRAD.5.12.4", "tavg1_2d_rad_Nx"),
    "CONST2DASM": ("https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2C0NXASM.5.12.4",  "const_2d_asm_Nx"),
    "CONST2DCTM": ("https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2C0NXCTM.5.12.4", "const_2d_ctm_Nx")
}


def get_merra_urls(time: np.datetime64) -> List[str]:
    """
    List MERRA2 URLS required to prepare the input data for a given time step.
    """
    m2i3nxasm_url = find_file_url(*MERRA2_PRODUCTS["M2I3NXASM"], time)
    m2i1nxasm_url = find_file_url(*MERRA2_PRODUCTS["M2I1NXASM"], time)
    m2t1nxlnd_url = find_file_url(*MERRA2_PRODUCTS["M2T1NXLND"], time)
    m2t1nxflx_url = find_file_url(*MERRA2_PRODUCTS["M2T1NXFLX"], time)
    m2t1nxrad_url = find_file_url(*MERRA2_PRODUCTS["M2T1NXRAD"], time)

    return [
        m2i3nxasm_url,
        m2i1nxasm_url,
        m2t1nxlnd_url,
        m2t1nxflx_url,
        m2t1nxrad_url,
    ]


@cache
def get_credentials() -> Tuple[str, str]:
    """
    Retrieves user name and password for GES DISC server through from environment variables or
    through user interaction.
    """
    ges_disc_user = os.environ.get("GES_DISC_USER", None)
    ges_disc_pw = os.environ.get("GES_DISC_PASSWORD", None)

    if (not ges_disc_user is None) and (not ges_disc_pw is None):
        return ges_disc_user, ges_disc_pw

    username = input("GES DISC username: ")
    password = getpass.getpass("GES DISC password: ")
    return username, password


def download_merra2_file(
        url: str,
        destination: Union[str, Path],
        force: bool = False
) -> Path:
    """
    Download MERRA2 file if it not already exists.

    Args:
        url: String containing the URL of the file to download.
        destination: The folder to which to download the file.
        force: Set to 'True' to force download even if file exists locally.

    Return:
        A Path object pointing to the local file.
    """
    destination = Path(destination)
    destination.mkdir(exist_ok=True, parents=True)

    filename = url.split("/")[-1]
    destination = destination / filename

    if not force and destination.exists():
        return destination

    auth = get_credentials()
    with requests.Session() as session:
        session.auth = auth
        redirect = session.get(url, auth=auth)
        response = session.get(redirect.url, auth=auth, stream=True)
        response.raise_for_status()

        with open(destination, "wb") as output:
            for chunk in response:
                output.write(chunk)

    return destination


def download_merra_files(
        time: np.datetime64,
        destination: Union[str, Path] = Path(".")
) -> List[str]:
    """
    Download MERRA2 files required to prepare the input data for a given time step.
    """
    urls = get_merra_urls(time)
    time = time.astype("datetime64[s]").item()
    files = []

    # Dynamic data
    year = time.year
    month = time.month
    day = time.day
    dest_dyn = Path(destination) / f"{year}/{month:02}/{day:02}"

    for url in tqdm(urls, desc="Downloading dynamic data"):
        files.append(download_merra2_file(url, dest_dyn))

    # Constant data
    urls_const = [find_file_url(*MERRA2_PRODUCTS["CONST2DCTM"], None)]
    dest_const = Path(destination) / f"constant/"
    for url in tqdm(urls_const, desc="Downloading constant data"):
        files.append(download_merra2_file(url, dest_const))

    return files


def get_required_input_files(time: np.datetime64) -> List[str]:
    """
    Get required Prithvi-WxC input files for given time.

    Args:
        time: A numpy.datetime64 object defining the input time.

    Return:
        A list containing the required input file names..
    """
    date = time.astype("datetime64[s]").item()
    return [
        date.strftime("MERRA2_sfc_%Y%m%d.nc"),
        date.strftime("MERRA_pres_%Y%m%d.nc"),
    ]


def get_prithvi_wxc_climatology(
        time: np.datetime64,
        climatology_dir: Path
):
    """
    Download climatology files for given times.

    Args:
        time: A numpy.datetime64 object specifying the time for which to download the climatology data.
        climatology_dir: The path in which to store the climatology files.
    """
    date = time.astype("datetime64[s]").item()
    doy = (date - datetime(date.year, 1, 1)).days + 1
    hour = (date.hour // 3) * 3

    fname_vert = f"climatology/climate_vertical_doy{doy:03}_hour{hour:02}.nc"
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        filename=fname_vert,
        local_dir=climatology_dir
    )
    fname_sfc = f"climatology/climate_surface_doy{doy:03}_hour{hour:02}.nc"
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        filename=fname_sfc,
        local_dir=climatology_dir
    )

def get_prithvi_wxc_scaling_factors(
        scaling_factor_dir: Path
        ):
    """
    Download scaling factor for the Prithvi-WxC model.
    """

    scale_in_surf = "musigma_surface.nc"
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        filename=f"climatology/{scale_in_surf}",
        local_dir=scaling_factor_dir,
    )
    scale_in_vert = "musigma_vertical.nc"
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        filename=f"climatology/{scale_in_vert}",
        local_dir=scaling_factor_dir,
    )
    scale_out_surf = "anomaly_variance_surface.nc"
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        filename=f"climatology/{scale_out_surf}",
        local_dir=scaling_factor_dir,
    )
    scale_out_vert = "anomaly_variance_vertical.nc"
    hf_hub_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        filename=f"climatology/{scale_out_vert}",
        local_dir=scaling_factor_dir,
    )


def get_prithvi_wxc_input_time_step(
        time: np.datetime64,
        input_data_dir: Path,
        download_dir: Path,
        force: bool = False
):
    """
    Download and prepare Prithvi-WxC input data for a single time step.

    Args:
        time: The time for which to prepare the input data.
        input_data_dir: The directory to which to write the extracted input data.
        download_dir: The directory to use to store the raw MERRA 2 data.
        force: Set to 'True' to force input extract even the output files already exist.

    """
    input_data_dir = Path(input_data_dir)
    download_dir = Path(download_dir)

    # Nothing do to if files already exist.
    input_files = [input_data_dir / input_file for input_file in get_required_input_files(time)]
    if not force and all([input_file.exists() for input_file in input_files]):
        return input_files

    if download_dir is None:
        tmp = tempfile.TemporaryDirectory()
        download_dir = Path(tmp.name)
    else:
        tmp = None

    try:
        merra_files = download_merra_files(time, download_dir)
        start_time = time.astype("datetime64[D]").astype("datetime64[h]")
        end_time = start_time + np.timedelta64(24, "h")
        time_steps = np.arange(start_time, end_time, np.timedelta64(3, "h"))

        vars_req = VERTICAL_VARS + SURFACE_VARS + STATIC_SURFACE_VARS

        all_data = []
        for path in merra_files:
            with xr.open_dataset(path) as data:
                if "const" in path.name:
                    vars = [
                        var for var in STATIC_SURFACE_VARS if var in data.variables
                    ]
                else:
                    vars = [
                        var for var in VERTICAL_VARS + SURFACE_VARS if var in data.variables
                    ]

                data = data[vars + ["time"]]
                if "lev" in data:
                    data = data.loc[{"lev": np.array(LEVELS)}]
                data = data.compute()

                for var in data:
                    if var in NAN_VALS:
                        nan = NAN_VALS[var]
                        data[var].data[:] = np.nan_to_num(data[var].data, nan=nan)

                # For static data without time dependence simply collapse the time dimension.
                if data.time.size == 1:
                    data = data[{"time": 0}]
                # For monthly static data, simply pick the right month
                elif data.time.size == 12:
                    month = start_time.astype("datetime64[s]").item().month
                    data = data[{"time": month - 1}]
                # Select time steps for three-hourly data, linearly interpolate for hourly data
                else:
                    method = "nearest"
                    if (data.time.data[0] - data.time.data[0].astype("datetime64[h]")) > 0:
                        for var in data:
                            data[var].data[1:] = 0.5 * (data[var].data[1:] + data[var].data[:-1])
                        new_time = data.time.data - 0.5 * (data.time.data[1] -  data.time.data[0])
                        data = data.assign_coords(time=new_time)

                    times = list(data.time.data)
                    inds = [times.index(t_s) for t_s in time_steps]
                    data = data[{"time": inds}]

                for var in data:
                    if var in NAN_VALS:
                        data[var].data[:] = np.nan_to_num(data[var].data, nan=NAN_VALS[var], copy=True)

                all_data.append(data)


        data = xr.merge(all_data, compat="override")

        input_data_dir.mkdir(exist_ok=True, parents=True)

        data_sfc = data[SURFACE_VARS + STATIC_SURFACE_VARS]
        encoding = {name: {"zlib": True} for name in data_sfc}
        data_sfc["time"] = (
            (data_sfc.time.astype("datetime64[m]").data - np.datetime64("2020-01-01", "m")).astype("timedelta64[m]").astype("int32")
        )
        encoding["time"] = {
            "dtype": "int32",
        }
        date = data.time.data[0].astype("datetime64[s]").item()
        output_file = date.strftime("MERRA2_sfc_%Y%m%d.nc")
        data_sfc.time.attrs = {
            "begin_time": 0,
            "begin_date": 20200101,
        }
        data_sfc.to_netcdf(input_data_dir / output_file, encoding=encoding)


        data_pres = data[VERTICAL_VARS]
        encoding = {name: {"zlib": True} for name in data_pres}
        data_pres["time"] = (
            (data_pres.time.astype("datetime64[m]").data - np.datetime64("2020-01-01", "m")).astype("timedelta64[m]").astype("int32")
        )
        encoding["time"] = {
            "dtype": "int32",
        }
        data_pres.time.attrs = {
            "begin_time": 0,
            "begin_date": 20200101,
        }

        output_file = date.strftime(date.strftime("MERRA_pres_%Y%m%d.nc"))
        data_pres.to_netcdf(input_data_dir / output_file, encoding=encoding)
    finally:
        if tmp is not None:
            tmp.cleanup()


def get_prithvi_wxc_input(
        time: np.datetime64,
        input_time_step: int,
        lead_time: int,
        input_data_dir: Path,
        download_dir: Optional[Path] = None,
):
    """
    Download and prepare Prithvi-WxC input data for a forecast initialized at a given time.

    Args:
        time: The time at which the forecast is initialized.
        input_time: The time difference in hours to the previous input time step.
        lead_time: The lead time up to which the forecast is made.
        download_dir: The directory to use to store the raw MERRA 2 data.
        input_data_dir:
    """
    input_times = [time - np.timedelta64(input_time_step, "h"), time]
    for input_time in input_times:
        get_prithvi_wxc_input_time_step(
            input_time,
            input_data_dir,
            download_dir=download_dir
        )

    output_times = time + np.arange(
        input_time_step,
        lead_time + 1,
        input_time_step
    ).astype("timedelta64[h]")
    print(output_times)
    for output_time in output_times:
        if output_time not in input_times:
            get_prithvi_wxc_input_time_step(
                output_time,
                input_data_dir,
                download_dir=download_dir
            )
        get_prithvi_wxc_climatology(
            output_time,
            input_data_dir
        )


def download_gdps_file(
        time: Union[np.datetime64, str],
        step: int,
        destination: Union[str, Path],
        force: bool = False
) -> Path:
    """
    Download MERRA2 file if it not already exists.

    Args:
        url: String containing the URL of the file to download.
        destination: The folder to which to download the file.
        force: Set to 'True' to force download even if file exists locally.

    Return:
        A Path object pointing to the local file.
    """
    if isinstance(time, str):
        time = np.datetime64(time)

    date = time.astype("datetime64[s]").item()
    if date.hour not in (0, 12):
        raise ValueError(
            "Initialization time must be at hour 0 or 12 of the day."
        )
    if time < np.datetime64("2022-07-01") or np.datetime64("2022-08-01") <= time:
        raise ValueError(
            "Initialization time must be within [2022-07-01, 2022-08-01)."
        )

    fname = date.strftime(f"%Y%m%d%H_{step:03}.nc")
    url = "https://hpfx.collab.science.gc.ca/~snow000/hrdps_domain/gdps_regridded/" + fname

    destination = Path(destination)
    destination.mkdir(exist_ok=True, parents=True)
    destination = destination / fname

    response = requests.get(url)
    response.raise_for_status()

    with open(destination, "wb") as output:
        for chunk in response:
            output.write(chunk)

    return destination


def download_eccc_static_data(
        destination: Union[str, Path],
        force: bool = False
) -> Path:
    """
    Download static high-resolution data for GDPS downscaling.

    Args:
        destination: The folder to which to download the file.
        force: Set to 'True' to force download even if file exists locally.

    Return:
        A Path object pointing to the local file.
    """
    fname = "geophy.nc"
    url = "https://hpfx.collab.science.gc.ca/~snow000/hrdps_domain/" + fname

    destination = Path(destination)
    destination.mkdir(exist_ok=True, parents=True)
    destination = destination / fname

    response = requests.get(url)
    response.raise_for_status()

    with open(destination, "wb") as output:
        for chunk in response:
            output.write(chunk)

    return destination
