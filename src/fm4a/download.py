"""
fm4a.download
=============

Provides download functionality for MERRA input data for the Prithvi-WxC model.
"""
from functools import cache
import getpass
from typing import List, Tuple

import numpy as np
import xarray as xr


M2I3NXASM_URL = (
    "https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVASM.5.12.4/%Y/%m/"
    "MERRA2_400.inst3_3d_asm_Nv.%Y%m%d.nc4"
)

M2I1NXASM_URL = (
    "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I1NXASM.5.12.4/%Y/%m/"
    "MERRA2_400.inst1_2d_asm_Nx.%Y%m%d.nc4"
)

M2T1NXLND_URL = (
    "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXLND.5.12.4/%Y/%m/"
    "MERRA2_400.tavg1_2d_lnd_Nx.%Y%m%d.nc4"
)

M2T1NXFLX_URL = (
    "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXFLX.5.12.4/%Y/%m/"
    "MERRA2_400.tavg1_2d_flx_Nx.%Y%m%d.nc4"
)

M2T1NXRAD_URL = (
    "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXRAD.5.12.4/%Y/%m/"
    "MERRA2_400.tavg1_2d_rad_Nx.%Y%m%d.nc4"
)

CONST2DASM_URL = (
    "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2C0NXASM.5.12.4/1980/"
    "MERRA2_101.const_2d_ctm_Nx.00000000.nc4"
)

CONST2DCTM_URL = (
    "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2C0NXCTM.5.12.4/1980/"
    "MERRA2_101.const_2d_ctm_Nx.00000000.nc4"
)



def get_merra_urls(time: np.datetime64) -> List[str]:
    """
    List MERRA2 URLS required to prepare the input data for a given time step.
    """
    time = time.astype("datetime64[s]").item()

    m2i3nxasm_url = time.strftime(M2I3NXASM_URL)
    m2i1nxasm_url = time.strftime(M2I1NXASM_URL)
    m2t1nxlnd_url = time.strftime(M2T1NXLND_URL)
    m2t1nxflx_url = time.strftime(M2T1NXFLX_URL)
    m2t1nxrad_url = time.strftime(M2T1NXRAD_URL)

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

    # Constant data
    urls_const = [
        CONST2DASM_URL,
        CONST2DCTM_URL,
    ]
    destination = Path(destination) / f"constant/"
    files = []
    for url in tqdm(urls_const, desc="Downloading constant data"):
        files.append(download_merra2_file(url))


    year = time.year
    month = time.month
    day = time.day
    destination = Path(destination) / f"{year}/{month:02}/{day:02}"

    for url in tqdm(urls, desc="Downloading dynamic data"):
        files.append(download_merra2_file(url))

    return files


def get_required_input_files(time: np.datetime64) -> List[str]:
    """
    Get required Prithvi-WxC input files for given time.

    Args:
        time:

    Return:
        A list containing the input files.
    """
    date = time.astype("datetime64[s]").item()
    return [
        date.strftime("MERRA2_sfc_%Y%m%d.nc"),
        date.strftime("MERRA2_pres_%Y%m%d.nc"),
    ]


def get_prithvi_wxc_input(
        time: np.datetime64,
        input_data_dir: Path,
        download_dir: Path
):
    merra_files = download_merra_files(time)
    start_time = time.astype("datetime64[D]").astype("datetime64[h]")
    end_time = start_time + np.timedelta64(24, "h")
    time_steps = np.arange(start_time, end_time, np.timedelta64(3, "h"))

    vars_req = VERTICAL_VARS + SURFACE_VARS + STATIC_SURFACE_VARS

    all_data = []
    for recs in merra_files:
        data_combined = []
        for rec in recs:
            with xr.open_dataset(rec.local_path) as data:
                vars = [
                    var for var in vars_req if var in data.variables
                ]
                data = data[vars + ["time"]]
                if "lev" in data:
                    data = data.loc[{"lev": np.array(LEVELS)}]
                data_combined.append(data.load())
        data = xr.concat(data_combined, "time").sortby("time")

        for var in data:
            if var in NAN_VALS:
                nan = NAN_VALS[var]
                data[var].data[:] = np.nan_to_num(data[var].data, nan=nan)

        if not "time" in data:
            continue

        if (data.time.data[0] - data.time.data[0].astype("datetime64[h]")) > 0:
            for var in data:
                data[var].data[1:] = 0.5 * (data[var].data[1:] + data[var].data[:-1])
            new_time = data.time.data - 0.5 * (data.time.data[1] -  data.time.data[0])
            data = data.assign_coords(time=new_time)

        times = list(data.time.data)
        inds = [times.index(t_s) for t_s in time_steps]
        data_t = data[{"time": inds}]

        all_data.append(data_t)


    data = xr.merge(all_data, compat="override")
    data = data.rename(
        lat="latitude",
        lon="longitude"
    )

    if domain.upper() != "MERRA":
        lons, lats = domains.get_lonlats(domain)
        lons = lons[0]
        lats = lats[..., 0]
        data = data.interp(longitude=lons, latitude=lats)

    output_path = Path(output_path) / f"dynamic/{year:04}/{month:02}/{day:02}"
    output_path.mkdir(exist_ok=True, parents=True)

    encoding = {name: {"zlib": True} for name in data}

    for time_ind in range(data.time.size):
        data_t = data[{"time": time_ind}]
        date = to_datetime(data_t.time.data)
        output_file = date.strftime("merra2_%Y%m%d%H%M%S.nc")
        data_t.to_netcdf(output_path / output_file, encoding=encoding)
