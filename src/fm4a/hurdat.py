"""
fm4a.hurdat
===========

Functionality to load hurricane track from the HURDAT 2 database.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr


def download_hurdat2() -> Path:
    """
    Download HURDAT 2 database.

    Return:
        A path object pointing to the download data if it isn't already present locally.
    """
    dest = Path() / "hurdat2.txt"
    if not dest.exists():
        url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt"
        response = requests.get(url)
        response.raise_for_status()
        with open(dest, "wb") as output:
            output.write(response.content)
    return dest


def parse_single_storm_xarray(filename, storm_id):
    storm_id = storm_id.strip().upper()

    with open(filename, 'r') as f:
        while True:
            header = f.readline()
            if not header:
                raise ValueError(f"Storm '{storm_id}' not found in file.")

            parts = [p.strip() for p in header.strip().split(',')]
            s_id, name, n_lines = parts[0], parts[1], int(parts[2])

            if s_id.upper() != storm_id:
                # Skip this storm
                for _ in range(n_lines):
                    f.readline()
                continue

            # Matching storm found
            times, record_ids, lats, lons, winds, pressures = [], [], [], [], [], []

            for _ in range(n_lines):
                line = f.readline().strip()
                fields = [x.strip() for x in line.split(',')]

                dt = pd.to_datetime(f"{fields[0]} {fields[1]}", format="%Y%m%d %H%M")
                lat = float(fields[4][:-1]) * (1 if fields[4][-1] == 'N' else -1)
                lon = float(fields[5][:-1]) * (1 if fields[5][-1] == 'E' else -1)
                wind = int(fields[6])
                pressure = fields[7]
                pressure = int(pressure) if pressure else np.nan
                record_id = fields[2]

                times.append(dt)
                lats.append(lat)
                lons.append(lon)
                winds.append(wind)
                pressures.append(pressure)
                record_ids.append(record_id)

            # Return as xarray
            return xr.Dataset(
                data_vars={
                    "lat": ("time", lats),
                    "lon": ("time", lons),
                    "wind": ("time", winds),
                    "pressure": ("time", pressures),
                    "record_id": ("time", record_ids),
                },
                coords={
                    "time": times,
                    "storm_id": s_id,
                    "storm_name": name,
                }
            )


def get_hurdat_track(storm_id: str) -> xr.Dataset:
    """
    Get HURDAT track for a storm with a given storm_id.
    """
    hurdat_file = download_hurdat2()
    return parse_single_storm_xarray(hurdat_file, storm_id)
