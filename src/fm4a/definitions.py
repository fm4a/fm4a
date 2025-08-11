"""
fm4a.definitions
================

Defines basic settings for the Prithvi-WxC model.
"""

LEVELS = [
    34.0, 39.0, 41.0, 43.0, 44.0, 45.0, 48.0, 51.0, 53.0, 56.0, 63.0, 68.0, 71.0, 72.0
][::-1]

SURFACE_VARS = [
    "EFLUX",
    "GWETROOT",
    "HFLUX",
    "LAI",
    "LWGAB",
    "LWGEM",
    "LWTUP",
    "PS",
    "QV2M",
    "SLP",
    "SWGNT",
    "SWTNT",
    "T2M",
    "TQI",
    "TQL",
    "TQV",
    "TS",
    "U10M",
    "V10M",
    "Z0M"
]

STATIC_SURFACE_VARS = [
    "FRACI",
    "FRLAND",
    "FROCEAN",
    "PHIS"
]

VERTICAL_VARS = [
    "CLOUD",
    "H",
    "OMEGA",
    "PL",
    "QI",
    "QL",
    "QV",
    "T",
    "U",
    "V"
]

NAN_VALS = {
    "GWETROOT": 1.0,
    "LAI": 0.0,
}


LONG_NAMES = {
    "EFLUX": "Total Latent Energy Flux",
    "GWETROOT": "Root Zone Soil Wetness",
    "HFLUX": "Sensible Heat Flux from Turbulence",
    "LAI": "Leaf Area Index",
    "LWGAB": "Surface Absorbed Longwave Radiation",
    "LWGEM": "Surface Emitted Longwave Radiation",
    "LWTUP": "TOA Upwelling Longwave Radiation",
    "PS": "Surface Pressure",
    "QV2M": "Specific Humidity at 2 m",
    "SLP": "Sea Level Pressure",
    "SWGNT": "Surface Net Downward Shortwave Flux",
    "SWTNT": "TOA Net Shortwave Downward Flux",
    "T2M": "Air Temperature at 2 m",
    "TQI": "Ice Water Path",
    "TQL": "Liquid Water Path",
    "TQV": "Total Precipitable Water Vapor",
    "TS": "Surface Skin Temperature",
    "U10M": "Eastward Wind at 10 Meters",
    "V10M": "Northward Wind at 10 Meters",
    "Z0M": "Surface Roughness Length for Momentum",
    "CLOUD": "Cloud Fraction",
    "H": "Geopotential Mid-level Height",
    "OMEGA": "Vertical Pressure Velocity",
    "PL": "Mid-level pressure",
    "QI": "Ice Water Mixing Ratio",
    "QL": "Liquid Water Mixing Ratio",
    "QV": "Specific Humidity",
    "T": "Air Temperature",
    "U": "Eastward Wind",
    "V": "Northward Wind",
    "LAT": "Latitude",
    "LON": "Longitude",
    "SIN_DOY": "Sine of the day of the year",
    "COS_DOY": "Cosine of the day of the year",
    "SIN_HOD": "Sine of the hour of the day",
    "COS_HOD": "Cosine of the hour of the day",
    "FRACI": "Sea Ice Fraction",
    "FRLAND": "Land Fraction",
    "FROCEAN": "Ocean Fraction",
    "PHIS": "Surface Geopotential Height"
}

UNITS = {
    "EFLUX": "W/m²",
    "GWETROOT": "",
    "HFLUX": "W/m²",
    "LAI": "m²/m²",
    "LWGAB": "W/m²",
    "LWGEM":  "W/m²",
    "LWTUP": "W/m²",
    "PS":  "Pa",
    "QV2M": "kg/kg",
    "SLP": "Pa",
    "SWGNT": "W/m²",
    "SWTNT": "W/m²",
    "T2M": "K",
    "TQI": "kg/m²",
    "TQL": "kg/m²",
    "TQV": "kg/m²",
    "TS": "K",
    "U10M": "m/s",
    "V10M": "m/s",
    "Z0M": "m ",
    "CLOUD": "",
    "H": "m",
    "OMEGA": "Pa/s",
    "PL": "Pa",
    "QI": "kg/kg",
    "QL": "kg/kg",
    "QV": "kg/kg",
    "T": "K",
    "U": "m/s",
    "V": "m/s",
    "LAT": "rad",
    "LON": "rad",
    "SIN_DOY": "",
    "COS_DOY": "",
    "SIN_HOD": "",
    "COS_HOD": "",
    "FRACI": "",
    "FRLAND": "",
    "FROCEAN": "",
    "PHIS": "m"
}


def get_dynamic_variable_name(index: int) -> str:
    """
    Get the variable name of one of the 160 variables included in the Prithvi-WxC input.

    Args:
        index: The index defining the variables.

    Return the name of the variables.
    """
    while index < 0:
        index = index + 160
    if index < 20:
        return SURFACE_VARS[index]

    prof_index = (index - 20) // 14
    return VERTICAL_VARS[index]


def get_static_variable_name(index: int) -> str:
    """
    Get the variable name of one of the 160 variables included in the Prithvi-WxC input.

    Args:
        index: The index defining the variables.

    Return the name of the variables.
    """
    while index < 0:
        index = index + 160
    if index < 20:
        return SURFACE_VARS[index]

    prof_index = (index - 20) // 14
    return VERTICAL_VARS[index]
