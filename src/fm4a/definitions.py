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
    "Z0M": "m "
}
