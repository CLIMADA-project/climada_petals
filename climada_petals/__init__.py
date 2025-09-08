"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

climada init
"""
from shutil import copyfile
from pathlib import Path

import climada
import climada_petals.util.config
from .util.constants import *

REPO_DATA = {
    'climada_petals/data/gdp_asset': [
        SYSTEM_DIR.joinpath('GDP2Asset_converter_2.5arcmin.nc'),
        DEMO_GDP2ASSET,
    ],
    'climada_petals/data/river_flood': [
        HAZ_DEMO_FLDDPH,
        HAZ_DEMO_FLDFRC,
    ],
    'climada_petals/data/crop_production': [
        DEMO_DIR.joinpath('crop_production_demo_data_yields_CHE.nc4'),
        DEMO_DIR.joinpath('crop_production_demo_data_cultivated_area_CHE.nc4'),
        DEMO_DIR.joinpath('FAOSTAT_data_producer_prices.csv'),
        DEMO_DIR.joinpath('FAOSTAT_data_production_quantity.csv'),
        DEMO_DIR.joinpath('hist_mean_mai-firr_1976-2005_DE_FR.hdf5'),
        DEMO_DIR.joinpath('histsoc_landuse-15crops_annual_FR_DE_DEMO_2001_2005.nc'),
    ],
    'climada_petals/data/relative_cropyield': [
        DEMO_DIR.joinpath('gepic_gfdl-esm2m_ewembi_historical_2005soc_co2_yield-whe-noirr_global_DEMO_TJANJIN_annual_1861_2005.nc'),
        DEMO_DIR.joinpath('pepic_miroc5_ewembi_historical_2005soc_co2_yield-whe-firr_global_annual_DEMO_TJANJIN_1861_2005.nc'),
        DEMO_DIR.joinpath('pepic_miroc5_ewembi_historical_2005soc_co2_yield-whe-noirr_global_annual_DEMO_TJANJIN_1861_2005.nc'),
        DEMO_DIR.joinpath('lpjml_ipsl-cm5a-lr_ewembi_historical_2005soc_co2_yield-whe-noirr_annual_FR_DE_DEMO_1861_2005.nc'),
    ],
    'climada_petals/data/tc_surge_bathtub': [
        DEMO_DIR.joinpath('SRTM15+V2.0_sample.tiff'),
    ],
    'climada_petals/data/wildfire': [
        DEMO_DIR.joinpath('Portugal_firms_June_2017.csv'),
        DEMO_DIR.joinpath('Portugal_firms_2016_17_18_MODIS.csv'),
    ],
}


def copy_repo_data(reload=False):
    for src_dir, path_list in REPO_DATA.items():
        for path in path_list:
            if not path.exists() or reload:
                src = Path(__file__).parent.parent.joinpath(src_dir, path.name)
                copyfile(src, path)

copy_repo_data()
