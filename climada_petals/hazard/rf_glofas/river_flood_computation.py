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

Top-level computation class for river flood inundation
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import logging
from typing import Iterable, Union, Optional

import xarray as xr
import geopandas as gpd
import numpy as np

from .rf_glofas import DEFAULT_DATA_DIR, dask_client
from .transform_ops import (
    download_glofas_discharge,
    return_period,
    return_period_resample,
    regrid,
    apply_flopros,
    flood_depth,
)

LOGGER = logging.getLogger(__file__)


class RiverFloodInundation:
    """Class for computing river flood inundations"""

    # Definitions for intermediate file paths

    def __init__(self, store_intermediates: bool = True):
        self.store_intermediates = store_intermediates
        self.flood_maps = xr.open_dataarray(
            DEFAULT_DATA_DIR / "flood-maps.nc",
            chunks=dict(return_period=-1, latitude="auto", longitude="auto"),
        )
        self.gumbel_fits = xr.open_dataset(
            DEFAULT_DATA_DIR / "gumbel-fit.nc", chunks="auto"
        )
        self.flopros = gpd.read_file(
            DEFAULT_DATA_DIR / "FLOPROS_shp_V1/FLOPROS_shp_V1.shp"
        )
        self._tempdir = TemporaryDirectory(prefix=".cache-", dir=DEFAULT_DATA_DIR)
        self.cache_dir = Path(self._tempdir.name)

        self._DISCHARGE_PATH = self.cache_dir / "discharge.nc"
        self._RETURN_PERIOD_PATH = self.cache_dir / "return-period.nc"
        self._RETURN_PERIOD_REGRID_PATH = self.cache_dir / "return-period-regrid.nc"
        self._RETURN_PERIOD_REGRID_PROTECT_PATH = (
            self.cache_dir / "return-period-regrid-protect.nc"
        )

    def run(self, **download_kwargs):
        self.download_forecast(**download_kwargs)
        self.return_period()
        self.regrid()
        inundation = self.flood_depth()

        self.apply_protection()
        inundation_protected = self.flood_depth()

        return xr.Dataset(
            dict(flood_depth=inundation, flood_depth_flopros=inundation_protected)
        )

    def download_forecast(
        self,
        countries: Union[str, Iterable[str]],
        forecast_date: str,
        lead_time_days: int = 10,
        preprocess: Optional[str] = None,
    ):
        leadtime_hour = list(
            map(str, (np.arange(lead_time_days, dtype=np.int_) * 24).flat)
        )
        forecast = download_glofas_discharge(
            product="forecast",
            date_from=forecast_date,
            date_to=None,
            countries=countries,
            preprocess=preprocess,
            leadtime_hour=leadtime_hour,
        )
        if self.store_intermediates:
            forecast.to_netcdf(self._DISCHARGE_PATH)
        return forecast

    def return_period(
        self,
        discharge: Optional[xr.DataArray] = None,
    ):
        if discharge is None:
            discharge = xr.open_dataarray(self._DISCHARGE_PATH, chunks="auto")
        rp = return_period(
            discharge, self.gumbel_fits["loc"], self.gumbel_fits["scale"]
        )

        if self.store_intermediates:
            rp.to_netcdf(self._RETURN_PERIOD_PATH)
        return rp

    def return_period_resample(
        self,
        num_bootstrap_samples: int,
        discharge: Optional[xr.DataArray] = None,
        fit_method: str = "MM",
        num_workers: int = 1,
    ):
        # Use smaller chunks so function does not suffocate
        if discharge is None:
            discharge = xr.open_dataarray(
                self._DISCHARGE_PATH, chunks=dict(longitude=50, latitude=50)
            )

        kwargs = dict(
            discharge=discharge,
            gev_loc=self.gumbel_fits["loc"],
            gev_scale=self.gumbel_fits["scale"],
            gev_samples=self.gumbel_fits["samples"],
            bootstrap_samples=num_bootstrap_samples,
            fit_method=fit_method,
        )

        if num_workers > 1:
            with dask_client(num_workers, 1, "2GB"):
                rp = return_period_resample(**kwargs)
        else:
            rp = return_period_resample(**kwargs)

        if self.store_intermediates:
            rp.to_netcdf(self._RETURN_PERIOD_PATH)
        return rp

    def regrid(
        self, return_period: Optional[xr.DataArray] = None, method: str = "bilinear"
    ):
        if return_period is None:
            return_period = xr.open_dataarray(
                self._RETURN_PERIOD_PATH,
                chunks=dict(
                    longitude=-1, latitude=-1, time=1, sample=1, number=1, step=1
                ),
            )

        return_period_regrid = regrid(return_period, self.flood_maps, method=method)
        if self.store_intermediates:
            return_period_regrid.to_netcdf(self._RETURN_PERIOD_REGRID_PATH)
        return return_period_regrid

    def apply_protection(self, return_period_regrid: Optional[xr.DataArray] = None):
        if return_period_regrid is None:
            return_period_regrid = xr.open_dataarray(
                self._RETURN_PERIOD_REGRID_PATH, chunks="auto"
            )

        return_period_regrid_protect = apply_flopros(self.flopros, return_period_regrid)

        if self.store_intermediates:
            return_period_regrid_protect.to_netcdf(
                self._RETURN_PERIOD_REGRID_PROTECT_PATH
            )
        return return_period_regrid_protect

    def flood_depth(self, return_period_regrid: Optional[xr.DataArray] = None):
        if return_period_regrid is None:
            if self._RETURN_PERIOD_REGRID_PROTECT_PATH.is_file():
                return_period_regrid = xr.open_dataarray(
                    self._RETURN_PERIOD_REGRID_PROTECT_PATH, chunks="auto"
                )
            else:
                return_period_regrid = xr.open_dataarray(
                    self._RETURN_PERIOD_REGRID_PATH, chunks="auto"
                )

        inundation = flood_depth(return_period_regrid, self.flood_maps)
        return inundation
