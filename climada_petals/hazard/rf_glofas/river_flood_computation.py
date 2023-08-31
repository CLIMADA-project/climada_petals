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
from contextlib import contextmanager

import xarray as xr
import geopandas as gpd
import numpy as np
import xesmf as xe

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


@contextmanager
def maybe_open_dataarray(
    arr: Optional[xr.DataArray] = None,
    *open_dataarray_args,
    **open_dataarray_kwargs,
):
    """Create a context for opening an xarray file or yielding the input array"""
    if arr is None:
        print(f"Opening file: {open_dataarray_args}")
        arr = xr.open_dataarray(*open_dataarray_args, **open_dataarray_kwargs)
        try:
            yield arr
        finally:
            print(f"Closing file: {open_dataarray_args}")
            arr.close()

    else:
        try:
            yield arr
        finally:
            pass


class RiverFloodInundation:
    """Class for computing river flood inundations"""

    # Definitions for intermediate file paths

    def __init__(
        self,
        store_intermediates: bool = True,
        cache_dir: Optional[Union[Path, str]] = None,
    ):
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

        self.regridder = None
        self._create_tempdir(cache_dir=cache_dir)

    def _create_tempdir(self, cache_dir):
        self._tempdir = TemporaryDirectory(
            prefix=".cache-", dir=cache_dir or DEFAULT_DATA_DIR
        )
        self.cache_dir = Path(self._tempdir.name)

        self._DISCHARGE_PATH = self.cache_dir / "discharge.nc"
        self._RETURN_PERIOD_PATH = self.cache_dir / "return-period.nc"
        self._RETURN_PERIOD_REGRID_PATH = self.cache_dir / "return-period-regrid.nc"
        self._RETURN_PERIOD_REGRID_PROTECT_PATH = (
            self.cache_dir / "return-period-regrid-protect.nc"
        )

    def clear_cache(self):
        self._tempdir.cleanup()
        self._create_tempdir()

    def run(self, download_type, download_kws=None, resample_kws=None):
        # Download data
        if download_kws is None:
            download_kws = {}
        if download_type == "forecast":
            self.download_forecast(**download_kws)
        elif download_type == "reanalysis":
            self.download_reanalysis(**download_kws)
        else:
            raise RuntimeError(f"Unknown download type: {download_type}")

        # Compute return period
        if resample_kws is None:
            self.return_period()
        else:
            self.return_period_resample(**resample_kws)

        # Regrid
        self.regrid()

        # Compute inundations
        inundation = self.flood_depth()
        self.apply_protection()
        inundation_protected = self.flood_depth()

        # Return data
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

    def download_reanalysis(
        self,
        countries: Union[str, Iterable[str]],
        year: str,
        preprocess: Optional[str] = None,
    ):
        reanalysis = download_glofas_discharge(
            product="historical",
            date_from=year,
            date_to=None,
            countries=countries,
            preprocess=preprocess,
        )
        if self.store_intermediates:
            reanalysis.to_netcdf(self._DISCHARGE_PATH)
        return reanalysis

    def return_period(
        self,
        discharge: Optional[xr.DataArray] = None,
    ):
        with maybe_open_dataarray(
            discharge, self._DISCHARGE_PATH, chunks="auto"
        ) as discharge:
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
        memory_per_worker: str = "2G",
    ):
        # Use smaller chunks so function does not suffocate
        with maybe_open_dataarray(
            discharge, self._DISCHARGE_PATH, chunks=dict(longitude=50, latitude=50)
        ) as discharge:

            kwargs = dict(
                discharge=discharge,
                gev_loc=self.gumbel_fits["loc"],
                gev_scale=self.gumbel_fits["scale"],
                gev_samples=self.gumbel_fits["samples"],
                bootstrap_samples=num_bootstrap_samples,
                fit_method=fit_method,
            )

            def work():
                rp = return_period_resample(**kwargs)
                if self.store_intermediates:
                    rp.to_netcdf(self._RETURN_PERIOD_PATH, engine="netcdf4")
                return rp

            if num_workers > 1:
                with dask_client(num_workers, 1, memory_per_worker):
                    return work()
            else:
                return work()

    def regrid(
        self,
        return_period: Optional[xr.DataArray] = None,
        method: str = "bilinear",
        reset_regridder: bool = True,
    ):
        with maybe_open_dataarray(
            return_period,
            self._RETURN_PERIOD_PATH,
            chunks=dict(longitude=-1, latitude=-1, time=1, sample=1, number=1, step=1),
        ) as return_period:

            if reset_regridder:
                self._regridder = None
            return_period_regrid, self._regridder = regrid(
                return_period,
                self.flood_maps,
                method=method,
                regridder=self._regridder,
                return_regridder=True,
            )

            if self.store_intermediates:
                return_period_regrid.to_netcdf(self._RETURN_PERIOD_REGRID_PATH)
            return return_period_regrid

    def apply_protection(self, return_period_regrid: Optional[xr.DataArray] = None):
        with maybe_open_dataarray(
            return_period_regrid, self._RETURN_PERIOD_REGRID_PATH, chunks="auto"
        ) as return_period_regrid:

            return_period_regrid_protect = apply_flopros(
                self.flopros, return_period_regrid
            )

            if self.store_intermediates:
                return_period_regrid_protect.to_netcdf(
                    self._RETURN_PERIOD_REGRID_PROTECT_PATH
                )
            return return_period_regrid_protect

    def flood_depth(self, return_period_regrid: Optional[xr.DataArray] = None):
        file_path = (
            self._RETURN_PERIOD_REGRID_PROTECT_PATH
            if self._RETURN_PERIOD_REGRID_PROTECT_PATH.is_file()
            else self._RETURN_PERIOD_REGRID_PATH
        )

        with maybe_open_dataarray(
            return_period_regrid, file_path, chunks="auto"
        ) as return_period_regrid:
            inundation = flood_depth(return_period_regrid, self.flood_maps)
            return inundation
