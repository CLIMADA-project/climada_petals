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
from pathlib import Path
from tempfile import TemporaryDirectory
import logging
from typing import Iterable, Union, Optional, Mapping, Any, Callable
from contextlib import contextmanager
from datetime import datetime
from collections import namedtuple

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
    save_file,
)

LOGGER = logging.getLogger(__file__)


@contextmanager
def maybe_open_dataarray(
    arr: Optional[xr.DataArray],
    filename: Union[str, Path],
    **open_dataarray_kwargs,
):
    """Create a context for opening an xarray file or yielding the input array

    This will open the file with ``xr.open_dataarray``

    Parameters
    ----------
    arr : xr.DataArray or None
        The input array. If this is *not* ``None`` it is simply returned.
    filename : Path or str
        The file to open as data array if ``arr`` is ``None``.
    open_dataarray_kwargs
        Keyword arguments passed to ``xr.open_dataarray``.

    Returns
    -------
    xr.DataArray
        Either ``arr`` or the array at ``filename``. If a file was opened, it will be
        closed when this context manager closes.
    """
    if arr is None:
        LOGGER.debug(f"Opening file: {filename}")
        arr = xr.open_dataarray(filename, **open_dataarray_kwargs)
        try:
            yield arr
        finally:
            LOGGER.debug(f"Closing file: {filename}")
            arr.close()

    else:
        yield arr


RiverFloodCachePaths = namedtuple(
    "RiverFloodCachePaths",
    [
        "discharge",
        "return_period",
        "return_period_regrid",
        "return_period_regrid_protect",
        "flood_depth",
        "flood_depth_protect",
    ],
)


class RiverFloodInundation:
    """Class for computing river flood inundations"""

    # Definitions for intermediate file paths
    def __init__(
        self,
        store_intermediates: bool = True,
        cache_dir: Union[Path, str] = DEFAULT_DATA_DIR / ".cache",
    ):
        """Initialize the instance

        Parameters
        ----------
        store_intermediates : bool (optional)
            Whether the data of each computation step should be stored in the cache
            directory. This is recommended especially for larger data. Only set this
            to ``False`` if the data operated on is very small (e.g., for a small
            country or region). Defaults to ``True``.
        cache_dir : Path or str (optional)
            The top-level cache directory where computation caches of this instance will
            be placed. Defaults to ``<climada>/data/river-flood-computation``, where
            ``<climada>`` is the Climada data directory indicated by
            ``local_data : system`` in the ``climada.conf``.
        """
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

    def _create_tempdir(self, cache_dir: Union[Path, str]):
        """Create a temporary directory inside the top-level cache dir

        Parameters
        ----------
        cache_dir : Path or str
            The directory where caches are placed. Each cache is a temporary
            subdirectory of ``cache_dir``. If the path does not exist, it will be
            created, including all parent directories.
        """
        # Create cache directory
        cache_dir = Path(cache_dir)
        if not cache_dir.is_dir():
            cache_dir.mkdir(parents=True)

        # Create temporary directory for cache
        self._tempdir = TemporaryDirectory(
            dir=cache_dir, prefix=datetime.today().strftime("%y%m%d-%H%M%S-")
        )

        # Define paths
        tempdir = Path(self._tempdir.name)
        self.cache_paths = RiverFloodCachePaths(
            discharge=tempdir / "discharge.nc",
            return_period=tempdir / "return-period.nc",
            return_period_regrid=tempdir / "return-period-regrid.nc",
            return_period_regrid_protect=tempdir / "return-period-regrid-protect.nc",
            flood_depth=tempdir / "flood-depth.nc",
            flood_depth_protect=tempdir / "flood-depth-protect.nc",
        )

    def clear_cache(self):
        """Clear the cache of this instance

        This will delete the temporary cache directory and create a new one by calling
        :py:meth:`_create_tempdir`.
        """
        cache_dir = Path(self._tempdir.name).parent
        self._tempdir.cleanup()
        self._create_tempdir(cache_dir=cache_dir)

    def compute(
        self,
        discharge: Optional[xr.DataArray] = None,
        apply_protection: Union[bool, str] = "both",
        resample_kws: Optional[Mapping[str, Any]] = None,
        regrid_kws: Optional[Mapping[str, Any]] = None,
    ):
        """Compute river flood inundation from downloaded river discharge

        After downloading discharge data, this will execute the pipeline for computing
        river flood inundaton. This pipeline has the following steps:

        - Compute the equivalent return period, either with :py:meth:`return_period`, or
          :py:meth:`return_period_resample`.
        - Regrid the return period data onto the grid of the flood hazard maps with
          :py:meth:`regrid`.
        - *Optional*: Apply the protection layer with :py:meth:`apply_protection`.
        - Compute the flood depth by interpolating flood hazard maps with
          :py:meth:`flood_depth`.

        Resampling, regridding, and the application of protection information are
        controlled via the parameters of this method.

        Parameters
        ----------
        discharge : xr.DataArray or None (optional)
            The discharge data to compute flood depths for. If ``None``, the cached
            discharge will be used. Defaults to ``None``.
        apply_protection : bool or "both" (optional)
            If the stored protection layer should be considered when computing the flood
            depth. If ``"both"``, this method will return a dataset with two flood depth
            arrays. Defaults to ``both``.
        resample_kws : Mapping (str, Any) or None (optional)
            Keyword arguments for :py:meth:`return_period_resample`. If ``None``,
            this method will call :py:meth:`return_period`. Otherwise, it will call
            :py:meth:`return_period_resample` and pass this parameter as keyword
            arguments. Defaults to ``None``.
        regrid_kws : Mapping (str, Any) or None (optional)
            Keyword arguments for :py:meth:`regrid`. Defaults to ``None``.

        Returns
        -------
        xr.Dataset
            Dataset containing the flood data with the same dimensions as the input
            discharge data. Depending on the choice of ``apply_protection``, this will
            contain one or two DataArrays.

        Raises
        ------
        RuntimeError
            If ``discharge`` is ``None``, but no discharge data is cached.
        """
        if discharge is None and not self.cache_paths.discharge.is_file():
            raise RuntimeError(
                "No discharge data. Please download a discharge with this object "
                "first or supply the data as argument to this function"
            )

        # Compute return period
        if resample_kws is None:
            self.return_period(discharge=discharge)
        else:
            self.return_period_resample(discharge=discharge, **resample_kws)

        # Regrid
        regrid_kws = regrid_kws if regrid_kws is not None else {}
        self.regrid(**regrid_kws)

        # Compute flood depth
        ds_flood = xr.Dataset()
        if not apply_protection or apply_protection == "both":
            ds_flood["flood_depth"] = self.flood_depth()

        # Compute flood depth with protection
        self.apply_protection()
        ds_flood["flood_depth_flopros"] = self.flood_depth()

        # Return data
        return ds_flood

    def download_forecast(
        self,
        countries: Union[str, Iterable[str]],
        forecast_date: str,
        lead_time_days: int = 10,
        preprocess: Optional[str] = None,
        **download_glofas_discharge_kwargs,
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
            **download_glofas_discharge_kwargs,
        )
        if self.store_intermediates:
            save_file(forecast, self.cache_paths.discharge, zlib=False)
        return forecast

    def download_reanalysis(
        self,
        countries: Union[str, Iterable[str]],
        year: str,
        preprocess: Optional[str] = None,
        **download_glofas_discharge_kwargs,
    ):
        reanalysis = download_glofas_discharge(
            product="historical",
            date_from=year,
            date_to=None,
            countries=countries,
            preprocess=preprocess,
            **download_glofas_discharge_kwargs,
        )
        if self.store_intermediates:
            save_file(reanalysis, self.cache_paths.discharge)
        return reanalysis

    def return_period(
        self,
        discharge: Optional[xr.DataArray] = None,
    ):
        with maybe_open_dataarray(
            discharge, self.cache_paths.discharge, chunks="auto"
        ) as discharge:
            rp = return_period(
                discharge, self.gumbel_fits["loc"], self.gumbel_fits["scale"]
            )

            if self.store_intermediates:
                save_file(rp, self.cache_paths.return_period)
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
            discharge,
            self.cache_paths.discharge,
            chunks=dict(longitude=50, latitude=50),
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
                    save_file(rp, self.cache_paths.return_period, zlib=False)
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
        # NOTE: Chunks must be small because resulting array is huge!
        with maybe_open_dataarray(
            return_period,
            self.cache_paths.return_period,
            chunks=dict(longitude=-1, latitude=-1, time=1, sample=1, number=1, step=1),
        ) as return_period:

            if reset_regridder:
                self.regridder = None
            return_period_regrid, self.regridder = regrid(
                return_period,
                self.flood_maps,
                method=method,
                regridder=self.regridder,
                return_regridder=True,
            )

            if self.store_intermediates:
                save_file(
                    return_period_regrid,
                    self.cache_paths.return_period_regrid,
                    zlib=False,
                )
            return return_period_regrid

    def apply_protection(self, return_period_regrid: Optional[xr.DataArray] = None):
        with maybe_open_dataarray(
            return_period_regrid, self.cache_paths.return_period_regrid, chunks="auto"
        ) as return_period_regrid:

            return_period_regrid_protect = apply_flopros(
                self.flopros, return_period_regrid
            )

            if self.store_intermediates:
                save_file(
                    return_period_regrid_protect,
                    self.cache_paths.return_period_regrid_protect,
                )
            return return_period_regrid_protect

    def flood_depth(self, return_period_regrid: Optional[xr.DataArray] = None):
        file_path = self.cache_paths.return_period_regrid
        store_path = self.cache_paths.flood_depth
        if return_period_regrid is None:
            if self.cache_paths.return_period_regrid_protect.is_file():
                file_path = self.cache_paths.return_period_regrid_protect
                store_path = self.cache_paths.flood_depth_protect

        with maybe_open_dataarray(
            return_period_regrid, file_path, chunks="auto"
        ) as return_period_regrid:
            inundation = flood_depth(return_period_regrid, self.flood_maps)

            if self.store_intermediates:
                save_file(inundation, store_path)
            return inundation

    # TODO: Remove
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
