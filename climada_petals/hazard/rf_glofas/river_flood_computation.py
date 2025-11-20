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
from typing import Union, Optional, Mapping, Any, Callable, List
from contextlib import contextmanager
from datetime import datetime
from collections import namedtuple
import shutil

import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

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
from .flood_maps import (
    open_flood_maps_extents,
    download_flood_map_tiles,
    open_flood_map_tiles,
    JRC_FLOOD_HAZARD_MAP_TILES_FILENAME,
)

LOGGER = logging.getLogger(__name__)


@contextmanager
def _maybe_open_dataarray(
    arr: Optional[xr.DataArray],
    filename: Union[str, Path],
    engine="netcdf4",
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
        LOGGER.debug("Opening file: %s", filename)
        kwargs = {"engine": engine} | open_dataarray_kwargs
        arr = xr.open_dataarray(filename, **kwargs)
        try:
            yield arr
        finally:
            LOGGER.debug("Closing file: %s", filename)
            arr.close()

    else:
        yield arr


def cleanup_cache_dir(
    cache_dir: Union[Path, str] = DEFAULT_DATA_DIR / ".cache", dry_run: bool = False
):
    """Delete the contents of the cache directory"""
    directories = [path for path in Path(cache_dir).iterdir() if path.is_dir()]
    for path in directories:
        LOGGER.debug("Removing directory: %s", path)
        if not dry_run:
            shutil.rmtree(path)
    if dry_run:
        LOGGER.debug("Dry run. No files removed")


_RiverFloodCachePaths = namedtuple(
    "RiverFloodCachePaths",
    [
        "discharge",
        "return_period",
        "return_period_regrid",
        "return_period_regrid_protect",
    ],
)


class RiverFloodCachePaths(_RiverFloodCachePaths):
    """Container for storing paths to caches for :py:class:`RiverFloodInundation`

    Depending on the state of the corresponding :py:class:`RiverFloodInundation`
    instance, files might be present or not. Please check this explicitly before
    accessing them.

    Attributes
    ----------
    discharge : pathlib.Path
        Path to the discharge data cache.
    return_period : pathlib.Path
        Path to the return period data cache.
    return_period_regrid : pathlib.Path
        Path to the regridded return period data cache.
    return_period_regrid_protect : pathlib.Path
        Path to the regridded return period data cache, where the return period was
        restricted by the protection limits.
    """

    @classmethod
    def from_dir(cls, cache_dir: Path):
        """Set default paths from a cache directory"""
        return cls(
            discharge=cache_dir / "discharge.nc",
            return_period=cache_dir / "return-period.nc",
            return_period_regrid=cache_dir / "return-period-regrid.nc",
            return_period_regrid_protect=cache_dir / "return-period-regrid-protect.nc",
        )


class RiverFloodInundation:
    """Class for computing river flood inundations

    Attributes
    ----------
    store_intermediates : bool
        If intermediate results are stored in the respective :py:attr:`cache_paths`
    cache_paths : RiverFloodCachePaths
        Paths pointing to potential intermediate results stored in a cache directory.
    flood_maps : xarray.DataArray
        Flood inundation on lat/lon grid for specific return periods.
    gumbel_fits : xarray.Dataset
        Gumbel parameters resulting from extreme value analysis of historical discharge
        data.
    flopros : geopandas.GeoDataFrame
        Spatially explicit information on flood protection levels.
    regridder : xesmf.Regridder
        Storage for re-using the XESMF regridder in case the computation is repeated
        on the same grid. This reduces the runtime of subsequent computations.
    """

    def __init__(
        self,
        store_intermediates: bool = True,
        data_dir: Union[Path, str] = DEFAULT_DATA_DIR,
        cache_dir: Union[Path, str] = DEFAULT_DATA_DIR / ".cache",
    ):
        """Initialize the instance

        Parameters
        ----------
        store_intermediates : bool, optional
            Whether the data of each computation step should be stored in the cache
            directory. This is recommended especially for larger data. Only set this
            to ``False`` if the data operated on is very small (e.g., for a small
            country or region). Defaults to ``True``.
        data_dir : Path or str, optional
            The directory where flood maps, Gumbel distribution parameters and the
            FLOPROS database are located. Defaults to
            ``<climada>/data/river-flood-computation``, where ``<climada>`` is the
            Climada data directory indicated by ``local_data : system`` in the
            ``climada.conf``. This directory must exist.
        cache_dir : Path or str, optional
            The top-level cache directory where computation caches of this instance will
            be placed. Defaults to ``<climada>/data/river-flood-computation/.cache``
            (see above for configuration). This directory (and all its parents) will be
            created.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"'data_dir' does not exist: {self.data_dir}")

        self.store_intermediates = store_intermediates
        self.flood_map_tiles = open_flood_maps_extents(
            self.data_dir / JRC_FLOOD_HAZARD_MAP_TILES_FILENAME
        )
        self.flood_maps_dir = self.data_dir / "flood-maps"
        self.gumbel_fits = xr.open_dataset(
            self.data_dir / "gumbel-fit.nc",
            chunks="auto",
            engine="netcdf4",
        )
        self.flopros = gpd.read_file(
            self.data_dir / "FLOPROS_shp_V1/FLOPROS_shp_V1.shp"
        )
        self.regridder = None
        self._create_tempdir(cache_dir=cache_dir)

    def __del__(self):
        """Upon deletion, make sure the temporary directory is cleaned up"""
        # NOTE: Deletion might also happen when __init__ did not succeed/conclude!
        try:
            getattr(self, "_tempdir").cleanup()
        except AttributeError:
            pass

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
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary directory for cache
        self._tempdir = TemporaryDirectory(
            dir=cache_dir, prefix=datetime.today().strftime("%y%m%d-%H%M%S-")
        )

        # Define cache paths
        self.cache_paths = RiverFloodCachePaths.from_dir(Path(self._tempdir.name))

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
        load_flood_maps_kws: Optional[Mapping[str, Any]] = None,
        resample_kws: Optional[Mapping[str, Any]] = None,
        regrid_kws: Optional[Mapping[str, Any]] = None,
    ):
        """Compute river flood inundation from downloaded river discharge

        After downloading discharge data, this will execute the pipeline for computing
        river flood inundaton. This pipeline has the following steps:

        - Compute the equivalent return period, either with :py:meth:`return_period`, or
          :py:meth:`return_period_resample`.
        - Load the flood maps matching the extent of the discharge data with
          :py:meth:`load_flood_maps`, potentially applying coarsening on them.
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
        load_flood_maps_kws : Mapping (str, Any), optional
            Keyward arguments for :py:meth:`load_flood_maps`.
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

        # Load flood maps
        load_flood_maps_defaults = {"reference": discharge}
        if load_flood_maps_kws is not None:
            load_flood_maps_defaults.update(load_flood_maps_kws)
        flood_maps = self.load_flood_maps(**load_flood_maps_defaults)

        # Regrid
        regrid_defaults = {"flood_maps": flood_maps}
        if regrid_kws is not None:
            regrid_defaults.update(regrid_kws)
        self.regrid(**regrid_defaults)

        # Compute flood depth
        ds_flood = xr.Dataset()
        if not apply_protection or apply_protection == "both":
            ds_flood["flood_depth"] = self.flood_depth(flood_maps=flood_maps)

        # Compute flood depth with protection
        self.apply_protection()
        ds_flood["flood_depth_flopros"] = self.flood_depth(flood_maps=flood_maps)

        # Return data
        return ds_flood

    def load_flood_maps(
        self,
        reference: Optional[xr.DataArray] = None,
        coarsening: int | None = 7,
        coarsen_agg=np.mean,
        overwrite_tiles: bool = False,
    ) -> xr.DataArray:
        """Load flood hazard maps for the area represented by the given reference

        Parameters
        ----------
        reference : xr.DataArray (optional)
            The array serving as spatial reference for the flood maps. All flood map
            tiles intersecting the bounds of the reference area will be downloaded and
            opened. If ``None`` (default), the cached discharge will be opened as
            reference.
        coarsening : int or None
            How many pixels in horizontal and vertical direction will be coarsened into
            a single pixel. This reduces memory load due to the computation, but also
            reduces accuracy. The original resolution is 3 arc seconds. Defaults to 7
            (coarse resolution of 21 arc seconds). If ``None``, does not apply
            coarsening.
        coarsen_agg
            Function used for coarsening pixels. Defaults to mean.
        overwrite_tiles : bool
            Overwrite already downloaded files. Defaults to ``False``.

        Returns
        -------
        flood_maps : xr.DataArray
            Flood hazard maps for the given area reference
        """
        with _maybe_open_dataarray(
            reference, self.cache_paths.discharge, chunks="auto"
        ) as reference:
            select = self.flood_map_tiles.geometry.intersects(
                shapely.Polygon(
                    [
                        [reference["longitude"].min(), reference["latitude"].min()],
                        [reference["longitude"].max(), reference["latitude"].min()],
                        [reference["longitude"].max(), reference["latitude"].max()],
                        [reference["longitude"].min(), reference["latitude"].max()],
                    ]
                )
            )
            tiles_select = self.flood_map_tiles.loc[select]
            flood_maps_dir = self.data_dir / "flood-maps"
            download_flood_map_tiles(
                output_dir=flood_maps_dir, tiles=tiles_select, overwrite=overwrite_tiles
            )
            flood_maps = open_flood_map_tiles(
                flood_maps_dir=flood_maps_dir, tiles=tiles_select
            )
            if coarsening is not None:
                return flood_maps.coarsen(
                    longitude=coarsening, latitude=coarsening, boundary="trim"
                ).reduce(func=coarsen_agg)

            return flood_maps

    def download_forecast(
        self,
        countries: Union[str, List[str]],
        forecast_date: Union[str, np.datetime64, datetime, pd.Timestamp],
        lead_time_days: int = 10,
        preprocess: Optional[Callable] = None,
        **download_glofas_discharge_kwargs,
    ) -> xr.DataArray:
        """Download GloFAS discharge ensemble forecasts

        If :py:attr:`store_intermediates` is true, the returned data is also stored in
        :py:attr:`cache_paths`.

        Parameters
        ----------
        countries : str or list of str
            Names or codes of countries to download data for. The downloaded data will
            be a lat/lon grid covering all specified countries.
        forecast_date
            The date at which the forecast was issued. Can be defined any way that is
            compatible with ``pandas.Timestamp``, see
            https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html
        lead_time_days : int, optional
            How many days of lead time to include in the downloaded forecast. Maximum
            is 30. Defaults to 10, in which case the 10 days following the
            ``forecast_date`` are included in the download.
        preprocess
            Callable for preprocessing data while loading it. See
            https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html
        download_glofas_discharge_kwargs
            Additional arguments to
            :py:func:`climada_petals.hazard.rf_glofas.transform_ops.download_glofas_discharge`

        Returns
        -------
        forecast : xr.DataArray
            Downloaded forecast as DataArray after preprocessing

        See Also
        --------
        :py:func:`climada_petals.hazard.rf_glofas.transform_ops.download_glofas_discharge`
        """
        leadtime_hour = list(
            map(str, (np.arange(1, lead_time_days + 1, dtype=np.int_) * 24).flat)
        )
        forecast = download_glofas_discharge(
            product="forecast",
            dates=pd.DatetimeIndex([forecast_date]),
            countries=countries,
            preprocess=preprocess,
            leadtime_hour=leadtime_hour,
            split_request=False,
            **download_glofas_discharge_kwargs,
        )
        if self.store_intermediates:
            save_file(forecast, self.cache_paths.discharge, zlib=False)
        return forecast

    def download_reanalysis(
        self,
        countries: Union[str, List[str]],
        year: int,
        preprocess: Optional[Callable] = None,
        **download_glofas_discharge_kwargs,
    ):
        """Download GloFAS discharge historical data

        If :py:attr:`store_intermediates` is true, the returned data is also stored in
        :py:attr:`cache_paths`.

        Parameters
        ----------
        countries : str or list of str
            Names or codes of countries to download data for. The downloaded data will
            be a lat/lon grid covering all specified countries.
        year : int
            The year to download data for.
        preprocess
            Callable for preprocessing data while loading it. See
            https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html
        download_glofas_discharge_kwargs
            Additional arguments to
            :py:func:`climada_petals.hazard.rf_glofas.transform_ops.download_glofas_discharge`

        Returns
        -------
        reanalysis : xr.DataArray
            Downloaded forecast as DataArray after preprocessing

        See Also
        --------
        :py:func:`climada_petals.hazard.rf_glofas.transform_ops.download_glofas_discharge`
        """
        reanalysis = download_glofas_discharge(
            product="historical",
            dates=pd.date_range(f"{year}-01-01", f"{year}-12-31"),
            countries=countries,
            preprocess=preprocess,
            split_request=False,
            **download_glofas_discharge_kwargs,
        )
        if self.store_intermediates:
            save_file(reanalysis, self.cache_paths.discharge, zlib=False)
        return reanalysis

    def return_period(
        self,
        discharge: Optional[xr.DataArray] = None,
    ) -> xr.DataArray:
        """Compute the return period for a given discharge

        If no discharge data is given as parameter, the discharge cache will be
        accessed.

        If :py:attr:`store_intermediates` is true, the returned data is also stored in
        :py:attr:`cache_paths`.

        Parameters
        ----------
        discharge : xr.DataArray, optional
            The discharge data to operate on. Defaults to ``None``, which indicates that
            data should be loaded from the cache

        Returns
        -------
        r_period : xr.DataArray
            Return period for each location of the input discharge.

        See Also
        --------
        :py:func:`climada_petals.hazard.rf_glofas.transform_ops.return_period`
        """
        with _maybe_open_dataarray(
            discharge, self.cache_paths.discharge, chunks="auto"
        ) as discharge:
            r_period = return_period(
                discharge, self.gumbel_fits["loc"], self.gumbel_fits["scale"]
            )

            if self.store_intermediates:
                save_file(r_period, self.cache_paths.return_period)
            return r_period

    def return_period_resample(
        self,
        num_bootstrap_samples: int,
        discharge: Optional[xr.DataArray] = None,
        fit_method: str = "MM",
        num_workers: int = 1,
        memory_per_worker: str = "2G",
    ):
        """Compute the return period for a given discharge using bootstrap sampling.

        For each input discharge value, this creates an ensemble of return periods by
        employing bootstrap sampling. The ensemble size is controlled with
        ``num_bootstrap_samples``.

        If :py:attr:`store_intermediates` is true, the returned data is also stored in
        :py:attr:`cache_paths`.

        Parameters
        ----------
        num_bootstrap_samples : int
            Number of bootstrap samples to compute for each discharge value.
        discharge : xr.DataArray, optional
            The discharge data to operate on. Defaults to ``None``, which indicates that
            data should be loaded from the cache.
        fit_method : str, optional
            Method for fitting data to bootstrapped samples.

            * ``"MM"``: Method of Moments
            * ``"MLE"``: Maximum Likelihood Estimation

        num_workers : int, optional
            Number of parallel processes to use when computing the samples.
        memory_per_worker : str, optional
            Memory to allocate for each process.

        Returns
        -------
        r_period : xr.DataArray
            Return period samples for each location of the input discharge.

        See Also
        --------
        :py:func:`climada_petals.hazard.rf_glofas.transform_ops.return_period_resample`
        """
        # Use smaller chunks so function does not suffocate
        with _maybe_open_dataarray(
            discharge,
            self.cache_paths.discharge,
            chunks=dict(longitude=50, latitude=50),
        ) as discharge_data:
            kwargs = dict(
                discharge=discharge_data,
                gev_loc=self.gumbel_fits["loc"],
                gev_scale=self.gumbel_fits["scale"],
                gev_samples=self.gumbel_fits["samples"],
                bootstrap_samples=num_bootstrap_samples,
                fit_method=fit_method,
            )

            def work():
                r_period = return_period_resample(**kwargs)
                if self.store_intermediates:
                    save_file(r_period, self.cache_paths.return_period, zlib=False)
                return r_period

            if num_workers > 1:
                with dask_client(num_workers, 1, memory_per_worker):
                    return work()
            else:
                return work()

    def regrid(
        self,
        r_period: Optional[xr.DataArray] = None,
        flood_maps: Optional[xr.DataArray] = None,
        method: str = "bilinear",
        reuse_regridder: bool = False,
    ):
        """Regrid the return period data onto the flood hazard map grid.

        This computes the regridding matrix for the given coordinates and then performs
        the actual regridding. The matrix is stored in :py:attr:`regridder`. If
        another regridding is performed on the same grid (but possibly different data),
        the regridder can be reused to save time. To control that, set
        ``reuse_regridder=True``.

        If :py:attr:`store_intermediates` is true, the returned data is also stored in
        :py:attr:`cache_paths`.

        Parameters
        ----------
        r_period : xr.DataArray, optional
            The return period data to regrid. Defaults to ``None``, which indicates that
            data should be loaded from the cache.
        flood_maps : xr.DataArray, optional
            The flood maps to use for regridding. Defaults to ``None``, which means that
            flood maps compatible to ``r_period`` will be downloaded and opened.
        method : str, optional
            Interpolation method of the return period data. Defaults to ``"bilinear"``.
            See https://xesmf.readthedocs.io/en/stable/notebooks/Compare_algorithms.html
        reuse_regridder : bool, optional
            Reuse the regridder stored if one is stored. Defaults to ``False``, which
            means that a new regridder is always built when calling this function.
            If ``True``, and no regridder is stored, it will be built nonetheless.

        Returns
        -------
        return_period_regrid : xr.DataArray
            The regridded return period data.

        See Also
        --------
        :py:func:`climada_petals.hazard.rf_glofas.transform_ops.regrid`
        """
        # NOTE: Chunks must be small because resulting array is huge!
        with _maybe_open_dataarray(
            r_period,
            self.cache_paths.return_period,
            chunks={
                "longitude": -1,
                "latitude": -1,
                "time": "auto",
                "sample": "auto",
                "number": "auto",
                "step": "auto",
            },
        ) as return_period_data:
            if flood_maps is None:
                flood_maps = self.load_flood_maps(reference=return_period_data)
            if not reuse_regridder:
                self.regridder = None

            return_period_regrid, self.regridder = regrid(
                return_period_data,
                flood_maps,
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
        """Limit the return period data by applying FLOPROS protection levels.

        This sets each return period value where the local FLOPROS protection level is
        not exceeded to NaN and returns the result. Protection levels are read from
        :py:attr:`flopros`.

        If :py:attr:`store_intermediates` is true, the returned data is also stored in
        :py:attr:`cache_paths`.

        Parameters
        ----------
        return_period_regrid : xr.DataArray, optional
            The return period data to regrid. Defaults to ``None``, which indicates that
            data should be loaded from the cache.

        Returns
        -------
        return_period_regrid_protect : xr.DataArray
            The regridded return period where each value that does not reach the
            protection limit is set to NaN.

        See Also
        --------
        :py:func:`climada_petals.hazard.rf_glofas.transform_ops.apply_flopros`
        """
        with _maybe_open_dataarray(
            return_period_regrid, self.cache_paths.return_period_regrid, chunks="auto"
        ) as return_period_regrid_data:
            return_period_regrid_protect = apply_flopros(
                self.flopros, return_period_regrid_data
            )

            if self.store_intermediates:
                save_file(
                    return_period_regrid_protect,
                    self.cache_paths.return_period_regrid_protect,
                    zlib=False,
                )
            return return_period_regrid_protect

    def flood_depth(
        self,
        return_period_regrid: Optional[xr.DataArray] = None,
        flood_maps: Optional[xr.DataArray] = None,
    ):
        """Compute the flood depth from regridded return period data.

        Interpolate the flood hazard maps stored in :py:attr`flood_maps` in the return
        period dimension at every location to compute the flood footprint.

        Note
        ----
        Even if :py:attr:`store_intermediates` is true, the returned data is **not**
        stored automatically! Use
        :py:func:`climada_petals.hazard.rf_glofas.transform_ops.save_file` to store
        the data yourself.

        Parameters
        ----------
        return_period_regrid : xr.DataArray, optional
            The regridded return period data to use for computing the flood footprint.
            Defaults to ``None`` which indicates that data should be loaded from the
            cache. If :py:attr:`RiverFloodCachePaths.return_period_regrid_protect`
            exists, that data is used. Otherwise, the "unprotected" data
            :py:attr:`RiverFloodCachePaths.return_period_regrid` is loaded.
        flood_maps : xr.DataArray, optional
            The flood maps to use for regridding. Defaults to ``None``, which means that
            flood maps compatible to ``return_period_regrid`` will be downloaded and
            opened.

        Returns
        -------
        inundation : xr.DataArray
            The flood inundation at every location of the flood hazard maps grid.
        """
        file_path = self.cache_paths.return_period_regrid
        if (
            return_period_regrid is None
            and self.cache_paths.return_period_regrid_protect.is_file()
        ):
            file_path = self.cache_paths.return_period_regrid_protect

        with _maybe_open_dataarray(
            return_period_regrid, file_path, chunks="auto"
        ) as return_period_regrid_data:
            if flood_maps is None:
                flood_maps = self.load_flood_maps(reference=return_period_regrid)

            inundation = flood_depth(return_period_regrid_data, flood_maps)
            return inundation
