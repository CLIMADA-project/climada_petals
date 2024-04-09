"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Function factories for reading sea levels from NetCDF files
"""

import logging
import pathlib
from typing import Any, Callable, Optional, Tuple, Union

import dask
import numpy as np
import pandas as pd
import xarray as xr

import climada.util.coordinates as u_coord


LOGGER = logging.getLogger(__name__)


def sea_level_from_nc(
    path : Union[pathlib.Path, str],
    t_agg : str = "mean",
    t_pad : Optional[np.timedelta64] = None,
    mod_zos : float = 0.0,
) -> Callable:
    """Generate a function that reads centroid sea levels from a NetCDF file

    The function that is generated can be used as an input for the `sea_level` parameter in
    `TCSurgeGeoClaw.from_tc_tracks`.

    The grid cell closest to the area's centroid that has valid entries is identified. Then the
    specified aggregation method (e.g. "mean" or "max") is applied over the time period.

    Parameters
    ----------
    path : Path or str
        Path to NetCDF file containing gridded sea level data with time resolution.
    t_agg : str, optional
        Aggregation method to apply over the time period. Supported methods: "mean", "min", "max".
        Default: "mean"
    t_pad : np.timedelta64, optional
        Padding to add around the time period. Default: 0.
    mod_zos : float, optional
        The scalar sea level rise is added to the sea level value that is extracted from the
        specified NetCDF file. Default: 0

    Returns
    -------
    fun : function (tuple, tuple) -> float
        The first argument is a tuple of floats (lon_min, lat_min, lon_max, lat_max), the second
        argument is a pair of np.datetime64 (start, end). The function returns the mean sea level
        in the specified region and time period.
    """
    t_agg = t_agg.lower()
    if t_agg not in ["mean", "min", "max"]:
        raise ValueError(f"Aggregation method not supported: {t_agg}")
    _sea_level_nc_info(path)
    def sea_level_fun(bounds, period, path=path, t_agg=t_agg, t_pad=t_pad, mod_zos=mod_zos):
        t_pad = np.timedelta64(0, "D") if t_pad is None or t_pad == 0 else t_pad
        period = (period[0] - t_pad, period[1] + t_pad)
        centroid = (0.5 * (bounds[0] + bounds[2]), 0.5 * (bounds[1] + bounds[3]))
        with xr.open_dataset(path) as ds:
            da_zos = _nc_rename_vars(ds)["zos"]
            da_period = [_get_closest_date_in_index(da_zos["time"], t) for t in period]
            da_zos = da_zos.sel(
                time=(da_zos["time"] >= da_period[0]) & (da_zos["time"] <= da_period[1])
            )
            lon, lat = _get_closest_valid_cell(da_zos, *centroid)
            da_zos = da_zos.sel(lon=lon, lat=lat)
            v_agg = getattr(da_zos, t_agg)().item()
        return v_agg + mod_zos
    return sea_level_fun


def _get_closest_valid_cell(
    ds_var : xr.DataArray,
    lon : float,
    lat : float,
    threshold_deg : float = 10.0,
) -> Tuple[float, float]:
    """Extract the grid cell with valid entries that is closest to the given location

    To be considered, a grid cell is required to have valid entries for all time steps.

    Parameters
    ----------
    ds_var : xr.DataArray
        Gridded data with "time" dimension.
    lon, lat : float
        The longitudinal and latitudinal coordinates of the location.
    threshold_deg : float, optional
        Threshold (in degrees) for a grid cell to be considered. Default: 10

    Returns
    -------
    lon, lat : float
        Longitudinal and latitudinal coordinates of the centroid of the grid cell that is closest
        to the specified location and has valid entries.
    """
    # store original longitudinal coordinates because they are normalized in the process
    lon_orig = ds_var["lon"].values.copy()

    # for performance reasons, restrict search to cells that are close enough
    bounds = (
        lon - threshold_deg,
        lat - threshold_deg,
        lon + threshold_deg,
        lat + threshold_deg,
    )

    # in this step, longitudinal coordinates are normalized to be consistent with `bounds`
    ds_var = _select_bounds(ds_var, bounds)

    finite_mask = np.isfinite(ds_var).all(dim="time")
    if not np.any(finite_mask):
        return None
    coords = xr.broadcast(*[getattr(ds_var, d) for d in finite_mask.dims])
    finite_coords = [c.values[finite_mask] for c in coords]
    lats, lons = finite_coords if finite_mask.dims[0] == "lat" else finite_coords[::-1]
    dist_sq = (lats - lat)**2 + (lons - lon)**2
    idx = np.argmin(dist_sq)
    lon_close, lat_close = lons[idx], lats[idx]

    # determine the un-normalized longitudinal coordinate in the original dataset
    lon_diff = np.mod(lon_orig - lon_close, 360)
    lon_diff[lon_diff > 180] -= 360
    lon_close = lon_orig[np.argmin(np.abs(lon_diff))]

    return lon_close, lat_close


def _get_closest_date_in_index(
    dt_index : pd.DatetimeIndex,
    date : np.datetime64,
) -> np.datetime64:
    """Extract the entry from the given DatetimeIndex that is closest to the given date

    If the date lies exactly between two consecutive entries in the index, the earlier date is
    returned.

    Parameters
    ----------
    dt_index : pd.DatetimeIndex
        The index from which to extract the entry that is closest to `date`.
    date : np.datetime64
        The date for which to search the closest entry in `dt_index`.

    Returns
    -------
    np.datetime64
    """
    i = dt_index.searchsorted(date, side="left")
    if i == 0:
        return dt_index.values[0]
    if i == dt_index.size:
        return dt_index.values[-1]
    if date - dt_index.values[i - 1] > dt_index.values[i] - date:
        return dt_index.values[i]
    return dt_index.values[i - 1]


def area_sea_level_from_monthly_nc(
    path : Union[pathlib.Path, str],
    t_pad : Optional[np.timedelta64] = None,
    mod_zos : float = 0.0,
) -> Callable:
    """Generate a function that reads area-aggregated sea levels from a NetCDF file

    The function that is generated can be used as an input for the `sea_level` parameter in
    `TCSurgeGeoClaw.from_tc_tracks`.

    The maximum over the specified area, then the mean over all affected months is taken.
    By specifying `t_pad`, neighboring months can also be marked as affected.

    Parameters
    ----------
    path : Path or str
        Path to NetCDF file containing monthly sea level data.
    t_pad : np.timedelta64, optional
        Padding to add around the time period. Default: 7 days.
    mod_zos : float, optional
        The scalar sea level rise is added to the sea level value that is extracted from the
        specified NetCDF file. Default: 0

    Returns
    -------
    fun : function (tuple, tuple) -> float
        The first argument is a tuple of floats (lon_min, lat_min, lon_max, lat_max), the second
        argument is a pair of np.datetime64 (start, end). The function returns the mean sea level
        in the specified region and time period.
    """
    _sea_level_nc_info(path)
    def sea_level_fun(bounds, period, path=path, t_pad=t_pad, mod_zos=mod_zos):
        t_pad = np.timedelta64(7, "D") if t_pad is None else t_pad
        period = (period[0] - t_pad, period[1] + t_pad)
        times = pd.date_range(*period, freq="12H")
        months = np.unique(np.stack((times.year, times.month), axis=-1), axis=0)
        return _mean_max_sea_level_monthly(path, months, bounds) + mod_zos
    return sea_level_fun


def _mean_max_sea_level_monthly(
    path : Union[pathlib.Path, str],
    months : np.ndarray,
    bounds : Tuple[float, float, float, float],
) -> float:
    """Mean of maxima over affected area in affected months

    Parameters
    ----------
    path : Path or str
        Path to NetCDF file containing monthly sea level data.
    months : ndarray
        each row is a tuple (year, month)
    bounds : tuple
        (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    zos : float
        Sea level height in meters
    """
    pad_deg = 0.25
    max_pad_deg = 5
    with xr.open_dataset(path) as ds:
        ds = _nc_rename_vars(ds)
        mask_time = np.any([
            (ds["time"].dt.year == m[0]) & (ds["time"].dt.month == m[1])
            for m in months
        ], axis=0)
        if np.count_nonzero(mask_time) != months.shape[0]:
            raise IndexError(
                "The sea level data set doesn't contain the required months: "
                + ", ".join(f"{m[0]:04d}-{m[1]:02d}" for m in months)
            )
        ds = ds.sel(time=mask_time)

        # enlarge bounds until the mean is valid or until max_pad_deg is reached
        i_pad = 0
        mean = np.nan
        bounds_padded = bounds
        while np.isnan(mean):
            if i_pad * pad_deg > max_pad_deg:
                raise IndexError(
                    f"The sea level data set doesn't intersect the specified bounds: {bounds}"
                )
            mean = _temporal_mean_of_max_within_bounds(ds, bounds_padded)
            bounds_padded = (
                bounds_padded[0] - pad_deg, bounds_padded[1] - pad_deg,
                bounds_padded[2] + pad_deg, bounds_padded[3] + pad_deg,
            )
            i_pad += 1
    return mean


def _temporal_mean_of_max_within_bounds(
    ds : xr.Dataset,
    bounds : Tuple[float, float, float, float],
) -> float:
    """Take the maximum over a given spatial extent, then the mean over the time dimension

    Any NaN-values in the data are ignored, unless all values within the specified bounds are NaN.
    For example, in case of a gridded sea level data set with NaN over land, the maximum will just
    be over the valid values. Only if all values within the spatial bounds are NaN, the maximum is
    assumed to be NaN. Similarly, when taking the mean over the time dimension, the NaN values are
    dropped before taking the mean.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset with temporal and spatial (lon/lat) dimensions and a "zos" data variable.
    bounds : tuple of floats
        The minimum and maximum values for each spatial dimension:
        (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    float
    """
    ds_zos = _select_bounds(ds["zos"], bounds)
    if 0 in ds_zos.shape:
        return np.nan
    values = ds_zos.values[:]
    if np.all(np.isnan(values)):
        return np.nan
    return np.nanmean(np.nanmax(values, axis=(1, 2)))


def _select_bounds_dim(
    ds : Union[xr.Dataset, xr.DataArray],
    dim : str,
    bounds : Tuple[float, float],
) -> Union[xr.Dataset, xr.DataArray]:
    """Restrict the data set's dimension to the specified bounds

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        The data set for which to restrict a dimension.
    dim : str
        The dimension within `ds` to restrict.
    bounds : tuple of floats
        The minimum and maximum value for `dim`.

    Returns
    -------
    xr.Dataset or xr.DataArray
    """
    ref_min, ref_max = bounds
    idx = ((ds[dim] <= ref_max) & (ds[dim] >= ref_min)).values.nonzero()[0]
    if idx.size < 2:
        d_min, d_max = ds[dim].values.min(), ds[dim].values.max()
        if d_min > ref_min or d_max < ref_max:
            LOGGER.warn(
                f"The dimension '{dim}' ({d_min} -- {d_max}) does not cover the range of the"
                f" reference dimension ({ref_min} -- {ref_max})."
            )
    sl_start, sl_end = (idx[0], idx[-1] + 1) if idx.size > 0 else (0, 0)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds = ds.isel(indexers={dim: slice(sl_start, sl_end)})
    return ds


def _select_bounds(
    ds : Union[xr.Dataset, xr.DataArray],
    bounds : Tuple[float, float, float, float],
) -> Union[xr.Dataset, xr.DataArray]:
    """Restrict the raster data set to the specified bounds

    In a first step, the longitudinal coordinate values are normalized to the longitudinal range
    indicated by `bounds`.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        The raster data to restrict.
    bounds : tuple of float
        The minimum and maximum values for each spatial dimension:
        (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    xr.Dataset or xr.DataArray
    """
    ds = _select_bounds_dim(ds, "lat", (bounds[1], bounds[3]))
    mid_lon = 0.5 * (bounds[0] + bounds[2])
    ds = ds.assign_coords(lon=u_coord.lon_normalize(ds["lon"].values.copy(), center=mid_lon))
    ds = ds.reindex(lon=np.unique(ds["lon"].values))
    ds = _select_bounds_dim(ds, "lon", (bounds[0], bounds[2]))
    return ds


def _sea_level_nc_info(path : Union[pathlib.Path, str]) -> None:
    """Log information about the spatiotemporal bounds of the specified NetCDF file.

    Parameters
    ----------
    path : Path or str
        Path to a NetCDF file with raster data and time dimension.
    """
    LOGGER.info("Reading sea level data from %s", path)

    with xr.open_dataset(path) as ds:
        ds = _nc_rename_vars(ds)
        ds_bounds = (
            ds["lon"].values.min(), ds["lat"].values.min(),
            ds["lon"].values.max(), ds["lat"].values.max(),
        )
        ds_period = (ds["time"][0], ds["time"][-1])
        LOGGER.info("Sea level data available within bounds %s", ds_bounds)
        LOGGER.info(
            "Sea level data available within period from %04d-%02d till %04d-%02d",
            ds_period[0].dt.year, ds_period[0].dt.month,
            ds_period[1].dt.year, ds_period[1].dt.month,
        )


def _nc_rename_vars(ds : xr.Dataset) -> xr.Dataset:
    """Rename several coordinate and data variable names to their defaults

    The default names are "lon", "lat", "time", and "zos" (for sea surface height).

    Parameters
    ----------
    ds : xr.Dataset
        Data set with longitudinal, latitudinal, and temporal dimensions, as well as a sea level
        data variable.

    Returns
    -------
    xr.Dataset
    """
    var_names = {
        'lon': ('coords', ["longitude", "lon", "x"]),
        'lat': ('coords', ["latitude", "lat", "y"]),
        'time': ('coords', ["time", "date", "datetime"]),
        'zos': ('variables', ["zos", "sla", "ssh", "adt"]),
    }
    for new_name, (var_type, all_names) in var_names.items():
        old_name = [c for c in getattr(ds, var_type) if c.lower() in all_names][0]
        if old_name != new_name:
            ds = ds.rename({old_name: new_name})
    return ds
