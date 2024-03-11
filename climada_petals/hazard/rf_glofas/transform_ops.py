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

Transformations for dantro data pipeline
"""

import logging
import re
from pathlib import Path
from typing import Optional, Union, List, Mapping, Any, Iterable, Tuple, Callable
from collections import deque
from copy import deepcopy

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import gumbel_r
import xesmf as xe
from numba import guvectorize

from climada.util.coordinates import get_country_geometries, country_to_iso
from .cds_glofas_downloader import glofas_request, CDS_DOWNLOAD_DIR

LOGGER = logging.getLogger(__name__)


def sel_lon_lat_slice(target: xr.DataArray, source: xr.DataArray) -> xr.DataArray:
    """Select a lon/lat slice from 'target' using coordinates of 'source'"""
    return target.sel({c: slice(*source[c][[0, -1]]) for c in ["longitude", "latitude"]})


def rp_comp(
    sample: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
    max_rp: Optional[float] = np.inf,
):
    """Compute the return period from a right-handed Gumbel distribution

    All parameters can be arrays, in which case numpy broadcasting rules apply.

    The return period of a sample :math:`x` from an extreme value distribution is
    defined as :math:`(1 - \\mathrm{cdf}(x))^{-1}`, where :math:`\\mathrm{cdf}` is the
    cumulative distribution function of said distribution.

    Parameters
    ----------
    sample : array
        Samples for which to compute the return period
    loc : array
        Loc parameter of the Gumbel distribution
    scale : array
        Scale parameter of the distribution
    max_rp : float, optional
        The maximum value of return periods. This avoids returning infinite values.
        Defaults to ``np.inf`` (no maximum).

    Returns
    -------
    np.ndarray
        The return period(s) for the input parameters
    """
    cdf = gumbel_r.cdf(sample, loc=loc, scale=scale)
    rp_from_cdf = np.where(cdf >= 1.0, np.inf, 1.0 / np.fmax(1.0 - cdf, np.spacing(1)))
    return np.fmin(rp_from_cdf, max_rp)


def reindex(
    target: xr.DataArray,
    source: xr.DataArray,
    tolerance: Optional[float] = None,
    fill_value: float = np.nan,
    assert_no_fill_value: bool = False,
) -> xr.DataArray:
    """Reindex target to source with nearest neighbor lookup

    Parameters
    ----------
    target : xr.DataArray
        Array to be reindexed.
    source : xr.DataArray
        Array whose coordinates are used for reindexing.
    tolerance : float (optional)
        Maximum distance between coordinates. If it is superseded, the ``fill_value`` is
        inserted instead of the nearest neighbor value. Defaults to NaN
    fill_value : float (optional)
        The fill value to use if coordinates are changed by a distance of more than
        ``tolerance``.
    assert_no_fill_value : bool (optional)
        Throw an error if fill values are found in the data after reindexing. This will
        also throw an error if the fill value is present in the ``target`` before
        reindexing (because the check afterwards would else not make sense)

    Returns
    -------
    target : xr.DataArray
        Target reindexed like 'source' with nearest neighbor lookup for the data.

    Raises
    ------
    ValueError
        If tolerance is exceeded when reindexing, in case ``assert_no_fill_value`` is
        ``True``.
    ValueError
        If ``target`` already contains the ``fill_value`` before reindexing, in case
        ``assert_no_fill_value`` is ``True``.
    """

    def has_fill_value(arr):
        return arr.isin(fill_value).any() or (
            np.isnan(fill_value) and arr.isnull().any()
        )

    # Check for fill values before
    if assert_no_fill_value and has_fill_value(target):
        raise ValueError(
            f"Array '{target.name}' does already contain reindex fill value"
        )

    # Reindex operation
    target = target.reindex_like(
        source, method="nearest", tolerance=tolerance, copy=False, fill_value=fill_value
    )

    # Check for fill values after
    if assert_no_fill_value and has_fill_value(target):
        raise ValueError(
            f"Reindexing '{target.name}' to '{source.name}' exceeds tolerance! "
            "Try interpolating the datasets or increasing the tolerance"
        )

    return target


def merge_flood_maps(flood_maps: Mapping[str, xr.DataArray]) -> xr.DataArray:
    """Merge the flood maps GeoTIFFs into one NetCDF file

    Adds a "zero" flood map (all zeros)

    Parameters
    ----------
    flood_maps : dict(str, xarray.DataArray)
        The mapping of GeoTIFF file paths to respective DataArray. Each flood map is
        identified through the folder containing it. The folders are expected to follow
        the naming scheme ``floodMapGL_rpXXXy``, where ``XXX`` indicates the return
        period of the respective map.
    """
    expr = re.compile(r"floodMapGL_rp(\d+)y")
    years = [int(expr.search(name).group(1)) for name in flood_maps]
    idx = np.argsort(years)
    darrs = list(flood_maps.values())
    darrs = [
        darrs[i].drop_vars("spatial_ref", errors="ignore").squeeze("band", drop=True)
        for i in idx
    ]

    # Add zero flood map
    # NOTE: Return period of 1 is the minimal value
    da_null_flood = xr.full_like(darrs[0], np.nan)
    darrs.insert(0, da_null_flood)

    # Concatenate and rename
    years = np.insert(np.array(years)[idx], 0, 1)
    da_flood_maps = xr.concat(darrs, pd.Index(years, name="return_period"))
    da_flood_maps = da_flood_maps.rename(x="longitude", y="latitude")
    return da_flood_maps.rename("flood_depth")


def fit_gumbel_r(
    input_data: xr.DataArray,
    time_dim: str = "year",
    fit_method: str = "MM",
    min_samples: int = 2,
):
    """Fit a right-handed Gumbel distribution to the data

    Parameters
    ----------
    input_data : xr.DataArray
        The input time series to compute the distributions for. It must contain the
        dimension specified as ``time_dim``.
    time_dim : str
        The dimension indicating time. Defaults to ``year``.
    fit_method : str
        The method used for computing the distribution. Either ``MLE`` (Maximum
        Likelihood Estimation) or ``MM`` (Method of Moments).
    min_samples : int
        The number of finite samples along the time dimension required for a
        successful fit. If there are fewer samples, the fit result will be NaN.

    Returns
    -------
    xr.Dataset
        A dataset on the same grid as the input data with variables

        * ``loc``: The loc parameter of the fitted distribution (mode)
        * ``scale``: The scale parameter of the fitted distribution
        * ``samples``: The number of samples used to fit the distribution at this
          coordinate
    """

    def fit(time_series):
        # Count finite samples
        samples = np.isfinite(time_series)
        samples_count = np.count_nonzero(samples)
        if samples_count < min_samples:
            return np.nan, np.nan, 0

        # Mask array
        return (*gumbel_r.fit(time_series[samples], method=fit_method), samples_count)

    # Apply fitting
    loc, scale, samples = xr.apply_ufunc(
        fit,
        input_data,
        input_core_dims=[[time_dim]],
        output_core_dims=[[], [], []],
        exclude_dims={time_dim},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.int32],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )

    return xr.Dataset(dict(loc=loc, scale=scale, samples=samples))


def download_glofas_discharge(
    product: str,
    date_from: str,
    date_to: Optional[str],
    num_proc: int = 1,
    download_path: Union[str, Path] = CDS_DOWNLOAD_DIR,
    countries: Optional[Union[List[str], str]] = None,
    preprocess: Optional[Callable] = None,
    open_mfdataset_kw: Optional[Mapping[str, Any]] = None,
    **request_kwargs,
) -> xr.DataArray:
    """Download the GloFAS data and return the resulting dataset

    Several parameters are passed directly to
    :py:func:`climada_petals.hazard.rf_glofas.cds_glofas_downloader.glofas_request`. See
    this functions documentation for further information.

    Parameters
    ----------
    product : str
        The string identifier of the product to download. See
        :py:func:`climada_petals.hazard.rf_glofas.cds_glofas_downloader.glofas_request`
        for supported products.
    date_from : str
        Earliest date to download. Specification depends on the ``product`` chosen.
    date_to : str or None
        Latest date to download. If ``None``, only download the ``date_from``.
        Specification depends on the ``product`` chosen.
    num_proc : int
        Number of parallel processes to use for downloading. Defaults to 1.
    download_path : str or pathlib.Path
        Directory to store the downloaded data. The directory (and all required parent
        directories!) will be created if it does not yet exist. Defaults to
        ``~/climada/data/glofas-discharge/``.
    countries : str or list of str, optional
        Countries to download data for. Uses the maximum extension of all countries for
        selecting the latitude/longitude range of data to download.
    preprocess : str, optional
        String expression for preprocessing the data before merging it into one dataset.
        Must be valid Python code. The downloaded data is passed as variable ``x``.
    open_mfdataset_kw : dict, optional
        Optional keyword arguments for the ``xarray.open_mfdataset`` function.
    request_kwargs:
        Keyword arguments for the Copernicus data store request.
    """
    # Create the download path if it does not yet exist
    LOGGER.debug("Preparing download directory: %s", download_path)
    download_path = Path(download_path)  # Make sure it is a Path
    download_path.mkdir(parents=True, exist_ok=True)

    # Determine area from 'countries'
    if countries is not None:
        LOGGER.debug("Choosing lat/lon bounds from countries %s", countries)
        # Fetch area and reorder appropriately
        # NOTE: 'area': north, west, south, east
        #       'extent': lon min (west), lon max (east), lat min (south), lat max (north)
        area = request_kwargs.get("area")
        if area is not None:
            LOGGER.debug("Subsetting country geometries with 'area'")
            area = [area[1], area[3], area[2], area[0]]

        # Fetch geometries and bounds of requested countries
        iso = country_to_iso(countries)
        geo = get_country_geometries(iso, extent=area)

        # NOTE: 'bounds': minx (west), miny (south), maxx (east), maxy (north)
        # NOTE: Explicitly cast to float to ensure that YAML parser can dump the data
        bounds = deque(map(float, geo.total_bounds.flat))
        bounds.rotate(1)

        # Insert into kwargs
        request_kwargs["area"] = list(bounds)

    # Request the data
    files = glofas_request(
        product=product,
        date_from=date_from,
        date_to=date_to,
        num_proc=num_proc,
        output_dir=download_path,
        request_kw=request_kwargs,
    )

    # Set arguments for 'open_mfdataset'
    open_kwargs = dict(
        chunks={}, combine="nested", concat_dim="time", preprocess=preprocess
    )
    if open_mfdataset_kw is not None:
        open_kwargs.update(open_mfdataset_kw)

    # Squeeze all dimensions except time
    arr = xr.open_mfdataset(files, **open_kwargs)["dis24"]
    dims = {dim for dim, size in arr.sizes.items() if size == 1} - {"time"}
    return arr.squeeze(dim=dims)

def max_from_isel(
    array: xr.DataArray, dim: str, selections: List[Union[Iterable, slice]]
) -> xr.DataArray:
    """Compute the maximum over several selections of an array dimension"""
    if not all(isinstance(sel, (Iterable, slice)) for sel in selections):
        raise TypeError(
            "This function only works with iterables or slices as selection"
        )

    data = [array.isel({dim: sel}) for sel in selections]
    return xr.concat(
        [da.max(dim=dim, skipna=True) for da in data],
        dim=pd.Index(list(range(len(selections))), name="select")
        # dim=xr.concat([da[dim].max() for da in data], dim=dim)
    )


def return_period(
    discharge: xr.DataArray,
    gev_loc: xr.DataArray,
    gev_scale: xr.DataArray,
    max_return_period: float = 1e4,
) -> xr.DataArray:
    """Compute the return period for a discharge from a Gumbel EV distribution fit

    Coordinates of the three datasets must match up to a tolerance of 1e-3 degrees. If
    they do not, an error is thrown.

    Parameters
    ----------
    discharge : xr.DataArray
        The discharge values to compute the return period for
    gev_loc : xr.DataArray
        The loc parameters for the Gumbel EV distribution
    gev_scale : xr.DataArray
        The scale parameters for the Gumbel EV distribution

    Returns
    -------
    xr.DataArray
        The equivalent return periods for the input discharge and Gumbel EV istributions

    See Also
    --------
    :py:func:`climada_petals.hazard.rf_glofas.transform_ops.rp`
    :py:func:`climada_petals.hazard.rf_glofas.transform_ops.return_period_resample`
    """
    reindex_kwargs = dict(tolerance=1e-3, fill_value=-1, assert_no_fill_value=True)
    gev_loc = reindex(gev_loc, discharge, **reindex_kwargs)
    gev_scale = reindex(gev_scale, discharge, **reindex_kwargs)

    # Apply and return
    return xr.apply_ufunc(
        rp_comp,
        discharge,
        gev_loc,
        gev_scale,
        max_return_period,
        dask="parallelized",
        output_dtypes=[np.float32],
    ).rename("Return Period")


def return_period_resample(
    discharge: xr.DataArray,
    gev_loc: xr.DataArray,
    gev_scale: xr.DataArray,
    gev_samples: xr.DataArray,
    bootstrap_samples: int,
    max_return_period: float = 1e4,
    fit_method: str = "MLE",
) -> xr.DataArray:
    """Compute resampled return periods for a discharge from a Gumbel EV distribution fit

    This function uses bootstrap resampling to incorporate the uncertainty in the EV
    distribution fit. Bootstrap resampling takes the fitted distribution, draws N samples
    from it (where N is the number of samples originally used to fit the distribution),
    and fits a new distribution onto these samples. This "bootstrapped" distribution is
    then used to compute the return period. Repeating this process yields an ensemble of
    distributions that captures the uncertainty in the original distribution fit.

    Coordinates of the three datasets must match up to a tolerance of 1e-3 degrees. If
    they do not, an error is thrown.

    Parameters
    ----------
    discharge : xr.DataArray
        The discharge values to compute the return period for
    gev_loc : xr.DataArray
        The loc parameters for the Gumbel EV distribution
    gev_scale : xr.DataArray
        The scale parameters for the Gumbel EV distribution
    gev_samples : xr.DataArray
        The samples used to fit the Gumbel EV distribution at every point
    bootstrap_samples : int
        The number of bootstrap samples to compute. Increasing this will improve the
        representation of uncertainty, but strongly increase computational costs later
        on.
    fit_method : str
        Method for fitting the Gumbel EV during resampling. Available methods are the
        Method of Moments ``MM`` or the Maximum Likelihood Estimation ``MLE`` (default).

    Returns
    -------
    xr.DataArray
        The equivalent return periods for the input discharge and Gumbel EV
        distributions. The data array will have an additional dimension ``sample``,
        representing the bootstrap samples for every point.

    See Also
    --------
    :py:func:`climada_petals.hazard.rf_glofas.transform_ops.rp`
    :py:func:`climada_petals.hazard.rf_glofas.transform_ops.return_period`
    """
    reindex_kwargs = dict(tolerance=1e-3, fill_value=-1, assert_no_fill_value=True)
    gev_loc = reindex(gev_loc, discharge, **reindex_kwargs)
    gev_scale = reindex(gev_scale, discharge, **reindex_kwargs)
    gev_samples = reindex(gev_samples, discharge, **reindex_kwargs).astype("int32")

    # All but 'longitude' and 'latitude' are core dimensions for this operation
    core_dims = list(discharge.dims)
    core_dims.remove("longitude")
    core_dims.remove("latitude")

    # Define input array layout
    # NOTE: This depends on the actual core dimensions put in, so we have to do this
    #       programmatically.
    # num_core_dims = len(core_dims)
    # arr_str_in = "[" + ", ".join([":" for _ in range(num_core_dims)]) + "]"
    # dims_str_in = "(" + ", ".join([f"c_{i}" for i in range(num_core_dims)]) + ")"
    # arr_str_out = arr_str_in[:-1] + ", :]"
    # dims_str_out = dims_str_in[:-1] + ", samples)"
    # print(arr_str_in, dims_str_in)
    # print(arr_str_out, dims_str_out)

    # Dummy array
    # dummy = xr.DataArray(
    #     np.empty((bootstrap_samples)),
    #     coords=dict(samples=list(range(bootstrap_samples))),
    # )

    # Define the vectorized function
    # @guvectorize(
    #     (
    #             f"(float32{arr_str_in}, float64, float64, int32,"
    #             f"float64,float64[:], float32{arr_str_out})"
    #     ),
    #     f"{dims_str_in}, (), (), (), (), (samples) -> {dims_str_out}",
    #     # nopython=True,
    # )
    def rp_sampling(
        dis: np.ndarray,
        loc: float,
        scale: float,
        samples: int,
        max_rp: float,
    ):
        """Compute multiple return periods using bootstrap sampling

        This function does not support broadcasting on the ``loc`` and ``scale``
        parameters.
        """
        # Return NaNs if we have no reliable samples
        finite_input = all((np.isfinite(x) for x in (loc, scale, samples)))
        if samples < 1 or not finite_input:
            return np.full((bootstrap_samples,) + dis.shape, np.nan)

        # Resample by drawing samples and re-fitting
        def resample_params():
            return gumbel_r.fit(
                gumbel_r.rvs(loc=loc, scale=scale, size=samples),
                method=fit_method,
            )

        # Resample the distribution and compute return periods from these resamples
        return np.array(
            [
                rp_comp(dis, *resample_params(), max_rp)
                for _ in range(bootstrap_samples)
            ],
            dtype=np.float32,
        )

    # Apply and return
    # NOTE: 'rp_sampling' requires scalar 'loc' and 'scale' parameters, so we
    #       define all but 'longitude' and 'latitude' dimensions as core dimensions
    # core_dims = set(discharge.dims) - {"longitude", "latitude"}
    return (
        xr.apply_ufunc(
            rp_sampling,
            discharge,
            gev_loc,
            gev_scale,
            gev_samples,
            max_return_period,
            input_core_dims=[list(core_dims), [], [], [], []],
            output_core_dims=[["sample"] + list(core_dims)],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32],
            dask_gufunc_kwargs=dict(
                output_sizes=dict(sample=bootstrap_samples), allow_rechunk=True
            ),
        )
        .rename("Return Period")
        .assign_coords(sample=np.arange(bootstrap_samples))
        .transpose(..., "sample")
    )


def interpolate_space(
    return_period: xr.DataArray,
    flood_maps: xr.DataArray,
    method: str = "linear",
) -> xr.DataArray:
    """Interpolate the return period in space onto the flood maps grid"""
    # Select lon/lat for flood maps
    flood_maps = sel_lon_lat_slice(flood_maps, return_period)

    # Interpolate the return period
    return return_period.interp(
        coords=dict(longitude=flood_maps["longitude"], latitude=flood_maps["latitude"]),
        method=method,
        kwargs=dict(fill_value=None),  # Extrapolate
    )


def regrid(
    return_period: xr.DataArray,
    flood_maps: xr.DataArray,
    method: str = "bilinear",
    regridder: Optional[xe.Regridder] = None,
    return_regridder: bool = False,
) -> Union[xr.DataArray, Tuple[xr.DataArray, xe.Regridder]]:
    """Regrid the return period onto the flood maps grid"""
    # Select lon/lat for flood maps
    flood_maps = sel_lon_lat_slice(flood_maps, return_period)

    # Mask return period so NaNs are not propagated
    rp = return_period.to_dataset(name="data")
    dims_to_remove = set(rp.sizes.keys()) - {"longitude", "latitude"}
    dims_to_remove = {dim: 0 for dim in dims_to_remove}
    rp["mask"] = xr.where(rp["data"].isel(dims_to_remove).isnull(), 0, 1)

    # NOTE: Masking here would omit all return periods outside flood plains
    #       (This might be desirable at some point?)
    flood = flood_maps.to_dataset(name="data")
    # flood["mask"] = xr.where(flood["data"].isel(return_period=-1).isnull(), 0, 1)

    # Perform regridding
    if regridder is None:
        regridder = xe.Regridder(
            rp,
            flood,
            method=method,
            extrap_method="nearest_s2d",
            # unmapped_to_nan=False,
        )

    return_period_regridded = regridder(return_period).rename(return_period.name)
    if return_regridder:
        return return_period_regridded, regridder

    return return_period_regridded


def apply_flopros(
    flopros_data: gpd.GeoDataFrame,
    return_period: Union[xr.DataArray, xr.Dataset],
    layer: str = "MerL_Riv",
) -> Union[xr.DataArray, xr.Dataset]:
    """Restrict the given return periods using FLOPROS data

    The FLOPROS database describes the regional protection to river flooding based on a
    return period. Users can choose from different database layers. For each coordinate
    in ``return_period``, the value from the respective FLOPROS database layer is
    selected. Any ``return_period`` lower than or equal to the FLOPROS protection value
    is discarded and set to ``NaN``.

    Parameters
    ----------
    flopros_data : PassthroughContainer
        The FLOPROS data bundled into a dantro.containers.PassthroughContainer
    return_period : xr.DataArray or xr.Dataset
        The return periods to be restricted by the FLOPROS data
    layer : str
        The FLOPROS database layer to evaluate

    Returns
    -------
    xr.DataArray or xr.Dataset
        The ``return_period`` data with all values below the local FLOPROS protection
        threshold set to ``NaN``.
    """
    # Make GeoDataframe from existing geometry
    latitude = return_period["latitude"].values
    longitude = return_period["longitude"].values
    lon, lat = np.meshgrid(longitude, latitude, indexing="ij")
    df_geometry = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon.flat, lat.flat), crs="EPSG:4326"
    )

    # Merge the DataFrames, setting the FLOPROS value for each point
    df_merged = df_geometry.sjoin(flopros_data, how="left", predicate="within")

    # Available layers
    # layers = [
    #     "DL_Min_Co",
    #     "DL_Max_Co",
    #     "PL_Min_Co",
    #     "PL_Max_Co",
    #     "MerL_Riv",
    #     "DL_Min_Riv",
    #     "DL_Max_Riv",
    #     "PL_Min_Riv",
    #     "PL_Max_Riv",
    #     "ModL_Riv",
    # ]

    def data_array_from_layer(col_name):
        """Create a xr.DataArray from a GeoDataFrame column"""
        return xr.DataArray(
            data=df_merged[col_name]
            .to_numpy(dtype=np.float32)
            .reshape((longitude.size, latitude.size)),
            dims=["longitude", "latitude"],
            coords=dict(longitude=longitude, latitude=latitude),
        )

    # Apply the FLOPROS protection
    flopros = data_array_from_layer(layer)
    return return_period.where(return_period > flopros)


def flood_depth(
    return_period: Union[xr.Dataset, xr.DataArray], flood_maps: xr.DataArray
) -> Union[xr.Dataset, xr.DataArray]:
    """Compute the flood depth from a return period and flood maps.

    At each lat/lon coordinate, take the return period(s) and use them to interpolate
    the flood maps at this position in the return period dimension. Take the interpolated
    values and return them as flood hazard footprints. Works with arbitrarily high
    dimensions in the ``return_period`` array. Interpolation is linear.

    Parameters
    ----------
    return_period : xr.DataArray or xr.Dataset
        The return periods for which to compute the flood depth. If this is a dataset,
        the function will compute the flood depth for each variable.
    flood_maps : xr.DataArray
        Maps that indicate flood depth at latitude/longitude/return period coordinates.

    Returns
    -------
    xr.DataArray or xr.Dataset
        The flood depths for the given return periods.
    """
    # Select lon/lat for flood maps
    flood_maps = sel_lon_lat_slice(flood_maps, return_period)

    # Clip infinite return periods
    return_period = return_period.clip(
        min=1, max=flood_maps["return_period"].max(), keep_attrs=True
    )

    # All but 'longitude' and 'latitude' are core dimensions for this operation
    core_dims = list(return_period.dims)
    core_dims.remove("longitude")
    core_dims.remove("latitude")

    # Define input array layout
    # NOTE: This depends on the actual core dimensions put in, so we have to do this
    #       programmatically.
    num_core_dims = len(core_dims)
    arr_str = "[" + ", ".join([":" for _ in range(num_core_dims)]) + "]"
    core_dims_str = "(" + ", ".join([f"c_{i}" for i in range(num_core_dims)]) + ")"

    # Define the vectorized function
    @guvectorize(
        f"(float32{arr_str}, float64[:], int64[:], float32{arr_str})",
        f"{core_dims_str}, (j), (j) -> {core_dims_str}",
        nopython=True,
    )
    def interpolate(return_period, hazard, return_periods, out_depth):
        """Linearly interpolate the hazard to a given return period

        Args:
            return_period (float): The return period to evaluate the hazard at
            hazard (np.array): The hazard at given return periods (dependent var)
            return_periods (np.array): The given return periods (independent var)

        Returns:
            float: The hazard at the requested return period.

            The hazard cannot become negative. Values beyond the given return periods
            range are extrapolated.
        """
        # Shortcut for only NaNs
        # NOTE: After rebuilding the hazard maps, the "1:" can be removed
        if np.all(np.isnan(hazard[1:])):
            out_depth[:] = np.full_like(return_period, np.nan)
            return

        # Make NaNs to zeros
        # NOTE: NaNs should be grouped at lower end of 'return_periods', so this should
        #       be sane.
        # hazard = np.nan_to_num(hazard)
        hazard = np.where(np.isnan(hazard), 0.0, hazard)

        # Use extrapolation and have 0.0 as minimum value
        out_depth[:] = np.interp(return_period, return_periods, hazard)
        out_depth[:] = np.maximum(out_depth, 0.0)

    # Perform operation
    out = xr.apply_ufunc(
        interpolate,
        return_period,
        flood_maps,
        flood_maps["return_period"],
        input_core_dims=[core_dims, ["return_period"], ["return_period"]],
        output_core_dims=[core_dims],
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )
    if isinstance(out, xr.DataArray):
        out = out.rename("Flood Depth")
    return out


def save_file(
    data: Union[xr.Dataset, xr.DataArray],
    output_path: Union[Path, str],
    encoding: Optional[Mapping[str, Any]] = None,
    engine: Optional[str] = "netcdf4",
    **encoding_defaults,
):
    """Save xarray data as a file with default compression

    Calls ``data.to_netcdf``.
    See https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html
    for the documentation of the underlying method.

    Parameters
    ----------
    data : xr.Dataset or xr.Dataarray
        The data to be stored in the file
    output_path : pathlib.Path or str
        The file path to store the data into. If it does not contain a suffix, ``.nc``
        is automatically appended. The enclosing folder must already exist.
    encoding : dict (optional)
        Encoding settings for every data variable. Will update the default settings.
    engine : str (optional)
        The engine used for writing the file. Defaults to ``"netcdf4"``.
    encoding_defaults
        Encoding settings shared by all data variables. This will update the default
        encoding settings, which are ``dict(dtype="float32", zlib=False, complevel=4)``.
    """
    # Promote to Dataset for accessing the data_vars
    if isinstance(data, xr.DataArray):
        data = data.to_dataset()

    # Store encoding
    default_encoding = dict(dtype="float32", zlib=False, complevel=4)
    default_encoding.update(**encoding_defaults)
    enc = {var: deepcopy(default_encoding) for var in data.data_vars}
    if encoding is not None:
        for key, settings in encoding.items():
            enc[key].update(settings)

    # Sanitize output path and write file
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".nc")
    data.to_netcdf(output_path, encoding=enc, engine=engine)
