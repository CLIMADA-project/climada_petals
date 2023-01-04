import logging
import re
from pathlib import Path
from typing import Optional, Union, List, Mapping, Any
from collections import deque
from collections.abc import Iterable

import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import gumbel_r
from scipy.interpolate import interp1d

from dantro.data_ops import is_operation
from dantro.groups import OrderedDataGroup

from climada.util.constants import SYSTEM_DIR
from climada.util.coordinates import get_country_geometries, country_to_iso
from climada_petals.util import glofas_request

LOGGER = logging.getLogger(__name__)


def sel_lon_lat_slice(target: xr.DataArray, source: xr.DataArray) -> xr.DataArray:
    """Select a lon/lat slice from 'target' using coordinates of 'source'"""
    lon = source["longitude"][[0, -1]]
    lat = source["latitude"][[0, -1]]
    return target.sel(longitude=slice(*lon), latitude=slice(*lat))


def reindex(
    target: xr.DataArray,
    source: xr.DataArray,
    tolerance=None,
    fill_value=np.nan,
    assert_no_fill_value=False,
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


@is_operation
def merge_flood_maps(flood_maps: OrderedDataGroup) -> xr.Dataset:
    """Merge the flood maps GeoTIFFs into one NetCDF file

    Adds a "zero" flood map (all zeros)

    Parameters
    ----------
    flood_maps : dantro.OrderedDataGroup
        The flood maps stored in a data group. Each flood map is expected to be an
        xarray Dataset named ``floodMapGL_rpXXXy``, where ``XXX`` indicates the return
        period of the respective map.

    """
    # print(flood_maps)
    expr = re.compile(r"floodMapGL_rp(\d+)y")
    years = [int(expr.match(name).group(1)) for name in flood_maps]
    idx = np.argsort(years)
    dsets = list(flood_maps.values())
    dsets = [dsets[i].drop_vars("spatial_ref").squeeze("band", drop=True) for i in idx]

    # Add zero flood map
    # NOTE: Return period of 1 is the minimal value
    ds_null_flood = xr.zeros_like(dsets[0])
    dsets.insert(0, ds_null_flood)

    # Concatenate and rename
    years = np.insert(np.array(years)[idx], 0, 1)
    ds_flood_maps = xr.concat(dsets, pd.Index(years, name="return_period"))
    ds_flood_maps = ds_flood_maps.rename(
        band_data="flood_depth", x="longitude", y="latitude"
    )
    return ds_flood_maps


@is_operation
def fit_gumbel_r(
    input_data: xr.DataArray, fit_method: str = "MLE", min_samples: int = 2
):
    """Fit a right-handed Gumbel distribution to the data

    input_data : xr.DataArray
        The input time series to compute the distributions for. It must contain the
        dimension ``year``.
    fit_method : str
        The method used for computing the distribution. Either ``MLE`` (Maximum
        Likelihood Estimation) or ``MM`` (Method of Moments).
    min_samples : int
        The number of finite samples along the ``year`` dimension required for a
        successful fit. If there are fewer samples, the fit result will be NaN.
    """

    def fit(time_series):
        # Count finite samples
        samples = np.isfinite(time_series)
        if np.count_nonzero(samples) < min_samples:
            return np.nan, np.nan

        # Mask array
        return gumbel_r.fit(time_series[samples], method=fit_method)

    # Apply fitting
    loc, scale = xr.apply_ufunc(
        fit,
        input_data,
        input_core_dims=[["year"]],
        output_core_dims=[[], []],
        exclude_dims={"year"},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64],
    )

    return xr.Dataset(dict(loc=loc, scale=scale))


@is_operation
def download_glofas_discharge(
    product: str,
    date_from: str,
    date_to: Optional[str],
    num_proc: int = 1,
    download_path: Union[str, Path] = Path(SYSTEM_DIR, "glofas-discharge"),
    countries: Optional[Union[List[str], str]] = None,
    preprocess: Optional[str] = None,
    open_mfdataset_kw: Optional[Mapping[str, Any]] = None,
    **request_kwargs,
) -> xr.DataArray:
    """Download the GloFAS data and return the resulting dataset

    Several parameters are passed directly to
    :py:func:`climada_petals.util.glofas_request`. See this functions documentation for
    further information.

    Parameters
    ----------
    product : str
        The string identifier of the product to download. See
        :py:func:`climada_petals.util.glofas_request` for supported products.
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
    open_kwargs = dict(chunks={}, combine="nested", concat_dim="time")
    if open_mfdataset_kw is not None:
        open_kwargs.update(open_mfdataset_kw)

    # Preprocessing
    if preprocess is not None:
        open_kwargs.update(preprocess=lambda x: eval(preprocess))

    # Open the data and return it
    return xr.open_mfdataset(files, **open_kwargs)["dis24"]


@is_operation
def max_from_isel(
    array: xr.DataArray, dim: str, selections: List[Union[Iterable, slice]]
):
    """Compute the maximum over several selections of an array dimension"""
    if not all(
        [isinstance(sel, Iterable) or isinstance(sel, slice) for sel in selections]
    ):
        raise TypeError(
            "This function only works with iterables or slices as selection"
        )

    data = [array.isel({dim: sel}) for sel in selections]
    return xr.concat(
        [da.max(dim=dim, skipna=True) for da in data],
        dim=pd.Index(list(range(len(selections))), name="select")
        # dim=xr.concat([da[dim].max() for da in data], dim=dim)
    )


@is_operation
def return_period(
    discharge: xr.DataArray, gev_loc: xr.DataArray, gev_scale: xr.DataArray
) -> xr.DataArray:
    """Compute the return period for a discharge from a Gumbel EV distribution fit

    Coordinates of the three datasets must match up to a tolerance of 1e-3 degrees. If
    they do not, an error is thrown.
    """
    gev_loc = reindex(
        gev_loc, discharge, tolerance=1e-3, fill_value=-1, assert_no_fill_value=True
    )
    gev_scale = reindex(
        gev_scale, discharge, tolerance=1e-3, fill_value=-1, assert_no_fill_value=True
    )

    # Compute the return period
    def rp(dis, loc, scale):
        return 1.0 / (1.0 - gumbel_r.cdf(dis, loc=loc, scale=scale))

    # Apply and return
    return xr.apply_ufunc(
        rp,
        discharge,
        gev_loc,
        gev_scale,
        dask="parallelized",
        output_dtypes=[np.float32],
    ).rename("Return Period")


@is_operation
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


@is_operation
def flood_depth(return_period: xr.DataArray, flood_maps: xr.DataArray) -> xr.Dataset:
    def interpolate(return_period, hazard, return_periods):
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
        if np.all(np.isnan(hazard)):
            return np.full_like(return_period, np.nan)

        # Make NaNs to zeros
        # NOTE: NaNs should be grouped at lower end of 'return_periods', so this should
        #       be sane.
        hazard = np.nan_to_num(hazard)

        # Use extrapolation and have 0.0 as minimum value
        ret = interp1d(
            return_periods,
            hazard,
            fill_value="extrapolate",
            assume_sorted=True,
            copy=False,
        )(return_period)
        ret = np.maximum(ret, 0.0)
        return ret

    # Select lon/lat for flood maps
    flood_maps = sel_lon_lat_slice(flood_maps, return_period)

    # All but 'longitude' and 'latitude' are core dimensions for this operation
    dims = set(return_period.dims)
    core_dims = dims - {"longitude", "latitude"}

    # Perform operation
    return xr.apply_ufunc(
        interpolate,
        return_period,
        flood_maps,
        flood_maps["return_period"],
        input_core_dims=[list(core_dims), ["return_period"], ["return_period"]],
        output_core_dims=[list(core_dims)],
        exclude_dims={"return_period"},  # Add 'step' and 'number' here?
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float32],
    ).rename("Flood Depth").to_dataset()


def save_file(
    data: Union[xr.Dataset, xr.DataArray],
    output_path: Union[Path, str],
    **encoding_kwargs,
):
    """Save xarray data as a file with default compression

    Parameters
    ----------
    data : xr.Dataset or xr.Dataarray
        The data to be stored in the file
    output_path : pathlib.Path or str
        The file path to store the data into. If it does not contain a suffix, ``.nc``
        is automatically appended. The enclosing folder must already exist.
    encoding_kwargs
        Optional keyword arguments for the encoding, which applies to every data
        variable. Default encoding settings are:
        ``dict(dtype="float32", zlib=True, complevel=4)``
    """
    # Promote to Dataset for accessing the data_vars
    if isinstance(data, xr.DataArray):
        data = data.to_dataset()

    # Store encoding
    encoding = dict(dtype="float32", zlib=True, complevel=4)
    encoding.update(encoding_kwargs)
    encoding = {var: encoding for var in data.data_vars}

    # Repeat encoding for each variable
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".nc")
    data.to_netcdf(output_path, encoding=encoding)


def finalize(
    *,
    data: Mapping[str, Any],
    to_file: Optional[Iterable[Union[str, Mapping[str, Any]]]] = None,
    to_dm: Optional[Iterable[Union[str, Mapping[str, Any]]]] = None,
    **kwargs,
):
    """Store tagged nodes in files or in the DataManager depending on the user input

    Parameters
    ----------
    data : dict
        The mapping of tagged data containers computed by the dantro transform DAG.
    to_file : list of str or dict (optional)
        Specs for writing data into files. If an entry is a single string, it is
        interpreted as the data tag to write and the filename to write to. If it is a
        ``dict``, the following items are interpreted:

        * ``tag``: The data tag to write.
        * ``filename`` (optional): The filename to write to. Defaults to the value of
          ``tag``.
        * ``encoding`` (dict, optional): The encoding when writing the file.

        This will call :py:func:`climada_petals.hazard.rf_glofas.transform_ops.save_file`,
        for each entry in ``to_file``.
    to_dm : list of str or dict (optional)
        Specs for storing data in the ``DataManager``. This requires the manager to be
        passed as node called ``data_manager`` in the transform DAG:

        .. code-block:: yaml

            transform:
              - pass: !dag_tag dm
              tag: data_manager

        If an entry in ``to_dm`` is a single string, it is interpreted as the data tag to
        store and the name of the data entry in the ``DataManager``. If it is a ``dict``,
        the following items are interpreted:

        * ``tag``: The data tag to store.
        * ``name`` (optional): The name of the target entry in the data manager.
          Defaults to the value of ``tag``.

    Examples
    --------

    Add the ``to_file`` and ``to_dm`` nodes to your evaluation config on the same level
    as ``transform``:

    .. code-block:: yaml

        eval:
          with_cache:
            to_file:
              - some_tag
              - tag: some_other_tag
                filename: other_tag_output
            to_dm:
              - some_tag
              - tag: some_other_tag
                name: other_tag_container

            transform:
              # ...
    """
    # Write data to files
    output_dir = Path(kwargs["out_path"]).parent
    for entry in to_file if to_file is not None else {}:
        if isinstance(entry, dict):
            tag = entry["tag"]
            filename = entry.get("filename", tag)
            encoding = entry.get("encoding", {})
        else:
            tag = entry
            filename = entry
            encoding = {}
        save_file(data[tag], output_dir / filename, **encoding)

    # Store data in DataManager
    for entry in to_dm if to_dm is not None else {}:
        if isinstance(entry, dict):
            tag = entry["tag"]
            name = entry.get("name", tag)
        else:
            tag = entry
            name = entry
        data["data_manager"].new_container(name, data=data[tag])
