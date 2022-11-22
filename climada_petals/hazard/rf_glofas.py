import sys
import logging
from pathlib import Path
from copy import deepcopy
from typing import Optional, Union, List
from collections import deque

import numpy as np
import xarray as xr
from scipy.stats import gumbel_r
from scipy.interpolate import interp1d
import pandas as pd

import dantro as dtr
from dantro.data_ops import is_operation
from dantro.data_loaders import AllAvailableLoadersMixin
from dantro.containers import XrDataContainer
from dantro.tools import load_yml
from dantro.plot import is_plot_func

from climada.hazard import Hazard
from climada.util.constants import SYSTEM_DIR
from climada.util.coordinates import get_country_geometries, country_to_iso
from climada_petals.hazard.river_flood import RiverFlood
from climada_petals.util import glofas_request

LOGGER = logging.getLogger(__name__)


def sel_lon_lat_slice(target: xr.DataArray, source: xr.DataArray) -> xr.DataArray:
    """Select a lon/lat slice from 'target' using coordinates of 'source'

    Warning
    -------
    This assumes the DataArrays are based on GloFAS data, where latitude runs from
    north to south (decreasing!)
    """
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
def download_glofas_discharge(
    product: str,
    date_from: str,
    date_to: Optional[str],
    num_proc: int,
    download_path: Union[str, Path] = Path(SYSTEM_DIR, "glofas-discharge"),
    countries: Optional[Union[List[str], str]] = None,
    **request_kwargs,
) -> xr.DataArray:
    """Download the GloFAS data and return the resulting dataset"""
    # Create the download path if it does not yet exist
    LOGGER.debug("Preparing download directory: %s", download_path)
    if isinstance(download_path, str):
        download_path = Path(download_path)
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
        bounds = deque(geo.total_bounds)
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

    # Open the data and return it
    return xr.open_mfdataset(files, chunks={}, combine="nested", concat_dim="time")[
        "dis24"
    ]


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
def flood_depth(return_period: xr.DataArray, flood_maps: xr.DataArray) -> xr.DataArray:
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
    ).rename("Flood Depth")


class ClimadaDataManager(AllAvailableLoadersMixin, dtr.DataManager):
    """A DataManager that can load many different file formats"""

    _HDF5_DSET_DEFAULT_CLS = XrDataContainer
    """Tells the HDF5 loader which container class to use"""

    _NEW_CONTAINER_CLS = XrDataContainer
    """Which container class to use when adding new containers"""


@is_plot_func(use_dag=True, required_dag_tags=("return_period"))
def write_results(*, data: dict, out_path: str, **plot_kwargs):
    data["return_period"].to_netcdf(out_path)


def show_results(*args, data, **kwargs):
    """Prints the arguments that are passed to it, with `data` being the
    results from the data transformation framework"""
    print(f"args:   {args}")
    print(f"kwargs: {kwargs}")
    print("")

    print(f"Data Transformation Results\n{'-'*27}")
    print("\n".join(f"-- {k}:\n{v}\n" for k, v in data.items()))
    print("")

    out_path = Path(kwargs["out_path"]).parent / "output.nc"
    print(f"Writing output data to {out_path}")
    data["flood_depth"].to_netcdf(out_path)


def return_flood_depth(*args, data, **kwargs):
    return data["flood_depth"]


def store_flood_depth_in_dm(*args, data, **kwargs):
    # flood_depth_cont = XrDataContainer(name="flood_depth", data=data["flood_depth"])
    data["data_manager"].new_container("flood_depth", data=data["flood_depth"])


class GloFASRiverFlood:
    def __init__(self, yaml_cfg_path):
        # Load config
        self.cfg = load_yml(yaml_cfg_path)

    def compute_hazard(self, **cfg_kwargs):
        # Update the config
        cfg = deepcopy(self.cfg)
        cfg.update(cfg_kwargs)

        # Create data directory
        data_dir = Path(self.cfg["data_dir"]).expanduser().absolute()
        data_dir.mkdir(parents=True, exist_ok=True)

        # Set up DataManager
        dm = ClimadaDataManager(data_dir, **cfg.get("data_manager", {}))

        # NOTE Can let the DataManager load something here, if desired ...
        # dm.load(...)
        # dm.load_from_cfg(...)

        dm.load_from_cfg(load_cfg=cfg["data_manager"]["load_cfg"], print_tree=True)

        # Set up the PlotManager ...
        pm = dtr.PlotManager(dm=dm, **cfg.get("plot_manager"))

        # ... and use it to invoke some evaluation routine
        pm.plot_from_cfg(plots_cfg=cfg.get("eval"))

        # Return xarray for `from_raster_xarray`??
        print(dm.tree)
        # print(dm["flood_depth"].data)
        # print(dm["area"])
        return dm["flood_depth"].data.to_dataset()

    def get_forecast(self, hazard_concat_dim="number", **cfg_kwargs) -> pd.Series:
        # Run the hazard computation pipeline
        ds_hazard = self.compute_hazard(**cfg_kwargs)

        # Squeeze: Drop all coordinates that are not dimensions
        ds_hazard = ds_hazard.squeeze()

        def create_hazard(ds: xr.Dataset) -> Hazard:
            """Create hazard from a GloFASRiverFlood hazard dataset"""
            return RiverFlood.from_raster_xarray(
                ds,
                hazard_type="RF",
                intensity="Flood Depth",
                intensity_unit="m",
                coordinate_vars=dict(event=hazard_concat_dim),
                data_vars=dict(date="time"),
            )

        # Iterate over all dimensions that are not lon, lat, or number
        # NOTE: Why would we have others than "time"? Multiple instances of 'max' over
        #       'step'? How would this look like in the DAG? Check this first!
        iter_dims = list(
            set(ds_hazard.dims) - {"longitude", "latitude", hazard_concat_dim}
        )
        if iter_dims:
            index = pd.MultiIndex.from_product(
                [ds_hazard[dim].values for dim in iter_dims], names=iter_dims
            )
            hazards = [
                create_hazard(ds_hazard.sel(dict(zip(iter_dims, idx))))
                for idx in index.to_flat_index()
            ]
        else:
            index = None
            hazards = [create_hazard(ds_hazard)]

        return pd.Series(hazards, index=index)


def run(_, cfg_file_path: str):
    grf = GloFASRiverFlood(cfg_file_path)
    grf.compute_hazard()


if __name__ == "__main__":
    run(*sys.argv)
