import sys
import logging
from pathlib import Path
from climada.util.constants import SYSTEM_DIR
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import xarray as xr
from scipy.stats import gumbel_r
from scipy.interpolate import interp1d

import dantro as dtr
from dantro.data_ops import is_operation
from dantro.data_loaders import AllAvailableLoadersMixin
from dantro.containers import XrDataContainer
from dantro.tools import load_yml
from dantro.plot import is_plot_func

from climada_petals.util import glofas_request

LOGGER = logging.getLogger(__name__)


@is_operation
def download_glofas_discharge(
    product: str,
    date_from: str,
    date_to: Optional[str],
    num_proc: int,
    download_path: Union[str, Path] = Path(SYSTEM_DIR, "glofas-discharge"),
    **request_kwargs,
) -> xr.DataArray:
    """Download the GloFAS data and return the resulting dataset"""
    # Create the download path if it does not yet exist
    LOGGER.debug("Preparing download directory: %s", download_path)
    download_path.mkdir(parents=True, exist_ok=True)

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
    """Compute the return period for a discharge from a Gumbel EV distribution fit"""
    # Make sure both objects are aligned (there might be slight coordinate differences)
    discharge = discharge.reindex_like(gev_loc, method="nearest", tolerance=1e-6)

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
        output_dtypes=[discharge.dtype],
    ).rename("Return Period")


@is_operation
def interpolate_space(
    return_period: xr.DataArray,
    flood_maps: xr.DataArray,
    method: str = "linear",
) -> xr.DataArray:
    """Interpolate the return period in space onto the flood maps grid"""
    return return_period.interp(
        coords=dict(longitude=flood_maps["longitude"], latitude=flood_maps["latitude"]),
        method=method,
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


def run(_, cfg_file_path: str):
    grf = GloFASRiverFlood(cfg_file_path)
    grf.compute_hazard()


if __name__ == "__main__":
    run(*sys.argv)
