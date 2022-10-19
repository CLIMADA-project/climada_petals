from pathlib import Path
from climada.util.config import SYSTEM_DIR
from copy import deepcopy

import numpy as np
import xarray as xr
from scipy.stats import gumbel_r

import dantro as dtr
from dantro.data_ops import is_operation
from dantro.data_loaders import AllAvailableLoadersMixin
from dantro.containers import XrDataContainer
from dantro.tools import load_yml


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


class ClimadaDataManager(AllAvailableLoadersMixin, dtr.DataManager):
    """A DataManager that can load many different file formats"""

    _HDF5_DSET_DEFAULT_CLS = XrDataContainer
    """Tells the HDF5 loader which container class to use"""


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
        data_dir.mkdir(parents=True)

        # Set up DataManager
        dm = ClimadaDataManager(data_dir, **cfg.get("data_manager", {}))

        # NOTE Can let the DataManager load something here, if desired ...
        # dm.load(...)
        # dm.load_from_cfg(...)

        # Set up the PlotManager ...
        pm = dtr.PlotManager(dm=dm, **cfg.get("plot_manager"))

        # ... and use it to invoke some evaluation routine
        pm.plot_from_cfg(plots_cfg=cfg.get("eval"), plot_only=cfg.get("eval_only"))

        # Return xarray for `from_raster_xarray`??
