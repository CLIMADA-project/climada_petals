import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Union

import xarray as xr
import pandas as pd
from dask.distributed import Client

import dantro as dtr
from dantro.data_loaders import AllAvailableLoadersMixin
from dantro.containers import XrDataContainer
from dantro.tools import load_yml

from climada.util.constants import SYSTEM_DIR
from climada_petals.hazard.river_flood import RiverFlood


LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_DIR = SYSTEM_DIR / "glofas-computation"
DEFAULT_SETUP_CFG = Path(__file__).parent.absolute() / "rf_glofas_util.yml"
DEFAULT_GLOFAS_CFG = Path(__file__).parent.absolute() / "rf_glofas.yml"


class ClimadaDataManager(AllAvailableLoadersMixin, dtr.DataManager):
    """A DataManager that can load many different file formats"""

    _HDF5_DSET_DEFAULT_CLS = XrDataContainer
    """Tells the HDF5 loader which container class to use"""

    _NEW_CONTAINER_CLS = XrDataContainer
    """Which container class to use when adding new containers"""


@contextmanager
def dask_client_or_not(*client_args, **client_kwargs):
    """Create a context with a dak.distributed.Client but only if arguments are present"""
    # Yield nothing if there are no arguments (sequential processing)
    if not client_args and not client_kwargs:
        yield None

    # Yield the client if there are arguments, and close it afterwards
    LOGGER.info("Creating dask.distributed.Client")
    client = Client(*client_args, **client_kwargs)
    try:
        yield client
    finally:
        LOGGER.info("Closing dask.distributed.Client")
        client.close()


def dantro_transform(yaml_cfg_path: Union[Path, str]):
    """Perform a transformation with the dantro Transformation DAG

    Parameters
    ----------
    yaml_cfg_path : Path or str
        The path to the configuration file specifying the transformation(s).
    dask_client_kwargs
        Keyword arguments for a ``dask.distributed.Client``. Will create one and hence
        compute in parallel only if these arguments are present. If you created a client
        outside this function, *do not* pass any arguments here.
    """
    # Load the config
    cfg = load_yml(yaml_cfg_path)

    # Check data directory
    data_dir = Path(cfg.get("data_dir", DEFAULT_DATA_DIR)).expanduser().absolute()
    if not data_dir.exists():
        # Default dir can be created
        if data_dir == DEFAULT_DATA_DIR:
            data_dir.mkdir(parents=False)
        # Custom dir should exist
        else:
            raise RuntimeError(
                f"Input data required, but data_dir does not exist: {data_dir}"
            )

    # Set up DataManager
    dm = ClimadaDataManager(data_dir, **cfg.get("data_manager", {}))
    dm.load_from_cfg(load_cfg=cfg["data_manager"]["load_cfg"], print_tree=True)

    # Set up the PlotManager ...
    pm = dtr.PlotManager(dm=dm, **cfg.get("plot_manager"))

    # ... and use it to invoke some evaluation routine
    pm.plot_from_cfg(plots_cfg=cfg.get("eval"))

    # Return the DataManager
    print(dm.tree)
    return dm


def setup(cfg: Union[str, Path] = DEFAULT_SETUP_CFG, **dask_client_kwargs):
    """Set up the data required for computing flood footprints from GloFAS discharge data

    This function will download historical GloFAS data from the Copernicus Data Store
    and compute a right-handed Gumbel distribution at every pixel from the yearly maximum
    of the time series.

    Additionally, it will load flood hazard maps and merge them into a single NetCDF file.

    Prerequisites
    -------------
    * Make sure you can download data from the Copernicus Data Store API following the
      instructions in :py:func:`climada_petals.util.glofas_request`.
    * Download the River Flood Hazard Maps at Global Scale from the
      `JRC Data Catalogue <https://data.jrc.ec.europa.eu/collection/id-0054>`_. Create
      the directory ``~/climdada/data/glofas-computation``, unzip the downloaded files
      and place the resulting directories into this directory.

    Parameters
    cfg : Path or str
        Path to the configuration file to use. DO NOT CHANGE THIS except you know exactly
        what you are doing.
    dask_client_kwargs
        Keywords arguments passed to the ``dask.distributed.Client`` for parallel
        computation.
    """
    with dask_client_or_not(**dask_client_kwargs):
        dantro_transform(cfg)


def compute_hazard_series(
    cfg: Union[Path, str] = DEFAULT_GLOFAS_CFG,
    hazard_concat_dim: str = "number",
    **dask_client_kwargs,
):
    """Compute a series of flood hazards from GloFAS discharge data

    This requires the Gumbel distribution fits and the flood maps to be stores as netCDF
    files in the ``data_dir`` specified in the ``cfg``. If you did not retrieve these
    files, execute `py:func:climada_petals.hazard.rf_glofas.setup`.

    We supply a default configuration that computes a flood hazard for a specific
    scenario. Copy this configuration somewhere, change it to your liking, and then
    pass the path to this file as ``cfg`` parameter here.

    The resulting flood data is usually multi-dimensional. For example, you might have
    downloaded ensemble data over an extended period of time. Therefore, this function
    returns a ``pandas.Series``. Each entry of the series is a ``Hazard`` object whose
    events have the same coordinates in this multi-dimensional space except the one
    given by ``hazard_concat_dim``. For example, if your data space has the dimensions
    ``time``, ``lead_time`` and ``number``, and you choose
    ``hazard_concat_dim="number"``, then the index of the series will be a ``MultiIndex``
    from ``time`` and ``lead_time``, and a single hazard object will contain all events
    along the ``number`` axis.

    Parameters
    ----------
    cfg : Path or str
        Path of the configuration file specifying the transformations.
    hazard_concat_dim : str
        The data dimension along which to concatenate events into a single hazard object.
        All other remaining dimensions (except ``longitude`` and ``latitude``) will
        become part of the ``MultiIndex`` for the returned series.
    dask_client_kwargs
        Keywords arguments passed to the ``dask.distributed.Client`` for parallel
        computation.

    Returns
    -------
    hazards : pd.Series
        A series of hazards with Hazard objects concatenated along ``hazard_concat_dim``
        and ``MultiIndex`` containing all other dimensions, except ``latitude``,
        ``longitude``, and the ``hazard_concat_dim``.
    """
    # Maybe create a client for parallel computing
    with dask_client_or_not(**dask_client_kwargs):

        # Perform transformation and retrieve result
        dm = dantro_transform(cfg)
        ds_hazard = dm["flood_depth"].data.to_dataset()

        def create_hazard(ds: xr.Dataset) -> RiverFlood:
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
