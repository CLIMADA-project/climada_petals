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

User Interface for GloFAS River Flood Module
"""

import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Union, Optional

import xarray as xr
import pandas as pd
import geopandas as gpd
from dask.distributed import Client

from climada.util.constants import SYSTEM_DIR
from climada.hazard import Hazard

LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_DIR = SYSTEM_DIR / "river-flood-computation"


@contextmanager
def dask_client(n_workers, threads_per_worker, memory_limit, *args, **kwargs):
    """Create a context with a ``dask.distributed.Client``.

    This is a lightweight wrapper and intended to expose only the most important
    parameters to end users.

    Parameters
    ----------
    n_workers : int
        Number of parallel processes to launch.
    threads_per_worker : int
        Compute threads launched by each worker.
    memory_limit : str
        Memory limit for each process. Example: 4 GB can be expressed as ``4000M`` or
        ``4G``.
    args, kwargs
        Additional (keyword) arguments passed to the ``dask.distributed.Client``
        constructor.

    Example
    -------
    >>> with dask_client(n_workers=2, threads_per_worker=2, memory_limit="4G"):
    ...     data_manager = dantro_transform("my_cfg.yml")
    """
    # Yield the client with the arguments, and close it afterwards
    LOGGER.info("Creating dask.distributed.Client")
    client = Client(
        *args,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        **kwargs,
    )
    try:
        yield client
    finally:
        LOGGER.info("Closing dask.distributed.Client")
        client.close()


def hazard_series_from_dataset(
    data: Union[Path, str, xr.Dataset], intensity: str, event_dim: str
) -> pd.Series:
    """Create a series of Hazard objects from a multi-dimensional dataset

    The input flood data is usually multi-dimensional. For example, you might have
    downloaded ensemble data over an extended period of time. Therefore, this function
    returns a ``pandas.Series``. Each entry of the series is a ``Hazard`` object whose
    events have the same coordinates in this multi-dimensional space except the one
    given by ``event_dim``. For example, if your data space has the dimensions ``time``,
    ``lead_time`` and ``number``, and you choose ``event_dim="number"``, then the index
    of the series will be a ``MultiIndex`` from ``time`` and ``lead_time``, and a single
    hazard object will contain all events along the ``number`` axis for a given
    MultiIndex.

    Parameters
    ----------
    data : xarray.Dataset or Path or str
        Data to load a hazard series from. May be an opened Dataset or a path to a file
        that can be opened by xarray.
    event_dim : str
        Name of the dimension to be used as event dimension in the hazards. All other
        dimension names except the dimensions for longitude and latitude will make up the
        hierarchy of the ``MultiIndex`` of the resulting series.

    Returns
    -------
    pandas.Series
        Series of ``RiverFlood`` objects with events along ``event_dim`` and with
        a ``MultiIndex`` of the remaining dimensions.

    Examples
    --------

    Execute the default pipeline and retrieve the Hazard series

    >>> from climada_petals.hazard.rf_glofas import dantro_transform
    >>> data_manager = dantro_transform()
    >>> dset = data_manager["flood_depth"].data
    >>> sorted(list(dset.dims.keys()))
    ["date", "latitude", "longitude", "number", "select"]

    >>> from climada_petals.hazard.rf_glofas import hazard_series_from_dataset
    >>> hazard_series_from_dataset(dset)
    select  time
    0       2022-08-10    <climada_petals.hazard.river_flood.RiverFlood ...
            2022-08-11    <climada_petals.hazard.river_flood.RiverFlood ...
    1       2022-08-10    <climada_petals.hazard.river_flood.RiverFlood ...
            2022-08-11    <climada_petals.hazard.river_flood.RiverFlood ...
    Length: 4, dtype: object
    """
    if not isinstance(data, xr.Dataset):
        data = xr.open_dataset(data, chunks="auto")

    def create_hazard(dataset: xr.Dataset) -> Hazard:
        """Create hazard from a GloFASRiverFlood hazard dataset"""
        return Hazard.from_xarray_raster(
            dataset,
            hazard_type="RF",
            intensity=intensity,
            intensity_unit="m",
            coordinate_vars=dict(event=event_dim),
            data_vars=dict(date="time"),
            rechunk=True,
        )

    # Iterate over all dimensions that are not lon, lat, or 'event_dim'
    iter_dims = list(set(data.dims) - {"longitude", "latitude", event_dim})
    if iter_dims:
        index = pd.MultiIndex.from_product(
            [data[dim].values for dim in iter_dims], names=iter_dims
        )
        hazards = [
            create_hazard(data.sel(dict(zip(iter_dims, idx))))
            for idx in index.to_flat_index()
        ]
    else:
        index = None
        hazards = [create_hazard(data)]

    return pd.Series(hazards, index=index)
