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
from contextlib import contextmanager
from typing import Union

import xarray as xr
import pandas as pd
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
    ...     xr.open_dataset("data.nc", chunks="auto").median()
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
    data: xr.Dataset, intensity: str, event_dim: str
) -> Union[pd.Series, Hazard]:
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
    data : xarray.Dataset
        Data to load a hazard series from.
    intensity : str
        Name of the dataset variable to read as hazard intensity.
    event_dim : str
        Name of the dimension to be used as event dimension in the hazards. All other
        dimension names except the dimensions for longitude and latitude will make up the
        hierarchy of the ``MultiIndex`` of the resulting series.

    Returns
    -------
    pandas.Series
        Series of ``Hazard`` objects with events along ``event_dim`` and with
        a ``MultiIndex`` of the remaining dimensions.

    Tip
    ---
    This function must transpose the underlying data in the dataset to convenietly build
    ``Hazard`` objects. To ensure that this is an efficient operation, avoid plugging
    the return value of
    :py:meth:`~climada_petals.hazard.rf_glofas.river_flood_computation.RiverFloodInundation.compute`
    directly into this function, especially for **large data**. Instead, save the data
    first using :py:func:`~climada_petals.hazard.rf_glofas.transform_ops.save_file`,
    then re-open the data with xarray and call this function on it.

    Examples
    --------

    Execute the default pipeline and retrieve the Hazard series

    >>> import xarray as xr
    >>> dset = xr.open_dataset("flood.nc")
    >>> sorted(list(dset.dims.keys()))
    ["date", "latitude", "longitude", "number", "sample"]

    >>> from climada_petals.hazard.rf_glofas import hazard_series_from_dataset
    >>> with xr.open_dataset("flood.nc") as dset:
    >>>     hazard_series_from_dataset(dset, "flood_depth_flopros", "number")
    date        sample
    2022-08-10  0       <climada.hazard.base.Hazard ...
                1       <climada.hazard.base.Hazard ...
    2022-08-11  0       <climada.hazard.base.Hazard ...
                1       <climada.hazard.base.Hazard ...
    Length: 4, dtype: object
    """
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
        return pd.Series(hazards, index=index)

    return create_hazard(data)
