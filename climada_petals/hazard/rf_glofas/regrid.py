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

xesmf regrid routine
"""

from typing import Optional

import xarray as xr
import numpy as np
import xesmf as xe
import rioxarray

from .transform_ops import sel_lon_lat_slice


def regrid(
    return_period: xr.DataArray,
    flood_maps: xr.DataArray,
    method: str = "bilinear",
    regridder: Optional[xe.Regridder] = None,
    return_regridder: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, xe.Regridder]:
    """Regrid the return period onto the flood maps grid
    
    Parameters
    ----------
    return_period : xr.DataArray
        The return period data to be regridded.
    flood_maps : xr.DataArray
        The flood maps serving as target grid onto which the return period data is
        regridded.
    method : str
        The regridding method. Default: ``"bilinear"``.
        See https://xesmf.readthedocs.io/en/stable/notebooks/Compare_algorithms.html
    regridder : xesmf.Regridder (optional)
        The regridder instance to use. If ``None`` (default), a new regridder is built.
        Use ``return_regridder`` to return the regridder instance.
    return_regridder : bool
        If ``True``, return the regridder instance. Default: ``False``.

    Returns
    -------
    return_period_regridded : xr.DataArray
        The regridded return period data.
    regridder : xesmf.Regridder
        The regridder instance for later reuse. Only returned if
        ``return_regridder=True``.
    """
    # Select lon/lat for flood maps
    flood_maps = sel_lon_lat_slice(flood_maps, return_period)

    # Mask return period so NaNs are not propagated
    rp = return_period.to_dataset(name="data")
    dims_to_remove = set(rp.sizes.keys()) - {"longitude", "latitude"}
    dims_to_remove = {dim: 0 for dim in dims_to_remove}
    rp["mask"] = xr.where(~np.isnan(rp["data"].isel(dims_to_remove)), 1, 0).astype(
        np.int8
    )

    # NOTE: Masking here would omit all return periods outside flood plains
    #       (This might be desirable at some point?)
    flood = flood_maps.isel(return_period=-1).to_dataset(name="data")
    flood["mask"] = xr.where(~np.isnan(flood["data"]), 1, 0).astype(np.int8)

    # Build regridder
    if regridder is None:
        regridder = xe.Regridder(
            rp,
            flood,
            method=method,
            extrap_method="nearest_s2d",
            # filename="/Users/ldr.riedel/Desktop/regridder/regridder.nc",
            # parallel=True,
            # unmapped_to_nan=False,
        )

    # Set chunksizes
    return_period = return_period.unify_chunks()
    chunksizes = {dim: np.max(sizes) for dim, sizes in return_period.chunksizes.items()}

    return_period_regridded = (
        regridder(return_period, output_chunks=chunksizes)
        .rename(return_period.name)
    )
    if return_regridder:
        return return_period_regridded, regridder

    return_period_regridded.rio.write_crs(flood_maps.rio.crs, inplace=True)
    return return_period_regridded
