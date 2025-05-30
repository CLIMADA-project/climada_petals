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

Functions for handling JRC flood hazard map tiles.
"""

from typing import Union, Optional
from pathlib import Path
import logging

import numpy as np
import xarray as xr
import requests
import geopandas as gpd
import pandas as pd

from .rf_glofas import DEFAULT_DATA_DIR

LOGGER = logging.getLogger(__name__)

JRC_FLOOD_HAZARD_MAP_TILES_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/tile_extents.geojson"
"""URL of the tile extents dataset"""

JRC_FLOOD_HAZARD_MAP_TILES_FILENAME = Path(JRC_FLOOD_HAZARD_MAP_TILES_URL).name

JRC_FLOOD_HAZARD_MAP_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/RP{rp}/ID{tile_id}_{name}_RP{rp}_depth.tif"
"""URL template of the flood hazard map tiles"""

JRC_FLOOD_HAZARD_MAP_RPS = [10, 20, 50, 75, 100, 200, 500]
"""Return periods of flood hazard maps"""


def tile_url(return_period: int, tile_id: int, name: str) -> str:
    """Get map tile URL by specifying return period, ID, and name

    Parameters
    ----------
    return_period : int
        The return period. See :py:const:`JRC_FLOOD_HAZARD_MAP_RPS`.
    tile_id : int
        The tile ID according to the extents dataset
    name : str
        The tile name according to the extents dataset

    Returns
    -------
    str
        The URL of the flood hazard map tile

    See Also
    --------
    :py:func:`open_flood_maps_extents`
        The function that loads the tile extents dataset.
    """
    return JRC_FLOOD_HAZARD_MAP_URL.format(rp=return_period, tile_id=tile_id, name=name)


def tile_filename(return_period: int, tile_id: int, name: str) -> str:
    """Get map tile file name by specifying return period, ID, and name

    The file name is the last component of the URL as returned by :py:func:`tile_url`.
    """
    return Path(tile_url(return_period=return_period, tile_id=tile_id, name=name)).name


def download_flood_maps_extents(
    filepath: Union[str, Path] = DEFAULT_DATA_DIR / JRC_FLOOD_HAZARD_MAP_TILES_FILENAME,
):
    """Download the extents dataset for the flood hazard map tiles

    This dataset contains information on which tile covers which geographical area.
    """
    output_path = Path(filepath).parent
    output_path.mkdir(exist_ok=True)

    response = requests.get(JRC_FLOOD_HAZARD_MAP_TILES_URL, timeout=10)
    with open(filepath, "wb") as file:
        file.write(response.content)


def open_flood_maps_extents(
    filepath: Union[str, Path] = DEFAULT_DATA_DIR / JRC_FLOOD_HAZARD_MAP_TILES_FILENAME,
) -> gpd.GeoDataFrame:
    """Open the extents dataset for the flood hazard map tiles.

    If the file does not exist, at the specified location, it will be downloaded with
    :py:func:`download_flood_maps_extents`.
    """
    if not Path(filepath).is_file():
        download_flood_maps_extents(filepath)
    return gpd.read_file(filepath)


def download_flood_map_tiles(
    output_dir: Union[str, Path] = DEFAULT_DATA_DIR / "flood-maps",
    tiles: Optional[gpd.GeoDataFrame] = None,
    overwrite: bool = False,
):
    """Download appropriate tiles of the flood hazard maps

    Parameters
    ----------
    output_dir : Path or str
        The directory where the tiles will be downloaded to.
    tiles : geopandas.GeoDataFrame, optional
        A subset of the extents dataset. If provided, this will only download the tiles
        specified by this subset. If ``None``, this will download all tiles.
    overwrite : bool
        If existing files should be overwritten or skipped.

    Return
    ------
    tiles : GeoDataFrame
        The tiles GeoDataFrame with paths added

    See Also
    --------
    :py:func:`open_flood_maps_extents`
        The function that loads the tile extents dataset.
    """
    if tiles is None:
        # Load all tiles
        tiles = open_flood_maps_extents()

    # Create output dir
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for return_period in JRC_FLOOD_HAZARD_MAP_RPS:
        # Set output path for the files
        rp_path = output_path / f"RP{return_period}"
        rp_path.mkdir(exist_ok=True)

        # Download the files (streaming, because they may be tens of MB large)
        for _, tile in tiles.iterrows():
            url = tile_url(
                return_period=return_period, tile_id=tile["id"], name=tile["name"]
            )
            filepath = rp_path / tile_filename(
                return_period=return_period, tile_id=tile["id"], name=tile["name"]
            )
            if filepath.is_file() and not overwrite:
                LOGGER.debug("Skipping file %s because it already exists", filepath)
                continue

            LOGGER.debug("Downloading %s", url)
            response = requests.get(url, stream=True, timeout=10)
            with open(filepath, "wb") as file:
                for chunk in response.iter_content(chunk_size=10 * 1024):
                    file.write(chunk)


def open_flood_map_tiles(
    flood_maps_dir: Union[str, Path] = DEFAULT_DATA_DIR / "flood-maps",
    tiles: Optional[gpd.GeoDataFrame] = None,
) -> xr.DataArray:
    """Open specific tiles of the flood hazard maps with xarray

    Parameters
    ----------
    flood_maps_dir : Path or str
        The directory containing the flood maps to load
    tiles : geopandas.GeoDataFrame, optional
        A subset of the extents dataset. If provided, this will only open the tiles
        specified by this subset. If ``None``, this will open all tiles.

    Returns
    -------
    xarray.DataArray
        The union of flood map tiles opened with xarray.

    See Also
    --------
    :py:func:`open_flood_maps_extents`
        The function that loads the tile extents dataset.
    """
    if tiles is None:
        tiles = open_flood_maps_extents()

    def open_rp(return_period: int) -> xr.DataArray:
        """Open and merge flood maps tiles for a single return period"""
        tif_files = [
            Path(flood_maps_dir)
            / f"RP{return_period}"
            / tile_filename(
                return_period=return_period, tile_id=tile["id"], name=tile["name"]
            )
            for _, tile in tiles.iterrows()
        ]
        with xr.open_mfdataset(
            tif_files,
            chunks="auto",
            combine="by_coords",
            engine="rasterio",
        ) as ds:
            return ds.drop_vars("spatial_ref", errors="ignore").squeeze(
                "band", drop=True
            )["band_data"]

    darrs = [open_rp(rp) for rp in JRC_FLOOD_HAZARD_MAP_RPS]
    da_null = xr.full_like(darrs[0], np.nan)

    return (
        xr.concat(
            [da_null] + darrs,
            pd.Index([1] + JRC_FLOOD_HAZARD_MAP_RPS, name="return_period"),
        )
        .rename(x="longitude", y="latitude")
        .rename("flood_depth")
    )
