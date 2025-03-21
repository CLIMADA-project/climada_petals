from typing import Union, Optional
from pathlib import Path
import logging

import numpy as np
import xarray as xr
import requests
import geopandas as gpd
import pandas as pd
import shapely

from .rf_glofas import DEFAULT_DATA_DIR
from .transform_ops import save_file

LOGGER = logging.getLogger(__name__)

JRC_FLOOD_HAZARD_MAP_TILES_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/tile_extents.geojson"

JRC_FLOOD_HAZARD_MAP_TILES_FILENAME = Path(JRC_FLOOD_HAZARD_MAP_TILES_URL).name

JRC_FLOOD_HAZARD_MAP_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/RP{rp}/ID{id}_{name}_RP{rp}_depth.tif"

JRC_FLOOD_HAZARD_MAP_RPS = [10, 20, 50, 75, 100, 200, 500]


def tile_url(return_period: int, id: int, name: str):
    return JRC_FLOOD_HAZARD_MAP_URL.format(rp=return_period, id=id, name=name)


def tile_filename(return_period: int, id: int, name: str):
    return Path(tile_url(return_period=return_period, id=id, name=name)).name


def download_flood_maps_extents(
    filepath: Union[str, Path] = DEFAULT_DATA_DIR / JRC_FLOOD_HAZARD_MAP_TILES_FILENAME,
):
    """Download the extents dataset for the flood hazard map tiles"""
    output_path = Path(filepath).parent
    output_path.mkdir(exist_ok=True)

    response = requests.get(JRC_FLOOD_HAZARD_MAP_TILES_URL)
    with open(filepath, "wb") as file:
        file.write(response.content)


def open_flood_maps_extents(
    filepath: Union[str, Path] = DEFAULT_DATA_DIR / JRC_FLOOD_HAZARD_MAP_TILES_FILENAME,
) -> gpd.GeoDataFrame:
    if not Path(filepath).is_file():
        download_flood_maps_extents(filepath)
    return gpd.read_file(filepath)


def download_flood_map_tiles(
    output_dir: Union[str, Path] = DEFAULT_DATA_DIR / "flood-maps",
    tiles: Optional[gpd.GeoDataFrame] = None,
    overwrite: bool = False,
):
    """Download appropriate tiles of the flood hazard maps

    Return
    ------
    tiles : GeoDataFrame
        The tiles GeoDataFrame with paths added
    """
    if tiles is None:
        tiles = open_flood_maps_extents()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for return_period in JRC_FLOOD_HAZARD_MAP_RPS:
        # Set output path for the files
        rp_path = output_path / f"RP{return_period}"
        rp_path.mkdir(exist_ok=True)

        # Download the files (streaming, because they may be tens of MB large)
        for _, tile in tiles.iterrows():
            url = tile_url(
                return_period=return_period, id=tile["id"], name=tile["name"]
            )
            filepath = rp_path / tile_filename(
                return_period=return_period, id=tile["id"], name=tile["name"]
            )
            if filepath.is_file() and not overwrite:
                LOGGER.debug("Skipping file %s because it already exists", filepath)
                continue

            LOGGER.debug("Downloading %s", url)
            response = requests.get(url, stream=True)
            with open(filepath, "wb") as file:
                for chunk in response.iter_content(chunk_size=10 * 1024):
                    file.write(chunk)


def rewrite_tile(filepath: Union[str, Path], overwrite: bool = False):
    """Rewrite a GeoTIFF tile as NetCDF"""
    output_path = Path(filepath).with_suffix(".nc")
    if overwrite or not output_path.is_file():
        LOGGER.debug("Rewriting %s as NetCDF", filepath)
        with xr.open_dataarray(filepath, chunks="auto", engine="rasterio") as da:
            da = da.drop_vars("spatial_ref", errors="ignore").squeeze("band", drop=True)
            save_file(da, output_path=output_path, zlib=True)

    return output_path


def open_flood_map_tiles(
    flood_maps_dir: Union[str, Path] = DEFAULT_DATA_DIR / "flood-maps",
    tiles: Optional[gpd.GeoDataFrame] = None,
) -> xr.DataArray:
    """Open specific tiles of the flood hazard maps with xarray"""
    if tiles is None:
        tiles = open_flood_maps_extents()

    def open_rp(return_period: int) -> xr.DataArray:
        """Open and merge flood maps tiles for a single return period"""
        tif_files = [
            Path(flood_maps_dir)
            / f"RP{return_period}"
            / tile_filename(
                return_period=return_period, id=tile["id"], name=tile["name"]
            )
            for _, tile in tiles.iterrows()
        ]
        # nc_files = [rewrite_tile(tif) for tif in tif_files]
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
