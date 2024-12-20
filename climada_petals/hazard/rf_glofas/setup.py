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

Module preparing data for the river flood inundation model
"""

from typing import Union
from pathlib import Path
from tempfile import TemporaryFile, TemporaryDirectory
from zipfile import ZipFile
import logging
import shutil

import xarray as xr
import requests
import geopandas as gpd

from .transform_ops import (
    merge_flood_maps,
    download_glofas_discharge,
    fit_gumbel_r,
    save_file,
)
from .rf_glofas import DEFAULT_DATA_DIR

LOGGER = logging.getLogger(__name__)

JRC_FLOOD_HAZARD_MAP_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/RP{rp}/ID{id}_{name}_RP{rp}_depth.tif"

JRC_FLOOD_HAZARD_MAP_RPS = [10, 20, 50, 75, 100, 200, 500]

FLOPROS_DATA = "https://nhess.copernicus.org/articles/16/1049/2016/nhess-16-1049-2016-supplement.zip"

GUMBEL_FIT_DATA = "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/641667/gumbel-fit.nc"


def download_flopros_database(output_dir: Union[str, Path] = DEFAULT_DATA_DIR):
    """Download the FLOPROS database and place it into the output directory.

    Download the supplementary material of `P. Scussolini et al.: "FLOPROS: an evolving
    global database of flood protection standards"
    <https://dx.doi.org/10.5194/nhess-16-1049-2016>`_, extract the zipfile, and retrieve
    the shapefile within. Discard the temporary data afterwards.
    """
    LOGGER.debug("Downloading FLOPROS database")

    # Download the file
    response = requests.get(FLOPROS_DATA, stream=True)
    with TemporaryFile(suffix=".zip") as file:
        for chunk in response.iter_content(chunk_size=10 * 1024):
            file.write(chunk)

        # Unzip the folder
        with TemporaryDirectory() as tmpdir:
            with ZipFile(file) as zipfile:
                zipfile.extractall(tmpdir)

            shutil.copytree(
                Path(tmpdir) / "Scussolini_etal_Suppl_info/FLOPROS_shp_V1",
                Path(output_dir) / "FLOPROS_shp_V1",
                dirs_exist_ok=True,
            )


def download_flood_hazard_maps(output_dir: Union[str, Path]):
    """Download the JRC flood hazard maps and unzip them

    This stores the downloaded zip files as temporary files which are discarded after
    unzipping.
    """
    LOGGER.debug("Downloading flood hazard maps")
    # Load information on the tiles
    tile_extents = gpd.read_file(
        "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/tile_extents.geojson"
    )

    for return_period in JRC_FLOOD_HAZARD_MAP_RPS:
        # Set output path for the files
        output_path = Path(output_dir) / f"RP{return_period}"
        output_path.mkdir(exist_ok=True)

        # Download the files (streaming, because they may be tens of MB large)
        for _, tile in tile_extents.iterrows():
            url = JRC_FLOOD_HAZARD_MAP_URL.format(
                rp=return_period, id=tile["id"], name=tile["name"]
            )
            LOGGER.debug("Downloading %s", url)
            filename = output_path / Path(url).name
            response = requests.get(url, stream=True)
            with open(filename, "w") as file:
                for chunk in response.iter_content(chunk_size=10 * 1024):
                    file.write(chunk)


def setup_flood_hazard_maps(flood_maps_dir: Path, output_dir=DEFAULT_DATA_DIR):
    """Download the flood hazard maps and merge them into a single NetCDF file

    Maps will be downloaded into ``flood_maps_dir`` if it does not exist. Then, the
    single maps are re-written as NetCDF files, if these do not exist. Finally, all maps
    are merged into a single dataset and written to the ``output_dir``. Because NetCDF
    files are more flexibly read and written, this procedure is more efficient that
    directly merging the GeoTIFF files into a single dataset.

    Parameters
    ----------
    flood_maps_dir : Path
        Storage directory of the flood maps as GeoTIFF files. Will be created if it does
        not exist, in which case the files are automatically downloaded.
    output_dir : Path
        Directory to store the flood maps dataset.
    """
    # Download flood maps if directory does not exist
    if not flood_maps_dir.is_dir():
        LOGGER.debug(
            "No flood maps found. Downloading GeoTIFF files to %s", flood_maps_dir
        )
        flood_maps_dir.mkdir()
        download_flood_hazard_maps(flood_maps_dir)

    # STEP 0: Rewrite tiles as NetCDFs ??

    # STEP 1: Merge all tiles for single RP into one NetCDF file
    LOGGER.debug("Merging flood hazard map tiles into NetCDF files")
    for return_period in JRC_FLOOD_HAZARD_MAP_RPS:
        flood_maps_rp_dir = flood_maps_dir / f"RP{return_period}"
        ds_rp_flood_maps = (
            xr.open_mfdataset(
                flood_maps_rp_dir / "*_depth.tif",
                chunks="auto",
                combine="by_coords",
                engine="rasterio",
            )
            .drop_vars("spatial_ref", errors="ignore")
            .squeeze("band", drop=True)
        )
        save_file(
            ds_rp_flood_maps,
            flood_maps_rp_dir / "depth.nc",
            zlib=True,
        )
        ds_rp_flood_maps.close()

    # STEP 2: Merge all RPs into one NetCDF file
    LOGGER.debug("Merging flood hazard maps into single dataset")
    flood_maps = {
        return_period: xr.open_dataset(
            flood_maps_dir / f"RP{return_period}" / "depth.nc",
            engine="netcdf4",
            chunks="auto",
        )["band_data"]
        for return_period in JRC_FLOOD_HAZARD_MAP_RPS
    }
    da_flood_maps = merge_flood_maps(flood_maps)
    save_file(da_flood_maps, output_dir / "flood-maps.nc", zlib=True)
    da_flood_maps.close()


def setup_gumbel_fit(
    output_dir=DEFAULT_DATA_DIR, num_downloads: int = 1, parallel: bool = False
):
    """Download historical discharge data and compute the Gumbel distribution fits.

    Data is downloaded from the Copernicus Climate Data Store (CDS).

    Parameters
    ----------
    output_dir
        The directory to place the resulting file
    num_downloads : int
        Number of parallel downloads from the CDS. Defaults to 1.
    parallel : bool
        Whether to preprocess data in parallel. Defaults to ``False``.
    """
    # Download discharge and preprocess
    LOGGER.debug("Downloading historical discharge data")
    discharge = download_glofas_discharge(
        "historical",
        "1979",
        "2023",
        num_proc=num_downloads,
        preprocess=lambda x: x.groupby("time.year").max(),
        open_mfdataset_kw=dict(
            concat_dim="year",
            chunks=dict(time=-1, longitude="auto", latitude="auto"),
            parallel=parallel,
        ),
    )
    discharge_file = output_dir / "discharge.nc"
    save_file(discharge, discharge_file)
    discharge.close()

    # Fit Gumbel
    LOGGER.debug("Fitting Gumbel distributions to historical discharge data")
    with xr.open_dataarray(
        discharge_file, chunks=dict(time=-1, year=-1, longitude=50, latitude=50)
    ) as discharge:
        fit = fit_gumbel_r(discharge, min_samples=10)
        save_file(fit, output_dir / "gumbel-fit.nc", dtype="float64")


def download_gumbel_fit(output_dir=DEFAULT_DATA_DIR):
    """Download the pre-computed Gumbel parameters from the ETH research collection.

    Download dataset of https://doi.org/10.3929/ethz-b-000641667
    """
    LOGGER.debug("Downloading Gumbel fit parameters")
    response = requests.get(GUMBEL_FIT_DATA, stream=True)
    with open(output_dir / "gumbel-fit.nc", "wb") as file:
        for chunk in response.iter_content(chunk_size=10 * 1024):
            file.write(chunk)


def setup_all(
    output_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    compute_gumbel_fit: bool = False,
    **setup_gumbel_fit_kwargs,
):
    """Set up the data for river flood computations.

    This performs two tasks:

    #. Downloading the JRC river flood hazard maps and merging them into a single NetCDF
       dataset.
    #. Downloading the FLOPROS flood protection database.
    #. Downloading the Gumbel distribution parameters fitted to GloFAS river discharge
       reanalysis data from 1979 to 2023.

    Parameters
    ----------
    output_dir : Path or str, optional
        The directory to store the datasets into.
    compute_gumbel_fit : bool
        If ``True``, recompute the Gumbel fits instead of downloading the data stored in
        the ETH research collection. See :py:func:`setup_gumbel_fit`.
    """
    # Make sure the path exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    setup_flood_hazard_maps(
        flood_maps_dir=DEFAULT_DATA_DIR / "flood-maps", output_dir=output_dir
    )
    download_flopros_database(output_dir)

    if compute_gumbel_fit:
        setup_gumbel_fit(output_dir, **setup_gumbel_fit_kwargs)
    else:
        download_gumbel_fit(output_dir)
