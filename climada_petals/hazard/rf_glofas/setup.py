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
from urllib.parse import urlparse
from zipfile import ZipFile
import logging
import shutil

import xarray as xr
import requests

from .transform_ops import (
    merge_flood_maps,
    download_glofas_discharge,
    fit_gumbel_r,
    save_file,
)
from .rf_glofas import DEFAULT_DATA_DIR

LOGGER = logging.getLogger(__name__)

JRC_FLOOD_HAZARD_MAPS = [
    "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp10y.zip",
    "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp20y.zip",
    "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp50y.zip",
    "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp100y.zip",
    "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp200y.zip",
    "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp500y.zip",
]

FLOPROS_DATA = \
    "https://nhess.copernicus.org/articles/16/1049/2016/nhess-16-1049-2016-supplement.zip"

GUMBEL_FIT_DATA = \
    "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/641667/gumbel-fit.nc"


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
    for url in JRC_FLOOD_HAZARD_MAPS:
        # Set output path for the archive
        file_name = Path(urlparse(url).path).stem
        output_path = Path(output_dir) / file_name
        output_path.mkdir(exist_ok=True)

        # Download the file (streaming, because they are around 45 MB)
        response = requests.get(url, stream=True)
        with TemporaryFile(suffix=".zip") as file:
            for chunk in response.iter_content(chunk_size=10 * 1024):
                file.write(chunk)

            # Unzip the file
            with ZipFile(file) as zipfile:
                zipfile.extractall(output_path)


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

    # Find flood maps
    flood_maps_paths = list(Path(flood_maps_dir).glob("**/floodMapGL_rp*y.tif"))
    flood_maps_paths_nc = [path.with_suffix(".nc") for path in flood_maps_paths]

    # Rewrite GeoTIFFs as NetCDFs
    LOGGER.debug("Rewriting flood hazard maps to NetCDF files")
    for path, path_nc in zip(flood_maps_paths, flood_maps_paths_nc):
        if not path_nc.is_file():
            # This uses rioxarray to open a GeoTIFF as an xarray DataArray:
            with xr.open_dataarray(path, engine="rasterio", chunks="auto") as d_arr:
                save_file(d_arr, path_nc, zlib=True)

    # Load NetCDFs and merge
    LOGGER.debug("Merging flood hazard maps into single dataset")
    flood_maps = {
        str(path): xr.open_dataset(path, engine="netcdf4", chunks="auto")["band_data"]
        for path in flood_maps_paths_nc
    }
    da_flood_maps = merge_flood_maps(flood_maps)
    save_file(da_flood_maps, output_dir / "flood-maps.nc", zlib=True)


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
        "2015",
        num_proc=num_downloads,
        preprocess=lambda x: x.groupby("time.year").max(),
        open_mfdataset_kw=dict(
            concat_dim="year",
            chunks=dict(time=-1, longitude="auto", latitude="auto"),
            parallel=parallel,
        ),
    )
    discharge_file = output_dir / "discharge.nc"
    discharge.to_netcdf(discharge_file, engine="netcdf4")
    discharge.close()

    # Fit Gumbel
    LOGGER.debug("Fitting Gumbel distributions to historical discharge data")
    with xr.open_dataarray(
        discharge_file, chunks=dict(time=-1, longitude=50, latitude=50)
    ) as discharge:
        fit = fit_gumbel_r(discharge, min_samples=10)
        fit.to_netcdf(output_dir / "gumbel-fit.nc", engine="netcdf4")


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
):
    """Set up the data for river flood computations.

    This performs two tasks:

    #. Downloading the JRC river flood hazard maps and merging them into a single NetCDF
       dataset.
    #. Downloading the FLOPROS flood protection database.
    #. Downloading the Gumbel distribution parameters fitted to GloFAS river discharge
       reanalysis data from 1979 to 2015.

    Parameters
    ----------
    output_dir : Path or str, optional
        The directory to store the datasets into.
    """
    # Make sure the path exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    setup_flood_hazard_maps(
        flood_maps_dir=DEFAULT_DATA_DIR / "flood-maps", output_dir=output_dir
    )
    download_flopros_database(output_dir)
    download_gumbel_fit(output_dir)
