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
import os
from typing import Union, Optional
from pathlib import Path
from tempfile import TemporaryFile
from urllib.parse import urlparse
from zipfile import ZipFile
import logging

import xarray as xr
import requests

from .transform_ops import merge_flood_maps, download_glofas_discharge, fit_gumbel_r
from .rf_glofas import dask_client, DEFAULT_DATA_DIR

LOGGER = logging.getLogger(__file__)

JRC_FLOOD_HAZARD_MAPS = [
    "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp10y.zip",
    "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp20y.zip",
    "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp50y.zip",
    "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp100y.zip",
    "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp200y.zip",
    "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/floodMapGL_rp500y.zip",
]


def download_flood_hazard_maps(output_dir: Union[str, Path]):
    """Download the JRC flood hazard maps and unzip them

    This stores the downloaded zip files as temporary files which are discarded after
    unzipping.
    """
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


def setup_flood_hazard_maps(output_dir, flood_maps_dir):
    # Download flood maps if no directory is given
    if flood_maps_dir is None:
        LOGGER.debug("Downloading and unzipping flood hazard maps")
        flood_maps_dir = output_dir / "flood-maps"
        flood_maps_dir.mkdir(exist_ok=True)
        download_flood_hazard_maps(flood_maps_dir)

    # Load flood maps
    LOGGER.debug("Merging flood hazard maps into single dataset")
    flood_maps_paths = Path(flood_maps_dir).glob("**/floodMapGL_rp*y.tif")
    flood_maps = {
        str(path): xr.open_dataarray(path, engine="rasterio", chunks="auto")
        for path in flood_maps_paths
    }
    da_flood_maps = merge_flood_maps(flood_maps)
    da_flood_maps.to_netcdf(output_dir / "flood-maps.nc")


def setup_gumbel_fit(output_dir, num_downloads: int = 1, parallel: bool = False):
    # Download discharge and preprocess
    LOGGER.debug("Downloading historical discharge data")
    discharge = download_glofas_discharge(
        "historical",
        "1979",
        "2015",
        num_proc=num_downloads,
        preprocess="x.groupby('time.year').max()",
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


def setup(
    output_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    flood_maps_dir: Optional[Union[str, Path]] = None,
    num_workers: int = 1,
    memory_limit: str = "4G",
):
    """Set up the data for river flood computations

    This performs two tasks:

    1. Merging the JRC river flood hazard maps into a single NetCDF dataset.
    2. Fitting right-handed Gumbel distributions to every grid cell in the GloFAS
       river discharge reanalysis data from 1979 to 2015, and storing the fit parameters
       as NetCDF dataset.

    If no data is provided, all will be downloaded automatically from the internet.

    Parameters
    ----------
    output_dir : Path or str, optional
        The directory to store the datasets into.
    flood_maps_dir : Path or str, optional
        The directory containing the single flood hazard map GeoTIFF files. If ``None``
        (default), these will be downloaded the the JRC data catalogue automatically.
        See :py:func:`download_flood_hazard_maps`.
    num_workers : int, optional
        The number of worker processes to use for fitting the Gumbel distributions in
        parallel. Defaults to 1.
    """
    # Make sure the path exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    setup_flood_hazard_maps(output_dir, flood_maps_dir)

    if num_workers > 1:
        with dask_client(num_workers, 1, memory_limit):
            setup_gumbel_fit(output_dir, num_downloads=num_workers, parallel=True)
    else:
        setup_gumbel_fit(output_dir, parallel=False)
