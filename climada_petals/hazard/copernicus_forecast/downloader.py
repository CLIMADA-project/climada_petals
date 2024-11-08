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

Functionality to download data from the Copernicus Data Stores.
"""

import logging
import cdsapi
from pathlib import Path

from climada.util.constants import SYSTEM_DIR

LOGGER = logging.getLogger(__name__)


def download_data(dataset, params, filename=None, overwrite=False):
    """Download data from Copernicus Data Stores (e.g., ds.climate.copernicus.eu,
    ads.atmosphere.copernicus.eu and ewds.climate.copernicus.eu) using specified dataset type and parameters.

    Parameters
    ----------
    dataset : str
        The dataset to retrieve (e.g., 'seasonal-original-single-levels', 'sis-heat-and-cold-spells').
    params : dict
        Dictionary containing the parameters for the CDS API call (e.g., variables, time range, area).
    filename : pathlib.Path or str
        Full path and filename where the downloaded data will be stored.
    overwrite : bool, optional
        If True, overwrite the file if it already exists. If False, skip downloading
        if the file is already present. The default is False.

    Raises
    ------
    FileNotFoundError
        Raised if the download attempt fails and the file is not found at the specified location.
    Exception
        Raised for any other error during the download process.
    """

    # Warning about terms and conditions
    cds_filepath = str(Path.home()) + "/.cdsapirc"
    with open(cds_filepath, "r") as file:
        url = file.read().split("\n")[0].split(" ")[1].strip().removesuffix("/api")
    LOGGER.warning(
        "Please ensure you have reviewed and accepted the terms and conditions "
        "for the use of this dataset. Access the terms here: "
        f"{url}/datasets/{dataset}?tab=download"
    )

    # Check if file exists and skip download if overwrite is False
    if filename:
        if Path(filename).exists() and not overwrite:
            LOGGER.warning(f"File {filename} already exists. Skipping download.")
            return

    try:
        # Initialize CDS API client
        c = cdsapi.Client()
        request = c.retrieve(dataset, params)

        # prepare filename if not given
        if not filename:
            filename = (
                SYSTEM_DIR
                / f'copernicus_data/{dataset}/{request.location.split("/")[-1]}'
            )

        # make parent directory
        output_dir = Path(filename).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # download data
        request.download(filename)

        # Check if the file was successfully downloaded
        if not Path(filename).exists():
            raise FileNotFoundError(f"Failed to download {filename}.")

        LOGGER.info(f"File successfully downloaded to {filename}.")

    except Exception as e:
        # user key is wrong
        if "401 Client Error" in str(e):
            error_message = (
                "Authentification failed. Please ensure the"
                "correct key in the .cdsapirc file (see instructions)."
            )
        # dataset does not exist
        elif "404 Client Error" in str(e):
            error_message = f'Dataset "{url}/datasets/{dataset} not found. Please ensure the correct store and dataset.'
        # terms not accepted
        elif "403 Client Error" in str(e):
            error_message = f"Required licences not accepted. Please accept here: {url}/datasets/{dataset}?tab=download"
        # parameter choice not available
        elif "MARS returned no data" in str(e):
            error_message = "No data available for the given Copernicus data store, dataset, and parameters. This may indicate unavailable or incorrect parameter selection. Please verify the existence of the data on the Climate Data Store website."
        # general error
        else:
            LOGGER.warning(f"Error downloading file {filename}: {e}")
            raise e

        raise Exception(error_message)
