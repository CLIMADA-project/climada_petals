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

---

Prerequisites:
1. CDS API client installation:
   pip install cdsapi

2. CDS account and API key:
   Register at https://cds.climate.copernicus.eu

3. CDS API configuration:
   Create a .cdsapirc file in your home directory with your API key and URL of the CDS you want to access.
   For instance, if you want to access the Climate Data Store, see here for instructions:
   https://cds.climate.copernicus.eu/how-to-api#install-the-cds-api-client

4. Dataset Terms and Conditions: After selecting the dataset to download, make 
   sure to accept the terms and conditions on the corresponding dataset webpage (under the "download" tab)
   in the CDS portal before running the script.
"""

import logging
import cdsapi
from pathlib import Path

from climada import CONFIG

DATA_DIR = CONFIG.hazard.copernicus.local_data.dir()
LOGGER = logging.getLogger(__name__)


def download_data(dataset, params, filename=None, datastore_url=None, overwrite=False):
    """Download data from Copernicus Data Stores (e.g., cds.climate.copernicus.eu,
    ads.atmosphere.copernicus.eu and ewds.climate.copernicus.eu) using specified dataset type and parameters.

    Parameters
    ----------
    dataset : str
        The dataset to retrieve (e.g., 'seasonal-original-single-levels', 'sis-heat-and-cold-spells').
    params : dict
        Dictionary containing the parameters for the CDS API call (e.g., variables, time range, area).
        To see which parameters are requested for the given dataset, go to the copernicus website of the dataset in the "download" tab,
        tick all required parameter choices. You find the params dicts as "request" dict in the "API request" section.
    filename : pathlib.Path or str
        Full path and filename where the downloaded data will be stored. If None, data will be saved with the filename as suggested by the data store. Defaults to None.
    datastore_url : str
        Url of the Copernicus data store to be accessed. If None, the url of the .cdsapirc file is used. Defaults to None.
    overwrite : bool, optional
        If True, overwrite the file if it already exists. If False, skip downloading
        if the file is already present. The default is False.

    Returns
    ----------
    Path
        Path to the downloaded file if the download was successfull.

    Raises
    ------
    FileNotFoundError
        Raised if the download attempt fails and the file is not found at the specified location.
    Exception
        Raised for any other error during the download process, with further details corresponding to typical errors.
    """

    # Warning about terms and conditions
    if not datastore_url:
        cds_filepath = str(Path.home()) + "/.cdsapirc"
        try:
            with open(cds_filepath, "r") as file:
                url = (
                    file.read()
                    .split("\n")[0]
                    .split(" ")[1]
                    .strip()
                    .removesuffix("/api")
                )
        except FileNotFoundError as e:
            raise FileNotFoundError("No .cdsapirc file in home directory.")
    else:
        if not datastore_url.endswith("/api"):
            raise ValueError("The given datastore_url must end with /api.")
        url = datastore_url.removesuffix("/api")

    # Check if file exists and skip download if overwrite is False
    if filename:
        if Path(filename).exists() and not overwrite:
            LOGGER.warning(f"File {filename} already exists. Skipping download.")
            return

    try:
        # Initialize CDS API client
        c = cdsapi.Client(url=datastore_url)
        request = c.retrieve(dataset, params)

        # prepare filename if not given
        if not filename:
            filename = DATA_DIR / f'{dataset}/{request.location.split("/")[-1]}'

            # Check if file exists and skip download if overwrite is False
            if Path(filename).exists() and not overwrite:
                LOGGER.warning(f"File {filename} already exists. Skipping download.")
                return

        # make parent directory
        output_dir = Path(filename).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # download data
        request.download(filename)

        # Check if the file was successfully downloaded
        if not Path(filename).exists():
            raise FileNotFoundError(f"Failed to download {filename}.")

        LOGGER.info(f"File successfully downloaded to {filename}.")

        return filename

    except Exception as e:
        # user key is wrong
        if "401 Client Error" in str(e):
            error_message = (
                "Authentification failed. Please ensure that the"
                "API key in the .cdsapirc file is correct (see instructions)."
            )
        # dataset does not exist
        elif "404 Client Error" in str(e):
            error_message = f'Dataset "{url}/datasets/{dataset} not found. Please ensure the correct store and dataset.'
        # terms not accepted
        elif "403 Client Error" in str(e):
            error_message = f"Required licences not accepted. Please accept here: {url}/datasets/{dataset}?tab=download"
        # parameter choice not available
        elif "MARS returned no data" in str(e) or "400 Client Error" in str(e):
            error_message = (
                "No data available for the given parameters. This may indicate unavailable or incorrect parameter selection. Please verify the existence of the data on the Climate Data Store website. "
                f'You can find which parameters are requested for the given dataset by indicating all required parameter choices at {url}/datasets/{dataset}?tab=download. The required params dict given in the "request" keyword in the "API request" section.'
            )
        # general error
        else:
            error_message = f"Unexpected error downloading file {filename} (for common error sources, check out https://confluence.ecmwf.int/display/CKB/Common+Error+Messages+for+CDS+Requests). Error description: {str(e)}"

        raise Exception(error_message)
