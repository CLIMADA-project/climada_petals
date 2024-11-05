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

A class to download data from the Copernicus Data Stores.
"""

import logging
import cdsapi
from pathlib import Path

from climada.util.constants import SYSTEM_DIR


class Downloader:
    """A class to download data from the Copernicus Data Stores, including cds.climate.copernicus.eu,
    ads.atmosphere.copernicus.eu and ewds.climate.copernicus.eu.

    This downloader allows for retrieving various climate datasets from the Copernicus Data Stores by
    specifying the dataset type and required parameters.

    Attributes
    ----------
    logger : logging.Logger
        Logger for the class to handle logging information.

    Methods
    -------
    download_data(dataset_type, params, filename, overwrite=False)
        Downloads data from CDS with user-specified dataset type and parameters.
    """

    def __init__(self):
        """Initialize the downloader with a logger."""
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
        )
        self.logger = logging.getLogger()

    def download_data(self, dataset_type, params, filename, overwrite=False):
        """Download data from Copernicus Data Stores using specified dataset type and parameters.

        Parameters
        ----------
        dataset_type : str
            The dataset type to retrieve (e.g., 'seasonal-original-single-levels', 'sis-heat-and-cold-spells').
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
        try:
            output_dir = Path(filename).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Check if file exists and skip download if overwrite is False
            if Path(filename).exists() and not overwrite:
                self.logger.info(f"File {filename} already exists. Skipping download.")
                return

            # Initialize CDS API client
            c = cdsapi.Client()
            c.retrieve(dataset_type, params, filename)

            # Check if the file was successfully downloaded
            if not Path(filename).exists():
                raise FileNotFoundError(f"Failed to download {filename}.")

            self.logger.info(f"File successfully downloaded to {filename}.")

        except Exception as e:
            self.logger.error(f"Error downloading file {filename}: {e}")
            raise e
