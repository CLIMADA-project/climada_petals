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

Functions for downloading GloFAS river discharge data from the Copernicus Climate Data
Store (CDS).
"""

from pathlib import Path
import multiprocessing as mp
from copy import deepcopy
from typing import Iterable, Mapping, Any, Optional, List, Union
import itertools as it
from datetime import datetime
import logging
import hashlib

from cdsapi import Client
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO
import pandas as pd
import numpy as np

from climada.util.constants import SYSTEM_DIR

LOGGER = logging.getLogger(__name__)

CDS_DOWNLOAD_DIR = Path(SYSTEM_DIR, "cds-download")

DEFAULT_REQUESTS = {
    "historical": {
        "variable": ["river_discharge_in_the_last_24_hours"],
        "product_type": ["consolidated"],
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "data_format": "grib2",
        "download_format": "unarchived",
        "hyear": ["1979"],
        "hmonth": [f"{month:02}" for month in range(1, 13)],
        "hday": [f"{day:02}" for day in range(1, 32)],
    },
    "forecast": {
        "variable": ["river_discharge_in_the_last_24_hours"],
        "product_type": ["ensemble_perturbed_forecasts"],
        "system_version": ["operational"],
        "hydrological_model": ["lisflood"],
        "data_format": "grib2",
        "download_format": "unarchived",
        "year": ["2022"],
        "month": ["08"],
        "day": ["01"],
        "leadtime_hour": (np.arange(1, 31) * 24).astype(str).tolist(),
    },
}
"""Default request keyword arguments to be updated by the user requests"""

CLIENT_KW_DEFAULT = {"quiet": False, "debug": False, "timeout": 240, "sleep_max": 480}
"""Default keyword argument for the API client"""


def datetime_index_to_request(
    index: pd.DatetimeIndex, product: str
) -> dict[str, list[str]]:
    """Create a request-compatible dict from a series"""
    prefix = "h" if product == "historical" else ""
    return {
        prefix + "year": list(map(str, index.year)),
        prefix + "month": list(map(lambda x: f"{x:02d}", index.month)),
        prefix + "day": list(map(lambda x: f"{x:02d}", index.day)),
    }


def request_to_md5(request: Mapping[Any, Any]) -> str:
    """Hash a string with the MD5 algorithm"""
    yaml = YAML()
    stream = StringIO()
    yaml.dump(request, stream)
    return hashlib.md5(stream.getvalue().encode("utf-8")).hexdigest()


def cleanup_download_dir(
    download_dir: Union[Path, str] = CDS_DOWNLOAD_DIR, dry_run: bool = False
):
    """Delete the contents of the download directory"""
    for filename in Path(download_dir).glob("*"):
        LOGGER.debug("Removing file: %s", filename)
        if not dry_run:
            filename.unlink()
    if dry_run:
        LOGGER.debug("Dry run. No files removed")


def glofas_request_single(
    product: str,
    request: Mapping[str, str | list[str]],
    outpath: Union[Path, str],
    use_cache: bool = True,
    client_kw: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Perform a single request for data from the Copernicus data store

    This will skip the download if a file was found at the target location with the same
    request. The request will be stored as YAML file alongside the target file and used
    for comparison. This behavior can be adjusted with the ``use_cache`` parameter.

    Parameters
    ----------
    product : str
        The string identifier of the product in the Copernicus data store
    request : dict
        The download request as dictionary
    outpath : str or Path
        The file path to store the download into (including extension)
    use_cache : bool (optional)
        Skip downloading if the target file exists and the accompanying request file
        contains the same request
    client_kw : dict (optional)
        Dictionary with keyword arguments for the ``cdsapi.Client`` used for downloading
    """
    # Define output file
    outpath = Path(outpath)
    request_hash = request_to_md5(request)
    outfile = outpath / (
        datetime.today().strftime("%y%m%d-%H%M%S") + f"-{request_hash}"
    )
    extension = ".grib" if request["data_format"] == "grib2" else ".nc"
    outfile = outfile.with_suffix(extension)

    # Check if request was issued before
    if use_cache:
        for filename in outpath.glob(f"*{extension}"):
            if request_hash == filename.stem.split("-")[-1]:
                LOGGER.info(
                    "Skipping request for file '%s' because it already exists", outfile
                )
                return filename.resolve()

    # Set up client and retrieve data
    LOGGER.info("Downloading file: %s", outfile)
    client_kw_default = deepcopy(CLIENT_KW_DEFAULT)
    if client_kw is not None:
        client_kw_default.update(client_kw)
    client = Client(**client_kw_default)
    client.retrieve(product, request, outfile)

    # Dump request
    yaml = YAML()
    yaml.dump(request, outfile.with_suffix(".yml"))

    # Return file path
    return outfile.resolve()


def glofas_request_multiple(
    product: str,
    requests: Iterable[Mapping[str, str | list[str]]],
    outdir: Union[Path, str],
    num_proc: int,
    use_cache: bool,
    client_kw: Optional[Mapping[str, Any]] = None,
) -> List[Path]:
    """Execute multiple requests to the Copernicus data store in parallel"""
    with mp.Pool(num_proc) as pool:
        return pool.starmap(
            glofas_request_single,
            zip(
                it.repeat(product),
                requests,
                it.repeat(outdir),
                it.repeat(use_cache),
                it.repeat(client_kw),
            ),
        )


def glofas_request(
    product: str,
    output_dir: Union[Path, str],
    *,
    num_proc: int = 1,
    use_cache: bool = True,
    client_kw: Optional[Mapping[str, Any]] = None,
    request_kw: Optional[Mapping[str, str | list[str]]] = None,
    requests: Optional[List[Mapping[str, str | list[str]]]] = None,
) -> List[Path]:
    """Request download of GloFAS data products from the Copernicus Data Store (CDS)

    Uses the Copernicus Data Store API (cdsapi) Python module.

    Notes
    -----
    Downloading data from the CDS requires authentication via a user key which is granted
    to each user upon registration. Do the following **before calling this function**:

    - Create an account at the Copernicus Data Store website:
      https://cds.climate.copernicus.eu/
    - Follow the instructions to install the CDS API key:
      https://cds.climate.copernicus.eu/api-how-to#install-the-cds-api-key

    Parameters
    ----------
    product : str
        The indentifier for the CMS product to download.

        - ``historical``: Historical reanalysis discharge data.
        - ``forecast``: Ensemble forecast discharge data.
    output_dir : Path
        Output directory for the downloaded data
    num_proc : int
        Number of processes used for parallel requests
    use_cache : bool (optional)
        Skip downloading if the target file exists and the accompanying request file
        contains the same request
    client_kw : dict (optional)
        Dictionary with keyword arguments for the ``cdsapi.Client`` used for downloading
    request_kw : dict(str: str), optional
        Dictionary to update the default request for the given product. If ``None``, the
        default request is issued.
    requests : list, optional
        A list of dictionaries for multiple requests. These will be used to update the
        default request after ``request_kw`` was applied. If ``None``, only one request
        will be issued.

    Returns
    -------
    list of Path
        Paths of the downloaded files

    See Also
    --------
    :py:const:`~climada_petals.hazard.rf_glofas.cds_glofas_downloader.DEFAULT_REQUESTS`
    """
    # Check if product exists
    glofas_product = f"cems-glofas-{product}"
    try:
        default_request = deepcopy(DEFAULT_REQUESTS[product])
    except KeyError as err:
        raise NotImplementedError(
            f"product = {product}. Choose from {list(DEFAULT_REQUESTS.keys())}"
        ) from err

    # Update with request_kw
    if request_kw is not None:
        default_request.update(**request_kw)

    def sanitize_request_lists(request):
        """Turn each item into a list if the default request item is a list"""
        default = deepcopy(DEFAULT_REQUESTS[product])
        request_sane = deepcopy(request)
        for key, default_value in default.items():
            if isinstance(default_value, list) and not isinstance(request[key], list):
                request_sane[key] = [request[key]]
        return request_sane

    # Single request
    if requests is None:
        return [
            glofas_request_single(
                glofas_product,
                sanitize_request_lists(default_request),
                output_dir,
                use_cache,
                client_kw,
            )
        ]

    # Request list
    requests = [
        sanitize_request_lists(deepcopy(default_request) | dict(req))
        for req in requests
    ]

    # Single request
    if len(requests) == 1:
        return [
            glofas_request_single(
                glofas_product, requests[0], output_dir, use_cache, client_kw
            )
        ]

    # Execute request
    return glofas_request_multiple(
        glofas_product, requests, output_dir, num_proc, use_cache, client_kw
    )
