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
from itertools import repeat
from datetime import date, datetime
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
        "variable": "river_discharge_in_the_last_24_hours",
        "product_type": "consolidated",
        "system_version": "version_3_1",
        "hydrological_model": "lisflood",
        "format": "grib",
        "hyear": "1979",
        "hmonth": [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        "hday": [f"{day:02}" for day in range(1, 32)],
    },
    "forecast": {
        "variable": "river_discharge_in_the_last_24_hours",
        "product_type": "ensemble_perturbed_forecasts",
        "system_version": "version_3_1",
        "hydrological_model": "lisflood",
        "format": "grib",
        "year": "2022",
        "month": "08",
        "day": "01",
        "leadtime_hour": (np.arange(1, 31) * 24).astype(str).tolist(),
    },
}
"""Default request keyword arguments to be updated by the user requests"""


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
    request: Mapping[str, Any],
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
    extension = ".grib" if request["format"] == "grib" else ".nc"
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
    client_kw_default = dict(quiet=False, debug=False)
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
    requests: Iterable[Mapping[str, str]],
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
                repeat(product),
                requests,
                repeat(outdir),
                repeat(use_cache),
                repeat(client_kw),
            ),
        )


def glofas_request(
    product: str,
    date_from: str,
    date_to: Optional[str],
    output_dir: Union[Path, str],
    num_proc: int = 1,
    use_cache: bool = True,
    request_kw: Optional[Mapping[str, str]] = None,
    client_kw: Optional[Mapping[str, Any]] = None,
) -> List[Path]:
    """Request download of GloFAS data products from the Copernicus Data Store (CDS)

    Uses the Copernicus Data Store API (cdsapi) Python module. The interpretation of the
    ``date`` parameters and the grouping of the downloaded data depends on the type of
    ``product`` requested.

    Available ``products``:

    - ``historical``: Historical reanalysis discharge data. ``date_from`` and ``date_to``
      are interpreted as integer years. Data for each year is placed into a single file.
    - ``forecast``: Forecast discharge data. ``date_from`` and ``date_to`` are
      interpreted as ISO date format strings. Data for each day is placed into a single
      file.

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
        The indentifier for the CMS product to download. See below for available options.
    date_from : str
        First date to download data for. Interpretation varies based on ``product``.
    date_to : str or None
        Last date to download data for. Interpretation varies based on ``product``. If
        ``None``, or the same date as ``date_from``, only download data for ``date_from``
    output_dir : Path
        Output directory for the downloaded data
    num_proc : int
        Number of processes used for parallel requests
    use_cache : bool (optional)
        Skip downloading if the target file exists and the accompanying request file
        contains the same request
    request_kw : dict(str: str)
        Dictionary to update the default request for the given product
    client_kw : dict (optional)
        Dictionary with keyword arguments for the ``cdsapi.Client`` used for downloading

    Returns
    -------
    list of Path
        Paths of the downloaded files
    """
    # Check if product exists
    try:
        default_request = deepcopy(DEFAULT_REQUESTS[product])
    except KeyError as err:
        raise NotImplementedError(
            f"product = {product}. Choose from {list(DEFAULT_REQUESTS.keys())}"
        ) from err

    # Update with request_kw
    if request_kw is not None:
        default_request.update(**request_kw)

    if product == "historical":
        # Interpret dates as years only
        year_from = int(date_from)
        year_to = int(date_to) if date_to is not None else year_from

        # List up all requests
        requests = [
            {"hyear": str(year)} for year in list(range(year_from, year_to + 1))
        ]

    elif product == "forecast":
        # Download single date if 'date_to' is 'None'
        date_from: date = date.fromisoformat(date_from)
        date_to: date = (
            date.fromisoformat(date_to) if date_to is not None else date_from
        )

        # List up all requests
        dates = pd.date_range(date_from, date_to, freq="D", inclusive="both").date
        requests = [
            {"year": str(d.year), "month": f"{d.month:02d}", "day": f"{d.day:02d}"}
            for d in dates
        ]

    else:
        NotImplementedError("Unknown product: %s" % product)

    requests = [{**default_request, **req} for req in requests]
    glofas_product = f"cems-glofas-{product}"

    # Execute request
    return glofas_request_multiple(
        glofas_product, requests, output_dir, num_proc, use_cache, client_kw
    )
