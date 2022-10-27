from pathlib import Path
import multiprocessing as mp
from copy import deepcopy
from typing import Iterable, Mapping, Any, Optional, List
from itertools import repeat
from datetime import date, timedelta

import cdsapi

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
        "system_version": "operational",
        "hydrological_model": "lisflood",
        "format": "netcdf",
        "year": "2022",
        "month": "08",
        "day": "01",
        "leadtime_hour": [
            "24",
            "48",
            "72",
            "96",
            "120",
            "144",
            "168",
            "192",
            "216",
            "240",
        ],
    },
}


def glofas_request_single(product, request, outfile):
    client_kw_default = dict(quiet=True, debug=False)
    # client_kw_default.update(client_kwargs)
    client = cdsapi.Client(**client_kw_default)
    client.retrieve(product, request, outfile)


def glofas_request_multiple(
    product: str,
    requests: Iterable[Mapping[str, str]],
    outfiles: Iterable[Path],
    num_proc: int,
):
    with mp.Pool(num_proc) as pool:
        pool.starmap(glofas_request_single, zip(repeat(product), requests, outfiles))


def glofas_request(
    product: str,
    date_from: str,
    date_to: Optional[str],
    output_dir: Path,
    num_proc: int = 1,
    request_kw: Optional[Mapping[str, str]] = None,
    client_kw: Optional[Mapping[str, Any]] = None,
) -> List[Path]:
    """Request download of GloFAS data products from the Copernicus Data Store (CDS)

    Uses the Copernicus Data Store API (cdsapi) Python module. The interpretation of the
    ``date`` parameters and the grouping of the downloaded data depends on the type of
    ``product`` requested.

    Prerequisites
    -------------
    Downloading data from the CDS requires authentication via a user key which is granted
    to each user upon registration.

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
    request_kw : dict(str: str)
        Dictionary to update the default request for the given product

    Returns
    -------
    list of Path
        Paths of the downloaded files

    Available products
    ------------------
    - ``historical``: Historical reanalysis discharge data. ``date_from`` and ``date_to``
      are interpreted as integer years. Data for each year is placed into a single file.
    - ``forecast``: Forecast discharge data. ``date_from`` and ``date_to`` are
      interpreted as ISO date format strings. Data for each day is placed into a single
      file.
    """
    # Check if product exists
    try:
        default_request = deepcopy(DEFAULT_REQUESTS[product])
    except KeyError:
        raise NotImplementedError(
            f"product = {product}. Choose from {list(DEFAULT_REQUESTS.keys())}"
        )

    # Update with request_kw
    if request_kw is not None:
        default_request.update(**request_kw)

    if product == "historical":
        # Interpret dates as years only
        year_from = int(date_from)
        year_to = int(date_to) if date_to is not None else year_from

        # Create request dict
        def get_request(year: int):
            request = deepcopy(default_request)
            request.update(hyear=str(year))
            return request

        # List up all requests
        glofas_product = f"cems-glofas-{product}"
        years = [year for year in range(year_from, year_to + 1)]
        requests = [get_request(year) for year in years]
        outfiles = [
            Path(output_dir, f"glofas-{product}-{year:04}.grib") for year in years
        ]

    elif product == "forecast":
        # Download single date if 'date_to' is 'None'
        date_from = date.fromisoformat(date_from)
        date_to = date.fromisoformat(date_to) if date_to is not None else date_from

        # Switch file extension based on selected format
        file_ext = "grib" if default_request["format"] == "grib" else "nc"

        # Range iterator over dates
        def date_range(begin: date, end: date):
            for day in range((end - begin).days):
                yield begin + timedelta(days=day)

        # Create request dict
        def get_request(date: date):
            request = deepcopy(default_request)
            request.update(
                year=str(date.year), month=f"{date.month:02d}", day=f"{date.day:02d}"
            )
            return request

        # List up all requests
        glofas_product = f"cems-glofas-{product}"
        dates = [d for d in date_range(date_from, date_to + timedelta(days=1))]
        requests = [get_request(d) for d in dates]
        product_id = default_request["product_type"].split("_")[0]
        outfiles = [
            Path(
                output_dir, f"glofas-{product}-{product_id}-{d.isoformat()}.{file_ext}"
            )
            for d in dates
        ]

    else:
        NotImplementedError()

    # Execute request
    glofas_request_multiple(glofas_product, requests, outfiles, num_proc)

    # Return the (absolute) filepaths
    return [file.resolve() for file in outfiles]
