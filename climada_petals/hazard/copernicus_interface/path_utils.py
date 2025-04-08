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
Path utility functions for seasonal forecast pipelines in CLIMADA.
Provides helpers to generate and check file paths used across the seasonal
forecast download, processing, and hazard generation modules.
"""

from pathlib import Path
from typing import List, Union
from climada_petals.hazard.copernicus_interface.time_utils import month_name_to_number

##########  Path Functions  ##########

def get_file_path(
    base_dir: Union[str, Path],
    originating_centre: str,
    year: Union[int, str],
    initiation_month_str: str,
    valid_period_str: str,
    data_type: str,
    index_metric: str,
    bounds_str: str,
    system: str,
    data_format: str = "grib",
) -> Union[Path, dict]:
    """
    Construct the file path or path dictionary for a given forecast dataset.

    Based on forecast metadata (provider, date, system, index, format, etc.), this
    function builds the expected path or dictionary of paths (for indices) that follow
    CLIMADA's seasonal forecast directory structure.

    Parameters
    ----------
    base_dir : str or Path
        Base directory where Copernicus seasonal data is stored.
    originating_centre : str
        Data provider (e.g., 'dwd').
    year : int or str
        Forecast initiation year.
    initiation_month_str : str
        Forecast initiation month in two-digit string format (e.g., '03').
    valid_period_str : str
        Valid period formatted as '<start>_<end>' (e.g., '06_08').
    data_type : str
        Type of data, one of: 'downloaded_data', 'processed_data', 'indices', 'hazard'.
    index_metric : str
        Name of the climate index (e.g., 'HW', 'TR', 'Tmax').
    bounds_str : str
        Bounding box string (e.g., 'W4_S44_E11_N48').
    system : str
        Forecast system (e.g., '21').
    data_format : str, optional
        File format: 'grib', 'netcdf', or 'hdf5' (autodetected by data_type if not provided).

    Returns
    -------
    pathlib.Path or dict
        A single file path (for non-index types) or a dictionary of paths (for 'indices').

    Raises
    ------
    ValueError
        If an unknown data_type is provided.

    Notes
    -----
    - The returned path follows the CLIMADA forecast folder structure.
    - Index files return a dict with 'daily', 'monthly', and 'stats' keys.
    """
    if data_type == "downloaded_data":
        data_type += f"/{data_format}"
    elif data_type == "hazard":
        data_type += f"/{index_metric}"
        data_format = "hdf5"
    elif data_type == "indices":
        data_type += f"/{index_metric}"
        data_format = "nc"
    elif data_type == "processed_data":
        data_format = "nc"
    else:
        raise ValueError(
            f"Unknown data type {data_type}. Must be in "
            "['downloaded_data', 'processed_data', 'indices', 'hazard']"
        )

    sub_dir = (
        f"{base_dir}/{originating_centre}/sys{system}/{year}/"
        f"init{initiation_month_str}/valid{valid_period_str}/{data_type}"
    )

    if data_type.startswith("indices"):
        return {
            timeframe: Path(
                f"{sub_dir}/{index_metric}_{bounds_str}_{timeframe}.{data_format}"
            )
            for timeframe in ["daily", "monthly", "stats"]
        }

    return Path(f"{sub_dir}/{index_metric}_{bounds_str}.{data_format}")


def check_existing_files(
    base_dir: Union[str, Path],
    originating_centre: str,
    index_metric: str,
    year: int,
    initiation_month: str,
    valid_period: List[str],
    bounds_str: str,
    system: str,
    download_format: str = "grib",
    print_flag: bool = False,
) -> str:
    """
    Inspect the existence of forecast data files for a given configuration.

    A manual debugging utility, this function checks whether the expected
    files (downloaded, processed, index, hazard) exist in the configured directory tree.

    Parameters
    ----------
    base_dir : str or Path
        Base directory where Copernicus seasonal data is stored.
    originating_centre : str
        Forecast data provider (e.g., 'dwd').
    index_metric : str
        Climate index to check (e.g., 'HW', 'TR', 'Tmax').
    year : int
        Forecast initiation year.
    initiation_month : str
        Initiation month as string (e.g., 'March').
    valid_period : list of str
        Valid forecast months, exactly two (e.g., ['June', 'August']).
    bounds_str : str
        Spatial bounds string used in filenames.
    system : str
        Forecast system version (e.g., '21').
    download_format : str, optional
        Format of the downloaded data. Default is 'grib'.
    print_flag : bool, optional
        Whether to print the existence check report.

    Returns
    -------
    str
        Summary report indicating which files exist.

    Raises
    ------
    ValueError
        If valid_period is not exactly two months long.

    Notes
    -----
    - This is a utility function for developers and users to validate pipeline outputs.
    - It is not called by the main forecast processing pipeline.
    """

    if len(valid_period) != 2:
        raise ValueError("valid_period must contain exactly two months.")

    initiation_month_str = f"{month_name_to_number(initiation_month):02d}"
    valid_period_str = "_".join(
        [f"{month_name_to_number(month):02d}" for month in valid_period]
    )

    (
        downloaded_data_path,
        processed_data_path,
        index_data_paths,
        hazard_data_path,
    ) = [
        get_file_path(
            base_dir,
            originating_centre,
            year,
            initiation_month_str,
            valid_period_str,
            data_type,
            index_metric,
            bounds_str,
            system,
            data_format=download_format,
        )
        for data_type in ["downloaded_data", "processed_data", "indices", "hazard"]
    ]

    response = ""
    if not downloaded_data_path.exists():
        response += "No downloaded data found for given time periods.\n"
    else:
        response += f"Downloaded data exist at: {downloaded_data_path}\n"

    if not processed_data_path.exists():
        response += "No processed data found for given time periods.\n"
    else:
        response += f"Processed data exist at: {processed_data_path}\n"

    if not any(path.exists() for path in index_data_paths.values()):
        response += "No index data found for given time periods.\n"
    else:
        response += f"Index data exist at: {index_data_paths}\n"

    if not hazard_data_path.exists():
        response += "No hazard data found for given time periods."
    else:
        response += f"Hazard data exist at: {hazard_data_path}"

    if print_flag:
        print(response)

    return response