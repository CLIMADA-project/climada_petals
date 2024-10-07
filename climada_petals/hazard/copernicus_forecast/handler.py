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

Module to handle seasonal forecast data from the Copernicus Climate Data Store (CDS)
in the U-CLIMADAPT project.

Prerequisites:
1. CDS API client installation:
   pip install cdsapi

2. CDS account and API key:
   Register at https://cds-beta.climate.copernicus.eu

3. CDS API configuration:
   Create a .cdsapirc file in your home directory with your API key and URL.
   For instructions, visit:
   https://cds-beta.climate.copernicus.eu/how-to-api#install-the-cds-api-client

4. Dataset Terms and Conditions: After selecting the dataset to download, make 
   sure to accept the terms and conditions on the corresponding dataset webpage 
   in the CDS portal before running the script.
"""

import logging
import calendar
import re
from pathlib import Path

import xarray as xr
import pandas as pd
import numpy as np
import cdsapi

from climada.hazard import Hazard
from climada.util.constants import SYSTEM_DIR
from climada.util.coordinates import get_country_geometries
import climada_petals.hazard.copernicus_forecast.indicator as indicator


# set path to store data
DATA_OUT = SYSTEM_DIR / "copernicus_forecast"

LOGGER = logging.getLogger(__name__)


class ForecastHandler:
    """A handler for downloading, processing, and calculating climate indices based on seasonal forecast data.

    This class is designed to handle various operations related to seasonal climate forecasts,
    including data retrieval from the Copernicus Climate Data Store (CDS), processing the
    downloaded data, calculating specific climate indices, and converting the results into
    hazard objects for further risk analysis.

    The `ForecastHandler` offers multiple methods for handling CDS API connections,
    calculating indices such as heat stress or temperature thresholds, and saving these
    indices in formats compatible with the CLIMADA hazard framework.

    Attributes
    ----------
    _FORMAT_GRIB : str
        Constant for the GRIB file format ("grib").
    _FORMAT_NC : str
        Constant for the NetCDF file format ("nc").
    data_dir : pathlib.Path
        Directory path where downloaded and processed data will be stored.
    key : str, optional
        CDS API key for accessing climate data. This should be configured through the `~/.cdsapirc` file.
    url : str, optional
        URL for the CDS API. If not provided, the default from `~/.cdsapirc` is used.

    Methods
    -------
    __init__(data_dir=DATA_OUT, url=None, key=None)
        Initializes the ForecastHandler instance, setting up the data directory and CDS API configurations.

    _get_bounds_for_area_selection(area_selection, margin=0.2)
        Determines the geographic bounds for a given area selection and adds a margin if specified.

    explain_index(index_metric)
        Provides an explanation and required input data for a specified climate index.

    _calc_min_max_lead(year, month, leadtime_months=1)
        Calculates the minimum and maximum lead time in hours for a given forecast start date.

    _download_multvar_multlead(filename, vars, year, month, l_hours, area, overwrite, format, originating_centre, system)
        Downloads multiple climate variables over multiple lead times from the Copernicus Climate Data Store (CDS).

    _download_data(data_out, year_list, month_list, bounds, overwrite, index_metric, format, originating_centre, system, max_lead_month)
        Handles the downloading of seasonal climate forecast data for specific years, months, and climate indices.

    _process_data(data_out, year_list, month_list, bounds, overwrite, index_metric, format)
        Processes downloaded forecast data into daily averages and saves results in NetCDF format.

    download_and_process_data(index_metric, year_list, month_list, area_selection, overwrite, format, originating_centre, system, max_lead_month, data_out=None)
        Downloads and processes climate forecast data for given years, months, and a specified climate index.

    calculate_index(index_metric, year_list, month_list, area_selection, overwrite, data_out=None)
        Calculates a specified climate index for given years, months, and geographical areas.

    save_index_to_hazard(index_metric, year_list, month_list, area_selection, overwrite, data_out=None)
        Converts the calculated climate indices into hazard objects compatible with the CLIMADA framework.

    _is_data_present(file, vars)
        Checks if the specified data file already exists in the given directory and meets the variable requirements.

    Example Usage
    -------------
    >>> handler = ForecastHandler()
    >>> handler.download_and_process_data("Tmean", [2021, 2022], [6, 7], "global", True, "netcdf", "ecmwf", "21", 6)
    >>> handler.calculate_index("TX30", [2022], [8], ["DEU", "CHE"], False)
    >>> handler.save_index_to_hazard("TX30", [2022], [8], ["DEU", "CHE"], True)

    Notes
    -----
    This class requires the `cdsapi` and `xarray` libraries for interacting with the CDS API and processing the data,
    respectively. Additionally, ensure that your CDS API key and URL are correctly set up in the `~/.cdsapirc` file.
    """

    _FORMAT_GRIB = "grib"
    _FORMAT_NC = "nc"

    def __init__(self, data_out=None):
        """Initialize the ForecastHandler instance.

        This method sets up logging and initializes the directory for storing
        downloaded and processed data.

        Parameters
        ----------
        data_out : pathlib.Path or str, optional
            Path to the directory where downloaded and processed data will be stored.
            If not provided, defaults to the global constant `DATA_OUT`.

        Notes
        -----
        The Copernicus Climate Data Store (CDS) API credentials (`url` and `key`) should be configured in the `~/.cdsapirc` file.
        These are not required as input parameters for this script.
        Ensure that your `.cdsapirc` file contains valid API credentials for successful data downloads.
        """
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
        )
        self.logger = logging.getLogger()
        self.data_out = Path(data_out) if data_out else DATA_OUT

    @staticmethod
    def _get_bounds_for_area_selection(area_selection, margin=0.2):
        """Determine the geographic bounds based on an area selection string.

        This function computes the geographic bounding box for the specified area selection.
        It supports multiple formats, including global selection, bounding box, or country ISO codes.
        The function also adds an optional margin to the bounding box to ensure a broader area is selected.

        Parameters
        ----------
        area_selection : str or list
            Specifies the area for data selection. This parameter can be:
            - "global": To select the entire globe.
            - A list of four floats representing [north, west, south, east] bounds.
            - A list of ISO alpha-3 country codes, e.g., ["DEU", "CHE"] for Germany and Switzerland.
        margin : float, optional
            Additional margin to be added to the bounds in degrees. This is applied to both
            latitude and longitude values. Default is 0.2.

        Returns
        -------
        list
            A list of four floats representing the calculated geographic bounds [north, west, south, east].
            If the area selection is invalid or unrecognized, it returns None.

        Raises
        ------
        ValueError
            Raised if area_selection contains unrecognized ISO codes.
        """
        # Handle "global" area selection
        if isinstance(area_selection, str):
            if area_selection.lower() == "global":
                return [90, -180, -90, 180]  # Global bounds: [north, west, south, east]
            else:
                raise ValueError(
                    f"Invalid string for area_selection: '{area_selection}'. Expected 'global' or a list of ISO codes."
                )  

        # Handle bounding box selection
        elif isinstance(area_selection, list):
            # Check if the list is a bounding box of four values
            if len(area_selection) == 4:
                try:
                    north, west, south, east = area_selection
                    lat_margin = margin * (north - south)
                    lon_margin = margin * (east - west)
                    north += lat_margin
                    east += lon_margin
                    south -= lat_margin
                    west -= lon_margin
                    return [north, west, south, east]
                except ValueError:
                    self.logger.error(
                        f"Invalid area selection bounds provided: {area_selection}. "
                        "Expected a list of four numerical values [north, west, south, east]."
                    )  
                    raise

            # Handle list of country ISO codes
            combined_bounds = [-90, 180, 90, -180]
            for iso in area_selection:
                geo = get_country_geometries(iso).to_crs(epsg=4326)
                bounds = geo.total_bounds
                if np.any(np.isnan(bounds)):
                    self.logger.warning(
                        f"ISO code '{iso}' not recognized. This region will not be included."
                    )  
                    continue

                min_lon, min_lat, max_lon, max_lat = bounds

                lat_margin = margin * (max_lat - min_lat)
                lon_margin = margin * (max_lon - min_lon)

                combined_bounds[0] = max(combined_bounds[0], max_lat + lat_margin)
                combined_bounds[1] = min(combined_bounds[1], min_lon - lon_margin)
                combined_bounds[2] = min(combined_bounds[2], min_lat - lat_margin)
                combined_bounds[3] = max(combined_bounds[3], max_lon + lon_margin)

            if combined_bounds == [-90, 180, 90, -180]:
                return None
            else:
                return combined_bounds

        else:
            raise ValueError(
                f"Invalid area_selection format: {area_selection}. "
                "Expected 'global', a list of ISO codes, or [north, west, south, east]."
            )

    def explain_index(self, index_metric):
        """Provide an explanation and required input data for the specified climate index.

        Parameters
        ----------
        index_metric : str
            The climate index identifier to be explained (e.g., 'TX30', 'Tmean').

        Raises
        ------
        TypeError
            Raised if `index_metric` is not provided as a string type.
        ValueError
            Raised if the specified `index_metric` is not recognized or does not exist.

        Returns
        -------
        dict
            A dictionary containing 'explanation' and 'input_data' if the index is found.
            Returns None if the index is not found or invalid.
        """
        if not isinstance(index_metric, str):
            raise TypeError(
                f"The function expects a string parameter, but received '{type(index_metric).__name__}'.\n"
                f"Did you mean to use quotation marks? For example, use 'TX30' instead of {index_metric}."
            )

        explanation = indicator.index_explanations(index_metric)
        if "error" not in explanation:
            print(
                f"Explanation for '{index_metric}': {explanation['explanation']}\nRequired Input Data: {explanation['input_data']}"
            )
        else:
            # Display an informative error message including valid indices
            valid_indices = ", ".join(explanation["valid_indices"])
            raise ValueError(
                f"Unknown index '{index_metric}'. Please use a valid index from the following list: {valid_indices}."
            )

    def _calc_min_max_lead(self, year, month, leadtime_months=1):
        """Calculate the minimum and maximum lead time in hours for a given start date.

        Parameters
        ----------
        year : int
            The starting year (e.g., 2023) for the forecast initialization.
        month : int
            The starting month (1-12) for the forecast initialization.
        leadtime_months : int, optional
            Number of months to include in the forecast period, by default 1.

        Returns
        -------
        tuple
            A tuple containing the minimum lead time (min_lead) and maximum lead time (max_lead) in hours.
        """
        total_timesteps = 0
        for m in range(month, month + leadtime_months):
            adjusted_year, adjusted_month = year, m
            if m > 12:
                adjusted_year += 1
                adjusted_month = m - 12

            num_days_month = calendar.monthrange(adjusted_year, adjusted_month)[1]
            timesteps = num_days_month * 24
            total_timesteps += timesteps

        max_lead = total_timesteps + 6
        return 0, max_lead

    def _download_multvar_multlead(
        self,
        filename,
        vars,
        year,
        month,
        l_hours,
        area,
        overwrite,
        format,
        originating_centre,
        system,
    ):
        """Download multiple climate variables over multiple lead times from the CDS.

        This method handles downloading multiple climate variables for a given year and month
        over various forecast lead times. The file is saved to the specified path, and the
        CDS API client is used to retrieve data based on the provided parameters.

        Parameters
        ----------
        filename : pathlib.Path or str
            Full path and filename where the downloaded data will be stored.
        vars : list[str]
            List of climate variables to download (e.g., temperature, precipitation).
        year : int
            The forecast initialization year (e.g., 2022).
        month : int
            The forecast initialization month (values 1-12).
        l_hours : list[int]
            List of lead times (in hours) to download. Each value represents a forecast hour
            relative to the initialization time (e.g., [0, 24, 48] for daily steps).
        area : list[float]
            Geographic bounds for data extraction specified as [north, west, south, east].
        overwrite : bool
            If True, overwrites the file if it already exists. If False, skips downloading
            if the file is already present.
        format : str
            File format for downloading data, either 'grib' or 'netcdf'. The choice impacts
            the processing speed and compatibility. GRIB files are often larger and more
            complex than NetCDF.
        originating_centre : str
            The name of the meteorological center providing the forecast (e.g., 'ecmwf', 'dwd').
        system : str
            The seasonal forecast system version (e.g., '21' for DWD).

        Raises
        ------
        FileNotFoundError
            Raised if the download attempt fails and the file is not found at the specified location.
        Exception
            Raised for any other error during the download process.

        Notes
        -----
        The CDS API client (`cdsapi.Client()`) uses the default configuration stored in the `.cdsapirc` file.
        Make sure the configuration file is correctly set up with your API key and URL before executing
        this function.
        """
        try:
            # Ensure the directory exists before downloading
            output_dir = Path(filename).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Directly instantiate the CDS API client using the default configuration from .cdsapirc
            c = cdsapi.Client()
            c.retrieve(
                "seasonal-original-single-levels",
                {
                    "format": format,
                    "originating_centre": originating_centre,
                    "area": area,
                    "system": system,
                    "variable": vars,
                    "month": f"{month:02d}",
                    "year": year,
                    "day": "01",
                    "leadtime_hour": l_hours,
                },
                filename,
            )

            # Check if the file was actually downloaded
            if not Path(filename).exists():  # Added check to confirm the file exists
                raise FileNotFoundError(
                    f"Failed to download {format} file to {filename}."
                )
            self.logger.info(
                f"{format.capitalize()} file successfully downloaded to {filename}."
            )

        except Exception as e:
            self.logger.error(
                f"{format.capitalize()} file {filename} could not be downloaded. Error: {e}"
            )
            raise e  # Re-raise the exception for higher-level handling

    def _download_data(
        self,
        data_out,
        year_list,
        month_list,
        bounds,
        overwrite,
        index_metric,
        format,
        originating_centre,
        system,
        max_lead_month,
    ):
        """Download climate forecast data for specified years, months, and a climate index.

        Parameters
        ----------
        data_out : pathlib.Path or str
            Base directory path for storing downloaded data.
        year_list : list[int]
            List of years for which to download data.
        month_list : list[int]
            List of months for which to download data.
        bounds : list[float]
            Geographical area bounds for data selection in the format [north, west, south, east].
        overwrite : bool
            If True, overwrites existing files.
        index_metric : str
            Climate index identifier for the requested data.
        format : str
            File format for download, either 'grib' or 'netcdf'.
        originating_centre : str
            The meteorological center producing the forecast (e.g., "dwd", "ecmwf").
        system : str
            The forecast system version (e.g., "21").
        max_lead_month : int
            Maximum lead time in months to download.

        Returns
        -------
        None
        """
        data_out = self.data_out
        index_params = indicator.get_index_params(index_metric)
        variables = index_params["variables"]
        vars_short = [indicator.VAR_SPECS[var]["short_name"] for var in variables]
        area_str = (
            f"area{int(bounds[1])}_{int(bounds[0])}_{int(bounds[2])}_{int(bounds[3])}"
        )

        for year in year_list:
            for month in month_list:
                # Prepare output paths
                out_dir = Path(f"{data_out}/input_data/{format}/{year}/{month:02d}")
                out_dir.mkdir(parents=True, exist_ok=True)

                # Construct the correct download file path
                file_extension = (
                    "grib" if format == self._FORMAT_GRIB else self._FORMAT_NC
                )
                download_file = (
                    out_dir
                    / f'{"_".join(vars_short)}_{area_str}_{year}{month:02d}.{file_extension}'
                )

                # Check if data already exists
                existing_file = self._is_data_present(download_file, variables)

                # Decide whether to download based on `overwrite` flag
                if existing_file and not overwrite:
                    self.logger.info(f"File {existing_file} already exists.")
                    continue  # Skip downloading
                else:
                    # Compute lead times
                    min_lead, max_lead = self._calc_min_max_lead(
                        year, month, max_lead_month
                    )
                    leadtimes = list(range(min_lead, max_lead, 6))
                    self.logger.info(f"{len(leadtimes)} leadtimes to download.")
                    self.logger.debug(f"Lead times are: {leadtimes}")

                    # Download data
                    self._download_multvar_multlead(
                        download_file,
                        variables,
                        year,
                        month,
                        leadtimes,
                        bounds,
                        overwrite,
                        format,
                        originating_centre,
                        system,
                    )

    def _process_data(
        self, data_out, year_list, month_list, bounds, overwrite, index_metric, format
    ):
        """Process the downloaded climate forecast data into daily average values.

        Parameters
        ----------
        data_out : pathlib.Path or str
            Base directory path for storing processed output data.
        year_list : list[int]
            List of years for which to process data.
        month_list : list[int]
            List of months for which to process data.
        bounds : list[float]
            Geographical area bounds for processing, specified as [north, west, south, east].
        overwrite : bool
            If True, overwrites existing processed files.
        index_metric : str
            Climate index identifier being processed (e.g., 'Tmean', 'TX30').
        format : str
            File format of the downloaded data, either 'grib' or 'nc'.

        Returns
        -------
        None
        """
        data_out = self.data_out
        index_params = indicator.get_index_params(index_metric)
        variables = index_params["variables"]
        vars_short = [indicator.VAR_SPECS[var]["short_name"] for var in variables]
        area_str = (
            f"area{int(bounds[1])}_{int(bounds[0])}_{int(bounds[2])}_{int(bounds[3])}"
        )

        for year in year_list:
            for month in month_list:

                output_dir = Path(
                    f"{data_out}/input_data/netcdf/daily/{year}/{month:02d}"
                )
                daily_file = (
                    output_dir
                    / f'{"_".join(vars_short)}_{area_str}_{year}{month:02d}.nc'
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                file_extension = (
                    "grib" if format == self._FORMAT_GRIB else self._FORMAT_NC
                )
                input_file = (
                    f"{data_out}/input_data/{format}/{year}/{month:02d}/"
                    f"{index_params['filename_lead']}_{area_str}_{year}{month:02d}.{file_extension}"
                )
                input_file = self._is_data_present(
                    input_file, index_params["variables"]
                )

                if input_file is None:
                    self.logger.error(
                        f"Input file {input_file} not found. Skipping processing for {year}-{month:02d}."
                    )
                    continue

                # Process and save the data
                if not daily_file.exists() or overwrite:
                    try:
                        if format == self._FORMAT_GRIB:
                            with xr.open_dataset(input_file, engine="cfgrib") as ds:
                                ds_mean = ds.coarsen(step=4, boundary="trim").mean()
                                ds_max = ds.coarsen(step=4, boundary="trim").max()
                                ds_min = ds.coarsen(step=4, boundary="trim").min()
                        else:
                            with xr.open_dataset(input_file) as ds:
                                ds_mean = ds.coarsen(step=4, boundary="trim").mean()
                                ds_max = ds.coarsen(step=4, boundary="trim").max()
                                ds_min = ds.coarsen(step=4, boundary="trim").min()

                        # Create a new dataset combining mean, max, and min values
                        combined_ds = xr.Dataset()
                        for var in vars_short:
                            combined_ds[f"{var}_mean"] = ds_mean[var]
                            combined_ds[f"{var}_max"] = ds_max[var]
                            combined_ds[f"{var}_min"] = ds_min[var]

                        # Save combined dataset to NetCDF
                        combined_ds.to_netcdf(
                            str(daily_file)
                        )  # Convert Path to string for compatibility

                    except FileNotFoundError:
                        self.logger.error(
                            f"{format.capitalize()} file does not exist, download failed."
                        )
                        continue
                else:
                    self.logger.info(f"Daily file {daily_file} already exists.")

    def download_and_process_data(
        self,
        index_metric,
        year_list,
        month_list,
        area_selection,
        overwrite,
        format,
        originating_centre,
        system,
        max_lead_month,
        data_out=None,
    ):
        """
        _summary_

        Parameters
        ----------
        index_metric : str
            Climate index identifier to be processed (e.g., 'Tmean', 'TR').
        year_list : list of int
            List of years for which to download and process data.
        month_list : list of int
            List of months for which to download and process data.
        area_selection : str or list
            Area specification for the data retrieval. It can be "global", a list of ISO-3 country codes,
            or a list of coordinates as [north, west, south, east].
        overwrite : bool
            If True, overwrites existing files.
        format : str
            File format for download and processing ('grib' or 'netcdf').
        originating_centre : str
            The meteorological center producing the forecast (e.g., 'ecmwf', 'dwd').
        system : str
            The forecast system version (e.g., '21').
        max_lead_month : int
            Maximum lead time in months for which forecast data should be downloaded.
        data_out : str or Path, optional
            Base directory path for storing data, by default None.

        Returns
        -------
        None
        """
        # if not data_out: data_out = self.data_dir
        self.data_out = Path(data_out) if data_out else self.data_out
        bounds = self._get_bounds_for_area_selection(area_selection)
        self._download_data(
            data_out,
            year_list,
            month_list,
            bounds,
            overwrite,
            index_metric,
            format,
            originating_centre,
            system,
            max_lead_month,
        )
        self._process_data(
            data_out, year_list, month_list, bounds, overwrite, index_metric, format
        )

    def calculate_index(
        self, index_metric, year_list, month_list, area_selection, overwrite, data_out=None
    ):
        """
        _summary_

        Parameters
        ----------
        index_metric : str
            The climate index to be calculated (e.g., 'Tmean', 'TR').
        year_list : list of int
            List of years for which to calculate the index.
        month_list : list of int
            List of months for which to calculate the index (values 1-12).
        area_selection : str or list
            Area specification for the calculation. This can be "global", a list of ISO-3 country codes
            (e.g., ["DEU", "CHE"]), or a list of coordinates specified as [north, west, south, east].
        overwrite : bool
            If True, overwrites existing files. If False, skips calculation if files already exist.
        data_out : str or Path, optional
            Base directory path for output data, by default None.

        Returns
        -------
        None
        """
        # if not data_out:data_out = self.data_dir
        self.data_out = Path(data_out) if data_out else self.data_out
        bounds = self._get_bounds_for_area_selection(area_selection)
        area_str = (
            f"area{int(bounds[1])}_{int(bounds[0])}_{int(bounds[2])}_{int(bounds[3])}"
        )
        index_params = indicator.get_index_params(index_metric)
        vars_short = [
            indicator.VAR_SPECS[var]["short_name"] for var in index_params["variables"]
        ]

        for year in year_list:
            for month in month_list:
                # Path to input file of daily variables
                input_file_name = (
                    self.data_out
                    / "input_data"
                    / "netcdf"
                    / "daily"
                    / str(year)
                    / f"{month:02d}"
                    / f'{"_".join(vars_short)}_{area_str}_{year}{month:02d}.nc'
                )

                grib_file_name = (
                    self.data_out
                    / "input_data"
                    / "grib"
                    / str(year)
                    / f"{month:02d}"
                    / f'{"_".join(vars_short)}_{area_str}_{year}{month:02d}.grib'
                )

                # Check if input data is present
                input_file_name = self._is_data_present(
                    str(input_file_name), index_params["variables"]
                )
                grib_file_name = self._is_data_present(
                    str(grib_file_name), index_params["variables"]
                )

                # Define output paths using Pathlib
                out_dir = (
                    self.data_out / "indices" / index_metric / str(year) / f"{month:02d}"
                )
                out_daily_path = (
                    out_dir / f"daily_{index_metric}_{area_str}_{year}{month:02d}.nc"
                )
                out_stats_path = (
                    out_dir
                    / "stats"
                    / f"stats_{index_metric}_{area_str}_{year}{month:02d}.nc"
                )
                out_monthly_path = (
                    out_dir / f"{index_metric}_{area_str}_{year}{month:02d}.nc"
                )
                out_stats_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if the index (monthly) file exists
                if out_monthly_path.exists() and not overwrite:
                    self.logger.info(f"Index file {out_monthly_path} already exists.")
                    continue

                # Calculate indices
                else:
                    if index_metric in ["HIS", "HIA", "Tmean", "Tmax", "Tmin"]:
                        ds_daily, ds_monthly, ds_stats = (
                            indicator.calculate_heat_indices_metrics(
                                input_file_name, index_metric
                            )
                        )
                    elif index_metric == "TR":
                        ds_daily, ds_monthly, ds_stats = indicator.calculate_TR(
                            grib_file_name, index_metric
                        )
                    elif index_metric == "TX30":
                        ds_daily, ds_monthly, ds_stats = indicator.calculate_tx30(
                            grib_file_name, index_metric
                        )
                    # TODO: add functionality
                    # elif index_metric == "HW":
                    #     indicator.calculate_and_save_heat_wave_days_per_lag(
                    #         data_out, year_list, month_list, index_metric, area_selection
                    #     )

                    else:
                        logging.error(
                            f"Index {index_metric} is not implemented. Supported indices "
                            "are 'HIS', 'HIA', 'Tmean', 'Tmax', 'Tmin', 'HotDays', 'TR', and 'HW'."
                        )

                    # Save files
                    self.logger.info(f"Writing index data to {out_monthly_path}.")
                    if index_metric in ["HIS", "HIA", "Tmean", "Tmax", "Tmin"]:
                        ds_daily.to_netcdf(str(out_daily_path))
                    ds_monthly.to_netcdf(str(out_monthly_path))
                    ds_stats.to_netcdf(str(out_stats_path))

                    # Confirm data saving
                    if out_monthly_path.exists() and out_stats_path.exists():
                        self.logger.info(f"Index {index_metric} successfully calculated and saved for {year}-{month:02d}.")
                        print(f"Data saved at:\n- Monthly index: {out_monthly_path}\n- Statistics: {out_stats_path}")  
                        if index_metric in ["HIS", "HIA", "Tmean", "Tmax", "Tmin"]:
                            print(f"- Daily index data: {out_daily_path}")  
                    else:
                        self.logger.warning(f"Index {index_metric} for {year}-{month:02d} may not have been saved correctly.")

    def save_index_to_hazard(
        self, index_metric, year_list, month_list, area_selection, overwrite, data_out=None
    ):
        """
        _summary_

        Parameters
        ----------
        index_metric : str
            The climate index identifier to be processed into hazard format (e.g., 'TR' for tropical nights,
            'TX30' for hot days, or 'HW' for heatwave days).
        year_list : list of int
            List of years for which to process the climate index (e.g., [2020, 2021]).
        month_list : list of int
            List of months for which to process the climate index (values 1-12).
        area_selection : str or list
            Specifies the geographical area for which data should be processed. This parameter can be:
            - "global" for worldwide data.
            - A list of ISO alpha-3 codes (e.g., ["DEU", "CHE"]) for multiple countries.
            - A list of four numbers representing [north, west, south, east] in degrees for a bounding box.
        overwrite : bool
            If True, forces the system to overwrite existing hazard files. If False, skips processing
            if files are already present for the specified year, month, and region.
        data_out : str or Path, optional
            The base directory path for storing output data. If not provided, the default data directory
            is used as defined in the class (`self.data_out`).
        Returns
        -------
        None
        Saves the processed climate indices as hazard objects in HDF5 format, ready for further analysis.
        """
        # Handle potential None values
        self.data_out = Path(data_out) if data_out else self.data_out
        if self.data_out is None:  # Ensure data_out is defined
            raise ValueError(
                "Data output directory must be provided or set in the class."
            )

        # Calculate the bounds and area string
        bounds = self._get_bounds_for_area_selection(area_selection)
        area_str = (
            f"area{int(bounds[1])}_{int(bounds[0])}_{int(bounds[2])}_{int(bounds[3])}"
        )

        hazard_type = index_metric
        intensity_variable = f"{index_metric}"

        # Set intensity unit based on the type of index
        if index_metric in ["TR", "TX30", "HW"]:
            intensity_unit = "days"
        else:
            intensity_unit = "°C"

        for year in year_list:
            for month in month_list:
                # Define input and output paths
                input_file_name = (
                    self.data_out
                    / "indices"
                    / index_metric
                    / str(year)
                    / f"{month:02d}"
                    / f"{hazard_type}_{area_str}_{year}{month:02d}.nc"
                )
                output_dir = (
                    self.data_out / "hazard" / index_metric / str(year) / f"{month:02d}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                try:
                    # Check if the file already exists
                    file_path = (
                        output_dir
                        / f"hazard_{hazard_type}_{area_str}_{year}{month:02d}.hdf5"
                    )
                    if file_path.exists() and not overwrite:
                        self.logger.info(f"Hazard file {file_path} already exists.")
                        continue

                    else:
                        # Open input file and process data
                        with xr.open_dataset(str(input_file_name)) as ds:
                            ds["step"] = xr.DataArray(
                                [f"{date}-01" for date in ds["step"].values],
                                dims=["step"],
                            )
                            ds["step"] = pd.to_datetime(ds["step"].values)
                            ensemble_members = ds["number"].values
                            hazard = []

                            # Iterate through ensemble members and create Hazard objects
                            for i, member in enumerate(ensemble_members):
                                ds_subset = ds.sel(number=member)
                                hazard.append(
                                    Hazard.from_xarray_raster(
                                        data=ds_subset,
                                        hazard_type=hazard_type,
                                        intensity_unit=intensity_unit,
                                        intensity=intensity_variable,
                                        coordinate_vars={
                                            "event": "step",
                                            "longitude": "longitude",
                                            "latitude": "latitude",
                                        },
                                    )
                                )
                                if i == 0:
                                    number_lead_times = len(hazard[0].event_name)
                                hazard[i].event_name = [
                                    f"member{member}"
                                ] * number_lead_times

                        # Concatenate and write hazards
                        hazard = Hazard.concat(hazard)
                        hazard.check()
                        hazard.write_hdf5(str(file_path))

                        print(
                            f"Completed processing for {year}-{month:02d}. Data saved in {output_dir}."
                        )

                except FileNotFoundError as e:
                    print(
                        f"File not found: {e.filename}"
                    )  # Error handling for missing files
                except Exception as e:
                    print(f"An error occurred: {e}")  # General error handling

        # Display the final hazard object for verification
        last_hazard_file = file_path
        hazard_obj = Hazard.from_hdf5(str(last_hazard_file))
        hazard_obj.plot_intensity(1, smooth=False)

    @staticmethod
    def _is_data_present(file, vars):
        """Check if data is already present in the given directory.

        This utility function verifies the presence of a data file in the specified directory.
        It checks if the file exists and if it contains the required climate variables, ensuring
        that the data is complete and meets the required specifications.

        Parameters
        ----------
        file : str or Path
            The file path or filename to be checked. If the input is a string, it is automatically
            converted to a Path object for compatibility.
        vars : list of str
            A list of variable names that should be present in the data file (e.g., ['2m_dewpoint_temperature', '2m_temperature']).
            The function uses these variables to verify if the required dataset is present.

        Returns
        -------
        Path or None
            - Returns the file path (Path object) if a matching file is found in the directory.
            - Returns None if no matching file is present or the directory does not contain the required data.
        """
        file = Path(file) if isinstance(file, str) else file
        vars_short = [indicator.VAR_SPECS[var]["short_name"] for var in vars]
        parent_dir = file.parent

        if not parent_dir.exists():
            return None

        # Adjust rest for Path compatibility
        rest = re.search(r"(area.*)", str(file)).group(0)
        for filename in parent_dir.iterdir():
            s = re.search(rf'.*{".*".join(vars_short)}.*{rest}', filename.name)
            if s:
                return parent_dir / s.group(0)
        return None
