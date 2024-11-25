import logging
import calendar
import re
from pathlib import Path, PosixPath
import datetime as dt
import requests

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cdsapi

from climada.hazard import Hazard
from climada.util.constants import SYSTEM_DIR
from climada.util.coordinates import get_country_geometries
from climada import CONFIG

import climada_petals.hazard.copernicus_interface.seasonal_statistics as seasonal_statistics
from climada_petals.hazard.copernicus_interface.downloader import download_data
from climada_petals.hazard.copernicus_interface.index_definitions import (
    IndexSpecEnum,
    get_short_name_from_variable,
)

# set path to store data
DATA_OUT = CONFIG.hazard.copernicus.seasonal_forecasts.dir()
LOGGER = logging.getLogger(__name__)


class SeasonalForecast:
    def __init__(
        self,
        index_metric,
        year_list,
        month_list,
        area_selection,
        format,
        originating_centre,
        system,
        max_lead_month,
        data_out=None,
    ):
        """
        Initialize the SeasonalForecast instance with user-defined parameters for index calculation.
        """
        self.index_metric = index_metric
        self.year_list = year_list
        self.month_list = [f"{month:02d}" for month in month_list]
        self.area_selection = area_selection  # Store user input for reference
        self.bounds = self._get_bounds_for_area_selection(
            area_selection
        )  # Convert user input to bounds
        self.area_str = f"area{int(self.bounds[1])}_{int(self.bounds[0])}_{int(self.bounds[2])}_{int(self.bounds[3])}"
        self.format = format
        self.originating_centre = originating_centre
        self.system = system
        self.max_lead_month = max_lead_month

        # initialze path handling
        self.data_out = Path(data_out) if data_out else DATA_OUT
        self.path_manager = PathManager(self.data_out)

        # Get index specifications
        index_spec = IndexSpecEnum.get_info(self.index_metric)
        self.variables = index_spec.variables
        self.variables_short = [
            get_short_name_from_variable(var) for var in self.variables
        ]

    @staticmethod
    def _get_bounds_for_area_selection(area_selection, margin=0.2):
        """Determine geographic bounds based on area selection, including global or country ISO codes."""
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
                    LOGGER.error(
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
                    LOGGER.warning(
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

    ##########  Download and process ##########

    def _download(self, overwrite=False):
        """Download data for specified years and months."""
        output_files = {}

        for year in self.year_list:
            for month in self.month_list:
                month_int = int(month)
                # Construct output file name
                output_file_name = self.path_manager.get_download_path(
                    self.originating_centre,
                    year,
                    month,
                    self.index_metric,
                    self.area_str,
                    self.format,
                )

                output_files[(year, month)] = _download_data(
                    output_file_name,
                    overwrite,
                    self.variables,
                    year,
                    month_int,
                    self.format,
                    self.originating_centre,
                    self.system,
                    self.bounds,
                    self.max_lead_month,
                )
        return output_files

    def _process(self, overwrite=False):
        """
        Process downloaded data into daily NetCDF format and return processed file paths.
        """
        processed_files = {}
        for year in self.year_list:
            for month in self.month_list:

                # Locate input and output file names
                input_file_name = self.path_manager.get_download_path(
                    self.originating_centre,
                    year,
                    month,
                    self.index_metric,
                    self.area_str,
                    self.format,
                )

                output_file_name = self.path_manager.get_daily_processed_path(
                    self.originating_centre,
                    year,
                    month,
                    self.index_metric,
                    self.area_str,
                )

                # Call _process_data for daily NetCDF file creation
                processed_files[(year, month)] = _process_data(
                    output_file_name,
                    overwrite,
                    input_file_name,
                    self.variables_short,
                    self.format,
                )

        return processed_files

    def download_and_process_data(self, overwrite=False):
        """
        Download and process climate forecast data based on the area selection.
        """

        # Call high-level methods for downloading and processing
        created_files = {}
        created_files["downloaded_data"] = self._download(
            overwrite=overwrite
        )  # Handles iteration and calls _download_data

        created_files["processed_data"] = self._process(
            overwrite=overwrite
        )  # Handles iteration and calls _process_data
        return created_files

    ##########  Calculate index ##########

    def calculate_index(
        self, overwrite=False, threshold=27, min_duration=3, max_gap=0, tr_threshold=20
    ):
        """ """
        index_outputs = {}

        for year in self.year_list:
            for month in self.month_list:
                LOGGER.info(f"Processing index {self.index_metric} for {year}-{month}.")

                if self.index_metric in ["TX30", "TR", "HW"]:
                    input_file_name = self.path_manager.get_download_path(
                        self.originating_centre,
                        year,
                        month,
                        self.index_metric,
                        self.area_str,
                        "grib",
                    )
                else:
                    input_file_name = self.path_manager.get_daily_processed_path(
                        self.originating_centre,
                        year,
                        month,
                        self.index_metric,
                        self.area_str,
                        format="nc",
                    )

                output_file_names = self.path_manager.get_index_paths(
                    self.originating_centre,
                    year,
                    month,
                    self.index_metric,
                    self.area_str,
                )

                try:
                    # Call _process_index to perform calculations for the given year and month
                    outputs = _calculate_index(
                        output_file_names,
                        overwrite,
                        input_file_name,
                        self.index_metric,
                        threshold=27,
                        min_duration=3,
                        max_gap=0,
                        tr_threshold=20,
                    )
                    index_outputs[(year, month)] = outputs
                except Exception as e:
                    LOGGER.error(
                        f"Error processing index {self.index_metric} for {year}-{month}: {e}"
                    )

        return index_outputs

    ##########  Calculate hazard and plot a sample ##########

    def save_index_to_hazard(self, overwrite=False):
        """ """
        hazard_outputs = {}

        for year in self.year_list:
            for month in self.month_list:
                LOGGER.info(
                    f"Creating hazard for index {self.index_metric} for {year}-{month}."
                )

                # Get the input index file and the hazard output file paths
                input_file_name = self.path_manager.get_index_paths(
                    self.originating_centre,
                    year,
                    month,
                    self.index_metric,
                    self.area_str,
                )["monthly"]
                output_file_name = self.path_manager.get_hazard_path(
                    self.originating_centre,
                    year,
                    month,
                    self.index_metric,
                    self.area_str,
                )

                try:
                    # Convert index file to hazard
                    hazard_outputs[(year, month)] = _convert_to_hazard(
                        output_file_name, overwrite, input_file_name, self.index_metric
                    )
                except Exception as e:
                    LOGGER.error(f"Failed to create hazard for {year}-{month}: {e}")

        return hazard_outputs


class PathManager:
    """Centralized path management for input, processed, and hazard data."""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)

    def construct_path(self, sub_dir, file_name):
        """Constructs a full path from a sub-directory and file name."""
        Path(self.base_dir / sub_dir).mkdir(parents=True, exist_ok=True)
        return self.base_dir / sub_dir / file_name

    def get_download_path(
        self, originating_centre, year, month, index_metric, area_str, format
    ):
        """
        Path for downloaded grib data.
        Example: input_data/dwd/grib/2002/07/HW_dwd_area4_49_33_20_200207.grib
        """
        sub_dir = f"{originating_centre}/{year}/{month}/downloaded_data/{format}"
        file_name = f"{index_metric.lower()}_{area_str}.{format}"
        return self.construct_path(sub_dir, file_name)

    def get_daily_processed_path(
        self, originating_centre, year, month, index_metric, area_str
    ):
        """
        Path for processed daily netcdf data.
        Example: input_data/netcdf/daily/dwd/2002/07/dwd_t2m_area4_49_33_20_200207.nc
        """
        sub_dir = f"{originating_centre}/{year}/{month}/processed_daily_data"
        file_name = f"{index_metric.lower()}_{area_str}.nc"
        return self.construct_path(sub_dir, file_name)

    def get_index_paths(self, originating_centre, year, month, index_metric, area_str):
        sub_dir = f"{originating_centre}/{year}/{month}/indeces/{index_metric}"
        return {
            timeframe: self.construct_path(sub_dir, f"{timeframe}_{area_str}.nc")
            for timeframe in ["daily", "monthly", "stats"]
        }

    def get_hazard_path(self, originating_centre, year, month, index_metric, area_str):
        """
        Path for hazard HDF5 output data.
        Example: hazard/dwd/HW/2002/07/hazard_HW_dwd_area4_49_33_20_200207.hdf5
        """
        sub_dir = f"{originating_centre}/{year}/{month}/hazard/{index_metric}"
        file_name = f"{area_str}.hdf5"
        return self.construct_path(sub_dir, file_name)


def handle_overwriting(function):
    """Decorator to manage overwriting functionality."""

    def wrapper(output_file_name, overwrite, *args, **kwargs):
        # if data exists and we do not want to overwrite
        if isinstance(output_file_name, PosixPath):
            if not overwrite and output_file_name.exists():
                LOGGER.info(f"{output_file_name} already exists.")
                return output_file_name
        elif isinstance(output_file_name, dict):
            if not overwrite and any(
                [path.exists() for path in output_file_name.values()]
            ):
                LOGGER.info(
                    f"A file of {[str(path) for path in output_file_name.values()]} already exists."
                )
                return output_file_name

        return function(output_file_name, overwrite, *args, **kwargs)

    return wrapper


@handle_overwriting
def _download_data(
    output_file_name,
    overwrite,
    variables,
    year,
    month,
    format,
    originating_centre,
    system,
    bounds,
    max_lead_month,
):
    """Download seasonal forecast data for a specific year and month."""
    try:

        # Calculate lead times
        # min_lead, max_lead = _calc_min_max_lead(year, month, max_lead_month)
        # leadtimes = list(range(min_lead, max_lead, 6))
        # leadtimes = self._calculate_leadtimes(year, month, self.max_lead_month)
        leadtimes = _calculate_leadtimes(year, month, max_lead_month)

        # Prepare download parameters
        download_params = {
            "format": format,
            "originating_centre": originating_centre,
            "area": bounds,
            "system": system,
            "variable": variables,
            "month": month,
            "year": year,
            "day": "01",
            "leadtime_hour": leadtimes,
        }

        # Perform download
        output_file_name = download_data(
            "seasonal-original-single-levels",
            download_params,
            output_file_name,
            overwrite=True,
        )
        return output_file_name

    except Exception as e:
        LOGGER.error(f"Error downloading data for {year}-{month}: {e}")
        raise


@handle_overwriting
def _process_data(output_file_name, overwrite, input_file_name, variables, format):
    """
    Process a single input file into the desired daily NetCDF format.
    """
    try:
        with xr.open_dataset(
            input_file_name,
            engine="cfgrib" if format == "grib" else None,
        ) as ds:
            # Coarsen the data
            ds_mean = ds.coarsen(step=4, boundary="trim").mean()
            ds_max = ds.coarsen(step=4, boundary="trim").max()
            ds_min = ds.coarsen(step=4, boundary="trim").min()

        # Create a new dataset combining mean, max, and min values
        combined_ds = xr.Dataset()
        for var in variables:
            combined_ds[f"{var}_mean"] = ds_mean[var]
            combined_ds[f"{var}_max"] = ds_max[var]
            combined_ds[f"{var}_min"] = ds_min[var]

        # Save the combined dataset to NetCDF
        combined_ds.to_netcdf(str(output_file_name))
        LOGGER.info(f"Daily file saved to {output_file_name}")

        return output_file_name

    except FileNotFoundError:
        LOGGER.error(f"Input file {input_file_name} does not exist, processing failed.")
        raise
    except Exception as e:
        LOGGER.error(f"Error during processing for {input_file_name}: {e}")
        raise


@handle_overwriting
def _calculate_index(
    output_file_names,
    overwrite,
    input_file_name,
    index_metric,
    threshold=27,
    min_duration=3,
    max_gap=0,
    tr_threshold=20,
):
    """ """
    # Define output paths
    daily_output_path = output_file_names["daily"]
    monthly_output_path = output_file_names["monthly"]
    stats_output_path = output_file_names["stats"]

    # Define index-specific parameters
    kwargs = {}
    if index_metric == "TR":
        kwargs["tr_threshold"] = tr_threshold
    elif index_metric == "HW":
        kwargs["threshold"] = threshold
        kwargs["min_duration"] = min_duration
        kwargs["max_gap"] = max_gap

    # Perform calculation
    if index_metric in [
        "HIS",
        "HIA",
        "Tmean",
        "Tmax",
        "Tmin",
        "HUM",
        "RH",
        "AT",
        "WBGT",
    ]:
        ds_daily, ds_monthly, ds_stats = (
            seasonal_statistics.calculate_heat_indices_metrics(
                input_file_name, index_metric
            )
        )
    elif index_metric == "TR":
        ds_daily, ds_monthly, ds_stats = seasonal_statistics.calculate_tr_days(
            input_file_name, index_metric, **kwargs
        )
    elif index_metric == "TX30":
        ds_daily, ds_monthly, ds_stats = seasonal_statistics.calculate_tx30_days(
            input_file_name, index_metric
        )
    elif index_metric == "HW":
        ds_daily, ds_monthly, ds_stats = seasonal_statistics.calculate_hw_days(
            input_file_name, index_metric, **kwargs
        )
    else:
        LOGGER.error(f"Index {index_metric} is not implemented.")
        return None

    # Save outputs
    if ds_daily is not None:
        ds_daily.to_netcdf(daily_output_path)
        LOGGER.info(f"Saved daily index to {daily_output_path}")
    if ds_monthly is not None:
        ds_monthly.to_netcdf(monthly_output_path)
        LOGGER.info(f"Saved monthly index to {monthly_output_path}")
    if ds_stats is not None:
        ds_stats.to_netcdf(stats_output_path)
        LOGGER.info(f"Saved stats index to {stats_output_path}")

    return {
        "daily": daily_output_path,
        "monthly": monthly_output_path,
        "stats": stats_output_path,
    }


@handle_overwriting
def _convert_to_hazard(output_file_name, overwrite, input_file_name, index_metric):
    """
    Convert an index file to a Hazard object and save it...
    """
    try:
        with xr.open_dataset(str(input_file_name)) as ds:
            # Update the 'step' dimension to include formatted date strings
            ds["step"] = xr.DataArray(
                [f"{date}-01" for date in ds["step"].values],
                dims=["step"],
            )
            ds["step"] = pd.to_datetime(ds["step"].values)

            # Get the ensemble members
            ensemble_members = ds["number"].values
            hazards = []

            # Determine intensity unit based on the index metric
            intensity_unit = (
                "%"
                if index_metric == "RH"
                else "days" if index_metric in ["TR", "TX30", "HW"] else "°C"
            )
            hazard_type = index_metric
            intensity_variable = index_metric  # Match exact variable name in dataset

            # Ensure the intensity variable exists in the dataset
            if intensity_variable not in ds.variables:
                raise KeyError(
                    f"No variable named '{intensity_variable}' in the dataset. "
                    f"Available variables: {list(ds.variables)}"
                )

            # Iterate through ensemble members and create Hazard objects
            for i, member in enumerate(ensemble_members):
                ds_subset = ds.sel(number=member)
                hazard = Hazard.from_xarray_raster(
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

                # Set event names for the ensemble member
                if i == 0:
                    number_lead_times = len(hazard.event_name)
                hazard.event_name = [f"member{member}"] * number_lead_times

                hazards.append(hazard)

            # Concatenate all hazards into one
            hazard = Hazard.concat(hazards)

            # Validate and save the Hazard object
            hazard.check()
            hazard.write_hdf5(str(output_file_name))
            return output_file_name

    except Exception as e:
        LOGGER.error(f"Failed to convert {input_file_name} to hazard: {e}")
        raise


def _calculate_leadtimes(year, month, leadtime_months=1):  
    month = int(month)
    total_hours = 0  
    leadtimes = [0]  
    for m in range(month, month + leadtime_months):  
        adjusted_year, adjusted_month = year, m  
        if m > 12:  
            adjusted_year += 1  
            adjusted_month = m - 12  

        num_days_month = calendar.monthrange(adjusted_year, adjusted_month)[
            1
        ]  
        total_hours += num_days_month * 24  

        # Append leadtimes at intervals of 6 hours  
        leadtimes.extend(range(leadtimes[-1] + 6, total_hours + 6, 6))  
    return leadtimes  
