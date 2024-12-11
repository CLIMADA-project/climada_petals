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
from datetime import date
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from climada.hazard import Hazard
from climada.util.constants import SYSTEM_DIR
from climada.util.coordinates import get_country_geometries
from climada import CONFIG

import climada_petals.hazard.copernicus_interface.seasonal_statistics as seasonal_statistics
from climada_petals.hazard.copernicus_interface.seasonal_statistics import index_explanations
from climada_petals.hazard.copernicus_interface.downloader import download_data
from climada_petals.hazard.copernicus_interface.index_definitions import (
    IndexSpecEnum,
    get_short_name_from_variable,
)

# set path to store data
DATA_OUT = CONFIG.hazard.copernicus.seasonal_forecasts.dir()
LOGGER = logging.getLogger(__name__)


# ----- Utility Functions -----
# Utility function for month name to number conversion (if not already defined)
def month_name_to_number(month_name):
    """
    Convert a month name or numeric string to its corresponding month number.

    Parameters
    ----------
    month_name : str
        Name of the month (e.g., 'March' or 'Mar') or a numeric string (e.g., '3').

    Returns
    -------
    int
        The month number (1-12) corresponding to the given name or string.

    Raises
    ------
    ValueError
        If the input is neither a valid month name nor a number within the range 1-12.
    """
    # If already a valid integer or numeric string, convert directly
    if isinstance(month_name, int) or month_name.isdigit():
        month_num = int(month_name)
        if 1 <= month_num <= 12:
            return month_num
    
    try:
        # Full month name
        return list(calendar.month_name).index(month_name.capitalize())
    except ValueError:
        try:
            # Abbreviated month name
            return list(calendar.month_abbr).index(month_name.capitalize())
        except ValueError:
            raise ValueError(f"Invalid month name or number: {month_name}")
        

def calculate_leadtimes(year, initiation_month, lead_time_months):
    """
    Calculate lead times in hours for forecast initiation and lead months.

    Parameters
    ----------
    year : int
        The starting year for the forecast.
    initiation_month : int or str
        The initiation month for the forecast.
    lead_time_months : list
        A list of forecast lead months, either as integers or strings.

    Returns
    -------
    list[int]
        A sorted list of lead times in hours, with each step being 6 hours.

    Raises
    ------
    ValueError
        If initiation month or lead time months are invalid.
    Exception
        For general errors during lead time calculation.
    """
    try:
        # Convert initiation month to numeric if it is a string
        if isinstance(initiation_month, str):
            initiation_month = month_name_to_number(initiation_month)

        # Convert lead time months to numeric
        lead_time_months = [
            month_name_to_number(month) if isinstance(month, str) else month
            for month in lead_time_months
        ]

        # Validate initiation month
        if not (1 <= initiation_month <= 12):
            raise ValueError("Initiation month must be in the range 1-12.")

        # Validate lead time months
        if not lead_time_months:
            raise ValueError("At least one lead time month must be specified.")

        # Reference starting date for initiation
        initiation_date = date(year, initiation_month, 1)

        # Include initiation month in lead time calculation
        all_months = [initiation_month] + lead_time_months

        # Calculate lead times
        leadtimes = []
        for i, forecast_month in enumerate(all_months):
            # Adjust the year for months crossing into the next year
            forecast_year = year
            if forecast_month < initiation_month and i > 0:
                forecast_year += 1

            # Calculate the start of the forecast month
            forecast_date = date(forecast_year, forecast_month, 1)

            # Calculate difference in hours between initiation and forecast month
            time_diff = forecast_date - initiation_date
            start_hour = time_diff.days * 24
            leadtimes.extend(range(start_hour, start_hour + 24 * 31, 6))  # Full month in 6-hour steps

        # Remove duplicates and sort
        leadtimes = sorted(set(leadtimes))

        LOGGER.debug(
            f"Calculated lead times for year {year}, initiation month {initiation_month}, "
            f"forecast months {lead_time_months}: {leadtimes}"
        )

        return leadtimes

    except Exception as e:
        LOGGER.error(f"Error in calculating lead times for year {year} and initiation month {initiation_month}: {e}")
        raise


def handle_overwriting(function):
    """
    Decorator to manage file overwriting during processing.

    Parameters
    ----------
    function : callable
        The function to decorate.

    Returns
    -------
    callable
        The wrapped function, which checks file existence before executing.

    Notes
    -----
    If the file already exists and overwriting is not allowed, the existing file path
    is returned directly without calling the wrapped function.
    """

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
                LOGGER.debug(
                    f"A file of {[str(path) for path in output_file_name.values()]} already exists."
                )
                return output_file_name

        return function(output_file_name, overwrite, *args, **kwargs)

    return wrapper


class PathManager:
    """Centralized path management for input, processed, and hazard data."""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)

    def construct_path(self, sub_dir, file_name):
        """
        Construct a full file path from a sub-directory and file name.

        Parameters
        ----------
        sub_dir : str
            Sub-directory relative to the base directory.
        file_name : str
            File name to append to the sub-directory.

        Returns
        -------
        Path
            Full path to the file.
        """
        Path(self.base_dir / sub_dir).mkdir(parents=True, exist_ok=True)
        return self.base_dir / sub_dir / file_name

    def get_download_path(self, originating_centre, year, initiation_month, index_metric, area_str, format):
        """        
        Get the file path for downloaded data.

        Parameters
        ----------
        originating_centre : str
            The data source (e.g., 'dwd').
        year : int
            The forecast year.
        initiation_month : int
            The initiation month.
        index_metric : str
            Climate index (e.g., 'HW').
        area_str : str
            Area identifier string.
        format : str
            File format ('grib' or 'netcdf').

        Returns
        -------
        Path
            Path to the downloaded data file.
        """
        initiation_month_str = str(initiation_month).zfill(2)
        sub_dir = f"{originating_centre}/{year}/{initiation_month_str}/downloaded_data/{format}"
        file_name = f"{index_metric.lower()}_{area_str}.{format}"
        return self.construct_path(sub_dir, file_name)

    def get_daily_processed_path(self, originating_centre, year, initiation_month, index_metric, area_str):
        """        
        Get the file path for daily processed data.

        Parameters
        ----------
        originating_centre : str
            The data source (e.g., 'dwd').
        year : int
            The forecast year.
        initiation_month : int
            The initiation month.
        index_metric : str
            Climate index (e.g., 'HW').
        area_str : str
            Area identifier string.

        Returns
        -------
        Path
            Path to the daily processed file.
        """
        # Update the sub-directory to use initiation_month
        sub_dir = f"{originating_centre}/{year}/{str(initiation_month).zfill(2)}/processed_data"
        file_name = f"{index_metric.lower()}_{area_str}_daily.nc"
        return self.construct_path(sub_dir, file_name)

    def get_index_paths(self, originating_centre, year, month, index_metric, area_str):
        """        
        Get file paths for daily, monthly, and stats index files.

        Parameters
        ----------
        originating_centre : str
            The data source (e.g., 'dwd').
        year : int
            The forecast year.
        month : int
            The month for the index file.
        index_metric : str
            Climate index (e.g., 'HW').
        area_str : str
            Area identifier string.

        Returns
        -------
        dict
            Dictionary with keys ['daily', 'monthly', 'stats'] and corresponding file paths.
        """
        month_str = str(month).zfill(2)
        sub_dir = f"{originating_centre}/{year}/{month_str}/indices/{index_metric}"
        return {
            timeframe: self.construct_path(sub_dir, f"{timeframe}_{area_str}.nc")
            for timeframe in ["daily", "monthly", "stats"]
        }
    
    def get_hazard_path(self, originating_centre, year, month, index_metric, area_str):
        """        
        Get the file path for a Hazard HDF5 file.

        Parameters
        ----------
        originating_centre : str
            The data source (e.g., 'dwd').
        year : int
            The forecast year.
        month : int
            The month for the hazard file.
        index_metric : str
            Climate index (e.g., 'HW').
        area_str : str
            Area identifier string.

        Returns
        -------
        Path
            Path to the Hazard HDF5 file."""
        month_str = str(month).zfill(2)
        sub_dir = f"{originating_centre}/{year}/{month_str}/hazard/{index_metric}"
        file_name = f"{area_str}.hdf5"
        return self.construct_path(sub_dir, file_name)



# ----- Main Class -----
class SeasonalForecast:
    """
        Class for managing the download, processing, and analysis of seasonal climate forecast data.
    """
    def __init__(
        self,
        index_metric,
        year_list,
        lead_time_months,
        initiation_month,
        area_selection,
        format,
        originating_centre,
        system,
        data_out=None,
    ):
        """
        Initialize the SeasonalForecast instance with user-defined parameters for index calculation.

        Parameters
        ----------
        index_metric : str
            Climate index to calculate (e.g., 'HW', 'TR', 'Tmax').
        year_list : list[int]
            Years for which data should be downloaded and processed.
        lead_time_months : list[str]
            Lead time months (e.g., ["June", "July", "August"]).
        initiation_month : list[str]
            Initiation months for the forecasts (e.g., ["March", "April"]).
        area_selection : list or str
            Geographic area for the analysis (e.g., "global", ["DEU"], or [north, west, south, east]).
        format : str
            Data format ('grib' or 'netcdf').
        originating_centre : str
            Data source (e.g., "dwd").
        system : str
            Model configuration (e.g., "21").
        data_out : Path, optional
            Directory for storing data. Defaults to a pre-configured directory.
        """
        # Ensure initiation_month is a list
        if not isinstance(initiation_month, list):
            initiation_month = [initiation_month]
        
        # Convert initiation months to numbers and format as two-digit strings
        processed_initiation_months = []
        for month in initiation_month:
            # If month is a string, convert to number
            if isinstance(month, str):
                month = month_name_to_number(month)
            
            # Ensure month is a two-digit string
            processed_initiation_months.append(f"{month:02d}")
   
        self.index_metric = index_metric
        self.year_list = year_list

        # Process lead time months
        processed_lead_time_months = []
        for month in lead_time_months:
            # If month is a string, convert to number
            if isinstance(month, str):
                month = month_name_to_number(month)
            
            # Ensure month is a two-digit string
            processed_lead_time_months.append(f"{month:02d}")
        
        self.lead_time_months = processed_lead_time_months
        self.initiation_month = processed_initiation_months
        self.area_selection = area_selection  # Store user input for reference
        self.bounds = self._get_bounds_for_area_selection(
            area_selection
        )  # Convert user input to bounds
        self.area_str = f"area{int(self.bounds[1])}_{int(self.bounds[0])}_{int(self.bounds[2])}_{int(self.bounds[3])}"
        self.format = format
        self.originating_centre = originating_centre
        self.system = system
       
        # initialze path handling
        self.data_out = Path(data_out) if data_out else DATA_OUT
        self.path_manager = PathManager(self.data_out)

        # Get index specifications
        index_spec = IndexSpecEnum.get_info(self.index_metric)
        self.variables = index_spec.variables
        self.variables_short = [
            get_short_name_from_variable(var) for var in self.variables
        ]

    def explain_index(self, index_metric=None):
        """
        Retrieve and print details about the selected climate index.

        Parameters
        ----------
        index_metric : str, optional
            The climate index to explain. If None, uses the instance's index_metric.

        Returns
        -------
        dict
            Explanation and input data required for the index.
        """
        index_metric = index_metric or self.index_metric
        explanation = index_explanations(index_metric)

        if "error" in explanation:
            print(f"Error: {explanation['error']}")
            print(f"Supported indices: {', '.join(explanation['valid_indices'])}")
        else:
            print(f"Explanation for {index_metric}: {explanation['explanation']}")
            print(f"Required variables: {', '.join(explanation['input_data'])}")

    
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
        """
        Download data for specified years and initiation months.

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite existing downloaded files.

        Returns
        -------
        dict
            Paths to the downloaded files, indexed by (year, initiation_month).
        """
        output_files = {}
        for year in self.year_list:
            for initiation_month in self.initiation_month:
                # Download for each lead time month
                output_file_name = self.path_manager.get_download_path(
                    self.originating_centre,
                    year,
                    initiation_month,
                    self.index_metric,
                    self.area_str,
                    self.format,
                )
                
                output_files[(year, initiation_month)] = _download_data(
                    output_file_name,
                    overwrite,
                    self.variables,
                    year,
                    initiation_month,
                    self.format,
                    self.originating_centre,
                    self.system,
                    self.bounds,
                    self.lead_time_months,  # Pass lead time months directly
                )
        
        return output_files

    def _process(self, overwrite=False):
        """
        Process the downloaded data into daily NetCDF format.

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite existing processed files.

        Returns
        -------
        dict
            Paths to the processed files, indexed by (year, initiation_month, month).
        """
        processed_files = {}
        for year in self.year_list:
            for initiation_month in self.initiation_month:
                # Locate input file name
                input_file_name = self.path_manager.get_download_path(
                    self.originating_centre,
                    year,
                    initiation_month,
                    self.index_metric,
                    self.area_str,
                    self.format,
                )
                
                # Generate output file name 
                output_file_name = self.path_manager.get_daily_processed_path(
                    self.originating_centre,
                    year,
                    initiation_month,
                    self.index_metric,
                    self.area_str,
                )
                
                # Process each lead time month
                for month in self.lead_time_months:
                    processed_files[(year, initiation_month, month)] = _process_data(
                        output_file_name,
                        overwrite,
                        input_file_name,
                        self.variables_short,
                        self.format,
                    )
        
        return processed_files


    def download_and_process_data(self, overwrite=False):
        """
        Download and process climate forecast data.

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite existing files.

        Returns
        -------
        dict
            A dictionary containing paths to downloaded and processed data.
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
        """
        Calculate the specified climate index based on the downloaded data.

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite existing files.
        threshold : float
            Threshold value for the index.
        min_duration : int
            Minimum duration for consecutive conditions (specific to certain indices).
        max_gap : int
            Maximum allowable gap between conditions (specific to certain indices).
        tr_threshold : float
            Threshold for tropical nights.

        Returns
        -------
        dict
            Outputs for each processed year and month, indexed by (year, month).
        """
        index_outputs = {}

        # Iterate over each year and initiation month
        for year in self.year_list:
            for initiation_month in self.initiation_month:
                LOGGER.info(
                    f"Processing index {self.index_metric} for year {year}, initiation month {initiation_month}."
                )

                # Convert initiation month to integer
                initiation_month_int = int(initiation_month)
                lead_time_months = [int(month) for month in self.lead_time_months]
                all_months = [initiation_month_int] + lead_time_months  # Include initiation month

                for month in all_months:
                    # Adjust year for forecast months that fall in the next calendar year
                    process_year = year
                    if month < initiation_month_int and month in lead_time_months:
                        process_year += 1

                    # Use initiation month folder for forecast months
                    output_month = initiation_month if month in lead_time_months else f"{month:02d}"

                    # Determine the input file based on index type
                    if self.index_metric in ["TX30", "TR", "HW"]:  # Metrics using GRIB
                        input_file_name = self.path_manager.get_download_path(
                            self.originating_centre,
                            year,  # Year is initiation year
                            initiation_month,  # Initiation month folder
                            self.index_metric,
                            self.area_str,
                            "grib",
                        )
                    else:  # Metrics using processed NC files
                        input_file_name = self.path_manager.get_daily_processed_path(
                            self.originating_centre,
                            year,  
                            initiation_month,  
                            self.index_metric,
                            self.area_str,
                        )

                    # Generate paths for index outputs 
                    output_file_names = self.path_manager.get_index_paths(
                        self.originating_centre,
                        year,  
                        output_month,  #
                        self.index_metric,
                        self.area_str,
                    )

                    # Process the index and handle exceptions
                    try:
                        outputs = _calculate_index(
                            output_file_names,
                            overwrite,
                            input_file_name,
                            self.index_metric,
                            threshold=threshold,
                            min_duration=min_duration,
                            max_gap=max_gap,
                            tr_threshold=tr_threshold,
                        )
                        index_outputs[(process_year, month)] = outputs

                    except FileNotFoundError:
                        LOGGER.warning(
                            f"File not found for {process_year}-{month:02d}. Skipping..."
                        )
                    except Exception as e:
                        LOGGER.error(
                            f"Error processing index {self.index_metric} for {process_year}-{month}: {e}"
                        )

        return index_outputs

    ##########  Calculate hazard  ##########

    def save_index_to_hazard(self, overwrite=False):
        """
        Convert the calculated index to a Hazard object and save it as HDF5.

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite existing files.

        Returns
        -------
        dict
            Paths to the saved Hazard files, indexed by (year, month).
        """
        hazard_outputs = {}

        for year in self.year_list:
            for initiation_month in self.initiation_month:
                LOGGER.info(
                    f"Creating hazard for index {self.index_metric} for year {year}, initiation month {initiation_month}."
                )

                # Convert initiation month to integer
                initiation_month_int = int(initiation_month)
                lead_time_months = [int(month) for month in self.lead_time_months]
                all_months = [initiation_month_int] + lead_time_months  # Include initiation month

                for month in all_months:
                    # Adjust year for forecast months crossing into the next year
                    process_year = year
                    if month < initiation_month_int and month in lead_time_months:
                        process_year += 1

                    # Use initiation month folder for forecast months
                    output_month = initiation_month if month in lead_time_months else f"{month:02d}"

                    # Get input index file paths and hazard output file paths
                    input_file_name = self.path_manager.get_index_paths(
                        self.originating_centre,
                        year,  
                        output_month,
                        self.index_metric,
                        self.area_str,
                    )["monthly"]
                    output_file_name = self.path_manager.get_hazard_path(
                        self.originating_centre,
                        year,  
                        output_month,
                        self.index_metric,
                        self.area_str,
                    )

                    try:
                        # Convert index file to Hazard
                        hazard_outputs[(process_year, month)] = _convert_to_hazard(
                            output_file_name, overwrite, input_file_name, self.index_metric
                        )
                    except FileNotFoundError:
                        LOGGER.warning(
                            f"Monthly index file not found for {process_year}-{month:02d}. Skipping..."
                        )
                    except Exception as e:
                        LOGGER.error(
                            f"Failed to create hazard for {process_year}-{month:02d}: {e}"
                        )

        return hazard_outputs
    


    def forecast_skills(self):
        """
        Access and plot forecast skill data for the handler's parameters, filtered by the selected area.

        Raises
        ------
        ValueError
            If the originating_centre is not "dwd".
        ValueError
            If the index_metric is not "Tmax".

        Returns
        -------
        None
            Generates plots for forecast skill metrics based on the handler's parameters and the selected area.
        """
        # Check if the originating_centre is "dwd"
        if self.originating_centre.lower() != "dwd":
            raise ValueError(
                "Forecast skill metrics are only available for the 'dwd' provider. "
                f"Current provider: {self.originating_centre}"
            )

        # Check if the index_metric is "Tmax"
        if self.index_metric.lower() != "tmax":
            raise ValueError(
                "Forecast skills are only available for the 'Tmax' index. "
                f"Current index: {self.index_metric}"
            )

        # Define the file path pattern for forecast skill data (change for Zenodo when ready)
        base_path = Path("/Users/daraya/Downloads")
        file_name_pattern = "tasmaxMSESS_subyr_gcfs21_shc{month}-climatology_r1i1p1_1990-2019.nc"

        # Get bounds for the selected area
        bounds = self._get_bounds_for_area_selection(self.area_selection)
        if not bounds:
            raise ValueError(f"No bounds found for area selection: {self.area_selection}")

        # Iterate over initiation months and access the corresponding file
        for month in self.initiation_month:
            # Convert the numeric month to a two-digit string for the file name
            month_str = f"{int(month):02d}"

            # Construct the file name and path
            file_path = base_path / file_name_pattern.format(month=month_str)

            if not file_path.exists():
                print(f"Skill data file for month {month_str} not found: {file_path}")
                continue

            # Load the data using xarray
            try:
                with xr.open_dataset(file_path) as ds:
                    # Subset the dataset by area bounds
                    north, west, south, east = bounds
                    subset_ds = ds.sel(
                        lon=slice(west, east),
                        lat=slice(north, south)
                    )

                    # Plot each variable
                    variables = ["tasmax_fc_mse", "tasmax_ref_mse", "tasmax_msess", "tasmax_msessSig"]
                    for var in variables:
                        if var in subset_ds:
                            plt.figure(figsize=(10, 8))
                            ax = plt.axes(projection=ccrs.PlateCarree())

                            # Adjust color scale to improve clarity
                            vmin = subset_ds[var].quantile(0.05).item()
                            vmax = subset_ds[var].quantile(0.95).item()

                            im = subset_ds[var].isel(time=0).plot(
                                ax=ax,
                                cmap="coolwarm",
                                vmin=vmin,
                                vmax=vmax,
                                add_colorbar=False
                            )

                            cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.1, shrink=0.7)
                            cbar.set_label(var, fontsize=10)

                            ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
                            ax.add_feature(cfeature.BORDERS, linestyle=':')
                            ax.add_feature(cfeature.COASTLINE)
                            ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.3)
                            ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

                            plt.title(f"{var} for month {month_str}, area {self.area_selection}")
                            plt.show()
                        else:
                            print(f"Variable {var} not found in dataset for month {month_str}.")

            except Exception as e:
                print(f"Failed to load or process data for month {month_str}: {e}")





# ----- Decorated Functions -----

@handle_overwriting
def _download_data(
    output_file_name,
    overwrite,
    variables,
    year,
    initiation_month,
    format,
    originating_centre,
    system,
    bounds,
    lead_time_months,
):
    """
    Download seasonal forecast data for a specific year and initiation month.

    Parameters
    ----------
    output_file_name : Path
        Path to save the downloaded data.
    overwrite : bool
        Whether to overwrite existing files.
    variables : list[str]
        List of variables to download.
    year : int
        The year for which data is being downloaded.
    initiation_month : int
        The month when the forecast is initiated.
    format : str
        File format for the downloaded data, either 'grib' or 'netcdf'.
    originating_centre : str
        The data source, e.g., 'dwd' for German Weather Service.
    system : str
        The specific forecast model or system version.
    bounds : list[float]
        Geographical bounds [north, west, south, east] for the data.
    lead_time_months : list[int]
        List of forecast months relative to the initiation month.

    Returns
    -------
    Path
        Path to the downloaded data file.
    """
    try:
        # Calculate lead times using the lead time months
        leadtimes = calculate_leadtimes(
            year,
            initiation_month=initiation_month,
            lead_time_months=lead_time_months
        )
        
        # Prepare download parameters
        download_params = {
            "format": format,
            "originating_centre": originating_centre,
            "area": bounds,
            "system": system,
            "variable": variables,
            "month": initiation_month,
            "year": year,
            "day": "01",
            "leadtime_hour": leadtimes,
        }
        
        # Perform download
        downloaded_file = download_data(
            "seasonal-original-single-levels",
            download_params,
            output_file_name,
            overwrite=overwrite,
        )
        
        return downloaded_file
    
    except Exception as e:
        LOGGER.error(f"Error downloading data for {year}-{initiation_month}: {e}")
        raise

@handle_overwriting
def _process_data(output_file_name, overwrite, input_file_name, variables, format):
    """
    Process a single input file into the desired daily NetCDF format.

    Parameters
    ----------
    output_file_name : Path
        Path to save the processed data.
    overwrite : bool
        Whether to overwrite existing files.
    input_file_name : Path
        Path to the input file.
    variables : list[str]
        Variables to process in the input file.
    format : str
        File format of the input file ('grib' or 'netcdf').

    Returns
    -------
    Path
        Path to the processed data file.
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
    """
    Calculate and save climate indices based on the input data.

    Parameters
    ----------
    output_file_names : dict
        Dictionary containing paths for daily, monthly, and stats output files.
    overwrite : bool
        Whether to overwrite existing files.
    input_file_name : Path
        Path to the input file.
    index_metric : str
        Climate index to calculate (e.g., 'HW', 'TR').
    threshold : float, optional
        Threshold for the index calculation (specific to the index type).
    min_duration : int, optional
        Minimum duration for events (specific to the index type).
    max_gap : int, optional
        Maximum gap allowed between events (specific to the index type).
    tr_threshold : float, optional
        Threshold for tropical nights (specific to the 'TR' index).

    Returns
    -------
    dict
        Paths to the saved index files.
    """
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
    Convert an index file to a Hazard object and save it as HDF5.

    Parameters
    ----------
    output_file_name : Path
        Path to save the Hazard file.
    overwrite : bool
        Whether to overwrite existing files.
    input_file_name : Path
        Path to the input index file.
    index_metric : str
        Climate index metric (e.g., 'HW', 'TR').

    Returns
    -------
    Path
        Path to the saved Hazard file.

    Raises
    ------
    KeyError
        If required variables are missing in the dataset.
    Exception
        If the conversion process fails.
    """
    try:
        with xr.open_dataset(str(input_file_name)) as ds:
            if "step" not in ds.variables:
                raise KeyError(
                    f"Missing 'step' variable in dataset for {input_file_name}."
                )

            ds["step"] = xr.DataArray(
                [f"{date}-01" for date in ds["step"].values],
                dims=["step"],
            )
            ds["step"] = pd.to_datetime(ds["step"].values)

            ensemble_members = ds.get("number", [0]).values
            hazards = []

            # Determine intensity unit and variable
            intensity_unit = (
                "%"
                if index_metric == "RH"
                else "days" if index_metric in ["TR", "TX30", "HW"] else "°C"
            )
            intensity_variable = index_metric

            if intensity_variable not in ds.variables:
                raise KeyError(
                    f"No variable named '{intensity_variable}' in the dataset. "
                    f"Available variables: {list(ds.variables)}"
                )

            # Create Hazard objects
            for member in ensemble_members:
                ds_subset = ds.sel(number=member) if "number" in ds.dims else ds
                hazard = Hazard.from_xarray_raster(
                    data=ds_subset,
                    hazard_type=index_metric,
                    intensity_unit=intensity_unit,
                    intensity=intensity_variable,
                    coordinate_vars={
                        "event": "step",
                        "longitude": "longitude",
                        "latitude": "latitude",
                    },
                )
                hazard.event_name = [
                    f"member{member}" for _ in range(len(hazard.event_name))
                ]
                hazards.append(hazard)

            hazard = Hazard.concat(hazards)
            hazard.check()
            hazard.write_hdf5(str(output_file_name))

        LOGGER.info(f"Hazard file saved to {output_file_name}")
        return output_file_name

    except Exception as e:
        LOGGER.error(f"Failed to convert {input_file_name} to hazard: {e}")
        raise
