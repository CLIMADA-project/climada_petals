import logging
import calendar
from pathlib import Path, PosixPath

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import date
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from climada.hazard import Hazard
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


# ----- Utility Functions -----
# Utility function for month name to number conversion (if not already defined)


def month_name_to_number(month):
    """_summary_

    Parameters
    ----------
    month : (int, str)
        month as integer or string

    Returns
    -------
    int
        month as integer

    """
    if isinstance(month, int):  # Already a number
        if 1 <= month <= 12:
            return month
        else:
            raise ValueError("Month number must be between 1 and 12.")
    elif isinstance(month, str):
        month = month.capitalize()  # Ensure consistent capitalization
        if month in calendar.month_name:
            return list(calendar.month_name).index(month)
        elif month in calendar.month_abbr:
            return list(calendar.month_abbr).index(month)
    raise ValueError(f"Invalid month input: {month}")


def calculate_leadtimes(year, initiation_month, valid_period):
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

    # Convert initiation month to numeric if it is a string
    if isinstance(initiation_month, str):
        initiation_month = month_name_to_number(initiation_month)

    # Convert valid_period to numeric
    valid_period = [
        month_name_to_number(month) if isinstance(month, str) else month
        for month in valid_period
    ]

    # compute years of valid period
    valid_years = np.array([year, year])
    if initiation_month > valid_period[0]:  # forecast for next year
        valid_years += np.array([1, 1])
    if valid_period[1] < valid_period[0]:  # forecast including two different years
        valid_years[1] += 1

    # Reference starting date for initiation
    initiation_date = date(year, initiation_month, 1)
    valid_period_start = date(valid_years[0], valid_period[0], 1)
    valid_period_end = date(
        valid_years[1],
        valid_period[1],
        calendar.monthrange(valid_years[1], valid_period[1])[1],
    )

    return list(
        range(
            (valid_period_start - initiation_date).days * 24,
            (valid_period_end - initiation_date).days * 24 + 24,
            6,
        )
    )


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
                return None
        elif isinstance(output_file_name, dict):
            if not overwrite and any(
                [path.exists() for path in output_file_name.values()]
            ):
                LOGGER.info(
                    f"A file of {[str(path) for path in output_file_name.values()]} already exists."
                )
                return None

        return function(output_file_name, overwrite, *args, **kwargs)

    return wrapper


# ----- Main Class -----
class SeasonalForecast:
    """
    Class for managing the download, processing, and analysis of seasonal climate forecast data.
    """

    def __init__(
        self,
        index_metric,
        year_list,
        valid_period,
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
        # initiate initiation month, valid period, and leadtimes
        if not isinstance(initiation_month, list):
            initiation_month = [initiation_month]
        if not isinstance(valid_period, list) or len(valid_period) != 2:
            raise ValueError("Valid period must be a list of two months.")
        self.initiation_month_str = [
            f"{month_name_to_number(month):02d}" for month in initiation_month
        ]
        self.valid_period = [month_name_to_number(month) for month in valid_period]
        self.valid_period_str = "_".join(
            [f"{month:02d}" for month in self.valid_period]
        )

        self.index_metric = index_metric
        self.year_list = year_list
        self.area_selection = area_selection  # Store user input for reference
        self.bounds = self._get_bounds_for_area_selection(
            area_selection
        )  # Convert user input to bounds
        self.area_str = f"area{int(self.bounds[1])}_{int(self.bounds[0])}_{int(self.bounds[2])}_{int(self.bounds[3])}"
        self.format = format
        self.originating_centre = originating_centre
        self.system = system

        # initialze base directory
        self.data_out = Path(data_out) if data_out else DATA_OUT

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
        print(
            f"Explanation for {index_metric}: {IndexSpecEnum.get_info(index_metric).explanation}"
        )
        print(
            f"Required variables: {', '.join(IndexSpecEnum.get_info(index_metric).variables)}"
        )

    @staticmethod
    def get_file_path(
        base_dir,
        originating_centre,
        year,
        initiation_month_str,
        valid_period_str,
        data_type,
        index_metric,
        area_str,
        format="",
    ):
        """
        Provide general file paths for forecast pipeline.
        """

        if data_type == "downloaded_data":
            data_type += f"/{format}"
        elif data_type == "hazard":
            data_type += f"/{index_metric}"
            format = "hdf5"
        elif data_type == "indices":
            data_type += f"/{index_metric}"
            format = "nc"
        elif data_type == "processed_data":
            format = "nc"
        else:
            raise ValueError(f"Unknown format {format}.")
        
        # prepare parent directory
        sub_dir = f"{base_dir}/{originating_centre}/{year}/init{initiation_month_str}/valid{valid_period_str}/{data_type}"
        
        if data_type.startswith("indices"):
            return {timeframe: Path(f"{sub_dir}/{index_metric}_{area_str}_{timeframe}.{format}")
                    for timeframe in ["daily", "monthly", "stats"]}
        else:
            return Path(f"{sub_dir}/{index_metric}_{area_str}.{format}")
    
    def get_pipeline_path(
        self,
        year,
        initiation_month_str,
        data_type
    ):
        """
        Provide (and possibly create) file paths for forecast pipeline.
        """

        file_path = self.get_file_path(
            self.data_out,
            self.originating_centre,
            year,
            initiation_month_str,
            self.valid_period_str,
            data_type,
            self.index_metric,
            self.area_str,
            self.format
        )

        # create directory if not existing
        if data_type == "indices":
            file_path["monthly"].parent.mkdir(parents=True, exist_ok=True)
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        return file_path

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
            for month_str in self.initiation_month_str:
                leadtimes = calculate_leadtimes(year, int(month_str), self.valid_period)

                # Generate output file name
                downloaded_data_path = self.get_pipeline_path(year, month_str, "downloaded_data")

                output_files[(year, month_str, self.valid_period_str)] = _download_data(
                    downloaded_data_path,
                    overwrite,
                    self.variables,
                    year,
                    month_str,
                    self.format,
                    self.originating_centre,
                    self.system,
                    self.bounds,
                    leadtimes,
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
            for month_str in self.initiation_month_str:
                # Locate input file name
                downloaded_data_path = self.get_pipeline_path(year, month_str, "downloaded_data")
                # Generate output file name
                processed_data_path = self.get_pipeline_path(year, month_str, "processed_data")

                processed_files[(year, month_str, self.valid_period_str)] = (
                    _process_data(
                        processed_data_path,
                        overwrite,
                        downloaded_data_path,
                        self.variables_short,
                        self.format,
                    )
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
        self,
        overwrite=False,
        hw_threshold=27,
        hw_min_duration=3,
        hw_max_gap=0,
        tr_threshold=20,
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
            for month_str in self.initiation_month_str:
                LOGGER.info(
                    f"Processing index {self.index_metric} for year {year}, initiation month {month_str}."
                )
                # Determine the input file based on index type
                if self.index_metric in ["TX30", "TR", "HW"]:  # Metrics using GRIB
                    input_data_path = self.get_pipeline_path(year, month_str, "downloaded_data")
                else:  # Metrics using processed NC files
                    input_data_path = self.get_pipeline_path(year, month_str, "processed_data")

                # Generate paths for index outputs
                index_data_paths = self.get_pipeline_path(year, month_str, "indices")

                # Process the index and handle exceptions
                try:
                    outputs = _calculate_index(
                        index_data_paths,
                        overwrite,
                        input_data_path,
                        self.index_metric,
                        tr_threshold=tr_threshold,
                        hw_min_duration=hw_min_duration,
                        hw_max_gap=hw_max_gap,
                        hw_threshold=hw_threshold,
                    )
                    index_outputs[(year, month_str, self.valid_period_str)] = outputs

                except FileNotFoundError:
                    LOGGER.warning(
                        f"File not found for {year}-{month_str}. Skipping..."
                    )
                except Exception as e:
                    LOGGER.error(
                        f"Error processing index {self.index_metric} for {year}-{month_str}: {e}"
                    )
                    raise e

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
            for month_str in self.initiation_month_str:
                LOGGER.info(
                    f"Creating hazard for index {self.index_metric} for year {year}, initiation month {month_str}."
                )
                # Get input index file paths and hazard output file paths
                index_data_path = self.get_pipeline_path(year, month_str, "indices")["monthly"]
                hazard_data_path = self.get_pipeline_path(year, month_str, "hazard")

                try:
                    # Convert index file to Hazard
                    hazard_outputs[(year, month_str, self.valid_period_str)] = (
                        _convert_to_hazard(
                            hazard_data_path,
                            overwrite,
                            index_data_path,
                            self.index_metric,
                        )
                    )
                except FileNotFoundError:
                    LOGGER.warning(
                        f"Monthly index file not found for {year}-{month_str}. Skipping..."
                    )
                except Exception as e:
                    LOGGER.error(f"Failed to create hazard for {year}-{month_str}: {e}")

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
            print(
                "Forecast skill metrics are only available for the 'dwd' provider. "
                f"Current provider: {self.originating_centre}"
            )
            return

        # Check if the index_metric is "Tmax"
        if self.index_metric.lower() != "tmax":
            print(
                "Forecast skills are only available for the 'Tmax' index. "
                f"Current index: {self.index_metric}"
            )
            return

        # Define the file path pattern for forecast skill data (change for Zenodo when ready)
        base_path = Path("/Users/daraya/Downloads")
        file_name_pattern = (
            "tasmaxMSESS_subyr_gcfs21_shc{month}-climatology_r1i1p1_1990-2019.nc"
        )

        # Get bounds for the selected area
        bounds = self._get_bounds_for_area_selection(self.area_selection)
        if not bounds:
            raise ValueError(
                f"No bounds found for area selection: {self.area_selection}"
            )

        # Iterate over initiation months and access the corresponding file
        for month in self.initiation_month_str:
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
                    subset_ds = ds.sel(lon=slice(west, east), lat=slice(north, south))

                    # Plot each variable
                    variables = [
                        "tasmax_fc_mse",
                        "tasmax_ref_mse",
                        "tasmax_msess",
                        "tasmax_msessSig",
                    ]
                    for var in variables:
                        if var in subset_ds:
                            plt.figure(figsize=(10, 8))
                            ax = plt.axes(projection=ccrs.PlateCarree())

                            # Adjust color scale to improve clarity
                            vmin = subset_ds[var].quantile(0.05).item()
                            vmax = subset_ds[var].quantile(0.95).item()

                            im = (
                                subset_ds[var]
                                .isel(time=0)
                                .plot(
                                    ax=ax,
                                    cmap="coolwarm",
                                    vmin=vmin,
                                    vmax=vmax,
                                    add_colorbar=False,
                                )
                            )

                            cbar = plt.colorbar(
                                im, ax=ax, orientation="vertical", pad=0.1, shrink=0.7
                            )
                            cbar.set_label(var, fontsize=10)

                            ax.set_extent(
                                [west, east, south, north], crs=ccrs.PlateCarree()
                            )
                            ax.add_feature(cfeature.BORDERS, linestyle=":")
                            ax.add_feature(cfeature.COASTLINE)
                            ax.add_feature(cfeature.LAND, edgecolor="black", alpha=0.3)
                            ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

                            plt.title(
                                f"{var} for month {month_str}, area {self.area_selection}"
                            )
                            plt.show()
                        else:
                            print(
                                f"Variable {var} not found in dataset for month {month_str}."
                            )

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
    leadtimes,
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
    leadtimes : list[int]
        List of leadtimes in hours.

    Returns
    -------
    Path
        Path to the downloaded data file.
    """
    try:

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
    tr_threshold=20,
    hw_threshold=27,
    hw_min_duration=3,
    hw_max_gap=0,
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
        "TR",
        "TX30",
        "HW",
    ]:
        ds_daily, ds_monthly, ds_stats = (
            seasonal_statistics.calculate_heat_indices_metrics(
                input_file_name,
                index_metric,
                tr_threshold=tr_threshold,
                hw_threshold=hw_threshold,
                hw_min_duration=hw_min_duration,
                hw_max_gap=hw_max_gap,
            )
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
