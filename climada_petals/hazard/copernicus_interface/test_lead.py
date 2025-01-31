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
from typing import List

#from climada.hazard import Hazard
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
        bounds,
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
        valid_period : list[str or int]
            A list of start and end month (given as integers or strings) of the valid period. Must have
            length two. If only one month is requested, use e.g. ["March", "March"].
        initiation_month : list[str]
            Initiation months for the forecasts (e.g., ["March", "April"]).
        bounds : list
            bounding box values (in EPSG 4326) in the order (min_lon, min_lat, max_lon, max_lat) or (west, south, east, north]).
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
        self.bounds = bounds
        self.bounds_str = f"boundsW{int(self.bounds[0])}_S{int(self.bounds[1])}_E{int(self.bounds[2])}_N{int(self.bounds[3])}"
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

    def check_existing_files(
        self,
        *,
        index_metric: str,
        year: int,
        initiation_month: str,
        valid_period: List[str],
        download_format="grib",
    ):

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
            self.get_file_path(
                self.data_out,
                self.originating_centre,
                year,
                initiation_month_str,
                valid_period_str,
                data_type,
                index_metric,
                self.bounds_str,
                format=download_format,
            )
            for data_type in ["downloaded_data", "processed_data", "indices", "hazard"]
        ]

        if not downloaded_data_path.exists():
            print("No downloaded data found for given time periods.")
        else:
            print(f"Downloaded data exist at: {downloaded_data_path}")
        if not processed_data_path.exists():
            print("No processed data found for given time periods.")
        else:
            print(f"Processed data exist at: {processed_data_path}")
        if not any([path.exists() for path in index_data_paths.values()]):
            print("No index data found for given time periods.")
        else:
            print(f"Index data exist at: {index_data_paths}")
        if not hazard_data_path.exists():
            print("No hazard data found for given time periods.")
        else:
            print(f"Hazard data exist at: {hazard_data_path}")

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
        bounds_str,
        format="grib",
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
            return {
                timeframe: Path(
                    f"{sub_dir}/{index_metric}_{bounds_str}_{timeframe}.{format}"
                )
                for timeframe in ["daily", "monthly", "stats"]
            }
        else:
            return Path(f"{sub_dir}/{index_metric}_{bounds_str}.{format}")

    def get_pipeline_path(self, year, initiation_month_str, data_type):
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
            self.bounds_str,
            self.format,
        )

        # create directory if not existing
        if data_type == "indices":
            file_path["monthly"].parent.mkdir(parents=True, exist_ok=True)
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        return file_path

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
        bounds_CDS_order = [
            self.bounds[3],
            *self.bounds[:3],
        ]  # convert bounds to CDS order
        for year in self.year_list:
            for month_str in self.initiation_month_str:
                leadtimes = calculate_leadtimes(year, int(month_str), self.valid_period)

                # Generate output file name
                downloaded_data_path = self.get_pipeline_path(
                    year, month_str, "downloaded_data"
                )

                output_files[f"{year}_init{month_str}_valid{self.valid_period_str}"] = (
                    _download_data(
                        downloaded_data_path,
                        overwrite,
                        self.variables,
                        year,
                        month_str,
                        self.format,
                        self.originating_centre,
                        self.system,
                        bounds_CDS_order,
                        leadtimes,
                    )
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
                downloaded_data_path = self.get_pipeline_path(
                    year, month_str, "downloaded_data"
                )
                # Generate output file name
                processed_data_path = self.get_pipeline_path(
                    year, month_str, "processed_data"
                )

                processed_files[
                    f"{year}_init{month_str}_valid{self.valid_period_str}"
                ] = _process_data(
                    processed_data_path,
                    overwrite,
                    downloaded_data_path,
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
                    input_data_path = self.get_pipeline_path(
                        year, month_str, "downloaded_data"
                    )
                else:  # Metrics using processed NC files
                    input_data_path = self.get_pipeline_path(
                        year, month_str, "processed_data"
                    )

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
                    index_outputs[
                        f"{year}_init{month_str}_valid{self.valid_period_str}"
                    ] = outputs

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
                index_data_path = self.get_pipeline_path(year, month_str, "indices")[
                    "monthly"
                ]
                hazard_data_path = self.get_pipeline_path(year, month_str, "hazard")

                try:
                    # Convert index file to Hazard
                    hazard_outputs[
                        f"{year}_init{month_str}_valid{self.valid_period_str}"
                    ] = _convert_to_hazard(
                        hazard_data_path,
                        overwrite,
                        index_data_path,
                        self.index_metric,
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

        # Iterate over initiation months and access the corresponding file
        for month_str in self.initiation_month_str:

            # Construct the file name and path
            file_path = base_path / file_name_pattern.format(month=month_str)

            if not file_path.exists():
                print(f"Skill data file for month {month_str} not found: {file_path}")
                continue

            # Load the data using xarray
            try:
                with xr.open_dataset(file_path) as ds:
                    # Subset the dataset by area bounds
                    west, south, east, north = self.bounds
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
                                f"{var} for month {month_str},  {self.bounds_str}"
                            )
                            plt.show()
                        else:
                            print(
                                f"Variable {var} not found in dataset for month {month_str}."
                            )

            except Exception as e:
                print(f"Failed to load or process data for month {month_str}: {e}")


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
    if isinstance(month, int): # Already a number
        if 1 <= month <= 12:
            return month
        else:
            raise ValueError("Month number must be between 1 and 12.")
    if isinstance(month, str):
        if not month.strip():
            # e.g. "" or "   "
            raise ValueError("Month cannot be empty.")
        month = month.capitalize() # Ensure consistent capitalization

        if month in calendar.month_name:
            return list(calendar.month_name).index(month)  # e.g., 'March' -> 3
        elif month in calendar.month_abbr:
            return list(calendar.month_abbr).index(month)  # e.g., 'Mar' -> 3

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
    valid_period : List[int or str]
        A list of start and end month (given as integers or strings) of the valid period. Must have
        length two. If only one month is requested, use e.g. ["March", "March"].
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

    Notes
    -----
    - The function determines the correct year(s) for the valid forecast period.
    - If the forecast extends into the next year, the valid period spans two years.
    - Lead times are computed in hourly intervals from the initiation month to the end of the forecast period.

    Example:
    ---------
    If the forecast is initiated in **December 2022** and the valid period is **January to February 2023**,
    the function will:
    - Recognize that the forecast extends into the next year (2023).
    - Compute lead times starting from **December 1, 2022** (0 hours) to **February 28, 2023**.
    - Generate lead times in 6-hour intervals, covering the entire forecast period from December 2022 through February 2023.

    """

    # Convert initiation month to numeric if it is a string
    if isinstance(initiation_month, str):
        initiation_month = month_name_to_number(initiation_month)

    # Convert valid_period to numeric
    valid_period = [
        month_name_to_number(month) if isinstance(month, str) else month
        for month in valid_period
    ]

        # We expect valid_period = [start, end]
    start_month, end_month = valid_period

    # Immediately check for reversed period
    if end_month < start_month:
        raise ValueError(
            "Reversed valid_period detected. The forecast cannot be called with "
            f"an end month ({end_month}) that is before the start month ({start_month})."
        )
    
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
            (valid_period_end - initiation_date).days * 24 +24,
            6,
        )
    )



# --- Print lead times for each combination ---
from datetime import datetime, timedelta
# Example lists for demonstration:
year_list = [2022]
initiation_month = ["January"]
valid_period = [ "April", "March"]

for y in year_list:
    for init_mth in initiation_month:
        # Get the raw hour offsets from your existing function
        leadtimes = calculate_leadtimes(y, init_mth, valid_period)

        # The reference date for hour=0 is the 1st day of the initiation month
        ref_datetime = datetime(y, month_name_to_number(init_mth), 1, 0, 0)

        print(f"Lead times for year={y}, init={init_mth}, valid={valid_period}:")

        # 1) Print the raw hour offsets
        print("Hour offsets:", leadtimes)

        # 2) Convert each offset to a datetime
        dt_list = [ref_datetime + timedelta(hours=offset) for offset in leadtimes]

        print("Corresponding datetimes:")
        for dt in dt_list:
            print(dt.isoformat())
