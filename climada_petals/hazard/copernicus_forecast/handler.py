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

Module to handle seasonal forecast data from the Copernicus Climate Data Store (CDS) in the U-CLIMADAPT project.

This module provides functionality for downloading, processing, calculating climate indices, and creating hazard objects
based on seasonal forecast data. It is designed to work with the CLIMADA (CLIMate ADAptation) platform for climate risk
assessment and adaptation strategies.

Features:
- Download seasonal forecast data from CDS
- Process raw data into climate indices
- Calculate various heat-related indices (e.g., Heat Index, Tropical Nights)
- Create CLIMADA Hazard objects for further risk analysis
- Visualize hazard data

Prerequisites:
1. CDS API client installation:
   pip install cdsapi

2. CDS account and API key:
   Register at https://cds-beta.climate.copernicus.eu

3. CDS API configuration:
   Create a .cdsapirc file in your home directory with your API key and URL.
   For instructions, visit: https://cds-beta.climate.copernicus.eu/how-to-api#install-the-cds-api-client

4. CLIMADA installation:
   Follow instructions at https://climada-python.readthedocs.io/en/stable/guide/install.html

Usage:
This module is typically imported and used within larger scripts or applications for climate data processing
and risk assessment. See individual function docstrings for specific usage instructions.

Note:
Ensure you have the necessary permissions and comply with CDS data usage policies when using this module.
"""

import os
import logging
import calendar

import xarray as xr
import pandas as pd
import numpy as np
import cdsapi

from climada.hazard import Hazard
from climada.util.coordinates import get_country_geometries
import climada_petals.hazard.copernicus_forecast.indicator as indicator


LOGGER = logging.getLogger(__name__)

class ForecastHandler:
    """
    A class to handle downloading, processing, and calculating climate indices
    and hazards based on seasonal forecast data from Copernicus Climate Data Store (CDS).
    """

    _FORMAT_GRIB = 'grib'
    _FORMAT_NC = 'nc'
    
    def __init__(self, data_dir='.', url = None, key = None):
        """
        Initializes the ForecastHandler instance.

        Parameters:
        data_dir (str): Path to the directory where downloaded and processed data will be stored.
            Defaults to the current directory ('.').
        url (str): url to the CDS API. Defaults to None, in which case the url from /.cdsapirc
            is used.
        key (str): CDS API key to the CDS API. Defaults to None, in which case the key from
            /.cdsapirc is used.

        Note:
        This method sets up logging and initializes the data directory for the instance.
        """
        logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
        self.logger = logging.getLogger()
        self.data_dir = data_dir
        self.key = key
        self.url = url
    
    @staticmethod
    def _get_bounds_for_area_selection(area_selection, margin=0.2):
        """
        Determines the geographic bounds based on an area selection string.

        Parameters:
        area_selection (str): Specifies the area for data selection.
        margin (float): Additional margin to be added to the bounds in degrees.

        Returns:
        list: A list of four floats representing the bounds [north, east, south, west].
        """
        if isinstance(area_selection, str):
            if area_selection.lower() == "global":
                return [90, -180, -90, 180]  # north, west, south, east
        else:
            # try if area was given in bounds
            try: 
                north, west, south, east = area_selection
                lat_margin = margin * (north - south)
                lon_margin = margin * (east - west)
                north += lat_margin
                east += lon_margin
                south -= lat_margin
                west -= lon_margin
                return [north, west, south, east]
            except:
                pass

            # check if countries are given 
            combined_bounds = [-90, 180, 90, -180]
            for iso in area_selection:
                geo = get_country_geometries(iso).to_crs(epsg=4326)
                bounds = geo.total_bounds
                if np.any(np.isnan(bounds)):
                    logging.warning(f"ISO code '{iso}' not recognized. This region will not be included." )

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

    def explain_index(self, tf_index):
        """
        Provides an explanation and input data for the selected index.

        Parameters:
        tf_index (str): The climate index identifier.

        Returns:
        str: A description of the selected index, its input data, or an error message if the index is invalid.
        """
        # Dictionary with explanations and input data for each index
        index_explanations = {
            "HIA": {
                "explanation": "Heat Index Adjusted: This indicator measures apparent temperature, considering both air temperature and humidity, providing a more accurate perception of how hot it feels.",
                "input_data": ["2m temperature (t2m)", "2m dewpoint temperature (d2m)"]
            },
            "HIS": {
                "explanation": "Heat Index Simplified: This indicator is a simpler version of the Heat Index, focusing on a quick estimate of perceived heat based on temperature and humidity.",
                "input_data": ["2m temperature (t2m)", "2m dewpoint temperature (d2m)"]
            },
            "Tmean": {
                "explanation": "Mean Temperature: This indicator calculates the average temperature over the specified period.",
                "input_data": ["2m temperature (t2m)"]
            },
            "Tmin": {
                "explanation": "Minimum Temperature: This indicator tracks the lowest temperature recorded over a specified period.",
                "input_data": ["2m temperature (t2m)"]
            },
            "Tmax": {
                "explanation": "Maximum Temperature: This indicator tracks the highest temperature recorded over a specified period.",
                "input_data": ["2m temperature (t2m)"]
            },
            "HW": {
                "explanation": "Heat Wave: This indicator identifies heat waves, defined as at least 3 consecutive days with temperatures exceeding a certain threshold.",
                "input_data": ["2m temperature (t2m)"]
            },
            "TR": {
                "explanation": "Tropical Nights: This indicator counts the number of nights where the minimum temperature remains above a certain threshold, typically 20°C.",
                "input_data": ["2m temperature (t2m)"]
            },
            "TX30": {
                "explanation": "Hot Days: This indicator counts the number of days where the maximum temperature exceeds 30°C.",
                "input_data": ["2m temperature (t2m)"]
            }
        }

        if tf_index in index_explanations:
            explanation = f"Selected Index: {tf_index}\n"
            explanation += f"Explanation: {index_explanations[tf_index]['explanation']}\n"
            explanation += f"Input Data: {', '.join(index_explanations[tf_index]['input_data'])}"
        else:
            explanation = f"Error: {tf_index} is not a valid index. Please choose from {list(self.index_explanations.keys())}"

        print(explanation)
        return None
    

    def _calc_min_max_lead(self, year, month, leadtime_months=1):
        """
        Calculates the minimum and maximum lead time in hours for a given starting (initidate.

        Parameters:
        year (int): The starting year (e.g., 2023).
        month (int): The starting month (1-12).
        leadtime_months (int): Number of months to include in the forecast period.

        Returns:
        tuple: (min_lead, max_lead) in hours.
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
        self, filename, vars, year, month, l_hours, area,
        overwrite, format, originating_centre, system
    ):
        """
        Downloads multiple climate variables over multiple lead times from the CDS.

        Parameters:
        filename (str): Full path and name for the downloaded file.
        vars (list of str): List of variable names to download.
        year (int): The forecast initialization year.
        month (int): The forecast initialization month.
        l_hours (list of int): List of lead times in hours to download.
        area (list of float): Geographic bounds [north, west, south, east].
        overwrite (bool): If True, overwrites existing files.
        format (str): File format for download, either 'grib' or 'nc'.
        originating_centre (str): The meteorological center producing the forecast.
        system (str): The forecast system version.

        Returns:
        None
        """
        # check if data already exists including all relevant data variables
        download_file = f'{filename}'
        data_already_exists = self._is_data_present(f'{download_file}', 'grib', vars)

        if data_already_exists and not overwrite:
            self.logger.info(f'Corresponding {format} file {download_file} already exists.')

        else:
            try:
                c = cdsapi.Client(url=self.url, key=self.key)
                c.retrieve(
                    'seasonal-original-single-levels',
                    {
                        'format': format,
                        'originating_centre': originating_centre,
                        'area': area,
                        'system': system,
                        'variable': vars,
                        'month': f"{month:02d}",
                        'year': year,
                        'day': '01',
                        'leadtime_hour': l_hours,
                    },
                    f'{download_file}'
                )
                self.logger.info(f'{format.capitalize()} file successfully downloaded to {download_file}.')
            except Exception as e:
                self.logger.error(f'{format.capitalize()} file {download_file} could not be downloaded. Error: {e}')

    def _download_data(
        self, data_out, year_list, month_list, bounds, overwrite, tf_index,
        format, originating_centre, system, max_lead_month
    ):
        """
        Downloads climate forecast data for specified years, months, and a climate index.

        Parameters:
        data_out (str): Base directory path for storing downloaded data.
        year_list (list of int): Years for which to download data.
        month_list (list of int): Months for which to download data.
        area_selection (str): Area specification.
        overwrite (bool): If True, overwrites existing files.
        tf_index (str): Climate index identifier.
        format (str): File format for download.
        originating_centre (str): The meteorological center producing the forecast.
        system (str): The forecast system version.
        max_lead_month (int): Maximum lead time in months to download.

        Returns:
        None
        """
        index_params = indicator.get_index_params(tf_index)
        area_str = f'{int(bounds[1])}_{int(bounds[0])}_{int(bounds[2])}_{int(bounds[3])}'

        for year in year_list:
            for month in month_list:
                out_dir = f"{data_out}/{format}/{year}/{month:02d}"
                os.makedirs(out_dir, exist_ok=True)

                min_lead, max_lead = self._calc_min_max_lead(year, month, max_lead_month)
                leadtimes = list(range(min_lead, max_lead, 6))
                self.logger.info(f"{len(leadtimes)} leadtimes to download.")
                self.logger.debug(f"which are: {leadtimes}")

                file_extension = 'grib' if format == self._FORMAT_GRIB else self._FORMAT_NC
                download_file = f"{out_dir}/{index_params['filename_lead']}_{area_str}_{year}{month:02d}.{file_extension}"

                if tf_index in ['HIS', 'HIA']:
                    variables = index_params['variables']
                else:
                    variables = [index_params['variables'][0]]
                
                self._download_multvar_multlead(
                    download_file, variables, year, month, leadtimes, bounds,
                    overwrite, format, originating_centre, system
                )

    def _process_data(self, data_out, year_list, month_list, bounds, overwrite, tf_index, format):
        """
        Processes the downloaded climate forecast data into daily average values.

        Parameters:
        data_out (str): Base directory path for storing processed output data.
        year_list (list of int): Years for which to process data.
        month_list (list of int): Months for which to process data.
        area_selection (str): Area specification.
        overwrite (bool): If True, overwrites existing processed files.
        tf_index (str): Climate index identifier being processed.
        format (str): File format of the downloaded data.

        Returns:
        None
        """
        index_params = indicator.get_index_params(tf_index)
        area_str = f'{int(bounds[1])}_{int(bounds[0])}_{int(bounds[2])}_{int(bounds[3])}'

        for year in year_list:
            for month in month_list:
                daily_out = f"{data_out}/netcdf/daily/{year}/{month:02d}"
                daily_file = f"{daily_out}/{index_params['filename_lead']}_{area_str}_{year}{month:02d}.nc"
                os.makedirs(daily_out, exist_ok=True)
                file_extension = 'grib' if format == self._FORMAT_GRIB else self._FORMAT_NC
                download_file = f"{data_out}/{format}/{year}/{month:02d}/{index_params['filename_lead']}_{area_str}_{year}{month:02d}.{file_extension}"
                
                # check if data already exists including all relevant data variables
                data_already_exists = self._is_data_present(
                    download_file, file_extension,
                    indicator.get_index_params(tf_index)['variables']
                )
                
                if not data_already_exists or overwrite:
                    try:
                        if format == self._FORMAT_GRIB:
                            with xr.open_dataset(download_file, engine="cfgrib") as ds:
                                ds_daily = ds.coarsen(step=4, boundary='trim').mean()
                        else:
                            with xr.open_dataset(download_file) as ds:
                                ds_daily = ds.coarsen(step=4, boundary='trim').mean()
                    except FileNotFoundError:
                        self.logger.error(f"{format.capitalize()} file does not exist, download failed.")
                        continue
                    ds_daily.to_netcdf(f"{daily_file}")
                else:
                    self.logger.info(f"Daily file {daily_file} already exists.")

    def download_and_process_data(
        self, data_out, year_list, month_list, area_selection, overwrite,
        tf_index, format, originating_centre, system, max_lead_month
    ):
        """
        Downloads and processes climate forecast data for specified parameters.

        Parameters:
        data_out (str): Base directory path for storing data.
        year_list (list of int): Years for which to download and process data.
        month_list (list of int): Months for which to download and process data.
        area_selection (str): Area specification.
        overwrite (bool): If True, overwrites existing files.
        tf_index (str): Climate index identifier to be processed.
        format (str): File format for download and processing.
        originating_centre (str): The meteorological center producing the forecast.
        system (str): The forecast system version.
        max_lead_month (int): Maximum lead time in months.

        Returns:
        None
        """

        bounds = self._get_bounds_for_area_selection(area_selection)
        self._download_data(data_out, year_list, month_list, bounds, overwrite, tf_index, format, originating_centre, system, max_lead_month)
        self._process_data(data_out, year_list, month_list, bounds, overwrite, tf_index, format)

    def calculate_index(
        self, data_out, year_list, month_list, area_selection, overwrite, tf_index
    ):
        """
        Calculates the specified climate index for given years and months.

        Parameters:
        data_out (str): Base directory path for output data.
        year_list (list of int): Years for which to calculate the index.
        month_list (list of int): Months for which to calculate the index (1-12).
        area_selection (str): Area specification.
        overwrite (bool): If True, overwrites existing files.
        tf_index (str): The climate index to be calculated.
        """
        bounds = self._get_bounds_for_area_selection(area_selection)
        area_str = f"{int(bounds[1])}_{int(bounds[0])}_{int(bounds[2])}_{int(bounds[3])}"
        index_params = indicator.get_index_params(tf_index)

        for year in year_list:
            for month in month_list:
                # path to input file of daily variables
                input_file_name = f"{data_out}/netcdf/daily/{year}/{month:02d}" \
                    f'/{index_params["filename_lead"]}_{area_str}_{year}{month:02d}.nc'
                grib_file_name = f"{data_out}/grib/{year}/{month:02d}" \
                    f"/{index_params['filename_lead']}_{area_str}_{year}{month:02d}.grib"
                
                # paths to output files
                output_dir = f"{data_out}/{tf_index}/{year}/{month:02d}"
                output_daily_path = f'{output_dir}/daily_{tf_index}_{area_str}_{year}{month:02d}.nc'
                output_stats_path = f'{output_dir}/stats/stats_{tf_index}_{area_str}_{year}{month:02d}.nc'
                output_monthly_path = f'{output_dir}/{tf_index}_{area_str}_{year}{month:02d}.nc'
                os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)

                # check if index (monthly) file exists
                if os.path.exists(output_monthly_path) and not overwrite:
                    self.logger.info(
                        f'Index file {tf_index}_{area_str}_{year}{month:02d}.nc already exists.'
                    )

                else:
                    if tf_index in ["HIS", "HIA", "Tmean", "Tmax", "Tmin"]:
                        # Calculate heat indices like HIS, HIA, Tmean, Tmax, Tmin
                        ds_daily, ds_monthly, ds_stats = indicator.calculate_heat_indices(input_file_name, tf_index)

                    elif tf_index == "TR":
                        # Handle Tropical Nights (TR)
                        ds_daily, ds_monthly, ds_stats = indicator.calculate_and_save_tropical_nights_per_lag(grib_file_name, tf_index)

                    elif tf_index == "TX30":
                        # Handle Hot Days (Tmax > 30°C)
                        ds_daily, ds_monthly, ds_stats = indicator.calculate_and_save_tx30_per_lag(grib_file_name, tf_index)
                    
                    # TODO: add functionality
                    # elif tf_index == "HW":
                        # Handle Heat Wave Days (3 consecutive days Tmax > threshold)
                        # calculate_and_save_heat_wave_days_per_lag(data_out, year_list, month_list, tf_index, area_selection)

                    else:
                        logging.error(f"Index {tf_index} is not implemented. Supported indices are 'HIS', 'HIA', 'Tmean', 'Tmax', 'Tmin', 'HotDays', 'TR', and 'HW'.")

                    # save files
                    self.logger.info(f"Writing index data to {output_monthly_path}.")
                    if tf_index in ["HIS", "HIA", "Tmean", "Tmax", "Tmin"]:
                        ds_daily.to_netcdf(output_daily_path)
                    ds_monthly.to_netcdf(output_monthly_path)
                    ds_stats.to_netcdf(output_stats_path)

    def save_index_to_hazard(self, year_list, month_list, area_selection, data_out, overwrite, tf_index):
        """
        Processes the calculated climate indices into hazard objects and saves them.

        This method converts the NetCDF files of climate indices into CLIMADA Hazard objects,
        which can be used for further risk analysis.
        """

        bounds = self._get_bounds_for_area_selection(area_selection)
        area_str = f"{int(bounds[1])}_{int(bounds[0])}_{int(bounds[2])}_{int(bounds[3])}"

        base_input_dir = os.path.join(data_out, tf_index)
        base_output_dir = os.path.join(data_out, f"{tf_index}/hazard")
        hazard_type = tf_index
        intensity_variable = f"{tf_index}"

        if tf_index in ["TR", "TX30", "HW"]:
            intensity_unit = "days"
        else:
            intensity_unit = "°C"

        for year in year_list:
            for month in month_list:

                month_str = f"{month:02d}"
                current_input_dir = f'{base_input_dir}/{year}/{month:02d}'
                nc_file_pattern = f"{hazard_type}_{area_str}_{year}{month_str}.nc"
                nc_file_path = os.path.join(current_input_dir, nc_file_pattern)
                current_output_dir = os.path.join(base_output_dir, f"{year}{month_str}")
                os.makedirs(current_output_dir, exist_ok=True)
                try:
                    ds = xr.open_dataset(nc_file_path)

                    ds["step"] = xr.DataArray([f"{date}-01" for date in ds["step"].values], dims=["step"])
                    ds["step"] = pd.to_datetime(ds["step"].values)

                    ensemble_members = ds["number"].values

                    for member in ensemble_members:
                        # check if data already exists
                        filename = f"hazard_{hazard_type}_member_{member}_{area_str}_{year}{month_str}.hdf5"
                        file_path = os.path.join(current_output_dir, filename)
                        if os.path.exists(file_path) and not overwrite:
                            self.logger.info(f'Index file ' \
                                f'{tf_index}_{area_str}_{year}{month:02d}.nc already exists.')
                        else:
                            ds_subset = ds.sel(number=member)
                            hazard = Hazard.from_xarray_raster(
                                data=ds_subset,
                                hazard_type=hazard_type,
                                intensity_unit=intensity_unit,
                                intensity=intensity_variable,
                                coordinate_vars={"event": "step", "longitude": "longitude", "latitude": "latitude"}
                            )

                            hazard.check()
                            hazard.write_hdf5(file_path)

                    print(f"Completed processing for {year}-{month_str}. Data saved in {current_output_dir}")

                except FileNotFoundError as e:
                    print(f"File not found: {e.filename}")
                except Exception as e:
                    print(f"An error occurred: {e}")

        # print final hazard
        last_hazard_file = file_path
        hazard_obj = Hazard.from_hdf5(last_hazard_file)
        hazard_obj.plot_intensity(1, smooth=False)
    
    @staticmethod
    def _is_data_present(file, format, vars):
        data_already_exists = os.path.isfile(file)
        if format == 'grib':
            engine = 'cfgrib'
        else:
            engine = None
        if data_already_exists:
            existing_variables = list(
                xr.open_dataset(file, engine=engine, decode_cf=False, chunks={}
            ).data_vars)
            vars_short = [indicator.VAR_SPECS[var]['short_name'] for var in vars]
            data_already_exists = set(vars_short).issubset(existing_variables)
        return data_already_exists
        