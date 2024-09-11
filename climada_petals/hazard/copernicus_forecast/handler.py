#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
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
   Register at https://cds.climate.copernicus.eu/#!/home

3. CDS API configuration:
   Create a .cdsapirc file in your home directory with your API key and URL.
   For instructions, visit: https://cds.climate.copernicus.eu/api-how-to

4. CLIMADA installation:
   Follow instructions at https://climada-python.readthedocs.io/en/stable/guide/install.html

Usage:
This module is typically imported and used within larger scripts or applications for climate data processing
and risk assessment. See individual function docstrings for specific usage instructions.

Note:
Ensure you have the necessary permissions and comply with CDS data usage policies when using this module.
"""


# Module_forescast_handler.py
import os
import logging
import calendar
import cdsapi
import xarray as xr
import pandas as pd
from climada.util.coordinates import country_to_iso, get_country_geometries
from climada import CONFIG
from climada.hazard import Hazard
import numpy as np




class ForecastHandler:
    """
    A class to handle downloading, processing, and calculating climate indices
    and hazards based on seasonal forecast data from Copernicus Climate Data Store (CDS).
    """

    FORMAT_GRIB = 'grib'
    FORMAT_NC = 'nc'

    
    def __init__(self, data_dir='.', URL = None, KEY = None):
        """
        Initializes the ForecastHandler instance.

        Parameters:
        data_dir (str): Path to the directory where downloaded and processed data will be stored.
                        Defaults to the current directory ('.').

        Note:
        This method sets up logging and initializes the data directory for the instance.
        """
        logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
        self.logger = logging.getLogger()
        self.data_dir = data_dir
        self.KEY = KEY
        self.URL = URL
        

    @staticmethod
    def get_index_params(index):
        """
        Retrieves parameters associated with a specific climate index.

        Parameters:
        index (str): The climate index identifier. Supported values include:
                     'HIA' (Heat Index Adjusted), 'HIS' (Heat Index Simplified),
                     'Tmean' (Mean Temperature), 'Tmin' (Minimum Temperature),
                     'Tmax' (Maximum Temperature), 'HW' (Heat Wave), 'TR' (Tropical Nights).

        Returns:
        dict: A dictionary containing the parameters for the specified index, including:
              - 'variables': List of required variables for the index calculation.
              - 'filename_lead': Prefix for filenames related to this index.
              - 'index_long_name': Full descriptive name of the index.

        If the index is not recognized, returns None.
        """
        var_specs = {
            "2m_temperature": {
                "unit": "K",
                "standard_name": "air_temperature",
                "short_name": "t2m",
                "full_name": "2m_temperature"
            },
            "2m_dewpoint_temperature": {
                "unit": "K",
                "standard_name": "dew_point_temperature",
                "short_name": "d2m",
                "full_name": "2m_dewpoint_temperature"
            },
        }

        index_params = {
            "HIA": {
                "variables": [var_specs["2m_temperature"]["full_name"], var_specs["2m_dewpoint_temperature"]["full_name"]],
                "filename_lead": "2m_temps",
                "index_long_name": "Heat_Index_Adjusted"
            },
            "HIS": {
                "variables": [var_specs["2m_temperature"]["full_name"], var_specs["2m_dewpoint_temperature"]["full_name"]],
                "filename_lead": "2m_temps",
                "index_long_name": "Heat_Index_Simplified"
            },
            "Tmean": {
                "variables": [var_specs["2m_temperature"]["full_name"]],
                "filename_lead": "2m_temps",
                "index_long_name": "Mean_Temperature"
            },
            "Tmin": {
                "variables": [var_specs["2m_temperature"]["full_name"]],
                "filename_lead": "2m_temps",
                "index_long_name": "Minimum_Temperature"
            },
            "Tmax": {
                "variables": [var_specs["2m_temperature"]["full_name"]],
                "filename_lead": "2m_temps",
                "index_long_name": "Maximum_Temperature"
            },
            
            "HW": {
                "variables": [var_specs["2m_temperature"]["full_name"]],
                "filename_lead": "2m_temps",
                "index_long_name": "Heat_Wave"
            },
            "TR": {
                "variables": [var_specs["2m_temperature"]["full_name"]],
                "filename_lead": "2m_temps",
                "index_long_name": "Tropical_Nights"
            }
        }
        return index_params.get(index)

    @staticmethod
    def calc_min_max_lead(year, month, leadtime_months=1):
        """
        Calculates the minimum and maximum lead time in hours for a given starting date.

        Parameters:
        year (int): The starting year (e.g., 2023).
        month (int): The starting month (1-12, where 1 is January and 12 is December).
        leadtime_months (int): Number of months to include in the forecast period. Defaults to 1.

        Returns:
        tuple: A tuple containing two integers:
               - min_lead (int): The minimum lead time in hours (always 0).
               - max_lead (int): The maximum lead time in hours, calculated based on the
                                 number of days in the specified months plus 6 hours.

        Note:
        This function accounts for varying month lengths and year transitions.
        """
        min_lead = 0
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
        return min_lead, max_lead

    @staticmethod
    def get_bounds_for_area_selection(area_selection, margin=0.2):
        """
        Determines the geographic bounds based on an area selection string.

        Parameters:
        area_selection (str): Specifies the area for data selection. Can be one of:
                              - 'global' for worldwide coverage
                              - A comma-separated string of coordinates (north,east,south,west)
                              - One or more country names separated by commas
        margin (float): Additional margin to be added to the bounds in degrees. 
                        Defaults to 0.2 degrees. Ignored for 'global' selection.

        Returns:
        list: A list of four floats representing the bounds [north, east, south, west].
              For 'global', returns [90, -180, -90, 180].
              For specific areas, returns the calculated bounds with added margins.

        Note:
        For country names, the function uses external utilities to convert names to 
        ISO codes and fetch geometries. The bounds are then calculated to encompass
        all specified countries with the added margin.
        """        
        if area_selection.lower() == 'global':
            return [90, -180, -90, 180]  # north, east, south, west
        else:
            try:
                user_bounds = list(map(float, area_selection.split(',')))
                if len(user_bounds) == 4:
                    north, east, south, west = user_bounds
                    north += margin
                    east -= margin
                    south -= margin
                    west += margin
                    return [north, east, south, west]
            except ValueError:
                pass

            countries = area_selection.split(",")
            combined_bounds = [180, 90, -180, -90]
            for country in countries:
                iso = country_to_iso(country.strip())
                geo = get_country_geometries(iso).to_crs(epsg=4326)
                bounds = geo.total_bounds
                min_lon, min_lat, max_lon, max_lat = bounds

                lat_margin = margin * (max_lat - min_lat)
                lon_margin = margin * (max_lon - min_lon)

                combined_bounds[0] = min(combined_bounds[0], min_lon - lon_margin)
                combined_bounds[1] = min(combined_bounds[1], min_lat - lat_margin)
                combined_bounds[2] = max(combined_bounds[2], max_lon + lon_margin)
                combined_bounds[3] = max(combined_bounds[3], max_lat + lat_margin)
            return [combined_bounds[3], combined_bounds[0], combined_bounds[1], combined_bounds[2]]

    def download_multvar_multlead(self, filename, vars, year, month, l_hours, area, overwrite, format, originating_centre, system):
        """
        Downloads multiple climate variables over multiple lead times from the Copernicus Climate Data Store (CDS).

        Parameters:
        filename (str): Full path and name for the downloaded file.
        vars (list of str): List of variable names to download (e.g., ['2m_temperature', '2m_dewpoint_temperature']).
        year (int): The forecast initialization year (e.g., 2023).
        month (int): The forecast initialization month (1-12, where 1 is January).
        l_hours (list of int): List of lead times in hours to download (e.g., [0, 6, 12, 18, 24]).
        area (list of float): Geographic bounds [north, west, south, east] in degrees.
        overwrite (bool): If True, overwrites existing files; if False, skips download if file exists.
        format (str): File format for download, either 'grib' or 'nc' (NetCDF).
        originating_centre (str): The meteorological center producing the forecast (e.g., 'ecmwf', 'dwd').
        system (str): The forecast system version (e.g., '5', '51').

        Returns:
        None

        Raises:
        Exception: If the download fails, an error message is logged.

        Note:
        This method uses the CDS API to retrieve data. Ensure you have proper credentials set up.
        """
        area_str = f'{int(area[1])}_{int(area[0])}_{int(area[2])}_{int(area[3])}'
        download_file = f'{filename}'
        if os.path.isfile(f'{download_file}') and not overwrite:
            self.logger.info(f'Corresponding {format} file {download_file} already exists.')
        else:
            try:
                c = cdsapi.Client(url=self.URL, key=self.KEY)
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
                    f'{download_file}')
                self.logger.info(f'{format.capitalize()} file successfully downloaded to {download_file}.')
            except Exception as e:
                self.logger.error(f'{format.capitalize()} file {download_file} could not be downloaded. Error: {e}')

    @staticmethod
    def create_directories(data_out, tf_index):
        """
        Creates necessary directories for storing output data for a specific climate index.

        Parameters:
        data_out (str): Base directory path where output data will be stored.
        tf_index (str): The climate index identifier (e.g., 'HIA', 'Tmean').

        Returns:
        str: Full path to the index-specific output directory.

        Note:
        This method creates both the base directory and a subdirectory for the specific index.
        If the directories already exist, it doesn't raise an error.
        """
        os.makedirs(data_out, exist_ok=True)
        index_out = f"{data_out}/{tf_index}"
        os.makedirs(index_out, exist_ok=True)
        return index_out

    def download_data(self, data_out, year_list, month_list, area_selection, overwrite, tf_index, format, originating_centre, system, max_lead_month):
        """
        Downloads climate forecast data for specified years, months, and a climate index.

        Parameters:
        data_out (str): Base directory path for storing downloaded data.
        year_list (list of int): Years for which to download data (e.g., [2022, 2023]).
        month_list (list of int): Months for which to download data (1-12, where 1 is January).
        area_selection (str): Area specification, can be 'global', coordinates, or country names.
        overwrite (bool): If True, overwrites existing files; if False, skips existing files.
        tf_index (str): Climate index identifier (e.g., 'HIA', 'Tmean').
        format (str): File format for download, either 'grib' or 'nc' (NetCDF).
        originating_centre (str): The meteorological center producing the forecast (e.g., 'ecmwf', 'dwd').
        system (str): The forecast system version (e.g., '5', '51').
        max_lead_month (int): Maximum lead time in months to download.

        Returns:
        None

        Note:
        This method organizes downloads by year and month, creating appropriate directory structures.
        It handles different requirements for various climate indices, especially for heat-related indices.
        """
        index_params = self.get_index_params(tf_index)
        area = self.get_bounds_for_area_selection(area_selection)
        area_str = f'{int(area[1])}_{int(area[0])}_{int(area[2])}_{int(area[3])}'

        for year in year_list:
            for month in month_list:
                out_dir = f"{data_out}/{format}/{year}/{month:02d}"
                os.makedirs(out_dir, exist_ok=True)

                min_lead, max_lead = self.calc_min_max_lead(year, month, max_lead_month)
                leadtimes = list(range(min_lead, max_lead, 6))
                self.logger.info(f"{len(leadtimes)} leadtimes to download.")
                self.logger.debug(f"which are: {leadtimes}")

                file_extension = 'grib' if format == self.FORMAT_GRIB else 'nc'
                download_file = f"{out_dir}/{index_params['filename_lead']}_{area_str}_{year}{month:02d}.{file_extension}"

                if tf_index in ['HIS', 'HIA']:
                    variables = index_params['variables']  # Both t2m and d2m
                else:
                    variables = [index_params['variables'][0]]  # Only t2m
                
                self.download_multvar_multlead(
                    download_file, variables, year, month, leadtimes, area,
                    overwrite, format, originating_centre, system
                )

    def process_data(self, data_out, year_list, month_list, area_selection, overwrite, tf_index, format):
        """
        Processes the downloaded climate forecast data into daily average values.

        Parameters:
        data_out (str): Base directory path for storing processed output data.
        year_list (list of int): Years for which to process data (e.g., [2022, 2023]).
        month_list (list of int): Months for which to process data (1-12, where 1 is January).
        area_selection (str): Area specification, can be 'global', coordinates, or country names.
        overwrite (bool): If True, overwrites existing processed files; if False, skips processing if file exists.
        tf_index (str): Climate index identifier (e.g., 'HIA', 'Tmean') being processed.
        format (str): File format of the downloaded data, either 'grib' or 'nc' (NetCDF).

        Returns:
        None

        Notes:
        - This method converts sub-daily data to daily averages.
        - Processed data is saved in NetCDF format, regardless of the input format.
        - The method creates a directory structure organized by year and month for the processed files.
        - If processing fails due to missing input files, an error is logged, and the method continues with the next file.
        """
        index_params = self.get_index_params(tf_index)
        area = self.get_bounds_for_area_selection(area_selection)
        area_str = f'{int(area[1])}_{int(area[0])}_{int(area[2])}_{int(area[3])}'

        for year in year_list:
            for month in month_list:
                daily_out = f"{data_out}/netcdf/daily/{year}/{month:02d}"
                daily_file = f"{daily_out}/{index_params['filename_lead']}_{area_str}_{year}{month:02d}.nc"
                os.makedirs(daily_out, exist_ok=True)
                file_extension = 'grib' if format == self.FORMAT_GRIB else 'nc'
                download_file = f"{data_out}/{format}/{year}/{month:02d}/{index_params['filename_lead']}_{area_str}_{year}{month:02d}.{file_extension}"

                if not os.path.exists(daily_file) or overwrite:
                    try:
                        if format == self.FORMAT_GRIB:
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
                    ds_daily = xr.load_dataset(daily_file)

    def download_and_process_data(self, data_out, year_list, month_list, area_selection, overwrite, tf_index, format, originating_centre, system, max_lead_month):
        """
        Downloads and processes climate forecast data for specified parameters.

        This method combines the downloading and processing steps into a single operation.

        Parameters:
        data_out (str): Base directory path for storing both downloaded and processed data.
        year_list (list of int): Years for which to download and process data (e.g., [2022, 2023]).
        month_list (list of int): Months for which to download and process data (1-12, where 1 is January).
        area_selection (str): Area specification, can be 'global', coordinates, or country names.
        overwrite (bool): If True, overwrites existing files; if False, skips existing files.
        tf_index (str): Climate index identifier (e.g., 'HIA', 'Tmean') to be processed.
        format (str): File format for download and processing, either 'grib' or 'nc' (NetCDF).
        originating_centre (str): The meteorological center producing the forecast (e.g., 'ecmwf', 'dwd').
        system (str): The forecast system version (e.g., '5', '51').
        max_lead_month (int): Maximum lead time in months to download and process.

        Returns:
        None

        Notes:
        - This method first calls `download_data` to retrieve the forecast data.
        - It then calls `process_data` to convert the downloaded data into daily averages.
        - The entire process is performed for each combination of year and month in the provided lists.
        """
        self.download_data(data_out, year_list, month_list, area_selection, overwrite, tf_index, format, originating_centre, system, max_lead_month)
        self.process_data(data_out, year_list, month_list, area_selection, overwrite, tf_index, format)

   
    @staticmethod
    def calculate_relative_humidity_percent(t2k, tdk):
        """
        Calculates the relative humidity percentage from temperature and dewpoint temperature.

        Parameters:
        t2k (float or array-like): 2-meter temperature in Kelvin.
        tdk (float or array-like): 2-meter dewpoint temperature in Kelvin.

        Returns:
        float or array-like: Relative humidity as a percentage.

        Notes:
        - This method uses the August-Roche-Magnus approximation for saturation vapor pressure.
        - The calculation is valid for temperatures between -45°C and 60°C.
        - Input temperatures are converted from Kelvin to Celsius within the function.
        - The output is capped between 0% and 100% to avoid unrealistic values due to numerical imprecision.
        - If dewpoint temperature exceeds air temperature, relative humidity is capped at 100%.

        Formula used:
        es = 6.11 * 10^((7.5 * T) / (237.3 + T))  # Saturation vapor pressure
        e = 6.11 * 10^((7.5 * Td) / (237.3 + Td))  # Actual vapor pressure
        RH = (e / es) * 100  # Relative humidity percentage

        Where:
        T: Temperature in Celsius
        Td: Dewpoint temperature in Celsius
        """

        # Convert temperatures from Kelvin to Celsius
        t2c = t2k - 273.15
        tdc = tdk - 273.15

        # Compute the saturation vapor pressure and actual vapor pressure
        es = 6.11 * 10.0 ** (7.5 * t2c / (237.3 + t2c))
        e = 6.11 * 10.0 ** (7.5 * tdc / (237.3 + tdc))

        # Calculate relative humidity
        rh = (e / es) * 100

        # Clip RH values between 0% and 100% to avoid unrealistic results
        rh = np.clip(rh, 0, 100)

        # Return the relative humidity percentage
        return rh


    @staticmethod
    def calculate_heat_index_simplified(t2k, tdk):
        """
        Calculates the simplified heat index based on temperature and dewpoint temperature.

        Parameters:
        t2k (float or array-like): 2-meter temperature in Kelvin.
        tdk (float or array-like): 2-meter dewpoint temperature in Kelvin.

        Returns:
        float or array-like: Simplified heat index in degrees Celsius.

        Notes:
        - This method uses a polynomial regression equation to estimate the heat index.
        - The equation is valid for temperatures above 20°C (68°F) and relative humidity above 40%.
        - Input temperatures are converted from Kelvin to Celsius within the function.
        - The simplified version is less accurate but computationally faster than the adjusted version.

        Formula:
        HI = -8.784695 + 1.61139411*T + 2.338549*RH + 0.14611605*T*RH 
             - 1.2308094e-2*T^2 - 1.6424828e-2*RH^2 + 2.211732e-3*T^2*RH 
             + 7.2546e-4*T*RH^2 - 3.582e-6*T^2*RH^2

        Where:
        T: Temperature in Celsius
        RH: Relative Humidity in percent
        """

        # Convert temperatures from Kelvin to Celsius
        t2c = t2k - 273.15

        # Calculate the relative humidity
        rh = ForecastHandler.calculate_relative_humidity_percent(t2k, tdk)

        # Check if any temperatures are below or equal to 20°C, where the heat index formula is invalid
        if np.any(t2c <= 20):
            logging.warning("Heat Index Simplified is only valid for temperatures above 20°C.")
            return t2c  # Return the temperature as the heat index for non-valid cases

        # Simplified heat index formula
        hi = (
            -8.784695 + 1.61139411 * t2c + 2.338549 * rh + 0.14611605 * t2c * rh +
            -1.2308094e-2 * t2c ** 2 + -1.6424828e-2 * rh ** 2 +
            2.211732e-3 * t2c ** 2 * rh + 7.2546e-4 * t2c * rh ** 2 +
            -3.582e-6 * t2c ** 2 * rh ** 2
        )

        return hi

    @staticmethod
    def calculate_heat_index_adjusted(t2k, tdk):
        """
        Calculates the adjusted heat index based on temperature and dewpoint temperature.

        Parameters:
        t2k (float or array-like): 2-meter temperature in Kelvin.
        tdk (float or array-like): 2-meter dewpoint temperature in Kelvin.

        Returns:
        float or array-like: Adjusted heat index in degrees Celsius.

        Notes:
        - This method uses the Rothfusz regression equation for a more accurate heat index calculation.
        - The equation is valid for temperatures above 26.7°C (80°F) and relative humidity above 40%.
        - Input temperatures are converted from Kelvin to Fahrenheit for calculation, then back to Celsius.
        - This version is more accurate but computationally more intensive than the simplified version.

        Formula:
        HI = -42.379 + 2.04901523*T + 10.14333127*RH - 0.22475541*T*RH 
             - 0.00683783*T^2 - 0.05481717*RH^2 + 0.00122874*T^2*RH 
             + 0.00085282*T*RH^2 - 0.00000199*T^2*RH^2

        Where:
        T: Temperature in Fahrenheit
        RH: Relative Humidity in percent
        """
        rh = ForecastHandler.calculate_relative_humidity_percent(t2k, tdk)
        t2f = (t2k - 273.15) * 9/5 + 32
        hi = (
            -42.379 + 2.04901523 * t2f + 10.14333127 * rh +
            -0.22475541 * t2f * rh + -0.00683783 * t2f ** 2 +
            -0.05481717 * rh ** 2 + 0.00122874 * t2f ** 2 * rh +
            0.00085282 * t2f * rh ** 2 + -0.00000199 * t2f ** 2 * rh ** 2
        )
        return (hi - 32) * 5/9  # convert back to Celsius

    @staticmethod
    def calculate_heat_index(da_t2k, da_tdk, index):
        if index == 'HIS':
            index_long_name = 'heat_index_simplified'
            unit = 'degC'
            tf_index_data = ForecastHandler.calculate_heat_index_simplified(da_t2k.data, da_tdk.data)
        elif index == 'HIA':
            index_long_name = 'heat_index_adjusted'
            unit = 'degC'
            tf_index_data = ForecastHandler.calculate_heat_index_adjusted(da_t2k.data, da_tdk.data)
        else:
            logging.error(f'Index {index} is not implemented, use either "HIS" or "HIA".')
            return None
        try:
            lon = da_tdk.lon
        except AttributeError:
            lon = da_tdk.longitude
        try:
            lat = da_tdk.lat
        except AttributeError:
            lat = da_tdk.latitude
        da_index = xr.DataArray(tf_index_data,
                                coords=da_tdk.coords,
                                dims=da_tdk.dims,
                                attrs={'description': index_long_name, 'units': unit})
        return da_index




    def calculate_heat_indices(self, data_out, year_list, month_list, area, overwrite, tf_index):
        """
        Calculates, processes, and saves heat indices or temperature-related metrics (Tmean, Tmax, Tmin, HIS, HIA)
        for specified years and months.
    
        Parameters:
        data_out (str): Base directory path for output data.
        year_list (list of int): Years for which to calculate indices.
        month_list (list of int): Months to calculate indices (1-12).
        area (list of float): Geographic area bounds [north, west, south, east] in degrees.
        overwrite (bool): If True, overwrites existing files; if False, skips processing for existing files.
        tf_index (str): The climate index being processed ('HIS', 'HIA', 'Tmean', 'Tmax', 'Tmin').
    
        Returns:
        None
    
        Notes:
        - Creates a directory structure organized by year and index type for output files.
        - Processes daily data into monthly values.
        - Calculates ensemble statistics, including mean, median, max, min, std, and percentiles.
        - Saves two types of NetCDF files:
          1. Monthly values for all ensemble members: {tf_index}_{year}{month}.nc
          2. Ensemble statistics: {tf_index}_{year}{month}_statistics.nc
        - Skips processing if output files already exist and overwrite is False.
        - Logs warnings for missing input files and information about saved outputs.
        """
        index_params = self.get_index_params(tf_index)
        index_out = f"{data_out}/{tf_index}"
        area_str = f'{int(area[1])}_{int(area[0])}_{int(area[2])}_{int(area[3])}'
    
        for year in year_list:
            if not os.path.exists(f'{index_out}/{year}'):
                os.makedirs(f'{index_out}/{year}')
        
            for month in month_list:
                month_str = f"{month:02d}"
                index_file = f'{index_out}/{year}/{tf_index}_{year}{month_str}.nc'
        
                if os.path.isfile(index_file) and not overwrite:
                    logging.info(f'Corresponding index file {index_file} already exists!')
                    continue  # Skip if the file already exists and overwrite is False
    
                daily_out = f'{data_out}/netcdf/daily/{year}/{month_str}'
            
                try:
                    with xr.open_dataset(f'{daily_out}/{index_params["filename_lead"]}_{area_str}_{year}{month_str}.nc') as daily_ds:
                        # Handling various indices
                        if tf_index == 'Tmean':
                            # Calculate mean temperature
                            da_index = daily_ds['t2m'] - 273.15  # Convert from Kelvin to Celsius
                            da_index.attrs["units"] = "degC"
                        elif tf_index == 'Tmax':
                            # Calculate max daily temperature
                            da_index = daily_ds['t2m'].resample(step='1D').max() - 273.15
                            da_index.attrs["units"] = "degC"
                        elif tf_index == 'Tmin':
                            # Calculate min daily temperature
                            da_index = daily_ds['t2m'].resample(step='1D').min() - 273.15
                            da_index.attrs["units"] = "degC"
                        elif tf_index == 'HIS':
                            # Calculate simplified heat index
                            da_index = self.calculate_heat_index(daily_ds['t2m'], daily_ds['d2m'], "HIS")
                        elif tf_index == 'HIA':
                            # Calculate adjusted heat index
                            da_index = self.calculate_heat_index(daily_ds['t2m'], daily_ds['d2m'], "HIA")
                        else:
                            raise ValueError(f"Unsupported index: {tf_index}")
    
                        # Save the index to a NetCDF file
                        ds_combined = xr.Dataset({tf_index: da_index})
                        ds_combined.to_netcdf(f'{daily_out}/{tf_index}_{year}{month_str}.nc')
                        logging.info(f"{tf_index} saved to {daily_out}/{tf_index}_{year}{month_str}.nc")
    
                except FileNotFoundError:
                    logging.warning(f'Data file {daily_out}/{index_params["filename_lead"]}_{area_str}_{year}{month_str}.nc does not exist!')
                    continue
                
                # Process data to align with forecast months like in tropical nights
                valid_times = pd.to_datetime(daily_ds.valid_time.values)
                forecast_months = valid_times.to_period('M')
                forecast_months_str = forecast_months.astype(str)
                step_to_month = dict(zip(daily_ds.step.values, forecast_months_str))
                forecast_month_da = xr.DataArray(list(step_to_month.values()), coords=[daily_ds.step], dims=['step'])
                da_index.coords['forecast_month'] = forecast_month_da
    
                # Calculate monthly means
                monthly_means = da_index.groupby('forecast_month').mean(dim='step')
                monthly_means = monthly_means.rename(tf_index)

                # Rename forecast_month to step
                monthly_means = monthly_means.rename({"forecast_month": "step"})

    
                # Save the monthly means for all members in one file
                ds_member_means = xr.Dataset(
                    {f'{tf_index}': monthly_means}
                )
    
                # Ensure 'number' dimension starts from 1 instead of 0
                ds_member_means = ds_member_means.assign_coords(number=ds_member_means.number)
    
                # Save the dataset
                ds_member_means.to_netcdf(index_file)
                logging.info(f'Monthly means saved to {index_file}')
                
                 # Now calculate ensemble statistics across members
                da_index_ens_mean = monthly_means.mean('number')
                da_index_ens_median = monthly_means.median('number')
                da_index_ens_max = monthly_means.max('number')
                da_index_ens_min = monthly_means.min('number')
                da_index_ens_std = monthly_means.std('number')
                percentile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
                ensemble_percentiles = monthly_means.quantile(percentile_levels, dim="number")
        
                ds_stats = xr.Dataset({
                    'ensemble_mean': da_index_ens_mean,
                    'ensemble_median': da_index_ens_median,
                    'ensemble_max': da_index_ens_max,
                    'ensemble_min': da_index_ens_min,
                    'ensemble_std': da_index_ens_std,
                })
                for i, level in enumerate(percentile_levels):
                    label = f"ensemble_p{int(level * 100)}"
                    ds_stats[label] = ensemble_percentiles.sel(quantile=level)
    
                stats_file = f"{index_out}/{year}/{tf_index}_{year}{month_str}_statistics.nc"
                ds_stats.to_netcdf(stats_file)
                logging.info(f'Ensemble statistics saved to {stats_file}')



    
    def calculate_and_save_tropical_nights_per_lag(self, base_path, year_list, month_list, tf_index, area):
        """
        Calculates, processes, and saves the tropical nights index for specified years and months.

        Parameters:
        base_path (str): Base directory path for input and output data.
        year_list (list of int): Years for which to calculate the index.
        month_list (list of int): Months for which to calculate the index (1-12).
        tf_index (str): The climate index being processed (should be 'TR' for Tropical Nights).
        area (list of float): Geographic area bounds [north, west, south, east] in degrees.

        Returns:
        None

        Notes:
        - Processes GRIB files to calculate tropical nights (nights with minimum temperature ≥ 20°C).
        - Creates a directory structure organized by year and index type for output files.
        - Saves two types of NetCDF files for each month:
          1. Tropical nights count for all ensemble members: {tf_index}_{year}{month}.nc
          2. Ensemble statistics: {tf_index}_{year}{month}_statistics.nc
        - Logs information about saved outputs and any errors encountered.

        The method performs the following steps for each year and month:
        1. Loads GRIB file data.
        2. Converts temperature from Kelvin to Celsius.
        3. Calculates daily minimum temperatures.
        4. Identifies tropical nights (min temp ≥ 20°C).
        5. Counts tropical nights per forecast month.
        6. Saves counts for all ensemble members.
        7. Calculates and saves ensemble statistics (mean, median, max, min, std, percentiles).

        Raises:
        FileNotFoundError: If the input GRIB file is not found.
        Exception: For any other errors during processing.
        """
        index_params = self.get_index_params(tf_index)
        area_str = f'{int(area[1])}_{int(area[0])}_{int(area[2])}_{int(area[3])}'
        for year in year_list:
            for month in month_list:
                month_str = f"{month:02d}"
                grib_file_path = f"{base_path}/grib/{year}/{month_str}/{index_params['filename_lead']}_{area_str}_{year}{month_str}.grib"
                output_dir = f"{base_path}/{tf_index}/{year}/"
                os.makedirs(output_dir, exist_ok=True)
                output_file_path = os.path.join(output_dir, f"{tf_index}_{year}{month_str}.nc")
                try:
                    ds = xr.open_dataset(grib_file_path, engine="cfgrib")
                    t2m_celsius = ds['t2m'] - 273.15
                    daily_min_temp = t2m_celsius.resample(step='1D').min()
                    valid_times = pd.to_datetime(ds.valid_time.values)
                    forecast_months = valid_times.to_period('M')
                    forecast_months_str = forecast_months.astype(str)
                    step_to_month = dict(zip(ds.step.values, forecast_months_str))
                    forecast_month_da = xr.DataArray(list(step_to_month.values()), coords=[ds.step], dims=['step'])
                    daily_min_temp.coords['forecast_month'] = forecast_month_da
                    tropical_nights = daily_min_temp >= 20
                    tropical_nights_count = tropical_nights.groupby('forecast_month').sum(dim='step')
                    tropical_nights_count = tropical_nights_count.rename(tf_index)

                    tropical_nights_count = tropical_nights_count.rename({"forecast_month": "step"})
                    tropical_nights_count.to_netcdf(output_file_path)
                    print(f"Tropical nights saved to {output_file_path}")

                    ensemble_mean = tropical_nights_count.mean(dim='number')
                    ensemble_median = tropical_nights_count.median(dim='number')
                    ensemble_max = tropical_nights_count.max(dim='number')
                    ensemble_min = tropical_nights_count.min(dim='number')
                    ensemble_std = tropical_nights_count.std(dim='number')
                    percentile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
                    ensemble_percentiles = tropical_nights_count.quantile(percentile_levels, dim="number")
                    ds_stats = xr.Dataset({
                        'ensemble_mean': ensemble_mean,
                        'ensemble_median': ensemble_median,
                        'ensemble_max': ensemble_max,
                        'ensemble_min': ensemble_min,
                        'ensemble_std': ensemble_std,
                    })
                    for i, level in enumerate(percentile_levels):
                        label = f"ensemble_p{int(level * 100)}"
                        ds_stats[label] = ensemble_percentiles.sel(quantile=level)
                    stats_output_dir = os.path.join(output_dir, "stats")
                    os.makedirs(stats_output_dir, exist_ok=True)
                    stats_file = os.path.join(stats_output_dir, f"{tf_index}_{year}{month_str}_statistics.nc")
                    ds_stats.to_netcdf(stats_file)
                    print(f'Tropical nights statistics saved to {stats_file}')
                except FileNotFoundError as e:
                    print(f'File not found: {e.filename}')
                except Exception as e:
                    print(f'An error occurred: {e}')




    def calculate_index(self, data_out, year_list, month_list, area_selection, overwrite, tf_index):
        """
        Calculates the specified climate index for given years and months.
    
        This method serves as a dispatcher to appropriate calculation methods based on the index type.
    
        Parameters:
        data_out (str): Base directory path for output data.
        year_list (list of int): Years for which to calculate the index.
        month_list (list of int): Months for which to calculate the index (1-12).
        area_selection (str): Area specification, can be 'global', coordinates, or country names.
        overwrite (bool): If True, overwrites existing files; if False, skips calculation if files exist.
        tf_index (str): The climate index to be calculated. Supported values are:
                        'HIS' (Heat Index Simplified),
                        'HIA' (Heat Index Adjusted),
                        'Tmean' (Mean Temperature),
                        'Tmax' (Maximum Temperature),
                        'Tmin' (Minimum Temperature),
                        'TR' (Tropical Nights).
    
        Returns:
        None
    
        Raises:
        ValueError: If an unsupported tf_index is provided.
    
        Notes:
        - For 'HIS', 'HIA', 'Tmean', 'Tmax', and 'Tmin', it calls the calculate_heat_indices method.
        - For 'TR', it calls the calculate_and_save_tropical_nights_per_lag method.
        - The area selection is converted to geographic bounds before being passed to the calculation methods.
        """
        area = self.get_bounds_for_area_selection(area_selection)
        
        # Handle heat indices and temperature-related indices
        if tf_index in ["HIS", "HIA", "Tmean", "Tmax", "Tmin"]:
            self.calculate_heat_indices(data_out, year_list, month_list, area, overwrite, tf_index)
        
        # Handle Tropical Nights
        elif tf_index == "TR":
            self.calculate_and_save_tropical_nights_per_lag(data_out, year_list, month_list, tf_index, area)
        
        else:
            logging.error(f'Index {tf_index} is not implemented. Supported indices are "HIS", "HIA", "Tmean", "Tmax", "Tmin", and "TR".')


    

    def process_and_save_hazards(self, year_list, month_list, data_out, tf_index):
        """
        Processes the calculated climate indices into hazard objects and saves them.

        This method converts the NetCDF files of climate indices into CLIMADA Hazard objects,
        which can be used for further risk analysis.

        Parameters:
        year_list (list of int): Years for which to process hazards.
        month_list (list of int): Months for which to process hazards (1-12).
        data_out (str): Base directory path for input and output data.
        tf_index (str): The climate index being processed ('HIS', 'HIA', 'Tmean', or 'TR').

        Returns:
        None

        Notes:
        - Input files are expected to be in NetCDF format.
        - Output files are saved in HDF5 format, compatible with CLIMADA.
        - Creates a directory structure: {data_out}/{tf_index}/hazard/{year}{month}/
        - Generates separate hazard files for each ensemble member.
        - After processing, plots the last generated hazard object.

        The method performs the following steps for each year and month:
        1. Loads the NetCDF file of the calculated index.
        2. Converts the time step information to datetime format.
        3. Creates a Hazard object for each ensemble member.
        4. Saves each Hazard object as an HDF5 file.
        5. Plots the last processed Hazard object.

        Raises:
        FileNotFoundError: If the input NetCDF file is not found.
        Exception: For any other errors during processing.
        """
        # Automatically fill parameters for processing and saving hazards based on tf_index
        base_input_dir = os.path.join(data_out, tf_index)
        base_output_dir = os.path.join(data_out, f"{tf_index}/hazard")
        hazard_type = tf_index
        intensity_variable = f"{tf_index}"
        intensity_unit = "days"    # Corrected intensity unit assignment based on the selected tf_index
        if tf_index == "TR":
            intensity_unit = "days"  # Tropical Nights are measured in days
        else:
            intensity_unit = "°C"  # All other indices are measured in degrees Celsius
        
        for year in year_list:
            for month in month_list:
                month_str = f"{month:02d}"
                current_input_dir = os.path.join(base_input_dir, str(year))
                nc_file_pattern = f"{hazard_type}_{year}{month_str}.nc"
                nc_file_path = os.path.join(current_input_dir, nc_file_pattern)
                current_output_dir = os.path.join(base_output_dir, f"{year}{month_str}")
                os.makedirs(current_output_dir, exist_ok=True)
                try:
                    ds = xr.open_dataset(nc_file_path)
                    ds["step"] = xr.DataArray([f"{date}-01" for date in ds["step"].values], dims=["step"])
                    ds["step"] = pd.to_datetime(ds["step"].values)
                    ensemble_members = ds["number"].values
                    for member in ensemble_members:
                        ds_subset = ds.sel(number=member)
                        hazard = Hazard.from_xarray_raster(
                            data=ds_subset,
                            hazard_type=hazard_type,
                            intensity_unit=intensity_unit,
                            intensity=intensity_variable,
                            coordinate_vars={"event": "step", "longitude": "longitude", "latitude": "latitude"}
                        )
                        hazard.check()
                        filename = f"hazard_{hazard_type}_member_{member}_{year}{month_str}.hdf5"
                        file_path = os.path.join(current_output_dir, filename)
                        hazard.write_hdf5(file_path)
                    print(f"Completed processing for {year}-{month_str}. Data saved in {current_output_dir}")
                except FileNotFoundError as e:
                    print(f"File not found: {e.filename}")
                except Exception as e:
                    print(f"An error occurred: {e}")

        # Retrieve the last saved hazard object and plot it
        last_hazard_file = file_path
        hazard_obj = Hazard.from_hdf5(last_hazard_file)
        self.plot_hazard(hazard_obj)

    @staticmethod
    def plot_hazard(hazard):
        """
        Plot the intensity of the hazard object.

        Parameters:
        hazard (Hazard): A Hazard object created from climate indices.
        """
        hazard.plot_intensity(1, smooth=False)
        