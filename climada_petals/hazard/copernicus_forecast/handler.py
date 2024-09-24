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


# Module_forescast_handler.py
import os
import logging
import calendar
import cdsapi
import xarray as xr
import pandas as pd
#from climada.util.coordinates import country_to_iso, get_country_geometries
#from climada import CONFIG
from climada.hazard import Hazard
#import numpy as np
import climada_petals.hazard.copernicus_forecast.indicator as indicator



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
        

        # Dictionary with explanations and input data for each index
        self.index_explanations = {
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
            "HotDays": {
                "explanation": "Hot Days: This indicator counts the number of days where the maximum temperature exceeds 30°C.",
                "input_data": ["2m temperature (t2m)"]
            }
        }

    def explain_index(self, tf_index):
        """
        Provides an explanation and input data for the selected index.

        Parameters:
        tf_index (str): The climate index identifier.

        Returns:
        str: A description of the selected index, its input data, or an error message if the index is invalid.
        """
        if tf_index in self.index_explanations:
            explanation = f"Selected Index: {tf_index}\n"
            explanation += f"Explanation: {self.index_explanations[tf_index]['explanation']}\n"
            explanation += f"Input Data: {', '.join(self.index_explanations[tf_index]['input_data'])}"
        else:
            explanation = f"Error: {tf_index} is not a valid index. Please choose from {list(self.index_explanations.keys())}"

        return explanation
    

    def calc_min_max_lead(self, year, month, leadtime_months=1):
        """
        Calculates the minimum and maximum lead time in hours for a given starting date.

        Parameters:
        year (int): The starting year (e.g., 2023).
        month (int): The starting month (1-12).
        leadtime_months (int): Number of months to include in the forecast period.

        Returns:
        tuple: (min_lead, max_lead) in hours.
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

    def download_multvar_multlead(
        self, filename, vars, year, month, l_hours, area, overwrite, format, originating_centre, system
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
                    f'{download_file}'
                )
                self.logger.info(f'{format.capitalize()} file successfully downloaded to {download_file}.')
            except Exception as e:
                self.logger.error(f'{format.capitalize()} file {download_file} could not be downloaded. Error: {e}')

    def download_data(
        self, data_out, year_list, month_list, area_selection, overwrite, tf_index, format, originating_centre, system, max_lead_month
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
        area = indicator.get_bounds_for_area_selection(area_selection)
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
                    variables = index_params['variables']
                else:
                    variables = [index_params['variables'][0]]
                
                self.download_multvar_multlead(
                    download_file, variables, year, month, leadtimes, area,
                    overwrite, format, originating_centre, system
                )

    def process_data(self, data_out, year_list, month_list, area_selection, overwrite, tf_index, format):
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
        area = indicator.get_bounds_for_area_selection(area_selection)
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

    def download_and_process_data(
        self, data_out, year_list, month_list, area_selection, overwrite, tf_index, format, originating_centre, system, max_lead_month
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
        self.download_data(data_out, year_list, month_list, area_selection, overwrite, tf_index, format, originating_centre, system, max_lead_month)
        self.process_data(data_out, year_list, month_list, area_selection, overwrite, tf_index, format)

    def calculate_index(self, data_out, year_list, month_list, area_selection, overwrite, tf_index):
        """
        Calculates the specified climate index for given years and months.

        Parameters:
        data_out (str): Base directory path for output data.
        year_list (list of int): Years for which to calculate the index.
        month_list (list of int): Months for which to calculate the index (1-12).
        area_selection (str): Area specification.
        overwrite (bool): If True, overwrites existing files.
        tf_index (str): The climate index to be calculated.

        Returns:
        None
        """
        indicator.calculate_index(data_out, year_list, month_list, area_selection, overwrite, tf_index)

    def process_and_save_hazards(self, year_list, month_list, data_out, tf_index):
        """
        Processes the calculated climate indices into hazard objects and saves them.

        This method converts the NetCDF files of climate indices into CLIMADA Hazard objects,
        which can be used for further risk analysis.
        """
        base_input_dir = os.path.join(data_out, tf_index)
        base_output_dir = os.path.join(data_out, f"{tf_index}/hazard")
        hazard_type = tf_index
        intensity_variable = f"{tf_index}"

        if tf_index in ["TR", "HotDays", "HW"]:
            intensity_unit = "days"
        else:
            intensity_unit = "°C"

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
