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

File to calculate differen seasonal forecast indices.
"""


import xarray as xr 
import numpy as np
import pandas as pd
import logging

VAR_SPECS = {
        "2m_temperature": {
            "unit": "K",
            "standard_name": "air_temperature",
            "short_name": "t2m",
            "full_name": "2m_temperature",
        },
        "2m_dewpoint_temperature": {
            "unit": "K",
            "standard_name": "dew_point_temperature",
            "short_name": "d2m",
            "full_name": "2m_dewpoint_temperature",
        },
    }

def get_index_params(index):
    """
    Retrieves parameters associated with a specific climate index.

    Parameters
    ----------
    index : str
        The climate index identifier for which the parameters are being retrieved. 
        It could be one of the following:
        - "HIA" : Heat Index Adjusted
        - "HIS" : Heat Index Simplified
        - "Tmean" : Mean Temperature
        - "Tmin" : Minimum Temperature
        - "Tmax" : Maximum Temperature
        - "HW" : Heat Wave
        - "TR" : Tropical Nights
        - "TX30" : Hot Days (Tmax > 30°C)

    Returns
    -------
    dict
        A dictionary containing the parameters associated with the specified index. 
        The dictionary includes:
        - "variables" : List of variable names required for the index calculation.
        - "filename_lead" : String prefix used in the filename.
        - "index_long_name" : Full descriptive name of the index.
    """
    index_params = {
        "HIA": {
            "variables": [
                VAR_SPECS["2m_temperature"]["full_name"],
                VAR_SPECS["2m_dewpoint_temperature"]["full_name"],
            ],
            "filename_lead": "2m_temps",
            "index_long_name": "Heat_Index_Adjusted",
        },
        "HIS": {
            "variables": [
                VAR_SPECS["2m_temperature"]["full_name"],
                VAR_SPECS["2m_dewpoint_temperature"]["full_name"],
            ],
            "filename_lead": "2m_temps",
            "index_long_name": "Heat_Index_Simplified",
        },
        "Tmean": {
            "variables": [VAR_SPECS["2m_temperature"]["full_name"]],
            "filename_lead": "2m_temps",
            "index_long_name": "Mean_Temperature",
        },
        "Tmin": {
            "variables": [VAR_SPECS["2m_temperature"]["full_name"]],
            "filename_lead": "2m_temps",
            "index_long_name": "Minimum_Temperature",
        },
        "Tmax": {
            "variables": [VAR_SPECS["2m_temperature"]["full_name"]],
            "filename_lead": "2m_temps",
            "index_long_name": "Maximum_Temperature",
        },
        "HW": {
            "variables": [VAR_SPECS["2m_temperature"]["full_name"]],
            "filename_lead": "2m_temps",
            "index_long_name": "Heat_Wave",
        },
        "TR": {
            "variables": [VAR_SPECS["2m_temperature"]["full_name"]],
            "filename_lead": "2m_temps",
            "index_long_name": "Tropical_Nights",
        },
        "TX30": {
            "variables": [VAR_SPECS["2m_temperature"]["full_name"]],
            "filename_lead": "2m_temps",
            "index_long_name": "Hot Days (Tmax > 30°C)",
        },
    }
    return index_params.get(index)

def calculate_relative_humidity_percent(t2k, tdk):
    """
    Calculates the relative humidity percentage from temperature and dewpoint temperature.

    Parameters
    ----------
    t2k : float or array-like
        2-meter temperature in Kelvin. This parameter represents the air temperature measured at a height of 2 meters above ground level.
    tdk : float or array-like
        2-meter dewpoint temperature in Kelvin. The dewpoint temperature is the temperature at which air becomes saturated and condensation begins.

    Returns
    -------
    float or array-like
        Relative humidity as a percentage. The result is constrained between 0 and 100, where 0 indicates completely dry air, and 100 indicates saturated air.
    """
    t2c = t2k - 273.15
    tdc = tdk - 273.15

    es = 6.11 * 10.0 ** (7.5 * t2c / (237.3 + t2c))
    e = 6.11 * 10.0 ** (7.5 * tdc / (237.3 + tdc))

    rh = (e / es) * 100
    rh = np.clip(rh, 0, 100)
    return rh


def calculate_heat_index_simplified(t2k, tdk):
    """
    Calculates the simplified heat index based on temperature and dewpoint temperature.

    Parameters
    ----------
    t2k : float or array-like
        2-meter air temperature in Kelvin. This value represents the temperature measured at a height of 2 meters above ground level.
    tdk : float or array-like
        2-meter dewpoint temperature in Kelvin. The dewpoint temperature is the temperature at which air becomes saturated and moisture begins to condense.

    Returns
    -------
    float or array-like
        Simplified heat index in degrees Celsius. This is an estimate of how hot it feels to the human body by combining the effects of temperature and relative humidity.
    """
    t2c = t2k - 273.15
    rh = calculate_relative_humidity_percent(t2k, tdk)
    hi = (
        -8.784695
        + 1.61139411 * t2c
        + 2.338549 * rh
        - 0.14611605 * t2c * rh
        + -1.2308094e-2 * t2c**2
        + -1.6424828e-2 * rh**2
        + 2.211732e-3 * t2c**2 * rh
        + 7.2546e-4 * t2c * rh**2
        - 3.582e-6 * t2c**2 * rh**2
    )
    return hi


def calculate_heat_index_adjusted(t2k, tdk):
    """
    Calculates the adjusted heat index based on temperature and dewpoint temperature.

    Parameters
    ----------
    t2k : float or array-like
        2-meter air temperature in Kelvin. This value represents the temperature measured at a height of 2 meters above ground level.
    tdk : float or array-like
        2-meter dewpoint temperature in Kelvin. The dewpoint temperature is the temperature at which the air becomes saturated and condensation begins.

    Returns
    -------
    float or array-like
        Adjusted heat index in degrees Celsius. This value indicates the perceived temperature, accounting for the combined effect of temperature and relative humidity. It is a refined version of the heat index formula that adjusts for extreme values of temperature and humidity.
    """
    rh = calculate_relative_humidity_percent(t2k, tdk)
    t2f = (t2k - 273.15) * 9 / 5 + 32
    hi = (
        -42.379
        + 2.04901523 * t2f
        + 10.14333127 * rh
        - 0.22475541 * t2f * rh
        + -0.00683783 * t2f**2
        + -0.05481717 * rh**2
        + 0.00122874 * t2f**2 * rh
        + 0.00085282 * t2f * rh**2
        - 0.00000199 * t2f**2 * rh**2
    )
    return (hi - 32) * 5 / 9  # converted back to Celsius


def calculate_heat_index(da_t2k, da_tdk, index):
    """
    Calculates the heat index based on temperature and dewpoint temperature using either the simplified or adjusted formula.

    Parameters
    ----------
    da_t2k : xarray.DataArray
        2-meter air temperature in Kelvin. This value represents the air temperature measured at a height of 2 meters above ground level.
    da_tdk : xarray.DataArray
        2-meter dewpoint temperature in Kelvin. The dewpoint temperature is the temperature at which the air becomes saturated and condensation begins.
    index : str
        Identifier for the type of heat index to calculate. Options are:
        - "HIS": Heat Index Simplified.
        - "HIA": Heat Index Adjusted.

    Returns
    -------
    xarray.DataArray
        The calculated heat index in degrees Celsius, represented as an `xarray.DataArray` with the same dimensions and coordinates as the input data. It includes the heat index values along with relevant metadata, such as units and a description.
    """
    if index == "HIS":
        index_long_name = "heat_index_simplified"
        unit = "degC"
        tf_index_data = calculate_heat_index_simplified(da_t2k.data, da_tdk.data)
    elif index == "HIA":
        index_long_name = "heat_index_adjusted"
        unit = "degC"
        tf_index_data = calculate_heat_index_adjusted(da_t2k.data, da_tdk.data)
    else:
        logging.error(f'Index {index} is not implemented, use either "HIS" or "HIA".')
        return None
    da_index = xr.DataArray(
        tf_index_data,
        coords=da_tdk.coords,
        dims=da_tdk.dims,
        attrs={"description": index_long_name, "units": unit},
    )
    return da_index


def calculate_heat_indices_metrics(input_file_name, tf_index):
    """
    Calculates heat indices or temperature metrics based on the provided input file and index type.

    Parameters
    ----------
    input_file_name : str
        Path to the input data file containing temperature and dewpoint information.
    tf_index : str
        The climate index to be processed. Supported indices include:
        - "Tmean" : Mean daily temperature
        - "Tmax" : Maximum daily temperature
        - "Tmin" : Minimum daily temperature
        - "HIS" : Simplified Heat Index
        - "HIA" : Adjusted Heat Index

    Returns
    -------
    tuple
        A tuple containing three `xarray.Dataset` objects:
        - `daily index` : The calculated daily index values.
        - `monthly index` : Monthly mean values of the index.
        - `index statistics` : Ensemble statistics calculated from the index.

    Raises
    ------
    ValueError
        If an unsupported index is provided.
    FileNotFoundError
        If the specified input file does not exist.
    """
    try:
        with xr.open_dataset(input_file_name) as daily_ds:
            # Handling various indices
            if tf_index == "Tmean":
                # Calculate mean temperature
                da_index = daily_ds["t2m_mean"] - 273.15  # Convert from Kelvin to Celsius
                da_index.attrs["units"] = "degC"
            elif tf_index == "Tmax":
                # Calculate max daily temperature
                da_index = (
                    daily_ds["t2m_max"].resample(step="1D").max() - 273.15
                )
                da_index.attrs["units"] = "degC"
            elif tf_index == "Tmin":
                # Calculate min daily temperature
                da_index = (
                    daily_ds["t2m_min"].resample(step="1D").min() - 273.15
                )
                da_index.attrs["units"] = "degC"
            elif tf_index == "HIS":
                # Calculate simplified heat index
                da_index = calculate_heat_index(
                    daily_ds["t2m_mean"], daily_ds["d2m_mean"], "HIS"
                )
            elif tf_index == "HIA":
                # Calculate adjusted heat index
                da_index = calculate_heat_index(
                    daily_ds["t2m_mean"], daily_ds["d2m_mean"], "HIA"
                )
            else:
                raise ValueError(f"Unsupported index: {tf_index}")

            # Save the index to a NetCDF file
            ds_combined = xr.Dataset({tf_index: da_index})

    except FileNotFoundError:
        logging.error(f'Data file {input_file_name} does not exist.')

    # Calculate statistics
    valid_times = pd.to_datetime(daily_ds.valid_time.values)
    forecast_months_str = valid_times.to_period("M").astype(str)
    step_to_month = dict(zip(daily_ds.step.values, forecast_months_str))
    forecast_month_da = xr.DataArray(
        list(step_to_month.values()), coords=[daily_ds.step], dims=["step"]
    )

    da_index.coords["forecast_month"] = forecast_month_da

    # Calculate monthly means
    monthly_means = da_index.groupby("forecast_month").mean(dim="step")
    monthly_means = monthly_means.rename(tf_index)
    monthly_means = monthly_means.rename({"forecast_month": "step"})
    ds_monthly = xr.Dataset({f"{tf_index}": monthly_means})

    # Ensure 'number' dimension starts from 1 instead of 0
    ds_monthly = ds_monthly.assign_coords(number=ds_monthly.number)

    # calculate ensemble statistics across members
    ds_stats = calculate_statistics_from_index(monthly_means)

    return ds_combined, ds_monthly, ds_stats


def calculate_TR(grib_file_path, tf_index):
    """
    Calculates and saves the tropical nights index, defined as the number of nights where the minimum temperature remains at or above 20°C.

    Parameters
    ----------
    grib_file_path : str
        Path to the input GRIB data file containing temperature data. The file should be structured to include 2-meter temperature values (`t2m`).
    tf_index : str
        The climate index being processed. This should specify the name for the tropical nights index, such as "TR" (Tropical Nights).

    Returns
    -------
    tuple
        A tuple containing:
        - `None` : No daily index is returned for this calculation.
        - `xarray.Dataset` : The monthly count of tropical nights, stored as an `xarray.Dataset` with the index values and relevant metadata.
        - `xarray.Dataset` : Statistics calculated across the monthly tropical nights index values, representing ensemble statistics.

    Raises
    ------
    FileNotFoundError
        If the specified input GRIB file does not exist.
    Exception
        For any other errors encountered during the data processing.
    """           
    try:
        # prepare dataarray
        with xr.open_dataset(grib_file_path, engine="cfgrib") as ds:
            t2m_celsius = ds["t2m"] - 273.15
            daily_min_temp = t2m_celsius.resample(step="1D").min()
            valid_times = pd.to_datetime(ds.valid_time.values)
            forecast_months_str = valid_times.to_period("M").astype(str)
            step_to_month = dict(zip(ds.step.values, forecast_months_str))
            forecast_month_da = xr.DataArray(list(
                step_to_month.values()), coords=[ds.step], dims=["step"]
            )
        daily_min_temp.coords["forecast_month"] = forecast_month_da

        #compute tropical nights
        tropical_nights = daily_min_temp >= 20
        tropical_nights_count = tropical_nights.groupby("forecast_month").sum(dim="step")
        tropical_nights_count = tropical_nights_count.rename(tf_index)
        tropical_nights_count = tropical_nights_count.rename({"forecast_month": "step"})

        # calculate statistics
        ds_stats = calculate_statistics_from_index(tropical_nights_count)
        
        return None, tropical_nights_count, ds_stats
    
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except Exception as e:
        print(f"An error occurred: {e}")


def calculate_tx30(grib_file_path, tf_index):
    """
    Calculates and saves the TX30 index, defined as the number of days with maximum temperature above 30°C.

    Parameters
    ----------
    grib_file_path : str
        Path to the input GRIB data file containing temperature data. The file should include 2-meter temperature values (`t2m`) for daily maximum temperature calculations.
    tf_index : str
        The climate index being processed. This should specify the name for the TX30 index, typically "TX30".

    Returns
    -------
    tuple
        A tuple containing:
        - `None` : No daily index is returned for this calculation.
        - `xarray.Dataset` : The monthly count of TX30 days, represented as an `xarray.Dataset` with the index values and relevant metadata.
        - `xarray.Dataset` : Statistics calculated across the monthly TX30 index values, representing ensemble statistics.

    Raises
    ------
    FileNotFoundError
        If the specified input GRIB file does not exist.
    Exception
        For any other errors encountered during the data processing.
    """
    try:
        # Prepare dataarray
        with xr.open_dataset(grib_file_path, engine="cfgrib") as ds:
            t2m_celsius = ds["t2m"] - 273.15
            daily_max_temp = t2m_celsius.resample(step="1D").max()
            valid_times = pd.to_datetime(ds.valid_time.values)
            forecast_months_str = valid_times.to_period("M").astype(str)
            step_to_month = dict(zip(ds.step.values, forecast_months_str))
            forecast_month_da = xr.DataArray(list(
                step_to_month.values()), coords=[ds.step], dims=["step"]
            )
        daily_max_temp.coords["forecast_month"] = forecast_month_da

        # Calculate TX30: Days where Tmax > 30°C
        tx30_days = daily_max_temp > 30  # Boolean array where True means a TX30 day
        
        # Count the number of TX30 days per forecast month
        tx30_days_count = tx30_days.groupby("forecast_month").sum(dim="step")
        tx30_days_count = tx30_days_count.rename(tf_index)
        tx30_days_count = tx30_days_count.rename({"forecast_month": "step"})
   
        # calculate statistics
        ds_stats = calculate_statistics_from_index(tx30_days_count)
        
        return None, tx30_days_count, ds_stats
    
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

def calculate_statistics_from_index(dataarray):
    """
    Calculates a set of ensemble statistics for the given data array, including mean, median, standard deviation, and selected percentiles.

    Parameters
    ----------
    dataarray : xarray.DataArray
        Input data array representing climate index values across multiple ensemble members.
        It should have a dimension named "number" corresponding to the different ensemble members.

    Returns
    -------
    xarray.Dataset
        A dataset containing the calculated statistics:
        - `ensemble_mean`: The mean value across the ensemble members.
        - `ensemble_median`: The median value across the ensemble members.
        - `ensemble_max`: The maximum value across the ensemble members.
        - `ensemble_min`: The minimum value across the ensemble members.
        - `ensemble_std`: The standard deviation across the ensemble members.
        - `ensemble_p05`, `ensemble_p25`, `ensemble_p50`, `ensemble_p75`, `ensemble_p95`: Percentile values (5th, 25th, 50th, 75th, and 95th) across the ensemble members.

    """
    # Calculate ensemble statistics (mean, median, etc.)
    ensemble_mean = dataarray.mean(dim="number")
    ensemble_median = dataarray.median(dim="number")
    ensemble_max = dataarray.max(dim="number")
    ensemble_min = dataarray.min(dim="number")
    ensemble_std = dataarray.std(dim="number")

    # create dataset
    ds_stats = xr.Dataset({
        "ensemble_mean": ensemble_mean,
        "ensemble_median": ensemble_median,
        "ensemble_max": ensemble_max,
        "ensemble_min": ensemble_min,
        "ensemble_std": ensemble_std,
    })

    # add percentiles
    percentile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    ensemble_percentiles = dataarray.quantile(percentile_levels, dim="number")
    for level in percentile_levels:
        label = f"ensemble_p{int(level * 100)}"
        ds_stats[label] = ensemble_percentiles.sel(quantile=level)

    return ds_stats

def index_explanations(tf_index):
    """
    Provides a detailed explanation and required input data for a given climate index.

    Parameters
    ----------
    tf_index : str
        The climate index identifier for which an explanation and input data are needed. 
        Supported indices include:
        - "HIA" : Heat Index Adjusted
        - "HIS" : Heat Index Simplified
        - "Tmean" : Mean Temperature
        - "Tmin" : Minimum Temperature
        - "Tmax" : Maximum Temperature
        - "HW" : Heat Wave
        - "TR" : Tropical Nights
        - "TX30" : Hot Days (Tmax > 30°C)

    Returns
    -------
    dict
        A dictionary containing two keys:
        - 'explanation': A detailed description of the climate index.
        - 'input_data': A list of required variables for calculating the index.
        
        If the index is not found, it returns a dictionary with:
        - 'error': Description of the issue.
        - 'valid_indices': A list of supported index identifiers.
    """
    index_explanations = {
        "HIA": {
            "explanation": "Heat Index Adjusted: This indicator measures apparent "\
                "temperature, considering both air temperature and humidity, providing a more "\
                "accurate perception of how hot it feels.",
            "input_data": ["2m temperature (t2m)", "2m dewpoint temperature (d2m)"]
        },
        "HIS": {
            "explanation": "Heat Index Simplified: This indicator is a simpler version of the "\
                "Heat Index, focusing on a quick estimate of perceived heat based on temperature "\
                "and humidity.",
            "input_data": ["2m temperature (t2m)", "2m dewpoint temperature (d2m)"]
        },
        "Tmean": {
            "explanation": "Mean Temperature: This indicator calculates the average temperature "\
                "over the specified period.",
            "input_data": ["2m temperature (t2m)"]
        },
        "Tmin": {
            "explanation": "Minimum Temperature: This indicator tracks the lowest temperature "\
                "recorded over a specified period.",
            "input_data": ["2m temperature (t2m)"]
        },
        "Tmax": {
            "explanation": "Maximum Temperature: This indicator tracks the highest temperature "\
                "recorded over a specified period.",
            "input_data": ["2m temperature (t2m)"]
        },
        "HW": {
            "explanation": "Heat Wave: This indicator identifies heat waves, defined as at least "\
                "3 consecutive days with temperatures exceeding a certain threshold.",
            "input_data": ["2m temperature (t2m)"]
        },
        "TR": {
            "explanation": "Tropical Nights: This indicator counts the number of nights where "\
                "the minimum temperature remains above a certain threshold, typically 20°C.",
            "input_data": ["2m temperature (t2m)"]
        },
        "TX30": {
            "explanation": "Hot Days: This indicator counts the number of days where the maximum "\
                "temperature exceeds 30°C.",
            "input_data": ["2m temperature (t2m)"]
        }
    }
    
    # Return the explanation if found; otherwise, provide valid index options
    return index_explanations.get(tf_index, {"error": "Unknown index", "valid_indices": list(index_explanations.keys())})
