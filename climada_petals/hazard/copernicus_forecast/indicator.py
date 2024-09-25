# indicator.py

import xarray as xr # Called in the .ipynb
import numpy as np
import pandas as pd
import logging
import os
# from climada.util.coordinates import country_to_iso, get_country_geometries # Called in the .ipynb

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

    Parameters:
    index (str): The climate index identifier.

    Returns:
    dict: A dictionary containing the parameters for the specified index.
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

    Parameters:
    t2k (float or array-like): 2-meter temperature in Kelvin.
    tdk (float or array-like): 2-meter dewpoint temperature in Kelvin.

    Returns:
    float or array-like: Relative humidity as a percentage.
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

    Parameters:
    t2k (float or array-like): 2-meter temperature in Kelvin.
    tdk (float or array-like): 2-meter dewpoint temperature in Kelvin.

    Returns:
    float or array-like: Simplified heat index in degrees Celsius.
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

    Parameters:
    t2k (float or array-like): 2-meter temperature in Kelvin.
    tdk (float or array-like): 2-meter dewpoint temperature in Kelvin.

    Returns:
    float or array-like: Adjusted heat index in degrees Celsius.
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
    return (hi - 32) * 5 / 9  # convert back to Celsius


def calculate_heat_index(da_t2k, da_tdk, index):
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


def calculate_heat_indices(input_file_name, tf_index):
    """
    Calculates and saves heat indices or temperature metrics.

    Parameters:
    data_out (str): Base directory path for output data.
    year_list (list of int): Years for which to calculate indices.
    month_list (list of int): Months to calculate indices (1-12).
    area_selection (str): Area specification.
    overwrite (bool): If True, overwrites existing files.
    tf_index (str): The climate index being processed.
    """

    # Calculate index
    try:
        with xr.open_dataset(input_file_name) as daily_ds:
            # Handling various indices
            if tf_index == "Tmean":
                # Calculate mean temperature
                da_index = daily_ds["t2m"] - 273.15  # Convert from Kelvin to Celsius
                da_index.attrs["units"] = "degC"
            elif tf_index == "Tmax":
                # Calculate max daily temperature
                da_index = (
                    daily_ds["t2m"].resample(step="1D").max() - 273.15
                )
                da_index.attrs["units"] = "degC"
            elif tf_index == "Tmin":
                # Calculate min daily temperature
                da_index = (
                    daily_ds["t2m"].resample(step="1D").min() - 273.15
                )
                da_index.attrs["units"] = "degC"
            elif tf_index == "HIS":
                # Calculate simplified heat index
                da_index = calculate_heat_index(
                    daily_ds["t2m"], daily_ds["d2m"], "HIS"
                )
            elif tf_index == "HIA":
                # Calculate adjusted heat index
                da_index = calculate_heat_index(
                    daily_ds["t2m"], daily_ds["d2m"], "HIA"
                )
            else:
                raise ValueError(f"Unsupported index: {tf_index}")

            # Save the index to a NetCDF file
            ds_combined = xr.Dataset({tf_index: da_index})

    except FileNotFoundError:
        logging.error(f'Data file {input_file_name} does not exist.')

    # Calculate statistics
    valid_times = pd.to_datetime(daily_ds.valid_time.values)
    forecast_months = valid_times.to_period("M")
    forecast_months_str = forecast_months.astype(str)
    step_to_month = dict(zip(daily_ds.step.values, forecast_months_str))
    forecast_month_da = xr.DataArray(
        list(step_to_month.values()), coords=[daily_ds.step], dims=["step"]
    )

    da_index.coords["forecast_month"] = forecast_month_da

    # Calculate monthly means
    monthly_means = da_index.groupby("forecast_month").mean(dim="step")
    monthly_means = monthly_means.rename(tf_index)

    # Rename forecast_month to step
    monthly_means = monthly_means.rename({"forecast_month": "step"})

    # Save the monthly means for all members in one file
    ds_monthly = xr.Dataset({f"{tf_index}": monthly_means})

    # Ensure 'number' dimension starts from 1 instead of 0
    ds_monthly = ds_monthly.assign_coords(number=ds_monthly.number)

    # Now calculate ensemble statistics across members
    da_index_ens_mean = monthly_means.mean("number")
    da_index_ens_median = monthly_means.median("number")
    da_index_ens_max = monthly_means.max("number")
    da_index_ens_min = monthly_means.min("number")
    da_index_ens_std = monthly_means.std("number")
    percentile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    ensemble_percentiles = monthly_means.quantile(percentile_levels, dim="number")

    ds_stats = xr.Dataset(
        {
            "ensemble_mean": da_index_ens_mean,
            "ensemble_median": da_index_ens_median,
            "ensemble_max": da_index_ens_max,
            "ensemble_min": da_index_ens_min,
            "ensemble_std": da_index_ens_std,
        }
    )
    for i, level in enumerate(percentile_levels):
        label = f"ensemble_p{int(level * 100)}"
        ds_stats[label] = ensemble_percentiles.sel(quantile=level)

    return ds_combined, ds_monthly, ds_stats


def calculate_and_save_tropical_nights_per_lag(grib_file_path, tf_index):
    """
    Calculates and saves the tropical nights index.

    Parameters:
    base_path (str): Base directory path for input and output data.
    year_list (list of int): Years for which to calculate the index.
    month_list (list of int): Months for which to calculate the index (1-12).
    tf_index (str): The climate index being processed ('TR').
    area_selection (str): Area specification.
    """            
    try:
        # calculate index
        ds = xr.open_dataset(grib_file_path, engine="cfgrib")
        t2m_celsius = ds["t2m"] - 273.15
        daily_min_temp = t2m_celsius.resample(step="1D").min()
        valid_times = pd.to_datetime(ds.valid_time.values)
        forecast_months = valid_times.to_period("M")
        forecast_months_str = forecast_months.astype(str)
        step_to_month = dict(zip(ds.step.values, forecast_months_str))
        forecast_month_da = xr.DataArray(list(step_to_month.values()), coords=[ds.step], dims=["step"])
        daily_min_temp.coords["forecast_month"] = forecast_month_da
        tropical_nights = daily_min_temp >= 20
        tropical_nights_count = tropical_nights.groupby("forecast_month").sum(dim="step")
        tropical_nights_count = tropical_nights_count.rename(tf_index)
        tropical_nights_count = tropical_nights_count.rename({"forecast_month": "step"})

        # calculate statistics
        ensemble_mean = tropical_nights_count.mean(dim="number")
        ensemble_median = tropical_nights_count.median(dim="number")
        ensemble_max = tropical_nights_count.max(dim="number")
        ensemble_min = tropical_nights_count.min(dim="number")
        ensemble_std = tropical_nights_count.std(dim="number")
        percentile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        ensemble_percentiles = tropical_nights_count.quantile(percentile_levels, dim="number")
        
        ds_stats = xr.Dataset({
            "ensemble_mean": ensemble_mean,
            "ensemble_median": ensemble_median,
            "ensemble_max": ensemble_max,
            "ensemble_min": ensemble_min,
            "ensemble_std": ensemble_std,
        })
        
        for i, level in enumerate(percentile_levels):
            label = f"ensemble_p{int(level * 100)}"
            ds_stats[label] = ensemble_percentiles.sel(quantile=level)
        
        return None, tropical_nights_count, ds_stats
    
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except Exception as e:
        print(f"An error occurred: {e}")


def calculate_and_save_tx30_per_lag(grib_file_path, tf_index):
    """
    Calculates and saves the TX30 index (Tmax > 30°C).

    Parameters:
    base_path (str): Base directory path for input and output data.
    year_list (list of int): Years for which to calculate the index.
    month_list (list of int): Months for which to calculate the index (1-12).
    tf_index (str): The climate index being processed ('TX30').
    area_selection (str): Area specification.
    """
    try:
        ds = xr.open_dataset(grib_file_path, engine="cfgrib")
        t2m_celsius = ds["t2m"] - 273.15
        daily_max_temp = t2m_celsius.resample(step="1D").max()
        valid_times = pd.to_datetime(ds.valid_time.values)
        forecast_months = valid_times.to_period("M")
        forecast_months_str = forecast_months.astype(str)
        step_to_month = dict(zip(ds.step.values, forecast_months_str))
        forecast_month_da = xr.DataArray(list(step_to_month.values()), coords=[ds.step], dims=["step"])
        daily_max_temp.coords["forecast_month"] = forecast_month_da

        # Calculate TX30: Days where Tmax > 30°C
        tx30_days = daily_max_temp > 30  # Boolean array where True means a TX30 day
        
        # Count the number of TX30 days per forecast month
        tx30_days_count = tx30_days.groupby("forecast_month").sum(dim="step")
        tx30_days_count = tx30_days_count.rename(tf_index)

        tx30_days_count = tx30_days_count.rename({"forecast_month": "step"})

        # Calculate ensemble statistics (mean, median, etc.)
        ensemble_mean = tx30_days_count.mean(dim="number")
        ensemble_median = tx30_days_count.median(dim="number")
        ensemble_max = tx30_days_count.max(dim="number")
        ensemble_min = tx30_days_count.min(dim="number")
        ensemble_std = tx30_days_count.std(dim="number")
        
        # Percentiles
        percentile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        ensemble_percentiles = tx30_days_count.quantile(percentile_levels, dim="number")
        
        # Store statistics in a dataset
        ds_stats = xr.Dataset({
            "ensemble_mean": ensemble_mean,
            "ensemble_median": ensemble_median,
            "ensemble_max": ensemble_max,
            "ensemble_min": ensemble_min,
            "ensemble_std": ensemble_std,
        })
        
        for i, level in enumerate(percentile_levels):
            label = f"ensemble_p{int(level * 100)}"
            ds_stats[label] = ensemble_percentiles.sel(quantile=level)
        
        return None, tx30_days_count, ds_stats
    
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

# def calculate_statistics_from_index(dataarray):
#     ensemble_mean = dataarray.mean(dim="number")
#     ensemble_median = dataarray.median(dim="number")
#     ensemble_max = dataarray.max(dim="number")
#     ensemble_min = dataarray.min(dim="number")
#     ensemble_std = dataarray.std(dim="number")
#     percentile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
#     ensemble_percentiles = dataarray.quantile(percentile_levels, dim="number")
    
#     ds_stats = xr.Dataset({
#         "ensemble_mean": ensemble_mean,
#         "ensemble_median": ensemble_median,
#         "ensemble_max": ensemble_max,
#         "ensemble_min": ensemble_min,
#         "ensemble_std": ensemble_std,
#     })
    
#     for i, level in enumerate(percentile_levels):
#         label = f"ensemble_p{int(level * 100)}"
#         ds_stats[label] = ensemble_percentiles.sel(quantile=level)

#     return ds_stats