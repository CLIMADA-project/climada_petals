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

File to calculate different seasonal forecast indices.
"""

import xarray as xr
import pandas as pd
import logging

from climada_petals.hazard.copernicus_forecast.index_definitions import IndexSpecEnum

from climada_petals.hazard.copernicus_forecast.heat_index import (
    calculate_heat_index,
    calculate_relative_humidity,
    calculate_humidex,
    calculate_wind_speed,
    calculate_apparent_temperature,
    calculate_wbgt_simple,
    calculate_tx30,
    calculate_tr,
    calculate_hw,
)

LOGGER = logging.getLogger(__name__)


def calculate_heat_indices_metrics(input_file_name, index_metric):
    """
    Calculates heat indices or temperature metrics based on the provided input file and index type.

    Parameters
    ----------
    input_file_name : str
        Path to the input data file containing temperature and dewpoint information.
    index_metric : str
        The climate index to be processed. Supported indices include:
        - "Tmean" : Mean daily temperature
        - "Tmax" : Maximum daily temperature
        - "Tmin" : Minimum daily temperature
        - "HIS" : Simplified Heat Index
        - "HIA" : Adjusted Heat Index
        - "RH"  : Relative Humidity
        - "HUM" : Humidex
        - "AT"  : Apparent Temperature
        - "WBGT": Wet Bulb Globe Temperature (Simple)

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
            if index_metric == "Tmean":
                da_index = daily_ds["t2m_mean"] - 273.15  # Kelvin to Celsius
                da_index.attrs["units"] = "degC"
            elif index_metric == "Tmax":
                da_index = daily_ds["t2m_max"].resample(step="1D").max() - 273.15
                da_index.attrs["units"] = "degC"
            elif index_metric == "Tmin":
                da_index = daily_ds["t2m_min"].resample(step="1D").min() - 273.15
                da_index.attrs["units"] = "degC"
            elif index_metric == "HIS":
                da_index = calculate_heat_index(
                    daily_ds["t2m_mean"], daily_ds["d2m_mean"], "HIS"
                )
            elif index_metric == "HIA":
                da_index = calculate_heat_index(
                    daily_ds["t2m_mean"], daily_ds["d2m_mean"], "HIA"
                )
            elif index_metric == "RH":
                da_index = calculate_relative_humidity(
                    daily_ds["t2m_mean"], daily_ds["d2m_mean"]
                )
            elif index_metric == "HUM":
                da_index = calculate_humidex(daily_ds["t2m_mean"], daily_ds["d2m_mean"])
            elif index_metric == "AT":
                u10 = daily_ds.get("u10_max", daily_ds.get("10m_u_component_of_wind"))
                v10 = daily_ds.get("v10_max", daily_ds.get("10m_v_component_of_wind"))
                if u10 is None or v10 is None:
                    raise KeyError("Wind component variables not found in the dataset.")
                wind_speed = calculate_wind_speed(u10, v10)
                da_index = calculate_apparent_temperature(
                    daily_ds["t2m_mean"], u10, v10, daily_ds["d2m_mean"]
                )
            elif index_metric == "WBGT":
                da_index = calculate_wbgt_simple(
                    daily_ds["t2m_mean"], daily_ds["d2m_mean"]
                )
            else:
                raise ValueError(f"Unsupported index: {index_metric}")

            ds_combined = xr.Dataset({index_metric: da_index})

    except FileNotFoundError:
        LOGGER.error(f"Data file {input_file_name} does not exist.")

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
    monthly_means = monthly_means.rename(index_metric)
    monthly_means = monthly_means.rename({"forecast_month": "step"})
    ds_monthly = xr.Dataset({f"{index_metric}": monthly_means})

    # Ensure 'number' dimension starts from 1 instead of 0
    ds_monthly = ds_monthly.assign_coords(number=ds_monthly.number)

    # calculate ensemble statistics across members
    ds_stats = calculate_statistics_from_index(monthly_means)

    return ds_combined, ds_monthly, ds_stats


def calculate_tr_days(grib_file_path, index_metric, tr_threshold=20):
    """
    Calculates and saves the tropical nights index, defined as the number of nights where the minimum temperature remains at or above a threshold.
    The default is 20°C.

    Parameters
    ----------
    grib_file_path : str
        Path to the input GRIB data file containing temperature data. The file should be structured to include 2-meter temperature values (`t2m`).
    index_metric : str
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
        # Open the seasonal forecast data
        with xr.open_dataset(grib_file_path, engine="cfgrib") as ds:
            t2m_celsius = ds["t2m"] - 273.15
            daily_min_temp = t2m_celsius.resample(step="1D").min()

            # Convert valid_time to monthly periods for grouping
            valid_times = pd.to_datetime(ds.valid_time.values)
            forecast_months_str = valid_times.to_period("M").astype(str)
            step_to_month = dict(zip(ds.step.values, forecast_months_str))
            forecast_month_da = xr.DataArray(
                list(step_to_month.values()), coords=[ds.step], dims=["step"]
            )
        daily_min_temp.coords["forecast_month"] = forecast_month_da

        # Use the generic TR calculation with a configurable threshold
        tropical_nights = calculate_tr(daily_min_temp, tr_threshold=tr_threshold)

        # Count tropical nights by month
        tropical_nights_count = tropical_nights.groupby("forecast_month").sum(
            dim="step"
        )
        tropical_nights_count = tropical_nights_count.rename(index_metric)
        tropical_nights_count = tropical_nights_count.rename({"forecast_month": "step"})

        # Calculate ensemble statistics, if relevant
        ds_stats = calculate_statistics_from_index(tropical_nights_count)
        return None, tropical_nights_count, ds_stats

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

    # **Updated: Return `(None, None, None)` to avoid unpacking error**
    return None, None, None


def calculate_tx30_days(grib_file_path, index_metric):
    """
    Calculate TX30 index for seasonal forecast data, specifically counting the number of days with Tmax > 30°C.

    Parameters
    ----------
    grib_file_path : str
        Path to the GRIB file containing temperature data.
    index_metric : str
        Name of the TX30 index, typically "TX30".

    Returns
    -------
    tuple
        - None: No daily index is returned.
        - xarray.Dataset: Monthly count of TX30 days.
        - xarray.Dataset: Ensemble statistics of the TX30 index.
    """
    try:
        # Open the seasonal forecast data
        with xr.open_dataset(grib_file_path, engine="cfgrib") as ds:
            t2m_celsius = ds["t2m"] - 273.15
            daily_max_temp = t2m_celsius.resample(
                step="1D"
            ).max()  # Ensure daily max temp is calculated

            # Convert valid_time to monthly periods for grouping
            valid_times = pd.to_datetime(ds.valid_time.values)
            forecast_months_str = valid_times.to_period("M").astype(str)
            step_to_month = dict(zip(ds.step.values, forecast_months_str))
            forecast_month_da = xr.DataArray(
                list(step_to_month.values()), coords=[ds.step], dims=["step"]
            )
        daily_max_temp.coords["forecast_month"] = forecast_month_da

        # Use the generic TX30 calculation
        tx30_days = calculate_tx30(daily_max_temp)

        # Count TX30 days by month
        tx30_days_count = tx30_days.groupby("forecast_month").sum(dim="step")
        tx30_days_count = tx30_days_count.rename(index_metric)
        tx30_days_count = tx30_days_count.rename({"forecast_month": "step"})

        # Calculate ensemble statistics, if relevant
        ds_stats = calculate_statistics_from_index(tx30_days_count)

        return None, tx30_days_count, ds_stats

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except Exception as e:
        print(f"An error occurred: {e}")


def calculate_hw_days(
    grib_file_path, index_metric, threshold=27, min_duration=3, max_gap=0
):
    """
    Calculates and saves the heatwave days index, which is defined as the number of days in each month where
    the daily mean temperature exceeds a specified threshold for a minimum consecutive duration. This index
    helps to identify periods of extreme heat events.

    Parameters
    ----------
    grib_file_path : str
        Path to the input GRIB data file containing temperature data. The file should include 2-meter temperature values (`t2m`).
    index_metric : str
        The climate index being processed. This should specify the name for the heatwave days index, such as "HW" (Heatwave Days).
    threshold : float, optional
        Temperature threshold in Celsius that defines a heatwave day. Defaults to 27°C.
    min_duration : int, optional
        Minimum consecutive days required to define a heatwave event. Defaults to 3 days.
    max_gap : int, optional
        Maximum allowable gap (in days) within a heatwave event for it to still count as a single event. Defaults to 0, meaning no gaps are allowed.

    Returns
    -------
    tuple
        A tuple containing:
        - `None` : No daily index is returned for this calculation.
        - `xarray.Dataset` : The monthly count of heatwave days, stored as an `xarray.Dataset` with the index values and relevant metadata.
        - `xarray.Dataset` : Statistics calculated across the monthly heatwave days index values, representing ensemble statistics.

    Raises
    ------
    FileNotFoundError
        If the specified input GRIB file does not exist.
    Exception
        For any other errors encountered during data processing.
    """
    try:
        with xr.open_dataset(grib_file_path, engine="cfgrib") as ds:
            # Convert to Celsius and calculate daily mean temperature
            t2m_celsius = ds["t2m"] - 273.15
            daily_mean_temp = t2m_celsius.resample(step="1D").mean()

            # Convert valid_time to monthly periods
            valid_times = pd.to_datetime(ds.valid_time.values)
            forecast_months_str = valid_times.to_period("M").astype(str)
            step_to_month = dict(zip(ds.step.values, forecast_months_str))
            forecast_month_da = xr.DataArray(
                list(step_to_month.values()), coords=[ds.step], dims=["step"]
            )
            daily_mean_temp.coords["forecast_month"] = forecast_month_da

            # Initialize DataArray for heatwave days
            hw_days = xr.zeros_like(daily_mean_temp, dtype=int)

            # Apply `calculate_hw` individually for each member, latitude, and longitude
            for member in range(daily_mean_temp.sizes["number"]):
                for lat in range(daily_mean_temp.sizes["latitude"]):
                    for lon in range(daily_mean_temp.sizes["longitude"]):
                        temp_series = daily_mean_temp.isel(
                            number=member, latitude=lat, longitude=lon
                        ).values
                        hw_event_days = calculate_hw(
                            temp_series, threshold, min_duration, max_gap
                        )
                        hw_days.loc[
                            dict(
                                number=member,
                                latitude=daily_mean_temp.latitude[lat],
                                longitude=daily_mean_temp.longitude[lon],
                            )
                        ] = hw_event_days

            # Count heatwave days by month
            hw_days_count = hw_days.groupby("forecast_month").sum(dim="step")
            hw_days_count = hw_days_count.rename(index_metric)
            hw_days_count = hw_days_count.rename({"forecast_month": "step"})

            # Placeholder for statistics calculation
            ds_stats = calculate_statistics_from_index(hw_days_count)

            return None, hw_days_count, ds_stats

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None


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
    ds_stats = xr.Dataset(
        {
            "ensemble_mean": ensemble_mean,
            "ensemble_median": ensemble_median,
            "ensemble_max": ensemble_max,
            "ensemble_min": ensemble_min,
            "ensemble_std": ensemble_std,
        }
    )

    # add percentiles
    percentile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    ensemble_percentiles = dataarray.quantile(percentile_levels, dim="number")
    for level in percentile_levels:
        label = f"ensemble_p{int(level * 100)}"
        ds_stats[label] = ensemble_percentiles.sel(quantile=level)

    return ds_stats


def index_explanations(index_metric):
    """
    Provides a detailed explanation and required input data for a given climate index.

    Parameters
    ----------
    index_metric : str
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
        - "HUM" : Humidex
        - "RH" : Relative Humidity
        - "AT" : Apparent Temperature
        - "WBGT" : Wet Bulb Globe Temperature (Simple)

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
            "explanation": "Heat Index Adjusted: This indicator measures apparent "
            "temperature, considering both air temperature and humidity, providing a more "
            "accurate perception of how hot it feels.",
            "input_data": ["2m temperature (t2m)", "2m dewpoint temperature (d2m)"],
        },
        "HIS": {
            "explanation": "Heat Index Simplified: This indicator is a simpler version of the "
            "Heat Index, focusing on a quick estimate of perceived heat based on temperature "
            "and humidity.",
            "input_data": ["2m temperature (t2m)", "2m dewpoint temperature (d2m)"],
        },
        "Tmean": {
            "explanation": "Mean Temperature: This indicator calculates the average temperature "
            "over the specified period.",
            "input_data": ["2m temperature (t2m)"],
        },
        "Tmin": {
            "explanation": "Minimum Temperature: This indicator tracks the lowest temperature "
            "recorded over a specified period.",
            "input_data": ["2m temperature (t2m)"],
        },
        "Tmax": {
            "explanation": "Maximum Temperature: This indicator tracks the highest temperature "
            "recorded over a specified period.",
            "input_data": ["2m temperature (t2m)"],
        },
        "HW": {
            "explanation": "Heat Wave: This indicator identifies heat waves, defined as at least "
            "3 consecutive days with temperatures exceeding a certain threshold.",
            "input_data": ["2m temperature (t2m)"],
        },
        "TR": {
            "explanation": "Tropical Nights: This indicator counts the number of nights where "
            "the minimum temperature remains above a certain threshold, typically 20°C.",
            "input_data": ["2m temperature (t2m)"],
        },
        "TX30": {
            "explanation": "Hot Days: This indicator counts the number of days where the maximum "
            "temperature exceeds 30°C.",
            "input_data": ["2m temperature (t2m)"],
        },
        "HUM": {
            "explanation": "Humidex: Perceived temperature combining temperature and humidity.",
            "input_data": ["2m temperature (t2m)", "2m dewpoint temperature (d2m)"],
        },
        "RH": {
            "explanation": "Relative Humidity: Measures humidity as a percentage.",
            "input_data": ["2m temperature (t2m)", "2m dewpoint temperature (d2m)"],
        },
        "AT": {
            "explanation": "Apparent Temperature: Perceived temperature considering wind and humidity.",
            "input_data": [
                "2m temperature (t2m)",
                "10m wind speed",
                "2m dewpoint temperature (d2m)",
            ],
        },
        "WBGT": {
            "explanation": "Wet Bulb Globe Temperature (Simple): Heat stress index combining temperature and humidity.",
            "input_data": ["2m temperature (t2m)", "2m dewpoint temperature (d2m)"],
        },
    }

    # Return the explanation if found; otherwise, provide valid index options
    return index_explanations.get(
        index_metric,
        {"error": "Unknown index", "valid_indices": list(index_explanations.keys())},
    )
