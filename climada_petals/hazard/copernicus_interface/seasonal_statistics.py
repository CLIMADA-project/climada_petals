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

import numpy as np
import xarray as xr
import pandas as pd
import logging

import climada_petals.hazard.copernicus_interface.heat_index as heat_index
from climada_petals.hazard.copernicus_interface.heat_index import kelvin_to_celsius


LOGGER = logging.getLogger(__name__)


def calculate_heat_indices_metrics(
    input_file_name,
    index_metric,
    tr_threshold=20,
    hw_threshold=27,
    hw_min_duration=3,
    hw_max_gap=0,
):
    """
    Computes heat indices or temperature-related metrics based on the specified climate index.

    Parameters
    ----------
    input_file_name : str
        Path to the input data file containing the variables required for the computation of the selected index. 
        The file should be in NetCDF (.nc) or GRIB format and contain relevant atmospheric data such as temperature, 
        dewpoint temperature, humidity, or wind speed. Information on the required variables for each index 
        can be found in the `index_definitions` class.

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
    tr_threshold : float, optional
        Temperature threshold (°C) for computing tropical nights (TR). 
        Default is 20°C, meaning nights with Tmin > 20°C are considered tropical.
    hw_threshold : float, optional
        Temperature threshold (°C) for detecting a heatwave (HW). 
        Default is 27°C, meaning a heatwave occurs if the temperature remains above this threshold for multiple days.
    hw_min_duration : int, optional
        Minimum consecutive days for a heatwave event to be detected. 
        Default is 3 days.
    hw_max_gap : int, optional
        Maximum allowable gap (in days) between heatwave days for them to still be considered part of the same event.
        Default is 0 days, meaning no gaps are allowed.

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

    engine = "cfgrib" if index_metric in ["TR", "TX30", "HW"] else "netcdf4"

    try:
        with xr.open_dataset(input_file_name, engine=engine) as daily_ds:
            # Handling various indices
            if index_metric == "Tmean":
                da_index = kelvin_to_celsius(daily_ds["t2m_mean"])  # Kelvin to Celsius
                da_index.attrs["units"] = "degC"
            elif index_metric == "Tmax":
                da_index = kelvin_to_celsius(daily_ds["t2m_max"].resample(step="1D").max())
                da_index.attrs["units"] = "degC"
            elif index_metric == "Tmin":
                da_index = kelvin_to_celsius(daily_ds["t2m_min"].resample(step="1D").min())
                da_index.attrs["units"] = "degC"
            elif index_metric in ["HIS", "HIA"]:
                da_index = heat_index.calculate_heat_index(
                    daily_ds["t2m_mean"], daily_ds["d2m_mean"], index_metric
                )
            elif index_metric == "RH":
                da_index = heat_index.calculate_relative_humidity(
                    daily_ds["t2m_mean"], daily_ds["d2m_mean"]
                )
            elif index_metric == "HUM":
                da_index = heat_index.calculate_humidex(
                    daily_ds["t2m_mean"], daily_ds["d2m_mean"]
                )
            elif index_metric == "AT":
                u10 = daily_ds.get("u10_max", daily_ds.get("10m_u_component_of_wind"))
                v10 = daily_ds.get("v10_max", daily_ds.get("10m_v_component_of_wind"))
                if u10 is None or v10 is None:
                    raise KeyError("Wind component variables not found in the dataset.")
                da_index = heat_index.calculate_apparent_temperature(
                    daily_ds["t2m_mean"], u10, v10, daily_ds["d2m_mean"]
                )
            elif index_metric == "WBGT":
                da_index = heat_index.calculate_wbgt_simple(
                    daily_ds["t2m_mean"], daily_ds["d2m_mean"]
                )
            elif index_metric == "TR":
                daily_ds["t2m"] = daily_ds["t2m"] - 273.15
                daily_min_temp = daily_ds["t2m"].resample(step="1D").min()
                da_index = heat_index.calculate_tr(
                    daily_min_temp, tr_threshold=tr_threshold
                )
            elif index_metric == "TX30":
                daily_ds["t2m"] = daily_ds["t2m"] - 273.15
                daily_max_temp = daily_ds["t2m"].resample(step="1D").max()
                da_index = heat_index.calculate_tx30(daily_max_temp)
            elif index_metric == "HW":
                daily_ds["t2m"] = daily_ds["t2m"] - 273.15
                daily_mean_temp = daily_ds["t2m"].resample(step="1D").mean()
                da_index = heat_index.calculate_hw(
                    daily_mean_temp, hw_threshold, hw_min_duration, hw_max_gap
                )
            else:
                raise ValueError(f"Unsupported index: {index_metric}")

            ds_combined = xr.Dataset({index_metric: da_index})

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}")
    except Exception as e:
        raise e

    # add monthly period label
    da_index.coords["forecast_month"] = monthly_periods_from_valid_times(daily_ds)

    # compute monthly means
    method = "count" if index_metric in ["TR", "TX30", "HW"] else "mean"
    ds_monthly = calculate_monthly_dataset(da_index, index_metric, method)

    # calculate ensemble statistics across members
    ds_stats = calculate_statistics_from_index(ds_monthly[f"{index_metric}"])

    return ds_combined, ds_monthly, ds_stats


def monthly_periods_from_valid_times(ds):
    """Create monthly labels from valid times of a dataframe

    Parameters
    ----------
    ds : xr.DataSet
        Dataset of daily values

    Returns
    -------
    xr.DataArray
        DataArray with monthly labels
    """
    valid_times = pd.to_datetime(ds.valid_time.values)
    forecast_months_str = valid_times.to_period("M").astype(str)
    step_to_month = dict(zip(ds.step.values, forecast_months_str))
    return xr.DataArray(list(step_to_month.values()), coords=[ds.step], dims=["step"])


def calculate_monthly_dataset(da_index, index_metric, method):
    """Calculate monthly means from daily data

    Parameters
    ----------
    da_index : xr.Dataset
        Dataset containing daily data
    index_metric : str
        index to be computed
    method : str
        method to combine daily data to monthly data. Available are "mean" and "count".

    Returns
    -------
    xr.DataSet
        Dataset of monthly averages
    """
    if method == "mean":
        monthly = da_index.groupby("forecast_month").mean(dim="step")
    elif method == "count":
        monthly = da_index.groupby("forecast_month").sum(dim="step")
    else:
        raise ValueError(
            f"Unknown method {method} to compute monthly data. Please use 'mean' or 'count'."
        )
    monthly = monthly.rename(index_metric)
    monthly = monthly.rename({"forecast_month": "step"})
    ds_monthly = xr.Dataset({f"{index_metric}": monthly})

    # Ensure 'number' dimension starts from 1 instead of 0
    ds_monthly = ds_monthly.assign_coords(number=ds_monthly.number)

    return ds_monthly


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
