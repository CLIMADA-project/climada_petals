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

---

Module for calculating various indices, including:
- Relative Humidity
- Heat Index (Simplified & Adjusted)
- Humidex
- Wind Speed
- Apparent Temperature
- Heatwaves (HW)
- Tropical Nights (TR)
"""

import xarray as xr
import numpy as np
import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)

# ------------------------
# Temperature Conversions
# ------------------------


def kelvin_to_fahrenheit(kelvin):
    return (kelvin - 273.15) * 9 / 5 + 32


def fahrenheit_to_kelvin(fahrenheit):
    return (fahrenheit - 32) * 5 / 9 + 273.15


def celsius_to_kelvin(temp_c):
    return temp_c + 273.15


def kelvin_to_celsius(temp_k):
    return temp_k - 273.15


# ------------------------
# Relative Humidity Calculation
# ------------------------
def calculate_relative_humidity(t2k, tdk, as_percentage=True):
    """
    Calculates the relative humidity with the option to return it either as a decimal value (0-1) or as a percentage (0-100).

    Parameters
    ----------
    t2k : float or array-like
        2-meter air temperature in Kelvin.
    tdk : float or array-like
        2-meter dew point temperature in Kelvin.
    as_percentage : bool, optional
        If True, returns relative humidity as a percentage (0-100). If False, returns it as a fraction (0-1).
        Default is True.

    Returns
    -------
    float or array-like
        Relative humidity as a percentage (0-100) or as a decimal value (0-1), depending on the `as_percentage` setting.
    """
    t2c = kelvin_to_celsius(t2k)
    tdc = kelvin_to_celsius(tdk)

    es = 6.11 * 10.0 ** (7.5 * t2c / (237.3 + t2c))
    e = 6.11 * 10.0 ** (7.5 * tdc / (237.3 + tdc))

    rh = e / es
    if as_percentage:
        rh *= 100  # Umwandlung in Prozent

    rh = np.clip(rh, 0, 100 if as_percentage else 1)  # Begrenzung auf sinnvolle Werte
    return rh


# ------------------------
# Heat Index Calculations
# ------------------------
def calculate_heat_index_simplified(t2k, tdk):
    """
    Calculates the simplified heat index (HIS) based on temperature and dewpoint temperature.

    The simplified heat index formula is only valid for temperatures above 20°C, as the heat index is specifically designed for warm to hot conditions where humidity significantly influences perceived temperature. Below 20°C, the function returns the actual air temperature instead of applying the heat index formula.

    The heat index is an empirical measure that estimates the perceived temperature by incorporating the effects of both temperature and humidity. It is commonly used in meteorology and climate studies to assess heat stress.

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

    Acknowledgment
    --------------
    This function is based on the Thermofeel library. The original implementation and methodology can be found in:
    Brimicombe, C., Bröde, P., and Calvi, P. (2022). Thermofeel: A python thermal comfort indices library. *SoftwareX*, 17, 101005. DOI: https://doi.org/10.1016/j.softx.2022.101005
    """

    t2_c = kelvin_to_celsius(t2k)
    rh = calculate_relative_humidity(t2k, tdk)  # Compute RH internally

    hiarray = [
        8.784695,
        1.61139411,
        2.338549,
        0.14611605,
        1.2308094e-2,
        1.6424828e-2,
        2.211732e-3,
        7.2546e-4,
        3.582e-6,
    ]

    hi = np.copy(t2_c)  # Default to air temperature

    # Apply heat index formula only where T > 20°C
    hi_filter = np.where(t2_c > 20)
    hi[hi_filter] = (
        -hiarray[0]
        + hiarray[1] * t2_c[hi_filter]
        + hiarray[2] * rh[hi_filter]
        - hiarray[3] * t2_c[hi_filter] * rh[hi_filter]
        - hiarray[4] * t2_c[hi_filter] ** 2
        - hiarray[5] * rh[hi_filter] ** 2
        + hiarray[6] * t2_c[hi_filter] ** 2 * rh[hi_filter]
        + hiarray[7] * t2_c[hi_filter] * rh[hi_filter] ** 2
        - hiarray[8] * t2_c[hi_filter] ** 2 * rh[hi_filter] ** 2
    )

    return hi  # Returns in Celsius


def calculate_heat_index_adjusted(t2k, tdk):
    """
    Calculates the adjusted heat index based on temperature and dewpoint temperature.

    This function refines the standard heat index calculation by incorporating adjustments
    for extreme values of temperature and relative humidity. The adjustments improve accuracy
    in conditions where the simplified formula may not be sufficient, particularly for high temperatures
    (> 80°F / ~27°C) and very low or high humidity levels.

    Parameters
    ----------
    t2k : float or array-like
        2-meter air temperature in Kelvin. This value represents the temperature measured at a height of 2 meters above ground level.
    tdk : float or array-like
        2-meter dewpoint temperature in Kelvin. The dewpoint temperature is the temperature at which the air becomes saturated and condensation begins.

    Returns
    -------
        float or array-like
        Adjusted heat index in degrees Celsius. This metric indicates the perceived temperature
        based on the combined effect of temperature and relative humidity.
        - If T ≤ 26.7°C (80°F), the function returns a simplified index.
        - If T > 26.7°C (80°F), additional corrections are applied to refine the heat index value.

    Acknowledgment
    --------------
    This function is based on the Thermofeel library. The original implementation and methodology can be found in:
    Brimicombe, C., Bröde, P., and Calvi, P. (2022). Thermofeel: A python thermal comfort indices library. *SoftwareX*, 17, 101005. DOI: https://doi.org/10.1016/j.softx.2022.101005
    """
    rh = calculate_relative_humidity(t2k, tdk)
    t2_f = kelvin_to_fahrenheit(t2k)

    hiarray = [
        42.379,
        2.04901523,
        10.1433312,
        0.22475541,
        0.00683783,
        0.05481717,
        0.00122874,
        0.00085282,
        0.00000199,
    ]

    hi_initial = 0.5 * (t2_f + 61 + ((t2_f - 68) * 1.2) + (rh * 0.094))

    hi = (
        -hiarray[0]
        + hiarray[1] * t2_f
        + hiarray[2] * rh
        - hiarray[3] * t2_f * rh
        - hiarray[4] * t2_f**2
        - hiarray[5] * rh**2
        + hiarray[6] * t2_f**2 * rh
        + hiarray[7] * t2_f * rh**2
        - hiarray[8] * t2_f**2 * rh**2
    )

    hi_filter1 = t2_f > 80
    hi_filter2 = t2_f < 112
    hi_filter3 = rh <= 13
    hi_filter4 = t2_f < 87
    hi_filter5 = rh > 85
    hi_filter6 = t2_f < 80
    hi_filter7 = (hi_initial + t2_f) / 2 < 80

    f_adjust1 = hi_filter1 & hi_filter2 & hi_filter3
    f_adjust2 = hi_filter1 & hi_filter4 & hi_filter5

    adjustment1 = (
        (13 - rh[f_adjust1]) / 4 * np.sqrt((17 - np.abs(t2_f[f_adjust1] - 95)) / 17)
    )

    adjustment2 = (rh[f_adjust2] - 85) / 10 * ((87 - t2_f[f_adjust2]) / 5)

    adjustment3 = 0.5 * (
        t2_f[hi_filter6]
        + 61.0
        + ((t2_f[hi_filter6] - 68.0) * 1.2)
        + (rh[hi_filter6] * 0.094)
    )

    # Apply adjustments
    hi[f_adjust1] -= adjustment1
    hi[f_adjust2] += adjustment2
    hi[hi_filter6] = adjustment3

    # Apply simplified formula for HI < 80°F
    hi[hi_filter7] = hi_initial[hi_filter7]

    # Convert heat index back to Kelvin
    hi_k = fahrenheit_to_kelvin(hi)
    hi_c = hi_k - 273.15
    return hi_c


# ------------------------
# Humidex Calculation
# ------------------------
def calculate_humidex(t2_k, td_k):
    """
    Calculate Humidex (°C)
    The Humidex is a thermal comfort index that represents the perceived temperature
    by incorporating both air temperature and humidity. It is commonly used in
    meteorology to assess heat stress and human discomfort in warm and humid conditions.
    The higher the Humidex value, the greater the level of discomfort.

    Parameters
    ----------
    t2_k : float or np.array
        2m temperature in Kelvin.
    td_k : float or np.array
        Dew point temperature in Kelvin.

    Returns
    -------
    float or np.array
        Humidex in Celsius.

    Acknowledgment
    --------------
    This function is based on the Thermofeel library. The original implementation and methodology can be found in:
    Brimicombe, C., Bröde, P., and Calvi, P. (2022). Thermofeel: A python thermal comfort indices library. *SoftwareX*, 17, 101005. DOI: https://doi.org/10.1016/j.softx.2022.101005
    """
    vp = 6.11 * np.exp(5417.7530 * ((1 / 273.16) - (1 / td_k)))
    h = 0.5555 * (vp - 10.0)
    humidex_k = t2_k + h
    humidex = kelvin_to_celsius(humidex_k)
    return humidex


# ------------------------
# Wind Speed Calculation
# ------------------------
def calculate_wind_speed(u10, v10):
    """
    Calculate wind speed (m/s) from the u and v components of the wind.

    Parameters
    ----------
    u10 : float or np.array
        10m eastward wind component in m/s.
    v10 : float or np.array
        10m northward wind component in m/s.

    Returns
    -------
    float or np.array
        Wind speed in m/s.

    Acknowledgment
    --------------
    This function is based on ECMWF (European Centre for Medium-Range Weather Forecasts) documentation for wind calculations https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398
    """
    return np.sqrt(u10**2 + v10**2)


# ------------------------
# Apparent Temperature Calculation
# ------------------------
def calculate_apparent_temperature(t2_k, u10, v10, tdk):
    """
    Calculate Apparent Temperature (°C)

    Parameters
    ----------
    t2_k : float or np.array
        2m temperature in Kelvin. Represents the air temperature measured at a height of 2 meters.
    u10 : float or np.array
        10m eastward wind component in m/s. Indicates the wind speed in the eastward direction at a height of 10 meters.
    v10 : float or np.array
        10m northward wind component in m/s. Indicates the wind speed in the northward direction at a height of 10 meters.
    tdk : float or np.array
        2m dewpoint temperature in Kelvin. Dew point temperature at which air becomes saturated and condensation begins.

    Returns
    -------
    float or np.array
        Apparent temperature in Celsius. This metric represents the perceived temperature considering both wind speed and humidity, accounting for heat loss or gain due to environmental factors.

    Acknowledgment
    --------------
    This function is based on the Thermofeel library. The original implementation and methodology can be found in:
    Brimicombe, C., Bröde, P., and Calvi, P. (2022). Thermofeel: A python thermal comfort indices library. *SoftwareX*, 17, 101005. DOI: https://doi.org/10.1016/j.softx.2022.101005
    """
    t2_c = kelvin_to_celsius(t2_k)
    rh = calculate_relative_humidity(t2_k, tdk)
    va = calculate_wind_speed(u10, v10)
    e = calculate_nonsaturation_vapour_pressure(t2_k, rh)
    at = t2_c + 0.33 * e - 0.7 * va - 4
    return at


def calculate_nonsaturation_vapour_pressure(t2_k, rh):
    """
    Calculate Non-Saturated Vapour Pressure (hPa)

    Parameters
    ----------
    t2_k : float or np.array
        2m temperature in Kelvin. Represents the temperature measured at 2 meters above ground level.
    rh : float or np.array
        Relative humidity as a percentage. Indicates the amount of moisture present in the air relative to the maximum it can hold.

    Returns
    -------
    float or np.array
        Non-saturated vapour pressure in hPa (equivalent to mBar). This pressure reflects the partial pressure of water vapor in air under non-saturated conditions.

    Acknowledgment
    --------------
    This function is based on the Thermofeel library. The original implementation and methodology can be found in:
    Brimicombe, C., Bröde, P., and Calvi, P. (2022). Thermofeel: A python thermal comfort indices library. *SoftwareX*, 17, 101005. DOI: https://doi.org/10.1016/j.softx.2022.101005
    """
    t2_c = kelvin_to_celsius(t2_k)
    ens = rh / 100 * 6.105 * np.exp(17.27 * t2_c / (237.7 + t2_c))
    return ens


# ------------------------
# Wet Bulb Globe Temperature (Simple) Calculation
# ------------------------
def calculate_wbgt_simple(t2_k, tdk):
    """
    Calculate Wet Bulb Globe Temperature (Simple)

    Parameters
    ----------
    t2_k : float or np.array
        2m temperature in Kelvin. This is the standard air temperature measured at a height of 2 meters.
    tdk : float or np.array
        Dew point temperature in Kelvin. Used to calculate relative humidity and overall heat stress.

    Returns
    -------
    float or np.array
        Wet Bulb Globe Temperature in Celsius. This index is used for heat stress assessments, combining temperature, humidity, and other factors to determine the perceived heat risk.

    Acknowledgment
    --------------
    This function is based on the Thermofeel library. The original implementation and methodology can be found in:
    Brimicombe, C., Bröde, P., and Calvi, P. (2022). Thermofeel: A python thermal comfort indices library. *SoftwareX*, 17, 101005. DOI: https://doi.org/10.1016/j.softx.2022.101005
    """
    rh = calculate_relative_humidity(t2_k, tdk)
    t2_c = kelvin_to_celsius(t2_k)
    e = calculate_nonsaturation_vapour_pressure(t2_k, rh)
    wbgt = 0.567 * t2_c + 0.393 * e + 3.94
    return wbgt


# ------------------------
# Heat Index Calculations
# ------------------------
def calculate_heat_index(da_t2k, da_tdk, index):
    """
    Calculates the heat index based on temperature and dewpoint temperature using
    either the simplified or adjusted formula as implemented in the Thermofeel library.

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

    Acknowledgment
    --------------
    This function is based on the Thermofeel library. The original implementation and methodology can be found in:
    Brimicombe, C., Bröde, P., and Calvi, P. (2022). Thermofeel: A python thermal comfort indices library. *SoftwareX*, 17, 101005. DOI: https://doi.org/10.1016/j.softx.2022.101005
    """
    if index == "HIS":
        index_long_name = "heat_index_simplified"
        unit = "degC"
        index_metric_data = calculate_heat_index_simplified(da_t2k.data, da_tdk.data)
    elif index == "HIA":
        index_long_name = "heat_index_adjusted"
        unit = "degC"
        index_metric_data = calculate_heat_index_adjusted(da_t2k.data, da_tdk.data)
    else:
        LOGGER.error(f'Index {index} is not implemented, use either "HIS" or "HIA".')
        return None
    da_index = xr.DataArray(
        index_metric_data,
        coords=da_tdk.coords,
        dims=da_tdk.dims,
        attrs={"description": index_long_name, "units": unit},
    )
    return da_index


# ------------------------
# Tropical Nights Calculation
# ------------------------
def calculate_tr(temperature_data, tr_threshold=20):
    """
    Calculate the Tropical Nights index, defined as the number of nights with minimum temperature above a given threshold.

    Parameters
    ----------
    temperature_data : xarray.DataArray
        DataArray containing daily minimum temperatures in Celsius.
    tr_threshold : float, optional
        Temperature threshold in Celsius for a tropical night. Default is 20°C.

    Returns
    -------
    xarray.DataArray
        Boolean DataArray where True indicates nights with Tmin > threshold.
    """
    tropical_nights = temperature_data >= tr_threshold
    return tropical_nights


# ------------------------
# TX30 Calculation
# ------------------------
def calculate_tx30(temperature_data, threshold=30):
    """
    Calculate TX30, the number of days with maximum temperature above the given threshold (default is 30°C).

    Parameters
    ----------
    temperature_data : xarray.DataArray
        DataArray containing daily maximum temperatures in Celsius. Can be from any dataset, not specific to seasonal forecasts.
    threshold : float, optional
        Temperature threshold in Celsius for a TX30 day. Default is 30°C.

    Returns
    -------
    xarray.DataArray
        Boolean DataArray where True indicates days where Tmax > threshold.
    """
    # Check that the input data is daily data. The caller should ensure data is resampled if needed.
    tx30_days = temperature_data > threshold
    return tx30_days


# ------------------------
# Heatwave Calculation
# ------------------------
def calculate_hw_1D(
    temperatures: np.ndarray,
    threshold: float = 27,
    min_duration: int = 3,
    max_gap: int = 0,
) -> list:
    """
    Identify and define heatwave events based on a sequence of daily temperatures.

    This function scans an array of temperature data to detect periods of heatwaves,
    defined as consecutive days where temperatures exceed a given threshold for a minimum duration.
    If two such periods are separated by days with temperatures below the threshold but within a specified maximum gap,
    they are merged into one continuous heatwave event.

    Parameters
    ----------
    temperatures : np.ndarray
        Array of daily temperatures.
    threshold : float, optional
        Temperature threshold above which days are considered part of a heatwave. Default is 27°C.
    min_duration : int, optional
        Minimum number of consecutive days required to define a heatwave event. Default is 3 days.
    max_gap : int, optional
        Maximum allowed gap (in days) of below-threshold temperatures to merge two consecutive heatwave events into one. Default is 0 days.

    Returns
    -------
        np.ndarray
        A binary mask (1D array) of the same length as `temperatures`, where:
        - `1` indicates a heatwave day.
        - `0` indicates a non-heatwave day.

    Acknowledgment
    --------------
    Adapted from Modelling marine heatwaves impact on shallow and upper mesophotic tropical coral reefs DOI:10.1088/1748-9326/ad89df
    """

    hw_days = np.zeros(len(temperatures), dtype=int)
    binary_array = np.where(temperatures >= threshold, 1, 0)
    events = []
    prev_continous_ones = 0

    for i, value in enumerate(binary_array):
        if value == 0:
            prev_continous_ones = 0
        elif value == 1:
            prev_continous_ones += 1
            if prev_continous_ones == min_duration:
                new_event_start = i - min_duration + 1
                should_merge_with_latest_event = (
                    len(events) > 0 and (new_event_start - events[-1][1] - 1) <= max_gap
                )
                if should_merge_with_latest_event:
                    events[-1] = (events[-1][0], i)
                else:
                    events.append((new_event_start, i))
            if prev_continous_ones >= min_duration:
                events[-1] = (events[-1][0], i)

    for start, end in events:
        hw_days[start : end + 1] = 1

    return hw_days


def calculate_hw(
    daily_mean_temp,
    threshold: float = 27,
    min_duration: int = 3,
    max_gap: int = 0,
    label_time_step="step",
):
    """
    Identify and define heatwave events based on a sequence of daily mean temperatures.

    This function detects heatwave events by applying a threshold-based approach to
    an xarray DataArray of daily mean temperatures. A heatwave is defined as a period
    where temperatures exceed a specified threshold for a minimum number of consecutive days.
    If two such periods are separated by a gap of below-threshold temperatures within
    a given maximum gap length, they are merged into a single heatwave event.

    Parameters
    ----------
    daily_mean_temp : xarray.DataArray
        An xarray DataArray containing daily mean temperatures. The time dimension should be labeled
        according to `label_time_step`.
    threshold : float, optional
        Temperature threshold above which days are considered part of a heatwave. Default is 27°C.
    min_duration : int, optional
        Minimum number of consecutive days required to define a heatwave event. Default is 3 days.
    max_gap : int, optional
        Maximum allowed gap (in days) of below-threshold temperatures to merge two consecutive
        heatwave events into one. Default is 0 days.
    label_time_step : str, optional
        Name of the time dimension in `daily_mean_temp`. Default is "step".

    Returns
    -------
    xarray.DataArray
        A DataArray of the same shape as `daily_mean_temp`, where heatwave periods
        are labeled with 1 (heatwave) and 0 (non-heatwave).

    Notes
    -----
    This function leverages `xarray.apply_ufunc` to apply the `calculate_hw_1D` function
    efficiently across all grid points, supporting vectorized operations and parallelized
    computation with Dask.
    """
    return xr.apply_ufunc(
        calculate_hw_1D,
        daily_mean_temp,
        input_core_dims=[[label_time_step]],
        output_core_dims=[[label_time_step]],
        vectorize=True,
        dask="parallelized",
        kwargs={
            "threshold": threshold,
            "min_duration": min_duration,
            "max_gap": max_gap,
        },
        output_dtypes=[int],
    )
