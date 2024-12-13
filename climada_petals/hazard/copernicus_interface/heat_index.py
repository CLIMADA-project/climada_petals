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
import numpy as np
import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)


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

    Acknowledgment
    --------------
    This function is based on the Thermofeel library. The original implementation and methodology can be found in:
    Brimicombe, C., Bröde, P., and Calvi, P. (2022). Thermofeel: A python thermal comfort indices library. *SoftwareX*, 17, 101005. DOI: https://doi.org/10.1016/j.softx.2022.101005
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

    Acknowledgment
    --------------
    This function is based on the Thermofeel library. The original implementation and methodology can be found in:
    Brimicombe, C., Bröde, P., and Calvi, P. (2022). Thermofeel: A python thermal comfort indices library. *SoftwareX*, 17, 101005. DOI: https://doi.org/10.1016/j.softx.2022.101005
    """
    t2c = t2k - 273.15
    rh = calculate_relative_humidity_percent(t2k, tdk)
    hi = (
        -8.784695
        + 1.61139411 * t2c
        + 2.338549 * rh
        - 0.14611605 * t2c * rh
        - 1.2308094e-2 * t2c**2
        - 1.6424828e-2 * rh**2
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

    Acknowledgment
    --------------
    This function is based on the Thermofeel library. The original implementation and methodology can be found in:
    Brimicombe, C., Bröde, P., and Calvi, P. (2022). Thermofeel: A python thermal comfort indices library. *SoftwareX*, 17, 101005. DOI: https://doi.org/10.1016/j.softx.2022.101005
    """
    rh = calculate_relative_humidity_percent(t2k, tdk)
    t2f = (t2k - 273.15) * 9 / 5 + 32
    hi = (
        -42.379
        + 2.04901523 * t2f
        + 10.14333127 * rh
        - 0.22475541 * t2f * rh
        - 0.00683783 * t2f**2
        - 0.05481717 * rh**2
        + 0.00122874 * t2f**2 * rh
        + 0.00085282 * t2f * rh**2
        - 0.00000199 * t2f**2 * rh**2
    )
    return (hi - 32) * 5 / 9  # converted back to Celsius


# Conversion utility functions consistent with thermofeel
def celsius_to_kelvin(temp_c):
    return temp_c + 273.15


def kelvin_to_celsius(temp_k):
    return temp_k - 273.15


def calculate_relative_humidity(t2_k, td_k):
    """
    Calculate Relative Humidity (%)

    Parameters
    ----------
    t2_k : float or np.array
        2m temperature in Kelvin.
    td_k : float or np.array
        Dew point temperature in Kelvin.

    Returns
    -------
    float or np.array
        Relative humidity in percentage.

    Acknowledgment
    --------------
    This function is based on the Thermofeel library. The original implementation and methodology can be found in:
    Brimicombe, C., Bröde, P., and Calvi, P. (2022). Thermofeel: A python thermal comfort indices library. *SoftwareX*, 17, 101005. DOI: https://doi.org/10.1016/j.softx.2022.101005
    """
    t2_c = kelvin_to_celsius(t2_k)
    td_c = kelvin_to_celsius(td_k)

    es = 6.11 * 10 ** (7.5 * t2_c / (237.3 + t2_c))
    e = 6.11 * 10 ** (7.5 * td_c / (237.3 + td_c))
    rh = (e / es) * 100
    return rh


def calculate_humidex(t2_k, td_k):
    """
    Calculate Humidex (°C)

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
    t2_c = kelvin_to_celsius(t2_k)
    vp = 6.11 * np.exp(5417.7530 * ((1 / 273.16) - (1 / td_k)))
    h = 0.5555 * (vp - 10.0)
    humidex = t2_c + h
    return humidex


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
    This function is based on the Thermofeel library. The original implementation and methodology can be found in:
    Brimicombe, C., Bröde, P., and Calvi, P. (2022). Thermofeel: A python thermal comfort indices library. *SoftwareX*, 17, 101005. DOI: https://doi.org/10.1016/j.softx.2022.101005
    """
    return np.sqrt(u10**2 + v10**2)


def calculate_apparent_temperature(t2_k, u10, v10, d2m_k):
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
    d2m_k : float or np.array
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
    rh = calculate_relative_humidity_percent(t2_k, d2m_k)
    va = calculate_wind_speed(u10, v10)
    e = rh / 100 * 6.105 * np.exp(17.27 * t2_c / (237.7 + t2_c))
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
    rh = calculate_relative_humidity_percent(t2_k, tdk)
    t2_c = kelvin_to_celsius(t2_k)
    e = calculate_nonsaturation_vapour_pressure(t2_k, rh)
    wbgt = 0.567 * t2_c + 0.393 * e + 3.94
    return wbgt


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


def calculate_tr(temperature_data, tr_threshold=20):
    """
    Calculate the Tropical Nights index, defined as the number of nights with minimum temperature above a given threshold.

    Parameters
    ----------
    temperature_data : xarray.DataArray
        DataArray containing daily minimum temperatures in Celsius.
    threshold : float, optional
        Temperature threshold in Celsius for a tropical night. Default is 20°C.

    Returns
    -------
    xarray.DataArray
        Boolean DataArray where True indicates nights with Tmin > threshold.
    """
    tropical_nights = temperature_data >= tr_threshold
    return tropical_nights


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
    list
        List of tuples representing start and end indices of detected heatwaves. Each tuple indicates the beginning and ending day of a heatwave period.
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
