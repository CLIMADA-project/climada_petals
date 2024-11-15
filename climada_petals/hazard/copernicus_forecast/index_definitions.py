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

This script defines the specifications and explanations for various indices used in seasonal forecasts
"""

import xarray as xr
import pandas as pd
import logging


"""
A module for defining climate index parameters and providing explanatory details.

The `index_definitions` module centralizes the definitions and explanations of various climate indices 
used within the seasonal forecast workflow. This module specifies necessary input variables, filename conventions, 
and index descriptions. Ensuring data consistency within the seasonal forecast module script.

The script includes parameter definitions and detailed explanations for a range of climate indices, such as Mean Temperature (Tmean), 
Tropical Nights (TR), and Heat Wave (HW). It is structured to enhance usability and extensibility across forecast-based climate risk 
assessments by enabling automated index configuration and documentation.

Attributes
----------
VAR_SPECS : dict
    Dictionary defining specifications for input variables commonly used in climate indices, including units, 
    standard names, and short names.

Methods
-------
get_index_params(index)
    Retrieves required parameters for a specific climate index, such as input variables and filename conventions.

index_explanations(index_metric)
    Provides a detailed explanation and lists required input data for a specified climate index.

Example Usage
-------------
>>> from index_definitions import get_index_params, index_explanations
>>> params = get_index_params("TR")
>>> print(params["variables"])
>>> explanation = index_explanations("HW")
>>> print(explanation["explanation"])

Notes
-----
This module relies on the `xarray` and `pandas` libraries for array manipulation and datetime handling, 
and `logging` for error and info logging.
"""


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
    "10m_u_component_of_wind": {
        "unit": "m/s",
        "standard_name": "eastward_wind",
        "short_name": "u10",
        "full_name": "10m_u_component_of_wind",
    },
    "10m_v_component_of_wind": {
        "unit": "m/s",
        "standard_name": "northward_wind",
        "short_name": "v10",
        "full_name": "10m_v_component_of_wind",
    },
    "10m_wind_gust_since_previous_post_processing": {
        "unit": "m/s",
        "standard_name": "wind_gust",
        "short_name": "wind_gust10m",
        "full_name": "10m_wind_gust_since_previous_post_processing",
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
        "RH": {
            "variables": [
                VAR_SPECS["2m_temperature"]["full_name"],
                VAR_SPECS["2m_dewpoint_temperature"]["full_name"],
            ],
            "filename_lead": "2m_temps",
            "index_long_name": "Relative_Humidity",
        },
        "HUM": {
            "variables": [
                VAR_SPECS["2m_temperature"]["full_name"],
                VAR_SPECS["2m_dewpoint_temperature"]["full_name"],
            ],
            "filename_lead": "2m_temps",
            "index_long_name": "Humidex",
        },
        "AT": {
            "variables": [
                VAR_SPECS["2m_temperature"]["full_name"],
                VAR_SPECS["10m_u_component_of_wind"]["full_name"],
                VAR_SPECS["10m_v_component_of_wind"]["full_name"],
                VAR_SPECS["2m_dewpoint_temperature"]["full_name"],
            ],
            "filename_lead": "2m_temps",
            "index_long_name": "Apparent_Temperature",
        },
        "WBGT": {
            "variables": [
                VAR_SPECS["2m_temperature"]["full_name"],
                VAR_SPECS["2m_dewpoint_temperature"]["full_name"],
            ],
            "filename_lead": "2m_temps",
            "index_long_name": "Wet_Bulb_Globe_Temperature_Simple",
        },
    }
    return index_params.get(index)


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
