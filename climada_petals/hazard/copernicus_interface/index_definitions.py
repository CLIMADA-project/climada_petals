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
Climate Index Definitions and Variable Handling Module

This module, `index_definitions.py`, defines and organizes the specifications, explanations, and variable mappings
for various climate indices used within the CLIMADA seasonal forecast workflow. It centralizes key parameters,
descriptions, filenames, and variable handling, ensuring consistent and extensible use across climate risk
assessment workflows based on seasonal forecasts.

Key features include:
- Index Definitions: The `IndexSpec` dataclass defines attributes for each climate index, such as units,
  standard and short names, full descriptions, required variables, and file naming conventions.
- Enumerated Indices: The `IndexSpecEnum` organizes supported climate indices, such as Mean Temperature (Tmean),
  Tropical Nights (TR), and Heat Wave (HW), each mapped to an `IndexSpec` instance for easy retrieval.
- Variable-to-Short-Name Mapping: The `get_short_name_from_variable` function provides a centralized mechanism
  for mapping variable names (e.g., "2m_temperature") to their corresponding short names (e.g., "t2m"), ensuring
  consistency in variable handling.

Attributes
----------
IndexSpec : dataclass
    Contains index-specific attributes, including units, standard and short names, full names, filenames,
    explanations, and variable requirements.

IndexSpecEnum : Enum
    Enumerates the supported climate indices, each mapped to its `IndexSpec` configuration.
    
Functions
---------
get_info(index_name)
    Retrieves the complete specifications of a specified climate index, including its attributes and required variables.
get_short_name_from_variable(variable)
    Maps a variable's standard name (e.g., "2m_temperature") to its short name (e.g., "t2m").

Example Usage
-------------
>>> from index_definitions import IndexSpecEnum, get_short_name_from_variable
>>> index_info = IndexSpecEnum.get_info("TR")
>>> print(index_info.explanation)
Tropical Nights: Counts nights with minimum temperatures above a certain threshold. Default threshold is 20°C.
>>> print(index_info.variables)
['2m_temperature']

>>> short_name = get_short_name_from_variable("2m_temperature")
>>> print(short_name)
t2m

Notes
-----
This module relies on the `xarray`, `pandas`, and `logging` libraries for array manipulation, datetime handling,
and error and information logging. It supports both index-based and variable-based workflows, allowing flexibility
in how indices and variables are processed and managed.
"""

from enum import Enum
from dataclasses import dataclass


@dataclass
class IndexSpec:
    unit: str
    full_name: str
    explanation: str
    variables: list


class IndexSpecEnum(Enum):
    HIA = IndexSpec(
        unit="C",
        full_name="Heat Index Adjusted",
        explanation="Heat Index Adjusted: A refined measure of apparent temperature that accounts for both air temperature and humidity. This index improves upon the simplified heat index by incorporating empirical corrections for extreme temperature and humidity conditions, ensuring a more accurate representation of perceived heat stress. If the temperature is ≤ 26.7°C (80°F), the index returns a simplified estimate.",
        variables=["2m_temperature", "2m_dewpoint_temperature"],
    )
    HIS = IndexSpec(
        unit="C",
        full_name="Heat Index Simplified",
        explanation="Heat Index Simplified: A quick estimate of perceived heat based on temperature and humidity, using an empirical formula designed for warm conditions (T > 20°C). If the temperature is ≤ 20°C, the heat index is set to the air temperature.",
        variables=["2m_temperature", "2m_dewpoint_temperature"],
    )
    Tmean = IndexSpec(
        unit="C",
        full_name="Mean Temperature",
        explanation="Mean Temperature: Calculates the average temperature over a specified period.",
        variables=["2m_temperature"],
    )
    Tmin = IndexSpec(
        unit="C",
        full_name="Minimum Temperature",
        explanation="Minimum Temperature: Tracks the lowest temperature recorded over a specified period.",
        variables=["2m_temperature"],
    )
    Tmax = IndexSpec(
        unit="C",
        full_name="Maximum Temperature",
        explanation="Maximum Temperature: Tracks the highest temperature recorded over a specified period.",
        variables=["2m_temperature"],
    )
    HW = IndexSpec(
        unit="Days",
        full_name="Heat Wave",
        explanation="Heat Wave: Identifies heat waves as periods with temperatures above a threshold. Default >= 27 °C for minimum 3 consecutive days.",
        variables=["2m_temperature"],
    )
    TR = IndexSpec(
        unit="Days",
        full_name="Tropical Nights",
        explanation="Tropical Nights: Counts nights with minimum temperatures above a certain threshold. Default threshold is 20°C.",
        variables=["2m_temperature"],
    )
    TX30 = IndexSpec(
        unit="Days",
        full_name="Hot Days (TX30)",
        explanation="Hot Days (TX30): Counts days with maximum temperature exceeding 30°C.",
        variables=["2m_temperature"],
    )
    RH = IndexSpec(
        unit="%",
        full_name="Relative Humidity",
        explanation="Relative Humidity: Measures humidity as a percentage.",
        variables=["2m_temperature", "2m_dewpoint_temperature"],
    )
    HUM = IndexSpec(
        unit="C",
        full_name="Humidex",
        explanation="Humidex: Perceived temperature combining temperature and humidity.",
        variables=["2m_temperature", "2m_dewpoint_temperature"],
    )
    AT = IndexSpec(
        unit="C",
        full_name="Apparent Temperature",
        explanation="Apparent Temperature: Perceived temperature considering wind and humidity.",
        variables=[
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
        ],
    )
    WBGT = IndexSpec(
        unit="C",
        full_name="Wet Bulb Globe Temperature",
        explanation="Wet Bulb Globe Temperature (Simple): Heat stress index combining temperature and humidity.",
        variables=["2m_temperature", "2m_dewpoint_temperature"],
    )

    @classmethod
    def get_info(cls, index_name: str):
        """
        Retrieve the complete information for a specified index.

        Parameters
        ----------
        index_name : str
            The name of the index (e.g., "HIA", "HIS", "Tmean").

        Returns
        -------
        IndexSpec
            Returns an instance of IndexSpec containing all relevant information.
            Raises a ValueError if the index is not found.
        """
        try:
            return cls[index_name].value
        except KeyError:
            raise ValueError(
                f"Unknown index '{index_name}'. Available indices: {', '.join(cls.__members__.keys())}"
            )


def get_short_name_from_variable(variable):
    """
    Retrieve the short name of a variable within an index based on its standard name.

    Parameters
    ----------
    variable : str
        The standard name of the climate variable (e.g., "2m_temperature", "10m_u_component_of_wind").

    Returns
    -------
    str or None
        The short name corresponding to the specified climate variable (e.g., "t2m" for "2m_temperature").
        Returns None if the variable is not recognized.

    Notes
    -----
    This function maps specific variable names to their short names, which are used across
    climate index definitions. These mappings are independent of the indices themselves
    but provide consistent naming conventions for variable processing and file management.

    Examples
    --------
    >>> get_short_name_from_variable("2m_temperature")
    't2m'

    >>> get_short_name_from_variable("10m_u_component_of_wind")
    'u10'

    >>> get_short_name_from_variable("unknown_variable")
    None
    """
    if variable == "2m_temperature":
        return "t2m"
    elif variable == "2m_dewpoint_temperature":
        return "d2m"
    elif variable == "10m_u_component_of_wind":
        return "u10"
    elif variable == "10m_v_component_of_wind":
        return "v10"
    else:
        return None

