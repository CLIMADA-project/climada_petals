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
Climate Index Definitions and Explanations Module

This module, `index_definitions.py`, defines the specifications and explanations for various indices
used within the CLIMADA seasonal forecast workflow. It centralises necessary parameters, descriptions, and filenames
for each climate index, ensuring consistent use across climate risk assessment workflows based on seasonal forecasts.

The provided definitions facilitate index-based analysis for indices such as Mean Temperature (Tmean), Tropical Nights (TR),
and Heat Wave (HW). Structured for extensibility and usability, the module supports automated index configuration,
standardised documentation, and file naming conventions across the seasonal forecast pipeline.

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
    Retrieves the complete specifications of a specified climate index, such as required input variables
    and naming conventions.

Example Usage
-------------
>>> from index_definitions import IndexSpecEnum
>>> index_info = IndexSpecEnum.get_info("TR")
>>> print(index_info.explanation)
>>> print(index_info.variables)

Notes
-----
This module relies on the `xarray`, `pandas`, and `logging` libraries for array manipulation, datetime handling,
and error and information logging.
"""

from enum import Enum
from dataclasses import dataclass


@dataclass
class IndexSpec:
    unit: str
    standard_name: str
    short_name: str
    full_name: str
    filename_lead: str
    explanation: str
    variables: list


class IndexSpecEnum(Enum):
    HIA = IndexSpec(
        unit="K",
        standard_name="air_temperature",
        short_name="t2m",
        full_name="2m_temperature",
        filename_lead="2m_temps",
        explanation="Heat Index Adjusted: This indicator measures apparent temperature, considering both air temperature and humidity, providing a more accurate perception of how hot it feels.",
        variables=["2m_temperature", "2m_dewpoint_temperature"],
    )
    HIS = IndexSpec(
        unit="K",
        standard_name="dew_point_temperature",
        short_name="d2m",
        full_name="2m_dewpoint_temperature",
        filename_lead="2m_temps",
        explanation="Heat Index Simplified: A simpler version focusing on a quick estimate of perceived heat.",
        variables=["2m_temperature", "2m_dewpoint_temperature"],
    )
    Tmean = IndexSpec(
        unit="K",
        standard_name="air_temperature",
        short_name="t2m",
        full_name="2m_temperature",
        filename_lead="2m_temps",
        explanation="Mean Temperature: Calculates the average temperature over a specified period.",
        variables=["2m_temperature"],
    )
    Tmin = IndexSpec(
        unit="K",
        standard_name="air_temperature",
        short_name="t2m",
        full_name="2m_temperature",
        filename_lead="2m_temps",
        explanation="Minimum Temperature: Tracks the lowest temperature recorded over a specified period.",
        variables=["2m_temperature"],
    )
    Tmax = IndexSpec(
        unit="K",
        standard_name="air_temperature",
        short_name="t2m",
        full_name="2m_temperature",
        filename_lead="2m_temps",
        explanation="Maximum Temperature: Tracks the highest temperature recorded over a specified period.",
        variables=["2m_temperature"],
    )
    HW = IndexSpec(
        unit="K",
        standard_name="air_temperature",
        short_name="t2m",
        full_name="2m_temperature",
        filename_lead="2m_temps",
        explanation="Heat Wave: Identifies heat waves as periods with temperatures above a threshold.",
        variables=["2m_temperature"],
    )
    TR = IndexSpec(
        unit="K",
        standard_name="air_temperature",
        short_name="t2m",
        full_name="2m_temperature",
        filename_lead="2m_temps",
        explanation="Tropical Nights: Counts nights with minimum temperatures above a certain threshold. Default threshold is 20°C.",
        variables=["2m_temperature"],
    )
    TX30 = IndexSpec(
        unit="K",
        standard_name="air_temperature",
        short_name="t2m",
        full_name="2m_temperature",
        filename_lead="2m_temps",
        explanation="Hot Days (TX30): Counts days with maximum temperature exceeding 30°C.",
        variables=["2m_temperature"],
    )
    RH = IndexSpec(
        unit="%",
        standard_name="relative_humidity",
        short_name="rh",
        full_name="relative_humidity",
        filename_lead="2m_temps",
        explanation="Relative Humidity: Measures humidity as a percentage.",
        variables=["2m_temperature", "2m_dewpoint_temperature"],
    )
    HUM = IndexSpec(
        unit="C",
        standard_name="humidex",
        short_name="hum",
        full_name="humidex",
        filename_lead="2m_temps",
        explanation="Humidex: Perceived temperature combining temperature and humidity.",
        variables=["2m_temperature", "2m_dewpoint_temperature"],
    )
    AT = IndexSpec(
        unit="C",
        standard_name="apparent_temperature",
        short_name="at",
        full_name="apparent_temperature",
        filename_lead="2m_temps",
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
        standard_name="wet_bulb_globe_temperature",
        short_name="wbgt",
        full_name="wet_bulb_globe_temperature",
        filename_lead="2m_temps",
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
