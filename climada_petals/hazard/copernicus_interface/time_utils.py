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
Time utility functions for seasonal forecast pipelines in CLIMADA.
Provides helpers to convert month names to numbers and calculate lead times
based on forecast configuration.
"""

import calendar
from datetime import date
import numpy as np

##########  Utility Functions  ##########


def month_name_to_number(month):
    """
    Convert a month name or number to its corresponding integer value.

    Accepts either an integer (1-12), full month name (e.g., 'March'),
    or abbreviated month name (e.g., 'Mar') and returns the corresponding
    month number (1-12).

    Parameters
    ----------
    month : int or str
        Month as an integer (1-12) or as a string (full or abbreviated month name).

    Returns
    -------
    int
        Month as an integer in the range 1-12.

    Raises
    ------
    ValueError
        If the input month is invalid, empty, or outside the valid range.
    """
    if isinstance(month, int):  # Already a number
        if 1 <= month <= 12:
            return month
        else:
            raise ValueError("Month number must be between 1 and 12.")
    if isinstance(month, str):
        if not month.strip():
            raise ValueError("Month cannot be empty.")  # e.g. "" or "   "
        month = month.capitalize()  # Ensure consistent capitalization
        if month in calendar.month_name:
            return list(calendar.month_name).index(month)
        elif month in calendar.month_abbr:
            return list(calendar.month_abbr).index(month)
    raise ValueError(f"Invalid month input: {month}")


def calculate_leadtimes(year, initiation_month, valid_period):
    """
    Calculate lead times in hours for a forecast period based on initiation and valid months.

    This function computes a list of lead times (in hours) for a seasonal forecast, starting
    from the initiation month to the end of the valid period. The lead times are generated
    in 6-hour steps, following the standard forecast output intervals.

    Parameters
    ----------
    year : int
        Year of the forecast initiation.
    initiation_month : int or str
        Initiation month of the forecast, as integer (1-12) or month name (e.g., 'March').
    valid_period : list of int or str
        List containing the start and end month of the valid period, either as integers (1-12)
        or month names (e.g., ['June', 'August']). Must contain exactly two elements.

    Returns
    -------
    list of int
        List of lead times in hours, sorted and spaced by 6 hours.

    Raises
    ------
    ValueError
        If initiation month or valid period months are invalid or reversed.

    Notes
    -----
    - The valid period may extend into the following year if the valid months are after December.
    - Lead times are calculated relative to the initiation date.
    - Each lead time corresponds to a 6-hour forecast step.

    Example:
    ---------
    If the forecast is initiated in **December 2022** and the valid period is **January
    to February 2023**,
    the function will:
    - Recognize that the forecast extends into the next year (2023).
    - Compute lead times starting from **December 1, 2022** (0 hours) to **February 28, 2023**.
    - Generate lead times in 6-hour intervals, covering the entire forecast period from
    December 2022 through February 2023.
    """

    # Convert initiation month to numeric if it is a string
    if isinstance(initiation_month, str):
        initiation_month = month_name_to_number(initiation_month)

    # Convert valid_period to numeric
    valid_period = [
        month_name_to_number(month) if isinstance(month, str) else month
        for month in valid_period
    ]

    # We expect valid_period = [start, end]
    start_month, end_month = valid_period

    # Immediately check for reversed period
    if end_month < start_month:
        raise ValueError(
            "Reversed valid_period detected. The forecast cannot be called with "
            f"an end month ({end_month}) that is before the start month ({start_month})."
        )

    # compute years of valid period
    valid_years = np.array([year, year])
    if initiation_month > valid_period[0]:  # forecast for next year
        valid_years += np.array([1, 1])
    if valid_period[1] < valid_period[0]:  # forecast including two different years
        valid_years[1] += 1

    # Reference starting date for initiation
    initiation_date = date(year, initiation_month, 1)
    valid_period_start = date(valid_years[0], valid_period[0], 1)
    valid_period_end = date(
        valid_years[1],
        valid_period[1],
        calendar.monthrange(valid_years[1], valid_period[1])[1],
    )

    return list(
        range(
            (valid_period_start - initiation_date).days * 24,
            (valid_period_end - initiation_date).days * 24 + 24,
            6,
        )
    )
