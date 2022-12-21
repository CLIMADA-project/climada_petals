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

Define HeatMortality class.
"""

__all__ = ['DailyMortality']

import logging
import datetime as dt
import numpy as np
import pandas as pd

from climada.entity.tag import Tag
from climada.entity.exposures.base import Exposures

LOGGER = logging.getLogger(__name__)

DEF_HAZ_TYPE = 'HT'
"""Default hazard type used in impact functions id."""

class DailyMortality(Exposures):
    """ Defines exposures from daily mortality counts
    """
    def __init__(self):
        """Empty constructor"""

        Exposures.__init__(self)
    
    @classmethod
    def from_pandas_list(cls, pd_list, lat, lon):
        """ Set up exposure from daily mortality time series

        Parameters
        ----------
        pd_list : list
            list of pandas dataframes. List needs to be in line with
            lat/lon arrays. Each pd.dataframe() must contain a column 'date'
            and 'deaths'.
        lat : np.array()
            Array of latitudes
        lon : np.array()
            Array of longitudes

        Returns
        -------
        exp : Exposure instance with daily mortality
                The category_id refers to the day-of-year

        """

        exp_df = pd.DataFrame(columns = ['value', 'latitude', 'longitude',
                                         'impf_', 'category_id'])

        for i, dat in enumerate(pd_list):
            exp_loc = pd.DataFrame(columns = ['value', 'latitude', 'longitude',
                                             'impf_', 'category_id'])
            # calculate mean daily mortality
            mort_clean = _clean_data_to_365day_year(dat)
            n_years = int(mort_clean.shape[0]/365)
            mort_doy = np.reshape(mort_clean.deaths.values, (n_years, 365))
            mort_doy_mean = np.nanmean(mort_doy,0)

            # fill expsure dataframe
            exp_loc['value'] = mort_doy_mean
            exp_loc['latitude'] = np.repeat(lat[i], 365)
            exp_loc['longitude'] = np.repeat(lon[i], 365)
            exp_loc['impf_'] = i
            exp_loc['category_id'] = np.arange(1, 366)

            exp_df = pd.concat([exp_df, exp_loc])

        exp_df.reset_index()
        # create expsure
        exp = Exposures(exp_df)
        exp.set_geometry_points()
        tag = Tag()
        tag.description = 'Mean daily mortality counts for each location. \
                            category_id refers to the doy-of-year.'
        exp.tag=tag,
        exp.value_unit='# lives'
        exp.check()

        return exp

def _remove_leap_days(data):
    date_series = pd.to_datetime(data.date)
    idx_leap_day = np.where((date_series.dt.is_leap_year.values == True) & \
                            (date_series.dt.day_of_year.values == 60))[0]
    data_clean = data.drop(idx_leap_day)
    return data_clean

def _clean_data_to_365day_year(data):
    ''' This function adds nan values to a time series data frame in order to
    fill up values to full years and removes leap days '''
    # make sure that data.date is a Timestamp series
    d_start = dt.datetime(data.date.min().year, 1 , 1)
    d_end = dt.datetime(data.date.max().year, 12 , 31)
    dat = pd.DataFrame()
    dat['date'] = pd.date_range(start=d_start, end=d_end, freq='D')

    dat = dat.merge(data, on='date', how='left')
    # correct for leap days
    dat_clean = _remove_leap_days(dat)

    return dat_clean
