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

Define WildFire class.
"""

__all__ = ['WildFire']

import logging
import numpy as np
from scipy import sparse
import os
import pandas as pd
from scipy.sparse import lil_matrix

from climada.hazard.base import Hazard
import climada.util.dates_times as u_dates
from climada.util import coordinates as u_coord

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'WF'
""" Hazard type acronym for Wild Fire"""

class WildFire(Hazard):

    """
    Hazard description

    """
    
    
    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
    
    
    
    """HOTSPOT-BASED HAZARD SET"""
    def assign_HS_2_centroids(HS_directory, centroids, output_csv='HS_assigned.csv'):
    
        target_coord = centroids.coord
        target = np.ascontiguousarray(target_coord, dtype='float64')
    
        csv_file_paths = _get_csv_file_paths(HS_directory)
        
        #LOOP OVER EVERY COUNTRY AND YEAR
        for idx, file in enumerate(csv_file_paths):
            print((idx+1)/len(csv_file_paths))
            
            df = pd.read_csv(file)
            
            #get coordinate
            coords_hs = np.vstack((df['latitude'], df['longitude'])).T
            coords = np.ascontiguousarray(coords_hs, dtype='float64')
            assigned_coord = u_coord.match_coordinates(coords, target)
            
            df['centroid'] = assigned_coord
            
            if idx == 0:
                df.to_csv(output_csv, index=False)
            else:
                # Append to the CSV file
                df.to_csv(output_csv, mode='a', header=False, index=False)
            
    

    def create_HS_haz_95th_percentile(output_csv, target_centroids):
    
        df = pd.read_csv(output_csv)
        dataframe = df[['acq_date', 'frp', 'centroid']]

        dataframe['acq_date'] = pd.to_datetime(dataframe['acq_date'])
        
        # Extract year and month from 'acq_date'
        dataframe['year'] = dataframe['acq_date'].dt.year
        dataframe['month'] = dataframe['acq_date'].dt.month
    
        group_by = ['year', 'month', 'centroid']
    
        # Group by 'year' and 'centroid' and compute the 95th percentile
        grouped_percentile = dataframe.groupby(group_by)['frp'].agg(
            lambda x: np.percentile(x, 95))
        grouped_percentile_df = grouped_percentile.reset_index()
    
        
        start_year = dataframe['year'].min()
        end_year = dataframe['year'].max()
        time_array = pd.date_range(start=str(start_year)+"-01-01", 
                                   end=str(end_year+1)+"-01-01",
                      freq='M')
        
        matrix_hazard = lil_matrix((len(time_array), target_centroids.lat.size))

        matrix_hazard[(grouped_percentile_df.year-start_year)*12+grouped_percentile_df.month-1 , 
                      grouped_percentile_df.centroid] = grouped_percentile_df.frp
    

        final_intensity = sparse.csr_matrix(matrix_hazard)
    

        haz_fre = create_wf_haz(final_intensity, target_centroids, 
                                time_array, units='MW')
        
        return haz_fre

    

def create_wf_haz(intensity, centroids, dates, units='', event_name=None):
    #dates as datetime.datetime object
   
    n_ev, n_centroids = intensity.shape
    
    dates_np = [np.datetime64(date) for date in dates]
    ordinals = u_dates.datetime64_to_ordinal(dates_np) 
    
    event_id = np.arange(1, n_ev+1, dtype=int)
    if event_name is None:
        event_name = [np.datetime_as_string(date, unit='D') for date in dates_np]
    
    frequency = np.ones(n_ev)/n_ev
    
    haz = Hazard(haz_type=HAZ_TYPE,
                  intensity=intensity,
                  centroids=centroids,  
                  units=units,
                  event_id=event_id,
                  event_name=event_name,
                  date=ordinals,
                  frequency=frequency,)
    
    return haz
    

def _get_csv_file_paths(directory):
    csv_file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_file_paths.append(os.path.join(root, file))
    return csv_file_paths
    
    
    
    
    
    """VISIBLE METHODS OF DEPRECATED FORMER MODULE"""
    class FirmsParams():
            LOGGER.warning(
                "The wildfire module described in Lüthi et al. (2021) has"
                "been depracted. To reproduce data with the previous calculation,"
                "use CLIMADA v6.0.1 or less.",
            )

    class ProbaParams():
            LOGGER.warning(
                "The wildfire module described in Lüthi et al. (2021) has"
                "been depracted. To reproduce data with the previous calculation,"
                "use CLIMADA v6.0.1 or less.",
            )

    def from_hist_fire_FIRMS(cls, df_firms, centr_res_factor=1.0, centroids=None):
            LOGGER.warning(
                "The wildfire module described in Lüthi et al. (2021) has"
                "been depracted. To reproduce data with the previous calculation,"
                "use CLIMADA v6.0.1 or less.",
            )

    def set_hist_fire_FIRMS(self, *args, **kwargs):
        """This function is deprecated, use WildFire.from_hist_fire_FIRMS instead."""
        LOGGER.warning("The use of WildFire.set_hist_fire_FIRMS is deprecated."
                        "Use WildFire.from_hist_fire_FIRMS .")
        self.__dict__ = WildFire.from_hist_fire_FIRMS(*args, **kwargs).__dict__

    @classmethod
    def from_hist_fire_seasons_FIRMS(cls, df_firms, centr_res_factor=1.0,
                                    centroids=None, hemisphere=None,
                                    year_start=None, year_end=None,
                                    keep_all_fires=False):

            LOGGER.warning(
                "The wildfire module described in Lüthi et al. (2021) has"
                "been depracted. To reproduce data with the previous calculation,"
                "use CLIMADA v6.0.1 or less.",
            )

    def set_hist_fire_seasons_FIRMS(self, *args, **kwargs):
            LOGGER.warning(
                "The wildfire module described in Lüthi et al. (2021) has"
                "been depracted. To reproduce data with the previous calculation,"
                "use CLIMADA v6.0.1 or less.",
            )

    def set_proba_fire_seasons(self, n_fire_seasons=1, n_ignitions=None,
                               keep_all_fires=False):
            LOGGER.warning(
                "The wildfire module described in Lüthi et al. (2021) has"
                "been depracted. To reproduce data with the previous calculation,"
                "use CLIMADA v6.0.1 or less.",
            )

    def combine_fires(self, event_id_merge=None, remove_rest=False,
                      probabilistic=False):
            LOGGER.warning(
                "The wildfire module described in Lüthi et al. (2021) has"
                "been depracted. To reproduce data with the previous calculation,"
                "use CLIMADA v6.0.1 or less.",
            )

    def summarize_fires_to_seasons(self, year_start=None, year_end=None,
                                   hemisphere=None):
            LOGGER.warning(
                "The wildfire module described in Lüthi et al. (2021) has"
                "been depracted. To reproduce data with the previous calculation,"
                "use CLIMADA v6.0.1 or less.",
            )

    def plot_fire_prob_matrix(self):
            LOGGER.warning(
                "The wildfire module described in Lüthi et al. (2021) has"
                "been depracted. To reproduce data with the previous calculation,"
                "use CLIMADA v6.0.1 or less.",
            )
