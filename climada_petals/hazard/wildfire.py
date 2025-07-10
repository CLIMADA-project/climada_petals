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
import datetime
import scipy as sp
from scipy.sparse import dok_matrix
import xarray as xr
import calendar

from climada.hazard.base import Hazard
import climada.util.dates_times as u_dates
from climada.util import coordinates as u_coord
from climada.util.api_client import Client

LENGTH_MODIS_TILE = 0.46331271653 #km documentation:463.31271653m
AREA_MODIS_TILE = LENGTH_MODIS_TILE**2


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
    
 
def sum_haz_per_timestep(haz, timestep, filename=None):
    """Aggregate in time"""

    dates0 = [datetime.datetime.fromordinal(date) for date in haz.date]
    # event_id, name und date funktioniert noch nicht!
    if timestep == 'year':
        dates = [datetime.date(date.year, 1, 1).toordinal() for date in dates0]
        time_array = pd.date_range(start=str(dates0[0].year)+"-01-01", end=str(dates0[-1].year+1)+"-01-01",
                      freq='Y')
        event_name = time_array.strftime("%Y").tolist()
    
    if timestep == 'season_StateOfWF':
        # 1. Filter to only full March–Feb seasons
    
        # Find first March or later
        for i_start, d in enumerate(dates0):
            if d.month == 3:
                break
    
        # Find last February or earlier
        for i_end in reversed(range(len(dates0))):
            if dates0[i_end].month == 2:
                break
    
        # Slice valid dates and intensity
        valid_dates0 = dates0[i_start:i_end+1]
        valid_intensity = haz.intensity[i_start:i_end+1, :]
        valid_fraction = haz.fraction[i_start:i_end+1, :]
    
        # 2. Assign March–Feb season years
        season_years = []
        for d in valid_dates0:
            season_year = d.year if d.month >= 3 else d.year - 1
            season_years.append(datetime.date(season_year, 3, 1).toordinal())
    
        dates = season_years
    
        # 3. Create time steps and labels
        start_year = valid_dates0[0].year if valid_dates0[0].month >= 3 else valid_dates0[0].year - 1
        end_year = valid_dates0[-1].year if valid_dates0[-1].month >= 3 else valid_dates0[-1].year - 1
    
        feb_days = [calendar.monthrange(y + 1, 2)[1] for y in range(start_year, end_year+1)]
        time_array = pd.to_datetime([f"{y + 1}-02-{day}" for y, day in zip(range(start_year, end_year+1), feb_days)])
        event_name = [f"{y}-{y+1}" for y in range(start_year, end_year+1)]
    
        # 4. Replace haz.intensity and fraction with trimmed version for aggregation
        haz.intensity = valid_intensity
        haz.fraction = valid_fraction

        
    elif timestep == 'month': 
        dates = [datetime.date(date.year, date.month, 1).toordinal() for date in dates0]
        time_array = pd.date_range(start=str(dates0[0].year)+"-01-01", end=str(dates0[-1].year+1)+"-01-01",
                      freq='M')
        event_name = time_array.strftime("%Y-%m").tolist()
        
    
    unique_tst, unique_idx = np.unique(dates, return_index=True)
    all_unique_tstp = np.append(unique_idx, (len(dates)-1))

    new_intensity = []
    new_fraction = []
    nr_tstp = len(unique_idx)
    for idx in range(nr_tstp):
        new_intensity.append(sp.sparse.csr_matrix(np.max(haz.intensity[
            all_unique_tstp[idx]:all_unique_tstp[idx+1],:],axis=0)))
        new_fraction.append(sp.sparse.csr_matrix(np.sum(haz.fraction[
            all_unique_tstp[idx]:all_unique_tstp[idx+1],:],axis=0)))
    
    intensity2 = sp.sparse.vstack(new_intensity).tocsr()
    fraction2 = sp.sparse.vstack(new_fraction).tocsr()
    
    haz_year = create_wf_haz(intensity2, haz.centroids, time_array, 
                             fraction=fraction2, units='', 
                             event_name=event_name)
    
    if filename is not None:
        haz_year.write_hdf5(filename)
    
    return haz_year    
 

def create_time_range(years, hemisphere):
    
    start_year, end_year = years
    
    if hemisphere == 'NH':
        start_day = str(start_year)+'-01-01'
        end_day = str(end_year)+'-12-31'
    elif hemisphere == 'SH':
        start_day = str(start_year)+'-07-01'
        end_day = str(end_year)+'-06-30'
    
    timerange = [start_day, end_day]
    
    return timerange


def resample_wf(haz, target_centr, units=''):
    """Aggregate in space""" 
    nonzero_idx = np.unique(haz.intensity.indices)
    start_coord = haz.centroids.coord[nonzero_idx,:]
    sub_intensity = haz.intensity[:,nonzero_idx]
    # start_coord = haz.centroids.coord
    # sub_intensity = haz.intensity
    assigned_coord = u_coord.match_coordinates(start_coord, target_centr.coord)
    

    event_ids = haz.event_id-1
    n_ev = len(event_ids)
    n_chunks = np.round(n_ev/12).astype(int)
    arrays = np.array_split(event_ids, n_chunks)
    
    new_intensity = dok_matrix((haz.intensity.shape[0], target_centr.lat.size))
 
    for array_chunk in arrays:
        pd_intensity = pd.DataFrame(sub_intensity[array_chunk,:].todense().T, columns=haz.event_id[array_chunk])
        # pd_intensity[pd_intensity == 0] = np.nan
        pd_intensity['assigned_coord'] = assigned_coord
        
        if units == 'km2':
            pd_intensity['assigned_coord'] = assigned_coord
            pd_agg = pd_intensity.groupby('assigned_coord').sum()*AREA_MODIS_TILE
        elif units == 'nonzero_mean':
            pd_intensity[pd_intensity == 0] = np.nan
            pd_intensity['assigned_coord'] = assigned_coord
            pd_agg = pd_intensity.groupby('assigned_coord').mean()
        elif units == 'sum':
            pd_intensity[pd_intensity == 0] = np.nan
            pd_intensity['assigned_coord'] = assigned_coord
            pd_agg = pd_intensity.groupby('assigned_coord').sum()
        elif units == 'max':
            pd_intensity[pd_intensity == 0] = np.nan
            pd_intensity['assigned_coord'] = assigned_coord
            pd_agg = pd_intensity.groupby('assigned_coord').max()
        elif units == '%':
            pd_intensity['assigned_coord'] = assigned_coord
            pd_agg = pd_intensity.groupby('assigned_coord').mean()*100
    
        new_intensity[array_chunk[0]:array_chunk[-1]+1, pd_agg.index.values] = pd_agg.values.T
    
    #add removing nan values from sparse matrix
        
    new_intensity2 = new_intensity.tocsr()
    
    haz_new_coord = create_wf_haz(new_intensity2, target_centr, haz.event_name, units)
    
    return haz_new_coord

def resample_to_fraction(haz, target_centr, threshold, nr_chunks=200):
    """Aggregate in space""" 
    #threshold in km
    
    total_centroids, _ = haz.centroids.coord.shape
    n_cent_chunk = np.round(total_centroids/nr_chunks).astype(int)
    
    nr_observations = lil_matrix((haz.intensity.shape[0], target_centr.lat.size))
    # nr_cells = dok_matrix((haz.intensity.shape[0], target_centr.lat.size))
    nr_cells = dok_matrix((1, target_centr.lat.size))
    
    for i in range(nr_chunks):
        print(i, 'of', nr_chunks)
        
        if i <= nr_chunks-2:
            start_coord = haz.centroids.coord[n_cent_chunk*i: n_cent_chunk*(i+1), :]
            pd_intensity = pd.DataFrame(haz.intensity[:,n_cent_chunk*i: n_cent_chunk*(i+1)].todense().T, 
                                        columns=haz.event_id)
        else: 
            start_coord = haz.centroids.coord[n_cent_chunk*i:, :]
            pd_intensity = pd.DataFrame(haz.intensity[:,n_cent_chunk*i:].todense().T, 
                                        columns=haz.event_id)
        assigned_coord = u_coord.match_coordinates(start_coord, target_centr.coord, threshold=threshold)
        pd_intensity['assigned_coord'] = assigned_coord
        
        pd_sum = pd_intensity.groupby('assigned_coord').sum()
        pd_count = pd_intensity.groupby('assigned_coord').count()[1]
        
        try: 
            df_sum = pd_sum.drop(-1, axis=0)
        except KeyError:
            df_sum = pd_sum
        try:
            df_count = pd_count.drop(-1, axis=0)
        except KeyError:
            df_count = pd_count
        
        nr_observations[:, df_sum.index.values] += df_sum.values.T
        nr_cells[0, df_count.index.values] += df_count.values.T
        
    # Convert dok_matrix to csr_matrix
    csr_fires = nr_observations.tocsr()
    csr_cells = nr_cells.tocsr()
    
    # Perform element-wise division
    fraction = csr_fires.multiply(csr_cells.power(-1))

    haz_new_coord = create_wf_haz(fraction, target_centr, haz.event_name, units='')
    
    return haz_new_coord

def get_CLIMADA_centr(**kwargs):
    # lon_min, lat_min, lon_max, lat_max = extent
    client = Client()
    centroids = client.get_centroids(**kwargs) #extent = (lon_min, lon_max, lat_min, lat_max)
    
    lat_centroids, lat_idx = np.unique(centroids, return_index=True)
    lat_centroids[np.argsort(lat_idx)]
    lon_centroids, lon_idx = np.unique(centroids, return_index=True)
    lon_centroids[np.argsort(lon_idx)]
    ds_centroids = xr.Dataset(
        {
            "lat": (["lat"], lat_centroids, {"units": "degrees_north"}),
            "lon": (["lon"], lon_centroids, {"units": "degrees_east"}),
        }
    )
    
    return centroids, ds_centroids

   
    
    


"""VISIBLE METHODS OF DEPRECATED FORMER MODULE"""
def set_hist_fire_seasons_FIRMS(self, *args, **kwargs):
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
