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
from tqdm import tqdm

from climada.hazard.base import Hazard
import climada.util.dates_times as u_dates
from climada.util import coordinates as u_coord
from climada.util.api_client import Client

LENGTH_MODIS_TILE = 0.46331271653 #km documentation:463.31271653m
AREA_MODIS_TILE = LENGTH_MODIS_TILE**2


LOGGER = logging.getLogger(__name__)
LOGGER.info("The wildfire module has been updated. The former module described"
"in Lüthi et al. (2021) has been depracted. To reproduce data with"
"the previous calculation, use CLIMADA v6.0.1 or less.")

HAZ_TYPE = 'WF'
""" Hazard type acronym for Wild Fire"""

class WildFire(Hazard):

    """
    Hazard description

    """
    
    
    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
    
    
    @classmethod
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
    
    
    """HOTSPOT-BASED HAZARD SET"""
    @staticmethod
    def assign_HS_2_centroids(HS_directory, centroids, output_csv='HS_assigned.csv'):
        """
        Create one CSV file with all hotspot detections. Add one column with the
        centroid closest to each detection. 

        Parameters
        ----------
        HS_directory : str
            Parent directory with all hotspot CSVs.
        centroids : Centroid
            CLIMADA centroid object to map the hotspots on.
        output_csv : str, optional
            Name for the output CSV. The default is 'HS_assigned.csv'.

        Returns
        -------
        None.

        """
    
        target_coord = centroids.coord
        target = np.ascontiguousarray(target_coord, dtype='float64')
    
        csv_file_paths = WildFire._get_csv_file_paths(HS_directory)
                
        for idx, file in tqdm(enumerate(csv_file_paths), total=len(csv_file_paths), desc="Processing"):

            df = pd.read_csv(file)
            
            #get coordinate
            coords_hs = np.vstack((df['latitude'], df['longitude'])).T
            coords = np.ascontiguousarray(coords_hs, dtype='float64')
            assigned_coord = u_coord.match_coordinates(coords, target)
            
            #extract relevant columns
            dataframe = df[['frp', 'acq_date']]
            dataframe['centroid'] = assigned_coord
            
            #drop rows of unassigned hotspots
            dataframe = dataframe.dropna()

            if idx == 0:
                dataframe.to_csv(output_csv, index=False)
            else:
                # Append to the CSV file
                dataframe.to_csv(output_csv, mode='a', header=False, index=False)
            
            pass
    
    @staticmethod
    def _get_csv_file_paths(directory):
        """
        Extract all CSV files in hotspot directory

        Parameters
        ----------
        directory : str
            Directory with containing Hotspot data.

        Returns
        -------
        csv_file_paths: list
            Sorted list of all CSV files.

        """
        csv_file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    csv_file_paths.append(os.path.join(root, file))
        return sorted(csv_file_paths)
    
    @staticmethod
    def assign_march_feb_season_HS(df):
        df['season_start_year'] = df.apply(lambda row: row['year'] if row['month'] >= 3 else row['year'] - 1, axis=1)
    
        season_month_counts = df.groupby('season_start_year')['month'].nunique()
        full_seasons = season_month_counts[season_month_counts == 12].index
    
        df.loc[~df['season_start_year'].isin(full_seasons), 'season_start_year'] = pd.NA
        

        #drop nan values in incomplete seasons
        df = df.dropna()
        df['season'] = df['season_start_year'].apply(lambda y: f"{y}-{y + 1}")
        
        
        start_year = df['season_start_year'].min()
        end_year = df['season_start_year'].max()
        # Generate season end dates (Feb 28 or 29) from start_year to end_year + 1
        time_array = pd.to_datetime([f'{y+1}-02-28' for y in range(start_year, end_year+1)])
        time_array = time_array.map(lambda d: d + pd.Timedelta(days=1) if d.is_leap_year else d)
        
        return df, time_array

        
    def _seasonal_assignment(dates, list_sparse_m):
        # 1. Filter to only full March–Feb seasons
    
        # Find first March or later
        for i_start, d in enumerate(dates):
            if d.month == 3:
                break
    
        # Find last February or earlier
        for i_end in reversed(range(len(dates))):
            if dates[i_end].month == 2:
                break
    
        # Slice valid dates and intensity
        valid_dates0 = dates[i_start:i_end+1]
        list_valid_m = []
        for matrix in list_sparse_m:
            list_valid_m.append(matrix[i_start:i_end+1, :])
    
        # 2. Assign March–Feb season years
        season_years = []
        for d in valid_dates0:
            season_year = d.year if d.month >= 3 else d.year - 1
            season_years.append(datetime.date(season_year, 3, 1).toordinal())
    
        # dates = season_years
    
        # 3. Create time steps and labels
        start_year = valid_dates0[0].year if valid_dates0[0].month >= 3 else valid_dates0[0].year - 1
        end_year = valid_dates0[-1].year if valid_dates0[-1].month >= 3 else valid_dates0[-1].year - 1
    
        feb_days = [calendar.monthrange(y + 1, 2)[1] for y in range(start_year, end_year+1)]
        time_array = pd.to_datetime([f"{y + 1}-02-{day}" for y, day in zip(range(start_year, end_year+1), feb_days)])
        event_name = [f"{y}-{y+1}" for y in range(start_year, end_year+1)]
        
        return valid_dates0, season_years, time_array, event_name, list_valid_m
    

    @classmethod
    def create_FRP_hazard(cls, output_csv, target_centroids, temporal_scale='month', 
                       aggregation='percentile', percentile=95):

        dataframe = pd.read_csv(output_csv)
        dataframe['acq_date'] = pd.to_datetime(dataframe['acq_date'])
        
        if temporal_scale == 'month':
            dataframe['year'] = dataframe['acq_date'].dt.year.astype(int)
            dataframe['month'] = dataframe['acq_date'].dt.month.astype(int)
            #dataframe['day'] = dataframe['acq_date'].dt.day.astype(int)
            group_by = ['year', 'month', 'centroid']
            
            start_year = dataframe['year'].min()
            end_year = dataframe['year'].max()
            time_array = pd.date_range(start=str(start_year)+"-01-01", 
                                       end=str(end_year+1)+"-01-01",
                          freq='M')
            event_name = None
            
        elif temporal_scale == 'season':
            dataframe, time_array = WildFire.assign_march_feb_season_HS(dataframe)
            start_year = dataframe['season_start_year'].min()
            group_by = ['season', 'centroid']
            event_name = dataframe['season']
        
            
        # Group by 'centroid' and temporal dimension and compute the aggregation metric
        if aggregation == 'percentile':
            grouped_percentile = dataframe.groupby(group_by)['frp'].agg(
                lambda x: np.percentile(x, percentile)).reset_index()
        elif aggregation == 'sum':
            grouped_percentile = dataframe.groupby(group_by)['frp'].agg(
                'sum').reset_index()
            
        matrix_hazard = lil_matrix((len(time_array), target_centroids.lat.size))
        
        if temporal_scale == 'month':
            matrix_hazard[(grouped_percentile.year-start_year)*12+grouped_percentile.month-1 , 
                          grouped_percentile.centroid] = grouped_percentile.frp
        elif temporal_scale == 'season':
            matrix_hazard[(grouped_percentile.year-start_year), 
                          grouped_percentile.centroid] = grouped_percentile.frp
    

        final_intensity = sparse.csr_matrix(matrix_hazard)
        haz_fre = cls.create_wf_haz(final_intensity, target_centroids, 
                                    time_array, 'MW', event_name)
        
        return haz_fre

        
    """Burnt area based hazard - see wildfire_tiles"""
    

    

    
    

    


    
    @classmethod
    def sum_haz_per_timestep(self, timestep, filename=None):
        """Aggregate in time"""
        
        cls = self.__class__
    
        dates0 = [datetime.datetime.fromordinal(date) for date in self.date]
        # event_id, name und date funktioniert noch nicht!
        if timestep == 'year':
            dates = [datetime.date(date.year, 1, 1).toordinal() for date in dates0]
            time_array = pd.date_range(start=str(dates0[0].year)+"-01-01", 
                                       end=str(dates0[-1].year+1)+"-01-01",
                                       freq='Y')
            event_name = time_array.strftime("%Y").tolist()
        
        if timestep == 'season':
            
            [valid_dates0, season_years, 
             time_array, event_name, 
             list_valid_m] = WildFire._seasonal_assignment(dates0, 
                                                            [self.intensity])#, 
                                                            # self.fraction])
        
            # 4. Replace haz.intensity and fraction with trimmed version for aggregation
            self.intensity = list_valid_m[0]
            # self.fraction = list_valid_m[1]
    
            
        elif timestep == 'month': 
            dates = [datetime.date(date.year, date.month, 1).toordinal() for date in dates0]
            time_array = pd.date_range(start=str(dates0[0].year)+"-01-01", 
                                       end=str(dates0[-1].year+1)+"-01-01",
                          freq='M')
            event_name = time_array.strftime("%Y-%m").tolist()
            
        
        unique_tst, unique_idx = np.unique(dates, return_index=True)
        all_unique_tstp = np.append(unique_idx, (len(dates)-1))
    
        new_intensity = []
        # new_fraction = []
        nr_tstp = len(unique_idx)
        for idx in range(nr_tstp):
            # new_intensity.append(sp.sparse.csr_matrix(np.max(self.intensity[
            #     all_unique_tstp[idx]:all_unique_tstp[idx+1],:],axis=0)))
            new_intensity.append(sp.sparse.csr_matrix(np.max(self.intensity[
                all_unique_tstp[idx]:all_unique_tstp[idx+1],:],axis=0)))
            # new_fraction.append(sp.sparse.csr_matrix(np.sum(self.fraction[
            #     all_unique_tstp[idx]:all_unique_tstp[idx+1],:],axis=0)))
        
        intensity2 = sp.sparse.vstack(new_intensity).tocsr()
        # fraction2 = sp.sparse.vstack(new_fraction).tocsr()
        
        haz_year = cls.create_wf_haz(intensity2, self.centroids, time_array, 
                                         # fraction=fraction2, 
                                         units='', 
                                         event_name=event_name)
        
        if filename is not None:
            haz_year.write_hdf5(filename)
        
        return haz_year    
 


    def resample_wf(self, target_centr, units=''):
        
        cls = self.__class__
        
        """Aggregate in space""" 
        nonzero_idx = np.unique(self.intensity.indices)
        start_coord = self.centroids.coord[nonzero_idx,:]
        sub_intensity = self.intensity[:,nonzero_idx]
        # start_coord = haz.centroids.coord
        # sub_intensity = haz.intensity
        assigned_coord = u_coord.match_coordinates(start_coord, target_centr.coord)
        
    
        event_ids = self.event_id-1
        n_ev = len(event_ids)
        n_chunks = np.round(n_ev/12).astype(int)
        arrays = np.array_split(event_ids, n_chunks)
        
        new_intensity = dok_matrix((self.intensity.shape[0], target_centr.lat.size))
     
        for array_chunk in arrays:
            pd_intensity = pd.DataFrame(sub_intensity[array_chunk,:].todense().T, columns=self.event_id[array_chunk])
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
        
        haz_new_coord = cls.create_wf_haz(new_intensity2, target_centr, self.event_name, units)
        
        return haz_new_coord
    
    
    def resample_to_fraction(self, target_centr, threshold, nr_chunks=200):
        
        cls = self.__class__
        
        """Aggregate in space""" 
        #threshold in km
        
        total_centroids, _ = self.centroids.coord.shape
        n_cent_chunk = np.round(total_centroids/nr_chunks).astype(int)
        
        nr_observations = lil_matrix((self.intensity.shape[0], target_centr.lat.size))
        # nr_cells = dok_matrix((haz.intensity.shape[0], target_centr.lat.size))
        nr_cells = dok_matrix((1, target_centr.lat.size))
        
        for i in range(nr_chunks):
            print(i, 'of', nr_chunks)
            
            if i <= nr_chunks-2:
                start_coord = self.centroids.coord[n_cent_chunk*i: n_cent_chunk*(i+1), :]
                pd_intensity = pd.DataFrame(self.intensity[:,n_cent_chunk*i: n_cent_chunk*(i+1)].todense().T, 
                                            columns=self.event_id)
            else: 
                start_coord = self.centroids.coord[n_cent_chunk*i:, :]
                pd_intensity = pd.DataFrame(self.intensity[:,n_cent_chunk*i:].todense().T, 
                                            columns=self.event_id)
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
    
        haz_new_coord = cls.create_wf_haz(fraction, target_centr, self.event_name, units='')
        
        return haz_new_coord

   
    
    

"""VISIBLE METHODS OF DEPRECATED FORMER MODULE"""
# class FirmsParams():
#         LOGGER.warning(
#             "The wildfire module described in Lüthi et al. (2021) has"
#             "been depracted. To reproduce data with the previous calculation,"
#             "use CLIMADA v6.0.1 or less.",
#         )

# class ProbaParams():
#         LOGGER.warning(
#             "The wildfire module described in Lüthi et al. (2021) has"
#             "been depracted. To reproduce data with the previous calculation,"
#             "use CLIMADA v6.0.1 or less.",
#         )

# @classmethod
# def from_hist_fire_seasons_FIRMS(cls, df_firms, centr_res_factor=1.0,
#                                 centroids=None, hemisphere=None,
#                                 year_start=None, year_end=None,
#                                 keep_all_fires=False):

#         LOGGER.warning(
#             "The wildfire module described in Lüthi et al. (2021) has"
#             "been depracted. To reproduce data with the previous calculation,"
#             "use CLIMADA v6.0.1 or less.",
#         )
        
# def set_hist_fire_seasons_FIRMS(self, *args, **kwargs):
#         LOGGER.warning(
#             "The wildfire module described in Lüthi et al. (2021) has"
#             "been depracted. To reproduce data with the previous calculation,"
#             "use CLIMADA v6.0.1 or less.",
#         )


# def from_hist_fire_FIRMS(cls, df_firms, centr_res_factor=1.0, centroids=None):
#         LOGGER.warning(
#             "The wildfire module described in Lüthi et al. (2021) has"
#             "been depracted. To reproduce data with the previous calculation,"
#             "use CLIMADA v6.0.1 or less.",
#         )

# def set_proba_fire_seasons(self, n_fire_seasons=1, n_ignitions=None,
#                            keep_all_fires=False):
#         LOGGER.warning(
#             "The wildfire module described in Lüthi et al. (2021) has"
#             "been depracted. To reproduce data with the previous calculation,"
#             "use CLIMADA v6.0.1 or less.",
#         )


# def combine_fires(self, event_id_merge=None, remove_rest=False,
#                   probabilistic=False):
#         LOGGER.warning(
#             "The wildfire module described in Lüthi et al. (2021) has"
#             "been depracted. To reproduce data with the previous calculation,"
#             "use CLIMADA v6.0.1 or less.",
#         )

# def summarize_fires_to_seasons(self, year_start=None, year_end=None,
#                                hemisphere=None):

#         LOGGER.warning(
#             "The wildfire module described in Lüthi et al. (2021) has"
#             "been depracted. To reproduce data with the previous calculation,"
#             "use CLIMADA v6.0.1 or less.",
#         )

# def plot_fire_prob_matrix(self):
#         LOGGER.warning(
#             "The wildfire module described in Lüthi et al. (2021) has"
#             "been depracted. To reproduce data with the previous calculation,"
#             "use CLIMADA v6.0.1 or less.",
#         )
