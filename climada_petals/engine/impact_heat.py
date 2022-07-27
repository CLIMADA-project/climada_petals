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

Define Impact and ImpactFreqCurve classes.
"""

__all__ = ['ImpactHeat']

import logging
import numpy as np
import pandas as pd
from scipy import sparse

from climada.engine.impact import Impact, ImpactFreqCurve


LOGGER = logging.getLogger(__name__)

class ImpactHeat(Impact, ImpactFreqCurve):
    """Impact for heat-related mortality. In contrast to conventional impact
    calculations, the heat mortality calculation relies on the annual cycle
    of death counts.
    
    Attributes
    ----------
    ens_member (optional) : np.array
        indicates SMILE ensemble member of each event
    imp_date (optional) : np.array
        indicates start date of events after aggregating to specific time
        windows using self.aggregate_to_window()

    """

    def __init__(self):
        """Empty initialization."""
        Impact.__init__(self)
        self.unit = ''

    def calc_heat_mort(self, exposures, impact_funcs, hazard,
                           save_mat=False):
        """Compute impact for heat-related mortality

        Parameters
        ----------
        exposures : climada_petals.entity.exposures.daily_mortality
            Daily mortality counts
        impact_funcs : climada_petals.entity.impact_funcs.heat_mortality
            impact functions / relativ risk curves
        hazard : climada_petals.hazard.heat 
        save_mat : bool
            self impact matrix: events x n_locations

        Returns
        -------
            climada.engine.impact.ImpactHeat
        """
        
        # calculate impacts using impact.calc
        self.calc(exposures, impact_funcs, hazard, save_mat=True)
        imp_mat = self.imp_mat.toarray()

        n_loc = int(imp_mat.shape[1]/365)
        n_years = int(imp_mat.shape[0]/365)
        
        # impacts are calculated for each day & exposure. Correct values are
        # there for on the diagonal of the tile of each daily array
        year_stack = np.stack(np.vsplit(
            np.concatenate(np.hsplit(imp_mat, n_loc), axis=0),
            n_loc*n_years), axis=2)
        imp_clean = np.zeros(0)
        for i in range(year_stack.shape[2]):
            imp_clean = np.concatenate([imp_clean,
                                        np.diag(year_stack[:,:,i])],0)
        imp_clean = np.reshape(imp_clean, [n_years*365, n_loc], 'F')
        
        # assign correct variables
        if save_mat:
            self.imp_mat = sparse.csr_matrix(imp_clean)
        self.at_event += np.squeeze(np.asarray(np.sum(imp_clean, axis=1)))
        
        # assign variables
        self.coord_exp = np.stack([
            np.reshape(exposures.gdf.latitude.values, [-1,365])[:,0],
            np.reshape(exposures.gdf.longitude.values, [-1,365])[:,0]], axis=1)
        self.eai_exp = np.sum(imp_clean * hazard.frequency[:, None], 0)
        self.aai_agg = sum(self.at_event * hazard.frequency)
        if hasattr(self, 'ens_member'):
            self.ens_member = hazard.ens_member
        
    def aggregate_to_years(self, save_mat=False):
        """Aggregate daily mortality counts to annual
        
        Parameters
        ----------
        save_mat : bool
            self impact matrix: events x locations

        Returns
        -------
            climada.engine.impact.ImpactHeat
        """
        if not hasattr(self, 'imp_mat'):
            raise ValueError('No imp_mat. Calculate impact with save_mat=True')
        
        if hasattr(self, 'ens_member'):
            summary_df = pd.DataFrame(columns = ['Year', 'ens_member'])
            summary_df.Year = pd.to_datetime(self.date).year
            summary_df.ens_member = self.ens_member
            summary_df = pd.concat([summary_df,
                                        pd.DataFrame(self.imp_mat.toarray())], axis=1)
            annual_aggregate = summary_df.groupby(['ens_member','Year']).sum()
        else:
            summary_df = pd.DataFrame(columns = ['Year'])
            summary_df.Year = pd.to_datetime(self.date).year
            summary_df = pd.concat([summary_df,
                                        pd.DataFrame(self.imp_mat.toarray())], axis=1)
            annual_aggregate = summary_df.groupby(['Year']).sum()
            
        annual_imp_mat = annual_aggregate.to_numpy()
        
        self.event_id = np.arange(1, annual_aggregate.shape[0]+1).astype(int)
        self.event_name = list(map(str, self.event_id))
        self.date = annual_aggregate.index.get_level_values('Year').values
        if hasattr(self, 'ens_member'):
            self.ens_member = annual_aggregate.index.get_level_values('ens_member').values
        if save_mat:
            self.imp_mat = sparse.csr_matrix(annual_imp_mat)
        freq = 1/(annual_imp_mat.shape[0])
        self.frequency = np.ones(self.event_id.size)*freq
        self.at_event = np.sum(annual_imp_mat, axis=1)
        self.eai_exp = np.sum(annual_imp_mat * self.frequency[:, None], 0)
        self.aai_agg = sum(self.at_event * self.frequency)


    def aggregate_to_window(self, window=21, save_mat=False, warm_season_only=True, 
                            len_season=180):
        """Aggregate daily mortality counts to specified time windows. Time
        windows are calculate to find the location specific maximum event
        per season. Hence dates between locations might no longer be in line.
        Therefore, the new attripute self.imp_date is introduced.
        
        Parameters
        ----------
        window : int
            number of days to aggregate impacts to
        save_mat : bool
            self impact matrix: events x locations
        warm_season_only : bool
            calculate impacts during warm season only
        len_season : int
            number of days of warm season
            
            
        Returns
        -------
            climada.engine.impact.ImpactHeat
        """
        if not hasattr(self, 'imp_mat'):
            raise ValueError('No imp_mat. Calculate impact with save_mat=True')
        
        # set warm season
        if warm_season_only:
            imp_mat_season, imp_mat_date = self._set_warm_season(len_season)
        else:
            imp_mat_season = self.imp_mat.toarray()
            imp_mat_date = np.tile(self.date, (imp_mat_season.shape[1], 1)).T
            len_season = 365
        
        # aggregate to window
        imp_mat, imp_date = self._set_time_window(window, imp_mat_season,
                                             imp_mat_date, len_season)
        
        # assign class attributes
        if hasattr(self, 'ens_member'):
            n_ens = len(np.unique(self.ens_member))
        else: n_ens = 1
        n_years = int(self.imp_mat.shape[0]/n_ens/365)
        
        self.event_id = np.arange(1, imp_mat.shape[0]+1).astype(int)
        self.event_name = list(map(str, self.event_id))
        years = imp_date.astype('datetime64[Y]').astype(int) + 1970
        self.date = pd.to_datetime(years[:,0].astype(str))
        if hasattr(self, 'ens_member'):
            self.ens_member = np.repeat(np.arange(0, n_ens),
                                        int(imp_mat.shape[0]/n_ens))
        if save_mat:
            self.imp_mat = sparse.csr_matrix(imp_mat)
            self.imp_date = imp_date
        freq = 1/(n_ens * n_years)
        self.frequency = np.ones(self.event_id.size)*freq
        self.at_event = np.sum(imp_mat, axis=1)
        self.eai_exp = np.sum(imp_mat * self.frequency[:, None], 0)
        self.aai_agg = sum(self.at_event * self.frequency)


    def _set_warm_season(self, len_season=180):
        """Identifies warm season based on mortality counts for each location.

        Parameters
        ----------
        len_season : int
            number of days of warm season
            
        Returns
        -------
            imp_mat_season : np.array
                self.imp_mat for warm season only
            imp_mat_date : np.array
                dates corresponding to imp_mat_season
        """
        if hasattr(self, 'ens_member'):
            n_ens = len(np.unique(self.ens_member))
        else: n_ens = 1
        n_years = int(self.imp_mat.shape[0]/n_ens/365)
        impacts = pd.DataFrame(self.imp_mat.toarray())
        rolling_impacts = impacts.rolling(len_season).mean().to_numpy()
        
        for i in range(rolling_impacts.shape[1]):
            ens_means = np.nanmean(np.reshape(rolling_impacts[:,i], [365,-1], 'F'), axis=1)
            season_idx = np.argmax(ens_means)
            ind = np.zeros(365, bool)
            if season_idx > 180:
                ind[season_idx-180:season_idx] = True
            else:
                ind[0:season_idx] = True
                ind[365-(180-season_idx):365] = True

            bool_waarm_season = np.tile(ind, n_ens*n_years)
            
            if i==0:
                imp_mat_season = self.imp_mat.toarray()[bool_waarm_season,i]
                imp_mat_date = self.date[bool_waarm_season]
            else:
                imp_mat_season = np.stack([imp_mat_season,
                                                 self.imp_mat.toarray() \
                                                 [bool_waarm_season,i]], axis=1)
                imp_mat_date = np.stack([imp_mat_date,
                                               self.date[bool_waarm_season]],
                                              axis=1) 

        return imp_mat_season, imp_mat_date


    def _set_time_window(self, window, imp_mat_season, imp_mat_date, len_season):
        """Identifies warm season based on mortality counts for each location.

        Parameters
        ----------
        window : int
            number of days to aggregate impacts to
        imp_mat_season : np.array
            self.imp_mat for warm season only
        imp_mat_date : np.array
            dates corresponding to imp_mat_season
        len_season : int
            number of days of warm season

        Returns
        -------
            imp_mat_season : np.array
                self.imp_mat for warm season only
            imp_mat_date : np.array
                dates corresponding to imp_mat_season
        """

        imp_events = np.zeros([0])
        date_events = np.zeros([0], dtype='datetime64[ns]')
        # loop over locations
        for i in range(imp_mat_season.shape[1]):
            imp_mat_loc = np.reshape(imp_mat_season[:,i], [len_season,-1], 'F')
            imp_mat_pd = pd.DataFrame(imp_mat_loc)
            imp_date_loc = np.reshape(imp_mat_date[:,i], [len_season,-1], 'F')
            # identify major events per season
            events = imp_mat_pd.rolling(window).sum().to_numpy()
            max_ind = np.nanargmax(events, axis=0)
            # arange events per season accordingly
            n_evts = len_season//window-1
            ind_start = np.mod(max_ind, window)
            
            # loop over all seasons and ensemble members
            evt_imp_loc = np.zeros([0])
            evt_date_loc = np.zeros([0], dtype='datetime64[ns]')

            for j in range(imp_mat_loc.shape[1]):
                memb_imp = np.reshape(imp_mat_loc[ind_start[j]:ind_start[j] + \
                                                 n_evts * window, j],
                                     [window,-1], 'F')
                evt_imp_loc = np.concatenate([evt_imp_loc,
                                              np.sum(memb_imp, axis=0)],0)
                memb_date = np.reshape(imp_date_loc[ind_start[j]:ind_start[j] + \
                                                   n_evts * window, j],
                                       [window,-1], 'F')
                evt_date_loc = np.concatenate([evt_date_loc,
                                              memb_date[0,:]],0)
            imp_events = np.concatenate([imp_events, evt_imp_loc], 0)
            date_events = np.concatenate([date_events, evt_date_loc], 0)
        imp_events = np.reshape(imp_events, [-1, imp_mat_season.shape[1]], 'F')
        date_events = np.reshape(date_events, [-1, imp_mat_season.shape[1]], 'F')

        return imp_events, date_events
        