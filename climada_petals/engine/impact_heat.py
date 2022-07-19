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
    ens_member : np.array
        indicates SMILE ensemble member of each event

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
        self.ens_member = hazard.ens_member
        
    def aggregate_to_years(self, save_mat=False):
        """Aggregate daily mortality counts to annual

        Returns
        -------
            climada.engine.impact.ImpactHeat
        """
        if not hasattr(self, 'imp_mat'):
            raise ValueError('No imp_mat. Calculate impact with save_mat=True')
        
        
        summary_df = pd.DataFrame(columns = ['Year', 'ens_member'])
        summary_df.Year = pd.to_datetime(self.date).year
        summary_df.ens_member = self.ens_member
        summary_df = pd.concat([summary_df,
                                    pd.DataFrame(self.imp_mat.toarray())], axis=1)
        annual_aggregate = summary_df.groupby(['ens_member','Year']).sum()
        annual_imp_mat = annual_aggregate.to_numpy()
        
        self.event_id = np.arange(1, annual_aggregate.shape[0]+1).astype(int)
        self.event_name = list(map(str, self.event_id))
        self.date = annual_aggregate.index.get_level_values('Year').values
        self.ens_member = annual_aggregate.index.get_level_values('ens_member').values
        if save_mat:
            self.imp_mat = sparse.csr_matrix(annual_imp_mat)
        freq = 1/(annual_imp_mat.shape[0])
        self.frequency = np.ones(self.event_id.size)*freq
        self.at_event = np.sum(annual_imp_mat, axis=1)
        self.eai_exp = np.sum(annual_imp_mat * self.frequency[:, None], 0)
        self.aai_agg = sum(self.at_event * self.frequency)
        