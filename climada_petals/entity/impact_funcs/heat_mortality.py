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

Define impact functions for WildFires.
"""

__all__ = ['ImpfSetHeat']

import logging
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet


LOGGER = logging.getLogger(__name__)

class ImpfSetHeat(ImpactFunc, ImpactFuncSet):
    """Impact function for wildfire."""

    def __init__(self, haz_type='Heat'):
        ImpactFunc.__init__(self)
        self.haz_type = haz_type

    @classmethod
    def from_pandas_list(cls, pd_list_RR):
        """ This function sets the impact functions set for heat mortality.
        For heat mortality, impact functions refer to the relativ risk (RR)
        of dying at a given temperature.
        
        These impact functions can be calculated using quasi-Poisson regression
        time series analyses with distributed lag nonlinear models (DLNM).
        A R-tutorial is available at https://pubmed.ncbi.nlm.nih.gov/30829832/
        (Vicedo-Cabrera et al. 2019, DOI: 10.1097/EDE.0000000000000982)
        
        Parameters
        ----------
        pd_list_RR : list
            list of pandas dataframes. List needs to be in line with
            lat/lon arrays of exposure and hazard. Each pd.dataframe() 
            must contain a column 'temp' and 'RRfit'.

        Returns
        -------
        Impf : climada.entity.impact_func.ImpfHeat instance

        """
        Impf_set = ImpactFuncSet()
        
        for i, dat in enumerate(pd_list_RR):
        
            Impf = cls()
        
            Impf.id = i
            Impf.name = "Relativ risk for"
            Impf.intensity_unit = "C"
            Impf.intensity = dat.temp
            Impf.mdd = dat.RRfit-1
            Impf.paa = np.ones(len(Impf.intensity))
            Impf.haz_type = 'Heat'
            
            Impf_set.append(Impf)
        
        return Impf_set