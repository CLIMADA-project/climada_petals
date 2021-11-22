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

Define impact functions for AgriculturalDroughts.
"""


__all__ = ['ImpfRelativeCropyield', 'IFRelativeCropyield']

import logging
from deprecation import deprecated
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc

LOGGER = logging.getLogger(__name__)

class ImpfRelativeCropyield(ImpactFunc):
    """Impact functions for agricultural droughts."""

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = 'RC'
        self.intensity_unit = ''
        #self.continent = ''

    def set_relativeyield(self, *args, **kwargs):
        """This function is deprecated, use ImpfRelativeCropyield.impf_relativeyield instead."""
        LOGGER.warning("The use of ImpfRelativeCropyield.set_relativeyield is deprecated."
                       "Use ImpfRelativeCropyield.impf_relativeyield instead.")
        self.__dict__ = ImpfRelativeCropyield.impf_relativeyield(*args, **kwargs).__dict__

    @classmethod
    def impf_relativeyield(cls):
        """Impact functions defining the impact as intensity

        Returns
        -------
        impf : climada.entity.impact_funcs.ImpfRelativeCropyield instance
        """
        impf = cls()
        impf.haz_type = 'RC'
        impf.id = 1
        impf.name = 'Relative Cropyield ISIMIP'
        impf.intensity_unit = ''
        # intensity = 0 when the crop production is equivalent to the historical mean
        # intensity = -1 for a complete crop failure
        # intensity = 1 for a crop production surplus of 100%
        # the impact function covers the common stretch of the hazard intensity
        # CLIMADA interpolates linearly in case of larger intensity values
        impf.intensity = np.arange(-1, 10)
        impf.mdr = (impf.intensity)
        impf.mdd = (impf.intensity)
        impf.paa = np.ones(len(impf.intensity))

        return impf


@deprecated(details="The class name IFRelativeCropyield is deprecated and won't be supported"
            "in a future version. Use ImpfRelativeCropyield instead")
class IFRelativeCropyield(ImpfRelativeCropyield):
    """Is ImpfRelativeCropyield now"""
