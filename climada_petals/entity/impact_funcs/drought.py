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

Define impact function for droughts.
"""

__all__ = ['ImpfDrought', 'IFDrought']

import logging
from deprecation import deprecated
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc

LOGGER = logging.getLogger(__name__)

class ImpfDrought(ImpactFunc):
    """Impact function for droughts."""

    def __init__(self):
        """Empty initialization.

        Parameters
        ----------
        impf_id : int, optional
            impact function id. Default: 1
        intensity : np.array, optional
            intensity array SPEI [-].
            default: intensity defintion 1 (minimum)
            default_sum: intensity definition 3 (sum over all drought months)

        Raises
        ------
        ValueError
        """
        ImpactFunc.__init__(self)

    def set_default(self, *args, **kwargs):
        """This function is deprecated, use ImpfDrought.from_default instead."""
        LOGGER.warning("The use of ImpfDrought.set_default is deprecated."
                       "Use ImpfDrought.from_default instead.")
        self.__dict__ = ImpfDrought.from_default(*args, **kwargs).__dict__

    def set_default_sum(self, *args, **kwargs):
        """This function is deprecated, use ImpfDrought.from_default_sum instead."""
        LOGGER.warning("The use of ImpfDrought.set_default_sum is deprecated."
                       "Use ImpfDrought.from_default_sum instead.")
        self.__dict__ = ImpfDrought.from_default_sum(*args, **kwargs).__dict__

    def set_default_sumthr(self, *args, **kwargs):
        """This function is deprecated, use ImpfDrought.from_default_sumthr instead."""
        LOGGER.warning("The use of ImpfDrought.set_default_sumthr is deprecated."
                       "Use ImpfDrought.from_default_sumthr instead.")
        self.__dict__ = ImpfDrought.from_default_sumthr(*args, **kwargs).__dict__

    def set_step(self, *args, **kwargs):
        """This function is deprecated, use ImpfDrought.from_step instead."""
        LOGGER.warning("The use of ImpfDrought.set_step is deprecated."
                       "Use ImpfDrought.from_step instead.")
        self.__dict__ = ImpfDrought.from_step(*args, **kwargs).__dict__

    @classmethod
    def from_default(cls):
        """
        Returns
        -------
        impf : ImpfDrought
            Default impact function.
        """
        impf = cls()
        impf.haz_type = "DR"
        impf.id = 1
        impf.name = "drought default"
        impf.intensity_unit = "NA"
        impf.intensity = [-6.5, -4, -1, 0]
        impf.mdd = [1, 1, 0, 0]
        impf.paa = [1, 1, 0, 0]
        return impf

    @classmethod
    def from_default_sum(cls):
        """
        Returns
        -------
        impf : ImpfDrought
            Default sum impact function.
        """
        impf = cls()
        impf.haz_type = "DR_sum"
        impf.id = 1
        impf.name = "drought default sum"
        impf.intensity_unit = "NA"
        impf.intensity = [-15, -12, -9, -7, -5, 0]
        impf.mdd = [1, 0.65, 0.5, 0.3, 0, 0]
        impf.paa = [1, 1, 1, 1, 0, 0]
        return impf

    @classmethod
    def from_default_sumthr(cls):
        """
        Returns
        -------
        impf : ImpfDrought
            Default sum threshold impact function.
        """
        impf = cls()
        impf.haz_type = "DR_sumthr"
        impf.id = 1
        impf.name = "drought default sum - thr"
        impf.intensity_unit = "NA"
        impf.intensity = [-8, -5, -2, 0]
        impf.mdd = [0.7, 0.3, 0, 0]
        impf.paa = [1, 1, 0, 0]
        return impf

    @classmethod
    def from_step(cls):
        """
        Returns
        -------
        impf : ImpfDrought
            step impact function.
        """
        impf = cls()
        impf.haz_type = "DR"
        impf.id = 1
        impf.name = "step"
        impf.intensity_unit = "NA"
        impf.intensity = np.arange(-4, 0)
        impf.mdd = np.ones(impf.intensity.size)
        impf.paa = np.ones(impf.mdd.size)
        return impf

@deprecated(details="The class name IFDrought is deprecated and won't be supported in a future "
                   +"version. Use ImpfDrought instead")
class IFDrought(ImpfDrought):
    """Is ImpfDrought now"""
