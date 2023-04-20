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

__all__ = ['ImpfWildfire']

import logging
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc

LOGGER = logging.getLogger(__name__)

class ImpfWildfire(ImpactFunc):
    """Impact function for wildfire."""

    def __init__(self, haz_type = 'WFsingle'):
        ImpactFunc.__init__(self)
        self.haz_type = haz_type
        LOGGER.warning('haz_type is set to %s.', self.haz_type)

    @classmethod
    def from_default_FIRMS(cls, i_half=295.01, impf_id=1):
        """ This function sets the impact curve to a sigmoid type shape, as
        common in impact modelling. We adapted the function as proposed by
        Emanuel et al. (2011) which hinges on two parameters (intercept (i_thresh)
        and steepness (i_half) of the sigmoid).

        .. math::
            f = \\frac{i^{3}}{1+i^{3}}

        with

        .. math::
            i = \\frac{MAX[(i_{lat, lon}-i_{thresh}), 0]}{i_{half}-i_{thresh}}

        The intercept is defined at the minimum intensity of a FIRMS value
        (295K) which leaves the steepness (i_half) the only parameter that
        needs to be calibrated.

        Here, i_half is set to 295 K as a result of the calibration
        performed by Lüthi et al. (in prep). This value is suited for an
        exposure resolution of 1 km.

        Calibration was further performed for:
            - 4 km: resulting i_half = 409.4 K
            - 10 km: resulting i_half = 484.4 K

        Calibration has been performed globally (using EMDAT data) and is
        based on 84 damage records since 2001.

        Intensity range is set between 295 K and 500 K as this is the typical
        range of FIRMS intensities.

        Parameters
        ----------
        i_half : float, optional, default = 295.01
            steepnes of the IF, [K] at which 50% of max. damage is expected
        if_id : int, optional, default = 1
            impact function id

        Returns
        -------
        Impf : climada.entity.impact_funcs.ImpfWildfire instance

        """

        Impf = cls()

        Impf.id = impf_id
        Impf.name = "wildfire default 1 km"
        Impf.intensity_unit = "K"
        Impf.intensity = np.arange(295, 500, 5)
        i_thresh = 295
        i_n = (Impf.intensity-i_thresh)/(i_half-i_thresh)
        Impf.paa = i_n**3 / (1 + i_n**3)
        Impf.mdd = np.ones(len(Impf.intensity))

        return Impf

    def set_default_FIRMS(self, *args, **kwargs):
        """This function is deprecated, use ImpfWildfire.from_default_FIRMS instead."""
        LOGGER.warning("The use of ImpfWildfire.set_default_FIRMS is deprecated."
                               "Use ImpfWildfire.from_default_FIRMS .")
        self.__dict__ = ImpfWildfire.from_default_FIRMS(*args, **kwargs).__dict__
