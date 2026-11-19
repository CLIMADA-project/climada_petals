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

Define the uncertainty output class for network cascade calculations
"""

__all__ = [
    "UncCascadeOutput",
]

import logging

from climada.engine.unsequa.unc_output import UncOutput

LOGGER = logging.getLogger(__name__)


class UncCascadeOutput(UncOutput):
    """Extension of UncOutput specific for CalcCascade, returned by the
    uncertainty() method.
    """

    def __init__(
        self,
        samples_df,
        unit,
        imp_met_unc_df,
    ):
        """Constructor

        Uncertainty output values of the network cascade for each sample

        Parameters
        ----------
        samples_df : pandas.DataFrame
            input parameters samples
        unit : str
            value unit
        imp_met_unc_df : pandas.DataFrame
            Network statistics for each sample (rows), e.g. the number of
            people losing access to each ci_type (columns), as computed by
            :py:func:`climada_petals.engine.networks.nw_utils.make_network_stat`.
        """
        super().__init__(samples_df, unit)
        self.imp_met_unc_df = imp_met_unc_df
        self.imp_met_sens_df = None
