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

Define Uncertainty class.
"""

__all__ = [
    "UncCascadeOutput",
]

import datetime as dt
import logging
from abc import ABC, abstractmethod
from itertools import zip_longest
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps as cm

from climada.engine.unsequa.unc_output import UncOutput

LOGGER = logging.getLogger(__name__)

class UncCascadeOutput(UncOutput):
    """Extension of UncOutput specific for CalcImpact, returned by the
    uncertainty() method.
    """

    def __init__(
        self,
        samples_df,
        unit,
        imp_met_unc_df,
    ):
        """Constructor

        Uncertainty output values from impact.calc for each sample

        Parameters
        ----------
        samples_df : pandas.DataFrame
            input parameters samples
        unit : str
            value unit
        imp_met_unc_df : pandas.DataFrame
            Each row contains the values of the number of people losing access to
            ci_types defined in the columns, computed over the sample (row of

        """
        super().__init__(samples_df, unit)
        self.imp_met_unc_df = imp_met_unc_df
        self.imp_met_sens_df = None
