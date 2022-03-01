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

Define impact functions for river flood .
"""

__all__ = ['ImpfRiverFlood', 'IFRiverFlood']

import logging
from deprecation import deprecated
import numpy as np
import pandas as pd

from climada.util.constants import RIVER_FLOOD_REGIONS_CSV
from climada.entity import ImpactFunc, ImpactFuncSet

LOGGER = logging.getLogger(__name__)

DEF_VAR_EXCEL = {'sheet_name': 'damagefunctions',
                 'col_name': {'func_id': 'DamageFunID',
                              'inten': 'Intensity',
                              'mdd': 'MDD',
                              'paa': 'PAA',
                              'mdr': 'MDR',
                              'name': 'name',
                              'peril': 'peril_ID',
                              'unit': 'Intensity_unit'
                              }
                 }



class ImpfRiverFlood(ImpactFunc):
    """Impact functions for tropical cyclones."""

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = 'RF'
        self.intensity_unit = 'm'
        self.continent = ''

    @classmethod
    def from_region(cls, region):
        """Create a new ImpfRiverFlood object with parameters for the specified world region.

        Parameters
        ----------
        region : str
            Use damage function parameters for this world region. Supported values: "Africa",
            "Asia", "Europe", "NorthAmerica", "Oceania", "SouthAmerica".

        Returns
        -------
        impf : ImpfRiverFlood
            New ImpfRiverFlood object with parameters for the specified world region.

        Raises
        ------
        ValueError
        """
        impf = cls()
        if region.lower() == "africa":
            impf.id = 1
            impf.name = "Flood Africa JRC Residential noPAA"
            impf.continent = 'Africa'
            impf.intensity = np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])
            impf.mdd = np.array([0.0000, 0.2199, 0.3782, 0.5306, 0.6356, 0.8169,
                                 0.9034, 0.9572, 1.0000, 1.0000])
            impf.mdr = np.array([0.0000, 0.2199, 0.3782, 0.5306, 0.6356, 0.8169,
                                 0.9034, 0.9572, 1.0000, 1.0000])
        elif region.lower() == "asia":
            impf.id = 2
            impf.name = "Flood Asia JRC Residential noPAA"
            impf.continent = 'Asia'
            impf.intensity = np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])
            impf.mdd = np.array([0.000, 0.3266, 0.4941, 0.6166, 0.7207, 0.8695,
                                 0.9315, 0.9836, 1.0000, 1.0000])
            impf.mdr = np.array([0.000, 0.3266, 0.4941, 0.6166, 0.7207, 0.8695,
                                 0.9315, 0.9836, 1.0000, 1.0000])
        elif region.lower() == "europe":
            impf.id = 3
            impf.name = "Flood Europe JRC Residential noPAA"
            impf.continent = 'Europe'
            impf.intensity = np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])
            impf.mdd = np.array([0.00, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 0.95,
                                 1.00, 1.00])
            impf.mdr = np.array([0.000, 0.250, 0.400, 0.500, 0.600, 0.750, 0.850,
                                 0.950, 1.000, 1.000])
        elif region.lower().replace(" ", "") == "northamerica":
            impf.id = 4
            impf.name = "Flood North America JRC Residential noPAA"
            impf.continent = 'NorthAmerica'
            impf.intensity = np.array([0., 0.1, 0.5, 1., 1.5, 2., 3., 4., 5.,
                                      6., 12.])
            impf.mdd = np.array([0.0000, 0.2018, 0.4433, 0.5828, 0.6825, 0.7840,
                                 0.8543, 0.9237, 0.9585, 1.0000, 1.0000])
            impf.mdr = np.array([0.0000, 0.2018, 0.4433, 0.5828, 0.6825, 0.7840,
                                 0.8543, 0.9237, 0.9585, 1.0000, 1.0000])
        elif region.lower() == "oceania":
            impf.id = 5
            impf.name = "Flood Oceania JRC Residential noPAA"
            impf.continent = 'Oceania'
            impf.intensity =  np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])
            impf.mdd = np.array([0.00, 0.48, 0.64, 0.71, 0.79, 0.93, 0.97, 0.98,
                                 1.00, 1.00])
            impf.mdr = np.array([0.000, 0.480, 0.640, 0.710, 0.790, 0.930, 0.970,
                                 0.980, 1.000, 1.000])
        elif region.lower().replace(" ", "") == "southamerica":
            impf.id = 6
            impf.name = "Flood South America JRC Residential noPAA"
            impf.continent = 'SouthAmerica'
            impf.intensity =  np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])
            impf.mdd = np.array([0.0000, 0.4908, 0.7112, 0.8420, 0.9494,
                                 0.9836, 1.0000, 1.0000, 1.0000, 1.0000])
            impf.mdr = np.array([0.0000, 0.4908, 0.7112, 0.8420, 0.9494, 0.9836,
                                 1.0000, 1.0000, 1.0000, 1.0000])
        else:
            raise ValueError(f"Unrecognized world region: {region}")
        impf.paa = np.ones(len(impf.intensity))
        return impf

    def set_RF_Impf_Africa(self, *args, **kwargs):
        """This function is deprecated, use ImpfRiverFlood.from_region instead."""
        LOGGER.warning("The use of ImpfRiverFlood.set_RF_Impf_Africa is deprecated."
                       "Use ImpfRiverFlood.from_region instead.")
        self.__dict__ = ImpfRiverFlood.from_region("Africa", *args, **kwargs).__dict__

    def set_RF_Impf_Asia(self, *args, **kwargs):
        """This function is deprecated, use ImpfRiverFlood.from_region instead."""
        LOGGER.warning("The use of ImpfRiverFlood.set_RF_Impf_Asia is deprecated."
                       "Use ImpfRiverFlood.from_region instead.")
        self.__dict__ = ImpfRiverFlood.from_region("Asia", *args, **kwargs).__dict__

    def set_RF_Impf_Europe(self, *args, **kwargs):
        """This function is deprecated, use ImpfRiverFlood.from_region instead."""
        LOGGER.warning("The use of ImpfRiverFlood.set_RF_Impf_Europe is deprecated."
                       "Use ImpfRiverFlood.from_region instead.")
        self.__dict__ = ImpfRiverFlood.from_region("Europe", *args, **kwargs).__dict__

    def set_RF_Impf_NorthAmerica(self, *args, **kwargs):
        """This function is deprecated, use ImpfRiverFlood.from_region instead."""
        LOGGER.warning("The use of ImpfRiverFlood.set_RF_Impf_NorthAmerica is deprecated."
                       "Use ImpfRiverFlood.from_region instead.")
        self.__dict__ = ImpfRiverFlood.from_region("NorthAmerica", *args, **kwargs).__dict__

    def set_RF_Impf_Oceania(self, *args, **kwargs):
        """This function is deprecated, use ImpfRiverFlood.from_region instead."""
        LOGGER.warning("The use of ImpfRiverFlood.set_RF_Impf_Oceania is deprecated."
                       "Use ImpfRiverFlood.from_region instead.")
        self.__dict__ = ImpfRiverFlood.from_region("Oceania", *args, **kwargs).__dict__

    def set_RF_Impf_SouthAmerica(self, *args, **kwargs):
        """This function is deprecated, use ImpfRiverFlood.from_region instead."""
        LOGGER.warning("The use of ImpfRiverFlood.set_RF_Impf_SouthAmerica is deprecated."
                       "Use ImpfRiverFlood.from_region instead.")
        self.__dict__ = ImpfRiverFlood.from_region("SouthAmerica", *args, **kwargs).__dict__


def flood_imp_func_set():
    """Builds impact function set for river flood, using standard files"""

    impf_set = ImpactFuncSet()

    impf_africa = ImpfRiverFlood.from_region("Africa")
    impf_set.append(impf_africa)

    impf_asia = ImpfRiverFlood.from_region("Asia")
    impf_set.append(impf_asia)

    impf_europe = ImpfRiverFlood.from_region("Europe")
    impf_set.append(impf_europe)

    impf_na = ImpfRiverFlood.from_region("NorthAmerica")
    impf_set.append(impf_na)

    impf_oceania = ImpfRiverFlood.from_region("Oceania")
    impf_set.append(impf_oceania)

    impf_sa = ImpfRiverFlood.from_region("SouthAmerica")
    impf_set.append(impf_sa)

    return impf_set


def assign_Impf_simple(exposure, country):
    info = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)
    impf_id = info.loc[info['ISO'] == country, 'Impf_RF'].values[0]
    exposure['Impf_RF'] = impf_id


@deprecated(details="The class name IFRiverFlood is deprecated and won't be supported in a future "
                   +"version. Use ImpfRiverFlood instead")
class IFRiverFlood(ImpfRiverFlood):
    """Is ImpfRiverFlood now"""
