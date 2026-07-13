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


SECTOR_CO_ID = {
    "Residential": 1,
    "Commercial": 2,
    "Industrial": 3,
    "Transport": 4,
    "Infrastructure": 5,
    "Agriculture": 6,
}
REGION_CO_ID = {
    "africa": 10,
    "asia": 20,
    "europe": 30,
    "northamerica": 40,
    "oceania": 50,
    "southamerica": 60,
}
VALID_REGIONS = "Africa, Asia, Europe, North America, Oceania, South America"


LOGGER = logging.getLogger(__name__)

class ImpfRiverFlood(ImpactFunc):
    """Impact functions for tropical cyclones."""

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = 'RF'
        self.intensity_unit = 'm'
        self.continent = ''

    @classmethod
    def from_jrc_region_sector(cls, region, sector="residential"):
        """Create a new ImpfRiverFlood object based on the specified world region and sector.
        Impact functions come from the following JRC publication:

        Huizinga, J., De Moel, H. and Szewczyk, W., Global flood depth-damage functions: Methodology
        and the database with guidelines, EUR 28552 EN, Publications Office of the European Union,
        Luxembourg, 2017, ISBN 978-92-79-67781-6, doi:10.2760/16510, JRC105688.

        Notes
        -----
        The impact functions assess percentage losses at 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6 meters of
        water. For North America, percentage losses higher than 0 are already registerd at 0 meters
        of water, because it accounts for the presence of basements (see main publication). Since
        this could be problematic when computing impacts, as one would have losses also when there
        is no flood, a 0.05 meters of water point is added to all functions (not only North America,
        for consistency), and this corresponds to the 0 meters point in the JRC functions.

        Parameters
        ----------
        region : str
            world region for which the impact function was defined. Supported values:
        "Africa", "Asia", "Europe", "North America", "Oceania", "South America".

        sector: str
            sector for which the impact function was defined. Supported values: "residential",
            "commercial", "industrial", "transport", "infrastructure", "agriculture".

        Returns
        -------
        impf : ImpfRiverFlood
            New ImpfRiverFlood object with parameters for the specified world region and sector.

        Raises
        ------
        ValueError
        """

        if sector == 'residential':
            impf_values, impf_id = from_jrc_impf_residential(region)

        elif sector == 'industrial':
            impf_values, impf_id = from_jrc_impf_industrial(region)

        elif sector == 'commercial':
            impf_values, impf_id = from_jrc_impf_commercial(region)

        elif sector == 'transport':
            impf_values, impf_id = from_jrc_impf_transport(region)

        elif sector == 'infrastructure':
            impf_values, impf_id = from_jrc_impf_infrastructure(region)

        elif sector == 'agriculture':
            impf_values, impf_id = from_jrc_impf_agriculture(region)

        else:
            raise ValueError(f"Unrecognized sector: {sector}")

        impf = cls()
        impf.name = f"Flood {region} JRC {sector.capitalize()} noPAA"
        impf.continent = f"{region}"
        impf.id = impf_id
        impf.mdd = impf_values
        impf.intensity = np.array([0., 0.05, 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])
        impf.paa = np.ones(len(impf.intensity))

        return impf

    def set_RF_Impf_Africa(self, *args, **kwargs):
        """This function is deprecated, use ImpfRiverFlood.from_region_sector instead."""
        LOGGER.warning("The use of ImpfRiverFlood.set_RF_Impf_Africa is deprecated."
                       "Use ImpfRiverFlood.from_region_sector instead.")
        self.__dict__ = ImpfRiverFlood.from_jrc_region_sector("Africa", *args, **kwargs).__dict__

    def set_RF_Impf_Asia(self, *args, **kwargs):
        """This function is deprecated, use ImpfRiverFlood.from_region_sector instead."""
        LOGGER.warning("The use of ImpfRiverFlood.set_RF_Impf_Asia is deprecated."
                       "Use ImpfRiverFlood.from_region_sector instead.")
        self.__dict__ = ImpfRiverFlood.from_jrc_region_sector("Asia", *args, **kwargs).__dict__

    def set_RF_Impf_Europe(self, *args, **kwargs):
        """This function is deprecated, use ImpfRiverFlood.from_region_sector instead."""
        LOGGER.warning("The use of ImpfRiverFlood.set_RF_Impf_Europe is deprecated."
                       "Use ImpfRiverFlood.from_region_sector instead.")
        self.__dict__ = ImpfRiverFlood.from_jrc_region_sector("Europe", *args, **kwargs).__dict__

    def set_RF_Impf_NorthAmerica(self, *args, **kwargs):
        """This function is deprecated, use ImpfRiverFlood.from_region_sector instead."""
        LOGGER.warning("The use of ImpfRiverFlood.set_RF_Impf_NorthAmerica is deprecated."
                       "Use ImpfRiverFlood.from_region_sector instead.")
        self.__dict__ = ImpfRiverFlood.from_jrc_region_sector("NorthAmerica", *args, **kwargs).__dict__

    def set_RF_Impf_Oceania(self, *args, **kwargs):
        """This function is deprecated, use ImpfRiverFlood.from_region_sector instead."""
        LOGGER.warning("The use of ImpfRiverFlood.set_RF_Impf_Oceania is deprecated."
                       "Use ImpfRiverFlood.from_region_sector instead.")
        self.__dict__ = ImpfRiverFlood.from_jrc_region_sector("Oceania", *args, **kwargs).__dict__

    def set_RF_Impf_SouthAmerica(self, *args, **kwargs):
        """This function is deprecated, use ImpfRiverFlood.from_region_sector instead."""
        LOGGER.warning("The use of ImpfRiverFlood.set_RF_Impf_SouthAmerica is deprecated."
                       "Use ImpfRiverFlood.from_region_sector instead.")
        self.__dict__ = ImpfRiverFlood.from_jrc_region_sector("SouthAmerica", *args, **kwargs).__dict__



def _from_jrc_impf(region, sector, impf_values_map):
    """Boiler plate function for the public from_jrc_impf_*** functions"""
    regkey = region.replace(' ', '').lower()
    if regkey not in REGION_CO_ID:
        raise ValueError(f"Unrecognized world region: {region}, must be one of {VALID_REGIONS}")
    impf_id = REGION_CO_ID[regkey] + SECTOR_CO_ID[sector]

    impf_values = impf_values_map.get(regkey)
    if impf_values is None:
        raise ValueError(f"No impact function implemented for the {sector} sector in {region}")
    return np.array(impf_values), impf_id



def from_jrc_impf_residential(region):
    """Create a new ImpfRiverFlood object for the residential sector of the a given
    world region.

    Parameters
    ----------
    region : str
        world region for which the impact function was defined. Supported values:
        "Africa", "Asia", "Europe", "North America", "Oceania", "South America".

    Returns
    -------
    impf : ImpfRiverFlood
        New ImpfRiverFlood object with parameters for the specified sector.

    Raises
    ------
    ValueError
    """
    return _from_jrc_impf(
        region=region,
        sector="Residential",
        impf_values_map={
            "africa":       [0., 0.,    0.22,  0.378, 0.531, 0.636, 0.817, 0.903, 0.957, 1., 1.],
            "asia":         [0., 0.,    0.327, 0.494, 0.617, 0.721, 0.87,  0.931, 0.984, 1., 1.],
            "europe":       [0., 0.,    0.25,  0.4,   0.5,   0.6,   0.75,  0.85,  0.95,  1., 1.],
            "northamerica": [0., 0.202, 0.443, 0.583, 0.683, 0.784, 0.854, 0.924, 0.959, 1., 1.],
            "oceania":      [0., 0.,    0.475, 0.640, 0.715, 0.788, 0.929, 0.967, 0.983, 1., 1.],
            "southamerica": [0., 0.,    0.491, 0.711, 0.842, 0.949, 0.984, 1.,    1.,    1., 1.],
        }
    )


def from_jrc_impf_commercial(region):
    """Create a new ImpfRiverFlood object for the commercial sector of the a given world region.

    Parameters
    ----------
    region : str
        world region for which the impact function was defined. Supported values:
        "Africa", "Asia", "Europe", "North America", "Oceania", "South America".

    Returns
    -------
    impf : ImpfRiverFlood
        New ImpfRiverFlood object with parameters for the specified sector.

    Raises
    ------
    ValueError
    """

    return _from_jrc_impf(
        region=region,
        sector="Commercial",
        impf_values_map={
            "asia":         [0., 0.,    0.377, 0.538, 0.659, 0.763, 0.883, 0.942, 0.981, 1., 1.],
            "europe":       [0., 0.,    0.15,  0.3,   0.45,  0.55,  0.75,  0.9,   1.,    1., 1.],
            "northamerica": [0., 0.018, 0.239, 0.374, 0.466, 0.552, 0.687, 0.822, 0.908, 1., 1.],
            "oceania":      [0., 0.,    0.239, 0.481, 0.674, 0.865, 1.,    1.,    1.,    1., 1.],
            "southamerica": [0., 0.,    0.611, 0.84 , 0.924, 0.992, 1.,    1.,    1.,    1., 1.],
        }
    )


def from_jrc_impf_industrial(region):
    """Create a new ImpfRiverFlood object for the industrial sector of the a given world region.

    Parameters
    ----------
    region : str
        world region for which the impact function was defined. Supported values:
        "Africa", "Asia", "Europe", "North America", "Oceania", "South America".

    Returns
    -------
    impf : ImpfRiverFlood
        New ImpfRiverFlood object with parameters for the specified sector.

    Raises
    ------
    ValueError
    """

    return _from_jrc_impf(
        region=region,
        sector="Industrial",
        impf_values_map={
            "africa":       [0., 0.,    0.063, 0.247, 0.403, 0.494, 0.685, 0.919, 1.,    1., 1.],
            "asia":         [0., 0.,    0.283, 0.482, 0.629, 0.717, 0.857, 0.909, 0.955, 1., 1.],
            "europe":       [0., 0.,    0.15,  0.27,  0.4,   0.52,  0.7,   0.85,  1.,    1., 1.],
            "northamerica": [0., 0.026, 0.323, 0.511, 0.637, 0.74,  0.86,  0.937, 0.98,  1., 1.],
            "southamerica": [0., 0.,    0.667, 0.889, 0.947, 1.,    1.,    1.,    1.,    1., 1.],
        }
    )


def from_jrc_impf_transport(region):
    """Create a new ImpfRiverFlood object for the transport sector of the a given world region.

    Parameters
    ----------
    region : str
        world region for which the impact function was defined. Supported values:
        "Africa", "Asia", "Europe", "North America", "Oceania", "South America".

    Returns
    -------
    impf : ImpfRiverFlood
        New ImpfRiverFlood object with parameters for the specified sector.

    Raises
    ------
    ValueError
    """
    return _from_jrc_impf(
        region=region,
        sector="Transport",
        impf_values_map={
            "asia":         [0., 0., 0.358, 0.572, 0.733, 0.847, 1., 1., 1., 1., 1.],
            "europe":       [0., 0., 0.317, 0.542, 0.702, 0.832, 1., 1., 1., 1., 1.],
            "southamerica": [0., 0., 0.088, 0.175, 0.596, 0.842, 1., 1., 1., 1., 1.],
        }
    )


def from_jrc_impf_infrastructure(region):
    """Create a new ImpfRiverFlood object for the infrastructure sector of the a given world region.

    Parameters
    ----------
    region : str
        world region for which the impact function was defined. Supported values:
        "Africa", "Asia", "Europe", "North America", "Oceania", "South America".

    Returns
    -------
    impf : ImpfRiverFlood
        New ImpfRiverFlood object with parameters for the specified sector.

    Raises
    ------
    ValueError
    """
    return _from_jrc_impf(
        region=region,
        sector="Infrastructure",
        impf_values_map={
            "asia":   [0., 0., 0.214, 0.373, 0.604, 0.71, 0.808, 0.887, 0.969, 1., 1.],
            "europe": [0., 0., 0.25,  0.42,  0.55,  0.65, 0.8,   0.9,   1.,    1., 1.],
        }
    )


def from_jrc_impf_agriculture(region):
    """Create a new ImpfRiverFlood object for the agriculture sector of the a given world region.

    Parameters
    ----------
    region : str
        world region for which the impact function was defined. Supported values:
        "Africa", "Asia", "Europe", "North America", "Oceania", "South America".

    Returns
    -------
    impf : ImpfRiverFlood
        New ImpfRiverFlood object with parameters for the specified sector.

    Raises
    ------
    ValueError
    """
    return _from_jrc_impf(
        region=region,
        sector="Agriculture",
        impf_values_map={
            "africa":       [0., 0.,    0.243, 0.472, 0.741, 0.917, 1.,   1.,    1.,    1., 1.],
            "asia":         [0., 0.,    0.135, 0.37 , 0.524, 0.558, 0.66, 0.834, 0.988, 1., 1.],
            "europe":       [0., 0.,    0.3,   0.55,  0.65,  0.75,  0.85, 0.95,  1.,    1., 1.],
            "northamerica": [0., 0.019, 0.268, 0.474, 0.551, 0.602, 0.76, 0.874, 0.954, 1., 1.],
        }
    )


def flood_imp_func_set(sector="residential"):
    """Builds impact function set for river flood, using standard files. By default, it reads
    functions for the residential sector"""

    impf_africa = ImpfRiverFlood.from_jrc_region_sector("Africa", sector=sector)
    impf_asia = ImpfRiverFlood.from_jrc_region_sector("Asia", sector=sector)
    impf_europe = ImpfRiverFlood.from_jrc_region_sector("Europe", sector=sector)
    impf_na = ImpfRiverFlood.from_jrc_region_sector("NorthAmerica", sector=sector)
    impf_oceania = ImpfRiverFlood.from_jrc_region_sector("Oceania", sector=sector)
    impf_sa = ImpfRiverFlood.from_jrc_region_sector("SouthAmerica", sector=sector)

    impf_set = ImpactFuncSet([
                impf_africa,
                impf_asia,
                impf_europe,
                impf_na,
                impf_oceania,
                impf_sa
    ])

    return impf_set


def assign_Impf_simple(exposure, country):
    info = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)
    impf_id = info.loc[info['ISO'] == country, 'Impf_RF'].values[0]
    exposure['Impf_RF'] = impf_id


@deprecated(details="The class name IFRiverFlood is deprecated and won't be supported in a future "
                   +"version. Use ImpfRiverFlood instead")
class IFRiverFlood(ImpfRiverFlood):
    """Is ImpfRiverFlood now"""
