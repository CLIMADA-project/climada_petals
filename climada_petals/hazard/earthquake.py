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

Define EQ earthquake (EarthQuake class).
"""

__all__ = ['Earthquake']

import numpy as np

from climada.hazard.base import Hazard
from climada.util import coordinates as u_coord

HAZ_TYPE = 'EQ'
"""Hazard type acronym for EarthQuake"""

MIN_MMI = 2
"""Minimal MMI value"""

MAX_DIST_DEG = 10
"""Maximum distance of the centroids from the epicenters in degrees"""

class Earthquake(Hazard):

    def from_Mw_depth(self, df, centroids):
        """

        Parameters
        ----------
        df : TYPE
            lat, lon, Mw, depth

        Returns
        -------
        None.

        """
        for idx, event in df.iterrows():
            epi_lat, epi_lon = event['lat', 'lon']
            mag, depth = event['mw', 'depth']
            lon_min, lat_min, lon_max, lat_max =\
                u_coord.latlon_bounds(epi_lat, epi_lon, buffer=MAX_DIST_DEG)
            bounds = (lon_min, lon_max, lat_min, lat_max)
            select_centroids = centroids.select(extent=bounds)
            intensity = self.footprint_MMI(select_centroids, epi_lat, epi_lon, mag, depth)

    def footprint_MMI(self, centroids, epi_lat, epi_lon, mag, depth):
        cent_lat, cent_lon = np.array([centroids.lat]), np.array([centroids.lon])
        dists = u_coord.dist_approx(cent_lat, cent_lon, epi_lat, epi_lon)
        self.attenuation_MMI(dists, mag, depth)

    def attenuation_MMI(self, dist, mag, depth, corr=0, a1=1.7, a2=1.5, a3=1.1726, a4=0.00106, b=0):
        """
        Modified Mercalli Intensity
        https://doi.org/10.1201/9781482271645

        Parameters
        ----------
        mag : TYPE
            DESCRIPTION.
        dist : TYPE
            DESCRIPTION.
        depth : TYPE
            DESCRIPTION.
        corr : TYPE, optional
            DESCRIPTION. The default is 0.
        a1 : TYPE, optional
            DESCRIPTION. The default is 1.7.
        a2 : TYPE, optional
            DESCRIPTION. The default is 1.5.
        a3 : TYPE, optional
            DESCRIPTION. The default is 1.1726.
        a4 : TYPE, optional
            DESCRIPTION. The default is 0.00106.
        b : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        max_MMI = 1.5 * mag -1
        return np.min(a1 + a2 * mag - a3 * np.log(dist + corr) - a4 * dist + b * depth, max_MMI)
