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
import pandas as pd
from scipy import sparse
from datetime import datetime

from climada.hazard.base import Hazard, TagHazard
from climada.util import coordinates as u_coord

HAZ_TYPE = 'EQ'
"""Hazard type acronym for EarthQuake"""

MIN_MMI = 3
"""Minimal MMI value"""

MAX_DIST_DEG = 3
"""Maximum distance of the centroids from the epicenters in degrees"""

class Earthquake(Hazard):
    """Earthquate class"""
    def __init__(self):
        Hazard.__init__(self, haz_type=HAZ_TYPE, pool=None)

    @classmethod
    def from_Mw_depth(cls, df, centroids):
        """
        Earthquakes from events epicenters positions, depth, and MW energy.
        Date column in format %Y-%m-%d %H:%M:%S.%f" .

        Parameters
        ----------
        df : DataFrame
            lat, lon, Mw, depth

        Returns
        -------
        Hazard: hazard Earthquake

        """

        format = "%Y-%m-%d %H:%M:%S.%f"
        dates = [datetime.strptime(date_str, format) for date_str in df.date]
        years = np.array([date.year for date in dates])

        quake = cls()
        quake.tag = TagHazard()
        n_years = 1
        quake.units = 'Mw'
        quake.centroids = centroids
        # following values are defined for each event
        quake.event_id = df.eventid.to_numpy()
        quake.frequency = np.repeat(1 / n_years, len(df))
        quake.event_name = df.eventid.astype('str').to_list()
        quake.date = np.array([date.toordinal() for date in dates])
        quake.orig = np.ones(len(df))
        # following values are defined for each event and centroid

        int_list = []
        cent_idx_list = []
        ev_idx_list = []

        df2 = pd.DataFrame({
            col: df[col].values
            for col in ['lat', 'lon', 'mw', 'depth']
        })

        for idx, event in df2.iterrows():
            epi_lat, epi_lon = event[['lat', 'lon']]
            mag, depth = event[['mw', 'depth']]

            lon_min, lat_min, lon_max, lat_max =\
                u_coord.latlon_bounds(np.array([epi_lat]), np.array([epi_lon]), buffer=MAX_DIST_DEG)
            lat, lon = centroids.lat, centroids.lon

            lon_max += 360 if lon_min > lon_max else 0
            lon_normalized = u_coord.lon_normalize(lon, center=0.5 * (lon_min + lon_max))
            mask = (
              (lon_normalized >= lon_min) & (lon_normalized <= lon_max) &
              (lat >= lat_min) & (lat <= lat_max)
            )
            cent_lat = lat[mask]
            cent_lon = lon[mask]
            if cent_lat.size > 0:
                int_list.append(quake.footprint_MMI(cent_lat, cent_lon, epi_lat, epi_lon, mag, depth))
                cent_idx_list.append(np.where(mask)[0])
                ev_idx_list.append(np.repeat(idx, sum(mask)))

        n_events = len(df)
        n_centroids = len(centroids.lat)
        col = np.hstack(cent_idx_list)
        row = np.hstack(ev_idx_list)
        data = np.hstack(int_list).ravel()
        quake.intensity = sparse.csr_matrix((data, (row, col)), shape=(n_events, n_centroids))  # events x centroids
        quake.fraction = quake.intensity.copy()
        quake.fraction.data.fill(1) # events x centroids
        return quake

    def footprint_MMI(self, cent_lat, cent_lon, epi_lat, epi_lon, mag, depth):
        """
        Compute the footprint (intensity in MMI) at centroids position for
        epicenter position, depth and magnitude in Mw

        Parameters
        ----------
        cent_lat : np.array
            centroids latitudes
        cent_lon : np.array
            centroids longitudes
        epi_lat : float
            epicenter latitude
        epi_lon : float
            epienter longitude
        mag : float
            magnitude of earthquake in MMI
        depth : float
            depth in km

        Returns
        -------
        np.array
            array of intensities

        """
        cent_lat, cent_lon = np.array([cent_lat]), np.array([cent_lon])
        dists = u_coord.dist_approx(cent_lat, cent_lon, np.array([[epi_lat]]), np.array([[epi_lon]]))
        return self.attenuation_MMI(dists, mag, depth)

    def attenuation_MMI(self, dist, mag, depth, corr=0.0, a1=1.7, a2=1.5, a3=1.1726, a4=0.00106, b=0.0):
        """
        Modified Mercalli Intensity (MMI)
        https://doi.org/10.1201/9781482271645

        Parameters
        ----------
        dist : np.array
            distances in KM
        mag : float
            earthquake magnitude in Mw
        depth : float
            distance in KM
        corr : float, optional
            see MMI. The default is 0.
        a1 : float, optional
            see MMI. The default is 1.7.
        a2 : float, optional
            see MMI. The default is 1.5.
        a3 : float optional
            see MMI. The default is 1.1726.
        a4 : float optional
            see MMI. The default is 0.00106.
        b : float, optional
            see MMI. The default is 0.

        Returns
        -------
        np.array
            MMI values for all distances

        """
        max_MMI = 1.5 * (mag - 1.0)
        mmi = a1 + a2 * mag - a3 * np.log(dist + corr) - a4 * dist + b * depth
        return np.clip(mmi, a_min=MIN_MMI, a_max=max_MMI)
