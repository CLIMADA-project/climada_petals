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
from datetime import datetime, timedelta

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
    def from_Mw_depth(cls, df, centroids, orig=True):
        """
        Earthquakes from events epicenters positions, depth, and MW energy.
        Date column in format %Y-%m-%d %H:%M:%S.%f" .

        Parameters
        ----------
        df : DataFrame
            lat, lon, Mw, depth, eventid, date

        Returns
        -------
        Hazard: hazard Earthquake

        """

        format_date = "%Y-%m-%d %H:%M:%S.%f"
        dates = [datetime.strptime(date_str, format_date) for date_str in df.date]
        years = np.array([date.year for date in dates])

        quake = cls()
        quake.tag.desription = \
        ('Earthquakes from events epicenters positions, depth, and MW energy. '
        'Using modified Mercalli Intensity (MMI) https://doi.org/10.1201/9781482271645')
        n_years = years.max() - years.min() + 1
        quake.units = 'MMI'
        quake.centroids = centroids
        # following values are defined for each event
        quake.event_id = df.eventid.to_numpy()
        quake.frequency = np.repeat(1 / n_years, len(df))
        quake.event_name = df.eventid.astype('str').to_list()
        quake.date = np.array([date.toordinal() for date in dates])
        if orig:
            quake.orig = np.ones(len(df))
        else:
            quake.orig = np.zeros(len(df))
        # following values are defined for each event and centroid

        int_list = []
        cent_idx_list = []
        ev_idx_list = []

        lat = df['lat'].to_numpy()
        lon = df['lon'].to_numpy()
        mw = df['mw'].to_numpy()
        depth = df['depth'].to_numpy()

        all_cent_lat, all_cent_lon = centroids.lat, centroids.lon

        for idx, (epi_lat, epi_lon, mag, depth) in enumerate(zip(lat, lon, mw, depth)):

            lon_min, lat_min, lon_max, lat_max =\
                u_coord.latlon_bounds(np.array([epi_lat]), np.array([epi_lon]), buffer=MAX_DIST_DEG)

            lon_max += 360
            all_cent_lon_normalized = u_coord.lon_normalize(all_cent_lon, center=0.5 * (lon_min + lon_max))
            mask = (
              (all_cent_lon_normalized >= lon_min) & (all_cent_lon_normalized <= lon_max) &
              (all_cent_lat >= lat_min) & (all_cent_lat <= lat_max)
            )
            cent_lat = all_cent_lat[mask]
            cent_lon = all_cent_lon[mask]
            if cent_lat.size > 0:
                int_list.append(quake.footprint_MMI(cent_lat, cent_lon, epi_lat, epi_lon, mag, depth))
                cent_idx_list.append(np.where(mask)[0])
                ev_idx_list.append(np.repeat(idx, np.count_nonzero(mask)))

        n_events = len(df)
        n_centroids = len(centroids.lat)
        col = np.hstack(cent_idx_list)
        row = np.hstack(ev_idx_list)
        data = np.hstack(int_list).ravel()
        quake.intensity = sparse.csr_matrix((data, (row, col)), shape=(n_events, n_centroids))  # events x centroids
        quake.fraction = quake.intensity.copy()
        quake.fraction.data.fill(1) # events x centroids
        return quake

    @classmethod
    def random_events(cls, df, centroids, n=1, loc_shift_deg=1.0, depth_mult=0.2, mag_mult=0.5):

        lat_orig = df['lat'].to_numpy()
        lon_orig = df['lon'].to_numpy()
        mw_orig = df['mw'].to_numpy()
        depth_orig = df['depth'].to_numpy()

        n_tot = n * len(df)

        lat_rnd = np.random.uniform(-loc_shift_deg/2, loc_shift_deg/2, size=n_tot)
        lon_rnd = np.random.uniform(-loc_shift_deg/2, loc_shift_deg/2, size=n_tot)

        mw_rnd = np.random.uniform(-mag_mult/2, mag_mult/2, size=n_tot)
        depth_rnd = np.random.uniform(-mag_mult/2, mag_mult/2, size=n_tot)

        format_date = "%Y-%m-%d %H:%M:%S.%f"
        dates = [datetime.strptime(date_str, format_date) for date_str in df.date]
        years = np.array([date.year for date in dates])
        n_years = np.max(years) - np.min(years)

        new_dates = []
        for n_rnd in range(1, n+1):
            for date in dates:
                new_year = date.year + n_rnd *  n_years
                if date.month == 2 and date.day==29:
                    new_dates.append(date.replace(year=new_year, day=28))
                else:
                    new_dates.append(date.replace(year=new_year))
        new_dates = [date.strftime(format_date) for date in new_dates]

        rnd_df = pd.DataFrame({
            'lat' : lat_rnd + np.repeat(lat_orig, n),
            'lon' : lon_rnd + np.repeat(lon_orig, n),
            'mw' : mw_rnd + np.repeat(mw_orig, n),
            'depth' : depth_rnd + np.repeat(depth_orig, n),
            'date' : new_dates,
            'eventid': np.arange(1, len(new_dates)+1)
            })

        return cls.from_Mw_depth(df=rnd_df, centroids=centroids, orig=False)


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
