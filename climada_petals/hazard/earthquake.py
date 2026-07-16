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

from climada.hazard.base import Hazard
from climada.util import coordinates as u_coord

HAZ_TYPE = 'EQ'
"""Hazard type acronym for EarthQuake"""

MIN_MMI = 3
"""Minimal MMI value"""

MAX_DIST_DEG = 3
"""Maximum distance of the centroids from the epicenters in degrees"""

class Earthquake(Hazard):
    """Earthquake hazard represented by Modified Mercalli Intensities.

    This class stores earthquake events and their spatial intensity footprints
    on a set of centroids. Intensities are expressed on the Modified Mercalli
    Intensity (MMI) scale.

    Attributes
    ----------
    haz_type : str
        Hazard type identifier.
    units : str
        Unit of the hazard intensity values.
    centroids : climada.hazard.centroids.centr.Centroids
        Spatial centroids associated with the hazard.
    intensity : scipy.sparse.csr_matrix
        Event intensity values with shape ``(n_events, n_centroids)``.
    """

    def __init__(self, **kwargs):
        """Initialize an empty earthquake hazard."""
        Hazard.__init__(self, haz_type=HAZ_TYPE, pool=None, **kwargs)

    @classmethod
    def from_Mw_depth(cls, df, centroids, orig=True):
        """Create an earthquake hazard from magnitude and depth data.

        Events outside the centroid extent expanded by ``MAX_DIST_DEG`` are
        discarded before intensity footprints are computed.

        The input catalogue must contain event locations, moment magnitudes,
        focal depths, event identifiers, and dates. Event dates are expected in
        the format ``%Y-%m-%d %H:%M:%S.%f``. MMI footprints are computed for
        centroids within ``MAX_DIST_DEG`` degrees of each epicentre.

        Parameters
        ----------
        df : pandas.DataFrame
            Earthquake catalogue with columns ``lat``, ``lon``, ``mw``,
            ``depth``, ``eventid``, and ``date``. Depth is given in kilometres
            and magnitude as moment magnitude Mw.
        centroids : climada.hazard.centroids.centr.Centroids
            Centroids at which earthquake intensities are computed.
        orig : bool, optional
            Whether the catalogue events are original historical events. The
            default is ``True``.

        Returns
        -------
        Earthquake
            Earthquake hazard containing catalogue events in the vicinity of the
            centroids and their MMI intensity footprints.

        Raises
        ------
        ValueError
            If no catalogue events are located within the buffered centroid
            extent.
        """
        all_cent_lat = centroids.lat
        all_cent_lon = centroids.lon

        # Select only catalogue events that could affect the centroid region.
        lon_min, lat_min, lon_max, lat_max = u_coord.latlon_bounds(
            all_cent_lat,
            all_cent_lon,
            buffer=MAX_DIST_DEG,
        )

        event_lon_normalized = u_coord.lon_normalize(
            df["lon"].to_numpy().copy(),
            center=0.5 * (lon_min + lon_max),
        )

        event_mask = (
            (event_lon_normalized >= lon_min)
            & (event_lon_normalized <= lon_max)
            & (df["lat"].to_numpy() >= lat_min)
            & (df["lat"].to_numpy() <= lat_max)
        )

        df = df.loc[event_mask].reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "No earthquake catalogue events are located within "
                f"{MAX_DIST_DEG} degrees of the centroid extent."
            )

        format_date = "%Y-%m-%d %H:%M:%S.%f"
        dates = [
            datetime.strptime(date_str, format_date)
            for date_str in df["date"]
        ]

        years = np.array([date.year for date in dates])
        n_years = years.max() - years.min() + 1

        lat = df["lat"].to_numpy()
        lon = df["lon"].to_numpy()
        mw = df["mw"].to_numpy()
        depths = df["depth"].to_numpy()

        int_list = []
        cent_idx_list = []
        ev_idx_list = []

        for event_idx, (epi_lat, epi_lon, mag, event_depth) in enumerate(
            zip(lat, lon, mw, depths)
        ):
            lon_min, lat_min, lon_max, lat_max = u_coord.latlon_bounds(
                np.array([epi_lat]),
                np.array([epi_lon]),
                buffer=MAX_DIST_DEG,
            )

            cent_lon_normalized = u_coord.lon_normalize(
                all_cent_lon.copy(),
                center=0.5 * (lon_min + lon_max),
            )

            centroid_mask = (
                (cent_lon_normalized >= lon_min)
                & (cent_lon_normalized <= lon_max)
                & (all_cent_lat >= lat_min)
                & (all_cent_lat <= lat_max)
            )

            centroid_indices = np.flatnonzero(centroid_mask)

            if centroid_indices.size == 0:
                continue

            intensities = footprint_MMI(
                all_cent_lat[centroid_mask],
                cent_lon_normalized[centroid_mask],
                epi_lat,
                epi_lon,
                mag,
                event_depth,
            )

            int_list.append(np.asarray(intensities).ravel())
            cent_idx_list.append(centroid_indices)
            ev_idx_list.append(
                np.full(centroid_indices.size, event_idx, dtype=int)
            )

        n_events = len(df)
        n_centroids = len(all_cent_lat)

        if int_list:
            data = np.concatenate(int_list)
            row = np.concatenate(ev_idx_list)
            col = np.concatenate(cent_idx_list)

            intensity = sparse.csr_matrix(
                (data, (row, col)),
                shape=(n_events, n_centroids),
            )
        else:
            intensity = sparse.csr_matrix(
                (n_events, n_centroids),
                dtype=float,
            )

        return cls(
            units="MMI",
            centroids=centroids,
            event_id=df["eventid"].to_numpy(),
            frequency=np.full(n_events, 1.0 / n_years),
            event_name=df["eventid"].astype(str).to_list(),
            date=np.array([date.toordinal() for date in dates]),
            orig=np.full(n_events, orig, dtype=bool),
            intensity=intensity,
        )

    @classmethod
    def uniform_random_events(
        cls,
        df,
        centroids,
        n=1,
        loc_shift_deg=1.0,
        depth_mult=0.2,
        mag_mult=0.5,
        buffer_deg=2.0,
    ):
        """Generate uniformly perturbed synthetic earthquake events.

        Events outside the buffered geographical extent of the centroids are
        excluded before synthetic events are generated. The selection buffer also
        accounts for the maximum location perturbation.

        Parameters
        ----------
        df : pandas.DataFrame
            Earthquake catalogue with columns ``lat``, ``lon``, ``mw``, ``depth``,
            ``eventid``, and ``date``.
        centroids : climada.hazard.centroids.centr.Centroids
            Centroids at which earthquake intensities are computed.
        n : int, optional
            Number of synthetic events generated per catalogue event. The default
            is ``1``.
        loc_shift_deg : float, optional
            Full width, in degrees, of the uniform perturbation interval applied
            independently to latitude and longitude. The default is ``1.0``.
        depth_mult : float, optional
            Full width of the relative uniform perturbation interval applied to
            event depth. The default is ``0.2``.
        mag_mult : float, optional
            Full width of the additive uniform perturbation interval applied to
            moment magnitude. The default is ``0.5``.
        buffer_deg : float, optional
            Additional buffer around the centroid extent in degrees. The default
            is ``2.0``.

        Returns
        -------
        Earthquake
            Earthquake hazard containing the generated synthetic events.
        """
        # Include events that may enter the buffered centroid region after the
        # maximum possible random location shift.
        selection_buffer = buffer_deg + loc_shift_deg / 2

        lat_min = np.min(centroids.lat) - selection_buffer
        lat_max = np.max(centroids.lat) + selection_buffer
        lon_min = np.min(centroids.lon) - selection_buffer
        lon_max = np.max(centroids.lon) + selection_buffer

        event_mask = (
            df["lat"].between(lat_min, lat_max)
            & df["lon"].between(lon_min, lon_max)
        )
        df = df.loc[event_mask].copy()

        lat_orig = df["lat"].to_numpy()
        lon_orig = df["lon"].to_numpy()
        mw_orig = df["mw"].to_numpy()
        depth_orig = df["depth"].to_numpy()

        n_tot = n * len(df)

        lat_rnd = np.random.uniform(
            -loc_shift_deg / 2,
            loc_shift_deg / 2,
            size=n_tot,
        )
        lon_rnd = np.random.uniform(
            -loc_shift_deg / 2,
            loc_shift_deg / 2,
            size=n_tot,
        )
        mw_rnd = np.random.uniform(
            -mag_mult / 2,
            mag_mult / 2,
            size=n_tot,
        )
        depth_rnd = np.random.uniform(
            -depth_mult / 2,
            depth_mult / 2,
            size=n_tot,
        )

        format_date = "%Y-%m-%d %H:%M:%S.%f"
        dates = [
            datetime.strptime(date_str, format_date)
            for date_str in df["date"]
        ]
        years = np.array([date.year for date in dates])
        n_years = np.max(years) - np.min(years)

        new_dates = []
        for n_rnd in range(1, n + 1):
            for date in dates:
                new_year = date.year + n_rnd * n_years
                if date.month == 2 and date.day == 29:
                    new_dates.append(date.replace(year=new_year, day=28))
                else:
                    new_dates.append(date.replace(year=new_year))

        new_dates = [date.strftime(format_date) for date in new_dates]

        rnd_df = pd.DataFrame(
            {
                "lat": lat_rnd + np.tile(lat_orig, n),
                "lon": lon_rnd + np.tile(lon_orig, n),
                "mw": mw_rnd + np.tile(mw_orig, n),
                "depth": depth_rnd + np.tile(depth_orig, n),
                "date": new_dates,
                "eventid": np.arange(1, len(new_dates) + 1),
            }
        )

        return cls.from_Mw_depth(
            df=rnd_df,
            centroids=centroids,
            orig=False,
        )


def footprint_MMI(cent_lat, cent_lon, epi_lat, epi_lon, mag, depth):
    """Compute an earthquake MMI footprint at specified centroids.

    Parameters
    ----------
    cent_lat : array_like
        Latitude coordinates of the centroids in degrees.
    cent_lon : array_like
        Longitude coordinates of the centroids in degrees.
    epi_lat : float
        Epicentre latitude in degrees.
    epi_lon : float
        Epicentre longitude in degrees.
    mag : float
        Earthquake moment magnitude Mw.
    depth : float
        Earthquake focal depth in kilometres.

    Returns
    -------
    numpy.ndarray
        Modified Mercalli Intensity values at the supplied centroids.
    """
    cent_lat, cent_lon = np.array([cent_lat]), np.array([cent_lon])
    dists = u_coord.dist_approx(cent_lat, cent_lon, np.array([[epi_lat]]), np.array([[epi_lon]]))
    return attenuation_MMI(dists, mag, depth)


def attenuation_MMI(
    dist, mag, depth, corr=0.0, a1=1.7, a2=1.5, a3=1.1726, a4=0.00106, b=0.0
):
    """Compute Modified Mercalli Intensity using an attenuation relation.

    The implemented relation follows the MMI attenuation formulation described
    in https://doi.org/10.1201/9781482271645. Values exceeding the
    magnitude-dependent maximum MMI are capped, while values below ``MIN_MMI``
    are set to zero.

    Parameters
    ----------
    dist : array_like
        Source-to-site distances in kilometres.
    mag : float
        Earthquake moment magnitude Mw.
    depth : float
        Earthquake focal depth in kilometres.
    corr : float, optional
        Additive distance correction inside the logarithmic term. The default is
        ``0.0``.
    a1 : float, optional
        Intercept coefficient. The default is ``1.7``.
    a2 : float, optional
        Magnitude coefficient. The default is ``1.5``.
    a3 : float, optional
        Log-distance attenuation coefficient. The default is ``1.1726``.
    a4 : float, optional
        Linear-distance attenuation coefficient. The default is ``0.00106``.
    b : float, optional
        Depth coefficient. The default is ``0.0``.

    Returns
    -------
    numpy.ndarray
        Modified Mercalli Intensity values. Values below ``MIN_MMI`` are zero.
    """
    max_mmi = 1.5 * (mag - 1.0)
    mmi = (
        a1
        + a2 * mag
        - a3 * np.log(np.asarray(dist) + corr)
        - a4 * np.asarray(dist)
        + b * depth
    )
    mmi = np.minimum(mmi, max_mmi)
    return np.where(mmi < MIN_MMI, 0.0, mmi)
