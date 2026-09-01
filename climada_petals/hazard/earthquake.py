"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define EQ earthquake (Earthquake class).
"""

__all__ = ["Earthquake"]

from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse

from climada.hazard.base import Hazard
from climada.util import coordinates as u_coord

HAZ_TYPE = "EQ"
"""Hazard type acronym for earthquakes."""

MIN_MMI = 3
"""Minimum non-zero MMI value."""

MAX_DIST_DEG = 3
"""Maximum centroid distance from an epicentre in degrees."""

_DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


class Earthquake(Hazard):
    """Earthquake hazard represented by Modified Mercalli Intensities.

    This class stores earthquake events and their spatial intensity footprints
    on a set of centroids. Intensities are expressed on the Modified Mercalli
    Intensity (MMI) scale.

    The default MII is computed using an attenuation relation.
    Alternative parametrizations are available at
    https://doi.org/10.1201/9781482271645, chapter 2.4.4.

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
        discarded before intensity footprints are computed. Event frequency is
        derived from the temporal coverage of the input catalogue before this
        geographic filtering is applied.

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
            Earthquake hazard containing catalogue events in the vicinity of
            the centroids and their MMI intensity footprints.

        Raises
        ------
        ValueError
            If the catalogue or centroid set is empty, or if no catalogue
            events are located within the buffered centroid extent.
        """
        if df.empty:
            raise ValueError("The earthquake catalogue is empty.")
        if centroids.lat.size == 0:
            raise ValueError("The centroid set is empty.")

        # Preserve the temporal coverage of the complete input catalogue so
        # that geographic filtering does not inflate event frequencies.
        catalogue_dates = [
            datetime.strptime(date_str, _DATE_FORMAT)
            for date_str in df["date"]
        ]
        catalogue_years = np.array([date.year for date in catalogue_dates])
        n_years = catalogue_years.max() - catalogue_years.min() + 1

        all_cent_lat = centroids.lat
        all_cent_lon = centroids.lon

        # Select only catalogue events that could affect the centroid region.
        lon_min, lat_min, lon_max, lat_max = u_coord.latlon_bounds(
            all_cent_lat,
            all_cent_lon,
            buffer=MAX_DIST_DEG,
        )
        lon_center = 0.5 * (lon_min + lon_max)
        event_lon_normalized = u_coord.lon_normalize(
            df["lon"].to_numpy().copy(),
            center=lon_center,
        )

        event_mask = (
            (event_lon_normalized >= lon_min)
            & (event_lon_normalized <= lon_max)
            & df["lat"].between(lat_min, lat_max).to_numpy()
        )
        df = df.loc[event_mask].reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "No earthquake catalogue events are located within "
                f"{MAX_DIST_DEG} degrees of the centroid extent."
            )

        dates = [
            datetime.strptime(date_str, _DATE_FORMAT)
            for date_str in df["date"]
        ]

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
            lon_center = 0.5 * (lon_min + lon_max)

            cent_lon_normalized = u_coord.lon_normalize(
                all_cent_lon.copy(),
                center=lon_center,
            )
            epi_lon_normalized = u_coord.lon_normalize(
                np.array([epi_lon]),
                center=lon_center,
            )[0]

            centroid_mask = (
                (cent_lon_normalized >= lon_min)
                & (cent_lon_normalized <= lon_max)
                & (all_cent_lat >= lat_min)
                & (all_cent_lat <= lat_max)
            )
            centroid_indices = np.flatnonzero(centroid_mask)

            if centroid_indices.size == 0:
                continue

            intensities = np.asarray(
                footprint_MMI(
                    all_cent_lat[centroid_mask],
                    cent_lon_normalized[centroid_mask],
                    epi_lat,
                    epi_lon_normalized,
                    mag,
                    event_depth,
                )
            ).ravel()

            # Do not store explicit zeros in the sparse intensity matrix.
            nonzero = intensities > 0
            if not np.any(nonzero):
                continue

            int_list.append(intensities[nonzero])
            cent_idx_list.append(centroid_indices[nonzero])
            ev_idx_list.append(
                np.full(np.count_nonzero(nonzero), event_idx, dtype=int)
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
            intensity.eliminate_zeros()
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
        excluded before synthetic events are generated. The selection buffer
        accounts for the maximum location perturbation.

        For every retained event, ``n`` synthetic events are generated by
        applying independent uniform perturbations to latitude, longitude,
        magnitude, and depth. Synthetic event dates are shifted into subsequent
        catalogue-length time periods.

        Parameters
        ----------
        df : pandas.DataFrame
            Earthquake catalogue with columns ``lat``, ``lon``, ``mw``,
            ``depth``, ``eventid``, and ``date``.
        centroids : climada.hazard.centroids.centr.Centroids
            Centroids at which earthquake intensities are computed.
        n : int, optional
            Number of synthetic events generated per catalogue event. The
            default is ``1``.
        loc_shift_deg : float, optional
            Full width, in degrees, of the uniform perturbation interval
            applied independently to latitude and longitude. The default is
            ``1.0``.
        depth_mult : float, optional
            Full width of the relative uniform perturbation interval applied to
            event depth. For example, ``0.2`` yields factors between 0.9 and
            1.1. The default is ``0.2``.
        mag_mult : float, optional
            Full width of the additive uniform perturbation interval applied to
            moment magnitude. The default is ``0.5``.
        buffer_deg : float, optional
            Additional buffer around the centroid extent in degrees. The
            default is ``2.0``.

        Returns
        -------
        Earthquake
            Earthquake hazard containing the generated synthetic events.

        Raises
        ------
        ValueError
            If ``n`` is not a positive integer, the catalogue or centroid set
            is empty, or no catalogue events lie within the buffered centroid
            extent.
        """
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("'n' must be a positive integer.")
        if df.empty:
            raise ValueError("The earthquake catalogue is empty.")
        if centroids.lat.size == 0:
            raise ValueError("The centroid set is empty.")

        # Determine the catalogue duration before geographic filtering so that
        # filtering does not shorten the synthetic time shifts.
        catalogue_dates = [
            datetime.strptime(date_str, _DATE_FORMAT)
            for date_str in df["date"]
        ]
        catalogue_years = np.array([date.year for date in catalogue_dates])
        n_years = catalogue_years.max() - catalogue_years.min() + 1

        selection_buffer = buffer_deg + loc_shift_deg / 2

        lon_min, lat_min, lon_max, lat_max = u_coord.latlon_bounds(
            centroids.lat,
            centroids.lon,
            buffer=selection_buffer,
        )
        lon_center = 0.5 * (lon_min + lon_max)
        event_lon_normalized = u_coord.lon_normalize(
            df["lon"].to_numpy().copy(),
            center=lon_center,
        )

        event_mask = (
            (event_lon_normalized >= lon_min)
            & (event_lon_normalized <= lon_max)
            & df["lat"].between(lat_min, lat_max).to_numpy()
        )
        df = df.loc[event_mask].reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "No catalogue events are located within the buffered "
                "centroid extent."
            )

        lat_orig = df["lat"].to_numpy()
        lon_orig = df["lon"].to_numpy()
        mw_orig = df["mw"].to_numpy()
        depth_orig = df["depth"].to_numpy()

        n_tot = n * len(df)

        lat_shift = np.random.uniform(
            -loc_shift_deg / 2,
            loc_shift_deg / 2,
            size=n_tot,
        )
        lon_shift = np.random.uniform(
            -loc_shift_deg / 2,
            loc_shift_deg / 2,
            size=n_tot,
        )
        mag_shift = np.random.uniform(
            -mag_mult / 2,
            mag_mult / 2,
            size=n_tot,
        )
        depth_factor = 1.0 + np.random.uniform(
            -depth_mult / 2,
            depth_mult / 2,
            size=n_tot,
        )

        new_lat = lat_shift + np.tile(lat_orig, n)
        new_lon = lon_shift + np.tile(lon_orig, n)
        new_mw = mag_shift + np.tile(mw_orig, n)
        new_depth = depth_factor * np.tile(depth_orig, n)

        new_lat = np.clip(new_lat, -90.0, 90.0)
        new_lon = u_coord.lon_normalize(new_lon)

        dates = [
            datetime.strptime(date_str, _DATE_FORMAT)
            for date_str in df["date"]
        ]

        # The numerical arrays are ordered by realization: the full retained
        # catalogue for realization 1, followed by realization 2, and so on.
        new_dates = []
        for realization in range(1, n + 1):
            for date in dates:
                new_year = date.year + realization * n_years
                if date.month == 2 and date.day == 29:
                    new_date = date.replace(year=new_year, day=28)
                else:
                    new_date = date.replace(year=new_year)
                new_dates.append(new_date.strftime(_DATE_FORMAT))

        rnd_df = pd.DataFrame(
            {
                "lat": new_lat,
                "lon": new_lon,
                "mw": new_mw,
                "depth": new_depth,
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
    cent_lat = np.array([cent_lat])
    cent_lon = np.array([cent_lon])
    dists = u_coord.dist_approx(
        cent_lat,
        cent_lon,
        np.array([[epi_lat]]),
        np.array([[epi_lon]]),
    )
    return attenuation_MMI(dists, mag, depth)


def attenuation_MMI(
    dist,
    mag,
    depth,
    corr=0.0,
    a1=1.7,
    a2=1.5,
    a3=1.1726,
    a4=0.00106,
    b=0.0,
):
    """Compute Modified Mercalli Intensity using an attenuation relation.

    The implemented relation follows the MMI attenuation formulation described
    in https://doi.org/10.1201/9781482271645 (alternative parametrizations are available at
    in Chapter 2.4.4).
    Values exceeding the magnitude-dependent maximum MMI are capped, while values
    below ``MIN_MMI`` are set to zero.


    Parameters
    ----------
    dist : array_like
        Source-to-site distances in kilometres.
    mag : float
        Earthquake moment magnitude Mw.
    depth : float
        Earthquake focal depth in kilometres.
    corr : float, optional
        Additive distance correction inside the logarithmic term. The default
        is ``0.0``.
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

    Raises
    ------
    ValueError
        If any value of ``dist + corr`` is negative.
    """
    dist = np.asarray(dist)

    if np.any(dist + corr < 0):
        raise ValueError("'dist + corr' must not be negative.")

    max_mmi = 1.5 * (mag - 1.0)

    # At zero distance and zero correction, log(0) deliberately produces
    # -infinity. The resulting positive infinity is subsequently capped at the
    # magnitude-dependent maximum MMI.
    with np.errstate(divide="ignore", invalid="ignore"):
        mmi = (
            a1
            + a2 * mag
            - a3 * np.log(dist + corr)
            - a4 * dist
            + b * depth
        )

    mmi = np.minimum(mmi, max_mmi)
    return np.where(mmi < MIN_MMI, 0.0, mmi)
