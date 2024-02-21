"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Class for periods and areas along TC track where centroids are reachable by surge
"""

import pathlib
from typing import List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from climada.hazard.trop_cyclone import NM_TO_KM
import climada.util.constants as u_const
import climada.util.coordinates as u_coord
from .plot import plot_surge_events


DEG_TO_NM = 60
"""Unit conversion factors."""


class TCSurgeEvents():
    """Periods and areas along TC track where centroids are reachable by surge

    When iterating over this object, it will return single events represented
    by dictionaries of this form:
        { 'period', 'time_mask', 'time_mask_buffered', 'wind_area',
          'landfall_area', 'surge_areas', 'centroid_mask' }

    Attributes
    ----------
    track : xr.Dataset
        Single tropical cyclone track.
    centroids : 2d ndarray
        Each row is a centroid [lat, lon]. These are supposed to be coastal points of interest.
    d_centroids : 2d ndarray
        For each eye position, distances to centroids.
    nevents : int
        Number of landfall events.
    period : list of tuples
        For each event, a pair of datetime objects indicating beginnig and end
        of landfall event period.
    time_mask : list of ndarray
        For each event, a mask along `track["time"]` indicating the landfall event period.
    time_mask_buffered : list of ndarray
        For each event, a mask along `track["time"]` indicating the landfall event period
        with added buffer for storm form-up.
    wind_area : list of tuples
        For each event, a rectangular box around the geographical area that is affected
        by storm winds during the (buffered) landfall event.
    landfall_area : list of tuples
        For each event, a rectangular box around the geographical area that is affected
        by storm surge during the landfall event.
    surge_areas : list of list of tuples
        For each event, a list of tight rectangular boxes around the centroids that will
        be affected by storm surge during the landfall event.
    centroid_mask : list of ndarray
        For each event, a mask along first axis of `centroids` indicating which centroids are
        reachable by surge during this landfall event.
    """
    keys = [
        'period', 'time_mask', 'time_mask_buffered', 'wind_area', 'landfall_area',
        'surge_areas', 'centroid_mask',
    ]
    """Keys that are available in each dict returned when iterating over this object"""

    maxlen_h = 48
    """The maximum length (in hours) of a landfall period. Longer periods are split up."""

    maxbreak_h = 12
    """The maximum duration (in hours) within a landfall period without affected areas."""

    period_buffer_d = 0.5
    """The buffer (in days) to add for the 'time_mask_buffered' attribute."""

    total_roci_factor = 2.5
    """The factor to use when deriving the 'wind_area' from the ROCI."""

    lf_roci_factor = 0.6
    """The factor to use when deriving the 'landfall_area' from the ROCI."""

    lf_rmw_factor = 2.0
    """The factor to use when deriving the 'landfall_area' from the RMW."""

    minwind_kt = 34
    """The wind speed threshold (in knots) for a track position to be considered."""

    def __init__(self, track : xr.Dataset, centroids : np.ndarray) -> None:
        """Determine temporal periods and geographical regions where the storm
        affects the centroids

        Parameters
        ----------
        track : xr.Dataset
            Single tropical cyclone track.
        centroids : ndarray of shape (ncentroids, 2)
            Each row is a centroid [lat, lon].
        """
        self.track = track
        self.centroids = centroids

        locs = np.stack([self.track["lat"], self.track["lon"]], axis=1)
        self.d_centroids = u_coord.dist_approx(
            locs[None, :, 0], locs[None, :, 1],
            self.centroids[None, :, 0], self.centroids[None, :, 1],
            method="geosphere",
        )[0]

        self._set_periods()
        self.time_mask = [self._period_to_mask(p) for p in self.period]
        self.time_mask_buffered = [
            self._period_to_mask(p, buffer=self.period_buffer_d)
            for p in self.period
        ]
        self._set_areas()
        self._remove_harmless_events()


    def __iter__(self):
        for i_event in range(self.nevents):
            yield {key: getattr(self, key)[i_event] for key in self.keys}


    def __len__(self) -> int:
        return self.nevents


    def _remove_harmless_events(self) -> None:
        """Remove events without affected areas (surge_areas)"""
        relevant_idx = [i for i in range(self.nevents) if len(self.surge_areas[i]) > 0]
        for key in self.keys:
            setattr(self, key, [getattr(self, key)[i] for i in relevant_idx])
        self.nevents = len(relevant_idx)


    def _set_periods(self) -> None:
        """Determine beginning and end of landfall events."""
        radii = np.fmax(
            self.lf_roci_factor * self.track["radius_oci"].values,
            self.lf_rmw_factor * self.track["radius_max_wind"].values,
        ) * NM_TO_KM
        centr_counts = np.count_nonzero(self.d_centroids < radii[:, None], axis=1)
        # below a certain wind speed, winds are not strong enough for significant surge
        mask = (centr_counts > 1) & (self.track["max_sustained_wind"] > self.minwind_kt)

        # convert landfall mask to (clustered) start/end pairs
        period = []
        start = end = None
        for i, date in enumerate(self.track["time"].values):
            if start is not None:
                # periods cover at most {maxlen_h} hours and a split will be forced
                # at breaks of more than {maxbreak_h} hours.
                exceed_maxbreak = (date - end) / np.timedelta64(1, 'h') > self.maxbreak_h
                exceed_maxlen = (date - start) / np.timedelta64(1, 'h') > self.maxlen_h
                if exceed_maxlen or exceed_maxbreak:
                    period.append((start, end))
                    start = end = None
            if mask[i]:
                end = date
                if start is None:
                    start = date
        if start is not None:
            period.append((start, end))
        self.period = period
        self.nevents = len(self.period)


    def _period_to_mask(
        self,
        period : Tuple[np.datetime64, np.datetime64],
        buffer : Union[Tuple[float, float], float] = 0.0,
    ) -> np.ndarray:
        """Compute buffered 1d-mask over track time series from period

        Parameters
        ----------
        period : pair of datetimes
            start/end of period
        buffer : float or pair of floats
            buffer to add (in days)

        Returns
        -------
        mask : ndarray
        """
        if not isinstance(buffer, tuple):
            buffer = (buffer, buffer)
        diff_start = np.array([(t - period[0]) / np.timedelta64(1, 'D')
                               for t in self.track["time"]])
        diff_end = np.array([(t - period[1]) / np.timedelta64(1, 'D')
                             for t in self.track["time"]])
        return (diff_start >= -buffer[0]) & (diff_end <= buffer[1])


    def _set_areas(self) -> None:
        """For each event, determine areas affected by wind and surge."""
        self.wind_area = []
        self.landfall_area = []
        self.surge_areas = []
        self.centroid_mask = []
        for i_event, mask_buf in enumerate(self.time_mask_buffered):
            track = self.track.sel(time=mask_buf)
            mask = self.time_mask[i_event][mask_buf]
            lf_radii = np.fmin(
                track["radius_oci"].values,
                np.fmax(
                    self.lf_roci_factor * track["radius_oci"].values,
                    self.lf_rmw_factor * track["radius_max_wind"].values,
                ),
            )

            # wind area (maximum bounds to consider)
            pad = self.total_roci_factor * track["radius_oci"].values / DEG_TO_NM
            self.wind_area.append(_round_bounds_enlarge(
                (track["lon"].values - pad).min(),
                (track["lat"].values - pad).min(),
                (track["lon"].values + pad).max(),
                (track["lat"].values + pad).max(),
                precision=5,
            ))

            # landfall area
            pad = lf_radii / DEG_TO_NM
            self.landfall_area.append(_round_bounds_enlarge(
                (track["lon"].values - pad)[mask].min(),
                (track["lat"].values - pad)[mask].min(),
                (track["lon"].values + pad)[mask].max(),
                (track["lat"].values + pad)[mask].max(),
                precision=2,
            ))

            # surge areas
            lf_radii *= NM_TO_KM
            centroids_mask = np.any(
                self.d_centroids[mask_buf][mask] < lf_radii[mask, None], axis=0)
            points = self.centroids[centroids_mask, ::-1]
            surge_areas = []
            if points.shape[0] > 0:
                pt_bounds = tuple(points.min(axis=0)) + tuple(points.max(axis=0))
                pt_size = (pt_bounds[2] - pt_bounds[0]) * (pt_bounds[3] - pt_bounds[1])
                if pt_size < (2 * lf_radii.max() / u_const.ONE_LAT_KM)**2:
                    small_bounds = [pt_bounds]
                else:
                    small_bounds, pt_size = _boxcover_points_along_axis(points, 3)
                min_size = 3. / (60. * 60.)
                if pt_size > (2 * min_size)**2:
                    for bounds in small_bounds:
                        surge_areas.append(_round_bounds_enlarge(
                            bounds[0] - min_size,
                            bounds[1] - min_size,
                            bounds[2] + min_size,
                            bounds[3] + min_size,
                            precision=1,
                        ))
            surge_areas = [tuple([float(b) for b in bounds]) for bounds in surge_areas]
            self.surge_areas.append(surge_areas)

            # centroids affected by surge
            centroids_mask = np.zeros(self.centroids.shape[0], dtype=bool)
            for bounds in surge_areas:
                centroids_mask |= (
                    (bounds[0] <= self.centroids[:, 1])
                    & (bounds[1] <= self.centroids[:, 0])
                    & (self.centroids[:, 1] <= bounds[2])
                    & (self.centroids[:, 0] <= bounds[3])
                )
            self.centroid_mask.append(centroids_mask)


    def plot_areas(
        self,
        path : Optional[Union[pathlib.Path, str]] = None,
        pad_deg : float = 5.5,
    ) -> None:
        """Plot areas associated with this track's landfall events

        Parameters
        ----------
        path : Path or str, optional
            If given, save the plots to the given location. Default: None
        pad_deg : float, optional
            Padding (in degrees) to add around total bounds. Default: 5.5
        """
        # plotting-related code is moved to the `.plot` submodule
        plot_surge_events(self, path, pad_deg)


def _round_bounds_enlarge(x_min, y_min, x_max, y_max, precision=1):
    """Round the given bounds to the specified precision, only enlarging the bounds

    The lower bounds are decreased (floor) and the upper bounds are increased (ceil).

    Parameters
    ----------
    x_min, y_min, x_max, y_max : float
        Two-dimensional coordinate bounds.
    precision : float, optional
        The rounding precision. For example, a value of 5 means that values are rounded to the
        next multiple of 5 so that 13 is rounded to 10 (using floor) or 15 (using ceil).
        Default: 1

    Returns
    -------
    x_min, y_min, x_max, y_max : float
        Enlarged and rounded coordinate bounds.
    """
    return (
        np.floor(x_min / precision) * precision,
        np.floor(y_min / precision) * precision,
        np.ceil(x_max / precision) * precision,
        np.ceil(y_max / precision) * precision,
    )


def _boxcover_points_along_axis(points : np.ndarray, nsplits : int) -> Tuple[List[Tuple], float]:
    """Cover n-dimensional points with grid-aligned boxes

    Parameters
    ----------
    points : ndarray of shape (npoints, ndims)
        Each row is an n-dimensional point.
    nsplits : int
        Maximum number of boxes to use.

    Returns
    -------
    boxes : list of tuples (x1_min, x2_min, ..., xn_min, x1_max, x2_max, ..., xn_max)
        Bounds of covering boxes.
    boxes_size : float
        Total volume/area of the covering boxes.
    """
    ndim = points.shape[1]
    bounds_min, bounds_max = points.min(axis=0), points.max(axis=0)
    final_boxes = []
    final_boxes_size = 1 + np.prod(bounds_max - bounds_min)
    for axis in range(ndim):
        splits = [
            ((nsplits - i) / nsplits) * bounds_min[axis] + (i / nsplits) * bounds_max[axis]
            for i in range(1, nsplits)
        ]
        boxes = []
        for i in range(nsplits):
            if i == 0:
                mask = points[:, axis] <= splits[0]
            elif i == nsplits - 1:
                mask = points[:, axis] > splits[-1]
            else:
                mask = (
                    (points[:, axis] <= splits[i])
                    & (points[:, axis] > splits[i - 1])
                )
            masked_points = points[mask, :]
            if masked_points.shape[0] > 0:
                boxes.append((masked_points.min(axis=0), masked_points.max(axis=0)))
        boxes_size = np.sum([np.prod(bmax - bmin) for bmin, bmax in boxes])
        if boxes_size < final_boxes_size:
            final_boxes = [tuple(bmin) + tuple(bmax) for bmin, bmax in boxes]
            final_boxes_size = boxes_size
    return final_boxes, final_boxes_size
