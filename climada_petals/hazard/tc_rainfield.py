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

Define TC rain hazard (TCRain class).
"""

__all__ = ['TCRain']

import datetime as dt
import itertools
import logging
from typing import Optional, Tuple, List

import numpy as np
import pathos.pools
from scipy import sparse
import xarray as xr

from climada.hazard import Hazard, TCTracks, TropCyclone, Centroids
from climada.hazard.trop_cyclone import (
    _track_to_si,
    _close_centroids,
    KM_TO_M,
    KN_TO_MS,
)
from climada.util import ureg
import climada.util.constants as u_const
import climada.util.coordinates as u_coord
from climada.util.tag import Tag

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'TR'
"""Hazard type acronym for TC rain"""

DEF_MAX_DIST_EYE_KM = 300
"""Default value for the maximum distance (in km) of a centroid to the TC center at which rain
rate calculations are done."""

DEF_INTENSITY_THRES = 0.1
"""Default value for the threshold below which rain amounts (in mm) are stored as 0."""

DEF_MAX_MEMORY_GB = 8
"""Default value of the memory limit (in GB) for rain computations (in each thread)."""

MODEL_RAIN = {'R-CLIPER': 0}
"""Enumerate different parametric TC rain models."""

D_TO_H = (1.0 * ureg.days).to(ureg.hours).magnitude
IN_TO_MM = (1.0 * ureg.inches).to(ureg.millimeters).magnitude
"""Unit conversion factors for JIT functions that can't use ureg"""

class TCRain(Hazard):
    """
    Contains rainfall from tropical cyclone events.

    Attributes
    ----------
    category : np.ndarray of ints
        for every event, the TC category using the Saffir-Simpson scale:

        * -1 tropical depression
        *  0 tropical storm
        *  1 Hurrican category 1
        *  2 Hurrican category 2
        *  3 Hurrican category 3
        *  4 Hurrican category 4
        *  5 Hurrican category 5
    basin : list of str
        Basin where every event starts:

        * 'NA' North Atlantic
        * 'EP' Eastern North Pacific
        * 'WP' Western North Pacific
        * 'NI' North Indian
        * 'SI' South Indian
        * 'SP' Southern Pacific
        * 'SA' South Atlantic
    rainrates : list of csr_matrix
        For each event, the rain rates (in mm/h) at each centroid and track position in a sparse
        matrix of shape (npositions, ncentroids).
    """
    intensity_thres = DEF_INTENSITY_THRES
    """intensity threshold for storage in mm"""

    vars_opt = Hazard.vars_opt.union({'category'})
    """Name of the variables that aren't needed to compute the impact."""

    def __init__(
        self,
        category: Optional[np.ndarray] = None,
        basin: Optional[List] = None,
        rainrates: Optional[List[sparse.csr_matrix]] = None,
        **kwargs,
    ):
        """Initialize values.

        Parameters
        ----------
        category : np.ndarray of int, optional
            For every event, the TC category using the Saffir-Simpson scale:
                -1 tropical depression
                0 tropical storm
                1 Hurrican category 1
                2 Hurrican category 2
                3 Hurrican category 3
                4 Hurrican category 4
                5 Hurrican category 5
        basin : list of str, optional
            Basin where every event starts:
                'NA' North Atlantic
                'EP' Eastern North Pacific
                'WP' Western North Pacific
                'NI' North Indian
                'SI' South Indian
                'SP' Southern Pacific
                'SA' South Atlantic
        rainrates : list of csr_matrix, optional
            For each event, the rain rates (in mm/h) at each centroid and track position in a
            sparse matrix of shape (npositions, ncentroids).
        **kwargs : Hazard properties, optional
            All other keyword arguments are passed to the Hazard constructor.
        """
        kwargs.setdefault('haz_type', HAZ_TYPE)
        Hazard.__init__(self, **kwargs)
        self.category = category if category is not None else np.array([], int)
        self.basin = basin if basin is not None else []
        self.rainrates = rainrates if rainrates is not None else []

    def set_from_tracks(self, *args, **kwargs):
        """This function is deprecated, use TCRain.from_tracks instead."""
        LOGGER.warning("The use of TCRain.set_from_tracks is deprecated."
                       "Use TCRain.from_tracks instead.")
        if "intensity_thres" not in kwargs:
            # some users modify the threshold attribute before calling `set_from_tracks`
            kwargs["intensity_thres"] = self.intensity_thres
        if self.pool is not None and 'pool' not in kwargs:
            kwargs['pool'] = self.pool
        self.__dict__ = TCRain.from_tracks(*args, **kwargs).__dict__

    @classmethod
    def from_tracks(
        cls,
        tracks: TCTracks,
        centroids: Optional[Centroids] = None,
        pool: Optional[pathos.pools.ProcessPool] = None,
        description: str = '',
        model: str = 'R-CLIPER',
        ignore_distance_to_coast: bool = False,
        store_rainrates: bool = False,
        metric: str = "equirect",
        intensity_thres: float = DEF_INTENSITY_THRES,
        max_latitude: float = 61,
        max_dist_inland_km: float = 1000,
        max_dist_eye_km: float = DEF_MAX_DIST_EYE_KM,
        max_memory_gb: float = DEF_MAX_MEMORY_GB,
    ):
        """
        Create new TCRain instance that contains rainfields from the specified tracks

        This function sets the `intensity` attribute to contain, for each centroid,
        the total amount of rain experienced over the whole period of each TC event in mm.
        The amount of rain is set to 0 if it does not exceed the threshold `intensity_thres`.

        The `category` attribute is set to the value of the `category`-attribute
        of each of the given track data sets.

        The `basin` attribute is set to the genesis basin for each event, which
        is the first value of the `basin`-variable in each of the given track data sets.

        Optionally, the time-dependent rain rates can be stored using the `store_rainrates`
        function parameter (see below).

        Parameters
        ----------
        tracks : climada.hazard.TCTracks
            Tracks of storm events.
        centroids : Centroids, optional
            Centroids where to model TC. Default: global centroids at 360 arc-seconds resolution.
        pool : pathos.pool, optional
            Pool that will be used for parallel computation of rain fields. Default: None
        description : str, optional
            Description of the event set. Default: "".
        model : str, optional
            Parametric rain model to use: only "R-CLIPER" is currently implemented.
            Default: "R-CLIPER".
        ignore_distance_to_coast : boolean, optional
            If True, centroids far from coast are not ignored. Default: False.
        store_rainrates : boolean, optional
            If True, the Hazard object gets a list `rainrates` of sparse matrices. For each track,
            the rain rates (in mm/h) at each centroid and track position are stored in a sparse
            matrix of shape (npositions, ncentroids). Default: False.
        metric : str, optional
            Specify an approximation method to use for earth distances:

            * "equirect": Distance according to sinusoidal projection. Fast, but inaccurate for
              large distances and high latitudes.
            * "geosphere": Exact spherical distance. Much more accurate at all distances, but slow.

            Default: "equirect".
        intensity_thres : float, optional
            Rain amounts (in mm) below this threshold are stored as 0. Default: 0.1
        max_latitude : float, optional
            No rain calculation is done for centroids with latitude larger than this parameter.
            Default: 61
        max_dist_inland_km : float, optional
            No rain calculation is done for centroids with a distance (in km) to the coast larger
            than this parameter. Default: 1000
        max_dist_eye_km : float, optional
            No rain calculation is done for centroids with a distance (in km) to the
            TC center ("eye") larger than this parameter. Default: 300
        max_memory_gb : float, optional
            To avoid memory issues, the computation is done for chunks of the track sequentially.
            The chunk size is determined depending on the available memory (in GB). Note that this
            limit applies to each thread separately if a `pool` is used. Default: 8

        Returns
        -------
        TCRain
        """
        num_tracks = tracks.size
        if centroids is None:
            centroids = Centroids.from_base_grid(res_as=360, land=True)

        if not centroids.coord.size:
            centroids.set_meta_to_lat_lon()

        if ignore_distance_to_coast:
            # Select centroids with lat <= max_latitude
            coastal_idx = (np.abs(centroids.lat) <= max_latitude).nonzero()[0]
        else:
            # Select centroids which are inside max_dist_inland_km and lat <= max_latitude
            if not centroids.dist_coast.size:
                centroids.set_dist_coast()
            coastal_idx = ((centroids.dist_coast <= max_dist_inland_km * 1000)
                           & (np.abs(centroids.lat) <= max_latitude)).nonzero()[0]

        # Filter early with a larger threshold, but inaccurate (lat/lon) distances.
        # Later, there will be another filtering step with more accurate distances in km.
        max_dist_eye_deg = max_dist_eye_km / (
            u_const.ONE_LAT_KM * np.cos(np.radians(max_latitude))
        )

        # Restrict to coastal centroids within reach of any of the tracks
        t_lon_min, t_lat_min, t_lon_max, t_lat_max = tracks.get_bounds(deg_buffer=max_dist_eye_deg)
        t_mid_lon = 0.5 * (t_lon_min + t_lon_max)
        coastal_centroids = centroids.coord[coastal_idx]
        u_coord.lon_normalize(coastal_centroids[:, 1], center=t_mid_lon)
        coastal_idx = coastal_idx[((t_lon_min <= coastal_centroids[:, 1])
                                   & (coastal_centroids[:, 1] <= t_lon_max)
                                   & (t_lat_min <= coastal_centroids[:, 0])
                                   & (coastal_centroids[:, 0] <= t_lat_max))]

        LOGGER.info('Mapping %s tracks to %s coastal centroids.', str(tracks.size),
                    str(coastal_idx.size))
        if pool:
            chunksize = min(num_tracks // pool.ncpus, 1000)
            tc_haz_list = pool.map(
                cls.from_single_track, tracks.data,
                itertools.repeat(centroids, num_tracks),
                itertools.repeat(coastal_idx, num_tracks),
                itertools.repeat(model, num_tracks),
                itertools.repeat(store_rainrates, num_tracks),
                itertools.repeat(metric, num_tracks),
                itertools.repeat(intensity_thres, num_tracks),
                itertools.repeat(max_dist_eye_km, num_tracks),
                itertools.repeat(max_memory_gb, num_tracks),
                chunksize=chunksize)
        else:
            last_perc = 0
            tc_haz_list = []
            for track in tracks.data:
                perc = 100 * len(tc_haz_list) / len(tracks.data)
                if perc - last_perc >= 10:
                    LOGGER.info("Progress: %d%%", perc)
                    last_perc = perc
                tc_haz_list.append(
                    cls._from_track(track, centroids, coastal_idx,
                                    model=model, store_rainrates=store_rainrates,
                                    metric=metric, intensity_thres=intensity_thres,
                                    max_dist_eye_km=max_dist_eye_km,
                                    max_memory_gb=max_memory_gb))
            if last_perc < 100:
                LOGGER.info("Progress: 100%")

        LOGGER.debug('Concatenate events.')
        haz = cls.concat(tc_haz_list)
        haz.pool = pool
        haz.intensity_thres = intensity_thres
        LOGGER.debug('Compute frequency.')
        TropCyclone.frequency_from_tracks(haz, tracks.data)
        haz.tag.append(Tag(description=description))
        return haz

    @classmethod
    def _from_track(
        cls,
        track: xr.Dataset,
        centroids: Centroids,
        coastal_idx: np.ndarray,
        model: str = 'R-CLIPER',
        store_rainrates: bool = False,
        metric: str = "equirect",
        intensity_thres: float = DEF_INTENSITY_THRES,
        max_dist_eye_km: float = DEF_MAX_DIST_EYE_KM,
        max_memory_gb: float = DEF_MAX_MEMORY_GB,
    ):
        """
        Generate a TC rain hazard object from a single track dataset

        Parameters
        ----------
        track : xr.Dataset
            Single tropical cyclone track.
        centroids : Centroids
            Centroids instance.
        coastal_idx : np.ndarray
            Indices of centroids close to coast.
        model : str, optional
            Parametric rain model to use: only "R-CLIPER" is currently implemented.
            Default: "R-CLIPER".
        store_rainrates : boolean, optional
            If True, store rain rates (in mm/h). Default: False.
        metric : str, optional
            Specify an approximation method to use for earth distances: "equirect" (faster) or
            "geosphere" (more accurate). See `dist_approx` function in `climada.util.coordinates`.
            Default: "equirect".
        intensity_thres : float, optional
            Rain amounts (in mm) below this threshold are stored as 0. Default: 0.1
        max_dist_eye_km : float, optional
            No rain calculation is done for centroids with a distance (in km) to the TC
            center ("eye") larger than this parameter. Default: 300
        max_memory_gb : float, optional
            To avoid memory issues, the computation is done for chunks of the track sequentially.
            The chunk size is determined depending on the available memory (in GB). Default: 8

        Returns
        -------
        TCRain
        """
        intensity_sparse, rainrates_sparse = _compute_rain_sparse(
            track=track,
            centroids=centroids,
            coastal_idx=coastal_idx,
            model=model,
            store_rainrates=store_rainrates,
            metric=metric,
            intensity_thres=intensity_thres,
            max_dist_eye_km=max_dist_eye_km,
            max_memory_gb=max_memory_gb,
        )

        new_haz = cls(haz_type=HAZ_TYPE)
        new_haz.tag = Tag(file_name=f'Name: {track.name}')
        new_haz.intensity_thres = intensity_thres
        new_haz.intensity = intensity_sparse
        if store_rainrates:
            new_haz.rainrates = [rainrates_sparse]
        new_haz.units = 'mm'
        new_haz.centroids = centroids
        new_haz.event_id = np.array([1])
        new_haz.frequency = np.array([1])
        new_haz.event_name = [track.sid]
        new_haz.fraction = sparse.csr_matrix(new_haz.intensity.shape)
        # store first day of track as date
        new_haz.date = np.array([
            dt.datetime(track["time"].dt.year.values[0],
                        track["time"].dt.month.values[0],
                        track["time"].dt.day.values[0]).toordinal()
        ])
        new_haz.orig = np.array([track.orig_event_flag])
        new_haz.category = np.array([track.category])
        new_haz.basin = [str(track["basin"].values[0])]
        return new_haz

def _compute_rain_sparse(
    track: xr.Dataset,
    centroids: Centroids,
    coastal_idx: np.ndarray,
    model: str = 'R-CLIPER',
    store_rainrates: bool = False,
    metric: str = "equirect",
    intensity_thres: float = DEF_INTENSITY_THRES,
    max_dist_eye_km: float = DEF_MAX_DIST_EYE_KM,
    max_memory_gb: float = DEF_MAX_MEMORY_GB,
) -> Tuple[sparse.csr_matrix, Optional[sparse.csr_matrix]]:
    """Version of `compute_rain` that returns sparse matrices and limits memory usage

    Parameters
    ----------
    track : xr.Dataset
        Single tropical cyclone track.
    centroids : Centroids
        Centroids instance.
    coastal_idx : np.ndarray
        Indices of centroids close to coast.
    model : str, optional
        Parametric rain model to use: only "R-CLIPER" is currently implemented.
        Default: "R-CLIPER".
    store_rainrates : boolean, optional
        If True, store rain rates. Default: False.
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See `dist_approx` function in `climada.util.coordinates`.
        Default: "equirect".
    intensity_thres : float, optional
        Wind speeds (in m/s) below this threshold are stored as 0. Default: 17.5
    max_dist_eye_km : float, optional
        No rain calculation is done for centroids with a distance (in km) to the TC
        center ("eye") larger than this parameter. Default: 300
    max_memory_gb : float, optional
        To avoid memory issues, the computation is done for chunks of the track sequentially.
        The chunk size is determined depending on the available memory (in GB). Default: 8

    Raises
    ------
    ValueError

    Returns
    -------
    intensity : csr_matrix
        Total amount of rain (in mm) in each centroid over the whole storm life time.
    rainrates : csr_matrix or None
        If store_rainrates is True, the rain rates at each centroid and track position
        are stored in a sparse matrix of shape (npositions,  ncentroids ).
        If store_rainrates is False, `None` is returned.
    """
    try:
        mod_id = MODEL_RAIN[model]
    except KeyError as err:
        raise ValueError(f'Model not implemented: {model}.') from err

    ncentroids = centroids.coord.shape[0]
    coastal_centr = centroids.coord[coastal_idx]
    npositions = track.sizes["time"]
    rainrates_shape = (npositions, ncentroids)
    intensity_shape = (1, ncentroids)

    # Split into chunks so that 5 arrays with `coastal_centr.size` entries can be stored for
    # each position in a chunk:
    memreq_per_pos_gb = (8 * 5 * max(1, coastal_centr.size)) / 1e9
    max_chunksize = max(2, int(max_memory_gb / memreq_per_pos_gb) - 1)
    n_chunks = int(np.ceil(npositions / max_chunksize))
    if n_chunks > 1:
        return _compute_rain_sparse_chunked(
            n_chunks,
            track,
            centroids,
            coastal_idx,
            model=model,
            store_rainrates=store_rainrates,
            metric=metric,
            intensity_thres=intensity_thres,
            max_dist_eye_km=max_dist_eye_km,
            max_memory_gb=max_memory_gb,
        )

    rainrates, reachable_centr_idx = compute_rain(
        track, coastal_centr, mod_id, metric=metric, max_dist_eye_km=max_dist_eye_km,
        max_memory_gb=0.8 * max_memory_gb,
    )
    reachable_coastal_centr_idx = coastal_idx[reachable_centr_idx]
    npositions = rainrates.shape[0]

    intensity = (rainrates * track["time_step"].values[:, None]).sum(axis=0)
    intensity[intensity < intensity_thres] = 0
    intensity_sparse = sparse.csr_matrix(
        (intensity, reachable_coastal_centr_idx, [0, intensity.size]),
        shape=intensity_shape)
    intensity_sparse.eliminate_zeros()

    rainrates_sparse = None
    if store_rainrates:
        n_reachable_coastal_centr = reachable_coastal_centr_idx.size
        indices = np.zeros((npositions, n_reachable_coastal_centr, 2), dtype=np.int64)
        indices[:, :, 0] = 2 * reachable_coastal_centr_idx[None]
        indices[:, :, 1] = 2 * reachable_coastal_centr_idx[None] + 1
        indices = indices.ravel()
        indptr = np.arange(npositions + 1) * n_reachable_coastal_centr * 2
        rainrates_sparse = sparse.csr_matrix((rainrates.ravel(), indices, indptr),
                                              shape=rainrates_shape)
        rainrates_sparse.eliminate_zeros()

    return intensity_sparse, rainrates_sparse

def _compute_rain_sparse_chunked(
    n_chunks: int,
    track: xr.Dataset,
    *args,
    **kwargs,
) -> Tuple[sparse.csr_matrix, Optional[sparse.csr_matrix]]:
    """Call `_compute_rain_sparse` for chunks of the track and re-assemble the results

    Parameters
    ----------
    n_chunks : int
        Number of chunks to use.
    track : xr.Dataset
        Single tropical cyclone track.
    args, kwargs :
        The remaining arguments are passed on to `_compute_rain_sparse`.

    Returns
    -------
    intensity, rainrates :
        See `_compute_rain_sparse` for a description of the return values.
    """
    npositions = track.sizes["time"]
    chunks = np.array_split(np.arange(npositions), n_chunks)
    intensities = []
    rainrates = []
    for i, chunk in enumerate(chunks):
        # generate an overlap of 2 time steps between consecutive chunks:
        chunk = ([] if i == 0 else chunks[i - 1][-2:].tolist()) + chunk.tolist()
        inten, rainr = _compute_rain_sparse(track.isel(time=chunk), *args, **kwargs)
        if rainr is None:
            rainrates = None
        else:
            rainrates.append(rainr[slice(
                    0 if i == 0 else 1,
                    -1 if i + 1 < n_chunks else None,
            )])
        intensities.append(inten)
    intensity = sparse.csr_matrix(sparse.vstack(intensities, format="csr").max(axis=0, ))
    if rainrates is not None:
        rainrates = sparse.vstack(rainrates, format="csr")
    return intensity, rainrates

def compute_rain(
    track: xr.Dataset,
    centroids: np.ndarray,
    model: int,
    metric: str = "equirect",
    max_dist_eye_km: float = DEF_MAX_DIST_EYE_KM,
    max_memory_gb: float = DEF_MAX_MEMORY_GB,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rain rate (in mm/h) of the tropical cyclone

    In a first step, centroids within reach of the track are determined so that rain rates will
    only be computed and returned for those centroids.

    Parameters
    ----------
    track : xr.Dataset
        Track information.
    centroids : np.ndarray with two dimensions
        Each row is a centroid [lat, lon].
        Centroids that are not within reach of the track are ignored.
    model : int
        TC rain model selection according to MODEL_RAIN.
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See `dist_approx` function in `climada.util.coordinates`.
        Default: "equirect".
    max_dist_eye_km : float, optional
        No rain calculation is done for centroids with a distance (in km) to the TC center
        ("eye") larger than this parameter. Default: 300
    max_memory_gb : float, optional
        To avoid memory issues, the computation is done for chunks of the track sequentially.
        The chunk size is determined depending on the available memory (in GB). Default: 8

    Returns
    -------
    rainrates : np.ndarray of shape (npositions, nreachable)
        Rain rates for each track position on those centroids within reach of the TC track.
    reachable_centr_idx : np.ndarray of shape (nreachable,)
        List of indices of input centroids within reach of the TC track.
    """
    # start with the assumption that no centroids are within reach
    npositions = track.sizes["time"]
    reachable_centr_idx = np.zeros((0,), dtype=np.int64)
    rainrates = np.zeros((npositions, 0), dtype=np.float64)

    # convert track variables to SI units
    si_track = _track_to_si(track, metric=metric)

    # normalize longitude values (improves performance of `dist_approx` and `_close_centroids`)
    u_coord.lon_normalize(centroids[:, 1], center=si_track.attrs["mid_lon"])

    # Filter early with a larger threshold, but inaccurate (lat/lon) distances.
    # There is another filtering step with more accurate distances in km later.
    max_dist_eye_deg = max_dist_eye_km / (
        u_const.ONE_LAT_KM * np.cos(np.radians(np.abs(si_track["lat"].values).max()))
    )

    # restrict to centroids within rectangular bounding boxes around track positions
    track_centr_msk = _close_centroids(
        si_track["lat"].values,
        si_track["lon"].values,
        centroids,
        max_dist_eye_deg,
    )
    track_centr = centroids[track_centr_msk]
    nreachable = track_centr.shape[0]
    if nreachable == 0:
        return rainrates, reachable_centr_idx

    # The memory requirements for each track position are estimated for the case of 10 arrays
    # containing `nreachable` float64 (8 Byte) values each. The chunking is only relevant in
    # extreme cases with a very high temporal and/or spatial resolution.
    memreq_per_pos_gb = (8 * 10 * nreachable) / 1e9
    max_chunksize = max(2, int(max_memory_gb / memreq_per_pos_gb) - 1)
    n_chunks = int(np.ceil(npositions / max_chunksize))
    if n_chunks > 1:
        return _compute_rain_chunked(
            n_chunks, track, centroids, model, metric=metric, max_dist_eye_km=max_dist_eye_km,
        )

    # compute distances (in m) to all centroids
    [d_centr] = u_coord.dist_approx(
        si_track["lat"].values[None], si_track["lon"].values[None],
        track_centr[None, :, 0], track_centr[None, :, 1],
        log=False, normalize=False, method=metric, units="m")

    # exclude centroids that are too far from or too close to the eye
    close_centr_msk = (d_centr <= max_dist_eye_km * KM_TO_M) & (d_centr > 1)
    if not np.any(close_centr_msk):
        return rainrates, reachable_centr_idx

    rainrates = _rcliper(si_track, d_centr, close_centr_msk)
    [reachable_centr_idx] = track_centr_msk.nonzero()
    return rainrates, reachable_centr_idx

def _compute_rain_chunked(
    n_chunks: int,
    track: xr.Dataset,
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Call `compute_rain` for chunks of the track and re-assemble the results

    Parameters
    ----------
    n_chunks : int
        Number of chunks to use.
    track : xr.Dataset
        Single tropical cyclone track.
    args, kwargs :
        The remaining arguments are passed on to `compute_rain`.

    Returns
    -------
    rainrates, reachable_centr_idx :
        See `compute_rain` for a description of the return values.
    """
    npositions = track.sizes["time"]
    chunks = np.array_split(np.arange(npositions), n_chunks)
    results = [
        compute_rain(track.isel(time=chunk), *args, **kwargs)
        for chunk in [
            # generate an overlap between consecutive chunks:
            ([] if i == 0 else chunks[i - 1][-2:].tolist()) + chunk.tolist()
            for i, chunk in enumerate(chunks)
        ]
    ]
    # concatenate the results into one
    reachable_centr_idx, reachable_centr_inv = np.unique(
        np.concatenate([d[1] for d in results]),
        return_inverse=True,
    )
    rainrates = np.zeros((npositions, reachable_centr_idx.size))
    split_indices = np.cumsum([d[1].size for d in results])[:-1]
    reachable_centr_inv = np.split(reachable_centr_inv, split_indices)
    for chunk, (arr, _), inv in zip(chunks, results, reachable_centr_inv):
        chunk_start, chunk_end = chunk[[0, -1]]
        # remove overlapping positions from chunk data
        offset = 0
        if chunk_start > 0:
            chunk_start -= 1
            offset_start = 1
        rainrates[chunk_start:chunk_end + 1, inv] = arr[offset:, :]
    return rainrates, reachable_centr_idx

def _rcliper(si_track, d_centr, close_centr):
    """Compute rain rate (in mm/h) from maximum wind speeds using the R-CLIPER model

    The model is defined in equations (3)-(5) and Table 2 (NHC) in the following publication:

    Tuleya et al. (2007): Evaluation of GFDL and Simple Statistical Model Rainfall Forecasts for
    U.S. Landfalling Tropical Storms. Weather and Forecasting 22(1): 56–70.
    https://doi.org/10.1175/WAF972.1

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `_track_to_si`.
    d_centr : np.ndarray of shape (npositions, ncentroids)
        Distance (in m) between centroids and track positions.
    close_centr : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.

    Returns
    -------
    ndarray of shape (npositions, ncentroids)
    """
    rainrate = np.zeros_like(d_centr)
    d_centr, v_max = [
        ar[close_centr] for ar in np.broadcast_arrays(
            d_centr, si_track["vmax"].values[:, None],
        )
    ]

    # bias-adjusted coefficients from Tuleya et al. 2007, Table 2 (NHC)
    # a1, a2, b1, b2 are in "inch per day" units
    # a3, a4, b3, b4 are in "km" units
    a1 = -1.10
    a2 = -1.60
    a3 = 64.5
    a4 = 150.0
    b1 = 3.96
    b2 = 4.80
    b3 = -13.0
    b4 = -16.0

    # u_norm : normalized maximum winds (unitless)
    u_norm = 1. + (v_max / KN_TO_MS - 35.) / 33.

    # rainr_0 : rain rate at r=0 (in "inch per day")
    rainr_0 = a1 + b1 * u_norm

    # rainr_m : rain rate at r=rad_m (in "inch per day")
    rainr_m = a2 + b2 * u_norm

    # rad_m : radial extent of the inner core (in km)
    rad_m = a3 + b3 * u_norm

    # rad_e : measure of the radial extent of the TC rain field (in km)
    rad_e = a4 + b4 * u_norm

    # convert radii to km
    d_centr_km = d_centr / KM_TO_M

    rainrate_close = np.zeros_like(d_centr)

    # rain rate inside of inner core
    msk = (d_centr_km <= rad_m)
    rainrate_close[msk] = (
        rainr_0[msk] + (rainr_m[msk] - rainr_0[msk]) * (d_centr_km[msk] / rad_m[msk])
    )

    # rain rate outside of inner core
    msk = (d_centr_km > rad_m)
    rainrate_close[msk] = rainr_m[msk] * np.exp(-(d_centr_km[msk] - rad_m[msk]) / rad_e[msk])

    rainrate_close[np.isnan(rainrate_close)] = 0
    rainrate_close[rainrate_close < 0] = 0

    # convert from "inch per day" to mm/h
    rainrate_close *= IN_TO_MM / D_TO_H

    rainrate[close_centr] = rainrate_close
    return rainrate
