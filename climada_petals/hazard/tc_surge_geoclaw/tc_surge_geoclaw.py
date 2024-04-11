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

Inundation from TC storm surges, modeled using the flow solver GeoClaw
"""

import datetime as dt
import logging
import pathlib
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import rasterio
from scipy import sparse
import xarray as xr

from climada import CONFIG
from climada.hazard import Centroids, Hazard, TropCyclone, TCTracks
from climada.hazard.tc_tracks import estimate_rmw, estimate_roci
import climada.util.coordinates as u_coord
from .geoclaw_runner import GeoClawRunner
from .setup_clawpack import setup_clawpack
from .tc_surge_events import TCSurgeEvents


LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'TCSurgeGeoClaw'
"""Hazard type acronym for this module."""

GEOCLAW_WORK_DIR = CONFIG.hazard.tc_surge_geoclaw.geoclaw_work_dir.dir()
"""Base directory for GeoClaw run data."""


class TCSurgeGeoClaw(Hazard):
    """TC storm surge heights in meters (m), modeled using GeoClaw.

    Note that this feature only works on Linux and Mac since Windows is not supported by GeoClaw.
    Due to the high computational demand, this functionality should be run on an HPC cluster with
    decent amounts of memory and processors (at least 32 GB and 8 cores) available. Only for
    testing purposes, it makes sense to run this functionality on a smaller machine at lower
    resolution.

    Attributes
    ----------
    category : ndarray of ints
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
    gauge_data : list of lists of dicts
        For each storm and each gauge, a dict containing the `location` of the gauge, and
        (for each landfall event) `base_sea_level`, `topo_height`, `time`, `height_above_geoid`,
        `height_above_ground`, and `amr_level` information.
        Due to this format, this data will NOT be stored when using `write_hdf5`. However, you
        can manually pickle it in a separate file using the `write_gauge_data` method.
    """
    vars_opt = Hazard.vars_opt.union({'category'})
    """Names of variables that are not needed to compute the impact."""

    def __init__(
        self,
        category: Optional[np.ndarray] = None,
        basin: Optional[List[str]] = None,
        gauge_data: Optional[List[List[Dict]]] = None,
        **kwargs,
    ):
        """Initialize values.

        Parameters
        ----------
        category : ndarray of int, optional
            For every event, the TC category using the Saffir-Simpson scale:

            * -1 tropical depression
            *  0 tropical storm
            *  1 Hurrican category 1
            *  2 Hurrican category 2
            *  3 Hurrican category 3
            *  4 Hurrican category 4
            *  5 Hurrican category 5
        basin : list of str, optional
            Basin where every event starts:

            * 'NA' North Atlantic
            * 'EP' Eastern North Pacific
            * 'WP' Western North Pacific
            * 'NI' North Indian
            * 'SI' South Indian
            * 'SP' Southern Pacific
            * 'SA' South Atlantic
        gauge_data : list of lists of dicts
            For each storm and each gauge, a dict containing the `location` of the gauge, and
            (for each landfall event) `base_sea_level`, `topo_height`, `time`,
            `height_above_geoid`, `height_above_ground`, and `amr_level` information.
            Due to this format, this data will NOT be stored when using `write_hdf5`. However, you
            can manually pickle it in a separate file using the `write_gauge_data` method.
        """
        kwargs.setdefault('haz_type', HAZ_TYPE)
        Hazard.__init__(self, **kwargs)
        self.category = category if category is not None else np.array([], int)
        self.basin = basin if basin is not None else []
        self.gauge_data = gauge_data if gauge_data is not None else []

    @classmethod
    def from_tc_tracks(
        cls,
        tracks : TCTracks,
        centroids : Centroids,
        topo_path : Union[pathlib.Path, str],
        max_dist_eye_deg : float = 5.5,
        max_dist_inland_km : float = 50.0,
        max_dist_offshore_km : float = 10.0,
        max_latitude : float = 61.0,
        geoclaw_kwargs : Optional[dict] = None,
        pool : Any = None,
    ):
        """Generate a TC surge hazard instance from a TCTracks object

        Note that this feature only works on Linux and Mac since Windows is not supported by
        GeoClaw. Due to the high computational demand, this functionality should be run on an HPC
        cluster with decent amounts of memory and processors (at least 32 GB and 8 cores)
        available. Only for testing purposes, it makes sense to run this functionality on a smaller
        machine at lower resolution.

        It is required to run this method (or the function `setup_clawpack`) with a working
        internet connection at least once to trigger the download and installation of the flow
        solver (GeoClaw).

        Before you can run several instances of this method in parallel, e.g. on an HPC cluster,
        make sure to run a single instance of this method since all instances of this method will
        be accessing the same version of the solver, and compilation might be triggered in all
        instances at the same time. The same applies if you want to recompile using the GeoClaw
        runner parameter ``recompile=True``.

        By default, the flow solver (GeoClaw) is configured to use multiple OpenMP threads with
        their number equal to the number of physical CPU cores in the machine. You can change this
        behavior by setting the environment variable OMP_NUM_THREADS to the desired number of
        threads. Note, however, that changes to OMP_NUM_THREADS will only be effective if you set
        the GeoClaw parameter ``recompile=True`` at least once.

        Parameters
        ----------
        tracks : TCTracks
            Tracks of tropical cyclone events.
        centroids : Centroids
            Centroids where to measure maximum surge heights.
        topo_path : Path or str
            Path to raster file containing gridded elevation data.
        max_dist_eye_deg : float, optional
            Maximum distance from a TC track node in degrees for a centroid to be considered
            as potentially affected. Default: 5.5
        max_dist_inland_km : float, optional
            Maximum inland distance of the centroids in kilometers. Default: 50
        max_dist_offshore_km : float, optional
            Maximum offshore distance of the centroids in kilometers. Default: 10
        max_latitude : float, optional
            Maximum latitude of potentially affected centroids. Default: 61
        geoclaw_kwargs : dict, optional
            Optional keyword arguments to pass to the GeoClaw runner. Currently supported:

            topo_res_as : float
                The resolution at which to extract topography data in arc-seconds. Needs to be at
                least 3 since lower values have been found to be unstable numerically. Default: 30
            gauges : list of pairs (lat, lon)
                The locations of tide gauges where to measure temporal changes in sea level height.
                This is used mostly for validation purposes. The result is stored in the
                `gauge_data` attribute.
            sea_level : float or function
                The sea level (above geoid) of the ocean at rest, used as a starting level for the
                surge simulation. Instead of a constant scalar value, a function can be specified
                that gets a `bounds` and a `period` argument and returns a scalar value. In this
                case, the first argument is a tuple of floats (lon_min, lat_min, lon_max, lat_max)
                and the second argument is a pair of np.datetime64 (start, end). For example, see
                the helper function `sea_level_from_nc` that reads the value from a NetCDF file.
                Default: 0
            outer_pad_deg : float
                An additional padding (in degrees) around the model domain where the automatic mesh
                refinement is disabled to stabilize boundary interactions. If you find that your
                run of GeoClaw is numerically unstable, takes exceedingly long, or produces
                unrealistic results, it might help to modify this parameter by a few degrees.
                Default: 5
            boundary_conditions : str
                One of "extrap" (extrapolation, non-reflecting outflow), "periodic", or "wall"
                (reflecting, solid wall boundary conditions). For more information about the
                possible settings, see the chapter "Boundary conditions" in the Clawpack
                documentation. Default: "extrap"
            output_freq_s : float
                Frequency of writing GeoClaw output files (for debug use) in 1/seconds. No output
                files are written if the value is 0.0. Default: 0.0
            recompile : bool
                If True, force the GeoClaw Fortran code to be recompiled. Note that, without
                recompilation, changes to environment variables like FC, FFLAGS or OMP_NUM_THREADS
                are ignored! Default: False
            resume_file : Path or str
                If given, use this file to remember the location of the run directory and resume
                operation later from this directory if it already exists. Default: None
            resume_i : int
                In case of a single storm, this points to the line number (starting from 0) in the
                ``resume`` file to consider for that event. Default: not specified.

            Default: None
        pool : an object with `map` functionality, optional
            If given, landfall events for each track are processed in parallel. Note that the
            solver for a single landfall event is using OpenMP multiprocessing capabilities
            already. You will only benefit from processing these OpenMP tasks in parallel if a
            sufficient number of CPUs is available, e.g. with MPI multitasking on a cluster.
            To control the use of OpenMP, set the environment variable `OMP_NUM_THREADS` to the
            number of cores and set the compiler flag `export FFLAGS='-O2 -fopenmp'`, following
            the [GeoClaw docs](https://www.clawpack.org/openmp.html).

        Returns
        -------
        TCSurgeGeoClaw
        """
        geoclaw_kwargs = {} if geoclaw_kwargs is None else geoclaw_kwargs

        if tracks.size == 0:
            raise ValueError("The given TCTracks object does not contain any tracks.")
        setup_clawpack()

        max_dist_coast_km = (max_dist_offshore_km, max_dist_inland_km)
        coastal_idx = _get_coastal_centroids_idx(
            centroids, max_dist_coast_km, max_latitude=max_latitude,
        )

        LOGGER.info('Computing TC surge of %d tracks on %d centroids.',
                    tracks.size, coastal_idx.size)

        haz = cls.concat([
            cls.from_xr_track(
                track,
                centroids,
                coastal_idx,
                topo_path,
                max_dist_eye_deg=max_dist_eye_deg,
                geoclaw_kwargs={"resume_i": resume_i, **geoclaw_kwargs},
                pool=pool,
            )
            for resume_i, track in enumerate(tracks.data)
        ])
        TropCyclone.frequency_from_tracks(haz, tracks.data)
        return haz

    @classmethod
    def from_xr_track(
        cls,
        track : xr.Dataset,
        centroids : Centroids,
        coastal_idx : np.ndarray,
        topo_path : Union[pathlib.Path, str],
        max_dist_eye_deg : float = 5.5,
        geoclaw_kwargs : Optional[dict] = None,
        pool : Any = None,
    ):
        """Generate a TC surge hazard from a single xarray track dataset

        Parameters
        ----------
        track : xr.Dataset
            A single tropical cyclone track.
        centroids : Centroids
            Centroids instance.
        coastal_idx : ndarray
            Indices of centroids close to coast.
        topo_path : Path or str
            Path to raster file containing gridded elevation data.
        max_dist_eye_deg : float, optional
            Maximum distance from a TC track node in degrees for a centroid to be considered
            as potentially affected. Default: 5.5
        geoclaw_kwargs : dict, optional
            Keyword arguments to pass to the GeoClaw runner. See ``from_tc_tracks`` for a list
            of supported arguments. Default: None
        pool : an object with `map` functionality, optional
            If given, landfall events are processed in parallel. Experimental feature for use with
            MPI. To control the use of OpenMP, set the environment variable `OMP_NUM_THREADS` to
            the number of cores and set the compiler flag `export FFLAGS='-O2 -fopenmp'`, following
            the [GeoClaw docs](https://www.clawpack.org/openmp.html).

        Returns
        -------
        TCSurgeGeoClaw
        """
        coastal_centroids = centroids.coord[coastal_idx]
        intensity = np.zeros(centroids.coord.shape[0])
        intensity[coastal_idx], gauge_data = _geoclaw_surge_from_track(
            track,
            coastal_centroids,
            topo_path,
            max_dist_eye_deg=max_dist_eye_deg,
            geoclaw_kwargs=geoclaw_kwargs,
            pool=pool,
        )
        intensity = sparse.csr_matrix(intensity)
        date = np.array([
            dt.datetime(
                track["time"].dt.year.values[0],
                track["time"].dt.month.values[0],
                track["time"].dt.day.values[0]
            ).toordinal()
        ])
        return cls(
            centroids=centroids,
            intensity=intensity,
            gauge_data=[gauge_data],
            category=np.array([track.attrs["category"]]),
            basin=[str(track["basin"].values[0])],
            event_id=np.array([1]),
            event_name=[track.attrs["sid"]],
            date=date,
            orig=np.array([track.attrs["orig_event_flag"]]),
            frequency=np.array([1]),
            fraction=sparse.csr_matrix(intensity.shape),
            units="m",
        )

    def write_hdf5(self, *args, **kwargs) -> None:
        """Wrapper for `Hazard.write_hdf5` that omits the `gauge_data` attribute.

        The HDF5 file format does not support very well the structure of the `gauge_data`
        attribute. Use `write_gauge_data` instead to pickle the data.
        """
        gauge_data = self.gauge_data
        delattr(self, "gauge_data")
        Hazard.write_hdf5(self, *args, **kwargs)
        self.gauge_data = gauge_data

    def write_gauge_data(self, file_name : Union[pathlib.Path, str]) -> None:
        """Write this object's gauge_data attribute to a file in pickle format

        Parameters
        ----------
        file_name : Path or str
            Full path, including file name, to the output file where pickled data is written.
        """
        with open(file_name, "wb") as fp:
            pickle.dump(self.gauge_data, fp)

    def read_gauge_data(self, file_name : Union[pathlib.Path, str]) -> None:
        """Overwrite this object's gauge_data attribute by data from a file

        Parameters
        ----------
        file_name : Path or str
            Full path, including file name, to the file where pickled data is stored.
        """
        with open(file_name, "rb") as fp:
            self.gauge_data = pickle.load(fp)


def _get_coastal_centroids_idx(
    centroids : Centroids,
    max_dist_coast_km : Union[float, Tuple[float, float]],
    max_latitude : float = 90.0,
) -> np.ndarray:
    """Get indices of coastal centroids

    Parameters
    ----------
    centroids : Centroids
        Centroids instance.
    max_dist_coast_km : pair of floats (offshore, inland)
        Maximum distance to coast offshore and inland. If a single float is given instead of a
        pair, the values for inland and offshore are assumed as equal.
    max_latitude : float, optional
        Maximum latitude cutoff. Default: 90.

    Returns
    -------
    coastal_idx : ndarray of type int
        Indices into given `centroids`.
    """
    try:
        max_dist_offshore_km, max_dist_inland_km = max_dist_coast_km
    except TypeError:
        max_dist_offshore_km = max_dist_inland_km = max_dist_coast_km

    if not centroids.coord.size:
        centroids.set_meta_to_lat_lon()

    if not centroids.dist_coast.size or np.all(centroids.dist_coast >= 0):
        centroids.set_dist_coast(signed=True, precomputed=True)
    coastal_msk = (centroids.dist_coast <= max_dist_offshore_km * 1000)
    coastal_msk &= (centroids.dist_coast >= -max_dist_inland_km * 1000)
    coastal_msk &= (np.abs(centroids.lat) <= max_latitude)
    return coastal_msk.nonzero()[0]


def _geoclaw_surge_from_track(
    track : xr.Dataset,
    centroids : np.ndarray,
    topo_path : Union[pathlib.Path, str],
    max_dist_eye_deg : float = 5.5,
    geoclaw_kwargs : Optional[dict] = None,
    pool : Any = None,
) -> Tuple[np.ndarray, List[dict]]:
    """Compute TC surge height on centroids from a single track dataset

    Parameters
    ----------
    track : xr.Dataset
        Single tropical cyclone track.
    centroids : 2d ndarray
        Points for which to record the maximum height of inundation. Each row is a lat-lon point.
    topo_path : Path or str
        Path to raster file containing gridded elevation data.
    max_dist_eye_deg : float, optional
        Maximum distance from a TC track node in degrees for a centroid to be considered
        as potentially affected. Default: 5.5
    geoclaw_kwargs : dict, optional
        Keyword arguments to pass to the GeoClaw runner. See ``TCSurgeGeoClaw.from_tc_tracks`` for
        a list of supported arguments. Default: None
    pool : an object with `map` functionality, optional
        If given, landfall events are processed in parallel. Experimental feature for use with
        MPI. To control the use of OpenMP, set the environment variable `OMP_NUM_THREADS` to
        the number of cores and set the compiler flag `export FFLAGS='-O2 -fopenmp'`, following
        the [GeoClaw docs](https://www.clawpack.org/openmp.html).

    Returns
    -------
    intensity : ndarray
        Surge height in meters.
    gauge_data : list of dicts
        For each gauge, a dict containing the `location` of the gauge, and (for each surge event)
        `base_sea_level`, `topo_height`, `time`, `height_above_geoid`, `height_above_ground`,
        and `amr_level` information.
    """
    geoclaw_kwargs = {} if geoclaw_kwargs is None else geoclaw_kwargs

    gauges = geoclaw_kwargs.get("gauges")
    gauges = [] if gauges is None else gauges

    # initialize gauge data
    gauge_data = [
        {
            'location': g,
            'base_sea_level': [],
            'topo_height': [],
            'time': [],
            'height_above_ground': [],
            'height_above_geoid': [],
            'amr_level': [],
        } for g in gauges
    ]

    # initialize intensity
    intensity = np.zeros(centroids.shape[0])

    # normalize longitudes of centroids and track
    track_bounds = u_coord.latlon_bounds(track["lat"].values, track["lon"].values)
    mid_lon = 0.5 * (track_bounds[0] + track_bounds[2])
    track['lon'].values[:] = u_coord.lon_normalize(track["lon"].values, center=mid_lon)
    centroids[:, 1] = u_coord.lon_normalize(centroids[:, 1], center=mid_lon)

    # restrict to centroids in rectangular bounding box around track
    track_centr_idx = (
        (track_bounds[1] - max_dist_eye_deg <= centroids[:, 0])
        & (track_bounds[3] + max_dist_eye_deg >= centroids[:, 0])
        & (track_bounds[0] - max_dist_eye_deg <= centroids[:, 1])
        & (track_bounds[2] + max_dist_eye_deg >= centroids[:, 1])
    ).nonzero()[0]

    # exclude centroids at too low/high topographic altitude
    with rasterio.Env(VRT_SHARED_SOURCE=0):
        # without this env-setting, reading might crash in a multi-threaded environment:
        # https://gdal.org/drivers/raster/vrt.html#multi-threading-issues
        centroids_height = u_coord.read_raster_sample(
            topo_path,
            centroids[track_centr_idx, 0],
            centroids[track_centr_idx, 1],
            intermediate_res=0.008,
        )
    track_centr_idx = track_centr_idx[(centroids_height > -10) & (centroids_height < 10)]
    track_centr = centroids[track_centr_idx]

    if track_centr.shape[0] == 0:
        LOGGER.info("No centroids within reach of this storm track.")
        return intensity, gauge_data

    # make sure that radius information is available
    if 'radius_oci' not in track.coords:
        track['radius_oci'] = xr.zeros_like(track['radius_max_wind'])
    track['radius_max_wind'][:] = estimate_rmw(
        track['radius_max_wind'].values,
        track['central_pressure'].values,
    )
    track['radius_oci'][:] = estimate_roci(
        track['radius_oci'].values,
        track['central_pressure'].values,
    )
    track['radius_oci'][:] = np.fmax(track['radius_max_wind'].values, track['radius_oci'].values)

    # create work directory
    resume_file = geoclaw_kwargs.pop("resume_file", None)
    resume_i = geoclaw_kwargs.pop("resume_i", 0)
    if resume_file is not None:
        if not resume_file.exists():
            resume_file.write_text("")
        resume_dirs = resume_file.read_text().strip()
        resume_dirs = [] if resume_dirs == "" else resume_dirs.split("\n")
        if resume_i >= len(resume_dirs):
            work_dir = _get_unused_work_dir(suffix=f"-{track.attrs['sid']}")
            resume_dirs.append(str(work_dir))
            resume_file.write_text("\n".join(resume_dirs))
        else:
            work_dir = pathlib.Path(resume_dirs[resume_i])
            work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = _get_unused_work_dir(suffix=f"-{track.attrs['sid']}")

    # get landfall events
    LOGGER.info("Determine georegions and temporal periods of landfall events ...")
    events = TCSurgeEvents(track, track_centr)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The 'geom_factory' function is deprecated",
            module="cartopy",
            category=DeprecationWarning,
        )
        events.plot_areas(path=work_dir.joinpath("event_areas.pdf"))

    if len(events) == 0:
        LOGGER.info("This storm doesn't affect any coastal areas.")
    else:
        LOGGER.info("Starting %d runs of GeoClaw ...", len(events))
        runners = [
            GeoClawRunner(
                work_dir,
                track.sel(time=event['time_mask_buffered']),
                event['period'][0],
                event,
                track_centr[event['centroid_mask']],
                topo_path,
                **(
                    # if specified, only recompile the first event, not every single one
                    geoclaw_kwargs if i_event == 0 else {**geoclaw_kwargs, "recompile": False}
                ),
            )
            for i_event, event in enumerate(events)
        ]

        if pool is not None:
            pool.map(GeoClawRunner.run, runners)
        else:
            [runner.run() for runner in runners]

        surge_h = []
        for event, runner in zip(events, runners):
            event_surge_h = np.zeros(track_centr.shape[0])
            event_surge_h[event['centroid_mask']] = runner.surge_h
            surge_h.append(event_surge_h)
            for igauge, new_gauge_data in enumerate(runner.gauge_data):
                if len(new_gauge_data['time']) > 0:
                    for var in [
                        'base_sea_level', 'topo_height', 'time', 'height_above_ground',
                        'height_above_geoid', 'amr_level',
                    ]:
                        gauge_data[igauge][var].append(new_gauge_data[var])

        # write results to intensity array
        intensity[track_centr_idx] = np.stack(surge_h, axis=0).max(axis=0)

    return intensity, gauge_data


def _get_unused_work_dir(suffix: str = ""):
    """Create an empty and non-occupied work directory for the GeoClaw run files

    The directory name will consist of a time stamp and a random integer between 0 and 99.

    Parameters
    ----------
    suffix : str, optional
        Append this string to the directory name, e.g. to make the name more human-readable.
        Default: ""

    Returns
    -------
    Path
    """
    GEOCLAW_WORK_DIR.mkdir(parents=True, exist_ok=True)
    work_dir_already_exists = True
    while work_dir_already_exists:
        path = GEOCLAW_WORK_DIR / (
            dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
            + f"{np.random.randint(0, 100):02d}{suffix}"
        )
        try:
            path.mkdir(parents=True)
            work_dir_already_exists = False
        except FileExistsError:
            work_dir_already_exists = True
    return path
