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

Inundation from TC storm surges, modeled using the library GeoClaw

This module requires a Fortran compiler (such as gfortran) to run!
"""

import __main__
import contextlib
import datetime as dt
import importlib
import inspect
import logging
import re
import pathlib
import pickle
import site
import subprocess
import sys
from typing import Optional, Tuple, List, Union, Dict, Callable, Any
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import dask
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.axes as maxes
import matplotlib.colors as mcolors
import matplotlib.cm as mcolormaps
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import rasterio
from scipy import sparse
import xarray as xr

from climada import CONFIG
from climada.hazard import Centroids, Hazard, TropCyclone, TCTracks
from climada.hazard.tc_tracks import estimate_rmw, estimate_roci
from climada.util import ureg
from climada.util.constants import ONE_LAT_KM
import climada.util.coordinates as u_coord

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'TCSurgeGeoClaw'
"""Hazard type acronym for this module."""

CLAWPACK_GIT_URL = CONFIG.hazard.tc_surge_geoclaw.resources.clawpack_git.str()
"""URL of the official Clawpack git repository."""

CLAWPACK_VERSION = CONFIG.hazard.tc_surge_geoclaw.resources.clawpack_version.str()
"""Version or git decorator (tag, branch) of Clawpack to use."""

CLAWPACK_SRC_DIR = CONFIG.hazard.tc_surge_geoclaw.clawpack_src_dir.dir()
"""Directory for Clawpack source checkouts (if it doesn't exist)"""

GEOCLAW_WORK_DIR = CONFIG.hazard.tc_surge_geoclaw.geoclaw_work_dir.dir()
"""Base directory for GeoClaw run data."""

KN_TO_MS = (1.0 * ureg.knot).to(ureg.meter / ureg.second).magnitude
NM_TO_KM = (1.0 * ureg.nautical_mile).to(ureg.kilometer).magnitude
MBAR_TO_PA = (1.0 * ureg.mbar).to(ureg.pascal).magnitude
DEG_TO_NM = 60
"""Unit conversion factors."""


class TCSurgeGeoClaw(Hazard):
    """TC storm surge heights in meters (m), modeled using GeoClaw.

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
    def __init__(
        self,
        category: Optional[np.ndarray] = None,
        basin: Optional[List[str]] = None,
        gauge_data: Optional[List[List[Dict]]] = None,
        **kwargs,
    ) -> None:
        """Initialize values.

        Parameters
        ----------
        category : ndarray of int, optional
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
        topo_path : Union[pathlib.Path, str],
        centroids : Optional[Centroids] = None,
        gauges : Optional[List] = None,
        topo_res_as : float = 30.0,
        node_max_dist_deg : float = 5.5,
        inland_max_dist_km : float = 50.0,
        offshore_max_dist_km : float = 10.0,
        max_latitude : float = 61.0,
        sea_level : float = 0.0,
        resume : Optional[Union[pathlib.Path, str]] = None,
        pool : Any = None,
    ):
        """Generate a TC surge hazard instance from a TCTracks object

        Parameters
        ----------
        tracks : TCTracks
            Tracks of tropical cyclone events.
        topo_path : Path or str
            Path to raster file containing gridded elevation data.
        centroids : Centroids, optional
            Centroids where to measure maximum surge heights. By default, a centroids grid at the
            resolution `topo_res_as` is generated in a bounding box around the given tracks using
            the method `TCTracks.generate_centroids`.
        gauges : list of pairs (lat, lon), optional
            The locations of tide gauges where to measure temporal changes in sea level height.
            This is used mostly for validation purposes. The result is stored in the `gauge_data`
            attribute.
        topo_res_as : float, optional
            The resolution at which to extract topography data in arc-seconds. Needs to be between
            3 and 90 (appx. between 90 and 3000 meters). Default: 30
        node_max_dist_deg : float, optional
            Maximum distance from a TC track node in degrees for a centroid to be considered
            as potentially affected. Default: 5.5
        inland_max_dist_km : float, optional
            Maximum inland distance of the centroids in kilometers. Default: 50
        offshore_max_dist_km : float, optional
            Maximum offshore distance of the centroids in kilometers. Default: 10
        max_latitude : float, optional
            Maximum latitude of potentially affected centroids. Default: 61
        sea_level : float or function, optional
            The sea level (above geoid) of the ocean at rest, used as a starting level for the
            surge simulation. Instead of a constant scalar value, a function can be specified that
            gets a `bounds` and a `period` argument and returns a scalar value. In this case, the
            first argument is a tuple of floats (lon_min, lat_min, lon_max, lat_max) and the
            second argument is a pair of np.datetime64 (start, end). For example, see the helper
            function `sea_level_from_nc` that reads the value from a NetCDF file. Default: 0
        resume : Path or str, optional
            If given, use this file to remember the location of the run directory and resume
            operation later from this directory if it already exists. Default: None
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
        haz : TCSurgeGeoClaw object
        """
        if tracks.size == 0:
            raise ValueError("The given TCTracks object does not contain any tracks.")
        _setup_clawpack()

        if centroids is None:
            centroids = tracks.generate_centroids(
                res_deg=topo_res_as / (60 * 60), buffer_deg=node_max_dist_deg,
            )

        max_dist_coast_km = (offshore_max_dist_km, inland_max_dist_km)
        coastal_idx = _get_coastal_centroids_idx(
            centroids, max_dist_coast_km, max_latitude=max_latitude,
        )

        LOGGER.info('Computing TC surge of %s tracks on %s centroids.',
                    str(tracks.size), str(coastal_idx.size))
        haz = cls.concat([
            cls.from_xr_track(
                t, centroids, coastal_idx, topo_path, topo_res_as=topo_res_as,
                node_max_dist_deg=node_max_dist_deg, gauges=gauges, sea_level=sea_level, pool=pool,
                resume=None if resume is None else (pathlib.Path(resume), resume_i),
            )
            for resume_i, t in enumerate(tracks.data)
        ])
        with _filter_xr_warnings():
            TropCyclone.frequency_from_tracks(haz, tracks.data)
        return haz

    @classmethod
    def from_xr_track(
        cls,
        track : xr.Dataset,
        centroids : Centroids,
        coastal_idx : np.ndarray,
        topo_path : Union[pathlib.Path, str],
        topo_res_as : float = 30.0,
        node_max_dist_deg : float = 5.5,
        gauges : Optional[List] = None,
        sea_level : float = 0.0,
        resume : Optional[Tuple[pathlib.Path, int]] = None,
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
        topo_res_as : float, optional
            The resolution at which to extract topography data in arc-seconds. Needs to be between
            3 and 90 (appx. between 90 and 3000 meters). Default: 30
        node_max_dist_deg : float, optional
            Maximum distance from a TC track node in degrees for a centroid to be considered
            as potentially affected. Default: 5.5
        gauges : list of pairs (lat, lon), optional
            The locations of tide gauges where to measure temporal changes in sea level height.
            This is used mostly for validation purposes. The result is stored in the `gauge_data`
            attribute.
        sea_level : float or function, optional
            The sea level (above geoid) of the ocean at rest, used as a starting level for the
            surge simulation. Instead of a constant scalar value, a function can be specified that
            gets a `bounds` and a `period` argument and returns a scalar value. In this case, the
            first argument is a tuple of floats (lon_min, lat_min, lon_max, lat_max) and the
            second argument is a pair of np.datetime64 (start, end). For example, see the helper
            function `sea_level_from_nc` that reads the value from a NetCDF file. Default: 0
        resume : tuple (Path, int), optional
            If given, use this file to remember the location of the run directory and resume
            operation later from this directory if it already exists. The integer points to the
            line number (starting from 0) in the file to consider. Default: None
        pool : an object with `map` functionality, optional
            If given, landfall events are processed in parallel. Experimental feature for use with
            MPI. To control the use of OpenMP, set the environment variable `OMP_NUM_THREADS` to
            the number of cores and set the compiler flag `export FFLAGS='-O2 -fopenmp'`, following
            the [GeoClaw docs](https://www.clawpack.org/openmp.html).

        Returns
        -------
        haz : TCSurgeGeoClaw object
        """
        coastal_centroids = centroids.coord[coastal_idx]
        intensity = np.zeros(centroids.coord.shape[0])
        intensity[coastal_idx], gauge_data = _geoclaw_surge_from_track(
            track, coastal_centroids, topo_path, gauges=gauges, sea_level=sea_level, pool=pool,
            topo_res_as=topo_res_as, node_max_dist_deg=node_max_dist_deg, resume=resume,
        )
        intensity = sparse.csr_matrix(intensity)
        with _filter_xr_warnings():
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
    max_dist_coast_km : float,
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
        offshore_max_dist_km, inland_max_dist_km = max_dist_coast_km
    except TypeError:
        offshore_max_dist_km = inland_max_dist_km = max_dist_coast_km

    if not centroids.coord.size:
        centroids.set_meta_to_lat_lon()

    if not centroids.dist_coast.size or np.all(centroids.dist_coast >= 0):
        centroids.set_dist_coast(signed=True, precomputed=True)
    coastal_msk = (centroids.dist_coast <= offshore_max_dist_km * 1000)
    coastal_msk &= (centroids.dist_coast >= -inland_max_dist_km * 1000)
    coastal_msk &= (np.abs(centroids.lat) <= max_latitude)
    return coastal_msk.nonzero()[0]


def _geoclaw_surge_from_track(
    track : xr.Dataset,
    centroids : np.ndarray,
    topo_path : Union[pathlib.Path, str],
    topo_res_as : float = 30.0,
    gauges : Optional[List] = None,
    sea_level : float = 0.0,
    resume : Optional[Tuple[pathlib.Path, int]] = None,
    pool : Any = None,
    node_max_dist_deg : float = 5.5,
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
    topo_res_as : float, optional
        The resolution at which to extract topography data in arc-seconds. Needs to be between
        3 and 90 (appx. between 90 and 3000 meters). Default: 30
    gauges : list of pairs (lat, lon), optional
        The locations of tide gauges where to measure temporal changes in sea level height.
        This is used mostly for validation purposes.
    sea_level : float or function, optional
        The sea level (above geoid) of the ocean at rest, used as a starting level for the
        surge simulation. Instead of a constant scalar value, a function can be specified that
        gets a `bounds` and a `period` argument and returns a scalar value. In this case, the
        first argument is a tuple of floats (lon_min, lat_min, lon_max, lat_max) and the
        second argument is a pair of np.datetime64 (start, end). For example, see the helper
        function `sea_level_from_nc` that reads the value from a NetCDF file. Default: 0
    resume : tuple (Path, int), optional
        If given, use this file to remember the location of the run directory and resume
        operation later from this directory if it already exists. The integer points to the
        line number (starting from 0) in the file to consider. Default: None
    pool : an object with `map` functionality, optional
        If given, landfall events are processed in parallel. Experimental feature for use with
        MPI. To control the use of OpenMP, set the environment variable `OMP_NUM_THREADS` to
        the number of cores and set the compiler flag `export FFLAGS='-O2 -fopenmp'`, following
        the [GeoClaw docs](https://www.clawpack.org/openmp.html).
    node_max_dist_deg : float, optional
        Maximum distance from a TC track node in degrees for a centroid to be considered
        as potentially affected. Default: 5.5

    Returns
    -------
    intensity : ndarray
        Surge height in meters.
    gauge_data : list of dicts
        For each gauge, a dict containing the `location` of the gauge, and (for each surge event)
        `base_sea_level`, `topo_height`, `time`, `height_above_geoid`, `height_above_ground`,
        and `amr_level` information.
    """
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
    track_bounds = u_coord.latlon_bounds(track.lat.values, track.lon.values)
    mid_lon = 0.5 * (track_bounds[0] + track_bounds[2])
    track['lon'][:] = u_coord.lon_normalize(track.lon.values, center=mid_lon)
    centroids[:, 1] = u_coord.lon_normalize(centroids[:, 1], center=mid_lon)

    # restrict to centroids in rectangular bounding box around track
    track_bounds_pad = np.array(track_bounds)
    track_bounds_pad[:2] -= node_max_dist_deg
    track_bounds_pad[2:] += node_max_dist_deg
    track_centr_msk = (track_bounds_pad[1] <= centroids[:, 0])
    track_centr_msk &= (centroids[:, 0] <= track_bounds_pad[3])
    track_centr_msk &= (track_bounds_pad[0] <= centroids[:, 1])
    track_centr_msk &= (centroids[:, 1] <= track_bounds_pad[2])
    track_centr_idx = track_centr_msk.nonzero()[0]

    # exclude centroids at too low/high topographic altitude
    with rasterio.Env(VRT_SHARED_SOURCE=0):
        # without this env-setting, reading might crash in a multi-threaded environment:
        # https://gdal.org/drivers/raster/vrt.html#multi-threading-issues
        centroids_height = u_coord.read_raster_sample(
            topo_path, centroids[track_centr_msk, 0], centroids[track_centr_msk, 1],
            intermediate_res=0.008)
    track_centr_idx = track_centr_idx[(centroids_height > -10) & (centroids_height < 10)]
    track_centr_msk.fill(False)
    track_centr_msk[track_centr_idx] = True
    track_centr = centroids[track_centr_msk]

    if track_centr.shape[0] == 0:
        LOGGER.info("No centroids within reach of this storm track.")
        return intensity, gauge_data

    # make sure that radius information is available
    if 'radius_oci' not in track.coords:
        track['radius_oci'] = xr.zeros_like(track['radius_max_wind'])
    track['radius_max_wind'][:] = estimate_rmw(track.radius_max_wind.values,
                                               track.central_pressure.values)
    track['radius_oci'][:] = estimate_roci(track.radius_oci.values, track.central_pressure.values)
    track['radius_oci'][:] = np.fmax(track.radius_max_wind.values, track.radius_oci.values)

    # create work directory
    if resume is not None:
        resume_file, resume_i = resume
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
            GeoclawRunner(
                work_dir, track.sel(time=event['time_mask_buffered']), event['period'][0], event,
                track_centr[event['centroid_mask']], topo_path, topo_res_as=topo_res_as,
                gauges=gauges, sea_level=sea_level,
            )
            for event in events
        ]

        if pool is not None:
            pool.map(GeoclawRunner.run, runners)
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
        intensity[track_centr_msk] = np.stack(surge_h, axis=0).max(axis=0)

    return intensity, gauge_data


def _get_unused_work_dir(suffix=""):
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


class GeoclawRunner():
    """"Wrapper for work directory setup and running of GeoClaw simulations.

    Attributes
    ----------
    surge_h : ndarray
        Maximum height of inundation recorded at given centroids.
    gauge_data : list of dicts
        For each gauge, a dict containing `location`, `base_sea_level`, `topo_height`, `time`,
        `height_above_geoid`, `height_above_ground`, and `amr_level` information.
    """
    def __init__(
        self,
        base_dir : str,
        track : xr.Dataset,
        time_offset : np.datetime64,
        areas : Dict,
        centroids : np.ndarray,
        topo_path : Union[pathlib.Path, str],
        topo_res_as : float = 30.0,
        gauges : Optional[List] = None,
        sea_level : float = 0.0,
    ) -> None:
        """Initialize GeoClaw working directory with ClawPack rundata

        Parameters
        ----------
        base_dir : str
            Location where to create the working directory.
        track : xr.Dataset
            Single tropical cyclone track.
        time_offset : np.datetime64
            Usually, time of landfall
        areas : dict
            Landfall event (single iterator output from TCSurgeEvents).
        centroids : ndarray
            Points for which to record the maximum height of inundation.
            Each row is a lat-lon point.
        topo_path : Path or str
            Path to raster file containing gridded elevation data.
        topo_res_as : float, optional
            The resolution at which to extract topography data in arc-seconds. Needs to be between
            3 and 90 (appx. between 90 and 3000 meters). Default: 30
        gauges : list of pairs (lat, lon), optional
            The locations of tide gauges where to measure temporal changes in sea level height.
            This is used mostly for validation purposes.
        sea_level : float or function, optional
            The sea level (above geoid) of the ocean at rest, used as a starting level for the
            surge simulation. Instead of a constant scalar value, a function can be specified that
            gets a `bounds` and a `period` argument and returns a scalar value. In this case, the
            first argument is a tuple of floats (lon_min, lat_min, lon_max, lat_max) and the
            second argument is a pair of np.datetime64 (start, end). For example, see the helper
            function `sea_level_from_nc` that reads the value from a NetCDF file. Default: 0
        """
        gauges = [] if gauges is None else gauges

        if topo_res_as < 3 or topo_res_as > 90:
            raise ValueError("Specify a topo resolution between 3 and 90 arc-seconds!")
        self.topo_resolution_as = [360, 120, topo_res_as]

        LOGGER.info("Prepare GeoClaw to determine surge on %d centroids", centroids.shape[0])
        self.track = track
        self.areas = areas
        self.centroids = centroids
        self.time_offset = time_offset
        self.time_offset_str = _dt64_to_pydt(self.time_offset).strftime("%Y-%m-%d-%H")
        self.topo_path = topo_path
        self.gauge_data = [
            {
                'location': g,
                'base_sea_level': 0,
                'topo_height': -32768.0,
                'time': [],
                'height_above_ground': [],
                'height_above_geoid': [],
                'amr_level': [],
                'in_domain': True,
            } for g in gauges
        ]
        self.sea_level_fun = sea_level
        if np.isscalar(sea_level):
            self.sea_level_fun = lambda bounds, period: sea_level
        self.surge_h = np.zeros(centroids.shape[0])

        # compute time horizon
        self.time_horizon = tuple([int((t - self.time_offset)  / np.timedelta64(1, 's'))
                                   for t in self.track["time"][[0, -1]]])

        # create work directory
        self.work_dir = base_dir.joinpath(self.time_offset_str)
        if self.work_dir.exists():
            LOGGER.info("Resuming in GeoClaw working directory: %s", self.work_dir)
        else:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Init GeoClaw working directory: %s", self.work_dir)

        # write Makefile
        path = self.work_dir.joinpath("Makefile")
        if not path.exists():
            with path.open("w") as file_p:
                file_p.write(f"""\
CLAW = {_clawpack_info()[0]}
CLAW_PKG = geoclaw
EXE = xgeoclaw
include $(CLAW)/geoclaw/src/2d/shallow/Makefile.geoclaw
SOURCES = $(CLAW)/riemann/src/rpn2_geoclaw.f \\
          $(CLAW)/riemann/src/rpt2_geoclaw.f \\
          $(CLAW)/riemann/src/geoclaw_riemann_utils.f
include $(CLAW)/clawutil/src/Makefile.common
""")
        path = self.work_dir.joinpath("setrun.py")
        if not path.exists():
            with path.open("w") as file_p:
                file_p.write("")

        self.write_rundata()


    def run(self) -> None:
        """Run GeoClaw script and set `surge_h` attribute."""
        self.stdout = ""
        self.stdout_printed = False
        if self.work_dir.joinpath("gc_terminated").exists():
            LOGGER.info("Skip running GeoClaw since it terminated previously ...")
            self.stdout = self.work_dir.joinpath("stdout.log").read_text()
        else:
            self._run_subprocess()
        LOGGER.info("Reading GeoClaw output ...")
        try:
            self.read_fgmax_data()
            self.read_gauge_data()
        except FileNotFoundError:
            self.print_stdout()
            LOGGER.warning("Reading GeoClaw output failed (see output above).")


    def _run_subprocess(self) -> None:
        LOGGER.info("Running GeoClaw in %s ...", self.work_dir)
        time_span = self.time_horizon[1] - self.time_horizon[0]
        perc = -100
        last_perc = -100
        stopped = False
        with subprocess.Popen(["make", ".output"],
                              cwd=self.work_dir,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as proc:
            for line in proc.stdout:
                line = line.decode()
                self.stdout += line
                with self.work_dir.joinpath("stdout.log").open("a") as fp:
                    fp.write(line)
                line = line.rstrip()
                error_strings = [
                    "ABORTING CALCULATION",
                    "Stopping calculation",
                    "  free list full with ",
                ]
                if any(err in line for err in error_strings):
                    stopped = True
                re_m = re.match(r".*t = ([-ED0-9\.\+]+)$", line)
                if re_m is not None:
                    time = float(re_m.group(1).replace("D", "E"))
                    perc = 100 * (time - self.time_horizon[0]) / time_span
                    if perc - last_perc >= 10:
                        # for parallelized output, print the time offset each time
                        LOGGER.info("%s: %d%%", self.time_offset_str, perc)
                        last_perc = perc
        self.work_dir.joinpath("gc_terminated").write_text("True")
        if perc < 99.9:
            # sometimes, GeoClaw fails without a specific error output
            stopped = True
        elif int(last_perc) != 100:
            LOGGER.info("%s: 100%%", self.time_offset_str)
        if proc.returncode != 0 or stopped:
            self.print_stdout()
            raise RuntimeError("GeoClaw run failed (see output above).")


    def print_stdout(self) -> None:
        """"Print standard (and error) output of GeoClaw run."""
        if not self.stdout_printed:
            LOGGER.info("Output of 'make .output' in GeoClaw work directory:")
            print(self.stdout)
            # make sure to print at most once
            self.stdout_printed = True


    def read_fgmax_data(self) -> None:
        """Read fgmax output data from GeoClaw working directory."""
        # pylint: disable=import-outside-toplevel
        from clawpack.geoclaw import fgmax_tools
        outdir = self.work_dir.joinpath("_output")
        fg_path = outdir.joinpath("fgmax0001.txt")

        if not fg_path.exists():
            raise FileNotFoundError("GeoClaw quit without creating fgmax data!")

        fgmax_grid = fgmax_tools.FGmaxGrid()
        fg_fname = self.work_dir.joinpath("fgmax_grids.data")
        with contextlib.redirect_stdout(None):
            fgmax_grid.read_fgmax_grids_data(1, fg_fname)
            fgmax_grid.read_output(outdir=outdir)
        assert fgmax_grid.point_style == 0
        self.surge_h[:] = fgmax_grid.h
        self.surge_h[fgmax_grid.arrival_time.mask] = 0


    def read_gauge_data(self) -> None:
        """Read gauge output data from GeoClaw working directory."""
        # pylint: disable=import-outside-toplevel
        from clawpack.pyclaw.gauges import GaugeSolution
        outdir = self.work_dir.joinpath("_output")
        for i_gauge, gauge in enumerate(self.gauge_data):
            if not gauge['in_domain']:
                continue
            gauge['base_sea_level'] = self.rundata.geo_data.sea_level
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                # suppress warnings about empty gauge files (which is not a problem for us)
                g = GaugeSolution(gauge_id=i_gauge + 1, path=outdir)
            if g.t is None:
                continue
            gauge['time'] = self.time_offset + g.t * np.timedelta64(1, 's')
            gauge['topo_height'] = g.q[1, -1] - g.q[0, -1]
            gauge['height_above_ground'] = g.q[0, :]
            gauge['height_above_geoid'] = g.q[1, :]
            gauge["amr_level"] = g.level


    def write_rundata(self) -> None:
        """Create rundata config files in work directory or read if already existent."""
        # pylint: disable=import-outside-toplevel
        if not self._read_rundata():
            self._set_rundata_claw()
            self._set_rundata_amr()
            self._set_rundata_geo()
            self._set_rundata_fgmax()
            self._set_rundata_storm()
            self._set_rundata_gauges()
            with contextlib.redirect_stdout(None), _backup_loggers():
                self.rundata.write(out_dir=self.work_dir)


    def _read_rundata(self) -> bool:
        """Read rundata object from files, return whether it was succesful

        Returns
        -------
        bool
        """
        import clawpack.amrclaw.data
        import clawpack.geoclaw.data
        self._clear_rundata()
        for dataobject in self.rundata.data_list:
            if isinstance(dataobject, clawpack.geoclaw.data.FixedGridData):
                # ignore since it's deprecated, hence unused
                continue
            fname = inspect.signature(dataobject.write).parameters["out_file"].default
            path = self.work_dir / fname
            if not path.exists():
                self._clear_rundata()
                return False
            is_gauge_data = isinstance(dataobject, clawpack.amrclaw.data.GaugeData)
            read_args = [] if is_gauge_data else [path]
            read_kwargs = dict(data_path=self.work_dir) if is_gauge_data else {}
            with contextlib.redirect_stdout(None):
                dataobject.read(*read_args, **read_kwargs)
        # resume from checkpoint if it exists and the previous run didn't finish
        chk_files = list(self.work_dir.glob("_output/fort.chk*"))
        if len(chk_files) > 1 and not self.work_dir.joinpath("gc_terminated").exists():
            idx_by_mtimes = np.argsort([p.stat().st_mtime for p in chk_files])
            # the latest might be corrupt after kill during I/O; use the previous
            self.rundata.clawdata.restart_file = chk_files[idx_by_mtimes[-2]].name
            self.rundata.clawdata.restart = True
            self.rundata.clawdata.write(out_file=self.work_dir / "claw.data")
        return True


    def _clear_rundata(self) -> None:
        """Reset the rundata object to its initial, empty state"""
        import clawpack.clawutil.data
        self.rundata = clawpack.clawutil.data.ClawRunData(pkg="geoclaw", num_dim=2)


    def _set_rundata_claw(self) -> None:
        """Set the rundata parameters in the `clawdata` category."""
        clawdata = self.rundata.clawdata
        clawdata.verbosity = 1
        clawdata.checkpt_style = -3
        clawdata.checkpt_interval = 25
        clawdata.num_output_times = 0
        clawdata.output_t0 = False
        clawdata.lower = self.areas['wind_area'][:2]
        clawdata.upper = self.areas['wind_area'][2:]
        clawdata.num_cells = [int(np.ceil((clawdata.upper[0] - clawdata.lower[0]) * 4)),
                              int(np.ceil((clawdata.upper[1] - clawdata.lower[1]) * 4))]
        clawdata.num_eqn = 3
        clawdata.num_aux = 3 + 1 + 3
        clawdata.capa_index = 2
        clawdata.t0, clawdata.tfinal = self.time_horizon
        clawdata.dt_initial = 0.8 / max(clawdata.num_cells)
        clawdata.cfl_desired = 0.75
        clawdata.num_waves = 3
        clawdata.limiter = ['mc', 'mc', 'mc']
        clawdata.use_fwaves = True
        clawdata.source_split = 'godunov'
        clawdata.bc_lower = ['extrap', 'extrap']
        clawdata.bc_upper = ['extrap', 'extrap']


    def _set_rundata_amr(self) -> None:
        """Set AMR-related rundata attributes."""
        clawdata = self.rundata.clawdata
        amrdata = self.rundata.amrdata
        refinedata = self.rundata.refinement_data
        amrdata.refinement_ratios_x = self.compute_refinement_ratios()
        amrdata.refinement_ratios_y = amrdata.refinement_ratios_x
        amrdata.refinement_ratios_t = amrdata.refinement_ratios_x
        amrdata.amr_levels_max = len(amrdata.refinement_ratios_x) + 1
        amrdata.aux_type = ['center', 'capacity', 'yleft', 'center', 'center', 'center', 'center']
        amrdata.regrid_interval = 3
        amrdata.regrid_buffer_width = 2
        amrdata.verbosity_regrid = 0
        regions = self.rundata.regiondata.regions
        t_1, t_2 = clawdata.t0, clawdata.tfinal
        maxlevel = amrdata.amr_levels_max
        x_1, y_1, x_2, y_2 = self.areas['wind_area']
        regions.append([1, 4, t_1, t_2, x_1, x_2, y_1, y_2])
        x_1, y_1, x_2, y_2 = self.areas['landfall_area']
        regions.append([max(1, maxlevel - 3), maxlevel, t_1, t_2, x_1, x_2, y_1, y_2])
        for area in self.areas['surge_areas']:
            x_1, y_1, x_2, y_2 = area
            regions.append([maxlevel - 1, maxlevel, t_1, t_2, x_1, x_2, y_1, y_2])
        refinedata.speed_tolerance = list(np.arange(1.0, maxlevel - 2))
        refinedata.variable_dt_refinement_ratios = True
        refinedata.wave_tolerance = 1.0


    def compute_refinement_ratios(self) -> None:
        # select the refinement ratios so that:
        # * the last but one resolution is less than self.topo_resolution_as[-1]
        # * the list of ratios is non-decreasing
        # * not more than 6 ratios
        # * no single ratio larger than 8
        clawdata = self.rundata.clawdata
        base_res = (clawdata.upper[0] - clawdata.lower[0]) / clawdata.num_cells[0]
        total_fact = base_res / (self.topo_resolution_as[-1] / 3600)
        n_ratios = min(5, int(np.round(np.log2(total_fact))))
        if n_ratios < 2:
            ratios = [2]
        else:
            ratios = [2] * (n_ratios - 2)
            target = total_fact / np.prod(ratios)
            ratio1 = np.arange(
                max(2, np.ceil(target / 8)),
                max(2, min(np.sqrt(target), 8)) + 1)
            ratio2 = np.fmax(ratio1, np.ceil(target / ratio1))
            i_ratio = np.argmin(ratio1 * ratio2)
            ratios += [int(ratio1[i_ratio]), int(ratio2[i_ratio])]
        ratios += [min(8, ratios[-1] + 1)]
        LOGGER.info("GeoClaw resolution in arc-seconds: %s",
                    str([f"{3600 * base_res / r:.2f}" for r in np.cumprod([1] + ratios)]))
        return ratios


    def _set_rundata_geo(self) -> None:
        """Set geo-related rundata attributes."""
        clawdata = self.rundata.clawdata
        frictiondata = self.rundata.friction_data
        geodata = self.rundata.geo_data
        topodata = self.rundata.topo_data

        # lat-lon coordinate system
        geodata.coordinate_system = 2

        # different friction on land and at sea
        geodata.friction_forcing = True
        frictiondata.variable_friction = True
        frictiondata.friction_regions.append([
            clawdata.lower, clawdata.upper, [np.infty, 0.0, -np.infty], [0.050, 0.025],
        ])
        geodata.dry_tolerance = 1.e-2

        # get sea level information for affected areas and time period
        tr_period = (self.track["time"].values[0], self.track["time"].values[-1])
        geodata.sea_level = np.mean([
            self.sea_level_fun(area, tr_period)
            for area in self.areas['surge_areas']
        ])

        # load elevation data, resolution depending on area of refinement
        topodata.topofiles = []
        areas = [
            self.areas['wind_area'],
            self.areas['landfall_area']
        ] + self.areas['surge_areas']
        resolutions = self.topo_resolution_as[:2]
        resolutions += [self.topo_resolution_as[2]] * len(self.areas['surge_areas'])
        dems_for_plot = []
        for res_as, bounds in zip(resolutions, areas):
            bounds, topo = _load_topography(self.topo_path, bounds, res_as)
            if 0 in topo.Z.shape:
                LOGGER.warning("Area is ignored because it is too small.")
                continue
            tt3_fname = 'topo_{}s_{}.tt3'.format(res_as, _bounds_to_str(bounds))
            tt3_fname = self.work_dir.joinpath(tt3_fname)
            topo.write(tt3_fname)
            topodata.topofiles.append([3, tt3_fname])
            dems_for_plot.append((bounds, topo.Z))
        _plot_dems(
            dems_for_plot,
            track=self.track,
            # for debugging purposes, you might want to plot the centroids as scatter:
            # centroids=self.centroids,
            path=self.work_dir.joinpath("dems.pdf"),
        )


    def _set_rundata_fgmax(self) -> None:
        """Set monitoring-related rundata attributes."""
        # pylint: disable=import-outside-toplevel
        from clawpack.geoclaw import fgmax_tools

        # monitor max height values on centroids
        self.rundata.fgmax_data.num_fgmax_val = 1
        fgmax_grid = fgmax_tools.FGmaxGrid()
        fgmax_grid.point_style = 0
        fgmax_grid.tstart_max = self.rundata.clawdata.t0
        fgmax_grid.tend_max = self.rundata.clawdata.tfinal
        fgmax_grid.dt_check = 0
        fgmax_grid.min_level_check = self.rundata.amrdata.amr_levels_max - 1
        fgmax_grid.arrival_tol = 1.e-2
        fgmax_grid.npts = self.centroids.shape[0]
        fgmax_grid.X = self.centroids[:, 1]
        fgmax_grid.Y = self.centroids[:, 0]
        self.rundata.fgmax_data.fgmax_grids.append(fgmax_grid)


    def _set_rundata_storm(self) -> None:
        """Set storm-related rundata attributes."""
        surge_data = self.rundata.surge_data
        surge_data.wind_forcing = True
        surge_data.drag_law = 1
        surge_data.pressure_forcing = True
        surge_data.storm_specification_type = 'holland80'
        surge_data.storm_file = str(self.work_dir.joinpath("track.storm"))
        gc_storm = _climada_xarray_to_geoclaw_storm(
            self.track, offset=_dt64_to_pydt(self.time_offset),
        )
        gc_storm.write(surge_data.storm_file, file_format='geoclaw')


    def _set_rundata_gauges(self) -> None:
        """Set gauge-related rundata attributes."""
        clawdata = self.rundata.clawdata
        for i_gauge, gauge in enumerate(self.gauge_data):
            lat, lon = gauge['location']
            if (clawdata.lower[0] > lon or clawdata.lower[1] > lat
                or clawdata.upper[0] < lon or clawdata.upper[1] < lat):
                # skip gauges outside of model domain
                gauge['in_domain'] = False
                continue
            self.rundata.gaugedata.gauges.append(
                [i_gauge + 1, lon, lat, clawdata.t0, clawdata.tfinal]
            )
        # q[0]: height above topography (above ground, where ground might be sea floor)
        self.rundata.gaugedata.q_out_fields = [0]


def _plot_dems(
    dems : List,
    track : Optional[xr.Dataset] = None,
    path : Optional[Union[pathlib.Path, str]] = None,
    centroids : Optional[Centroids] = None,
) -> None:
    """Plot given DEMs as rasters to one worldmap

    Parameters
    ----------
    dems : list of pairs
        pairs (bounds, heights)
    path : Path or str or None
        If given, save plot in this location. Default: None
    track : xr.Dataset
        If given, overlay the tropical cyclone track. Default: None
    centroids : ndarray
        If given, overlay as scatter points. Default: None
    """
    # adjust properties of the colorbar (ignored by cartopy axes)
    matplotlib.rc('axes', linewidth=0.5)
    matplotlib.rc('font', size=7, family='serif')
    matplotlib.rc('xtick', top=True, direction='out')
    matplotlib.rc('xtick.major', size=2.5, width=0.5)
    matplotlib.rc('ytick', right=True, direction='out')
    matplotlib.rc('ytick.major', size=2.5, width=0.5)

    total_bounds = (
        min([bounds[0] for bounds, _ in dems]),
        min([bounds[1] for bounds, _ in dems]),
        max([bounds[2] for bounds, _ in dems]),
        max([bounds[3] for bounds, _ in dems]),
    )
    mid_lon = 0.5 * (total_bounds[0] + total_bounds[2])
    aspect_ratio = 1.124 * ((total_bounds[2] - total_bounds[0])
                            / (total_bounds[3] - total_bounds[1]))
    fig = plt.figure(
        figsize=(10, 10 / aspect_ratio) if aspect_ratio >= 1 else (aspect_ratio * 10, 10),
        dpi=100)
    proj_data = ccrs.PlateCarree()
    proj_ax = ccrs.PlateCarree(central_longitude=mid_lon)
    axes = fig.add_subplot(111, projection=proj_ax)
    axes.spines['geo'].set_linewidth(0.5)
    axes.set_extent(
        (total_bounds[0], total_bounds[2], total_bounds[1], total_bounds[3]), crs=proj_data)

    # add axes tick labels
    grid = axes.gridlines(draw_labels=True, alpha=0.2, linewidth=0)
    grid.top_labels = grid.right_labels = False
    grid.xformatter, grid.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    cmap, cnorm = _colormap_coastal_dem(axes=axes)
    for bounds, heights in dems:
        # a bug (?) in cartopy breaks imshow with a transform different from `proj_ax`, so we
        # manually shift the central longitude in the `extent` attribute
        axes.imshow(heights, origin='lower', transform=proj_ax, cmap=cmap, norm=cnorm,
                    extent=(bounds[0] - mid_lon, bounds[2] - mid_lon, bounds[1], bounds[3]))
        _plot_bounds(axes, bounds, transform=proj_data, color='k', linewidth=0.5)
    axes.coastlines(resolution='10m', linewidth=0.5)
    if track is not None:
        axes.plot(track.lon, track.lat, transform=proj_data, color='k', linewidth=0.5)
    if centroids is not None:
        axes.scatter(centroids[:, 1], centroids[:, 0], transform=proj_data, s=0.1, alpha=0.5)
    fig.subplots_adjust(left=0.02, bottom=0.03, right=0.89, top=0.99, wspace=0, hspace=0)
    if path is None or not hasattr(__main__, '__file__'):
        plt.show()
    if path is not None:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(path)
        plt.close(fig)


def _colormap_coastal_dem(
    axes : Optional[maxes.Axes] = None,
) -> Tuple[mcolors.Colormap, mcolors.Normalize]:
    """Return colormap and normalization for coastal areas of DEMs

    Parameters
    ----------
    axes : matplotlib.axes.Axes, optional
        If given, add a colorbar to the right of it. Default: None

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        A colormap ranging from dark and light blue over yellow and light and dark green to brown.
    cnorm : LinearSegmentedNormalize
        A nonlinear normalize class ranging from -8000 till 1000 that makes sure that small values
        around 0 (coastal areas) are better distinguished.
    """
    cmap_terrain = [
        (0, 0, 0),
        (3, 73, 114),
        (52, 126, 255),
        (146, 197, 222),
        (255, 251, 171),
        (165, 230, 162),
        (27, 149, 29),
        (32, 114, 11),
        (117, 84, 0),
    ]
    cmap_terrain = mcolors.LinearSegmentedColormap.from_list(
        "coastal_dem", [tuple(c / 255 for c in rgb) for rgb in cmap_terrain])
    cnorm_coastal_dem = LinearSegmentedNormalize([-8000, -1000, -10, -5, 0, 5, 10, 100, 1000])
    if axes:
        cbar_ax = inset_axes(axes, width="5%", height="100%",
                             loc='lower left', bbox_to_anchor=(1.02, 0., 0.5, 1),
                             bbox_transform=axes.transAxes, borderpad=0)
        cbar = plt.colorbar(mcolormaps.ScalarMappable(cmap=cmap_terrain), cax=cbar_ax)
        cbar.set_ticks(cnorm_coastal_dem.values)
        cbar.set_ticklabels(cnorm_coastal_dem.vthresh)
    return cmap_terrain, cnorm_coastal_dem


class LinearSegmentedNormalize(mcolors.Normalize):
    """Piecewise linear color normalization."""
    def __init__(self, vthresh : List[float]):
        """Initialize normalization

        Parameters
        ----------
        vthresh : list of floats
            Equally distributed to the interval [0,1].
        """
        self.vthresh = vthresh
        self.values = np.linspace(0, 1, len(self.vthresh))
        mcolors.Normalize.__init__(self, vmin=vthresh[0], vmax=vthresh[-1], clip=False)

    def __call__(self, value : float, clip : Any = None) -> np.ndarray:
        return np.ma.masked_array(np.interp(value, self.vthresh, self.values))


def _climada_xarray_to_geoclaw_storm(
    track : xr.Dataset,
    offset : Optional[dt.datetime] = None,
) -> Any:
    """Convert CLIMADA's xarray TC track to GeoClaw storm object

    Parameters
    ----------
    track : xr.Dataset
        Single tropical cyclone track.
    offset : datetime
        Time zero for internal use in GeoClaw.

    Returns
    -------
    gc_storm : clawpack.geoclaw.surge.storm.Storm
    """
    # pylint: disable=import-outside-toplevel
    from clawpack.geoclaw.surge.storm import Storm
    gc_storm = Storm()
    gc_storm.t = _dt64_to_pydt(track["time"].values)
    if offset is not None:
        gc_storm.time_offset = offset
    gc_storm.eye_location = np.stack([track.lon, track.lat], axis=-1)
    gc_storm.max_wind_speed = track.max_sustained_wind.values * KN_TO_MS
    gc_storm.max_wind_radius = track.radius_max_wind.values * NM_TO_KM * 1000
    gc_storm.central_pressure = track.central_pressure.values * MBAR_TO_PA
    gc_storm.storm_radius = track.radius_oci.values * NM_TO_KM * 1000
    return gc_storm


def sea_level_from_nc(
    path : Union[pathlib.Path, str],
    t_agg : str = "mean",
    t_pad : Optional[np.timedelta64] = None,
    mod_zos : float = 0.0,
) -> Callable:
    """Generate a function that reads centroid sea levels from a NetCDF file

    The function that is generated can be used as an input for the `sea_level` parameter in
    `TCSurgeGeoClaw.from_tc_tracks`.

    The grid cell closest to the area's centroid that has valid entries is identified. Then the
    specified aggregation method (e.g. "mean" or "max") is applied over the time period.

    Parameters
    ----------
    path : Path or str
        Path to NetCDF file containing gridded sea level data with time resolution.
    t_agg : str, optional
        Aggregation method to apply over the time period. Supported methods: "mean", "min", "max".
        Default: "mean"
    t_pad : np.timedelta64, optional
        Padding to add around the time period. Default: 0.
    mod_zos : float, optional
        The scalar sea level rise is added to the sea level value that is extracted from the
        specified NetCDF file. Default: 0

    Returns
    -------
    fun : function (tuple, tuple) -> float
        The first argument is a tuple of floats (lon_min, lat_min, lon_max, lat_max), the second
        argument is a pair of np.datetime64 (start, end). The function returns the mean sea level
        in the specified region and time period.
    """
    t_agg = t_agg.lower()
    if t_agg not in ["mean", "min", "max"]:
        raise ValueError(f"Aggregation method not supported: {t_agg}")
    _sea_level_nc_info(path)
    def sea_level_fun(bounds, period, path=path, t_agg=t_agg, t_pad=t_pad, mod_zos=mod_zos):
        t_pad = np.timedelta64(0, "D") if t_pad is None or t_pad == 0 else t_pad
        period = (period[0] - t_pad, period[1] + t_pad)
        centroid = (0.5 * (bounds[0] + bounds[2]), 0.5 * (bounds[1] + bounds[3]))
        with _filter_xr_warnings(), xr.open_dataset(path) as ds:
            da_zos = _nc_rename_vars(ds)["zos"]
            period = [_get_closest_date_in_index(da_zos["time"], t) for t in period]
            da_zos = da_zos.sel(time=(da_zos["time"] >= period[0]) & (da_zos["time"] <= period[1]))
            lon, lat = _get_closest_valid_cell(da_zos, *centroid)
            da_zos = da_zos.sel(lon=lon, lat=lat)
            v_agg = getattr(da_zos, t_agg)().item()
        return v_agg + mod_zos
    return sea_level_fun


def _get_closest_valid_cell(
    ds_var : xr.DataArray,
    lon : float,
    lat : float,
    threshold_deg : float = 10.0,
) -> Tuple[float, float]:
    """Extract the grid cell with valid entries that is closest to the given location

    To be considered, a grid cell is required to have valid entries for all time steps.

    Parameters
    ----------
    ds_var : xr.DataArray
        Gridded data with "time" dimension.
    lon, lat : float
        The longitudinal and latitudinal coordinates of the location.
    threshold_deg : float, optional
        Threshold (in degrees) for a grid cell to be considered. Default: 10

    Returns
    -------
    lon, lat : float
        Longitudinal and latitudinal coordinates of the centroid of the grid cell that is closest
        to the specified location and has valid entries.
    """
    # store original longitudinal coordinates because they are normalized in the process
    lon_orig = ds_var["lon"].values.copy()

    # for performance reasons, restrict search to cells that are close enough
    bounds = (lon - threshold_deg, lat - threshold_deg,
              lon + threshold_deg, lat + threshold_deg)
    ds_var = _select_bounds(ds_var, bounds)

    finite_mask = np.isfinite(ds_var).all(dim="time")
    if not np.any(finite_mask):
        return None
    coords = xr.broadcast(*[getattr(ds_var, d) for d in finite_mask.dims])
    finite_coords = [c.values[finite_mask] for c in coords]
    lats, lons = finite_coords if finite_mask.dims[0] == "lat" else finite_coords[::-1]
    dist_sq = (lats - lat)**2 + (lons - lon)**2
    idx = np.argmin(dist_sq)
    lon_close, lat_close = lons[idx], lats[idx]
    lon_diff = np.mod(lon_orig - lon_close, 360)
    lon_diff[lon_diff > 180] -= 360
    lon_close = lon_orig[np.argmin(np.abs(lon_diff))]
    return lon_close, lat_close


def _get_closest_date_in_index(
    dt_index : pd.DatetimeIndex,
    date : np.datetime64,
) -> np.datetime64:
    """Extract the entry from the given DatetimeIndex that is closest to the given date

    If the date lies exactly between two consecutive entries in the index, the earlier date is
    returned.

    Parameters
    ----------
    dt_index : pd.DatetimeIndex
        The index from which to extract the entry that is closest to `date`.
    date : np.datetime64
        The date for which to search the closest entry in `dt_index`.

    Returns
    -------
    np.datetime64
    """
    i = dt_index.searchsorted(date, side="left")
    if i == 0:
        return dt_index.values[0]
    if i == dt_index.size:
        return dt_index.values[-1]
    if date - dt_index.values[i - 1] > dt_index.values[i] - date:
        return dt_index.values[i]
    return dt_index.values[i - 1]


def area_sea_level_from_monthly_nc(
    path : Union[pathlib.Path, str],
    t_pad : Optional[np.timedelta64] = None,
    mod_zos : float = 0.0,
) -> Callable:
    """Generate a function that reads area-aggregated sea levels from a NetCDF file

    The function that is generated can be used as an input for the `sea_level` parameter in
    `TCSurgeGeoClaw.from_tc_tracks`.

    The maximum over the specified area, then the mean over all affected months is taken.
    By specifying `t_pad`, neighboring months can also be marked as affected.

    Parameters
    ----------
    path : Path or str
        Path to NetCDF file containing monthly sea level data.
    t_pad : np.timedelta64, optional
        Padding to add around the time period. Default: 7 days.
    mod_zos : float, optional
        The scalar sea level rise is added to the sea level value that is extracted from the
        specified NetCDF file. Default: 0

    Returns
    -------
    fun : function (tuple, tuple) -> float
        The first argument is a tuple of floats (lon_min, lat_min, lon_max, lat_max), the second
        argument is a pair of np.datetime64 (start, end). The function returns the mean sea level
        in the specified region and time period.
    """
    _sea_level_nc_info(path)
    def sea_level_fun(bounds, period, path=path, t_pad=t_pad, mod_zos=mod_zos):
        t_pad = np.timedelta64(7, "D") if t_pad is None else t_pad
        period = (period[0] - t_pad, period[1] + t_pad)
        times = pd.Series([0, 0], index=list(period)).resample("12H").ffill(limit=1).index
        months = np.unique(np.stack((times.year, times.month), axis=-1), axis=0)
        return _mean_max_sea_level(path, months, bounds) + mod_zos
    return sea_level_fun


def _mean_max_sea_level(
    path : Union[pathlib.Path, str],
    months : np.ndarray,
    bounds : Tuple[float, float, float, float],
) -> float:
    """Mean of maxima over affected area in affected months

    Parameters
    ----------
    path : Path or str
        Path to NetCDF file containing monthly sea level data.
    months : ndarray
        each row is a tuple (year, month)
    bounds : tuple
        (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    zos : float
        Sea level height in meters
    """
    pad_deg = 0.25
    max_pad_deg = 5
    with _filter_xr_warnings(), xr.open_dataset(path) as ds:
        ds = _nc_rename_vars(ds)
        mask_time = np.any([(ds["time"].dt.year == m[0]) & (ds["time"].dt.month == m[1])
                           for m in months], axis=0)
        if np.count_nonzero(mask_time) != months.shape[0]:
            raise IndexError("The sea level data set doesn't contain the required months: %s"
                             % ", ".join(f"{m[0]:04d}-{m[1]:02d}" for m in months))
        ds = ds.sel(time=mask_time)

        # enlarge bounds until the mean is valid or until max_pad_deg is reached
        i_pad = 0
        mean = np.nan
        bounds_padded = bounds
        while np.isnan(mean):
            if i_pad * pad_deg > max_pad_deg:
                raise IndexError(
                    f"The sea level data set doesn't intersect the specified bounds: {bounds}")
            mean = _temporal_mean_of_max_within_bounds(ds, bounds_padded)
            bounds_padded = (
                bounds_padded[0] - pad_deg, bounds_padded[1] - pad_deg,
                bounds_padded[2] + pad_deg, bounds_padded[3] + pad_deg)
            i_pad += 1
    return mean


def _temporal_mean_of_max_within_bounds(
    ds : xr.Dataset,
    bounds : Tuple[float, float, float, float],
) -> float:
    """Take the maximum over a given spatial extent, then the mean over the time dimension

    Any NaN-values in the data are ignored, unless all values within the specified bounds are NaN.
    For example, in case of a gridded sea level data set with NaN over land, the maximum will just
    be over the valid values. Only if all values within the spatial bounds are NaN, the maximum is
    assumed to be NaN. Similarly, when taking the mean over the time dimension, the NaN values are
    dropped before taking the mean.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset with temporal and spatial (lon/lat) dimensions and a "zos" data variable.
    bounds : tuple of floats
        The minimum and maximum values for each spatial dimension:
        (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    float
    """
    ds_zos = _select_bounds(ds["zos"], bounds)
    if 0 in ds_zos.shape:
        return np.nan
    values = ds_zos.values[:]
    if np.all(np.isnan(values)):
        return np.nan
    return np.nanmean(np.nanmax(values, axis=(1, 2)))


def _select_bounds_dim(
    ds : Union[xr.Dataset, xr.DataArray],
    dim : str,
    bounds : Tuple[float, float],
) -> Union[xr.Dataset, xr.DataArray]:
    """Restrict the data set's dimension to the specified bounds

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        The data set for which to restrict a dimension.
    dim : str
        The dimension within `ds` to restrict.
    bounds : tuple of floats
        The minimum and maximum value for `dim`.

    Returns
    -------
    xr.Dataset or xr.DataArray
    """
    ref_min, ref_max = bounds
    idx = ((ds[dim] <= ref_max) & (ds[dim] >= ref_min)).values.nonzero()[0]
    if idx.size < 2:
        d_min, d_max = ds[dim].values.min(), ds[dim].values.max()
        if d_min > ref_min or d_max < ref_max:
            LOGGER.warn(
                f"The dimension '{dim}' ({d_min} -- {d_max}) does not cover the range of the"
                f" reference dimension ({ref_min} -- {ref_max})."
            )
    sl_start, sl_end = (idx[0], idx[-1] + 1) if idx.size > 0 else (0, 0)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds = ds.isel(indexers={dim: slice(sl_start, sl_end)})
    return ds


def _select_bounds(
    ds : Union[xr.Dataset, xr.DataArray],
    bounds : Tuple[float, float, float, float],
) -> Union[xr.Dataset, xr.DataArray]:
    """Restrict the raster data set to the specified bounds

    In a first step, the longitudinal coordinate values are normalized to the longitudinal range
    indicated by `bounds`.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        The raster data to restrict.
    bounds : tuple of float
        The minimum and maximum values for each spatial dimension:
        (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    xr.Dataset or xr.DataArray
    """
    ds = _select_bounds_dim(ds, "lat", (bounds[1], bounds[3]))
    mid_lon = 0.5 * (bounds[0] + bounds[2])
    ds = ds.assign_coords(lon=u_coord.lon_normalize(ds["lon"].values.copy(), center=mid_lon))
    ds = ds.reindex(lon=np.unique(ds["lon"].values))
    ds = _select_bounds_dim(ds, "lon", (bounds[0], bounds[2]))
    return ds


def _sea_level_nc_info(path : Union[pathlib.Path, str]) -> None:
    """Log information about the spatiotemporal bounds of the specified NetCDF file.

    Parameters
    ----------
    path : Path or str
        Path to a NetCDF file with raster data and time dimension.
    """
    LOGGER.info("Reading sea level data from %s", path)

    with _filter_xr_warnings(), xr.open_dataset(path) as ds:
        ds = _nc_rename_vars(ds)
        ds_bounds = (ds["lon"].values.min(), ds["lat"].values.min(),
                     ds["lon"].values.max(), ds["lat"].values.max())
        ds_period = (ds["time"][0], ds["time"][-1])
        LOGGER.info("Sea level data available within bounds %s", ds_bounds)
        LOGGER.info("Sea level data available within period from %04d-%02d till %04d-%02d",
                    ds_period[0].dt.year, ds_period[0].dt.month,
                    ds_period[1].dt.year, ds_period[1].dt.month)

@contextlib.contextmanager
def _filter_xr_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="distutils Version classes are deprecated",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Index.ravel returning ndarray is deprecated; in a future version",
            module="xarray",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in cast",
            module="xarray",
            category=RuntimeWarning,
        )
        yield

def _nc_rename_vars(ds : xr.Dataset) -> xr.Dataset:
    """Rename several coordinate and data variable names to their defaults

    The default names are "lon", "lat", "time", and "zos" (for sea surface height).

    Parameters
    ----------
    ds : xr.Dataset
        Data set with longitudinal, latitudinal, and temporal dimensions, as well as a sea level
        data variable.

    Returns
    -------
    xr.Dataset
    """
    var_names = {
        'lon': ('coords', ["longitude", "lon", "x"]),
        'lat': ('coords', ["latitude", "lat", "y"]),
        'time': ('coords', ["time", "date", "datetime"]),
        'zos': ('variables', ["zos", "sla", "ssh", "adt"]),
    }
    for new_name, (var_type, all_names) in var_names.items():
        old_name = [c for c in getattr(ds, var_type) if c.lower() in all_names][0]
        if old_name != new_name:
            ds = ds.rename({old_name: new_name})
    return ds


def _load_topography(
    path : Union[pathlib.Path, str],
    bounds : Tuple[float, float, float, float],
    res_as : float,
) -> Tuple[Tuple[float, float, float, float], Any]:
    """Load topographical elevation data in specified bounds and resolution

    The bounds of the returned topodata are always larger than the requested bounds to make sure
    that the pixel centers still cover the requested region.

    Parameters
    ----------
    path : Path or str
        Path to raster file containing elevation data above reference geoid.
    bounds : tuple
        Bounds (lon_min, lat_min, lon_max, lat_max) of region of interest.
    res_as : float
        Resolution in arc-seconds.

    Returns
    -------
    bounds : tuple
        Bounds (lon_min, lat_min, lon_max, lat_max) actually covered by the returned topodata.
    topo : clawpack.geoclaw.topotools.Topography object
        The object's x, y and Z attributes contain the loaded topodata.
    """
    # pylint: disable=import-outside-toplevel
    from clawpack.geoclaw import topotools

    LOGGER.info("Load elevation data [%s, %s] from %s", res_as, bounds, path)
    res = res_as / (60 * 60)
    with rasterio.Env(VRT_SHARED_SOURCE=0):
        # without this env-setting, reading might crash in a multi-threaded environment:
        # https://gdal.org/drivers/raster/vrt.html#multi-threading-issues
        zvalues, transform = u_coord.read_raster_bounds(
            path, bounds, res=res, bands=[1], resampling="bilinear", global_origin=(-180, 90))
    zvalues = zvalues[0]
    xres, _, xmin, _, yres, ymin = transform[:6]
    xmax, ymax = xmin + zvalues.shape[1] * xres, ymin + zvalues.shape[0] * yres
    if xres < 0:
        zvalues = np.flip(zvalues, axis=1)
        xres, xmin, xmax = -xres, xmax, xmin
    if yres < 0:
        zvalues = np.flip(zvalues, axis=0)
        yres, ymin, ymax = -yres, ymax, ymin
    xmin, xmax = u_coord.lon_normalize(
        np.array([xmin, xmax]), center=0.5 * (bounds[0] + bounds[2]))
    bounds = (xmin, ymin, xmax, ymax)
    xcoords = np.arange(xmin + xres / 2, xmax, xres)
    ycoords = np.arange(ymin + yres / 2, ymax, yres)

    nan_msk = np.isnan(zvalues)
    nan_count = nan_msk.sum()
    if nan_count > 0:
        LOGGER.warning("Elevation data contains %d NaN values that are replaced with -1000!",
                       nan_count)
        zvalues[nan_msk] = -1000

    topo = topotools.Topography()
    topo.set_xyZ(xcoords, ycoords, zvalues.astype(np.float64))
    return bounds, topo


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
    keys = ['period', 'time_mask', 'time_mask_buffered', 'wind_area',
            'landfall_area', 'surge_areas', 'centroid_mask']
    maxlen_h = 48
    maxbreak_h = 12
    period_buffer_d = 0.5
    total_roci_factor = 2.5
    lf_roci_factor = 0.6
    lf_rmw_factor = 2.0
    minwind_kt = 34

    def __init__(self, track : xr.Dataset, centroids : np.ndarray) -> None:
        """Determine temporal periods and geographical regions where the storm
        affects the centroids

        Parameters
        ----------
        track : xr.Dataset
            Single tropical cyclone track.
        centroids : 2d ndarray
            Each row is a centroid [lat, lon].
        """
        self.track = track
        self.centroids = centroids

        locs = np.stack([self.track.lat, self.track.lon], axis=1)
        self.d_centroids = u_coord.dist_approx(
            locs[None, :, 0], locs[None, :, 1],
            self.centroids[None, :, 0], self.centroids[None, :, 1],
            method="geosphere")[0]

        self._set_periods()
        self.time_mask = [self._period_to_mask(p) for p in self.period]
        self.time_mask_buffered = [self._period_to_mask(p, buffer=self.period_buffer_d)
                                   for p in self.period]
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
        radii = np.fmax(self.lf_roci_factor * self.track.radius_oci.values,
                        self.lf_rmw_factor * self.track.radius_max_wind.values) * NM_TO_KM
        centr_counts = np.count_nonzero(self.d_centroids < radii[:, None], axis=1)
        # below a certain wind speed, winds are not strong enough for significant surge
        mask = (centr_counts > 1) & (self.track.max_sustained_wind > self.minwind_kt)

        # convert landfall mask to (clustered) start/end pairs
        period = []
        start = end = None
        for i, date in enumerate(self.track["time"]):
            if start is not None:
                # periods cover at most 36 hours and a split will be forced
                # at breaks of more than 12 hours.
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
        self.period = [(s.values[()], e.values[()]) for s, e in period]
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
        # total area (maximum bounds to consider)
        pad = 1 + self.total_roci_factor * self.track.radius_oci / DEG_TO_NM
        self.total_area = (
            float((self.track.lon - pad).min()),
            float((self.track.lat - pad).min()),
            float((self.track.lon + pad).max()),
            float((self.track.lat + pad).max()),
        )
        self.wind_area = []
        self.landfall_area = []
        self.surge_areas = []
        self.centroid_mask = []
        for i_event, mask_buf in enumerate(self.time_mask_buffered):
            track = self.track.sel(time=mask_buf)
            mask = self.time_mask[i_event][mask_buf]
            lf_radii = np.fmin(
                track.radius_oci.values,
                np.fmax(self.lf_roci_factor * track.radius_oci.values,
                        self.lf_rmw_factor * track.radius_max_wind.values))

            # wind area (maximum bounds to consider)
            pad = self.total_roci_factor * track.radius_oci / DEG_TO_NM
            self.wind_area.append((
                float((track.lon - pad).min()),
                float((track.lat - pad).min()),
                float((track.lon + pad).max()),
                float((track.lat + pad).max()),
            ))

            # landfall area
            pad = lf_radii / DEG_TO_NM
            self.landfall_area.append((
                float((track.lon - pad)[mask].min()),
                float((track.lat - pad)[mask].min()),
                float((track.lon + pad)[mask].max()),
                float((track.lat + pad)[mask].max()),
            ))

            # surge areas
            lf_radii *= NM_TO_KM
            centroids_mask = np.any(
                self.d_centroids[mask_buf][mask] < lf_radii[mask, None], axis=0)
            points = self.centroids[centroids_mask, ::-1]
            surge_areas = []
            if points.shape[0] > 0:
                pt_bounds = list(points.min(axis=0)) + list(points.max(axis=0))
                pt_size = (pt_bounds[2] - pt_bounds[0]) * (pt_bounds[3] - pt_bounds[1])
                if pt_size < (2 * lf_radii.max() / ONE_LAT_KM)**2:
                    small_bounds = [pt_bounds]
                else:
                    small_bounds, pt_size = _boxcover_points_along_axis(points, 3)
                min_size = 3. / (60. * 60.)
                if pt_size > (2 * min_size)**2:
                    for bounds in small_bounds:
                        bounds[:2] = [v - min_size for v in bounds[:2]]
                        bounds[2:] = [v + min_size for v in bounds[2:]]
                        surge_areas.append(bounds)
            surge_areas = [tuple([float(b) for b in bounds]) for bounds in surge_areas]
            self.surge_areas.append(surge_areas)

            # centroids affected by surge
            centroids_mask = np.zeros(self.centroids.shape[0], dtype=bool)
            for bounds in surge_areas:
                centroids_mask |= ((bounds[0] <= self.centroids[:, 1])
                                   & (bounds[1] <= self.centroids[:, 0])
                                   & (self.centroids[:, 1] <= bounds[2])
                                   & (self.centroids[:, 0] <= bounds[3]))
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
        """
        total_bounds = (min(self.centroids[:, 1].min(), self.track.lon.min()) - pad_deg,
                        min(self.centroids[:, 0].min(), self.track.lat.min()) - pad_deg,
                        max(self.centroids[:, 1].max(), self.track.lon.max()) + pad_deg,
                        max(self.centroids[:, 0].max(), self.track.lat.max()) + pad_deg)
        mid_lon = 0.5 * float(total_bounds[0] + total_bounds[2])
        proj_data = ccrs.PlateCarree()
        aspect_ratio = 1.124 * ((total_bounds[2] - total_bounds[0])
                                / (total_bounds[3] - total_bounds[1]))
        fig = plt.figure(
            figsize=(10, 10 / aspect_ratio) if aspect_ratio >= 1 else (aspect_ratio * 10, 10),
            dpi=100)
        axes = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=mid_lon))
        axes.spines['geo'].set_linewidth(0.5)
        axes.set_extent(
            (total_bounds[0], total_bounds[2], total_bounds[1], total_bounds[3]), crs=proj_data)

        # add axes tick labels
        grid = axes.gridlines(draw_labels=True, alpha=0.2, transform=proj_data, linewidth=0)
        grid.top_labels = grid.right_labels = False
        grid.xformatter = LONGITUDE_FORMATTER
        grid.yformatter = LATITUDE_FORMATTER

        # plot coastlines
        axes.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.1)

        # plot TC track with masks
        axes.plot(self.track.lon, self.track.lat, transform=proj_data, color='k', linewidth=0.5)
        for mask in self.time_mask_buffered:
            axes.plot(self.track.lon[mask], self.track.lat[mask], transform=proj_data,
                      color='k', linewidth=1.5)

        # plot rectangular areas
        linestep = max(0.5, 1 - 0.1 * self.nevents)
        linew = 1 + linestep * self.nevents
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i_event, mask in enumerate(self.time_mask):
            axes.plot(self.track.lon[mask], self.track.lat[mask], transform=proj_data,
                      color=color_cycle[i_event], linewidth=3)
            linew -= linestep
            areas = [
                self.wind_area[i_event],
                self.landfall_area[i_event],
            ] + self.surge_areas[i_event]
            for bounds in areas:
                _plot_bounds(
                    axes, bounds, transform=proj_data,
                    color=color_cycle[i_event], linewidth=linew,
                )

        # plot track data points
        axes.scatter(self.track.lon, self.track.lat, transform=proj_data, s=2)

        # adjust and output to file or screen
        fig.subplots_adjust(left=0.01, bottom=0.03, right=0.99, top=0.99, wspace=0, hspace=0)
        if path is None or not hasattr(__main__, '__file__'):
            plt.show()
        if path is not None:
            canvas = FigureCanvasAgg(fig)
            canvas.print_figure(path)
            plt.close(fig)


def _plot_bounds(axes : maxes.Axes, bounds : Tuple[float, float, float, float], **kwargs) -> None:
    """Plot given bounds as rectangular boundary lines

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        Target Axes to plot to.
    bounds : tuple
        (lon_min, lat_min, lon_max, lat_max)
    **kwargs :
        Keyword arguments that are passed on to the `plot` function.
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    axes.plot([lon_min, lon_min, lon_max, lon_max, lon_min],
              [lat_min, lat_max, lat_max, lat_min, lat_min], **kwargs)


def _boxcover_points_along_axis(points : np.ndarray, nsplits : int) -> Tuple[List[Tuple], float]:
    """Cover n-dimensional points with grid-aligned boxes

    Parameters
    ----------
    points : ndarray
        Each row is an n-dimensional point.
    nsplits : int
        Maximum number of boxes to use.

    Returns
    -------
    boxes : list of tuples
        Bounds of covering boxes.
    boxes_size : float
        Total volume/area of the covering boxes.
    """
    ndim = points.shape[1]
    bounds_min, bounds_max = points.min(axis=0), points.max(axis=0)
    final_boxes = []
    final_boxes_size = 1 + np.prod(bounds_max - bounds_min)
    for axis in range(ndim):
        splits = [((nsplits - i) / nsplits) * bounds_min[axis]
                  + (i / nsplits) * bounds_max[axis]
                  for i in range(1, nsplits)]
        boxes = []
        for i in range(nsplits):
            if i == 0:
                mask = points[:, axis] <= splits[0]
            elif i == nsplits - 1:
                mask = points[:, axis] > splits[-1]
            else:
                mask = (points[:, axis] <= splits[i]) \
                    & (points[:, axis] > splits[i - 1])
            masked_points = points[mask, :]
            if masked_points.shape[0] > 0:
                boxes.append((masked_points.min(axis=0), masked_points.max(axis=0)))
        boxes_size = np.sum([np.prod(bmax - bmin) for bmin, bmax in boxes])
        if boxes_size < final_boxes_size:
            final_boxes = [list(bmin) + list(bmax) for bmin, bmax in boxes]
            final_boxes_size = boxes_size
    return final_boxes, final_boxes_size


def _clawpack_info() -> Tuple[Optional[pathlib.Path], Tuple[str]]:
    """Information about the available clawpack version

    Returns
    -------
    path : Path or None
        If the python package clawpack is not available, None is returned.
        Otherwise, the CLAW source path is returned.
    decorators : tuple of str
        Strings describing the available version of clawpack. If it's a git
        checkout, the first string will be the full commit hash and the
        following strings will be git decorators such as tags or branch names
        that point to this checkout.
    """
    git_cmd = ["git", "log", "--pretty=format:%H%D", "-1"]
    try:
        # pylint: disable=import-outside-toplevel
        import clawpack
    except ImportError:
        return None, ()

    ver = clawpack.__version__
    path = pathlib.Path(clawpack.__file__).parent.parent
    LOGGER.info("Found Clawpack version %s in %s", ver, path)

    proc = subprocess.Popen(git_cmd, stdout=subprocess.PIPE, cwd=path)
    out = proc.communicate()[0].decode()
    if proc.returncode != 0:
        return path, (ver,)
    decorators = [out[:40]] + out[40:].split(", ")
    decorators = [d.replace("tag: ", "") for d in decorators]
    decorators = [d.replace("HEAD -> ", "") for d in decorators]
    return path, decorators


def _setup_clawpack(version : str = CLAWPACK_VERSION, overwrite: bool = False) -> None:
    """Install the specified version of clawpack if not already present

    Parameters
    ----------
    version : str, optional
        A git (short or long) hash, branch name or tag.
    overwrite : bool, optional
        If ``True``, perform a fresh install even if an existing installation is found.
        Defaults to ``False``.
    """
    path, git_ver = _clawpack_info()
    if overwrite or (
        path is None or version not in git_ver and version not in git_ver[0]
    ):
        LOGGER.info("Installing Clawpack version %s", version)
        src_path = CLAWPACK_SRC_DIR
        pkg = f"git+{CLAWPACK_GIT_URL}@{version}#egg=clawpack"
        cmd = [sys.executable, "-m", "pip", "install", "--src", src_path, "-e", pkg]
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            LOGGER.warning(f"pip install failed with return code {exc.returncode} and stdout:")
            print(exc.output.decode("utf-8"))
            raise RuntimeError("pip install failed with return code %d (see output above)."
                               "Make sure that a Fortran compiler (e.g. gfortran) is available on "
                               "your machine before using tc_surge_geoclaw!") from exc
        importlib.reload(site)
        importlib.invalidate_caches()

    with _backup_loggers(), warnings.catch_warnings():
        # pylint: disable=unused-import,import-outside-toplevel
        warnings.filterwarnings(
            "ignore",
            message="unclosed <socket.socket",
            module="clawpack",
            category=ResourceWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="the imp module is deprecated",
            module="clawpack",
            category=DeprecationWarning,
        )
        import clawpack.pyclaw


@contextlib.contextmanager
def _backup_loggers():
    """Context that reverts changes to the `disabled` states of all loggers afterwards

    Some modules (such as clawpack.pyclaw) use logging.config.fileConfig which disables all
    registered loggers; this context manager reverts these changes afterwards.
    """
    try:
        logger_state = {name: logger.disabled
                        for name, logger in logging.root.manager.loggerDict.items()
                        if isinstance(logger, logging.Logger)}
        yield logger_state
    finally:
        for name, logger in logging.root.manager.loggerDict.items():
            if name in logger_state and not logger_state[name]:
                logger.disabled = False


def _bounds_to_str(bounds : Tuple[float, float, float, float]) -> str:
    """Convert longitude/latitude bounds to a human-readable string

    Example
    -------
    >>> _bounds_to_str((-4.2, 1.0, -3.05, 2.125))
    '1N-2.125N_4.2W-3.05W'

    Parameters
    ----------
    bounds : tuple
        (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    string : str
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    return '{:.4g}{}-{:.4g}{}_{:.4g}{}-{:.4g}{}'.format(
        abs(lat_min), 'N' if lat_min >= 0 else 'S',
        abs(lat_max), 'N' if lat_max >= 0 else 'S',
        abs(lon_min), 'E' if lon_min >= 0 else 'W',
        abs(lon_max), 'E' if lon_max >= 0 else 'W')


def _dt64_to_pydt(
    date : Union[np.datetime64, np.ndarray],
) -> Union[dt.datetime, List[dt.datetime]]:
    """Convert datetime64 value or array to python datetime object or list

    Parameters
    ----------
    date : np.datetime64 or array

    Returns
    -------
    dt : datetime or list of datetime objects
    """
    result = pd.Series(date).dt.to_pydatetime()
    if isinstance(date, np.datetime64):
        return result[0]
    return list(result)
