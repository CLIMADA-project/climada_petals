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
from pathlib import Path
from typing import Optional, Tuple, List, Union

import numpy as np
import pathos.pools
from scipy import sparse
import xarray as xr

from climada.hazard import Hazard, TCTracks, TropCyclone, Centroids
from climada.hazard.trop_cyclone import (
    get_close_centroids,
    compute_angular_windspeeds,
    tctrack_to_si,
    GRADIENT_LEVEL_TO_SURFACE_WINDS,
    H_TO_S,
    KM_TO_M,
    KN_TO_MS,
    MODEL_VANG,
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

MODEL_RAIN = {'R-CLIPER': 0, 'TCR': 1}
"""Enumerate different parametric TC rain models."""

D_TO_H = (1.0 * ureg.days).to(ureg.hours).magnitude
IN_TO_MM = (1.0 * ureg.inches).to(ureg.millimeters).magnitude
M_TO_MM = (1.0 * ureg.meter).to(ureg.millimeter).magnitude
"""Unit conversion factors for JIT functions that can't use ureg"""

H_TROP = 4000
"""Depth (in m) of lower troposphere"""

DELTA_T_TROPOPAUSE = 100
"""Difference between surface and tropopause temperature (in K): T_s - T_t"""

T_ICE_K = 273.16
"""Freezing temperatur of water (in K), for conversion between K and °C"""

L_EVAP_WATER = 2.5e6
"""Latent heat of the evaporation of water (in J/kg)"""

M_WATER = 18.01528
"""Molar mass of water vapor (in g/mol)"""

M_DRY_AIR = 28.9634
"""Molar mass of dry air (in g/mol)"""

R_GAS = 8.3144621
"""Molar gas constant (in J/molK)"""

R_DRY_AIR = 1000 * R_GAS / M_DRY_AIR
"""Specific gas constant of dry air (in J/kgK)"""

RHO_A_OVER_RHO_L = 0.00117
"""Density of water vapor divided by density of liquid water"""

DEF_ELEVATION_TIF = u_const.SYSTEM_DIR / "topography_land_360as.tif"
"""Topography (land surface elevation, 0 over oceans) raster data at 0.1 degree resolution

The data is copied from the reference MATLAB implementation (source unknown)
"""

DEF_DRAG_TIF = u_const.SYSTEM_DIR / "c_drag_500.tif"
"""Gradient-level drag coefficient raster data at 0.25 degree resolution

The ERA5 'forecast_surface_roughness' (fsr) variable has been transformed into drag
coefficients (C_D) following eqs. (7) and (8) in the following work:

    Feldmann et al. (2019): Estimation of Atlantic Tropical Cyclone Rainfall Frequency in the
    United States. Journal of Applied Meteorology and Climatology 58(8): 1853–1866.
    https://doi.org/10.1175/JAMC-D-19-0011.1
"""

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
        model_kwargs: Optional[dict] = None,
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

        Currently, two models are supported to compute the rain rates: R-CLIPER and TCR. The
        R-CLIPER model is documented in Tuleya et al. 2007. The TCR model was used by
        Zhu et al. 2013 and Emanuel 2017 for the first time and is documented in detail in
        Lu et al. 2018. This implementation of TCR includes improvements proposed in
        Feldmann et al. 2019. TCR's accuracy is much higher than R-CLIPER's at the cost of
        additional computational and data requirements.

        When using the TCR model make sure that your TC track data includes the along-track
        variables "t600" (temperature at 600 hPa) and "u850"/"v850" (wind speed at 850 hPa). Both
        can be extracted from reanalysis or climate model outputs. For "t600", use the value at the
        storm center. For "u850"/"v850", use the average over the 200-500 km annulus around the
        storm center. If "u850"/"v850" is missing, this implementation sets the shear component of
        the vertical velocity to 0. If "t600" is missing, the saturation specific humidity is set
        to a universal estimate of 0.01 kg/kg. Both assumptions can have a large effect on the
        results (see Lu et al. 2018).

        The implementation of the R-CLIPER model currently does not allow modifications, so that
        `model_kwargs` is ignored with `model="R-CLIPER"`. While the TCR model can be configured in
        several ways, it is usually safe to go with the default settings. Here is the complete list
        of `model_kwargs` and their meaning with `model="TCR"` (in alphabetical order):

        c_drag_tif : Path or str, optional
            Path to a GeoTIFF file containing gridded drag coefficients (bottom friction). If not
            specified, an ERA5-based data set provided with CLIMADA is used. Default: None
        e_precip : float, optional
            Precipitation efficiency (unitless), the fraction of the vapor flux falling to the
            surface as rainfall (Lu et al. 2018, eq. (14)). Default: 0.9
        elevation_tif : Path or str, optional
            Path to a GeoTIFF file containing digital elevation model data (in m). If not
            specified, a topography at 0.1 degree resolution provided with CLIMADA is used.
            Default: None
        matlab_ref_mode : bool, optional
            This implementation is based on a (proprietary) reference implementation in MATLAB.
            However, some bug fixes have been applied in the CLIMADA implementation compared to the
            reference. If this parameter is True, do not apply the bug fixes, but reproduce the
            exact behavior of the reference implementation. Default: False
        max_w_foreground : float, optional
            The maximum value (in m/s) at which to clip the vertical velocity w before subtracting
            the background subsidence velocity w_rad. Default: 7.0
        min_c_drag : float, optional
            The drag coefficient is clipped to this minimum value (esp. over ocean). Default: 0.001
        q_900 : float, optional
            If the track data does not include "t600" values, assume this constant value of
            saturation specific humidity (in kg/kg) at 900 hPa. Default: 0.01
        res_radial_m : float, optional
            Resolution (in m) in radial direction. This is used for the computation of discrete
            derivatives of the horizontal wind fields and derived quantities. Default: 2000.0
        w_rad : float, optional
            Background subsidence velocity (in m/s) under radiative cooling. Default: 0.005
        wind_model : str, optional
            Parametric wind field model to use, see the `TropCyclone` class. Default: "ER11".

        Emanuel (2017): Assessing the present and future probability of Hurricane Harvey’s
        rainfall. Proceedings of the National Academy of Sciences 114(48): 12681–12684.
        https://doi.org/10.1073/pnas.1716222114

        Lu et al. (2018): Assessing Hurricane Rainfall Mechanisms Using a Physics-Based Model:
        Hurricanes Isabel (2003) and Irene (2011). Journal of the Atmospheric
        Sciences 75(7): 2337–2358. https://doi.org/10.1175/JAS-D-17-0264.1

        Feldmann et al. (2019): Estimation of Atlantic Tropical Cyclone Rainfall Frequency in the
        United States. Journal of Applied Meteorology and Climatology 58(8): 1853–1866.
        https://doi.org/10.1175/JAMC-D-19-0011.1

        Tuleya et al. (2007): Evaluation of GFDL and Simple Statistical Model Rainfall Forecasts
        for U.S. Landfalling Tropical Storms. Weather and Forecasting 22(1): 56–70.
        https://doi.org/10.1175/WAF972.1

        Zhu et al. (2013): Estimating tropical cyclone precipitation risk in Texas. Geophysical
        Research Letters 40(23): 6225–6230. https://doi.org/10.1002/2013GL058284

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
            Parametric rain model to use: "R-CLIPER" (faster and requires less inputs, but
            much less accurate, statistical approach, Tuleya et al. 2007), "TCR" (physics-based
            approach, requires non-standard along-track variables, Zhu et al. 2013).
            Default: "R-CLIPER".
        model_kwargs: dict, optional
            If given, forward these kwargs to the selected model. Default: None
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
                itertools.repeat(model_kwargs, num_tracks),
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
                                    model=model, model_kwargs=model_kwargs,
                                    store_rainrates=store_rainrates,
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
        model_kwargs: Optional[dict] = None,
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
            Parametric rain model to use: "R-CLIPER" (faster and requires less inputs, but
            much less accurate, statistical approach), "TCR" (physics-based approach, requires
            non-standard along-track variables). Default: "R-CLIPER".
        model_kwargs: dict, optional
            If given, forward these kwargs to the selected model. Default: None
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
            model_kwargs=model_kwargs,
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
    model_kwargs: Optional[dict] = None,
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
        Parametric rain model to use: "R-CLIPER" (faster and requires less inputs, but
        much less accurate, statistical approach), "TCR" (physics-based approach, requires
        non-standard along-track variables). Default: "R-CLIPER".
    model_kwargs: dict, optional
        If given, forward these kwargs to the selected model. Default: None
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
            model_kwargs=model_kwargs,
            store_rainrates=store_rainrates,
            metric=metric,
            intensity_thres=intensity_thres,
            max_dist_eye_km=max_dist_eye_km,
            max_memory_gb=max_memory_gb,
        )

    rainrates, reachable_centr_idx = compute_rain(
        track,
        coastal_centr,
        mod_id,
        model_kwargs=model_kwargs,
        metric=metric,
        max_dist_eye_km=max_dist_eye_km,
        max_memory_gb=0.8 * max_memory_gb,
    )
    reachable_coastal_centr_idx = coastal_idx[reachable_centr_idx]
    npositions = rainrates.shape[0]

    # obtain total rainfall in mm by multiplying by time step size (in hours) and summing up
    intensity = (rainrates * track["time_step"].values[:, None]).sum(axis=0)

    intensity[intensity < intensity_thres] = 0
    intensity_sparse = sparse.csr_matrix(
        (intensity, reachable_coastal_centr_idx, [0, intensity.size]),
        shape=intensity_shape)
    intensity_sparse.eliminate_zeros()

    rainrates_sparse = None
    if store_rainrates:
        n_reachable_coastal_centr = reachable_coastal_centr_idx.size
        indices = np.broadcast_to(
            reachable_coastal_centr_idx[None],
            (npositions, n_reachable_coastal_centr),
        ).ravel()
        indptr = np.arange(npositions + 1) * n_reachable_coastal_centr
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
    model_kwargs: Optional[dict] = None,
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
    model_kwargs: dict, optional
        If given, forward these kwargs to the selected model. Default: None
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
    model_kwargs = {} if model_kwargs is None else model_kwargs

    # start with the assumption that no centroids are within reach
    npositions = track.sizes["time"]
    reachable_centr_idx = np.zeros((0,), dtype=np.int64)
    rainrates = np.zeros((npositions, 0), dtype=np.float64)

    # convert track variables to SI units
    si_track = _track_to_si_with_q_and_shear(track, metric=metric, **model_kwargs)

    # normalize longitude values (improves performance of `dist_approx` and `get_close_centroids`)
    u_coord.lon_normalize(centroids[:, 1], center=si_track.attrs["mid_lon"])

    # Filter early with a larger threshold, but inaccurate (lat/lon) distances.
    # There is another filtering step with more accurate distances in km later.
    max_dist_eye_deg = max_dist_eye_km / (
        u_const.ONE_LAT_KM * np.cos(np.radians(np.abs(si_track["lat"].values).max()))
    )

    # restrict to centroids within rectangular bounding boxes around track positions
    track_centr_msk = get_close_centroids(
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

    d_centr = _centr_distances(si_track, track_centr, metric=metric, **model_kwargs)

    # exclude centroids that are too far from or too close to the eye
    close_centr_msk = (d_centr[""] <= max_dist_eye_km * KM_TO_M) & (d_centr[""] > 1)
    if not np.any(close_centr_msk):
        return rainrates, reachable_centr_idx

    if model == MODEL_RAIN["R-CLIPER"]:
        rainrates = _rcliper(si_track, d_centr[""], close_centr_msk, **model_kwargs)
    elif model == MODEL_RAIN["TCR"]:
        rainrates = _tcr(si_track, track_centr, d_centr, close_centr_msk, **model_kwargs)
    else:
        raise NotImplementedError
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

def _track_to_si_with_q_and_shear(
    track: xr.Dataset,
    metric: str = "equirect",
    q_900: float = 0.01,
    matlab_ref_mode: bool = False,
    **kwargs,
) -> xr.Dataset:
    """Convert track data to SI units and add Q (humidity) and vshear variables

    If the track data set does not contain the "q900" variable, but "t600", we compute the humidity
    assuming a moist adiabatic lapse rate (see `_qs_from_t_diff_level`).

    If the track data set does not contain the "vshear" variable, but "v850", we compute the wind
    shear based on the Beta Advection Model (BAM):

      v_trans = 0.8 * v850 + 0.2 * v250 + v_beta
      => 5 * (v_trans - v_beta - v850) v250 - v850 =: v_shear

    Paramaters
    ----------
    track : xr.Dataset
        TC track data.
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See `dist_approx` function in `climada.util.coordinates`.
        Default: "equirect".
    q_900 : float, optional
        If the track data does not include "t600" values, assume this constant value of saturation
        specific humidity (in kg/kg) at 900 hPa. Default: 0.01
    matlab_ref_mode : bool, optional
        Do not apply the fixes to the reference MATLAB implementation. Default: False
    kwargs : dict
        Additional kwargs are ignored.

    Returns
    -------
    xr.Dataset
    """
    si_track = tctrack_to_si(track, metric=metric)

    if "q900" in track.variables:
        si_track["q900"] = track["q900"].copy()
    elif "t600" not in track.variables:
        si_track["q900"] = ("time", np.full_like(si_track["lat"].values, q_900))
    else:
        # MATLAB computes Q at 950 hPa instead of 900 hPa (which is used in Lu et al. 2018)
        pres_in = 600
        pres_out = 950 if matlab_ref_mode else 900
        si_track["q900"] = ("time", _qs_from_t_diff_level(
            track["t600"].values,
            si_track["vmax"].values,
            pres_in,
            pres_out,
            matlab_ref_mode=matlab_ref_mode,
        ))

    if "ushear" in track.variables:
        si_track["vshear"] = (["time", "component"], (
            np.stack([track[f"{d}shear"].values.copy() for d in ["v", "u"]], axis=1)
        ))
    elif "u850" in track.variables:
        si_track["v850"] = (["time", "component"], (
            np.stack([track[f"{d}850"].values.copy() for d in ["v", "u"]], axis=1)
        ))

        # We set v_drift (or v_beta) to be a 2.5 m/s drift in meridional direction (away from the
        # equator), which seems to be common in the literature (e.g. Emanuel et al. 2006). But the
        # MATLAB implementation uses 1.5 m/s.
        si_track["vdrift"] = xr.zeros_like(si_track["v850"])
        si_track["vdrift"].values[:, 0] = (
            (1.5 if matlab_ref_mode else 2.5)
            * si_track.attrs["latsign"]
            * np.cos(np.radians(si_track["lat"].values))
        )
        si_track["vshear"] = 5 * (si_track["vtrans"] - si_track["vdrift"] - si_track["v850"])

    return si_track

def _centr_distances(
    si_track: xr.Dataset,
    centroids: np.ndarray,
    metric: str = "equirect",
    res_radial_m: float = 2000.0,
    **kwargs,
) -> dict:
    """Compute distances of centroids to storm locations required for `_compute_vertical_velocity`

    In addition to the distances to the centroids, the distances to staggered centroid locations,
    as well as the unit vectors pointing from the storm center to each centroid are returned.

    Parameters
    ----------
    si_track : xr.Dataset
        TC track data in SI units, see `tctrack_to_si`.
    centroids : ndarray
        Each row is a pair of lat/lon coordinates.
    metric : str, optional
        Approximation method to use for earth distances: "equirect" (faster) or "geosphere" (more
        accurate). See `dist_approx` function in `climada.util.coordinates`.
        Default: "equirect".
    res_radial_m : float, optional
        Spatial resolution (in m) in radial direction. Default: 2000
    kwargs : dict
        Additional keyword arguments are ignored.

    Returns
    -------
    dict
    """
    # d_centr : Distance (in m) from eyes to centroids .
    # v_centr : Vector pointing from storm center to centroids. The directional components are
    #           lat-lon, i. e. the y (meridional) direction is listed first.
    [d_centr], [v_centr] = u_coord.dist_approx(
        si_track["lat"].values[None], si_track["lon"].values[None],
        centroids[None, :, 0], centroids[None, :, 1],
        log=True, normalize=False, method=metric, units="m")

    return {
        "": d_centr,
        "+": d_centr + res_radial_m,
        "-": np.fmax(0, d_centr - res_radial_m),
        "+h": d_centr + 0.5 * res_radial_m,
        "-h": np.fmax(0, d_centr - 0.5 * res_radial_m),
        "dir": v_centr / np.fmax(1e-3, d_centr[:, :, None]),
    }

def _rcliper(
    si_track: xr.Dataset,
    d_centr: np.ndarray,
    close_centr: np.ndarray,
) -> np.ndarray:
    """Compute rain rate (in mm/h) from maximum wind speeds using the R-CLIPER model

    The model is defined in equations (3)-(5) and Table 2 (NHC) in the following publication:

    Tuleya et al. (2007): Evaluation of GFDL and Simple Statistical Model Rainfall Forecasts for
    U.S. Landfalling Tropical Storms. Weather and Forecasting 22(1): 56–70.
    https://doi.org/10.1175/WAF972.1

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `tctrack_to_si`.
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

    # rain rate inside and outside of inner core
    msk = (d_centr_km <= rad_m)
    rainrate_close[msk] = (
        rainr_0[msk] + (rainr_m[msk] - rainr_0[msk]) * (d_centr_km[msk] / rad_m[msk])
    )
    msk = ~msk
    rainrate_close[msk] = rainr_m[msk] * np.exp(-(d_centr_km[msk] - rad_m[msk]) / rad_e[msk])

    rainrate_close[np.isnan(rainrate_close)] = 0
    rainrate_close[rainrate_close < 0] = 0

    # convert from "inch per day" to mm/h
    rainrate_close *= IN_TO_MM / D_TO_H

    rainrate[close_centr] = rainrate_close
    return rainrate

def _tcr(
    si_track: xr.Dataset,
    centroids: np.ndarray,
    d_centr: dict,
    close_centr: np.ndarray,
    e_precip: float = 0.9,
    **kwargs,
) -> np.ndarray:
    """Compute rain rate (in mm/h) using the TCR model

    This follows the TCR model that was used by Zhu et al. 2013 and Emanuel 2017 for the first time
    and documented in Lu et al. 2018. This implementation includes improvements proposed by
    Feldmann et al. 2019:

    Zhu et al. (2013): Estimating tropical cyclone precipitation risk in Texas. Geophysical
    Research Letters 40(23): 6225–6230. https://doi.org/10.1002/2013GL058284

    Emanuel (2017): Assessing the present and future probability of Hurricane Harvey’s rainfall.
    Proceedings of the National Academy of Sciences 114(48): 12681–12684.
    https://doi.org/10.1073/pnas.1716222114

    Lu et al. (2018): Assessing Hurricane Rainfall Mechanisms Using a Physics-Based Model:
    Hurricanes Isabel (2003) and Irene (2011). Journal of the Atmospheric
    Sciences 75(7): 2337–2358. https://doi.org/10.1175/JAS-D-17-0264.1

    Feldmann et al. (2019): Estimation of Atlantic Tropical Cyclone Rainfall Frequency in the
    United States. Journal of Applied Meteorology and Climatology 58(8): 1853–1866.
    https://doi.org/10.1175/JAMC-D-19-0011.1

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `tctrack_to_si`.
    d_centr : dict
        Output of `_centr_distances`.
    close_centr : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.
    e_precip : float, optional
        Precipitation efficiency (unitless), the fraction of the vapor flux falling to the surface
        as rainfall (Lu et al. 2018, eq. (14)). Default: 0.9
    kwargs :
        The remaining arguments are passed on to _compute_vertical_velocity.

    Returns
    -------
    ndarray of shape (npositions, ncentroids)
    """
    # w is of shape (ntime, ncentroids)
    w = _compute_vertical_velocity(si_track, centroids, d_centr, close_centr, **kwargs)

    # derive vertical vapor flux wq by multiplying with saturation specific humidity Q900
    wq = si_track["q900"].values[:, None] * w

    # convert rainrate from "meters per second" to "milimeters per hour"
    rainrate = (M_TO_MM * H_TO_S) * e_precip * RHO_A_OVER_RHO_L * wq

    return rainrate

def _compute_vertical_velocity(
    si_track: xr.Dataset,
    centroids: np.ndarray,
    d_centr: dict,
    close_centr: np.ndarray,
    wind_model: str = "ER11",
    elevation_tif: Optional[Union[str, Path]] = None,
    c_drag_tif: Optional[Union[str, Path]] = None,
    w_rad: float = 0.005,
    res_radial_m: float = 2000.0,
    min_c_drag: float = 0.001,
    max_w_foreground: float = 7.0,
    matlab_ref_mode: bool = False,
) -> np.ndarray:
    """Compute the vertical wind velocity at locations along a tropical cyclone track

    This implements eqs. (6)-(13) from:

    Lu et al. (2018): Assessing Hurricane Rainfall Mechanisms Using a Physics-Based Model:
    Hurricanes Isabel (2003) and Irene (2011). Journal of the Atmospheric
    Sciences 75(7): 2337–2358. https://doi.org/10.1175/JAS-D-17-0264.1

    with improvements from:

    Feldmann et al. (2019): Estimation of Atlantic Tropical Cyclone Rainfall Frequency in the
    United States. Journal of Applied Meteorology and Climatology 58(8): 1853–1866.
    https://doi.org/10.1175/JAMC-D-19-0011.1

    Parameters
    ----------
    si_track : xr.Dataset
        TC track data in SI units, see `tctrack_to_si`.
    centroids : ndarray
        Each row is a pair of lat/lon coordinates.
    d_centr : ndarray of shape (npositions, ncentroids)
        Distances from storm centers to centroids.
    close_centr : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.
    wind_model : str, optional
        Parametric wind field model to use, see TropCyclone. Default: "ER11".
    elevation_tif : Path or str, optional
        Path to a GeoTIFF file containing digital elevation model data (in m). If not specified, a
        topography at 0.1 degree resolution provided with CLIMADA is used. Default: None
    c_drag_tif : Path or str, optional
        Path to a GeoTIFF file containing gridded drag coefficients (bottom friction). If not
        specified, an ERA5-based data set provided with CLIMADA is used. Default: None
    w_rad : float, optional
        Background subsidence velocity (in m/s) under radiative cooling. Default: 0.005
    res_radial_m : float, optional
        Spatial resolution (in m) in radial direction. Default: 2000
    min_c_drag : float, optional
        The drag coefficient is clipped to this minimum value (esp. over ocean). Default: 0.001
    max_w_foreground : float, optional
        The maximum value (in m/s) at which to clip the vertical velocity w before subtracting the
        background subsidence velocity w_rad. The default value is taken from the MATLAB reference
        implementation. Default: 7.0
    matlab_ref_mode : bool, optional
        Do not apply the fixes to the reference MATLAB implementation. Default: False

    Returns
    -------
    ndarray of shape (ntime, ncentroids)
    """
    h_winds = _horizontal_winds(
        si_track, d_centr, close_centr, MODEL_VANG[wind_model], matlab_ref_mode=matlab_ref_mode,
    )

    # Currently, the `close_centr` mask is ignored in the computation of the components, but it is
    # applied only afterwards. This is because it seems like the code readability would suffer a
    # lot from this. However, this might be one aspect where computational performance can be
    # improved in the future.
    w = np.zeros_like(d_centr[""])

    w_f_plus_w_t = _w_frict_stretch(
        si_track, d_centr, h_winds, centroids,
        res_radial_m=res_radial_m, c_drag_tif=c_drag_tif, min_c_drag=min_c_drag,
        matlab_ref_mode=matlab_ref_mode,
    )[close_centr]

    w_h = _w_topo(
        si_track, d_centr, h_winds, centroids,
        elevation_tif=elevation_tif, matlab_ref_mode=matlab_ref_mode,
    )[close_centr]

    w_s = _w_shear(
        si_track, d_centr, h_winds,
        res_radial_m=res_radial_m,
        matlab_ref_mode=matlab_ref_mode,
    )[close_centr]

    w[close_centr] = np.fmax(np.fmin(w_f_plus_w_t + w_h + w_s, max_w_foreground) - w_rad, 0)
    return w

def _horizontal_winds(
    si_track: xr.Dataset,
    d_centr: dict,
    close_centr: np.ndarray,
    model: int,
    matlab_ref_mode: bool = False,
) -> dict:
    """Compute all horizontal wind speed variables required for `_compute_vertical_velocity`

    Wind speeds are not only computed on the given centroids and for the given times, but also at
    staggered locations for further use in finite difference computations.

    Parameters
    ----------
    si_track : xr.Dataset
        TC track data in SI units, see `tctrack_to_si`.
    d_centr : dict
        Output of `_centr_distances`.
    close_centr : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.
    model : int
        Wind profile model selection according to MODEL_VANG.
    matlab_ref_mode : bool, optional
        Do not apply the fixes to the reference MATLAB implementation. Default: False

    Returns
    -------
    dict
    """
    ntime = si_track.sizes["time"]
    ncentroids = d_centr[""].shape[1]

    # all of the following are in meters per second:
    winds = {
        # the cyclostrophic wind direction, the meridional direction is listed first!
        "dir": (
            si_track.attrs["latsign"]
            * np.array([1.0, -1.0])[..., :]
            * d_centr["dir"][:, :, ::-1]
        ),
        # radial windprofile without influence from coriolis force
        "nocoriolis": _windprofile(
            si_track, d_centr[""], close_centr, model,
            cyclostrophic=True, matlab_ref_mode=matlab_ref_mode,
        ),
    }
    # winds['r±,t±'] : radial windprofile with offset in radius and/or time
    steps = ["", "+", "-"]
    for rstep in steps:
        for tstep in steps:
            result = np.zeros((ntime, ncentroids))
            if tstep == "":
                result[:, :] = _windprofile(
                    si_track, d_centr[rstep], close_centr, model,
                    cyclostrophic=False, matlab_ref_mode=matlab_ref_mode,
                )
            else:
                # NOTE: For the computation of time derivatives, the eye of the storm is held
                #       fixed while only the wind profile varies (see MATLAB code)
                sl = slice(2, None) if tstep == "+" else slice(None, -2)
                result[1:-1, :] = _windprofile(
                    si_track.isel(time=sl), d_centr[rstep][1:-1], close_centr[1:-1], model,
                    cyclostrophic=False, matlab_ref_mode=matlab_ref_mode,
                )
            winds[f"r{rstep},t{tstep}"] = result

    # winds['r±h,t'] : radial windprofile with half-sized offsets in radius
    for rstep in ["+", "-"]:
        winds[f"r{rstep}h,t"] = 0.5 * (winds["r,t"] + winds[f"r{rstep},t"])

    return winds

def _windprofile(
    si_track: xr.Dataset,
    d_centr: dict,
    close_centr: np.ndarray,
    model: int,
    cyclostrophic: bool = False,
    matlab_ref_mode: bool = False,
) -> np.ndarray:
    """Compute (absolute) angular wind speeds according to a parametric wind profile

    Wrapper around `compute_angular_windspeeds` (from climada.trop_cyclone) that adjusts the
    Coriolis parameter if matlab_ref_mode is True.

    Parameters
    ----------
    si_track : xr.Dataset
        TC track data in SI units, see `tctrack_to_si`.
    d_centr : ndarray of shape (npositions, ncentroids)
        Distances from storm centers to centroids.
    close_centr : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.
    model : int
        Wind profile model selection according to MODEL_VANG.
    cyclostrophic : bool, optional
        If True, don't apply the influence of the Coriolis force (set the Coriolis terms to 0).
        Default: False
    matlab_ref_mode : bool, optional
        Do not apply the fixes to the reference MATLAB implementation. Default: False

    Returns
    -------
    ndarray of shape (npositions, ncentroids)
    """
    if matlab_ref_mode:
        # Following formula (2) and the remark in Section 3 of Emanuel & Rotunno (2011), the
        # Coriolis parameter is chosen to be 5e-5 (independent of latitude) in the MATLAB
        # implementation.
        si_track = si_track.copy(deep=True)
        si_track["cp"].values[:] = 5e-5
    return compute_angular_windspeeds(
        si_track, d_centr, close_centr, model, cyclostrophic=cyclostrophic,
    )

def _w_shear(
    si_track: xr.Dataset,
    d_centr: dict,
    h_winds: dict,
    res_radial_m: float = 2000.0,
    matlab_ref_mode: bool = False,
) -> np.ndarray:
    """Compute the shear component of the vertical wind velocity

    This implements eq. (12) from:

    Lu et al. (2018): Assessing Hurricane Rainfall Mechanisms Using a Physics-Based Model:
    Hurricanes Isabel (2003) and Irene (2011). Journal of the Atmospheric
    Sciences 75(7): 2337–2358. https://doi.org/10.1175/JAS-D-17-0264.1

    Parameters
    ----------
    si_track : xr.Dataset
        TC track data in SI units, see `tctrack_to_si`.
    d_centr : dict
        Output of `_centr_distances`.
    h_winds : dict
        Output of `_horizontal_winds`.
    res_radial_m : float, optional
        Spatial resolution (in m) in radial direction. Default: 2000
    matlab_ref_mode : bool, optional
        Do not apply the fixes to the reference MATLAB implementation. Default: False

    Returns
    -------
    ndarray of shape (ntime, ncentroids)
    """
    if "vshear" not in si_track.variables:
        return np.zeros_like(h_winds["r,t"])

    # fac_scalar : scalar factor from eq. (12) in Lu et al. 2018:
    #                 g / (cp * (Ts - Tt) * (1 - ep) * N**2
    #              g : gravitational acceleration (10 m*s**-2)
    #              cp : isobaric specific heat of dry air (1000 J*(kg*K)**-1)
    #              Ts - Tt : difference between surface and tropopause temperature (100 K)
    #              ep : precipitation efficiency (0.5)
    #              N : buoyancy frequency for dry air (2e-2 s**-1)
    fac_scalar = 0.5

    # the following are fixes of bugs in the MATLAB implementation
    fac = fac_scalar * (
        si_track["cp"].values[:, None]
        + (2.0 if matlab_ref_mode else 1.0) * h_winds["r,t"] / (1 + d_centr[""])
        + (h_winds["r+,t"] - h_winds["r-,t"]) / ((1.0 if matlab_ref_mode else 2.0) * res_radial_m)
    )

    return h_winds["nocoriolis"] * fac * (
        d_centr["dir"] * si_track["vshear"].values[:, None, :]
    ).sum(axis=-1)

def _w_topo(
    si_track: xr.Dataset,
    d_centr: dict,
    h_winds: dict,
    centroids: np.ndarray,
    elevation_tif: Optional[Union[str, Path]] = None,
    matlab_ref_mode: bool = False,
) -> np.ndarray:
    """Compute the topographic component w_h of the vertical wind velocity

    This implements eq. (7) from:

    Lu et al. (2018): Assessing Hurricane Rainfall Mechanisms Using a Physics-Based Model:
    Hurricanes Isabel (2003) and Irene (2011). Journal of the Atmospheric
    Sciences 75(7): 2337–2358. https://doi.org/10.1175/JAS-D-17-0264.1

    Parameters
    ----------
    si_track : xr.Dataset
        TC track data in SI units, see `tctrack_to_si`.
    d_centr : dict
        Output of `_centr_distances`.
    h_winds : dict
        Output of `_horizontal_winds`.
    centroids : ndarray
        Each row is a pair of lat/lon coordinates.
    elevation_tif : Path or str, optional
        Path to a GeoTIFF file containing digital elevation model data (in m). If not specified, a
        topography at 0.1 degree resolution provided with CLIMADA is used. Default: None
    matlab_ref_mode : bool, optional
        Do not apply the fixes to the reference MATLAB implementation. Default: False

    Returns
    -------
    ndarray of shape (ntime, ncentroids)
    """
    if elevation_tif is None:
        elevation_tif = DEF_ELEVATION_TIF

    # Note that the gradient of the raster products is smoothed (as in the reference MATLAB
    # implementation), even though it should be piecewise constant since the data itself is read
    # with bilinear interpolation.
    h, h_grad = u_coord.read_raster_sample_with_gradients(
        elevation_tif, centroids[:, 0], centroids[:, 1], method=("linear", "linear"),
    )

    # only consider interaction with terrain over land
    mask_onland = (h > -1)
    h[~mask_onland] = -1
    h_grad[~mask_onland, :] *= 0

    # reduce effect of translation speed and orography outside of storm core
    vtrans_red = si_track["vtrans"].values[:, None, :] * (
        np.clip((300 * KM_TO_M - d_centr[""]) / (50 * KM_TO_M), 0, 1)[:, :, None]
    )
    h_grad_red = h_grad[None, :, :] * (
        np.clip((150 * KM_TO_M - d_centr[""]) / (30 * KM_TO_M), 0.2, 0.6)[:, :, None]
    )
    return (
        (vtrans_red + h_winds["nocoriolis"][..., None] * h_winds["dir"]) * h_grad_red
    ).sum(axis=-1)

def _w_frict_stretch(
    si_track: xr.Dataset,
    d_centr: dict,
    h_winds: dict,
    centroids: np.ndarray,
    res_radial_m: float = 2000.0,
    c_drag_tif: Optional[Union[str, Path]] = None,
    min_c_drag: float = 0.001,
    matlab_ref_mode: bool = False,
) -> np.ndarray:
    """Compute the sum of the frictional and stretching components w_f and w_t

    This implements eq. (6) and (11) from:

    Lu et al. (2018): Assessing Hurricane Rainfall Mechanisms Using a Physics-Based Model:
    Hurricanes Isabel (2003) and Irene (2011). Journal of the Atmospheric
    Sciences 75(7): 2337–2358. https://doi.org/10.1175/JAS-D-17-0264.1

    Parameters
    ----------
    si_track : xr.Dataset
        TC track data in SI units, see `tctrack_to_si`.
    d_centr : dict
        Output of `_centr_distances`.
    h_winds : dict
        Output of `_horizontal_winds`.
    centroids : ndarray
        Each row is a pair of lat/lon coordinates.
    res_radial_m : float, optional
        Spatial resolution (in m) in radial direction. Default: 2000
    c_drag_tif : Path or str, optional
        Path to a GeoTIFF file containing gridded drag coefficients (bottom friction). If not
        specified, an ERA5-based data set provided with CLIMADA is used. Default: None
    min_c_drag : float, optional
        The drag coefficient is clipped to this minimum value (esp. over ocean). Default: 0.001
    matlab_ref_mode : bool, optional
        Do not apply the fixes to the reference MATLAB implementation. Default: False

    Returns
    -------
    ndarray of shape (ntime, ncentroids)
    """
    # sum of frictional and stretching components w_f and w_t
    if c_drag_tif is None:
        c_drag_tif = DEF_DRAG_TIF

    # vnet : absolute value of the total surface wind
    vnet = {
        f"r{rstep}h,t": np.linalg.norm(
            (
                h_winds[f"r{rstep}h,t"][..., None] * h_winds["dir"]
                + si_track["vtrans"].values[:, None, :]
            ),
            axis=-1,
        )
        for rstep in ["+", "-"]
    }

    # Note that the gradient of the raster products is smoothed (as in the reference MATLAB
    # implementation), even though it should be piecewise constant since the data itself is read
    # with bilinear interpolation.
    cd, cd_grad = u_coord.read_raster_sample_with_gradients(
        c_drag_tif, centroids[:, 0], centroids[:, 1], method=("linear", "linear"),
    )

    mask_onland = (cd >= min_c_drag)
    cd[~mask_onland] = min_c_drag
    cd_grad[~mask_onland, :] *= 0

    cd_hstep = (cd_grad[None] * (0.5 * res_radial_m * d_centr["dir"])).sum(axis=-1)
    cd = {
        "": cd,
        "r+h": np.clip(cd + cd_hstep, 0.0, 0.01),
        "r-h": np.clip(cd - cd_hstep, 0.0, 0.01),
    }

    # tau : azimuthal surface stress, equation (8) in (Lu et al. 2018)
    tau = {
        f"r{rstep}h,t": (
            -cd[f"r{rstep}h"] * h_winds[f"r{rstep}h,t"] * vnet[f"r{rstep}h,t"]
        ) for rstep in ["+", "-"]
    }

    # evaluate derivative of angular momentum M wrt r
    #   M = r * V + 0.5 * f * r^2
    #   dMdr = r * (f + dVdr) + V
    dMdr = {
        f"r{rstep}h,t": (
            d_centr[f"{rstep}h"] * (
                si_track["cp"].values[:, None]
                + (1 if rstep == "+" else -1) * (
                    h_winds[f"r{rstep},t"] - h_winds[f"r,t"]
                ) / res_radial_m
            ) + h_winds[f"r{rstep}h,t"]
        )
        for rstep in ["+", "-"]
    }

    # compute derivative of angular momentum M wrt t
    #   M = r * V + 0.5 * f * r^2
    #   dMdt = r * dVdt
    # combine equations (6) and (11) in (Lu et al. 2018):
    #   w_f + w_t = (1 / r) * (d/dr) [pre_wf_wt]
    #   where [pre_wf_wt] := [r^2 / dMdr * (H_TROP * dVdt - tau)]
    pre_wf_wt = {
        f"r{rstep}h,t": (
            d_centr[rstep]**2 / np.fmax(10, dMdr[f"r{rstep}h,t"]) * (
                # NOTE: The attenuation factor is not in Lu et al. 2018.
                #       It ranges from -1 (at the storm center) to 1 (at RMW and higher).
                np.fmin(1, (-1 + 2 * (d_centr[rstep] / si_track["rad"].values[:, None])**2)) *
                # continue with "official" formula:
                H_TROP * (
                    h_winds[f"r{rstep},t+"] - h_winds[f"r{rstep},t-"]
                ) / (2 * si_track["tstep"].values[:, None])
                - tau[f"r{rstep}h,t"]
            )
        ) for rstep in ["+", "-"]
    }

    return (pre_wf_wt["r+h,t"] - pre_wf_wt["r-h,t"]) / (res_radial_m * d_centr[""])

def _qs_from_t_diff_level(
    temps_in: np.ndarray,
    vmax: np.ndarray,
    pres_in: float,
    pres_out: float,
    cap_heat_air: float = 1005.0,
    max_iter: int = 5,
    matlab_ref_mode: bool = False,
) -> np.ndarray:
    """Compute the humidity from temperature assuming a moist adiabatic lapse rate

    The input temperatures may be given on a different pressure level than the output humidities.
    When computing Q from T on the same pressure level, see `_qs_from_t_same_level` instead.

    The approach assumes that the lapse rate dT/dz is given by the law for the moist adiabatic
    lapse rate so that the following expression for the "total entropy" is a conserved quantity
    across pressure levels (see eq. (4.5.9) in Emanuel (1994): Atmospheric convection):

        s = cp * log(T) + Lv * r / T - Rd * log(p) = const.,

    where:

        T : temperature,
        r : mixing ratio (equals Q/(1-Q) where Q is saturation specific humidity),
        p : pressure,
        cp : isobaric specific heat of dry air,
        Lv : latent heat of the evaporation of water,
        Rd : specific gas constant of dry air.

    Since it's possible to compute Q from T on the same pressure level (see
    `_qs_from_t_same_level`), we can use this relationship to compute Q at one pressure level from
    T given on a different pressure level. However, since we can't solve the equation for T
    analytically, we use the Newton-Raphson method to find the solution.

    Parameters
    ----------
    temps_in : ndarray
        Temperatures (in K) at the pressure level pres_in.
    vmax : ndarray
        Maximum surface wind speeds (in m/s).
    pres_in : float
        Pressure level (in hPa) at which the input temperatures are given.
    pres_out : float
        Pressure level (in hPa) at which the output humidity is computed.
    cap_heat_air : float, optional
        Isobaric specific heat of dry air (in J/(kg*K)). The value depends on env. conditions and
        lies, for example, between 1003 (for 0 °C) and 1012 (typical room conditions).
        Default: 1005
    max_iter : int, optional
        The number of Newton-Raphson steps to take. Default: 5
    matlab_ref_mode : bool, optional
        Do not apply the fixes to the reference MATLAB implementation. Default: False

    Returns
    -------
    q_out : ndarray
        For each temperature value in temps_in, a value of saturation specific humidity (in kg/kg)
        at the pressure level pres_out.
    """
    # c_vmax : rescale factor from (squared) surface to (squared) gradient winds
    #          MATLAB code uses c_vmax=1.6 (source unknown)
    c_vmax = 1.6 if matlab_ref_mode else GRADIENT_LEVEL_TO_SURFACE_WINDS**-2

    # first, calculate q_in from temps_in
    q_in, _ = _qs_from_t_same_level(
        pres_in, np.fmax(T_ICE_K - 50, temps_in), matlab_ref_mode=matlab_ref_mode)

    # derive (temps_out, q_out) from (temps_in, q_in) iteratively (Newton-Raphson method)
    q_out = np.zeros_like(q_in)
    temps_out = temps_in.copy() + 20  # first guess, assuming that pres_out > pres_in

    # exclude missing data (fill values) in the inputs
    mask = (temps_in > 100)

    # r_in : mixing ratio at pressure level pres_in
    if matlab_ref_mode:
        # In the MATLAB implementation, _qs_from_t_same_level does actually return the mixing
        # ratio instead of the specific humidity.
        r_in = q_in[mask]
    else:
        r_in = q_in[mask] / (1 - q_in[mask])

    # s : Total entropy, which is conserved across pressure levels when assuming a moist adiabatic
    #     lapse rate. The additional vmax-term is a correction to account for the fact that the
    #     eyewall is warmer than the environment at 600 hPa (thermal wind balance).
    s_in = (
        cap_heat_air * np.log(temps_in[mask])
        + L_EVAP_WATER * r_in / temps_in[mask]
        - R_DRY_AIR * np.log(pres_in)
        + c_vmax * vmax[mask]**2 / DELTA_T_TROPOPAUSE
    )

    # solve `s_out(T_out) - s_in = 0` using the Newton-Raphson method
    for it in range(max_iter):
        # compute new estimate of q_out from current estimate of T_out
        q_out[mask], dQdT = _qs_from_t_same_level(
            pres_out, temps_out[mask], gradient=True, matlab_ref_mode=matlab_ref_mode,
        )
        if matlab_ref_mode:
            # In the MATLAB implementation, _qs_from_t_same_level does actually return the mixing
            # ratio instead of the specific humidity.
            r_out = q_out[mask]
            drdT = dQdT
        else:
            r_out = q_out[mask] / (1 - q_out[mask])
            drdT = dQdT / (1 - q_out[mask])**2
        s_out = (
            cap_heat_air * np.log(temps_out[mask])
            + L_EVAP_WATER * r_out / temps_out[mask]
            - R_DRY_AIR * np.log(pres_out)
        )
        dsdT = (
            cap_heat_air * temps_out[mask]
            + L_EVAP_WATER * (drdT * temps_out[mask] - r_out)
        ) / temps_out[mask]**2

        # take Newton step
        temps_out[mask] -= (s_out - s_in) / dsdT

    return q_out

def _qs_from_t_same_level(
    p_ref: float,
    temps: np.ndarray,
    gradient: bool = False,
    tetens_coeffs: str = "Buck1981",
    matlab_ref_mode: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute saturation specific humidity from temperature at a given pressure level

    This uses the Tetens (or Magnus) formula for saturation vapor pressure (over water) with
    coefficients given in:

    Murray (1967): On the Computation of Saturation Vapor Pressure. Journal of Applied Meteorology
    and Climatology 6(1): 203–204. http://doi.org/10.1175/1520-0450(1967)006<0203:OTCOSV>2.0.CO;2

    Bolton (1980): The Computation of Equivalent Potential Temperature. Monthly Weather Review
    108(7): 1046–1053. https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2

    Buck (1981): New Equations for Computing Vapor Pressure and Enhancement Factor. Journal of
    Applied Meteorology and Climatology 20(12): 1527-1532.
    http://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2

    The default coefficients (Buck 1981) are also used by the ECMWF, see Section 7.2.1 (b) of the
    following document:

    ECMWF (2023): IFS Documentation CY48R1, Part IV: Physical Processes.
    https://www.ecmwf.int/en/elibrary/81370-ifs-documentation-cy48r1-part-iv-physical-processes

    Parameters
    ----------
    p_ref : float
        Reference pressure level (in hPa) at which the input temperatures are given and at which
        output humidity values are computed.
    temps : ndarray
        Temperatures (in K) at the pressure level p_ref.
    gradient : bool, optional
        If True, compute the derivative of the functional relationship between Q and T.
    tetens_coeffs : str, optional
        Coefficients to use for the Tetens formula. One of "Buck1981", "Bolton1980",
        or "Murray1967". This is overwritten if `matlab_ref_mode` is True because the reference
        MATLAB implementation uses the "Bolton1980" coefficients. Default: "Buck1981"
    matlab_ref_mode : bool, optional
        Do not apply the fixes to the reference MATLAB implementation. Default: False

    Returns
    -------
    qs : ndarray
        For each temperature value in temp, a value of saturation specific humidity (in kg/kg).
    dQdT : ndarray
        If `gradient` is False, this is None. Otherwise, the derivative of Q with respect to T is
        returned.
    """
    if matlab_ref_mode:
        tetens_coeffs = "Bolton1980"

    if tetens_coeffs == "Murray1967":
        a = 17.2693882
        b = 35.86
        c = 6.1078
    elif tetens_coeffs == "Bolton1980":
        a = 17.67
        b = 29.65
        c = 6.112
    elif tetens_coeffs == "Buck1981":
        a = 17.502
        b = 32.19
        c = 6.1121
    else:
        raise NotImplementedError

    # es : saturation vapor pressure (in hPa)
    es = c * np.exp(a * (temps - T_ICE_K) / (temps - b))

    fact = M_WATER / M_DRY_AIR
    if matlab_ref_mode:
        # in the reference implementation, the formula for the "mixing ratio" is used which is
        # almost the same as the "specific humidity" in practice
        qs = fact * es / (p_ref - es)
    else:
        qs = fact * es / (p_ref - (1 - fact) * es)

    dQdT = None
    if gradient:
        if matlab_ref_mode:
            # Specific gas constant of water vapor (in J / kgK)
            # (overwritten by 491 instead of 461 in the MATLAB code)
            r_water = 1000 * R_GAS / M_WATER
            r_water = 491
            # this approximation of the derivative is used in the MATLAB code:
            dQdT = (L_EVAP_WATER / r_water) / temps**2 * qs
        else:
            dQdT = a * (T_ICE_K - b) / (temps - b)**2 * qs * (1 + (1 - fact) * qs / fact)

    return qs, dQdT
