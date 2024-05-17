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
    H_TO_S,
    KM_TO_M,
    KN_TO_MS,
    MODEL_VANG,
)
from climada.util import ureg
from climada.util.api_client import Client
import climada.util.constants as u_const
import climada.util.coordinates as u_coord

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

GRADIENT_LEVEL_TO_SURFACE_WINDS = 0.8
"""Gradient-to-surface wind reduction factor according to Table 2 in:

Franklin, J.L., Black, M.L., Valde, K. (2003): GPS Dropwindsonde Wind Profiles in Hurricanes and
Their Operational Implications. Weather and Forecasting 18(1): 32–44.
https://doi.org/10.1175/1520-0434(2003)018<0032:GDWPIH>2.0.CO;2

Note that we here use a value different from the one in ``climada.hazard.trop_cyclone`` because the
focus is not only on the eyewall region, but also on the outer vortex, which is a little more
important for precipitation than for wind effects.
"""


def default_elevation_tif():
    """Topography (land surface elevation, 0 over oceans) raster data at 0.1 degree resolution

    SRTM data upscaled to 0.1 degree resolution using the "average" method of gdalwarp.
    """
    client = Client()
    dsi = client.get_dataset_info(name='topography_land_360as', status='package-data')
    _, [elevation_tif] = client.download_dataset(dsi)
    return elevation_tif


def default_drag_tif():
    """Gradient-level drag coefficient raster data at 0.25 degree resolution

    The ERA5 'forecast_surface_roughness' (fsr) variable has been transformed into drag
    coefficients (C_D) following eqs. (7) and (8) in the following work:

        Feldmann et al. (2019): Estimation of Atlantic Tropical Cyclone Rainfall Frequency in the
        United States. Journal of Applied Meteorology and Climatology 58(8): 1853–1866.
        https://doi.org/10.1175/JAMC-D-19-0011.1
    """
    client = Client()
    dsi = client.get_dataset_info(name='c_drag_500', status='package-data')
    _, [drag_tif] = client.download_dataset(dsi)
    return drag_tif


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
        centroids: Centroids = None,
        pool: Optional[pathos.pools.ProcessPool] = None,
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

        This function sets the ``intensity`` attribute to contain, for each centroid,
        the total amount of rain experienced over the whole period of each TC event in mm.
        The amount of rain is set to 0 if it does not exceed the threshold ``intensity_thres``.

        The ``category`` attribute is set to the value of the ``category``-attribute
        of each of the given track data sets.

        The ``basin`` attribute is set to the genesis basin for each event, which
        is the first value of the ``basin``-variable in each of the given track data sets.

        Optionally, the time-dependent rain rates can be stored using the ``store_rainrates``
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
            Centroids where to model TC. Default: centroids at 360 arc-seconds resolution within
            tracks' bounds.
        pool : pathos.pool, optional
            Pool that will be used for parallel computation of rain fields. Default: None
        model : str, optional
            Parametric rain model to use: "R-CLIPER" (faster and requires less inputs, but
            much less accurate, statistical approach, Tuleya et al. 2007), "TCR" (physics-based
            approach, requires non-standard along-track variables, Zhu et al. 2013).
            Default: "R-CLIPER".
        model_kwargs: dict, optional
            If given, forward these kwargs to the selected model. The implementation of the
            R-CLIPER model currently does not allow modifications, so that ``model_kwargs`` is
            ignored with ``model="R-CLIPER"``. While the TCR model can be configured in several ways,
            it is usually safe to go with the default settings. Here is the complete list of
            ``model_kwargs`` and their meaning with ``model="TCR"`` (in alphabetical order):

            c_drag_tif : Path or str, optional
                Path to a GeoTIFF file containing gridded drag coefficients (bottom friction). If
                not specified, an ERA5-based data set provided with CLIMADA is used. Default: None
            e_precip : float, optional
                Precipitation efficiency (unitless), the fraction of the vapor flux falling to the
                surface as rainfall (Lu et al. 2018, eq. (14)). Note that we follow the MATLAB
                reference implementation and use 0.5 as a default value instead of the 0.9 that was
                proposed in Lu et al. 2018. Default: 0.5
            elevation_tif : Path or str, optional
                Path to a GeoTIFF file containing digital elevation model data (in m). If not
                specified, an SRTM-based topography at 0.1 degree resolution provided with CLIMADA
                is used. Default: None
            matlab_ref_mode : bool, optional
                This implementation is based on a (proprietary) reference implementation in MATLAB.
                However, some (minor) changes have been applied in the CLIMADA implementation
                compared to the reference:

                * In the computation of horizontal wind speeds, we compute the Coriolis parameter
                  from latitude. The MATLAB code assumes a constant parameter value (5e-5).
                * As a rescaling factor from surface to gradient winds, we use a factor from the
                  literature. The factor in MATLAB is very similar, but does not specify a
                  source.
                * Instead of the "specific humidity", the (somewhat simpler) formula for the
                  "mixing ratio" is used in the MATLAB code. These quantities are almost the same
                  in practice.
                * We use the approximation of the Clausius-Clapeyron equation used by the ECMWF
                  (Buck 1981) instead of the one used in the MATLAB code (Bolton 1980).

                Since it might be useful to have a version that replicates the behavior of the
                reference implementation, this parameter can be set to True to enforce the exact
                behavior of the reference implementation. Default: False
            max_w_foreground : float, optional
                The maximum value (in m/s) at which to clip the vertical velocity w before
                subtracting the background subsidence velocity w_rad. Default: 7.0
            min_c_drag : float, optional
                The drag coefficient is clipped to this minimum value (esp. over ocean).
                Default: 0.001
            q_950 : float, optional
                If the track data does not include "t600" values, assume this constant value of
                saturation specific humidity (in kg/kg) at 950 hPa. Default: 0.01
            res_radial_m : float, optional
                Resolution (in m) in radial direction. This is used for the computation of discrete
                derivatives of the horizontal wind fields and derived quantities. Default: 2000.0
            w_rad : float, optional
                Background subsidence velocity (in m/s) under radiative cooling. Default: 0.005
            wind_model : str, optional
                Parametric wind field model to use, see the ``TropCyclone`` class. Default: "ER11".

            Default: None
        ignore_distance_to_coast : boolean, optional
            If True, centroids far from coast are not ignored. Default: False.
        store_rainrates : boolean, optional
            If True, the Hazard object gets a list ``rainrates`` of sparse matrices. For each track,
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
            limit applies to each thread separately if a ``pool`` is used. Default: 8

        Returns
        -------
        TCRain
        """
        num_tracks = tracks.size
        if centroids is None:
            centroids = Centroids.from_pnt_bounds(tracks.get_bounds(), res=0.1)

        if ignore_distance_to_coast:
            # Select centroids with lat <= max_latitude
            [idx_centr_filter] = (np.abs(centroids.lat) <= max_latitude).nonzero()
        else:
            # Select centroids which are inside max_dist_inland_km and lat <= max_latitude
            [idx_centr_filter] = (
                (centroids.get_dist_coast() <= max_dist_inland_km * 1000)
                & (np.abs(centroids.lat) <= max_latitude)
            ).nonzero()

        # Filter early with a larger threshold, but inaccurate (lat/lon) distances.
        # Later, there will be another filtering step with more accurate distances in km.
        max_dist_eye_deg = max_dist_eye_km / (
            u_const.ONE_LAT_KM * np.cos(np.radians(max_latitude))
        )

        # Restrict to coastal centroids within reach of any of the tracks
        t_lon_min, t_lat_min, t_lon_max, t_lat_max = tracks.get_bounds(deg_buffer=max_dist_eye_deg)
        t_mid_lon = 0.5 * (t_lon_min + t_lon_max)
        filtered_centroids = centroids.coord[idx_centr_filter]
        u_coord.lon_normalize(filtered_centroids[:, 1], center=t_mid_lon)
        idx_centr_filter = idx_centr_filter[
            (t_lon_min <= filtered_centroids[:, 1])
            & (filtered_centroids[:, 1] <= t_lon_max)
            & (t_lat_min <= filtered_centroids[:, 0])
            & (filtered_centroids[:, 0] <= t_lat_max)
        ]

        LOGGER.info('Mapping %s tracks to %s coastal centroids.', str(tracks.size),
                    str(idx_centr_filter.size))
        if pool:
            chunksize = max(min(num_tracks // pool.ncpus, 1000), 1)
            tc_haz_list = pool.map(
                cls._from_track, tracks.data,
                itertools.repeat(centroids, num_tracks),
                itertools.repeat(idx_centr_filter, num_tracks),
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
                    cls._from_track(track, centroids, idx_centr_filter,
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
        return haz

    @classmethod
    def _from_track(
        cls,
        track: xr.Dataset,
        centroids: Centroids,
        idx_centr_filter: np.ndarray,
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
        idx_centr_filter : np.ndarray
            Indices of centroids to restrict to (e.g. sufficiently close to coast).
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
            "geosphere" (more accurate). See ``dist_approx`` function in ``climada.util.coordinates``.
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
            idx_centr_filter=idx_centr_filter,
            model=model,
            model_kwargs=model_kwargs,
            store_rainrates=store_rainrates,
            metric=metric,
            intensity_thres=intensity_thres,
            max_dist_eye_km=max_dist_eye_km,
            max_memory_gb=max_memory_gb,
        )

        new_haz = cls(haz_type=HAZ_TYPE)
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
        new_haz.date = np.array([dt.datetime(
            track["time"].dt.year.values[0],
            track["time"].dt.month.values[0],
            track["time"].dt.day.values[0]
        ).toordinal()])
        new_haz.orig = np.array([track.orig_event_flag])
        new_haz.category = np.array([track.category])
        new_haz.basin = [str(track["basin"].values[0])]
        return new_haz

def _compute_rain_sparse(
    track: xr.Dataset,
    centroids: Centroids,
    idx_centr_filter: np.ndarray,
    model: str = 'R-CLIPER',
    model_kwargs: Optional[dict] = None,
    store_rainrates: bool = False,
    metric: str = "equirect",
    intensity_thres: float = DEF_INTENSITY_THRES,
    max_dist_eye_km: float = DEF_MAX_DIST_EYE_KM,
    max_memory_gb: float = DEF_MAX_MEMORY_GB,
) -> Tuple[sparse.csr_matrix, Optional[sparse.csr_matrix]]:
    """Version of ``compute_rain`` that returns sparse matrices and limits memory usage

    Parameters
    ----------
    track : xr.Dataset
        Single tropical cyclone track.
    centroids : Centroids
        Centroids instance.
    idx_centr_filter : np.ndarray
        Indices of centroids to restrict to (e.g. sufficiently close to coast).
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
        "geosphere" (more accurate). See ``dist_approx`` function in ``climada.util.coordinates``.
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
        If store_rainrates is False, ``None`` is returned.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs

    try:
        mod_id = MODEL_RAIN[model]
    except KeyError as err:
        raise ValueError(f'Model not implemented: {model}.') from err

    ncentroids = centroids.coord.shape[0]
    npositions = track.sizes["time"]
    rainrates_shape = (npositions, ncentroids)
    intensity_shape = (1, ncentroids)

    # initialise arrays for the assumption that no centroids are within reach
    rainrates_sparse = (
        sparse.csr_matrix(([], ([], [])), shape=rainrates_shape)
        if store_rainrates else None
    )
    intensity_sparse = sparse.csr_matrix(([], ([], [])), shape=intensity_shape)

    # The TCR model requires at least three track positions because both forward and backward
    # differences in time are used.
    if npositions == 0 or model == "TCR" and npositions < 3:
        return intensity_sparse, rainrates_sparse

    # convert track variables to SI units
    si_track = _track_to_si_with_q_and_shear(track, metric=metric, **model_kwargs)

    # when done properly, finding and storing the close centroids is not a memory bottle neck and
    # can be done before chunking:
    centroids_close, mask_centr, mask_centr_alongtrack = get_close_centroids(
        si_track=si_track,
        centroids=centroids.coord[idx_centr_filter],
        buffer_km=max_dist_eye_km,
        metric=metric,
    )
    idx_centr_filter = idx_centr_filter[mask_centr]
    n_centr_close = centroids_close.shape[0]
    if n_centr_close == 0:
        return intensity_sparse, rainrates_sparse

    # the total memory requirement in GB if we compute everything without chunking:
    # 8 Bytes per entry (float64), 25 arrays
    total_memory_gb = npositions * n_centr_close * 8 * 25 / 1e9
    if total_memory_gb > max_memory_gb and npositions > 3:
        # If the number of positions is down to 3 already, we do not split any further. In that
        # case, we just take the risk and try to do the computation anyway. It might still work
        # since we have only computed an upper bound for the number of affected centroids.

        # Split the track into chunks, compute the result for each chunk, and combine:
        return _compute_rain_sparse_chunked(
            mask_centr_alongtrack,
            track,
            centroids,
            idx_centr_filter,
            model=model,
            model_kwargs=model_kwargs,
            store_rainrates=store_rainrates,
            metric=metric,
            intensity_thres=intensity_thres,
            max_dist_eye_km=max_dist_eye_km,
            max_memory_gb=max_memory_gb,
        )

    rainrates, idx_centr_reachable = compute_rain(
        si_track,
        centroids_close,
        mod_id,
        model_kwargs=model_kwargs,
        metric=metric,
        max_dist_eye_km=max_dist_eye_km,
    )
    idx_centr_filter = idx_centr_filter[idx_centr_reachable]
    npositions = rainrates.shape[0]

    # obtain total rainfall in mm by multiplying by time step size (in hours) and summing up
    intensity = (rainrates * track["time_step"].values[:, None]).sum(axis=0)

    intensity[intensity < intensity_thres] = 0
    intensity_sparse = sparse.csr_matrix(
        (intensity, idx_centr_filter, [0, intensity.size]),
        shape=intensity_shape)
    intensity_sparse.eliminate_zeros()

    rainrates_sparse = None
    if store_rainrates:
        n_centr_filter = idx_centr_filter.size
        indices = np.broadcast_to(idx_centr_filter[None], (npositions, n_centr_filter)).ravel()
        indptr = np.arange(npositions + 1) * n_centr_filter
        rainrates_sparse = sparse.csr_matrix((rainrates.ravel(), indices, indptr),
                                              shape=rainrates_shape)
        rainrates_sparse.eliminate_zeros()

    return intensity_sparse, rainrates_sparse

def _compute_rain_sparse_chunked(
    mask_centr_alongtrack: np.ndarray,
    track: xr.Dataset,
    *args,
    max_memory_gb: float = DEF_MAX_MEMORY_GB,
    **kwargs,
) -> Tuple[sparse.csr_matrix, Optional[sparse.csr_matrix]]:
    """Call ``_compute_rain_sparse`` for chunks of the track and re-assemble the results

    Parameters
    ----------
    mask_centr_alongtrack : np.ndarray of shape (npositions, ncentroids)
        Each row is a mask that indicates the centroids within reach for one track position.
    track : xr.Dataset
        Single tropical cyclone track.
    max_memory_gb : float, optional
        Maximum memory requirements (in GB) for the computation of a single chunk of the track.
        Default: 8
    args, kwargs :
        The remaining arguments are passed on to ``_compute_rain_sparse``.

    Returns
    -------
    intensity, rainrates :
        See ``_compute_rain_sparse`` for a description of the return values.
    """
    npositions = track.sizes["time"]
    # The memory requirements for each track position are estimated for the case of 25 arrays
    # containing `nreachable` float64 (8 Byte) values each. The chunking is only relevant in
    # extreme cases with a very high temporal and/or spatial resolution.
    max_nreachable = max_memory_gb * 1e9 / (8 * 25 * npositions)
    split_pos = [0]
    chunk_size = 3
    while split_pos[-1] + chunk_size < npositions:
        chunk_size += 1
        # create overlap between consecutive chunks
        chunk_start = max(0, split_pos[-1] - 2)
        chunk_end = chunk_start + chunk_size
        nreachable = mask_centr_alongtrack[chunk_start:chunk_end].any(axis=0).sum()
        if nreachable > max_nreachable:
            split_pos.append(chunk_end - 1)
            chunk_size = 3
    split_pos.append(npositions)

    intensity = []
    rainrates = []
    for prev_chunk_end, chunk_end in zip(split_pos[:-1], split_pos[1:]):
        chunk_start = max(0, prev_chunk_end - 2)
        inten, rainr = _compute_rain_sparse(
            track.isel(time=slice(chunk_start, chunk_end)), *args,
            max_memory_gb=max_memory_gb, **kwargs,
        )
        intensity.append(inten)
        rainrates.append(rainr)

    intensity = sparse.csr_matrix(sparse.vstack(intensity).max(axis=0))
    if rainrates[0] is not None:
        # eliminate the overlap between consecutive chunks
        rainrates = (
            [rainrates[0][:-1, :]]
            + [rainr[1:-1, :] for rainr in rainrates[1:-1]]
            + [rainrates[-1][1:, :]]
        )
        rainrates = sparse.vstack(rainrates, format="csr")
    return intensity, rainrates

def compute_rain(
    si_track: xr.Dataset,
    centroids: np.ndarray,
    model: int,
    model_kwargs: Optional[dict] = None,
    metric: str = "equirect",
    max_dist_eye_km: float = DEF_MAX_DIST_EYE_KM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rain rate (in mm/h) of the tropical cyclone

    In a first step, centroids within reach of the track are determined so that rain rates will
    only be computed and returned for those centroids. Still, since computing the distance of
    the storm center to the centroids is computationally expensive, make sure to pre-filter the
    centroids and call this function only for those centroids that are potentially affected.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si``. Which data variables are used in the computation of the rain
        rates depends on the selected model.
    centroids : np.ndarray with two dimensions
        Each row is a centroid [lat, lon]. Centroids that are not within reach of the track are
        ignored. Longitudinal coordinates are assumed to be normalized consistently with the
        longitudinal coordinates in ``si_track``.
    model : int
        TC rain model selection according to MODEL_RAIN.
    model_kwargs: dict, optional
        If given, forward these kwargs to the selected model. Default: None
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See ``dist_approx`` function in ``climada.util.coordinates``.
        Default: "equirect".
    max_dist_eye_km : float, optional
        No rain calculation is done for centroids with a distance (in km) to the TC center
        ("eye") larger than this parameter. Default: 300

    Returns
    -------
    rainrates : np.ndarray of shape (npositions, nreachable)
        Rain rates for each track position on those centroids within reach of the TC track.
    idx_centr_reachable : np.ndarray of shape (nreachable,)
        List of indices of input centroids within reach of the TC track.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs

    # start with the assumption that no centroids are within reach
    npositions = si_track.sizes["time"]
    idx_centr_reachable = np.zeros((0,), dtype=np.int64)
    rainrates = np.zeros((npositions, 0), dtype=np.float64)

    # exclude centroids that are too far from or too close to the eye
    d_centr = _centr_distances(si_track, centroids, metric=metric, **model_kwargs)
    mask_centr_close = (d_centr[""] <= max_dist_eye_km * KM_TO_M) & (d_centr[""] > 1)
    if not np.any(mask_centr_close):
        return rainrates, idx_centr_reachable

    # restrict to the centroids that are within reach of any of the positions
    mask_centr_close_any = mask_centr_close.any(axis=0)
    mask_centr_close = mask_centr_close[:, mask_centr_close_any]
    d_centr = {key: d[:, mask_centr_close_any, ...] for key, d in d_centr.items()}
    centroids = centroids[mask_centr_close_any]

    if model == MODEL_RAIN["R-CLIPER"]:
        rainrates = _rcliper(si_track, d_centr[""], mask_centr_close, **model_kwargs)
    elif model == MODEL_RAIN["TCR"]:
        rainrates = _tcr(
            si_track, centroids, d_centr, mask_centr_close, **model_kwargs,
        )
    else:
        raise NotImplementedError
    [idx_centr_reachable] = mask_centr_close_any.nonzero()
    return rainrates, idx_centr_reachable

def _track_to_si_with_q_and_shear(
    track: xr.Dataset,
    metric: str = "equirect",
    q_950: float = 0.01,
    matlab_ref_mode: bool = False,
    **_kwargs,
) -> xr.Dataset:
    """Convert track data to SI units and add Q (humidity) and vshear variables

    If the track data set does not contain the "q950" variable, but "t600", we compute the humidity
    assuming a moist adiabatic lapse rate (see ``_qs_from_t_diff_level``).

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
        "geosphere" (more accurate). See ``dist_approx`` function in ``climada.util.coordinates``.
        Default: "equirect".
    q_950 : float, optional
        If the track data does not include "t600" values, assume this constant value of saturation
        specific humidity (in kg/kg) at 950 hPa. Default: 0.01
    matlab_ref_mode : bool, optional
        Do not apply the changes to the reference MATLAB implementation. Default: False
    _kwargs : dict
        Additional kwargs are ignored.

    Returns
    -------
    xr.Dataset
    """
    si_track = tctrack_to_si(track, metric=metric)

    if "q950" in track.variables:
        si_track["q950"] = track["q950"].copy()
    elif "t600" not in track.variables:
        si_track["q950"] = ("time", np.full_like(si_track["lat"].values, q_950))
    else:
        # Note that we follow the MATLAB reference in computing Q at 950 hPa as opposed to the
        # pressure level used in Lu et al. 2018 (900 hPa)
        pres_in = 600
        pres_out = 950
        si_track["q950"] = ("time", _qs_from_t_diff_level(
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

        # v_drift (or v_beta) is set to be a 1.4 m/s drift in meridional direction (away from the
        # equator), because that's the value used in the proprietary synthetic track generator by
        # WindRiskTech. Note, however, that a value of 2.5 m/s seems to be more common in the
        # literature (e.g. Emanuel et al. 2006 or Lee et al. 2018).
        v_beta_lat = 1.4
        si_track["vdrift"] = xr.zeros_like(si_track["v850"])
        si_track["vdrift"].values[:, 0] = (
            v_beta_lat
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
    **_kwargs,
) -> dict:
    """Compute distances of centroids to storm locations required for ``_compute_vertical_velocity``

    In addition to the distances to the centroids, the distances to staggered centroid locations,
    as well as the unit vectors pointing from the storm center to each centroid are returned.

    Parameters
    ----------
    si_track : xr.Dataset
        TC track data in SI units, see ``tctrack_to_si``.
    centroids : ndarray
        Each row is a pair of lat/lon coordinates.
    metric : str, optional
        Approximation method to use for earth distances: "equirect" (faster) or "geosphere" (more
        accurate). See ``dist_approx`` function in ``climada.util.coordinates``.
        Default: "equirect".
    res_radial_m : float, optional
        Spatial resolution (in m) in radial direction. Default: 2000
    _kwargs : dict
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
    mask_centr_close: np.ndarray,
) -> np.ndarray:
    """Compute rain rate (in mm/h) from maximum wind speeds using the R-CLIPER model

    The model is defined in equations (3)-(5) and Table 2 (NHC) in the following publication:

    Tuleya et al. (2007): Evaluation of GFDL and Simple Statistical Model Rainfall Forecasts for
    U.S. Landfalling Tropical Storms. Weather and Forecasting 22(1): 56–70.
    https://doi.org/10.1175/WAF972.1

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si``. Only the "vmax" data variable is used.
    d_centr : np.ndarray of shape (npositions, ncentroids)
        Distance (in m) between centroids and track positions.
    mask_centr_close : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.

    Returns
    -------
    ndarray of shape (npositions, ncentroids)
    """
    rainrate = np.zeros_like(d_centr)
    d_centr, v_max = [
        ar[mask_centr_close] for ar in np.broadcast_arrays(
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

    rainrate[mask_centr_close] = rainrate_close
    return rainrate

def _tcr(
    si_track: xr.Dataset,
    centroids: np.ndarray,
    d_centr: dict,
    mask_centr_close: np.ndarray,
    e_precip: float = 0.5,
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
        Output of ``tctrack_to_si``. Which data variables are used in the computation of the rain
        rates depends on the selected wind model.
    centroids : ndarray of shape (ncentroids, 2)
        Each row is a pair of lat/lon coordinates.
    d_centr : dict
        Output of ``_centr_distances``.
    mask_centr_close : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.
    e_precip : float, optional
        Precipitation efficiency (unitless), the fraction of the vapor flux falling to the surface
        as rainfall (Lu et al. 2018, eq. (14)). Note that we follow the MATLAB reference
        implementation and use 0.5 as a default value instead of the 0.9 that was proposed in
        Lu et al. 2018. Default: 0.5
    kwargs :
        The remaining arguments are passed on to _compute_vertical_velocity.

    Returns
    -------
    ndarray of shape (npositions, ncentroids)
    """
    # w is of shape (ntime, ncentroids)
    w = _compute_vertical_velocity(si_track, centroids, d_centr, mask_centr_close, **kwargs)

    # derive vertical vapor flux wq by multiplying with saturation specific humidity Q950
    wq = si_track["q950"].values[:, None] * w

    # convert rainrate from "meters per second" to "milimeters per hour"
    rainrate = (M_TO_MM * H_TO_S) * e_precip * RHO_A_OVER_RHO_L * wq

    return rainrate

def _compute_vertical_velocity(
    si_track: xr.Dataset,
    centroids: np.ndarray,
    d_centr: dict,
    mask_centr_close: np.ndarray,
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
        Output of ``tctrack_to_si``. Which data variables are used depends on the wind model.
    centroids : ndarray of shape (ncentroids, 2)
        Each row is a pair of lat/lon coordinates.
    d_centr : ndarray of shape (npositions, ncentroids)
        Distances from storm centers to centroids.
    mask_centr_close : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.
    wind_model : str, optional
        Parametric wind field model to use, see TropCyclone. Default: "ER11".
    elevation_tif : Path or str, optional
        Path to a GeoTIFF file containing digital elevation model data (in m). If not specified, an
        SRTM-based topography at 0.1 degree resolution provided with CLIMADA is used. Default: None
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
        Do not apply the changes to the reference MATLAB implementation. Default: False

    Returns
    -------
    ndarray of shape (ntime, ncentroids)
    """
    h_winds = _horizontal_winds(
        si_track,
        d_centr,
        mask_centr_close,
        MODEL_VANG[wind_model],
        matlab_ref_mode=matlab_ref_mode,
    )

    # Currently, the `mask_centr_close` is ignored in the computation of the components, but it is
    # applied only afterwards. This is because it seems like the code readability would suffer a
    # lot from this. However, this might be one aspect where computational performance can be
    # improved in the future.
    w = np.zeros_like(d_centr[""])

    w_f_plus_w_t = _w_frict_stretch(
        si_track, d_centr, h_winds, centroids,
        res_radial_m=res_radial_m, c_drag_tif=c_drag_tif, min_c_drag=min_c_drag,
    )[mask_centr_close]

    w_h = _w_topo(
        si_track, d_centr, h_winds, centroids, elevation_tif=elevation_tif,
    )[mask_centr_close]

    w_s = _w_shear(si_track, d_centr, h_winds, res_radial_m=res_radial_m)[mask_centr_close]

    w[mask_centr_close] = np.fmax(np.fmin(w_f_plus_w_t + w_h + w_s, max_w_foreground) - w_rad, 0)
    return w

def _horizontal_winds(
    si_track: xr.Dataset,
    d_centr: dict,
    mask_centr_close: np.ndarray,
    model: int,
    matlab_ref_mode: bool = False,
) -> dict:
    """Compute all horizontal wind speed variables required for ``_compute_vertical_velocity``

    Wind speeds are not only computed on the given centroids and for the given times, but also at
    staggered locations for further use in finite difference computations.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si``. Which data variables are used depends on the wind model.
    d_centr : dict
        Output of ``_centr_distances``.
    mask_centr_close : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.
    model : int
        Wind profile model selection according to MODEL_VANG.
    matlab_ref_mode : bool, optional
        Do not apply the changes to the reference MATLAB implementation. Default: False

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
            si_track, d_centr[""], mask_centr_close, model,
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
                    si_track, d_centr[rstep], mask_centr_close, model,
                    cyclostrophic=False, matlab_ref_mode=matlab_ref_mode,
                )
            else:
                # NOTE: For the computation of time derivatives, the eye of the storm is held
                #       fixed while only the wind profile varies (see MATLAB code)
                sl = slice(2, None) if tstep == "+" else slice(None, -2)
                result[1:-1, :] = _windprofile(
                    si_track.isel(time=sl), d_centr[rstep][1:-1], mask_centr_close[1:-1], model,
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
    mask_centr_close: np.ndarray,
    model: int,
    cyclostrophic: bool = False,
    matlab_ref_mode: bool = False,
) -> np.ndarray:
    """Compute (absolute) angular wind speeds according to a parametric wind profile

    Wrapper around ``compute_angular_windspeeds`` (from climada.trop_cyclone) that adjusts the
    Coriolis parameter if matlab_ref_mode is True.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si``. Which data variables are used depends on the wind model.
    d_centr : ndarray of shape (npositions, ncentroids)
        Distances from storm centers to centroids.
    mask_centr_close : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.
    model : int
        Wind profile model selection according to MODEL_VANG.
    cyclostrophic : bool, optional
        If True, don't apply the influence of the Coriolis force (set the Coriolis terms to 0).
        Default: False
    matlab_ref_mode : bool, optional
        Do not apply the changes to the reference MATLAB implementation. Default: False

    Returns
    -------
    ndarray of shape (npositions, ncentroids)
    """
    if matlab_ref_mode:
        # In the MATLAB implementation, the Coriolis parameter is chosen to be 5e-5 (independent of
        # latitude), following formula (2) and the remark in Section 3 of Emanuel & Rotunno (2011).
        si_track = si_track.copy(deep=True)
        si_track["cp"].values[:] = 5e-5
    return compute_angular_windspeeds(
        si_track, d_centr, mask_centr_close, model, cyclostrophic=cyclostrophic,
    )

def _w_shear(
    si_track: xr.Dataset,
    d_centr: dict,
    h_winds: dict,
    res_radial_m: float = 2000.0,
) -> np.ndarray:
    """Compute the shear component of the vertical wind velocity

    This implements eq. (12) from:

    Lu et al. (2018): Assessing Hurricane Rainfall Mechanisms Using a Physics-Based Model:
    Hurricanes Isabel (2003) and Irene (2011). Journal of the Atmospheric
    Sciences 75(7): 2337–2358. https://doi.org/10.1175/JAS-D-17-0264.1

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si``. The data variables used by this function are "cp" and "vshear".
        If the "vshear" variable is not available, the result is 0 everywhere.
    d_centr : dict
        Output of ``_centr_distances``.
    h_winds : dict
        Output of ``_horizontal_winds``.
    res_radial_m : float, optional
        Spatial resolution (in m) in radial direction. Default: 2000

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
    # While Lu et al. 2018 assume a factor of 0.5, we follow the MATLAB reference implementation
    # and use a factor of 1.0 because the precipitation efficiency (ep) is higher (0.75 instead
    # of 0.5) in the TC than in the normal tropical environment.
    fac_scalar = 1.0

    fac = fac_scalar * (
        si_track["cp"].values[:, None]
        + h_winds["r,t"] / (1 + d_centr[""])
        + (h_winds["r+,t"] - h_winds["r-,t"]) / (2.0 * res_radial_m)
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
) -> np.ndarray:
    """Compute the topographic component w_h of the vertical wind velocity

    This implements eq. (7) from:

    Lu et al. (2018): Assessing Hurricane Rainfall Mechanisms Using a Physics-Based Model:
    Hurricanes Isabel (2003) and Irene (2011). Journal of the Atmospheric
    Sciences 75(7): 2337–2358. https://doi.org/10.1175/JAS-D-17-0264.1

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si``. Only the "vtrans" data variable is used by this function.
    d_centr : dict
        Output of ``_centr_distances``.
    h_winds : dict
        Output of ``_horizontal_winds``.
    centroids : ndarray
        Each row is a pair of lat/lon coordinates.
    elevation_tif : Path or str, optional
        Path to a GeoTIFF file containing digital elevation model data (in m). If not specified, an
        SRTM-based topography at 0.1 degree resolution provided with CLIMADA is used. Default: None

    Returns
    -------
    ndarray of shape (ntime, ncentroids)
    """
    if elevation_tif is None:
        elevation_tif = default_elevation_tif()

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
) -> np.ndarray:
    """Compute the sum of the frictional and stretching components w_f and w_t

    This implements eq. (6) and (11) from:

    Lu et al. (2018): Assessing Hurricane Rainfall Mechanisms Using a Physics-Based Model:
    Hurricanes Isabel (2003) and Irene (2011). Journal of the Atmospheric
    Sciences 75(7): 2337–2358. https://doi.org/10.1175/JAS-D-17-0264.1

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si``. The data variables used by this function are "cp", "rad",
        "tstep" and "vtrans".
    d_centr : dict
        Output of ``_centr_distances``.
    h_winds : dict
        Output of ``_horizontal_winds``.
    centroids : ndarray
        Each row is a pair of lat/lon coordinates.
    res_radial_m : float, optional
        Spatial resolution (in m) in radial direction. Default: 2000
    c_drag_tif : Path or str, optional
        Path to a GeoTIFF file containing gridded drag coefficients (bottom friction). If not
        specified, an ERA5-based data set provided with CLIMADA is used. Default: None
    min_c_drag : float, optional
        The drag coefficient is clipped to this minimum value (esp. over ocean). Default: 0.001

    Returns
    -------
    ndarray of shape (ntime, ncentroids)
    """
    # sum of frictional and stretching components w_f and w_t
    if c_drag_tif is None:
        c_drag_tif = default_drag_tif()

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
    When computing Q from T on the same pressure level, see ``_r_from_t_same_level`` instead since
    Q = r / (1 + r) for the mixing ratio r.

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
    ``_r_from_t_same_level``), we can use this relationship to compute Q at one pressure level from
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
        Do not apply the changes to the reference MATLAB implementation. Default: False

    Returns
    -------
    q_out : ndarray
        For each temperature value in temps_in, a value of saturation specific humidity (in kg/kg)
        at the pressure level pres_out.
    """
    # c_vmax : rescale factor from (squared) surface to (squared) gradient winds
    #          In the MATLAB implementation, c_vmax=1.6 is used, but this is almost the same as the
    #          value used here (0.8**-2).
    c_vmax = 1.6 if matlab_ref_mode else GRADIENT_LEVEL_TO_SURFACE_WINDS**-2

    # In the MATLAB implementation, the "Bolton1980" coefficients, and an approximative form of the
    # derivative are used in the computation of the mixing ratio.
    r_from_t_kwargs = dict(
        tetens_coeffs="Bolton1980" if matlab_ref_mode else "Buck1981",
        use_cc_derivative=matlab_ref_mode,
    )

    # first, calculate mixing ratio r_in from temps_in
    r_in, _ = _r_from_t_same_level(pres_in, np.fmax(T_ICE_K - 50, temps_in), **r_from_t_kwargs)

    # derive (temps_out, r_out) from (temps_in, r_in) iteratively (Newton-Raphson method)
    r_out = np.zeros_like(r_in)
    temps_out = temps_in.copy() + 20  # first guess, assuming that pres_out > pres_in

    # exclude missing data (fill values) in the inputs
    mask = (temps_in > 100)

    # s : Total entropy, which is conserved across pressure levels when assuming a moist adiabatic
    #     lapse rate. The additional vmax-term is a correction to account for the fact that the
    #     eyewall is warmer than the environment at 600 hPa (thermal wind balance). For a reference
    #     of the vmax-term, see, e.g., the considerations in the following article:
    #
    #         Emanuel, K. (1986): An Air-Sea Interaction Theory for Tropical Cyclones. Part I:
    #         Steady-State Maintenance. Journal of the Atmospheric Sciences 43(6): 585–605.
    #         https://doi.org/10.1175/1520-0469(1986)043<0585:AASITF>2.0.CO;2
    #
    #     Compared to eq. (34), the last term is neglected due to V >> fr. The boundary layer
    #     theta_e under the eyewall is equated with that of the far environment, and the saturation
    #     theta_e of the eyewall (which is constant along an M surface) is equated with the
    #     saturation theta_e of the sea surface.
    s_in = (
        cap_heat_air * np.log(temps_in[mask])
        + L_EVAP_WATER * r_in[mask] / temps_in[mask]
        - R_DRY_AIR * np.log(pres_in)
        + c_vmax * vmax[mask]**2 / DELTA_T_TROPOPAUSE
    )

    # solve `s_out(T_out) - s_in = 0` using the Newton-Raphson method
    for it in range(max_iter):
        # compute new estimate of r_out from current estimate of T_out
        r_out[mask], drdT = _r_from_t_same_level(
            pres_out, temps_out[mask], gradient=True, **r_from_t_kwargs,
        )
        s_out = (
            cap_heat_air * np.log(temps_out[mask])
            + L_EVAP_WATER * r_out[mask] / temps_out[mask]
            - R_DRY_AIR * np.log(pres_out)
        )
        dsdT = (
            cap_heat_air * temps_out[mask]
            + L_EVAP_WATER * (drdT * temps_out[mask] - r_out[mask])
        ) / temps_out[mask]**2

        # take Newton step
        temps_out[mask] -= (s_out - s_in) / dsdT

    if matlab_ref_mode:
        # In the MATLAB implementation, this function does actually return the mixing ratio which
        # is almost the same as the "specific humidity" in practice.
        q_out = r_out
    else:
        q_out = r_out / (1 + r_out)
    return q_out

def _r_from_t_same_level(
    p_ref: float,
    temps: np.ndarray,
    gradient: bool = False,
    tetens_coeffs: str = "Buck1981",
    use_cc_derivative: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute the mixing ratio from temperature at a given pressure level

    The physical background is the Clausius-Clapeyron equation for water vapor pressure under
    typical atmospheric conditions, but since this differential equation does not have an explicit
    solution, the implementation uses the Tetens (or August-Roche-Magnus) approximation formula
    with coefficients given in:

    Murray (1967): On the Computation of Saturation Vapor Pressure. Journal of Applied Meteorology
    and Climatology 6(1): 203–204. http://doi.org/10.1175/1520-0450(1967)006<0203:OTCOSV>2.0.CO;2

    Bolton (1980): The Computation of Equivalent Potential Temperature. Monthly Weather Review
    108(7): 1046–1053. https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2

    Buck (1981): New Equations for Computing Vapor Pressure and Enhancement Factor. Journal of
    Applied Meteorology and Climatology 20(12): 1527-1532.
    https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2

    Alduchov and Eskridge (1996): Improved Magnus Form Approximation of Saturation Vapor Pressure.
    Journal of Applied Meteorology and Climatology 35(4): 601–609.
    https://doi.org/10.1175/1520-0450(1996)035<0601:imfaos>2.0.co;2

    The default coefficients (Buck 1981) are also used by the ECMWF, see Section 7.2.1 (b) of the
    following document:

    ECMWF (2023): IFS Documentation CY48R1, Part IV: Physical Processes.
    https://www.ecmwf.int/en/elibrary/81370-ifs-documentation-cy48r1-part-iv-physical-processes

    Parameters
    ----------
    p_ref : float
        Reference pressure level (in hPa) at which the input temperatures are given and at which
        output mixing ratio values are computed.
    temps : ndarray
        Temperatures (in K) at the pressure level p_ref.
    gradient : bool, optional
        If True, compute the derivative of the functional relationship between Q and T.
    tetens_coeffs : str, optional
        Coefficients to use for the Tetens formula. One of "Alduchov1996", "Buck1981",
        "Bolton1980", or "Murray1967". Default: "Buck1981"
    use_cc_derivative : bool, optional
        Instead of the actual derivative, use an approximation of the derivative (that is used in
        the MATLAB code) based on the original Clausius-Clapeyron equation for water vapor under
        typical atmospheric conditions:

          des/dT = (Lv * es) / (Rv * T**2)

        The approximation is used in the MATLAB reference implementation under the assumption that
        it is only used in the Newton-Raphson iteration where little errors do not matter.
        Default: False

    Returns
    -------
    r : ndarray
        For each temperature value in temp, a value of saturation specific humidity (in kg/kg).
    drdT : ndarray
        If ``gradient`` is False, this is None. Otherwise, the derivative of Q with respect to T is
        returned.
    """
    try:
        a, b, c = {
            "Murray1967": (17.2693882, 35.86, 6.1078),
            "Bolton1980": (17.67, 29.65, 6.112),
            "Buck1981": (17.502, 32.19, 6.1121),
            "Alduchov1996": (17.625, 30.12, 6.1094),
        }[tetens_coeffs]
    except KeyError as err:
        raise ValueError(f"Unknown Tetens coefficients: {tetens_coeffs}") from err

    # es : saturation vapor pressure (in hPa)
    es = c * np.exp(a * (temps - T_ICE_K) / (temps - b))

    fact = M_WATER / M_DRY_AIR
    r_mix = fact * es / (p_ref - es)

    drdT = None
    if gradient:
        if use_cc_derivative:
            # Specific gas constant of water vapor (in J / kgK)
            r_water = 1000 * R_GAS / M_WATER
            drdT = (L_EVAP_WATER / r_water) / temps**2 * r_mix
            # Note that the full C-C formula including the deriative of r_mix with respect to es
            # (as in dr/dT = dr/des * des/dT) would be as follows:
            # drdT = (L_EVAP_WATER / r_water) / temps**2 * r_mix * (1 + r_mix / fact)
        else:
            drdT = a * (T_ICE_K - b) / (temps - b)**2 * r_mix * (1 + r_mix / fact)

    return r_mix, drdT
