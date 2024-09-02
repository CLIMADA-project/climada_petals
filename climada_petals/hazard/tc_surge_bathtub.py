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

Define TCSurgeBathtub class.
"""

__all__ = ['TCSurgeBathtub']

import copy
import logging
import warnings

import numpy as np
from scipy import sparse
from scipy.spatial import KDTree, QhullError
from tqdm import tqdm
import rasterio.warp

from climada.hazard.base import Hazard
from climada.hazard import Centroids
import climada.util.coordinates as u_coord
from scipy.interpolate import griddata

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'TCSurgeBathtub'
"""Hazard type acronym for this module"""

MAX_DIST_COAST = 50
"""Maximum inland distance of the centroids in km."""

MAX_ELEVATION = 10
"""Maximum elevation of the centroids in m."""

MAX_LATITUDE = 61
"""Maximum latitude of potentially affected centroids."""


class TCSurgeBathtub(Hazard):
    """TC surge heights in m, a bathtub model with wind-surge relationship and inland decay."""

    def __init__(self):
        Hazard.__init__(self, HAZ_TYPE)


    @staticmethod
    def from_tc_winds(wind_haz, topo_path, inland_decay_rate=0.2, add_sea_level_rise=0.0, sea_level_rise_gdf=None, higher_res=None):
        """Compute tropical cyclone surge from input winds.

        Parameters
        ----------
        wind_haz : TropCyclone
            Tropical cyclone wind hazard object.
        topo_path : str
            Path to a raster file containing gridded elevation data.
        inland_decay_rate : float, optional
            Decay rate of surge when moving inland in meters per km. Set to 0 to deactivate
            this effect. The default value of 0.2 is taken from Section 5.2.1 of the monograph
            Pielke and Pielke (1997): Hurricanes: their nature and impacts on society.
            https://rogerpielkejr.com/2016/10/10/hurricanes-their-nature-and-impacts-on-society/
        add_sea_level_rise : float, optional
            Sea level rise effect in meters to be added to surge height.
        """
        centroids = copy.deepcopy(wind_haz.centroids)
        intensity = copy.deepcopy(wind_haz.intensity)

        if not centroids.coord.size:
            centroids.set_meta_to_lat_lon()

        coastal_msk = _calc_coastal_mask(centroids, intensity)
        centroids, intensity = centroids.select(sel_cen=coastal_msk), intensity[:,coastal_msk]
        if intensity.size == 0:
            haz = TCSurgeBathtub()
            haz.centroids = centroids
            haz.units = 'm'
            haz.event_id = wind_haz.event_id
            haz.event_name = wind_haz.event_name
            haz.date = wind_haz.date
            haz.orig = wind_haz.orig
            haz.frequency = wind_haz.frequency
            haz.intensity = intensity
            haz.fraction = None
            return haz

        if higher_res is not None:
            centroids, intensity = _downscale_sparse_matrix(intensity, centroids, higher_res)
            coastal_msk = _calc_coastal_mask(centroids, intensity)

        # Load elevation at coastal centroids
        coastal_centroids_h = u_coord.read_raster_sample(
            topo_path, centroids.lat[coastal_msk], centroids.lon[coastal_msk])

        if sea_level_rise_gdf is not None:
            gdf_coords = np.array(list(zip(sea_level_rise_gdf.geometry.y, sea_level_rise_gdf.geometry.x)))
            sea_level_rise_values = sea_level_rise_gdf['sea_level_change'].values / 1000
            kdtree = KDTree(gdf_coords)
            _, idx = kdtree.query(centroids.coord[coastal_msk])
            add_sea_level_rise = sea_level_rise_values[idx]

        # Update selected coastal centroids to exclude high-lying locations
        # We only update the previously selected centroids (for which elevation info was obtained)
        elevation_msk = (coastal_centroids_h >= 0)
        elevation_msk &= (coastal_centroids_h - add_sea_level_rise <= MAX_ELEVATION)
        coastal_msk[coastal_msk] = elevation_msk

        # Elevation data and coastal/non-coastal indices are used later in the code
        coastal_centroids_h = coastal_centroids_h[elevation_msk]
        coastal_idx = coastal_msk.nonzero()[0]

        # to each centroid, assign its position in `coastal_idx`, and assign an out-of-bounds
        # index to all centroids that are not contained in `coastal_idx`
        cent_to_coastal_idx = np.full(coastal_msk.shape, coastal_idx.size, dtype=np.int64)
        cent_to_coastal_idx[coastal_msk] = np.arange(coastal_idx.size)

        # Initialize intensity array at coastal centroids
        inten_surge = intensity.copy()
        inten_surge.data[~coastal_msk[inten_surge.indices]] = 0
        inten_surge.eliminate_zeros()

        # Conversion of wind to surge using the linear wind-surge relationship from
        # figure 2 of the following paper:
        #
        #   Xu, Liming (2010): A Simple Coastline Storm Surge Model Based on Pre-run SLOSH Outputs.
        #   In: 29th Conference on Hurricanes and Tropical Meteorology, 10–14 May. Tucson, Arizona.
        #   https://ams.confex.com/ams/pdfpapers/168806.pdf
        inten_surge.data = 0.1023 * np.fmax(inten_surge.data - 26.8224, 0) + 1.8288

        if inland_decay_rate != 0:
            # Add decay according to distance from coast
            dist_coast_km = np.abs(centroids.get_dist_coast()[coastal_idx]) / 1000
            coastal_centroids_h += inland_decay_rate * dist_coast_km
        if isinstance(add_sea_level_rise, np.ndarray):
            coastal_centroids_h -= add_sea_level_rise[elevation_msk]
        else:
            coastal_centroids_h -= add_sea_level_rise

        # Efficient way to subtract from selected columns of sparse csr matrix
        inten_surge.data -= coastal_centroids_h[cent_to_coastal_idx[inten_surge.indices]]

        # Discard negative (invalid/unphysical) surge height values
        inten_surge.data = np.fmax(inten_surge.data, 0)
        inten_surge.eliminate_zeros()

        # Get fraction of (large) centroid cells on land according to the given (high-res) DEM
        # only store the result in locations with non-zero surge height
        # fract_surge = inten_surge.copy()
        # this will probably not work
        # fract_surge.data[:] = _fraction_on_land(centroids, topo_path)[fract_surge.indices]

        # Set other attributes
        haz = TCSurgeBathtub()
        haz.centroids = centroids
        haz.units = 'm'
        haz.event_id = wind_haz.event_id
        haz.event_name = wind_haz.event_name
        haz.date = wind_haz.date
        haz.orig = wind_haz.orig
        haz.frequency = wind_haz.frequency
        haz.intensity = inten_surge
        # this has to be done too
        haz.fraction = None
        return haz

def _calc_coastal_mask(centroids, intensity):
    """Calculate a coastal mask to identify centroids affected by wind within a specified distance from the coast.

    This function determines which centroids are affected by wind intensity and are located within a maximum distance
    from the coast and within a specified latitude range. The centroids are filtered based on the following conditions:
    - They are within the maximum distance from the coast (defined by `MAX_DIST_COAST`).
    - They have an absolute latitude less than or equal to `MAX_LATITUDE`.
    - They have positive wind intensity values.

    Parameters
    ----------
    centroids : Centroids
        Centroids to select from.

    intensity : numpy.ndarray
        A 2D numpy array where each row corresponds to a time point and each column corresponds to a centroid.
        The array contains wind intensity values. A positive intensity value indicates wind-affected centroids.

    Returns
    -------
    numpy.ndarray
        A 1D boolean numpy array (mask) where `True` indicates centroids that are:
        - Affected by wind (positive intensity).
        - Within `MAX_DIST_COAST` kilometers from the coast.
        - Within the latitude range of ±`MAX_LATITUDE` degrees.

    """
    coastal_msk = (intensity > 0).sum(axis=0).A1 > 0
    coastal_msk &= (centroids.get_dist_coast(signed=True) < 0)
    coastal_msk &= (centroids.get_dist_coast(signed=True) >= -MAX_DIST_COAST * 1000)
    coastal_msk &= (np.abs(centroids.lat) <= MAX_LATITUDE)
    return coastal_msk


def _calc_extent(coords):
    """Calculate the minimum and maximum values of latitudes and longitudes from given coordinates.

    Parameters
    ----------
    coords : numpy.ndarray
        2D array where each row contains latitude and longitude coordinates in degrees.

    Returns
    -------
    min_lat : float
        Minimum latitude value in degrees.
    max_lat : float
        Maximum latitude value in degrees.
    min_lon : float
        Minimum longitude value in degrees.
    max_lon : float
        Maximum longitude value in degrees.

    Notes
    -----
    This function assumes that `coords` is a 2D numpy array with latitude in the first column (index 0)
    and longitude in the second column (index 1).

    Examples
    --------
    >>> coords = np.array([[0, 0], [90, 180], [-90, -180]])
    >>> _calc_extent(coords)
    (-90.0, 90.0, -180.0, 180.0)

    """
    latitudes = coords[:, 0]
    longitudes = coords[:, 1]

    # Calculate the bounds
    min_lat = np.min(latitudes)
    max_lat = np.max(latitudes)
    min_lon = np.min(longitudes)
    max_lon = np.max(longitudes)
    return min_lat, max_lat, min_lon, max_lon



def _create_hr_coordinates(coords, resolution):
    """
    Generate a high-resolution grid of coordinates based on the given bounds and resolution.

    This function calculates the bounding box for the provided coordinates and then
    creates a high-resolution grid within these bounds using the specified resolution.

    Parameters
    ----------
    coords : numpy.ndarray
        A 2D numpy array of shape (N, 2) containing latitude and longitude coordinates.
    resolution : float
        The resolution for the high-resolution grid. This specifies the step size for
        generating the grid points.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array of shape (M, 2) containing the generated high-resolution latitude and longitude coordinates.

    Notes
    -----

    This function uses numpy.meshgrid to first generate the dense grid of coordinates which can quickly become
    huge and long to create for large total extent of `coords` and "too" precise resolution.
    """
    min_lat, max_lat, min_lon, max_lon = _calc_extent(coords)

    lat = np.arange(min_lat, max_lat + resolution, resolution)
    lon = np.arange(min_lon, max_lon + resolution, resolution)

    # Create a meshgrid of latitude and longitude values
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

def _match_coords(hr_coords, lr_coords, hr_res, tree=None):
    """
    Match high-resolution coordinates to low-resolution coordinates based on proximity.

    Parameters
    ----------
    hr_coords : numpy.ndarray
        High-resolution coordinates stored in a 2D array where each row represents a point with latitude and longitude.

    lr_coords : numpy.ndarray
        Low-resolution coordinates stored in a 2D array where each row represents a point with latitude and longitude.

    hr_res : float
        Resolution of the high-resolution coordinates, used to determine proximity threshold for matching.

    tree : scipy.spatial.KDTree, optional
        KDTree built on `hr_coords` for efficient nearest neighbor search. If not provided, it will be constructed
        using `hr_coords`.

    Returns
    -------
    numpy.ndarray
        Selected high-resolution coordinates that match closely with the low-resolution coordinates, ensuring no duplicates.

    Notes
    -----
    Matches low-resolution coordinates to high-resolution coordinates are within a distance threshold
    calculated as 4 times the `hr_res`.
    """
    if tree is None:
        tree = KDTree(hr_coords)

    threshold = 4 * hr_res
    distances, indices = tree.query(lr_coords, k=100, distance_upper_bound=threshold)
    within_threshold = distances <= threshold
    selected_coords = hr_coords[indices[within_threshold]]
    _,unique_indices = np.unique(selected_coords, axis=0, return_index=True)
    unique_indices.sort()
    return selected_coords[unique_indices]

def _downscale_coordinates(lowres_coords, higher_res, bins_res=5.0):
    """
    Downscale low-resolution coordinates to higher resolution within defined grid cells.

    Parameters
    ----------
    lowres_coords : numpy.ndarray
        Low-resolution coordinates stored in a 2D array where each row represents a point with latitude and longitude.

    higher_res : float
        Resolution to which the coordinates will be downscaled, in degrees.

    bins_res : float, optional
        Resolution of the binning grid cells in degrees. Default is 5.0 degrees.

    Returns
    -------
    numpy.ndarray
        High-resolution coordinates downscaled from the low-resolution input, ensuring no duplicates.

    Notes
    -----
    This function divides the extent of `lowres_coords` into grid cells defined by `bins_res`. For each grid cell,
    it identifies low-resolution coordinates that fall within or near the cell boundaries using a KDTree. It then
    creates high-resolution coordinates by refining the points within each cell and matches these to the original
    low-resolution coordinates to ensure uniqueness.
    """

    # could take a tuple instead of a single value in the future
    lat_resolution = bins_res  # degrees
    lon_resolution = bins_res  # degrees
    lat_min, lat_max, lon_min, lon_max = _calc_extent(lowres_coords)
    lat_bins = np.arange(lat_min, lat_max, lat_resolution)
    lon_bins = np.arange(lon_min, lon_max, lon_resolution)

    bins_with_data = []
    highres_coords = []
    tree = KDTree(lowres_coords)
    for lat in lat_bins:
        for lon in lon_bins:
            # Define the current grid cell (subpart)
            cell_min = [lat, lon]
            cell_max = [lat + lat_resolution, lon + lon_resolution]

            # Check if any data points fall within the current grid cell
            # Using the KDTree to find points within the cell
            indices = tree.query_ball_point(cell_min, lat_resolution * np.sqrt(2), p=2.0)
            cell_coords = lowres_coords[indices]

            # Filter points within the exact cell boundaries
            within_cell = [point for point in cell_coords
                           if cell_min[0] <= point[0] < cell_max[0]
                           and cell_min[1] <= point[1] < cell_max[1]]

            if within_cell:
                bins_with_data.append((cell_min, cell_max))
                highres_coords.append(_create_hr_coordinates(cell_coords, higher_res))

    highres_coords = np.vstack(highres_coords)
    highres_coords = _match_coords(highres_coords, lowres_coords, higher_res)
    return highres_coords


def _downscale_sparse_matrix(matrix, centroids, higher_res, method="linear"):
    """Downscale a sparse matrix of intensities to a higher resolution grid.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Sparse matrix where each row represents intensity values associated with low-resolution coordinates.

    centroids : Centroids
        Centroids of low-resolution coordinates associated with the sparse matrix.

    higher_res : float
        Resolution to which the coordinates will be downscaled, in degrees.

    method : str, optional
        Method to use for interpolation of intensities. Default is "linear".
        Other options include "nearest" and "cubic". See `scipy.interpolate.griddata` for details.

    Returns
    -------
    Centroids
        Centroids object containing high-resolution coordinates derived from the downscaled grid.

    scipy.sparse.csr_matrix
        Sparse matrix of downscaled values corresponding to the high-resolution grid.

    Notes
    -----
    This function first downscales the low-resolution coordinates (`centroids.coord`) to a higher resolution grid
    using `_downscale_coordinates`. It then interpolates values from the sparse matrix `matrix` to this
    high-resolution grid using the specified `method`. The resulting values are stored in a sparse matrix format.
    """

    intensities = []
    lowres_coords = centroids.coord
    hr_coordinates_full = _downscale_coordinates(lowres_coords, higher_res)
    for i in range(matrix.shape[0]):
        # Need at least 3 points to interpolates
        if matrix[i].size > 3:
            # In some rare case, all points are aligned, current solution is to ignore TC (far from coast anyway)
            if _is_a_line(lowres_coords[matrix[i].indices]):
                warnings.warn(
                    f"Coordinates are aligned, interpolation is not possible, will ignore this event (TC {i+1}, with {matrix[i].size} non-zero intensity after filter)."
                )
                intensities.append(
                    sparse.csr_matrix([], shape=(1, hr_coordinates_full[:, 0].size))
                )
            else:
                try:
                    values = matrix[i].data
                    new_matrix = griddata(
                        lowres_coords[matrix[i].indices],
                        values,
                        hr_coordinates_full,
                        method=method,
                        fill_value=0,
                    )
                    intensities.append(
                        sparse.csr_matrix(new_matrix, shape=(1, hr_coordinates_full[:, 0].size))
                    )
                except QhullError as qhullerr:
                    warnings.warn(
                        f"Scipy could not compute the Qhull for this event. Ignoring.\nThe event (TC {i+1}) had {matrix[i].size} non zero intensity centroids.\nHere is the error:\n======={qhullerr}\n======="
                    )
        else:
            intensities.append(
                sparse.csr_matrix([], shape=(1, hr_coordinates_full[:, 0].size))
            )

    new_centroids = Centroids.from_lat_lon(
        hr_coordinates_full[:, 0], hr_coordinates_full[:, 1]
    )
    new_intensity = sparse.vstack(intensities)
    return new_centroids, new_intensity


def _is_a_line(coords):
    """Determine whether coordinates are aligned. Used to check interpolation is possible."""
    return (
        np.all(coords[:, 0] == coords[0, 0]) or
        np.all(coords[:, 1] == coords[0, 1])
    )

def _fraction_on_land(centroids, topo_path):
    """Determine fraction of each centroid cell that is on land.

    Typically, the resolution of the provided DEM data set is much higher than the resolution
    of the centroids so that the centroid cells might be partly above and partly below sea level.
    This function computes for each centroid cell the fraction of its area that is on land.

    Parameters
    ----------
    centroids : Centroids
        Centroids to consider
    topo_path : str
        Path to a raster file containing gridded elevation data.

    Returns
    -------
    fractions : ndarray of shape (ncentroids,)
        For each centroid, the fraction of it's cell area that is on land according to the DEM.
    """
    bounds = np.array(centroids.total_bounds)
    shape = [0, 0]
    shape[0], shape[1], cen_trans = u_coord.pts_to_raster_meta(
        points_bounds=bounds,
        res=min(u_coord.get_resolution(centroids.lat, centroids.lon)))

    read_raster_buffer = 0.5 * max(np.abs(cen_trans[0]), np.abs(cen_trans[4]))
    bounds += read_raster_buffer * np.array([-1., -1., 1., 1.])
    on_land, dem_trans = u_coord.read_raster_bounds(topo_path, bounds, resampling="bilinear")
    on_land = (on_land > 0).astype(np.float64)

    with rasterio.open(topo_path, 'r') as src:
        dem_crs = src.crs
        dem_nodata = src.nodata

    fractions = np.zeros(shape, dtype=np.float64)
    rasterio.warp.reproject(source=on_land, destination=fractions,
                            src_transform=dem_trans, src_crs=dem_crs,
                            dst_transform=cen_trans, dst_crs=centroids.crs,
                            resampling=rasterio.warp.Resampling.average,
                            src_nodata=dem_nodata, dst_nodata=0.0)

    return fractions.reshape(-1)
