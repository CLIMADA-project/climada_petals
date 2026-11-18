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

"""

import geopandas as gpd
import pandas as pd
import shapely
import numpy as np
import pyproj
from scipy.spatial import cKDTree
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
import logging
from pathlib import Path
import urllib.request

# requests, time, u_coords, TJ_TO_GWH, HRS_PER_YEAR: used by PowerFunctionalData on
# feature/nw_update (moves to its own module with the power PR)
import requests
import time

from climada.util import coordinates as u_coords
from climada.util.constants import EARTH_RADIUS_KM
from climada.util.coordinates import compute_geodesic_lengths
from climada_petals.util.constants import DICT_SPEEDS

# =============================================================================
# Shared names, used by all network modules (this module imports no other
# network module, so all of them can import these without import cycles)
# =============================================================================

#: ci_types whose components are lines (edges): their exposures are built from
#: the edges, impacts are aggregated back onto the edges, plots draw them as lines.
LINE_EXPOSURES = ["road", "rail", "power_line"]

#: default end-user ci_type for access checks, statistics and plots
DEF_END_USER = "people"

# Access states of end users, in the ``access_state_{source}_{target}`` node
# columns: set by ``Network.initialize_supply``, updated by the cascade.
#: nodes that are not end users of the dependency
ACCESS_UNDEFINED = "undefined"
#: end users without access in the base state
ACCESS_NO_BASE = "no base access"
ACCESS_UNDISRUPTED = "access undisrupted"
#: end users that lost their source but reach another one
ACCESS_NEW_SOURCE = "access new source"
#: end users that lost access because the path to the source is disrupted
ACCESS_DISRUPTED_VIA = "access disrupted via"
#: end users that lost access because their source failed
ACCESS_DISRUPTED_SOURCE = "access disrupted source"
#: end users of a capacity-based dependency whose sources no longer provide enough
ACCESS_DISRUPTED = "access disrupted"
#: states counted as a loss of access (plots and statistics)
ACCESS_LOSS_STATES = (ACCESS_DISRUPTED_VIA, ACCESS_DISRUPTED_SOURCE, ACCESS_DISRUPTED)

#: node column holding the population of end-user nodes (e.g. people cells)
POP_COL = "counts"

#: edge column marking the edges added by the module (cluster links, physical
#: links, dependency edges): True on them, missing on the input edges
ADDED_COL = "added"

# Energy conversion factors
TJ_TO_GWH = 0.277778
HRS_PER_YEAR = 8760

LOGGER = logging.getLogger(__name__)


def _added_mask(edges):
    """Boolean mask of the edges added by the module (see :data:`ADDED_COL`)."""
    if ADDED_COL not in edges.columns:
        return pd.Series(False, index=edges.index)
    return edges[ADDED_COL].eq(True)


def _dependency_name(source, target):
    """Edge ``ci_type`` of the dependency edges from ``source`` to ``target``

    This is the only name under which the cascade (capacity propagation,
    access checks, rerouting) looks up dependency edges.

    Parameters
    ----------
    source : str
        ``ci_type`` of the providing vertices.
    target : str
        ``ci_type`` of the dependent vertices.

    Returns
    -------
    str
        ``"dependency_{source}_{target}"``
    """
    return f"dependency_{source}_{target}"


# =============================================================================
# Distances in metres
# =============================================================================
# Longitude/latitude coordinates (e.g. EPSG:4326) are never compared in
# degrees: away from the equator, a degree of longitude (east-west) is shorter
# than a degree of latitude (north-south), by a factor cos(latitude).
# Distances in metres are measured along the Earth's surface. For the
# nearest-neighbour searches, the points are placed on a sphere: the straight
# line through the sphere between two points is shorter than the distance along
# its surface, but ranks points in the same order and converts exactly to and
# from it (_metres_to_chord, _chord_to_metres).

#: radius (m) of the sphere used for the nearest-neighbour searches
EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000


def _metre_factor(crs):
    """Metres per unit of a projected CRS (1 for metres)."""
    return crs.axis_info[0].unit_conversion_factor


def _edge_lengths_metres(edges):
    """Compute edge lengths in metres for any CRS.

    For geographic CRS (coordinates in degrees), geodesic distances are
    computed via :func:`climada.util.coordinates.compute_geodesic_lengths`.
    For projected CRS, ``shapely.length`` is converted to metres with the unit
    of the CRS.

    Parameters
    ----------
    edges : gpd.GeoDataFrame
        Edge GeoDataFrame with a geometry column and a CRS set.

    Returns
    -------
    np.ndarray
        Edge lengths in metres.

    Raises
    ------
    ValueError
        If the edges have no CRS: the coordinate units are then unknown, and
        assuming degrees would silently give wrong lengths for projected data.
    """
    if edges.crs is None:
        raise ValueError(
            "Edges have no CRS, so edge lengths cannot be computed in metres. "
            "Set it explicitly, e.g. edges.set_crs('EPSG:4326')."
        )
    if not edges.crs.is_geographic:
        return shapely.length(edges.geometry) * _metre_factor(edges.crs)

    # Geographic CRS – use geodesic distances.
    return compute_geodesic_lengths(edges).values


def _metric_coords(points, crs):
    """Coordinates for nearest-neighbour searches with distances in metres.

    Parameters
    ----------
    points : sequence of shapely.Point
        Points (e.g. a geometry column).
    crs : pyproj.CRS or str
        CRS of the points.

    Returns
    -------
    np.ndarray
        Projected CRS: ``(n, 2)`` x, y in metres. Longitude/latitude
        coordinates: ``(n, 3)`` positions on a sphere of radius
        :data:`EARTH_RADIUS_M`. The straight-line distance between two such
        positions (through the sphere, the chord) converts to and from the
        distance along the surface with :func:`_metres_to_chord` /
        :func:`_chord_to_metres`.
    """
    crs = pyproj.CRS(crs)
    points = np.asarray(points, dtype=object)
    x, y = shapely.get_x(points), shapely.get_y(points)
    if not crs.is_geographic:
        return np.column_stack([x, y]) * _metre_factor(crs)
    lon, lat = np.radians(x), np.radians(y)
    return EARTH_RADIUS_M * np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def _metres_to_chord(dist_m):
    """Straight-line distance through the sphere (chord) between two points
    that are ``dist_m`` metres apart along its surface."""
    arc = np.minimum(np.asarray(dist_m, dtype=float), np.pi * EARTH_RADIUS_M)
    return 2 * EARTH_RADIUS_M * np.sin(arc / (2 * EARTH_RADIUS_M))


def _chord_to_metres(chord):
    """Distance along the surface of the sphere (m) between two points whose
    straight-line distance through the sphere (chord) is ``chord``."""
    ratio = np.clip(np.asarray(chord, dtype=float) / (2 * EARTH_RADIUS_M), 0, 1)
    return 2 * EARTH_RADIUS_M * np.arcsin(ratio)


def _search_coords_and_radius(points, crs, dist_thresh):
    """Search coordinates and radius for a distance threshold.

    With ``crs`` None, the raw coordinates and ``dist_thresh`` in coordinate
    units are used; otherwise ``dist_thresh`` is in metres (see
    :func:`_metric_coords`). Returns the coordinates, the search radius and
    whether the coordinates are positions on the sphere, whose distances are
    then chords (see :func:`_chord_to_metres`).
    """
    if crs is None:
        points = np.asarray(points, dtype=object)
        coords = np.column_stack([shapely.get_x(points), shapely.get_y(points)])
        return coords, dist_thresh, False
    on_sphere = pyproj.CRS(crs).is_geographic
    coords = _metric_coords(points, crs)
    radius = dist_thresh
    if on_sphere and np.isfinite(dist_thresh):
        radius = _metres_to_chord(dist_thresh)
    return coords, radius, on_sphere


# =============================================================================
# Spatial analysis util functions
# =============================================================================


def make_edge_geometries(vs_geoms_from, vs_geoms_to):
    """Create straight LineString geometries between pairs of node geometries.

    Parameters
    ----------
    vs_geoms_from : list of shapely.Point
        Origin point geometries.
    vs_geoms_to : list of shapely.Point
        Destination point geometries.

    Returns
    -------
    list of shapely.LineString
        Straight-line geometries connecting each from-to pair.
    """
    return [
        shapely.geometry.LineString([geom_from, geom_to])
        for geom_from, geom_to in zip(vs_geoms_from, vs_geoms_to)
    ]


def _ckdnearest(vs_assign, gdf_base, k=1, dist_thresh=np.inf, crs=None):
    """Find the k nearest base points of each point to assign (cKDTree).

    Parameters
    ----------
    vs_assign : gpd.GeoDataFrame, pd.DataFrame or pandas.Series
        Points to assign (a ``geometry`` column), or a single row.
    gdf_base : gpd.GeoDataFrame or pd.DataFrame
        Candidate points (a ``geometry`` column).
    k : int, optional
        Number of nearest neighbours to find. Default: 1.
    dist_thresh : float, optional
        Maximum distance for valid matches. Default: np.inf.
    crs : pyproj.CRS or str, optional
        CRS of the points. If given, ``dist_thresh`` and the returned
        distances are in metres; for longitude/latitude coordinates they are
        measured along the Earth's surface. If None, both are in the units of
        the coordinates (degrees for longitude/latitude, where an east-west
        degree is shorter than a north-south degree away from the equator).
        Default: None.

    Returns
    -------
    dist : np.ndarray
        Distances to the k nearest neighbours (``inf`` beyond ``dist_thresh``).
    labels : np.ndarray
        Index labels of ``gdf_base`` of the nearest neighbours (float); NaN
        beyond ``dist_thresh``.
    """
    if isinstance(vs_assign, (gpd.GeoDataFrame, pd.DataFrame)):
        assign_points = vs_assign.geometry.values
    else:  # a single row
        assign_points = [vs_assign.geometry]
    base_coords, radius, on_sphere = _search_coords_and_radius(
        gdf_base.geometry.values, crs, dist_thresh
    )
    assign_coords, _, _ = _search_coords_and_radius(assign_points, crs, dist_thresh)
    dist, idx = cKDTree(base_coords).query(
        assign_coords, k=k, distance_upper_bound=radius
    )
    found = dist < np.inf
    if on_sphere:
        dist = np.where(found, _chord_to_metres(dist), np.inf)
    labels = np.full(dist.shape, np.nan)
    labels[found] = gdf_base.index.values[idx[found]]
    return dist, labels


# =============================================================================
# Network statistics
# =============================================================================


def get_tot_pop(network, enduser=DEF_END_USER, pop_col=POP_COL):
    """Total population of the end-user nodes.

    Parameters
    ----------
    network : Network
        Network with end-user nodes.
    enduser : str, optional
        ``ci_type`` of the end users. Default: :data:`DEF_END_USER`.
    pop_col : str, optional
        Node column holding the population. Default: :data:`POP_COL`.

    Returns
    -------
    float
    """
    nodes = network.nodes
    return nodes.loc[nodes["ci_type"] == enduser, pop_col].sum()


def _access_col(ci_type, enduser):
    return f"access_state_{ci_type}_{enduser}"


def _get_access_loss_mask(network, ci_type, enduser=DEF_END_USER):
    """Boolean mask of the nodes that lost access to ``ci_type``."""
    return network.nodes[_access_col(ci_type, enduser)].isin(ACCESS_LOSS_STATES)


def _get_no_base_access_mask(network, ci_type, enduser=DEF_END_USER):
    """Boolean mask of the nodes without base access to ``ci_type``."""
    return network.nodes[_access_col(ci_type, enduser)] == ACCESS_NO_BASE


def _get_access_undisrupted_mask(network, ci_type, enduser=DEF_END_USER):
    """Boolean mask of the nodes with undisrupted access to ``ci_type``."""
    return network.nodes[_access_col(ci_type, enduser)] == ACCESS_UNDISRUPTED


def _get_dir_affected_mask(network, enduser=DEF_END_USER):
    """Boolean mask of the end-user nodes with direct impacts."""
    nodes = network.nodes
    return (nodes["ci_type"] == enduser) & (nodes["imp_dir"] > 0)


def _get_imp_infra(network, ci_type, threshold=1, edge_col="distance", node_col=None):
    """Amount of infrastructure of ``ci_type`` with ``func_tot < threshold``.

    Line infrastructure (``ci_type`` present in the edges): sum of ``edge_col``
    over the edges, without the edges added by the module (e.g. the road links
    of :meth:`NetworkCalcs.add_physical_links`). Point infrastructure: number of nodes, or sum of
    ``node_col`` if given. ``threshold=np.inf`` gives the total amount.

    Returns
    -------
    float or int

    Raises
    ------
    ValueError
        If ``ci_type`` is neither in the edges nor in the nodes.
    """
    if ci_type in network.edges["ci_type"].unique():
        edges = network.edges
        infra_mask = (
            (edges["ci_type"] == ci_type)
            & (edges["func_tot"] < threshold)
            & ~_added_mask(edges)
        )
        return edges.loc[infra_mask, edge_col].sum()
    if ci_type in network.nodes["ci_type"].unique():
        nodes = network.nodes
        infra_mask = (nodes["ci_type"] == ci_type) & (nodes["func_tot"] < threshold)
        if not node_col:
            return int(infra_mask.sum())
        return nodes.loc[infra_mask, node_col].sum()
    raise ValueError(f"CI type '{ci_type}' not found in network nodes or edges.")


def make_network_stat(
    network,
    ci_types=None,
    threshold_func=1,
    edge_col="distance",
    node_col=None,
    enduser=DEF_END_USER,
    pop_col=POP_COL,
):
    """
    Compute network impact statistics.

    Parameters
    ----------
    network : Network
        Network object with nodes and edges, after the cascade.
    ci_types : list, optional
        CI types to analyse. If None: all node ``ci_type`` values except
        ``enduser``.
    threshold_func : float, optional
        Infrastructure with ``func_tot`` below this value counts as impacted.
        Default is 1 (anything not fully functional).
    edge_col : str, optional
        Edge column summed for impacted line infrastructure. Default is
        'distance' (metres).
    node_col : str, optional
        Node column summed for impacted point infrastructure. If None, the
        number of nodes is counted.
    enduser : str, optional
        ``ci_type`` of the end users. Default: :data:`DEF_END_USER`.
    pop_col : str, optional
        Node column holding the population of the end users. Default:
        :data:`POP_COL`.

    Returns
    -------
    dict
        Dictionary with keys:
            - 'total_population': total population of the end users
            - 'directly_affected': population directly impacted (imp_dir > 0)
            - 'indirectly_affected': population losing access to any CI type
            - 'total_affected': directly or indirectly affected (counted once)
            - For each ci_type:
                - f'{ci_type}_infra_impact': impacted infrastructure (sum of
                  ``edge_col`` for lines, without the edges added by the
                  module; number of nodes or sum of ``node_col`` for points)
                - f'{ci_type}_infra_total': the same for all infrastructure
            - For each ci_type with an end-user dependency (a column
              ``access_state_{ci_type}_{enduser}``); other types, e.g. power
              plants, have no access statistics:
                - f'{ci_type}_access_loss': population losing access
                  (states in :data:`ACCESS_LOSS_STATES`)
                - f'{ci_type}_no_base_access': population without access in
                  the base state
                - f'{ci_type}_access_undisrupted': population keeping access
    """
    nodes = network.nodes
    if ci_types is None:
        ci_types = [ct for ct in nodes["ci_type"].unique() if ct != enduser]

    pop_mask = nodes["ci_type"] == enduser
    dir_affected_mask = _get_dir_affected_mask(network, enduser)
    stats = {
        "total_population": get_tot_pop(network, enduser, pop_col),
        "directly_affected": nodes.loc[dir_affected_mask, pop_col].sum(),
    }

    def pop(mask):
        return nodes.loc[pop_mask & mask, pop_col].sum()

    indir_affected_mask = pd.Series(False, index=nodes.index)
    for ci_type in ci_types:
        if _access_col(ci_type, enduser) in nodes.columns:
            access_loss_mask = _get_access_loss_mask(network, ci_type, enduser)
            stats[f"{ci_type}_access_loss"] = pop(access_loss_mask)
            stats[f"{ci_type}_no_base_access"] = pop(
                _get_no_base_access_mask(network, ci_type, enduser)
            )
            stats[f"{ci_type}_access_undisrupted"] = pop(
                _get_access_undisrupted_mask(network, ci_type, enduser)
            )
            # indirectly affected: losing access to ANY ci_type
            indir_affected_mask |= access_loss_mask

        stats[f"{ci_type}_infra_impact"] = _get_imp_infra(
            network, ci_type, threshold_func, edge_col, node_col
        )
        stats[f"{ci_type}_infra_total"] = _get_imp_infra(
            network, ci_type, np.inf, edge_col, node_col
        )

    stats["indirectly_affected"] = pop(indir_affected_mask)
    # counted once when directly and indirectly affected
    stats["total_affected"] = pop(indir_affected_mask | dir_affected_mask)

    return stats


# =============================================================================
# Population data (WorldPop)
# =============================================================================


def get_worldpop_data(save_path, iso3=None, year=2024, res=1000):
    """Download a WorldPop population raster (release R2025A, constrained).

    The file is only downloaded if it is not in ``save_path`` yet.

    Parameters
    ----------
    save_path : str or pathlib.Path
        Directory to save the file in (created if missing).
    iso3 : str, optional
        ISO 3166-1 alpha-3 country code. If None, the global 1 km mosaic is
        downloaded (large file). Required for 100 m data.
    year : int, optional
        Year of the population estimate. Default: 2024.
    res : int, optional
        Resolution in metres: 1000 (UN-adjusted counts) or 100. Default: 1000.

    Returns
    -------
    pathlib.Path
        Path of the raster file.

    Raises
    ------
    ValueError
        If ``res`` is not 100 or 1000, or if ``res`` is 100 without ``iso3``.
    """
    if res not in (100, 1000):
        raise ValueError(f"res must be 100 or 1000 (metres), got {res!r}.")
    base_url = (
        f"https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/{year}/"
    )
    if iso3 is None:
        if res == 100:
            raise ValueError("ISO3 code must be provided for 100m resolution data.")
        download_url = (
            base_url + "0_Mosaicked/v1/1km_ua/constrained/"
            f"global_pop_{year}_CN_1km_R2025A_UA_v1.tif"
        )
    elif res == 1000:
        download_url = (
            base_url + f"{iso3}/v1/1km_ua/constrained/"
            f"{iso3.lower()}_pop_{year}_CN_1km_R2025A_UA_v1.tif"
        )
    else:
        download_url = (
            base_url + f"{iso3}/v1/100m/constrained/"
            f"{iso3.lower()}_pop_{year}_CN_100m_R2025A_v1.tif"
        )

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    local_filepath = save_path / download_url.split("/")[-1]
    if local_filepath.is_file():
        LOGGER.info("File already exists: %s", local_filepath)
    else:
        LOGGER.info("Downloading %s to %s", download_url, local_filepath)
        urllib.request.urlretrieve(download_url, local_filepath)
    return local_filepath


def _extent_window(dataset, extent):
    """Window of whole raster cells overlapping ``extent`` (all cells if None).

    ``extent`` is a shapely geometry or ``(xmin, ymin, xmax, ymax)``; the
    window is rounded outwards to whole cells, so that its transform lies on
    the raster grid.
    """
    if extent is None:
        return Window(0, 0, dataset.width, dataset.height)
    xmin, ymin, xmax, ymax = (
        extent if isinstance(extent, (tuple, list)) else extent.bounds
    )
    col0, row0 = ~dataset.transform * (xmin, ymax)
    col1, row1 = ~dataset.transform * (xmax, ymin)
    eps = 1e-9  # do not add a cell for a floating-point difference
    col0, row0 = max(int(np.floor(col0 + eps)), 0), max(int(np.floor(row0 + eps)), 0)
    col1 = min(int(np.ceil(col1 - eps)), dataset.width)
    row1 = min(int(np.ceil(row1 - eps)), dataset.height)
    if col1 <= col0 or row1 <= row0:
        raise ValueError(f"The extent {extent} does not overlap the raster.")
    return Window(col0, row0, col1 - col0, row1 - row0)


def _resample_res(filepath, upscale_factor, nodata=None, extent=None):
    """Read a population raster, optionally cropped and resampled.

    Nodata cells (e.g. water in WorldPop) count as 0 people, also when
    aggregating. Resampling uses the average of the overlapping cells times
    the number of original cells per new cell, and the result is rescaled so
    that the total equals the total of the original cells in the extent.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the raster file (band 1 is read).
    upscale_factor : float
        Factor by which to change the resolution: > 1 finer cells, < 1
        coarser cells (e.g. 0.5 aggregates 2 x 2 cells into one).
    nodata : float, optional
        Nodata value. Default: the raster's nodata value.
    extent : shapely geometry or tuple, optional
        Extent to crop to: a geometry (its bounds are used) or
        ``(xmin, ymin, xmax, ymax)``. Default: the whole raster.

    Returns
    -------
    arr : np.ndarray
        2D array of population counts.
    transform : rasterio.Affine
        Affine transform of ``arr``.
    crs : rasterio.crs.CRS or None
        CRS of the raster.
    """
    if not upscale_factor > 0:
        raise ValueError(f"upscale_factor must be positive, got {upscale_factor!r}.")
    with rasterio.open(filepath) as dataset:
        window = _extent_window(dataset, extent)
        src_transform = dataset.window_transform(window)
        crs = dataset.crs
        nodata = dataset.nodata if nodata is None else nodata
        arr = dataset.read(1, window=window).astype(float)

    # before any resampling, so that nodata cannot leak into the averages
    if nodata is not None:
        arr[arr == nodata] = 0.0
    arr[np.isnan(arr)] = 0.0
    if upscale_factor == 1:
        return arr, src_transform, crs

    height = int(arr.shape[0] * upscale_factor)
    width = int(arr.shape[1] * upscale_factor)
    if height < 1 or width < 1:
        raise ValueError("upscale_factor is too small for the extent: no cell left.")
    transform = src_transform * rasterio.Affine.scale(1 / upscale_factor)
    resampled = np.zeros((height, width))
    work_crs = crs if crs is not None else "EPSG:4326"  # same CRS: no reprojection
    reproject(
        arr,
        resampled,
        src_transform=src_transform,
        src_crs=work_crs,
        dst_transform=transform,
        dst_crs=work_crs,
        resampling=Resampling.average,
    )
    resampled *= (1 / upscale_factor) ** 2
    # rows / columns cut off by the integer number of new cells
    if resampled.sum() > 0:
        resampled *= arr.sum() / resampled.sum()
    return resampled, transform, crs


def load_resampled_raster(
    filepath, upscale_factor=1.0, nodata=None, extent=None, crs=None
):
    """Load a population raster as points, optionally cropped and resampled.

    Each non-zero cell becomes a point at the cell centre, with its
    population in the :data:`POP_COL` column. See :func:`_resample_res` for
    the handling of nodata and resampling.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the raster file.
    upscale_factor : float, optional
        Factor by which to change the resolution (> 1 finer, < 1 coarser).
        Default: 1.0 (no resampling).
    nodata : float, optional
        Nodata value. Default: the raster's nodata value.
    extent : shapely geometry or tuple, optional
        Extent to crop to: a geometry (its bounds are used) or
        ``(xmin, ymin, xmax, ymax)``; cells overlapping it are kept.
        Default: the whole raster.
    crs : str or pyproj.CRS, optional
        CRS of the output. Default: the raster's CRS.

    Returns
    -------
    gpd.GeoDataFrame
        Columns :data:`POP_COL` and 'geometry', one row per non-zero cell.

    Raises
    ------
    ValueError
        If the raster has no CRS and ``crs`` is not given.
    """
    arr, transform, raster_crs = _resample_res(filepath, upscale_factor, nodata, extent)
    crs = crs if crs is not None else raster_crs
    if crs is None:
        raise ValueError(f"{filepath} has no CRS: pass it as crs=...")

    # cell centres
    xs = transform.c + (np.arange(arr.shape[1]) + 0.5) * transform.a
    ys = transform.f + (np.arange(arr.shape[0]) + 0.5) * transform.e
    grid_x, grid_y = np.meshgrid(xs, ys)
    gdf = gpd.GeoDataFrame(
        {
            POP_COL: arr.ravel(),
            "geometry": gpd.points_from_xy(grid_x.ravel(), grid_y.ravel()),
        },
        crs=crs,
    )
    return gdf[gdf[POP_COL] != 0].reset_index(drop=True)


def get_pop_cutoff(gdf_people, cutoff, pop_col=POP_COL):
    """Population threshold below which cells hold less than ``cutoff`` of the people.

    Cells are grouped in bins of 10 people ((0, 10], (10, 20], ...); the
    result is the left edge of the first bin at which the cumulative share of
    the total population reaches ``cutoff``. Cells with a population at or
    below the result hold less than ``cutoff`` of the total, so they can be
    dropped (e.g. ``gdf[gdf[pop_col] >= result]``).

    Parameters
    ----------
    gdf_people : gpd.GeoDataFrame
        Population grid points.
    cutoff : float
        Share of the total population (0 to 1).
    pop_col : str, optional
        Column holding the population. Default: :data:`POP_COL`.

    Returns
    -------
    float
        Population threshold (a multiple of 10); 0 if nothing can be dropped.

    Raises
    ------
    ValueError
        If ``cutoff`` is not between 0 and 1.
    """
    if not 0 <= cutoff <= 1:
        raise ValueError(f"cutoff must be between 0 and 1, got {cutoff!r}.")
    counts = gdf_people[pop_col]
    total = counts.sum()
    if total <= 0:
        return 0.0
    # the last edge reaches the largest cell, at least one bin
    bin_edges = np.arange(0, max(counts.max(), 0) + 10, 10)
    if len(bin_edges) < 2:
        bin_edges = np.array([0, 10])
    shares = counts.groupby(pd.cut(counts, bin_edges), observed=False).sum() / total
    reached = shares.cumsum().to_numpy() >= cutoff - 1e-12  # rounding of the sum
    return float(shares.index[np.argmax(reached)].left)


# =============================================================================
# Distance Threshold setting
# =============================================================================


def set_travel_distance_threshs(
    df_dependencies, iso3, sources, target=DEF_END_USER, hrs_max=1
):
    """Set road travel distance thresholds based on country-specific speeds.

    Sets the distance threshold (in metres) of the dependencies from
    ``sources`` to ``target`` to the distance travelled in ``hrs_max`` hours
    at the average road speed of the country: ``speed (km/h) x 1000 x
    hrs_max``. Other rows are unchanged.

    Data taken from *Road Quality and Mean Speed Score*,
    Author/Editor: Marian Moszoro; Mauricio Soto,
    ISBN: 9798400210440/1018-5941.

    Parameters
    ----------
    df_dependencies : pd.DataFrame
        Dependencies DataFrame with 'source', 'target', and 'thresh_dist'
        columns.
    iso3 : str
        ISO 3166-1 alpha-3 country code. Countries without a speed in
        ``DICT_SPEEDS`` get the speed of "other".
    sources : str or list of str
        ``source`` ci_types of the dependencies to set, e.g.
        ``["healthcare", "education"]``.
    target : str, optional
        ``target`` ci_type of the dependencies to set. Default:
        :data:`DEF_END_USER`.
    hrs_max : float, optional
        Maximum travel time in hours. Default: 1.

    Returns
    -------
    pd.DataFrame
        Copy of the dependencies DataFrame with 'thresh_dist' set for the
        rows with a source in ``sources`` and target ``target``.
    """
    speed_kmh = DICT_SPEEDS.get(iso3, DICT_SPEEDS["other"])
    thresh_dist = int(speed_kmh * 1000 * hrs_max)

    if isinstance(sources, str):
        sources = [sources]
    df_dependencies = df_dependencies.copy()
    rows = df_dependencies["source"].isin(sources) & (
        df_dependencies["target"] == target
    )
    df_dependencies.loc[rows, "thresh_dist"] = thresh_dist
    return df_dependencies
