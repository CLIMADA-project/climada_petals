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


def _ckdnearest(vs_assign, gdf_base, k=1, dist_thresh=np.inf):
    """Find the k nearest neighbours using a cKDTree.

    See https://gis.stackexchange.com/a/301935.

    Parameters
    ----------
    vs_assign : gpd.GeoDataFrame or pandas.Series
        GeoDataFrame or single row with point geometries to assign.
    gdf_base : gpd.GeoDataFrame
        GeoDataFrame of candidate destination points.
    k : int, optional
        Number of nearest neighbours to find. Default: 1.
    dist_thresh : float, optional
        Maximum distance for valid matches. Default: np.inf.

    Returns
    -------
    dist : np.ndarray
        Distances to the k nearest neighbours.
    vld_ind_formatted : np.ndarray
        Indices into ``gdf_base`` of valid nearest neighbours. Invalid
        matches (beyond ``dist_thresh``) are set to NaN.
    """
    # TODO: this mixed input options (1 vertex vs gdf) is not nicely solved
    if isinstance(vs_assign, (gpd.GeoDataFrame, pd.DataFrame)):
        n_assign = np.array(list(vs_assign.geometry.apply(lambda x: (x.x, x.y))))
    else:
        n_assign = np.array([(vs_assign.geometry.x, vs_assign.geometry.y)])
    n_base = np.array(list(gdf_base.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(n_base)
    dist, idx = btree.query(n_assign, k=k, distance_upper_bound=dist_thresh)

    vld_ind = gdf_base.iloc[idx[dist < np.inf]].index.values
    vld_ind_formatted = np.empty(dist.shape)
    vld_ind_formatted[:] = np.nan
    vld_ind_formatted[np.where(dist < np.inf)] = vld_ind
    return dist, vld_ind_formatted


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


# =============================================================================
# Power Supply & Demand Data
# =============================================================================


class PowerFunctionalData:

    def load_pplants_wri(self, path_pplants_wri, iso3):
        """
        load power plant gdf from an excel file containing the WRI global
        power plant database.
        """
        gdf_pp_world = gpd.read_file(path_pplants_wri, crs="EPSG:4326")
        gdf_pp = gdf_pp_world[gdf_pp_world.country == f"{iso3}"][
            ["estimated_generation_gwh_2017", "latitude", "longitude", "name"]
        ]
        del gdf_pp_world
        if not gdf_pp.empty:
            gdf_pp = gpd.GeoDataFrame(
                gdf_pp, geometry=gpd.points_from_xy(gdf_pp.longitude, gdf_pp.latitude)
            )
            gdf_pp = gdf_pp.drop(["latitude", "longitude"], axis=1)

        return gdf_pp

    def query_elaccess_wb(self, iso3):
        ind = "EG.ELC.ACCS.ZS"
        query = f"http://api.worldbank.org/v2/country/{iso3.lower()}/indicator/{ind}?mrnev=1&format=json"
        trials = 0
        while trials < 5:
            trials += 1
            try:
                response = requests.get(query)
                return response.json()[1][0]["value"]
            except:
                time.sleep(5)
                LOGGER.info("Sleeping for 5 secs until next WB data query try")

    def load_eltargets(self, cntry_shape, path_et):
        meta_et, arr_et = u_coords.read_raster(
            path_et, src_crs={"epsg": "4326"}, geometry=[cntry_shape]
        )
        grid = u_coords.raster_to_meshgrid(
            meta_et["transform"], meta_et["width"], meta_et["height"]
        )
        gdf_et = gpd.GeoDataFrame(
            {
                "counts": arr_et.squeeze().flatten(),
                "geometry": gpd.points_from_xy(grid[0].flatten(), grid[1].flatten()),
            }
        )
        gdf_et = gdf_et[gdf_et.counts != 0].reset_index(drop=True)
        gdf_et["geometry"] = gdf_et.geometry.buffer(meta_et["transform"][0] / 2)
        return gdf_et

    def assign_el_targets(self, gdf_people, iso3, cntry_shape, path_et):

        el_rate = self.query_elaccess_wb(iso3)

        if el_rate < 90:
            gdf_et = self.load_eltargets(cntry_shape, path_et)
            gdf_people["electrified"] = (
                gdf_people.sjoin(gdf_et, how="left", predicate="within")["counts_right"]
                .groupby(level=0)
                .sum()
                > 0
            )
            counter = 0
            while (
                gdf_people[gdf_people["electrified"]].counts.sum()
                < (gdf_people.counts.sum() * el_rate / 100)
                and counter < 10
            ):
                gdf_et["geometry"] = gdf_et.geometry.buffer(0.01)
                gdf_people["electrified"] = (
                    gdf_people.sjoin(gdf_et, how="left", predicate="within")[
                        "counts_right"
                    ]
                    .groupby(level=0)
                    .sum()
                    > 0
                )
                counter += 1
        else:
            gdf_people["electrified"] = True

        return gdf_people.electrified

    def assign_el_prod_consump(self, gdf_people, gdf_pplants, iso3, path_final_cons):
        """
        Takes a countries' annual electricity consumption value (as
        gathered by the IEA in https://www.iea.org/data-and-statistics/data-tables?,
        "Total Final Consumption, Electricity") and assigns this
        i) to gdf_people, proportional to population count
        ii) to gdf_pplants, either proportional to plant capacity reported in
        the global power plant database, or equally distributed.

        If no consumption value is available for the country, dummy demands
        proportional to population are assigned to gdf_people, and equally
        distributed onto gdf_pplants.

        Returns
        ------
        gdf_people with column el_consumption
        gdf_pplants with column el_generation
        """

        # easiest default for population and power plants - assign count dummies
        gdf_people["el_consumption"] = 0
        gdf_people.loc[gdf_people.electrified, "el_consumption"] = gdf_people.loc[
            gdf_people.electrified, "counts"
        ]
        gdf_pplants["el_generation"] = gdf_people["el_consumption"].sum() / len(
            gdf_pplants
        )

        # check if better data available from IEA & WRI
        df_final_cons = pd.read_csv(path_final_cons)
        final_cons = np.nan
        if iso3 in df_final_cons.ISO3.values:
            final_cons = df_final_cons[
                df_final_cons.ISO3 == iso3
            ].el_consumption.values[0]
        if not np.isnan(final_cons):
            # assign electricity consumption to population
            gdf_people.loc[gdf_people.electrified, "el_consumption"] = (
                gdf_people.loc[gdf_people.electrified, "counts"]
                / gdf_people.loc[gdf_people.electrified, "counts"].sum()
                * final_cons
            )
            if "estimated_generation_gwh_2017" in gdf_pplants.columns:
                # assign electricity generation to power plants
                gdf_pplants["estimated_generation_gwh_2017"] = pd.to_numeric(
                    gdf_pplants.estimated_generation_gwh_2017, errors="coerce"
                )
                gdf_pplants["estimated_generation_gwh_2017"].fillna(
                    np.nanmean(gdf_pplants["estimated_generation_gwh_2017"]),
                    inplace=True,
                )
                gdf_pplants["el_generation"] = (
                    gdf_pplants.estimated_generation_gwh_2017
                    / gdf_pplants.estimated_generation_gwh_2017.sum()
                    * final_cons
                )
                gdf_pplants = gdf_pplants.drop("estimated_generation_gwh_2017", axis=1)
            else:
                gdf_pplants["el_generation"] = final_cons / len(gdf_pplants)

        return gdf_people, gdf_pplants

    def assign_edemand_iea(self, gdf_people, path_elcons_iea):
        """
        Assigns annual electricity consumptions to power
        clusters based on per capita consuption statistics
        retrieved from IEA.org --> Heat & Electricity --> Electricity Consumption
        per capita.
        Returns a pd.Series with el_consumption in GWh for the last reported year
        (currently 2019)
        """

        df_el_cons = pd.read_csv(path_elcons_iea, skiprows=4)

        if df_el_cons.Units.iloc[0] != "MWh/capita":
            LOGGER.warning(
                "Units of per capita electricity consumption are"
                + f"different than expected. ({df_el_cons.Units.iloc[0]})"
            )

        per_cap_cons = df_el_cons["Electricity consumption/population"].iloc[-1]
        LOGGER.info(
            "Taking per capita electricity consumption value for year"
            + f" {df_el_cons['Unnamed: 0'].iloc[-1]}"
        )
        return gdf_people.counts * per_cap_cons / 1000

    def assign_esupply_iea(self, gdf_pplants, path_elgen_iea):
        """
        Assigns annual electricity generation (in GWh) to each power plant
        reported in the global power plant database from the WRI.
        Electricity generation is taken from IEA.org (expects a column
        'estimated_generation_gwh_2017') and re-distributed upon
        power plants proportionally to the generation values given in the
        WRI database, and the rest distributed equally on missing values.
        """

        df_el_gen = pd.read_csv(path_elgen_iea, skiprows=4)

        # Latest Electricity Generation data from the IEA (2019)
        if df_el_gen.iloc[-1]["Units"] != "GWh":
            LOGGER.warning("Expected different units for generation.")
        gen = np.nansum(
            df_el_gen[
                list(set(df_el_gen.columns.values).difference({"Units", "Unnamed: 0"}))
            ]
            .iloc[-1]
            .values
        )

        # generation from WRI power plants database (usually incomplete)
        gdf_pplants.estimated_generation_gwh_2017 = pd.to_numeric(
            gdf_pplants.estimated_generation_gwh_2017, errors="coerce"
        )

        plant_gen = gdf_pplants.estimated_generation_gwh_2017.fillna(
            np.nanmean(gdf_pplants.estimated_generation_gwh_2017)
        )

        return np.array(plant_gen / sum(plant_gen) * gen)

    def assign_impexp_iea(self, gdf_pplants, path_elimpexp_iea, var_name):
        """
        Places an import / export node outside the country,
        to account for this delta as reported in IEA.org
        """
        # Latest annual Import/Export data from the IEA (2019)
        df_el_impexp = pd.read_csv(path_elimpexp_iea, skiprows=4)

        if df_el_impexp.iloc[-1]["Units"] != "TJ":
            LOGGER.warning("Expected different units for import/export.")
        if "Exports" not in df_el_impexp.columns:
            df_el_impexp["Exports"] = 0
            LOGGER.warning("No export column. Setting exports to 0.")
        if "Imports" not in df_el_impexp.columns:
            df_el_impexp["Imports"] = 0
            LOGGER.warning("No import column. Setting imports to 0.")

        el_imp = df_el_impexp.iloc[-1]["Imports"] * TJ_TO_GWH
        el_exp = df_el_impexp.iloc[-1]["Exports"] * TJ_TO_GWH
        imp_exp = el_imp + el_exp

        return gpd.GeoDataFrame(
            {
                "geometry": [
                    shapely.geometry.Point(
                        max(gdf_pplants.geometry.x) + 1, max(gdf_pplants.geometry.y) + 1
                    )
                ],
                "name": ["imp_exp_balance"],
                f"{var_name}": [imp_exp],
            }
        )

    def balance_el_generation(self, gdf_pplants, per_cap_cons, pop_no):

        gdf_pplants.estimated_generation_gwh_2017 = pd.to_numeric(
            gdf_pplants.estimated_generation_gwh_2017, errors="coerce"
        )

        tot_cons_mw = per_cap_cons * pop_no / HRS_PER_YEAR
        tot_prod_mw = (
            gdf_pplants.estimated_generation_gwh_2017.sum() * 1000 / HRS_PER_YEAR
        )

        unassigned = tot_cons_mw - tot_prod_mw
        nans = np.isnan(gdf_pplants.estimated_generation_gwh_2017).sum()

        gdf_pplants["el_gen_mw"] = (
            gdf_pplants.estimated_generation_gwh_2017 * 1000 / HRS_PER_YEAR
        )
        gdf_pplants["el_gen_mw"][np.isnan(gdf_pplants.el_gen_mw)] = unassigned / nans

        return gdf_pplants
