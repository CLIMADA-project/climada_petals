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
from matplotlib import pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap

from climada.util import coordinates as u_coords
from climada_petals.util.constants import DICT_SPEEDS

# =============================================================================
# Shared names, used by all network modules (this module imports no other
# network module, so all of them can import these without import cycles)
# =============================================================================

#: ci_types whose components are lines (edges): their exposures are built from
#: the edges, impacts are aggregated back onto the edges, plots draw them as lines.
LINE_EXPOSURES = ["road", "rail", "power_line"]

DEF_END_USER = "people"  # default end-user type for access checks and statistics

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

# Plots
MPL_MARKERS = list(mpl.markers.MarkerStyle.markers.keys())
# Use matplotlib-recognised linestyles to avoid ValueError when applying to plot()
MPL_LINE_STYLES = [
    "solid",
    "dashed",
    "dashdot",
    "dotted",
]

#: colour class of each access state in :func:`access_plot` (-2 to 2)
STATUS_MAP = {
    ACCESS_UNDISRUPTED: -2,
    ACCESS_NEW_SOURCE: -1,
    ACCESS_NO_BASE: 0,
    ACCESS_DISRUPTED_VIA: 1,
    ACCESS_DISRUPTED_SOURCE: 2,
    ACCESS_DISRUPTED: 2,  # also a failure at the source side
}
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
# Plots
# =============================================================================
def enduser_plot(
    self, enduser="people", axes=None, projection=ccrs.PlateCarree(), **kwargs
):
    """Plot population nodes as a scatter plot.

    Parameters
    ----------
    self : Network
        Network instance containing nodes with ci_type enduser.
    enduser : str, optional
        The enduser type to plot. Default: 'people'.
    axes : cartopy.mpl.geoaxes.GeoAxes, optional
        Axes to plot on. If None, a new figure and axes are created.
    projection : cartopy.crs.Projection, optional
        Map projection. Default: PlateCarree.
    **kwargs
        Additional keyword arguments passed to ``axes.scatter``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : cartopy.mpl.geoaxes.GeoAxes
        The axes with the population plot.
    """
    plot_df = self.nodes[self.nodes.ci_type == enduser]
    if axes is None:
        fig, axes = plt.subplots(1, 1, subplot_kw=dict(projection=projection))
    else:
        fig = axes.get_figure()
    axes.scatter(
        plot_df.geometry.x,
        plot_df.geometry.y,
        s=plot_df["value"],
        label=enduser,
        transform=projection,
        alpha=0.5,
        color="grey",
        **kwargs,
    )
    return fig, axes


def infra_plot(
    self,
    ci_types=None,
    plot_col="ci_type",
    enduser="people",
    mask_func_ci=True,
    road_mask=True,
    axes=None,
    legend=True,
    legend_loc="upper right",
    projection=ccrs.PlateCarree(),
    pop_kwargs=dict(),
    ci_kwargs=dict(),
    cbar_kwargs=dict(),
):
    """Plot critical infrastructure network elements.

    Plots nodes and edges for each CI type with distinct markers, colors,
    and line styles. Point infrastructure (non-people, non-line) is rendered
    as a coloured disc with a white letter identifier on top for high
    visibility. When ``plot_col`` is not 'ci_type', a functional status
    colorbar (disrupted/functioning) is added.

    Parameters
    ----------
    self : Network
        Network instance containing nodes and edges.
    ci_types : list of str, optional
        CI types to plot. If None, all unique ci_types in nodes are used.
    plot_col : str, optional
        Column to use for coloring. Default: 'ci_type'.
    enduser : str, optional
        Enduser type to plot as background. Default: 'people'.
    axes : cartopy.mpl.geoaxes.GeoAxes, optional
        Axes to plot on. If None, a new figure and axes are created.
    projection : cartopy.crs.Projection, optional
        Map projection. Default: PlateCarree.
    pop_kwargs : dict, optional
        Additional keyword arguments for population scatter plot.
    ci_kwargs : dict, optional
        Additional keyword arguments for CI element plots.
    cbar_kwargs : dict, optional
        Additional keyword arguments for the colorbar.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : cartopy.mpl.geoaxes.GeoAxes
        The axes with the infrastructure plot.
    """
    if not ci_types:
        ci_types = self.nodes.ci_type.unique()
    if axes is None:
        fig, axes = plt.subplots(1, 1, subplot_kw=dict(projection=projection))
    else:
        fig = axes.get_figure()
    colors = ci_kwargs.pop("colors", mpl.cm.tab20.colors)

    # Define colormap based on plot_col
    if plot_col == "ci_type":
        # Each CI type gets its own color
        use_status_colorbar = False
        vmin, vmax = None, None
    else:
        # Functional status: 0=red (disrupted), 1=black (functioning)
        status_cmap = mpl.colors.ListedColormap(["r", "k"])
        use_status_colorbar = True
        vmin, vmax = 0, 1

    for i, ci_type in enumerate(ci_types):
        point_marker = MPL_MARKERS[i % len(MPL_MARKERS)]
        line_style = MPL_LINE_STYLES[i % len(MPL_LINE_STYLES)]
        color = colors[i]

        # Set colormap for this CI type
        if plot_col == "ci_type":
            cmap = mpl.colors.ListedColormap([color])
        else:
            cmap = status_cmap

        if ci_type == enduser:
            fig, axes = enduser_plot(
                self,
                enduser=enduser,
                axes=axes,
                projection=projection,
                zorder=0,
                **pop_kwargs,
            )
        elif ci_type in LINE_EXPOSURES:
            plot_df = self.edges[self.edges.ci_type == ci_type]
            if mask_func_ci:
                plot_df = plot_df[plot_df["func_tot"] < 1]
            if road_mask & (ci_type == "road"):
                # roads added by the module (links to facilities, people)
                plot_df = plot_df[~_added_mask(plot_df)]

            plot_df.plot(
                plot_col,
                ax=axes,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                transform=projection,
                label=ci_type,
                linestyle=line_style,
                zorder=1,
                **ci_kwargs,
            )
        else:
            plot_df = self.nodes[self.nodes.ci_type == ci_type]
            if mask_func_ci:
                plot_df = plot_df[plot_df["func_tot"] < 1]
            marker_letter = f"${ci_type[0].upper()}$"

            # Background coloured disc for visibility
            bg_kw = dict(
                s=350,
                marker="o",
                transform=projection,
                zorder=i + 1,
                edgecolor="white",
                linewidth=1.5,
            )
            if plot_col == "ci_type":
                bg_kw["color"] = color
            else:
                bg_kw.update(
                    c=plot_df[plot_col].values,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
            axes.scatter(
                plot_df.geometry.x,
                plot_df.geometry.y,
                **bg_kw,
                **ci_kwargs,
            )
            # White letter marker on top
            axes.scatter(
                plot_df.geometry.x,
                plot_df.geometry.y,
                s=100,
                color="white",
                marker=marker_letter,
                transform=projection,
                label=ci_type,
                zorder=i + 2,
            )

    if legend:
        axes.legend(loc=legend_loc)

    # Add a single status colorbar if plotting functional status
    if use_status_colorbar and not mask_func_ci:
        orientation = cbar_kwargs.pop("orientation", "horizontal")
        shrink = cbar_kwargs.pop("shrink", 0.55)
        sm = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=status_cmap
        )
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=axes, orientation=orientation, shrink=shrink, **cbar_kwargs
        )
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Disrupted", "Functioning"], rotation=30)
        cbar.set_label("CI status")

    return fig, axes


def _get_dependencies(self, ci_types):
    """Get unique dependency edge types for the given CI types.

    Parameters
    ----------
    self : Network
        Network instance containing edges.
    ci_types : list of str
        CI types to filter dependency edges for.

    Returns
    -------
    np.ndarray
        Array of unique dependency edge type names.
    """
    dep_list = [
        dep
        for dep in self.edges.ci_type
        for ci_type in ci_types
        if "dependency_" in dep and ci_type in dep
    ]
    return np.unique(dep_list)


def dep_plot(self, ci_types=None, axes=None, projection=ccrs.PlateCarree(), **kwargs):
    """Plot dependency edges between CI types.

    Parameters
    ----------
    self : Network
        Network instance containing edges with dependency types.
    ci_types : list of str, optional
        CI types to plot dependencies for. If None, all unique ci_types
        in nodes are used.
    axes : cartopy.mpl.geoaxes.GeoAxes, optional
        Axes to plot on. If None, a new figure and axes are created.
    projection : cartopy.crs.Projection, optional
        Map projection. Default: PlateCarree.
    **kwargs
        Additional keyword arguments passed to ``GeoDataFrame.plot``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : cartopy.mpl.geoaxes.GeoAxes
        The axes with the dependency plot.
    """
    if not ci_types:
        ci_types = self.nodes.ci_type.unique()
    if axes is None:
        fig, axes = plt.subplots(1, 1, subplot_kw=dict(projection=projection))
    else:
        fig = axes.figure
    dependencies = _get_dependencies(self, ci_types)
    colors = kwargs.pop("colors", mpl.cm.tab20.colors)
    for i, dep in enumerate(dependencies):
        color = colors[i]
        plot_df = self.edges[self.edges.ci_type == dep]
        plot_df.plot(
            ax=axes,
            color=color,
            transform=projection,
            label=dep,
            alpha=0.5,
            zorder=10,
            **kwargs,
        )
    axes.legend()
    # f.suptitle(title ,fontweight="bold",y=0.92)
    return fig, axes


def access_plot(
    self,
    ci_type,
    enduser="people",
    access_states=None,
    axes=None,
    projection=ccrs.PlateCarree(),
    plot_kwargs=dict(),
    cbar_kwargs=dict(),
):
    """Plot population access status for a given CI type.

    Visualises the access state of population clusters to a specific
    CI service, color-coded by access status and sized by population count.

    Parameters
    ----------
    self : Network
        Network instance containing nodes with ci_type 'people' and
        access state columns.
    ci_type : str
        The CI type to assess access for (e.g. 'healthcare', 'power').
    enduser : str, optional
        The type of end user to plot access for. Default is 'people'.
    access_states : list of str, optional
        Subset of access states to plot. If None, all states are plotted.
        Possible values: the keys of :data:`STATUS_MAP`.
    axes : cartopy.mpl.geoaxes.GeoAxes, optional
        Axes to plot on. If None, a new figure and axes are created.
    projection : cartopy.crs.Projection, optional
        Map projection. Default: PlateCarree.
    plot_kwargs : dict, optional
        Additional keyword arguments for ``axes.scatter``.
    cbar_kwargs : dict, optional
        Additional keyword arguments for the colorbar.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : cartopy.mpl.geoaxes.GeoAxes
        The axes with the access plot.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 1, subplot_kw=dict(projection=projection))
    else:
        fig = axes.figure

    cmap = plot_kwargs.pop(
        "cmap", ListedColormap(["#046582", "#158204", "grey", "#CF8018", "#BB8082"])
    )
    gdf_enduser = self.nodes[self.nodes.ci_type == enduser]

    service = f"access_state_{ci_type}_{enduser}"

    # get access states
    if access_states is not None:
        gdf_enduser = gdf_enduser[gdf_enduser[service].isin(access_states)]

    # lowercase for safety
    cvals = gdf_enduser[service].str.lower().map(STATUS_MAP).fillna(0)
    scatter = axes.scatter(
        gdf_enduser.geometry.x,
        gdf_enduser.geometry.y,
        c=cvals,
        s=gdf_enduser["counts"] / max(gdf_enduser["counts"]) * 100,
        transform=projection,
        cmap=cmap,
        vmin=-2,
        vmax=2,
        **plot_kwargs,
    )
    n_ppl_access_loss = gdf_enduser[
        gdf_enduser[service].isin(ACCESS_LOSS_STATES)
    ].counts.sum()
    n_ppl_no_base_access = gdf_enduser[
        gdf_enduser[service] == ACCESS_NO_BASE
    ].counts.sum()

    text = (
        f"{n_ppl_no_base_access:.2f} {enduser} without base access to {ci_type}"
        + f"\n{n_ppl_access_loss:.2f} {enduser} loosing access to {ci_type}"
    )
    axes.text(
        0.015,
        0.015,
        text,
        transform=axes.transAxes,
        bbox=dict(ec="gray", fc="gray", alpha=0.2),
    )
    # add colorbar
    orientation = cbar_kwargs.pop("orientation", "horizontal")
    shrink = cbar_kwargs.pop("shrink", 0.55)
    cbar = fig.colorbar(
        scatter, ax=axes, orientation=orientation, shrink=shrink, **cbar_kwargs
    )
    cbar.set_ticks([-8 / 5, -4 / 5, 0, 4 / 5, 8 / 5])
    # one label per colour class; states sharing a class share the label
    class_states = {}
    for state, value in STATUS_MAP.items():
        class_states.setdefault(value, []).append(state)
    cbar.set_ticklabels(
        [" / ".join(class_states[value]) for value in sorted(class_states)],
        rotation=30,
    )
    cbar.set_label("Access to service")  # ,fontweight="bold" fontsize=24
    return fig, axes


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


def get_tot_pop(network, enduser="people", pop_col=POP_COL):
    """Total population of the end-user nodes.

    Parameters
    ----------
    network : Network
        Network with end-user nodes.
    enduser : str, optional
        ``ci_type`` of the end users. Default: 'people'.
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


def _get_access_loss_mask(network, ci_type, enduser="people"):
    """Boolean mask of the nodes that lost access to ``ci_type``."""
    return network.nodes[_access_col(ci_type, enduser)].isin(ACCESS_LOSS_STATES)


def _get_no_base_access_mask(network, ci_type, enduser="people"):
    """Boolean mask of the nodes without base access to ``ci_type``."""
    return network.nodes[_access_col(ci_type, enduser)] == ACCESS_NO_BASE


def _get_access_undisrupted_mask(network, ci_type, enduser="people"):
    """Boolean mask of the nodes with undisrupted access to ``ci_type``."""
    return network.nodes[_access_col(ci_type, enduser)] == ACCESS_UNDISRUPTED


def _get_dir_affected_mask(network, enduser="people"):
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
    enduser="people",
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
        ``ci_type`` of the end users. Default is 'people'.
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
