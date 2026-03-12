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

"""

import geopandas as gpd
import pandas as pd
import shapely
import numpy as np
from scipy.spatial import cKDTree
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
import logging
from pathlib import Path
import urllib.request
import requests
import time
import copy as cp
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap

from climada.util import coordinates as u_coords
from climada_petals.util.constants import DICT_SPEEDS

# Critical infrastructure charactersitics constants
LINE_EXPOSURES = ["road", "rail"]

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
# list(Line2D.markers.keys())

STATUS_MAP = {
    "access undisrupted": -2,
    "access new source": -1,
    "no base access": 0,
    "access disrupted via": 1,
    "access disrupted source": 2,
}
LOGGER = logging.getLogger(__name__)


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
    axes=None,
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

    axes.legend()

    # Add a single status colorbar if plotting functional status
    if use_status_colorbar:
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
        gdf_enduser[service].isin(["access disrupted via", "access disrupted source"])
    ].counts.sum()
    n_ppl_no_base_access = gdf_enduser[
        gdf_enduser[service].isin(["no base access"])
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
    cbar.set_ticklabels(
        STATUS_MAP.keys(), rotation=30
    )  # ,fontweight="bold"#fontsize=18
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


def _preselect_destinations(vs_assign, vs_base, dist_thresh):
    """Preselect candidate destination indices within a distance threshold.

    Uses a cKDTree to find, for each point in ``vs_assign``, all points
    in ``vs_base`` within ``dist_thresh``.

    Parameters
    ----------
    vs_assign : gpd.GeoDataFrame
        GeoDataFrame of points to assign.
    vs_base : gpd.GeoDataFrame
        GeoDataFrame of candidate destination points.
    dist_thresh : float
        Maximum distance threshold for candidate matching.

    Returns
    -------
    list of list of int
        For each point in ``vs_assign``, a list of indices into ``vs_base``
        within ``dist_thresh``.
    """
    points_base = np.array([(x.x, x.y) for x in vs_base["geometry"]])
    point_tree = cKDTree(points_base)

    points_assign = np.array([(x.x, x.y) for x in vs_assign["geometry"]])
    ix_matches = []
    for assign_loc in points_assign:
        ix_matches.append(point_tree.query_ball_point(assign_loc, dist_thresh))
    return ix_matches


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


def window_from_extent(xmin, ymin, xmax, ymax, transform):
    """Create a rasterio Window from geographic extent bounds.

    Parameters
    ----------
    xmin : float
        Minimum x coordinate (left).
    ymin : float
        Minimum y coordinate (bottom).
    xmax : float
        Maximum x coordinate (right).
    ymax : float
        Maximum y coordinate (top).
    transform : rasterio.Affine
        Affine transform of the raster.

    Returns
    -------
    rasterio.windows.Window
        Window corresponding to the given extent.
    """
    col_start, row_start = ~transform * (xmin, ymax)
    col_stop, row_stop = ~transform * (xmax, ymin)
    return Window.from_slices((row_start, row_stop), (col_start, col_stop))


def _resample_res(filepath, upscale_factor, nodata, extent=None):
    """Resample a raster file to a different resolution.

    Reads a raster, optionally crops to an extent, and resamples by the
    given upscale factor using average resampling.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the raster file.
    upscale_factor : float
        Factor by which to upscale the resolution. Values > 1 increase
        the number of pixels (finer resolution).
    nodata : float
        Nodata value to replace with 0.
    extent : tuple of float, optional
        Geographic extent as (xmin, ymin, xmax, ymax) to crop the raster.
        Default: None (use full raster).

    Returns
    -------
    arr : np.ndarray
        Resampled raster array with nodata replaced by 0.
    transform : rasterio.Affine
        Affine transform of the resampled raster.
    """

    with rasterio.open(filepath) as dataset:
        # Get the initial transform and metadata
        transform = dataset.transform
        meta = dataset.meta.copy()

        if extent:
            # Create a rasterio window from the extent
            window = from_bounds(*extent, transform=transform)
        else:
            # If no extent is provided, use the full dataset
            window = rasterio.windows.Window(0, 0, dataset.width, dataset.height)

        # Get the windowed transform and dimensions
        window_transform = dataset.window_transform(window)
        window_width = int(window.width * upscale_factor)
        window_height = int(window.height * upscale_factor)

        # Read and resample the data within the window
        arr = dataset.read(
            out_shape=(dataset.count, window_height, window_width),
            resampling=Resampling.average,
            window=window,
        )
        # Update metadata for the cropped and resampled array
        meta.update(
            height=window_height, width=window_width, transform=window_transform
        )

        # Adjust the transform for the scaling
        transform = window_transform * rasterio.Affine.scale(
            1 / upscale_factor, 1 / upscale_factor
        )

    # Replace nodata values with 0 and adjust array values
    arr = np.where(arr == nodata, 0, arr)
    arr = arr * (1 / upscale_factor) ** 2

    return arr, transform


def load_resampled_raster(filepath, upscale_factor, nodata=-99999.0, extent=None):
    """Load a raster file, resample it, and return as a GeoDataFrame.

    Resamples the raster by the given factor and converts non-zero cells
    to point geometries with population counts. Applies a correction factor
    to account for aggregation over/under-estimation.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the raster file.
    upscale_factor : float
        Factor by which to upscale the resolution.
    nodata : float, optional
        Nodata value in the raster. Default: -99999.0.
    extent : tuple of float, optional
        Geographic extent as (xmin, ymin, xmax, ymax). Default: None.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with 'counts' and 'geometry' columns for non-zero
        raster cells.
    """

    arr, transform = _resample_res(filepath, upscale_factor, nodata, extent)

    grid = u_coords.raster_to_meshgrid(transform, arr.shape[-1], arr.shape[-2])
    gdf = gpd.GeoDataFrame(
        {
            "counts": arr.squeeze().flatten(),
            "geometry": gpd.points_from_xy(grid[0].flatten(), grid[1].flatten()),
        }
    )
    gdf = gdf[gdf.counts != 0].reset_index(drop=True)

    # manual correction for over-estimate after aggregation:
    arr_orig, __ = _resample_res(filepath, 1, nodata, extent)
    corr_factor = arr_orig.squeeze().flatten().sum() / arr.squeeze().flatten().sum()
    gdf["counts"] = gdf.counts * corr_factor

    return gdf


# =============================================================================
# General results analysis util functions
# =============================================================================


def number_noservice(service, graph, enduser="people"):
    """Calculate the population without access to a given service.

    Parameters
    ----------
    service : str
        Service name (e.g. 'power', 'healthcare').
    graph : GraphCalcs
        Graph calculation object with an igraph graph attribute.
    enduser : str, optional
        The type of end user to calculate access for. Default is 'people'.

    Returns
    -------
    float
        Total population without access to the service.
    """

    no_service = 1 - np.array(
        graph.graph.vs.select(ci_type=enduser)[f"actual_supply_{service}_{enduser}"]
    )
    pop = np.array(graph.graph.vs.select(ci_type=enduser)["counts"])
    return (no_service * pop).sum()


def number_noservices(graph, services, enduser="people"):
    """Calculate the population without access for multiple services.

    Parameters
    ----------
    graph : GraphCalcs
            Graph calculation object with an igraph graph attribute.
    services : list of str
        Services to evaluate.
    enduser : str, optional
        The type of end user to calculate access for. Default is 'people'.

    Returns
    -------
    dict
        Dictionary mapping each service name to the population count
        without access.
    """

    servstats_dict = {}
    for service in services:
        servstats_dict[service] = number_noservice(service, graph, enduser)
    return servstats_dict


def number_noservice_df(service, df, enduser="people"):
    """Calculate the population without access to a service from a DataFrame.

    Counts population having 0 or negative values in the service state column.

    Parameters
    ----------
    service : str
        Service name to check availability for (e.g. 'power', 'healthcare').
    df : gpd.GeoDataFrame or pd.DataFrame
        DataFrame containing population clusters with columns 'ci_type',
        'counts', and the service state columns.
    enduser : str, optional
        The type of end user to calculate access for. Default is 'people'.

    Returns
    -------
    float
        Total population without access to the service.

    See Also
    --------
    number_noservice : Same calculation performed on the igraph graph.
    """
    return df[
        (df.ci_type == enduser) & (df[f"actual_supply_{service}_{enduser}"] <= 0)
    ].counts.sum()


def number_noservices_df(
    df,
    services,
    enduser="people",
):
    """Calculate the population without access for multiple services from a DataFrame.

    Parameters
    ----------
    df : gpd.GeoDataFrame or pd.DataFrame
        DataFrame containing population clusters and service state columns.
    services : list of str
        Services to evaluate.
    enduser : str, optional
        The type of end user to calculate access for. Default is 'people'.

    Returns
    -------
    dict
        Dictionary mapping each service name to the population count
        without access.

    See Also
    --------
    number_noservices : Same calculation performed on the igraph graph.
    """
    servstats_dict = {}
    for service in services:
        servstats_dict[service] = number_noservice_df(service, df, enduser)
    return servstats_dict


def disaster_impact_service_geoseries(service, pre_graph, post_graph, enduser="people"):
    """Get geometries of population clusters that lost access to a service.

    Parameters
    ----------
    service : str
        Service name (e.g. 'power', 'healthcare').
    pre_graph : GraphCalcs
        Graph calculation object representing the pre-disaster state.
    post_graph : GraphCalcs
        Graph calculation object representing the post-disaster state.
    enduser : str, optional
        The type of end user to calculate access for. Default is 'people'.

    Returns
    -------
    gpd.GeoSeries
        Geometries of population clusters that lost service access.
    """

    no_service_post = 1 - np.array(
        post_graph.graph.vs.select(ci_type=enduser)[
            f"actual_supply_{service}_{enduser}"
        ]
    )
    no_service_pre = 1 - np.array(
        pre_graph.graph.vs.select(ci_type=enduser)[f"actual_supply_{service}_{enduser}"]
    )
    geom = np.array(post_graph.graph.vs.select(ci_type=enduser)["geom_wkt"])
    return gpd.GeoSeries.from_wkt(
        geom[np.where((no_service_post - no_service_pre) > 0)]
    )


def disaster_impact_service(service, pre_graph, post_graph, enduser="people"):
    """Calculate population losing access to a service due to a disaster.

    Parameters
    ----------
    service : str
        Service name (e.g. 'power', 'healthcare').
    pre_graph : GraphCalcs
        Graph calculation object representing the pre-disaster state.
    post_graph : GraphCalcs
        Graph calculation object representing the post-disaster state.
    enduser : str, optional
        The type of end user to calculate access for. Default is 'people'.

    Returns
    -------
    float
        Total population that lost access to the service.
    """

    no_service_post = 1 - np.array(
        post_graph.graph.vs.select(ci_type=enduser)[
            f"actual_supply_{service}_{enduser}"
        ]
    )
    no_service_pre = 1 - np.array(
        pre_graph.graph.vs.select(ci_type=enduser)[f"actual_supply_{service}_{enduser}"]
    )
    pop = np.array(pre_graph.graph.vs.select(ci_type=enduser)["counts"])
    return ((no_service_post - no_service_pre) * pop).sum()


def disaster_impact_allservices(
    pre_graph,
    post_graph,
    services,
    enduser="people",
):
    """Calculate population losing access across all services due to a disaster.

    Parameters
    ----------
    pre_graph : GraphCalcs
        Graph calculation object representing the pre-disaster state.
    post_graph : GraphCalcs
        Graph calculation object representing the post-disaster state.
    services : list of str
        Services to evaluate
    enduser : str, optional
        The type of end user to calculate access for. Default is 'people'.

    Returns
    -------
    dict
        Dictionary mapping each service to the change in population
        without access (post minus pre).
    """

    dict_pre = number_noservices(pre_graph, services, enduser=enduser)
    dict_post = number_noservices(post_graph, services, enduser=enduser)
    dict_delta = {}
    for key, value in dict_post.items():
        dict_delta[key] = value - dict_pre[key]
    return dict_delta


def disaster_impact_allservices_df(
    df_pre,
    df_post,
    services,
    enduser="people",
):
    """Calculate population losing access across all services from DataFrames.

    Parameters
    ----------
    df_pre : gpd.GeoDataFrame or pd.DataFrame
        DataFrame representing the pre-disaster state.
    df_post : gpd.GeoDataFrame or pd.DataFrame
        DataFrame representing the post-disaster state.
    services : list of str
        Services to evaluate.
    enduser : str, optional
        The type of end user to calculate access for. Default is 'people'.

    Returns
    -------
    dict
        Dictionary mapping each service (with '_access' suffix) to the
        change in population without access. If 'people' was in services,
        also includes the directly impacted population count.

    See Also
    --------
    disaster_impact_allservices : Same calculation on igraph graphs.
    """
    services = cp.deepcopy(services)
    dict_delta = {}
    if enduser in services:
        services.remove(enduser)
        # directly affected people
        dict_delta[enduser] = sum(df_post[df_post.ci_type == enduser].imp_dir)
    dict_pre = number_noservices_df(df_pre, services, enduser=enduser)
    dict_post = number_noservices_df(df_post, services, enduser=enduser)
    for key, value in dict_post.items():
        dict_delta[key + "_access"] = value - dict_pre[key]
    return dict_delta


def get_graphstats(graph):
    """Get summary statistics of a network graph.

    Parameters
    ----------
    graph : GraphCalcs
        Graph calculation object with an igraph graph attribute.

    Returns
    -------
    dict
        Dictionary with keys 'no_edges', 'no_nodes', 'edge_types',
        and 'node_types' (Counter objects for type distributions).
    """
    from collections import Counter

    stats_dict = {}
    stats_dict["no_edges"] = len(graph.graph.es)
    stats_dict["no_nodes"] = len(graph.graph.vs)
    stats_dict["edge_types"] = Counter(graph.graph.es["ci_type"])
    stats_dict["node_types"] = Counter(graph.graph.vs["ci_type"])
    return stats_dict


# =============================================================================
# Worldpop Data
# =============================================================================
def get_worldpop_data(iso3, save_path, res=100):
    """Download WorldPop population raster data for a country.

    Downloads UN-adjusted population data for 2020 from WorldPop at the
    specified resolution. Skips download if the file already exists.

    Parameters
    ----------
    iso3 : str
        ISO 3166-1 alpha-3 country code.
    save_path : str or pathlib.Path
        Directory path to save the downloaded file.
    res : int, optional
        Resolution in metres: 100 or 1000. Default: 100.
    """

    if res == 1000:
        download_url = (
            "https://data.worldpop.org/GIS/Population/"
            + f"Global_2000_2020_1km_UNadj/2020/{iso3}/"
            + f"{iso3.lower()}_ppp_2020_1km_Aggregated_UNadj.tif"
        )
    elif res == 100:
        download_url = (
            "https://data.worldpop.org/GIS/Population/"
            + f"Global_2000_2020/2020/{iso3}/{iso3.lower()}_ppp_2020_UNadj.tif"
        )

    local_filepath = Path(save_path, download_url.split("/")[-1])

    if not Path(local_filepath).is_file():
        LOGGER.info(f"Downloading file as {local_filepath}")
        urllib.request.urlretrieve(download_url, local_filepath)
    else:
        LOGGER.info(f"file already exists as {local_filepath}")


def get_pop_cutoff(gdf_people, cutoff):
    """Find population count cutoff for filtering low-density grid points.

    Determines the maximum population value per grid point that cumulatively
    accounts for less than a specified fraction of the total population.

    Parameters
    ----------
    gdf_people : gpd.GeoDataFrame
        GeoDataFrame with a 'counts' column representing population per
        grid point.
    cutoff : float
        Cumulative population fraction threshold (0 to 1).

    Returns
    -------
    float
        Left boundary of the bin interval at which the cumulative
        population fraction first exceeds ``cutoff``.
    """
    # redefine bins as high res data might have less than 100 max count values
    bins = list(np.arange(start=0, stop=gdf_people["counts"].max(), step=10))
    # bins = [0, 10, 20, 35, 50, 75]
    # bins.extend(
    #    list(np.linspace(start=100, stop=gdf_people["counts"].max(), num=30)))
    df_cum = (
        gdf_people.groupby(pd.cut(gdf_people["counts"], bins)).sum(numeric_only=True)
        / gdf_people["counts"].sum()
    )
    cutoff_bool = (df_cum.cumsum() >= cutoff).counts.values
    cutoff_interval = df_cum.index.categories[cutoff_bool][0]
    return cutoff_interval.left


# =============================================================================
# Distance Threshold setting
# =============================================================================


def set_travel_distance_threshs(df_dependencies, iso3, hrs_max=1):
    """Set road travel distance thresholds based on country-specific speeds.

    Sets the distance threshold (in metres) for health and education
    dependencies to people, based on the average road speed in the
    given country and a maximum travel time.

    Data taken from *Road Quality and Mean Speed Score*,
    Author/Editor: Marian Moszoro; Mauricio Soto,
    ISBN: 9798400210440/1018-5941.

    Parameters
    ----------
    df_dependencies : pd.DataFrame
        Dependencies DataFrame with 'source', 'target', and 'thresh_dist'
        columns.
    iso3 : str
        ISO 3166-1 alpha-3 country code.
    hrs_max : float, optional
        Maximum travel time in hours. Default: 1.

    Returns
    -------
    pd.DataFrame
        Updated dependencies DataFrame with 'thresh_dist' set for
        health/education-to-people dependencies.
    """
    try:
        thresh_dist = int(DICT_SPEEDS[iso3] * 1000 * hrs_max)
    except KeyError:
        thresh_dist = int(DICT_SPEEDS["other"] * 1000 * hrs_max)

    df_dependencies.loc[
        (df_dependencies.source == "health")
        | (df_dependencies.source == "education")
        & (df_dependencies.target == "people"),
        "thresh_dist",
    ] = thresh_dist

    return df_dependencies
