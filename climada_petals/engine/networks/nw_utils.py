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
def population_plot(self, axes=None, projection=ccrs.PlateCarree(), **kwargs):
    plot_df = self.nodes[self.nodes.ci_type == "people"]
    if axes is None:
        fig, axes = plt.subplots(1, 1, subplot_kw=dict(projection=projection))
    else:
        fig = axes.get_figure()
    axes.scatter(
        plot_df.geometry.x,
        plot_df.geometry.y,
        s=plot_df["value"],
        label="population",
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
    axes=None,
    projection=ccrs.PlateCarree(),
    pop_kwargs=dict(),
    ci_kwargs=dict(),
    cbar_kwargs=dict(),
):
    """Infrastructure plots"""
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

        if ci_type == "people":
            fig, axes = population_plot(
                self, axes=axes, projection=projection, zorder=0, **pop_kwargs
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
            marker = f"${ci_type[0].upper()}$"
            plot_df.plot(
                plot_col,
                ax=axes,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                transform=projection,
                label=ci_type,
                markersize=200,
                zorder=i + 1,
                marker=marker,
                edgecolor="white",
                linewidth=0.05,
                **ci_kwargs,
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
    dep_list = [
        dep
        for dep in self.edges.ci_type
        for ci_type in ci_types
        if "dependency_" in dep and ci_type in dep
    ]
    return np.unique(dep_list)


def dep_plot(self, ci_types=None, axes=None, projection=ccrs.PlateCarree(), **kwargs):
    """dependencies plots"""
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
    axes=None,
    projection=ccrs.PlateCarree(),
    plot_kwargs=dict(),
    cbar_kwargs=dict(),
):
    if axes is None:
        fig, axes = plt.subplots(1, 1, subplot_kw=dict(projection=projection))
    else:
        fig = axes.figure

    cmap = plot_kwargs.pop(
        "cmap", ListedColormap(["#046582", "#158204", "grey", "#CF8018", "#BB8082"])
    )
    gdf_ppl = self.nodes[self.nodes.ci_type == "people"]

    service = f"access_state_{ci_type}_people"

    # lowercase for safety
    cvals = gdf_ppl[service].str.lower().map(STATUS_MAP).fillna(0)
    scatter = axes.scatter(
        gdf_ppl.geometry.x,
        gdf_ppl.geometry.y,
        c=cvals,
        s=gdf_ppl["counts"] / max(gdf_ppl["counts"]) * 100,
        transform=projection,
        cmap=cmap,
        vmin=-2,
        vmax=2,
        **plot_kwargs,
    )
    n_ppl_access_loss = gdf_ppl[
        gdf_ppl[service].isin(["access disrupted via", "access disrupted source"])
    ].counts.sum()
    n_ppl_no_base_access = gdf_ppl[
        gdf_ppl[service].isin(["no base access"])
    ].counts.sum()

    text = (
        f"{n_ppl_no_base_access:.2f} people without base access to {ci_type}"
        + f"\n{n_ppl_access_loss:.2f} people loosing access to {ci_type}"
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
    """
    create straight shapely LineString geometries between lists of
    from and to nodes, to be added to newly created edges as attributes
    """
    return [
        shapely.geometry.LineString([geom_from, geom_to])
        for geom_from, geom_to in zip(vs_geoms_from, vs_geoms_to)
    ]


def _preselect_destinations(vs_assign, vs_base, dist_thresh):
    points_base = np.array([(x.x, x.y) for x in vs_base["geometry"]])
    point_tree = cKDTree(points_base)

    points_assign = np.array([(x.x, x.y) for x in vs_assign["geometry"]])
    ix_matches = []
    for assign_loc in points_assign:
        ix_matches.append(point_tree.query_ball_point(assign_loc, dist_thresh))
    return ix_matches


def _ckdnearest(vs_assign, gdf_base, k=1, dist_thresh=np.inf):
    """
    see https://gis.stackexchange.com/a/301935

    Parameters
    ----------
    vs_assign : gpd.GeoDataFrame or Point
    gdf_base : gpd.GeoDataFrame

    Returns
    ----------

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
    col_start, row_start = ~transform * (xmin, ymax)
    col_stop, row_stop = ~transform * (xmax, ymin)
    return Window.from_slices((row_start, row_stop), (col_start, col_stop))


def _resample_res(filepath, upscale_factor, nodata, extent=None):

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


def service_dict():
    return {
        "power": "actual_supply_power_line_people",
        "healthcare": "actual_supply_healthcare_people",
        "education": "actual_supply_education_people",
        "telecom": "actual_supply_celltower_people",
        "road": "actual_supply_road_people",
        "water": "actual_supply_wastewater_people",
    }


def number_noservice(service, graph):

    no_service = 1 - np.array(
        graph.graph.vs.select(ci_type="people")[service_dict()[service]]
    )
    pop = np.array(graph.graph.vs.select(ci_type="people")["counts"])
    return (no_service * pop).sum()


def number_noservices(
    graph, services=["power", "healthcare", "education", "telecom", "mobility", "water"]
):

    servstats_dict = {}
    for service in services:
        servstats_dict[service] = number_noservice(service, graph)
    return servstats_dict


def number_noservice_df(service, df, service_dict=service_dict()):
    """
    Number of population having '0' or '-1' in service state for a respective service

    Parameters
    -----------
    service : str
        the service to check (in-)availability for
    df : dataframe
        a (geo-)dataframe containing information on population clusters
        and the respective state of certain sevices, e.g. from a df_res extracted
        for saving from disrupted graph, or from network.nodes instance

    Note
    -----
    same as number_noservice, just that it's performed on a df, not on the graph
    """
    return df[(df.ci_type == "people") & (df[service_dict[service]] <= 0)].counts.sum()


def number_noservices_df(
    df, services=["power", "healthcare", "education", "telecom", "mobility", "water"]
):
    """
    Note
    -----
    same as number_noservices, just that it's performed on a df, not on the graph
    """
    servstats_dict = {}
    for service in services:
        servstats_dict[service] = number_noservice_df(service, df)
    return servstats_dict


def disaster_impact_service_geoseries(service, pre_graph, post_graph):

    no_service_post = 1 - np.array(
        post_graph.graph.vs.select(ci_type="people")[service_dict()[service]]
    )
    no_service_pre = 1 - np.array(
        pre_graph.graph.vs.select(ci_type="people")[service_dict()[service]]
    )
    geom = np.array(post_graph.graph.vs.select(ci_type="people")["geom_wkt"])
    return gpd.GeoSeries.from_wkt(
        geom[np.where((no_service_post - no_service_pre) > 0)]
    )


def disaster_impact_service(service, pre_graph, post_graph):

    no_service_post = 1 - np.array(
        post_graph.graph.vs.select(ci_type="people")[service_dict()[service]]
    )
    no_service_pre = 1 - np.array(
        pre_graph.graph.vs.select(ci_type="people")[service_dict()[service]]
    )
    pop = np.array(pre_graph.graph.vs.select(ci_type="people")["counts"])
    return ((no_service_post - no_service_pre) * pop).sum()


def disaster_impact_allservices(
    pre_graph,
    post_graph,
    services=["power", "healthcare", "education", "telecom", "mobility", "water"],
):

    dict_pre = number_noservices(pre_graph, services)
    dict_post = number_noservices(post_graph, services)
    dict_delta = {}
    for key, value in dict_post.items():
        dict_delta[key] = value - dict_pre[key]
    return dict_delta


def disaster_impact_allservices_df(
    df_pre,
    df_post,
    services=["power", "healthcare", "education", "telecom", "mobility", "water"],
):
    """
    Note
    -----
    same as disaster_impact_allservices, just that it's performed on a df,
    not on the graph
    """
    services = cp.deepcopy(services)
    dict_delta = {}
    if "people" in services:
        services.remove("people")
        # directly affected people
        dict_delta["people"] = sum(df_post[df_post.ci_type == "people"].imp_dir)
    dict_pre = number_noservices_df(df_pre, services)
    dict_post = number_noservices_df(df_post, services)
    for key, value in dict_post.items():
        dict_delta[key + "_access"] = value - dict_pre[key]
    return dict_delta


def get_graphstats(graph):
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
    """
    find the maximum population value per grid point which accounts cumulatively across
    the entire gdf for less than a cutoff fraction of the entire population number
    to decrease the
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
    """
    set road travel distance threshold according to the average distance
    covered within a specified amount of hours in the respective country.

    Data taken from Road Quality and Mean Speed Score
    Author/Editor:Marian Moszoro ; Mauricio Soto
    ISBN: 9798400210440/1018-5941
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
