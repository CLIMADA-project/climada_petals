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

Plotting TC surge events and topography data for GeoClaw setups
"""

import __main__
import pathlib
from typing import Any, List, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.axes as maxes
import matplotlib.colors as mcolors
import matplotlib.cm as mcolormaps
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import xarray as xr

from climada.hazard import Centroids


def plot_surge_events(
    obj : Any,
    path : Optional[Union[pathlib.Path, str]] = None,
    pad_deg : float = 5.5,
) -> None:
    """Plot areas associated with a track's landfall events

    Parameters
    ----------
    obj : TCSurgeEvents
        The TCSurgeEvents instance containing information about landfall events.
    path : Path or str, optional
        If given, save the plots to the given location. Default: None
    pad_deg : float, optional
        Padding (in degrees) to add around total bounds. Default: 5.5
    """
    total_bounds = (
        min(obj.centroids[:, 1].min(), obj.track["lon"].min()) - pad_deg,
        min(obj.centroids[:, 0].min(), obj.track["lat"].min()) - pad_deg,
        max(obj.centroids[:, 1].max(), obj.track["lon"].max()) + pad_deg,
        max(obj.centroids[:, 0].max(), obj.track["lat"].max()) + pad_deg,
    )
    mid_lon = 0.5 * float(total_bounds[0] + total_bounds[2])
    proj_data = ccrs.PlateCarree()
    proj_plot = ccrs.PlateCarree(central_longitude=mid_lon)
    aspect_ratio = 1.124 * (
        (total_bounds[2] - total_bounds[0])
        / (total_bounds[3] - total_bounds[1])
    )
    fig = plt.figure(
        # the longer side is 10 inches long, the other is scaled according to the aspect ratio
        figsize=(10, 10 / aspect_ratio) if aspect_ratio >= 1 else (aspect_ratio * 10, 10),
        dpi=100,
    )
    axes = fig.add_subplot(111, projection=proj_plot)
    axes.spines['geo'].set_linewidth(0.5)
    axes.set_extent(
        (total_bounds[0], total_bounds[2], total_bounds[1], total_bounds[3]),
        crs=proj_data,
    )

    # add axes tick labels
    grid = axes.gridlines(draw_labels=True, alpha=0.2, transform=proj_data, linewidth=0)
    grid.top_labels = grid.right_labels = False
    grid.xformatter = LONGITUDE_FORMATTER
    grid.yformatter = LATITUDE_FORMATTER

    # plot coastlines
    axes.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.1)

    # plot TC track with masks
    axes.plot(
        obj.track["lon"],
        obj.track["lat"],
        transform=proj_data,
        color='k',
        linewidth=0.5,
    )
    for mask in obj.time_mask_buffered:
        axes.plot(
            obj.track["lon"][mask],
            obj.track["lat"][mask],
            transform=proj_data,
            color='k',
            linewidth=1.5,
        )

    # plot rectangular areas
    linestep = max(0.5, 1 - 0.1 * obj.nevents)
    linew = 1 + linestep * obj.nevents
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i_event, mask in enumerate(obj.time_mask):
        axes.plot(
            obj.track["lon"][mask],
            obj.track["lat"][mask],
            transform=proj_data,
            color=color_cycle[i_event],
            linewidth=3,
        )
        linew -= linestep
        areas = [
            obj.wind_area[i_event],
            obj.landfall_area[i_event],
        ] + obj.surge_areas[i_event]
        for bounds in areas:
            _plot_bounds(
                axes, bounds, transform=proj_data,
                color=color_cycle[i_event], linewidth=linew,
            )

    # plot track data points
    axes.scatter(
        obj.track["lon"],
        obj.track["lat"],
        transform=proj_data,
        s=2,
    )

    # adjust and output to file or screen
    fig.subplots_adjust(left=0.01, bottom=0.03, right=0.99, top=0.99, wspace=0, hspace=0)
    if path is None or not hasattr(__main__, '__file__'):
        plt.show()
    if path is not None:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(path)
        plt.close(fig)


def plot_dems(
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
        axes.plot(track["lon"], track["lat"], transform=proj_data, color='k', linewidth=0.5)
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
    axes.plot(
        [lon_min, lon_min, lon_max, lon_max, lon_min],
        [lat_min, lat_max, lat_max, lat_min, lat_min],
        **kwargs,
    )
