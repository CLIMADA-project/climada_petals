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

Plots of networks: infrastructure, end users, dependencies and access states.

The functions are also methods of :class:`~climada_petals.engine.networks.nw_base.Network`
(``plot_infra``, ``plot_enduser``, ``plot_dep``, ``plot_access``).

Coordinates are plotted in the CRS of the network (``transform``); the
``projection`` argument only sets the map projection of new axes.
"""

import logging

import cartopy.crs as ccrs
import matplotlib as mpl
import numpy as np
import pyproj
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from climada_petals.engine.networks.nw_utils import (
    ACCESS_DISRUPTED,
    ACCESS_DISRUPTED_SOURCE,
    ACCESS_DISRUPTED_VIA,
    ACCESS_LOSS_STATES,
    ACCESS_NEW_SOURCE,
    ACCESS_NO_BASE,
    ACCESS_UNDISRUPTED,
    DEF_END_USER,
    LINE_EXPOSURES,
    POP_COL,
    _added_mask,
    _dependency_name,
)

LOGGER = logging.getLogger(__name__)

MPL_MARKERS = list(mpl.markers.MarkerStyle.markers.keys())
# matplotlib line styles (other names raise a ValueError in plot())
MPL_LINE_STYLES = ["solid", "dashed", "dashdot", "dotted"]

#: colour class of each access state in :func:`access_plot` (-2 to 2)
STATUS_MAP = {
    ACCESS_UNDISRUPTED: -2,
    ACCESS_NEW_SOURCE: -1,
    ACCESS_NO_BASE: 0,
    ACCESS_DISRUPTED_VIA: 1,
    ACCESS_DISRUPTED_SOURCE: 2,
    ACCESS_DISRUPTED: 2,  # also a failure at the source side
}


def _data_crs(network):
    """Cartopy CRS of the network's coordinates (``transform`` of the plots)."""
    if network.crs is None:
        return ccrs.PlateCarree()
    crs = pyproj.CRS(network.crs)
    if crs.is_geographic:
        return ccrs.PlateCarree()
    epsg = crs.to_epsg()
    if epsg is not None:
        try:
            return ccrs.epsg(epsg)
        except ValueError:  # not a projection cartopy can build from the code
            pass
    return ccrs.CRS(crs)


def _new_axes(axes, projection):
    """Figure and axes: new ones with ``projection`` if ``axes`` is None."""
    if axes is None:
        return plt.subplots(1, 1, subplot_kw=dict(projection=projection))
    return axes.get_figure(), axes


def _marker_sizes(population, max_size=100):
    """Marker areas proportional to the population, the largest ``max_size``."""
    largest = population.max() if len(population) else 0
    if not largest > 0:
        return np.zeros(len(population))
    return population / largest * max_size


def enduser_plot(
    network,
    enduser=DEF_END_USER,
    axes=None,
    projection=ccrs.PlateCarree(),
    pop_col=POP_COL,
    marker_scale=1.0,
    **kwargs,
):
    """Plot end-user nodes as a scatter plot, sized by population.

    Parameters
    ----------
    network : Network
        Network with end-user nodes.
    enduser : str, optional
        ``ci_type`` of the end users. Default: :data:`DEF_END_USER`.
    axes : cartopy.mpl.geoaxes.GeoAxes, optional
        Axes to plot on. If None, a new figure and axes are created.
    projection : cartopy.crs.Projection, optional
        Map projection of new axes. Default: PlateCarree.
    pop_col : str, optional
        Node column with the population (marker size). Default: :data:`POP_COL`.
    marker_scale : float, optional
        Factor applied to all marker sizes, e.g. 2 for markers twice as
        large. Marker sizes stay proportional to the population: the most
        populated node has an area of ``100 * marker_scale`` points^2.
        Default: 1.
    **kwargs
        Passed to ``axes.scatter``; they override the defaults (grey,
        ``alpha=0.5``). ``s`` replaces the population-based sizes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : cartopy.mpl.geoaxes.GeoAxes
        The axes with the population plot.
    """
    plot_df = network.nodes[network.nodes.ci_type == enduser]
    fig, axes = _new_axes(axes, projection)
    style = dict(
        s=_marker_sizes(plot_df[pop_col], 100 * marker_scale),
        label=enduser,
        alpha=0.5,
        color="grey",
    )
    style.update(kwargs)
    axes.scatter(
        plot_df.geometry.x, plot_df.geometry.y, transform=_data_crs(network), **style
    )
    return fig, axes


def infra_plot(
    network,
    ci_types=None,
    plot_col="ci_type",
    enduser=DEF_END_USER,
    mask_func_ci=True,
    road_mask=True,
    axes=None,
    legend=True,
    legend_loc="upper right",
    projection=ccrs.PlateCarree(),
    pop_kwargs=None,
    ci_kwargs=None,
    cbar_kwargs=None,
    marker_scale=1.0,
):
    """Plot critical infrastructure network elements.

    Plots nodes and edges for each CI type with distinct markers, colors,
    and line styles. Point infrastructure (not end users, not line types) is
    rendered as a coloured disc with a white letter identifier on top for
    high visibility. When ``plot_col`` is not 'ci_type', a functional status
    colorbar (disrupted/functioning) is added.

    Parameters
    ----------
    network : Network
        Network instance containing nodes and edges.
    ci_types : list of str, optional
        CI types to plot. If None, all unique ci_types in nodes are used.
    plot_col : str, optional
        Column to use for coloring. Default: 'ci_type'.
    enduser : str, optional
        End-user type, plotted as background with :func:`enduser_plot`.
        Default: :data:`DEF_END_USER`.
    mask_func_ci : bool, optional
        Plot only infrastructure that is not fully functional
        (``func_tot < 1``). Ignored if there is no ``func_tot`` column (before
        ``initialize_base_state``). Default: True.
    road_mask : bool, optional
        Do not plot the road edges added by the module (links to facilities
        and people, see :data:`ADDED_COL`). Default: True.
    axes : cartopy.mpl.geoaxes.GeoAxes, optional
        Axes to plot on. If None, a new figure and axes are created.
    legend : bool, optional
        Add a legend. Default: True.
    legend_loc : str, optional
        Location of the legend. Default: 'upper right'.
    projection : cartopy.crs.Projection, optional
        Map projection of new axes. Default: PlateCarree.
    pop_kwargs : dict, optional
        Keyword arguments for :func:`enduser_plot`.
    ci_kwargs : dict, optional
        Keyword arguments for the CI element plots (lines and point discs);
        they override the defaults. ``colors`` sets the colour cycle of the CI
        types. Default colours: ``tab20``.
    cbar_kwargs : dict, optional
        Keyword arguments for the colorbar.
    marker_scale : float, optional
        Factor applied to all marker sizes, e.g. 2 for markers twice as
        large: the discs of point infrastructure (with their letter) and the
        end-user markers, which stay proportional to the population.
        ``pop_kwargs={"marker_scale": ...}`` sets the end users separately.
        Default: 1.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : cartopy.mpl.geoaxes.GeoAxes
        The axes with the infrastructure plot.
    """
    # copies: the caller's dicts are not modified by the pop() calls below
    pop_kwargs = dict(pop_kwargs or {})
    ci_kwargs = dict(ci_kwargs or {})
    cbar_kwargs = dict(cbar_kwargs or {})
    if ci_types is None:
        ci_types = network.nodes.ci_type.unique()
    fig, axes = _new_axes(axes, projection)
    transform = _data_crs(network)
    colors = ci_kwargs.pop("colors", mpl.cm.tab20.colors)
    pop_kwargs.setdefault("marker_scale", marker_scale)

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
        color = colors[i % len(colors)]

        # Set colormap for this CI type
        if plot_col == "ci_type":
            cmap = mpl.colors.ListedColormap([color])
        else:
            cmap = status_cmap

        if ci_type == enduser:
            fig, axes = enduser_plot(
                network,
                enduser=enduser,
                axes=axes,
                projection=projection,
                zorder=0,
                **pop_kwargs,
            )
        elif ci_type in LINE_EXPOSURES:
            plot_df = network.edges[network.edges.ci_type == ci_type]
            if mask_func_ci and "func_tot" in plot_df.columns:
                plot_df = plot_df[plot_df["func_tot"] < 1]
            if road_mask and ci_type == "road":
                plot_df = plot_df[~_added_mask(plot_df)]

            line_kw = dict(
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                transform=transform,
                label=ci_type,
                linestyle=line_style,
                zorder=1,
            )
            line_kw.update(ci_kwargs)
            plot_df.plot(plot_col, ax=axes, **line_kw)
        else:
            plot_df = network.nodes[network.nodes.ci_type == ci_type]
            if mask_func_ci and "func_tot" in plot_df.columns:
                plot_df = plot_df[plot_df["func_tot"] < 1]
            marker_letter = f"${ci_type[0].upper()}$"

            # Background coloured disc for visibility
            bg_kw = dict(
                s=350 * marker_scale,
                marker="o",
                transform=transform,
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
            bg_kw.update(ci_kwargs)
            axes.scatter(plot_df.geometry.x, plot_df.geometry.y, **bg_kw)
            # White letter marker on top
            axes.scatter(
                plot_df.geometry.x,
                plot_df.geometry.y,
                s=100 * marker_scale,
                color="white",
                marker=marker_letter,
                transform=transform,
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


def _get_dependencies(network, ci_types):
    """Dependency edge types with a source or target in ``ci_types``.

    The edge types are matched exactly against the ``dependency_{source}_{target}``
    names of all pairs of CI types in the network (no substring matching, so
    e.g. 'power' does not select the dependencies of 'power_line').

    Parameters
    ----------
    network : Network
        Network instance containing edges.
    ci_types : list of str
        CI types to select dependency edges for.

    Returns
    -------
    np.ndarray
        Sorted unique dependency edge types present in the edges.
    """
    edge_types = set(network.edges["ci_type"].dropna().unique())
    all_types = set(network.nodes["ci_type"].dropna().unique()) | edge_types
    wanted = {
        _dependency_name(source, target)
        for source in all_types
        for target in all_types
        if source in ci_types or target in ci_types
    }
    return np.array(sorted(edge_types & wanted))


def dep_plot(
    network, ci_types=None, axes=None, projection=ccrs.PlateCarree(), **kwargs
):
    """Plot dependency edges between CI types.

    Parameters
    ----------
    network : Network
        Network instance containing edges with dependency types.
    ci_types : list of str, optional
        CI types to plot dependencies for. If None, all unique ci_types
        in nodes are used.
    axes : cartopy.mpl.geoaxes.GeoAxes, optional
        Axes to plot on. If None, a new figure and axes are created.
    projection : cartopy.crs.Projection, optional
        Map projection of new axes. Default: PlateCarree.
    **kwargs
        Passed to ``GeoDataFrame.plot``; they override the defaults
        (``alpha=0.5``, ``zorder=10``). ``colors`` sets the colour cycle
        (default ``tab20``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : cartopy.mpl.geoaxes.GeoAxes
        The axes with the dependency plot.
    """
    if ci_types is None:
        ci_types = network.nodes.ci_type.unique()
    fig, axes = _new_axes(axes, projection)
    transform = _data_crs(network)
    kwargs = dict(kwargs)
    colors = kwargs.pop("colors", mpl.cm.tab20.colors)
    for i, dep in enumerate(_get_dependencies(network, ci_types)):
        style = dict(color=colors[i % len(colors)], label=dep, alpha=0.5, zorder=10)
        style.update(kwargs)
        plot_df = network.edges[network.edges.ci_type == dep]
        plot_df.plot(ax=axes, transform=transform, **style)
    axes.legend()
    return fig, axes


def access_plot(
    network,
    ci_type,
    enduser=DEF_END_USER,
    access_states=None,
    axes=None,
    projection=ccrs.PlateCarree(),
    plot_kwargs=None,
    cbar_kwargs=None,
    pop_col=POP_COL,
    marker_scale=1.0,
):
    """Plot the access state of end users to a CI type.

    Visualises the access state of the end-user nodes to a CI service,
    color-coded by access state (:data:`STATUS_MAP`) and sized by population.
    States not in :data:`STATUS_MAP` are shown in the colour of
    'no base access', with a warning.

    Parameters
    ----------
    network : Network
        Network instance containing end-user nodes and access state
        columns.
    ci_type : str
        The CI type to assess access for (e.g. 'healthcare', 'power').
    enduser : str, optional
        The type of end user to plot access for. Default:
        :data:`DEF_END_USER`.
    access_states : list of str, optional
        Subset of access states to plot. If None, all states are plotted.
        Possible values: the keys of :data:`STATUS_MAP`.
    axes : cartopy.mpl.geoaxes.GeoAxes, optional
        Axes to plot on. If None, a new figure and axes are created.
    projection : cartopy.crs.Projection, optional
        Map projection of new axes. Default: PlateCarree.
    plot_kwargs : dict, optional
        Additional keyword arguments for ``axes.scatter``; they override the
        defaults. ``cmap`` replaces the colours of the 5 classes, ``s`` the
        population-based sizes.
    cbar_kwargs : dict, optional
        Additional keyword arguments for the colorbar.
    pop_col : str, optional
        Node column with the population (marker size and the numbers in the
        text box). Default: :data:`POP_COL`.
    marker_scale : float, optional
        Factor applied to all marker sizes, e.g. 2 for markers twice as
        large. Marker sizes stay proportional to the population: the most
        populated node has an area of ``100 * marker_scale`` points^2.
        Default: 1.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : cartopy.mpl.geoaxes.GeoAxes
        The axes with the access plot.
    """
    # copies: the caller's dicts are not modified by the pop() calls below
    plot_kwargs = dict(plot_kwargs or {})
    cbar_kwargs = dict(cbar_kwargs or {})
    fig, axes = _new_axes(axes, projection)

    gdf_enduser = network.nodes[network.nodes.ci_type == enduser]

    service = f"access_state_{ci_type}_{enduser}"

    # get access states
    if access_states is not None:
        gdf_enduser = gdf_enduser[gdf_enduser[service].isin(access_states)]

    # lowercase for safety
    states = gdf_enduser[service].str.lower()
    unknown = sorted(set(states.dropna()) - set(STATUS_MAP))
    if unknown:
        LOGGER.warning(
            "access_plot: access states %s are not in STATUS_MAP; shown in the "
            "colour of '%s'.",
            unknown,
            ACCESS_NO_BASE,
        )
    cvals = states.map(STATUS_MAP).fillna(STATUS_MAP[ACCESS_NO_BASE])
    style = dict(
        c=cvals,
        s=_marker_sizes(gdf_enduser[pop_col], 100 * marker_scale),
        transform=_data_crs(network),
        cmap=ListedColormap(["#046582", "#158204", "grey", "#CF8018", "#BB8082"]),
        vmin=-2,
        vmax=2,
    )
    style.update(plot_kwargs)
    scatter = axes.scatter(gdf_enduser.geometry.x, gdf_enduser.geometry.y, **style)
    n_ppl_access_loss = gdf_enduser.loc[
        gdf_enduser[service].isin(ACCESS_LOSS_STATES), pop_col
    ].sum()
    n_ppl_no_base_access = gdf_enduser.loc[
        gdf_enduser[service] == ACCESS_NO_BASE, pop_col
    ].sum()

    text = (
        f"{n_ppl_no_base_access:.2f} {enduser} without base access to {ci_type}"
        + f"\n{n_ppl_access_loss:.2f} {enduser} losing access to {ci_type}"
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
    cbar.set_label("Access to service")
    return fig, axes
