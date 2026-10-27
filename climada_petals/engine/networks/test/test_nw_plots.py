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

Tests for the network plots (nw_plots)

Most tests are smoke tests: the plot runs without error and returns the
figure and axes. ``impact_network`` comes from ``conftest.py``.
"""

import cartopy.crs as ccrs
import geopandas as gpd
import pytest
from matplotlib import pyplot as plt
from shapely.geometry import LineString, Point

from climada_petals.engine.networks import nw_plots as nwp
from climada_petals.engine.networks import nw_utils as nwu
from climada_petals.engine.networks.nw_base import Network


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ========================================================================
# Helpers
# ========================================================================


def test_status_map_knows_all_cascade_states():
    states_written_by_cascade = {
        nwu.ACCESS_UNDISRUPTED,
        nwu.ACCESS_NEW_SOURCE,
        nwu.ACCESS_NO_BASE,
        nwu.ACCESS_DISRUPTED_VIA,
        nwu.ACCESS_DISRUPTED_SOURCE,
        nwu.ACCESS_DISRUPTED,  # GraphCalcs._propagate_check_fail
    }
    assert states_written_by_cascade <= set(nwp.STATUS_MAP)


def test_data_crs(impact_network):
    """Geographic data: PlateCarree; projected data: the projection of its EPSG code."""
    assert isinstance(nwp._data_crs(impact_network), ccrs.PlateCarree)

    projected = Network.reproject(impact_network, "EPSG:3857")
    crs = nwp._data_crs(projected)

    assert isinstance(crs, ccrs.Projection)
    # cartopy CRSs are pyproj CRSs
    assert crs.to_epsg() == 3857


def _dependency_network():
    nodes = gpd.GeoDataFrame(
        {
            "ci_type": ["power", "power_line", "people"],
            "geometry": [Point(0, 0), Point(1, 0), Point(2, 0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    edges = gpd.GeoDataFrame(
        {
            "from_id": [1, 0],
            "to_id": [2, 2],
            "ci_type": ["dependency_power_line_people", "dependency_power_people"],
            "geometry": [LineString([(1, 0), (2, 0)]), LineString([(0, 0), (2, 0)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    return Network(edges=edges, nodes=nodes)


def test_get_dependencies_exact_names():
    """'power' does not select the dependencies of 'power_line'."""
    network = _dependency_network()

    assert list(nwp._get_dependencies(network, ["power"])) == [
        "dependency_power_people"
    ]
    assert list(nwp._get_dependencies(network, ["power_line"])) == [
        "dependency_power_line_people"
    ]
    assert list(nwp._get_dependencies(network, ["people"])) == [
        "dependency_power_line_people",
        "dependency_power_people",
    ]


# ========================================================================
# Plots
# ========================================================================


def test_infra_plot(impact_network):
    fig, axes = impact_network.plot_infra(mask_func_ci=False)

    assert fig is axes.get_figure()


def test_infra_plot_functional_states(impact_network):
    fig, axes = impact_network.plot_infra(plot_col="func_tot", mask_func_ci=False)

    assert fig is axes.get_figure()


def test_infra_plot_without_func_tot(impact_network):
    """mask_func_ci (default) is ignored before the functional states exist."""
    impact_network.nodes = impact_network.nodes.drop(columns="func_tot")
    impact_network.edges = impact_network.edges.drop(columns="func_tot")

    fig, axes = impact_network.plot_infra()

    assert fig is axes.get_figure()


def test_infra_plot_does_not_modify_kwargs(impact_network):
    """The dicts passed as arguments are not changed (they used to be popped)."""
    ci_kwargs = {"colors": ["red", "blue", "green"]}
    cbar_kwargs = {"shrink": 0.3, "orientation": "vertical"}

    impact_network.plot_infra(
        plot_col="func_tot",
        mask_func_ci=False,
        ci_kwargs=ci_kwargs,
        cbar_kwargs=cbar_kwargs,
    )

    assert ci_kwargs == {"colors": ["red", "blue", "green"]}
    assert cbar_kwargs == {"shrink": 0.3, "orientation": "vertical"}


def test_enduser_plot_style_override(impact_network):
    """Keyword arguments override the default style (no duplicate keyword)."""
    fig, axes = impact_network.plot_enduser(color="red", alpha=1)

    assert fig is axes.get_figure()


def test_access_plot(impact_network):
    fig, axes = impact_network.plot_access("healthcare")

    assert fig is axes.get_figure()


def test_access_plot_projected_network(impact_network):
    """Data in EPSG:3857 on a Mercator map."""
    projected = Network.reproject(impact_network, "EPSG:3857")

    fig, axes = projected.plot_access("healthcare", projection=ccrs.Mercator())

    assert fig is axes.get_figure()


def test_access_plot_counts_all_loss_states(impact_network):
    """'access disrupted' counts as a loss of access, as in make_network_stat."""
    nodes = impact_network.nodes
    nodes.loc[1, "access_state_healthcare_people"] = "access disrupted"

    _, axes = impact_network.plot_access("healthcare")

    no_base_line, loss_line = axes.texts[0].get_text().splitlines()
    assert no_base_line.startswith("20.00 ")
    assert loss_line.startswith("50.00 ")
    stats = nwu.make_network_stat(impact_network)
    assert stats["healthcare_access_loss"] == 50


def test_access_plot_unknown_state_warns(impact_network, nw_caplog):
    impact_network.nodes.loc[0, "access_state_healthcare_people"] = "something else"

    impact_network.plot_access("healthcare")

    assert "not in STATUS_MAP" in nw_caplog.text


def test_dep_plot(impact_network):
    fig, axes = impact_network.plot_dep(ci_types=["healthcare"])

    assert fig is axes.get_figure()
