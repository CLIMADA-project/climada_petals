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

Test nw_impacts module
"""

import pytest
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy import sparse
from shapely.geometry import Point, LineString
from unittest.mock import MagicMock

from climada.hazard import Hazard, Centroids
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet

from climada_petals.engine.networks.nw_impacts import (
    gdf_from_network,
    exposure_from_nodes,
    exposure_from_edges,
    NetworkImpactCalc,
)
from climada_petals.engine.networks.test.fixtures_test_networks import *


# ========================================================================
# Fixtures for nw_impacts tests
# ========================================================================


@pytest.fixture
def impact_network(network_with_ci_types):
    """Network with CI types and initialised functional states."""
    network_with_ci_types.initialize_funcstates()
    return network_with_ci_types


@pytest.fixture
def simple_hazard():
    """Hazard with TC type, centroids at test network node locations."""
    lats = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    lons = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    return Hazard(
        haz_type="TC",
        centroids=Centroids(lat=lats, lon=lons),
        event_id=np.array([1]),
        event_name=["ev1"],
        date=np.array([1]),
        frequency=np.array([1.0]),
        intensity=sparse.csr_matrix(np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])),
        fraction=sparse.csr_matrix(np.ones((1, 5))),
    )


@pytest.fixture
def healthcare_impf():
    """Step impact function for healthcare CI (string id matching ci_type)."""
    return ImpactFunc(
        haz_type="TC",
        id="healthcare",
        intensity=np.array([0, 20, 40, 60]),
        mdd=np.array([0, 0.0, 0.5, 1.0]),
        paa=np.array([1, 1, 1, 1]),
        intensity_unit="m/s",
        name="healthcare_TC",
    )


@pytest.fixture
def road_impf():
    """Step impact function for road CI (string id matching ci_type)."""
    return ImpactFunc(
        haz_type="TC",
        id="road",
        intensity=np.array([0, 20, 40, 60]),
        mdd=np.array([0, 0.0, 0.5, 1.0]),
        paa=np.array([1, 1, 1, 1]),
        intensity_unit="m/s",
        name="road_TC",
    )


@pytest.fixture
def mock_impact_mat():
    """Create a mock Impact object with a given imp_mat.

    Returns a factory function accepting a 2-D array.
    """

    def _factory(arr):
        mock = MagicMock()
        mock.imp_mat = sparse.csr_matrix(np.atleast_2d(arr))
        return mock

    return _factory


# ========================================================================
# Tests for gdf_from_network
# ========================================================================


def test_gdf_from_network_nodes(network_with_ci_types):
    """Filter nodes by ci_type returns only matching rows."""
    result = gdf_from_network(network_with_ci_types.nodes, "road")
    assert len(result) == 3
    assert all(result["ci_type"] == "road")


def test_gdf_from_network_edges(network_with_ci_types):
    """Filter edges by ci_type returns only matching rows."""
    result = gdf_from_network(network_with_ci_types.edges, "road")
    assert len(result) == 4
    assert all(result["ci_type"] == "road")


def test_gdf_from_network_no_match(network_with_ci_types):
    """Non-existent ci_type returns empty dataframe."""
    result = gdf_from_network(network_with_ci_types.nodes, "nonexistent")
    assert len(result) == 0
    assert isinstance(result, gpd.GeoDataFrame)


def test_gdf_from_network_returns_copy(network_with_ci_types):
    """Modifying the result must not mutate the original network."""
    result = gdf_from_network(network_with_ci_types.nodes, "road")
    result["new_col"] = 99
    assert "new_col" not in network_with_ci_types.nodes.columns


def test_gdf_from_network_healthcare(network_with_ci_types):
    """Filter for healthcare returns single node."""
    result = gdf_from_network(network_with_ci_types.nodes, "healthcare")
    assert len(result) == 1
    assert result.iloc[0]["ci_type"] == "healthcare"


# ========================================================================
# Tests for exposure_from_nodes
# ========================================================================


def test_exposure_from_nodes_default_value(impact_network):
    """No value/value_col ⟶ default value=1, tag=ci_type."""
    exp = exposure_from_nodes(impact_network, "healthcare")
    assert isinstance(exp, Exposures)
    assert len(exp.gdf) == 1
    assert exp.gdf["value"].iloc[0] == 1
    assert exp.description == "healthcare"


def test_exposure_from_nodes_with_value(impact_network):
    """Explicit value is assigned to all exposure points."""
    exp = exposure_from_nodes(impact_network, "healthcare", value=100)
    assert exp.gdf["value"].iloc[0] == 100


def test_exposure_from_nodes_value_zero(impact_network):
    """value=0 must be respected, not treated as falsy."""
    exp = exposure_from_nodes(impact_network, "healthcare", value=0)
    assert exp.gdf["value"].iloc[0] == 0


def test_exposure_from_nodes_with_value_col(impact_network):
    """value_col reads values from an existing column."""
    impact_network.nodes["importance"] = [10, 20, 30, 40, 50]
    exp = exposure_from_nodes(impact_network, "healthcare", value_col="importance")
    assert exp.gdf["value"].iloc[0] == 50  # healthcare is node 4


def test_exposure_from_nodes_value_overrides_value_col(impact_network):
    """When both value and value_col are provided, value wins."""
    impact_network.nodes["importance"] = [10, 20, 30, 40, 50]
    exp = exposure_from_nodes(
        impact_network, "healthcare", value=999, value_col="importance"
    )
    assert exp.gdf["value"].iloc[0] == 999


def test_exposure_from_nodes_custom_tag(impact_network):
    """Custom tag overrides default ci_type tag."""
    exp = exposure_from_nodes(impact_network, "healthcare", tag="hospital")
    assert exp.description == "hospital"


def test_exposure_from_nodes_multiple(impact_network):
    """Multiple matching nodes are included."""
    exp = exposure_from_nodes(impact_network, "road")
    assert len(exp.gdf) == 3
    assert exp.description == "road"


def test_exposure_from_nodes_has_lat_lon(impact_network):
    """set_lat_lon adds latitude and longitude columns."""
    exp = exposure_from_nodes(impact_network, "healthcare")
    assert "latitude" in exp.gdf.columns
    assert "longitude" in exp.gdf.columns


def test_exposure_from_nodes_geometry_preserved(impact_network):
    """Resulting exposure has Point geometries matching the network nodes."""
    exp = exposure_from_nodes(impact_network, "healthcare")
    assert exp.gdf.geometry.iloc[0].geom_type == "Point"


# ========================================================================
# Tests for exposure_from_edges
# ========================================================================


def test_exposure_from_edges_basic(impact_network):
    """Edge exposure with default disaggregation produces point geometries."""
    exp = exposure_from_edges(impact_network, "road", res=100)
    assert isinstance(exp, Exposures)
    assert len(exp.gdf) > 0
    assert all(exp.gdf.geometry.geom_type == "Point")
    assert exp.description == "road"


def test_exposure_from_edges_custom_tag(impact_network):
    """Custom tag overrides default ci_type tag for edge exposures."""
    exp = exposure_from_edges(impact_network, "road", res=100, tag="highway")
    assert exp.description == "highway"


def test_exposure_from_edges_disagg_produces_more_points(impact_network):
    """Disaggregation at fine resolution produces more points than edges."""
    n_edges = len(impact_network.edges[impact_network.edges["ci_type"] == "road"])
    exp = exposure_from_edges(impact_network, "road", res=100)
    assert len(exp.gdf) >= n_edges


def test_exposure_from_edges_has_value(impact_network):
    """Each disaggregated point has a value column."""
    exp = exposure_from_edges(impact_network, "road", res=100)
    assert "value" in exp.gdf.columns
    assert all(exp.gdf["value"] > 0)


def test_exposure_from_edges_has_lat_lon(impact_network):
    """set_lat_lon adds latitude and longitude columns."""
    exp = exposure_from_edges(impact_network, "road", res=100)
    assert "latitude" in exp.gdf.columns
    assert "longitude" in exp.gdf.columns


# ========================================================================
# Tests for NetworkImpactCalc.__init__
# ========================================================================


def test_network_impact_calc_init(impact_network):
    """Constructor stores all inputs."""
    impf_set = ImpactFuncSet()
    haz = MagicMock()
    nic = NetworkImpactCalc(
        impf_set=impf_set,
        impf_thresh_set={"healthcare": 0.5},
        haz=haz,
        network=impact_network,
        exp_list=[],
    )
    assert nic.impf_set is impf_set
    assert nic.network is impact_network
    assert nic.haz is haz
    assert nic.exp_list == []
    assert nic.impf_thresh_set == {"healthcare": 0.5}


# ========================================================================
# Tests for NetworkImpactCalc.calc_point_impacts
# ========================================================================


def test_calc_point_impacts_returns_impact(simple_hazard, healthcare_impf):
    """calc_point_impacts returns an Impact with imp_mat."""
    impf_set = ImpactFuncSet([healthcare_impf])
    exp = Exposures(
        lat=np.array([4.0]),
        lon=np.array([4.0]),
        value=np.array([1.0]),
    )
    exp.gdf["impf_TC"] = "healthcare"

    imp = NetworkImpactCalc.calc_point_impacts(simple_hazard, exp, impf_set)

    assert imp.imp_mat is not None
    assert imp.imp_mat.shape == (1, 1)


def test_calc_point_impacts_intensity_matters(simple_hazard, healthcare_impf):
    """Higher hazard intensity produces larger impacts."""
    impf_set = ImpactFuncSet([healthcare_impf])
    # Two points: low-intensity centroid (0,0) and high-intensity centroid (4,4)
    exp = Exposures(
        lat=np.array([0.0, 4.0]),
        lon=np.array([0.0, 4.0]),
        value=np.array([1.0, 1.0]),
    )
    exp.gdf["impf_TC"] = "healthcare"

    imp = NetworkImpactCalc.calc_point_impacts(simple_hazard, exp, impf_set)

    impacts = imp.imp_mat.toarray().flatten()
    assert impacts[0] < impacts[1]


def test_calc_point_impacts_is_static():
    """calc_point_impacts is callable on the class without an instance."""
    assert isinstance(NetworkImpactCalc.__dict__["calc_point_impacts"], staticmethod)


# ========================================================================
# Tests for NetworkImpactCalc.impacts_to_network
# ========================================================================


def test_impacts_to_network_is_static():
    """impacts_to_network is callable on the class without an instance."""
    assert isinstance(NetworkImpactCalc.__dict__["impacts_to_network"], staticmethod)


def test_impacts_to_network_below_thresh(impact_network, mock_impact_mat):
    """Impact below threshold ⟶ func_internal=1, func_tot unchanged."""
    nodes = impact_network.nodes.copy()
    imp = mock_impact_mat([[0.1]])

    result = NetworkImpactCalc.impacts_to_network(nodes, imp, "healthcare", 0.5)

    hc = result[result["ci_type"] == "healthcare"]
    assert hc["func_internal"].iloc[0] == 1
    assert hc["imp_dir"].iloc[0] == pytest.approx(0.1)
    assert hc["func_tot"].iloc[0] == 1


def test_impacts_to_network_above_thresh(impact_network, mock_impact_mat):
    """Impact above threshold ⟶ func_internal=0, func_tot=0."""
    nodes = impact_network.nodes.copy()
    imp = mock_impact_mat([[0.8]])

    result = NetworkImpactCalc.impacts_to_network(nodes, imp, "healthcare", 0.5)

    hc = result[result["ci_type"] == "healthcare"]
    assert hc["func_internal"].iloc[0] == 0
    assert hc["func_tot"].iloc[0] == 0


def test_impacts_to_network_equal_thresh(impact_network, mock_impact_mat):
    """Impact exactly at threshold ⟶ func_internal=1 (<= comparison)."""
    nodes = impact_network.nodes.copy()
    imp = mock_impact_mat([[0.5]])

    result = NetworkImpactCalc.impacts_to_network(nodes, imp, "healthcare", 0.5)

    hc = result[result["ci_type"] == "healthcare"]
    assert hc["func_internal"].iloc[0] == 1


def test_impacts_to_network_func_tot_is_minimum(impact_network, mock_impact_mat):
    """func_tot = min(func_internal, existing func_tot)."""
    nodes = impact_network.nodes.copy()
    # Pre-set road func_tot to 0
    nodes.loc[nodes["ci_type"] == "road", "func_tot"] = 0

    # Road impacts all below threshold ⟶ func_internal=1
    imp = mock_impact_mat([[0.1, 0.2, 0.3]])

    result = NetworkImpactCalc.impacts_to_network(nodes, imp, "road", 0.5)

    road = result[result["ci_type"] == "road"]
    assert all(road["func_internal"] == 1)
    # func_tot = min(1, 0) = 0 because pre-existing func_tot was 0
    assert all(road["func_tot"] == 0)


def test_impacts_to_network_multiple_ci(impact_network, mock_impact_mat):
    """Only rows matching exp_tag are updated; others remain unchanged."""
    nodes = impact_network.nodes.copy()
    imp = mock_impact_mat([[0.8]])

    result = NetworkImpactCalc.impacts_to_network(nodes, imp, "healthcare", 0.5)

    # Healthcare is disrupted
    assert result[result["ci_type"] == "healthcare"]["func_internal"].iloc[0] == 0
    # Road nodes are unaffected
    assert all(result[result["ci_type"] == "road"]["func_internal"] == 1)
    assert all(result[result["ci_type"] == "road"]["func_tot"] == 1)


def test_impacts_to_network_records_imp_dir(impact_network, mock_impact_mat):
    """imp_dir column stores the raw impact value."""
    nodes = impact_network.nodes.copy()
    imp = mock_impact_mat([[0.42]])

    result = NetworkImpactCalc.impacts_to_network(nodes, imp, "healthcare", 0.5)

    assert result[result["ci_type"] == "healthcare"]["imp_dir"].iloc[
        0
    ] == pytest.approx(0.42)


def test_impacts_to_network_edges(impact_network, mock_impact_mat):
    """impacts_to_network works on edges too."""
    edges = impact_network.edges.copy()
    imp = mock_impact_mat([[0.1, 0.2, 0.8, 0.9]])

    result = NetworkImpactCalc.impacts_to_network(edges, imp, "road", 0.5)

    assert result["func_internal"].iloc[0] == 1  # 0.1 <= 0.5
    assert result["func_internal"].iloc[1] == 1  # 0.2 <= 0.5
    assert result["func_internal"].iloc[2] == 0  # 0.8 > 0.5
    assert result["func_internal"].iloc[3] == 0  # 0.9 > 0.5


# ========================================================================
# Tests for NetworkImpactCalc.disrupt_network
# ========================================================================


def test_disrupt_network_node_exposure(impact_network, simple_hazard, healthcare_impf):
    """Disruption with node exposure updates node func states."""
    impf_set = ImpactFuncSet([healthcare_impf])
    exp = exposure_from_nodes(impact_network, "healthcare", value=1)

    nic = NetworkImpactCalc(
        impf_set=impf_set,
        impf_thresh_set={"healthcare": 0.3},
        haz=simple_hazard,
        network=impact_network,
        exp_list=[exp],
    )

    network_disr, imp_dict = nic.disrupt_network()

    # Healthcare node at (4,4) has intensity=50 ⟶ high impact
    hc = network_disr.nodes[network_disr.nodes["ci_type"] == "healthcare"]
    assert hc["func_internal"].iloc[0] == 0  # above threshold
    assert "healthcare" in imp_dict
    # Original network is not mutated
    assert all(impact_network.nodes["func_tot"] == 1)


def test_disrupt_network_edge_exposure(impact_network, simple_hazard, road_impf):
    """Disruption with edge exposure updates edge func states."""
    impf_set = ImpactFuncSet([road_impf])
    exp = exposure_from_edges(impact_network, "road", res=100)

    nic = NetworkImpactCalc(
        impf_set=impf_set,
        impf_thresh_set={"road": 0.3},
        haz=simple_hazard,
        network=impact_network,
        exp_list=[exp],
    )

    network_disr, imp_dict = nic.disrupt_network()

    assert "imp_dir" in network_disr.edges.columns
    assert "func_internal" in network_disr.edges.columns
    assert "road" in imp_dict
    # Original network is not mutated
    assert all(impact_network.edges["func_tot"] == 1)


def test_disrupt_network_returns_deepcopy(
    impact_network, simple_hazard, healthcare_impf
):
    """disrupt_network returns a deep copy; original is preserved."""
    impf_set = ImpactFuncSet([healthcare_impf])
    exp = exposure_from_nodes(impact_network, "healthcare", value=1)

    nic = NetworkImpactCalc(
        impf_set=impf_set,
        impf_thresh_set={"healthcare": 0.3},
        haz=simple_hazard,
        network=impact_network,
        exp_list=[exp],
    )

    network_disr, _ = nic.disrupt_network()

    # Disrupted network is a different object
    assert network_disr is not impact_network
    assert network_disr.nodes is not impact_network.nodes


def test_disrupt_network_multiple_exposures(
    impact_network, simple_hazard, healthcare_impf, road_impf
):
    """Disruption with multiple exposures updates both nodes and edges."""
    impf_set = ImpactFuncSet([healthcare_impf, road_impf])
    exp_nodes = exposure_from_nodes(impact_network, "healthcare", value=1)
    exp_edges = exposure_from_edges(impact_network, "road", res=100)

    nic = NetworkImpactCalc(
        impf_set=impf_set,
        impf_thresh_set={"healthcare": 0.3, "road": 0.3},
        haz=simple_hazard,
        network=impact_network,
        exp_list=[exp_nodes, exp_edges],
    )

    network_disr, imp_dict = nic.disrupt_network()

    assert "healthcare" in imp_dict
    assert "road" in imp_dict
    assert "func_internal" in network_disr.nodes.columns
    assert "func_internal" in network_disr.edges.columns


def test_disrupt_network_empty_exp_list(impact_network, simple_hazard):
    """Empty exposure list returns unchanged network copy."""
    impf_set = ImpactFuncSet()
    nic = NetworkImpactCalc(
        impf_set=impf_set,
        impf_thresh_set={},
        haz=simple_hazard,
        network=impact_network,
        exp_list=[],
    )

    network_disr, imp_dict = nic.disrupt_network()

    assert len(imp_dict) == 0
    assert all(network_disr.nodes["func_tot"] == 1)
    assert all(network_disr.edges["func_tot"] == 1)
