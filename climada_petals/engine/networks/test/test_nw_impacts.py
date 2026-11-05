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

Tests for the network impacts (nw_impacts)

Tests marked with ``_xfail(step, ...)`` document known bugs; they are fixed in
the given step of the project doc ``claude/nw-impacts-review.md`` (strict: the
step must remove the marker).
"""

from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import LineString

import climada.util.lines_polys_handler as u_lp
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.hazard import Centroids, Hazard

from climada_petals.engine.networks.nw_impacts import (
    NetworkImpactCalc,
    exposure_from_edges,
    exposure_from_nodes,
    gdf_from_network,
)
from climada_petals.engine.networks.nw_calcs import NetworkCalcs
from climada_petals.engine.networks.nw_utils import ADDED_COL


def _xfail(step, reason):
    """xfail marker for a known bug fixed in the given step of the review doc."""
    return pytest.mark.xfail(
        strict=True, reason=f"Known bug, fixed in step {step}: {reason}"
    )


# ========================================================================
# Fixtures
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


HAZ_DIAGONAL = [[10.0, 20.0, 30.0, 40.0, 50.0]]  # as simple_hazard


def _make_hazard(intensity, haz_type="TC"):
    """Hazard on the 5 centroids (i, i), i = 0..4, of the toy network.

    Parameters
    ----------
    intensity : array-like
        Shape ``(n_events, 5)``; one row per event.
    """
    intensity = np.atleast_2d(np.asarray(intensity, dtype=float))
    n_ev = intensity.shape[0]
    coords = np.arange(5, dtype=float)
    return Hazard(
        haz_type=haz_type,
        centroids=Centroids(lat=coords, lon=coords),
        event_id=np.arange(1, n_ev + 1),
        event_name=[f"ev{i}" for i in range(1, n_ev + 1)],
        date=np.arange(1, n_ev + 1),
        frequency=np.ones(n_ev) / n_ev,
        intensity=sparse.csr_matrix(intensity),
        fraction=sparse.csr_matrix(np.ones_like(intensity)),
    )


def _make_impf(fun_id, intensity=(0, 20, 40, 60), mdd=(0, 0, 0.5, 1)):
    """Impact function with id ``fun_id`` (default: same shape as
    ``road_impf``: 0 up to 20, 0.25 at 30, 0.5 at 40, 0.75 at 50)."""
    return ImpactFunc(
        haz_type="TC",
        id=fun_id,
        intensity=np.array(intensity, dtype=float),
        mdd=np.array(mdd, dtype=float),
        paa=np.ones(len(intensity)),
        intensity_unit="m/s",
        name=f"{fun_id}_TC",
    )


def _run_disrupt(network, exp_list, impfs, thresholds, hazard):
    """Build a NetworkImpactCalc and run ``disrupt_network``."""
    nic = NetworkImpactCalc(
        impf_set=ImpactFuncSet(impfs),
        impf_thresh_set=thresholds,
        haz=hazard,
        network=network,
        exp_list=exp_list,
    )
    return nic.disrupt_network()


def _ci_values(gdf, ci_type, col):
    """Values of ``col`` for the rows of ``gdf`` of type ``ci_type``."""
    return gdf.loc[gdf["ci_type"] == ci_type, col].tolist()


# ========================================================================
# Filtering network components
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
# Exposures from nodes
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


@_xfail("I2", "an unknown ci_type returns an empty exposure")
def test_exposure_from_nodes_unknown_ci_type_raises(impact_network):
    with pytest.raises(ValueError, match="(?i)no nodes"):
        exposure_from_nodes(impact_network, "school")


# ========================================================================
# Exposures from edges
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


def test_exposure_from_edges_value_from_column(impact_network):
    """With disagg_val=None, values are taken from disagg_col and divided
    over the points, so the total per edge is preserved."""
    exp = exposure_from_edges(
        impact_network,
        "road",
        res=10000,
        disagg_val=None,
        disagg_col="distance",
        disagg_met=u_lp.DisaggMethod.DIV,
    )

    total = impact_network.edges.loc[
        impact_network.edges["ci_type"] == "road", "distance"
    ].sum()
    assert exp.gdf["value"].sum() == pytest.approx(total)


def test_exposure_from_edges_has_lat_lon(impact_network):
    """set_lat_lon adds latitude and longitude columns."""
    exp = exposure_from_edges(impact_network, "road", res=100)
    assert "latitude" in exp.gdf.columns
    assert "longitude" in exp.gdf.columns


@_xfail("I2", "disagg_val=None without disagg_col raises KeyError(None)")
def test_exposure_from_edges_needs_value_source(impact_network):
    """disagg_val=None without disagg_col -> clear ValueError."""
    with pytest.raises(ValueError, match="disagg_col"):
        exposure_from_edges(impact_network, "road", res=10_000, disagg_val=None)


@_xfail("I2", "an unknown ci_type returns an empty exposure")
def test_exposure_from_edges_unknown_ci_type_raises(impact_network):
    with pytest.raises(ValueError, match="(?i)no edges"):
        exposure_from_edges(impact_network, "rail", res=10_000)


@_xfail("I2", "links added by the module (ADDED_COL) are exposed (DI-3)")
def test_exposure_from_edges_skips_added_links(impact_network):
    """Straight links created by add_physical_links / link_clusters are not
    infrastructure: they are not exposed (as in make_network_stat)."""
    edges = impact_network.edges
    added = edges.iloc[[0]].copy()
    added["from_id"] = 4
    added["to_id"] = 0
    added["geometry"] = [LineString([(4, 4), (0, 0)])]
    impact_network.edges = pd.concat(
        [edges.assign(**{ADDED_COL: False}), added.assign(**{ADDED_COL: True})],
        ignore_index=True,
    )

    exp = exposure_from_edges(impact_network, "road", res=10_000)

    assert exp.gdf.index.get_level_values(0).unique().tolist() == [0, 1, 2, 3]


# ========================================================================
# Initialisation
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
# Point impacts
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
# Impacts written to the network
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
# Network disruption
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

    # Healthcare node at (4,4) has intensity=50 ⟶ impact 0.75 > 0.3
    hc = network_disr.nodes[network_disr.nodes["ci_type"] == "healthcare"]
    assert hc["imp_dir"].iloc[0] == pytest.approx(0.75)
    assert hc["func_internal"].iloc[0] == 0  # above threshold
    assert hc["func_tot"].iloc[0] == 0
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

    # Edge 0 lies between centroids with intensity 10 and 20, where the
    # impact function is 0; all other edges reach intensities >= 30.
    assert network_disr.edges["func_internal"].tolist() == [1, 0, 0, 0]
    assert network_disr.edges["imp_dir"].iloc[0] == 0
    assert (network_disr.edges["imp_dir"].iloc[1:] > 0).all()
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


def test_disrupt_network_edge_imp_dir_is_sum_of_points(impact_network):
    """imp_dir of an edge is the sum of the impacts at its points."""
    # intensity 30 everywhere -> 0.25 at every point (value 1)
    exp = exposure_from_edges(impact_network, "road", res=10_000)
    n_pnts = exp.gdf.groupby(level=0, sort=False).size().to_numpy()

    network_disr, imp_dict = _run_disrupt(
        impact_network,
        [exp],
        [_make_impf("road")],
        {"road": 1e9},
        _make_hazard([[30.0] * 5]),
    )

    assert _ci_values(network_disr.edges, "road", "imp_dir") == pytest.approx(
        (0.25 * n_pnts).tolist()
    )
    assert imp_dict["road"].imp_mat.shape == (1, 4)


def test_disrupt_network_edge_fails_if_one_part_is_damaged(impact_network):
    """An edge fails when part of it is fully damaged (a road cut at one place
    is cut). Holds for the current rule (sum of point impacts) and for the
    per-point damage-ratio rule of DI-1; it would not hold for a mean rule."""
    # only centroid 2 (2, 2) is damaging: edges 1 and 2 touch it
    exp = exposure_from_edges(impact_network, "road", res=10_000)

    network_disr, _ = _run_disrupt(
        impact_network,
        [exp],
        [_make_impf("road")],
        {"road": 0.5},
        _make_hazard([[10.0, 10.0, 60.0, 10.0, 10.0]]),
    )

    assert _ci_values(network_disr.edges, "road", "func_internal") == [1, 0, 0, 1]


def test_disrupt_network_keeps_existing_failures(impact_network):
    """func_tot = min(func_internal, func_tot): a component that already failed
    (e.g. in the base-state cascade) stays failed when the hazard spares it."""
    impact_network.nodes.loc[
        impact_network.nodes["ci_type"] == "healthcare", "func_tot"
    ] = 0
    exp = exposure_from_nodes(impact_network, "healthcare", value=1)

    network_disr, _ = _run_disrupt(
        impact_network,
        [exp],
        [_make_impf("healthcare")],
        {"healthcare": 0.5},
        _make_hazard([[10.0] * 5]),  # impact 0 everywhere
    )

    assert _ci_values(network_disr.nodes, "healthcare", "func_internal") == [1]
    assert _ci_values(network_disr.nodes, "healthcare", "func_tot") == [0]


@_xfail("I3", "multi-event hazards fail with an unrelated pandas error")
def test_disrupt_network_multi_event_raises(impact_network):
    """A hazard with several events raises a clear error."""
    exp = exposure_from_nodes(impact_network, "healthcare", value=1)
    haz = _make_hazard([[10, 20, 30, 40, 50], [50, 40, 30, 20, 10]])

    with pytest.raises(ValueError, match="one event"):
        _run_disrupt(
            impact_network, [exp], [_make_impf("healthcare")], {"healthcare": 0.3}, haz
        )


@_xfail("I4", "impacts are written to the rows with ci_type == tag")
def test_disrupt_network_custom_tag(impact_network):
    """The tag selects the impact function and threshold; impacts are written
    to the nodes the exposure was built from."""
    exp = exposure_from_nodes(impact_network, "healthcare", value=1, tag="hospital")

    network_disr, imp_dict = _run_disrupt(
        impact_network,
        [exp],
        [_make_impf("hospital")],
        {"hospital": 0.3},
        _make_hazard(HAZ_DIAGONAL),
    )

    assert _ci_values(network_disr.nodes, "healthcare", "func_internal") == [0]
    assert "hospital" in imp_dict


@_xfail("I4", "edges whose ci_type is not in LINE_EXPOSURES are handled as nodes")
def test_disrupt_network_edges_of_any_ci_type(impact_network):
    """Line exposures are recognised from the exposure itself, not from the
    hard-coded LINE_EXPOSURES list."""
    impact_network.edges["ci_type"] = "pipeline"
    exp = exposure_from_edges(impact_network, "pipeline", res=10_000)

    network_disr, _ = _run_disrupt(
        impact_network,
        [exp],
        [_make_impf("pipeline")],
        {"pipeline": 0.2},
        _make_hazard(HAZ_DIAGONAL),
    )

    assert _ci_values(network_disr.edges, "pipeline", "func_internal") == [1, 0, 0, 0]
    assert (network_disr.nodes["func_internal"] == 1).all()


@_xfail("I3", "without func_internal, func_tot of all other rows becomes NaN")
def test_disrupt_network_requires_funcstates(network_with_ci_types):
    """Without initialised functional states, disrupt_network raises instead
    of writing NaN to func_tot of every row it does not touch."""
    exp = exposure_from_nodes(network_with_ci_types, "healthcare", value=1)

    with pytest.raises(ValueError, match="initialize_funcstates"):
        _run_disrupt(
            network_with_ci_types,
            [exp],
            [_make_impf("healthcare")],
            {"healthcare": 0.3},
            _make_hazard(HAZ_DIAGONAL),
        )


@_xfail("I3", "AttributeError on the empty list returned by get_func")
def test_disrupt_network_missing_impf_raises(impact_network):
    """No impact function for the exposure tag -> clear ValueError."""
    exp = exposure_from_nodes(impact_network, "healthcare", value=1)

    with pytest.raises(ValueError, match="impact function"):
        _run_disrupt(
            impact_network,
            [exp],
            [_make_impf("road")],
            {"healthcare": 0.3},
            _make_hazard(HAZ_DIAGONAL),
        )


@_xfail("I3", "TypeError when comparing with None")
def test_disrupt_network_missing_threshold_raises(impact_network):
    """No threshold for the exposure tag -> clear ValueError."""
    exp = exposure_from_nodes(impact_network, "healthcare", value=1)

    with pytest.raises(ValueError, match="threshold"):
        _run_disrupt(
            impact_network,
            [exp],
            [_make_impf("healthcare")],
            {"road": 0.3},
            _make_hazard(HAZ_DIAGONAL),
        )


@_xfail("I4", "impacts are matched to rows by position, not by label")
def test_disrupt_network_exposure_not_from_network_raises(
    impact_network, network_with_remote_node_missing_edge
):
    """An exposure with points that are not rows of the network raises."""
    other = network_with_remote_node_missing_edge  # healthcare nodes 4 and 5
    other.initialize_funcstates()
    exp = exposure_from_nodes(other, "healthcare", value=1)

    with pytest.raises(ValueError, match="not in the network"):
        _run_disrupt(
            impact_network,
            [exp],
            [_make_impf("healthcare")],
            {"healthcare": 0.3},
            _make_hazard(HAZ_DIAGONAL),
        )


@_xfail("I5", "threshold compared with the absolute impact (DI-1)")
def test_disrupt_network_threshold_on_damage_ratio_nodes(impact_network):
    """A fully flooded people cell fails whatever its population count
    (resampled rasters give fractional counts, e.g. 0.3)."""
    impact_network.nodes["counts"] = [0.3, np.nan, np.nan, np.nan, np.nan]
    exp = exposure_from_nodes(impact_network, "people", value_col="counts")
    people_impf = _make_impf("people", intensity=(0, 5, 6, 100), mdd=(0, 0, 1, 1))

    network_disr, _ = _run_disrupt(
        impact_network, [exp], [people_impf], {"people": 0.5}, _make_hazard(HAZ_DIAGONAL)
    )

    assert _ci_values(network_disr.nodes, "people", "imp_dir") == pytest.approx([0.3])
    assert _ci_values(network_disr.nodes, "people", "func_internal") == [0]


@_xfail("I5", "edge result depends on res: sum of point impacts (DI-1)")
@pytest.mark.parametrize("res", [10_000, 50_000])
def test_disrupt_network_threshold_on_damage_ratio_edges(impact_network, res):
    """A damage ratio of 0.25 at every point, below the threshold 0.3, leaves
    every edge functional at any disaggregation resolution."""
    exp = exposure_from_edges(impact_network, "road", res=res)

    network_disr, _ = _run_disrupt(
        impact_network,
        [exp],
        [_make_impf("road")],
        {"road": 0.3},
        _make_hazard([[30.0] * 5]),
    )

    assert _ci_values(network_disr.edges, "road", "func_internal") == [1, 1, 1, 1]


# ========================================================================
# Disruption followed by the cascade
# ========================================================================


def test_disrupt_then_cascade_cuts_healthcare_access(network_calcs, dependency_table):
    """End to end, as in the tutorial: base-state cascade -> exposures from the
    base network -> disrupt_network -> cascade on the disrupted network.
    The hospital (intensity 50, impact 0.75 > 0.3) fails, people lose
    healthcare access but keep road access."""
    network_calcs.initialize_base_state()
    network_calcs.setup_dependencies()
    network_calcs.cascade(friction_surf=None, rerouting=False)
    base = network_calcs.network

    exp = exposure_from_nodes(base, "healthcare", value=1)
    network_disr, _ = _run_disrupt(
        base,
        [exp],
        [_make_impf("healthcare")],
        {"healthcare": 0.3},
        _make_hazard(HAZ_DIAGONAL),
    )

    nc_disr = NetworkCalcs(network=network_disr, dep_table=dependency_table)
    nc_disr.cascade(friction_surf=None, rerouting=False)
    nodes = nc_disr.network.nodes

    assert _ci_values(nodes, "healthcare", "func_tot") == [0]
    assert _ci_values(nodes, "people", "actual_supply_healthcare_people") == [0]
    assert _ci_values(nodes, "people", "access_state_healthcare_people") == [
        "access disrupted source"
    ]
    assert _ci_values(nodes, "people", "actual_supply_road_people") == [1]
    # the base network is not modified
    assert _ci_values(base.nodes, "healthcare", "func_tot") == [1]
