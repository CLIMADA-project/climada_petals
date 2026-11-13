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
    _impacts_to_network,
    exposure_from_edges,
    exposure_from_nodes,
    gdf_from_network,
)
from climada_petals.engine.networks.nw_calcs import NetworkCalcs
from climada_petals.engine.networks.nw_utils import ADDED_COL



# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def funcstate_network(network_with_ci_types):
    """Network with CI types and initialised functional states."""
    network_with_ci_types.initialize_funcstates()
    return network_with_ci_types


# intensity at the 5 nodes (i, i): 10 at node 0 ... 50 at the hospital (node 4)
HAZ_DIAGONAL = [[10.0, 20.0, 30.0, 40.0, 50.0]]


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
    """Impact function with id ``fun_id`` (default MDD: 0 up to intensity
    20, then linear: 0.25 at 30, 0.5 at 40, 0.75 at 50)."""
    return ImpactFunc(
        haz_type="TC",
        id=fun_id,
        intensity=np.array(intensity, dtype=float),
        mdd=np.array(mdd, dtype=float),
        paa=np.ones(len(intensity)),
        intensity_unit="m/s",
        name=f"{fun_id}_TC",
    )


def _run_disrupt(
    network, exp_list, impfs, thresholds, hazard, edge_rule="max_ratio"
):
    """Build a NetworkImpactCalc and run ``disrupt_network``."""
    nic = NetworkImpactCalc(
        impf_set=ImpactFuncSet(impfs),
        impf_thresh_set=thresholds,
        haz=hazard,
        network=network,
        exp_list=exp_list,
        edge_rule=edge_rule,
    )
    return nic.disrupt_network()


def _ci_values(gdf, ci_type, col):
    """Values of ``col`` for the rows of ``gdf`` of type ``ci_type``."""
    return gdf.loc[gdf["ci_type"] == ci_type, col].tolist()


def _with_added_link(network):
    """Append a road edge 4 -> 0 marked as added by the module (``ADDED_COL``,
    missing on the input edges, as after add_physical_links); return its label."""
    edges = network.edges
    added = edges.iloc[[0]].copy()
    added["id"] = len(edges)
    added["from_id"] = 4
    added["to_id"] = 0
    added["geometry"] = [LineString([(4, 4), (0, 0)])]
    network.edges = pd.concat(
        [edges, added.assign(**{ADDED_COL: True})], ignore_index=True
    )
    return len(network.edges) - 1


# ========================================================================
# Filtering network components
# ========================================================================


def test_gdf_from_network_nodes(network_with_ci_types):
    """Filter nodes by ci_type returns only matching rows, with their labels."""
    result = gdf_from_network(network_with_ci_types.nodes, "road")
    assert result.index.tolist() == [1, 2, 3]
    assert all(result["ci_type"] == "road")


def test_gdf_from_network_edges(network_with_ci_types):
    """Filter edges by ci_type returns only matching rows, with their labels."""
    edges = network_with_ci_types.edges
    edges.loc[1, "ci_type"] = "rail"  # all edges of the toy network are roads
    result = gdf_from_network(edges, "road")
    assert result.index.tolist() == [0, 2, 3]
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


# ========================================================================
# Exposures from nodes
# ========================================================================


def test_exposure_from_nodes_default_value(funcstate_network):
    """No value/value_col ⟶ default value=1, tag=ci_type."""
    exp = exposure_from_nodes(funcstate_network, "healthcare")
    assert isinstance(exp, Exposures)
    assert len(exp.gdf) == 1
    assert exp.gdf["value"].iloc[0] == 1
    assert exp.description == "healthcare"


def test_exposure_from_nodes_with_value(funcstate_network):
    """Explicit value is assigned to all exposure points."""
    exp = exposure_from_nodes(funcstate_network, "healthcare", value=100)
    assert exp.gdf["value"].iloc[0] == 100


def test_exposure_from_nodes_value_zero(funcstate_network):
    """value=0 must be respected, not treated as falsy."""
    exp = exposure_from_nodes(funcstate_network, "healthcare", value=0)
    assert exp.gdf["value"].iloc[0] == 0


def test_exposure_from_nodes_with_value_col(funcstate_network):
    """value_col reads values from an existing column."""
    funcstate_network.nodes["importance"] = [10, 20, 30, 40, 50]
    exp = exposure_from_nodes(funcstate_network, "healthcare", value_col="importance")
    assert exp.gdf["value"].iloc[0] == 50  # healthcare is node 4


def test_exposure_from_nodes_value_overrides_value_col(funcstate_network):
    """When both value and value_col are provided, value wins."""
    funcstate_network.nodes["importance"] = [10, 20, 30, 40, 50]
    exp = exposure_from_nodes(
        funcstate_network, "healthcare", value=999, value_col="importance"
    )
    assert exp.gdf["value"].iloc[0] == 999


def test_exposure_from_nodes_custom_tag(funcstate_network):
    """Custom tag overrides default ci_type tag."""
    exp = exposure_from_nodes(funcstate_network, "healthcare", tag="hospital")
    assert exp.description == "hospital"


def test_exposure_from_nodes_multiple(funcstate_network):
    """Multiple matching nodes are included."""
    exp = exposure_from_nodes(funcstate_network, "road")
    assert len(exp.gdf) == 3
    assert exp.description == "road"


def test_exposure_from_nodes_lat_lon(funcstate_network):
    """Latitude and longitude of the points are those of the nodes."""
    exp = exposure_from_nodes(funcstate_network, "healthcare")
    assert exp.latitude.tolist() == [4.0]
    assert exp.longitude.tolist() == [4.0]


def test_exposure_from_nodes_geometry_preserved(funcstate_network):
    """Resulting exposure has Point geometries matching the network nodes."""
    exp = exposure_from_nodes(funcstate_network, "healthcare")
    assert exp.gdf.geometry.iloc[0].geom_type == "Point"


def test_exposure_from_nodes_unknown_ci_type_raises(funcstate_network):
    with pytest.raises(ValueError, match="(?i)no nodes"):
        exposure_from_nodes(funcstate_network, "school")


def test_exposure_from_nodes_missing_value_col_raises(funcstate_network):
    with pytest.raises(ValueError, match="value_col='counts'"):
        exposure_from_nodes(funcstate_network, "people", value_col="counts")


@pytest.mark.parametrize("as_array", [False, True])
def test_exposure_from_nodes_mask(funcstate_network, as_array):
    """mask selects a subset of the nodes of the ci_type (Series or array)."""
    mask = funcstate_network.nodes["id"].isin([1, 3])
    if as_array:
        mask = mask.to_numpy()
    exp = exposure_from_nodes(funcstate_network, "road", mask=mask, tag="road_13")
    assert exp.gdf.index.tolist() == [1, 3]
    assert exp.description == "road_13"


@pytest.mark.parametrize(
    "mask, match",
    [
        (np.array([True, False]), "one value per row"),
        (np.array([0, 1, 1, 0, 0]), "boolean"),
        (pd.Series([True] * 5, index=[10, 11, 12, 13, 14]), "same index"),
    ],
)
def test_exposure_from_nodes_invalid_mask_raises(funcstate_network, mask, match):
    with pytest.raises(ValueError, match=match):
        exposure_from_nodes(funcstate_network, "road", mask=mask)


def test_exposure_from_nodes_mask_selects_nothing_raises(funcstate_network):
    mask = funcstate_network.nodes["ci_type"] == "people"  # no road node
    with pytest.raises(ValueError, match="mask selects no nodes"):
        exposure_from_nodes(funcstate_network, "road", mask=mask)


# ========================================================================
# Exposures from edges
# ========================================================================


def test_exposure_from_edges_basic(funcstate_network):
    """Edge exposure with default disaggregation produces point geometries."""
    exp = exposure_from_edges(funcstate_network, "road", res=10_000)
    assert isinstance(exp, Exposures)
    assert len(exp.gdf) > 0
    assert all(exp.gdf.geometry.geom_type == "Point")
    assert exp.description == "road"


def test_exposure_from_edges_custom_tag(funcstate_network):
    """Custom tag overrides default ci_type tag for edge exposures."""
    exp = exposure_from_edges(funcstate_network, "road", res=10_000, tag="highway")
    assert exp.description == "highway"


# The 4 toy edges are 156.8-156.9 km long (geodesic): round(length / res)
# points per edge, at least one
@pytest.mark.parametrize(
    "res, n_pnts", [(10_000, 16), (50_000, 3), (100_000, 2), (400_000, 1)]
)
def test_exposure_from_edges_points_per_edge(funcstate_network, res, n_pnts):
    """Each edge is disaggregated into round(length / res) points (min. 1)."""
    exp = exposure_from_edges(funcstate_network, "road", res=res)
    pnts_per_edge = exp.gdf.groupby(level=0, sort=False).size()
    assert pnts_per_edge.index.tolist() == [0, 1, 2, 3]
    assert pnts_per_edge.tolist() == [n_pnts] * 4


def test_exposure_from_edges_has_value(funcstate_network):
    """Without disagg_val and disagg_col, every point has the value 1."""
    exp = exposure_from_edges(funcstate_network, "road", res=10_000)
    assert (exp.gdf["value"] == 1).all()


def test_exposure_from_edges_disagg_col_without_disagg_val(funcstate_network):
    """disagg_col is used when disagg_val is not given (FIX: every point gets
    the value of its edge)."""
    exp = exposure_from_edges(
        funcstate_network, "road", res=10_000, disagg_col="osm_id"
    )
    values = exp.gdf["value"].groupby(level=0, sort=False)
    assert values.min().tolist() == [100, 101, 102, 103]
    assert values.max().tolist() == [100, 101, 102, 103]


def test_exposure_from_edges_disagg_val_overrides_disagg_col(funcstate_network):
    exp = exposure_from_edges(
        funcstate_network, "road", res=10_000, disagg_val=5, disagg_col="osm_id"
    )
    assert (exp.gdf["value"] == 5).all()


def test_exposure_from_edges_value_from_column(funcstate_network):
    """With disagg_val=None, values are taken from disagg_col and divided
    over the points, so the total per edge is preserved."""
    exp = exposure_from_edges(
        funcstate_network,
        "road",
        res=10000,
        disagg_val=None,
        disagg_col="distance",
        disagg_met=u_lp.DisaggMethod.DIV,
    )

    total = funcstate_network.edges.loc[
        funcstate_network.edges["ci_type"] == "road", "distance"
    ].sum()
    assert exp.gdf["value"].sum() == pytest.approx(total)


def test_exposure_from_edges_lat_lon(funcstate_network):
    """Latitude and longitude of the points lie on the edges (on the diagonal
    lat == lon of the toy network, between 0 and 4)."""
    exp = exposure_from_edges(funcstate_network, "road", res=10_000)
    assert exp.latitude == pytest.approx(exp.longitude)
    assert ((exp.latitude > 0) & (exp.latitude < 4)).all()


def test_exposure_from_edges_missing_disagg_col_raises(funcstate_network):
    with pytest.raises(ValueError, match="disagg_col='width'"):
        exposure_from_edges(funcstate_network, "road", res=10_000, disagg_col="width")


def test_exposure_from_edges_unknown_ci_type_raises(funcstate_network):
    with pytest.raises(ValueError, match="(?i)no edges"):
        exposure_from_edges(funcstate_network, "rail", res=10_000)


def test_exposure_from_edges_mask(funcstate_network):
    """mask selects a subset of the edges of the ci_type."""
    mask = funcstate_network.edges["osm_id"].isin([100, 102])
    exp = exposure_from_edges(funcstate_network, "road", res=10_000, mask=mask)
    assert exp.gdf.index.get_level_values(0).unique().tolist() == [0, 2]


def test_exposure_from_edges_includes_added_links_by_default(funcstate_network):
    """Links added by add_physical_links / link_clusters are exposed like the
    other edges of the ci_type."""
    label = _with_added_link(funcstate_network)

    exp = exposure_from_edges(funcstate_network, "road", res=10_000)

    assert exp.gdf.index.get_level_values(0).unique().tolist() == [0, 1, 2, 3, label]


def test_exposure_from_edges_skips_added_links(funcstate_network):
    """With include_added=False, the added links are not exposed."""
    _with_added_link(funcstate_network)

    exp = exposure_from_edges(
        funcstate_network, "road", res=10_000, include_added=False
    )

    assert exp.gdf.index.get_level_values(0).unique().tolist() == [0, 1, 2, 3]


def test_exposure_from_edges_only_added_links_raises(funcstate_network):
    label = _with_added_link(funcstate_network)
    mask = funcstate_network.edges.index == label
    with pytest.raises(ValueError, match="include_added=True"):
        exposure_from_edges(
            funcstate_network, "road", res=10_000, mask=mask, include_added=False
        )


# ========================================================================
# Initialisation
# ========================================================================


def test_network_impact_calc_init(funcstate_network):
    """Constructor stores all inputs."""
    impf_set = ImpactFuncSet()
    haz = MagicMock()
    nic = NetworkImpactCalc(
        impf_set=impf_set,
        impf_thresh_set={"healthcare": 0.5},
        haz=haz,
        network=funcstate_network,
        exp_list=[],
    )
    assert nic.impf_set is impf_set
    assert nic.network is funcstate_network
    assert nic.haz is haz
    assert nic.exp_list == []
    assert nic.impf_thresh_set == {"healthcare": 0.5}
    assert nic.edge_rule == "max_ratio"  # default


# ========================================================================
# Point impacts
# ========================================================================


def test_calc_point_impacts_returns_impact():
    """calc_point_impacts returns an Impact with imp_mat."""
    impf_set = ImpactFuncSet([_make_impf("healthcare")])
    exp = Exposures(
        lat=np.array([4.0]),
        lon=np.array([4.0]),
        value=np.array([1.0]),
    )
    exp.gdf["impf_TC"] = "healthcare"

    imp = NetworkImpactCalc.calc_point_impacts(
        _make_hazard(HAZ_DIAGONAL), exp, impf_set
    )

    assert imp.imp_mat is not None
    assert imp.imp_mat.shape == (1, 1)


def test_calc_point_impacts_intensity_matters():
    """Higher hazard intensity produces larger impacts."""
    impf_set = ImpactFuncSet([_make_impf("healthcare")])
    # Two points: low-intensity centroid (0,0) and high-intensity centroid (4,4)
    exp = Exposures(
        lat=np.array([0.0, 4.0]),
        lon=np.array([0.0, 4.0]),
        value=np.array([1.0, 1.0]),
    )
    exp.gdf["impf_TC"] = "healthcare"

    imp = NetworkImpactCalc.calc_point_impacts(
        _make_hazard(HAZ_DIAGONAL), exp, impf_set
    )

    impacts = imp.imp_mat.toarray().flatten()
    assert impacts[0] < impacts[1]


# ========================================================================
# Impacts written to the network
# ========================================================================


def test_impacts_to_network_writes_rows(funcstate_network):
    """imp_dir and func_internal are written to the given rows; func_tot is
    the minimum of func_internal and the previous func_tot."""
    nodes = funcstate_network.nodes.copy()

    _impacts_to_network(nodes, pd.Index([4]), [0.8], [0])

    assert nodes.loc[4, ["imp_dir", "func_internal", "func_tot"]].tolist() == [
        0.8,
        0,
        0,
    ]


def test_impacts_to_network_other_rows_unchanged(funcstate_network):
    nodes = funcstate_network.nodes.copy()

    _impacts_to_network(nodes, pd.Index([4]), [0.8], [0])

    others = nodes.drop(index=4)
    assert (others["func_internal"] == 1).all()
    assert (others["func_tot"] == 1).all()
    assert (others["imp_dir"] == 0).all()


def test_impacts_to_network_func_tot_is_minimum(funcstate_network):
    """A row that already failed stays failed, even if functional now."""
    nodes = funcstate_network.nodes.copy()
    nodes.loc[[1, 2, 3], "func_tot"] = [0, 1, 0]

    _impacts_to_network(nodes, pd.Index([1, 2, 3]), [0.1, 0.2, 0.9], [1, 1, 0])

    assert nodes.loc[[1, 2, 3], "func_internal"].tolist() == [1, 1, 0]
    assert nodes.loc[[1, 2, 3], "func_tot"].tolist() == [0, 1, 0]


def test_impacts_to_network_by_label_not_position(funcstate_network):
    """Values follow the order of the labels, not the order of the rows."""
    nodes = funcstate_network.nodes.copy()

    _impacts_to_network(nodes, pd.Index([3, 1]), [0.9, 0.1], [0, 1])

    assert nodes.loc[[1, 3], "imp_dir"].tolist() == [0.1, 0.9]
    assert nodes.loc[[1, 3], "func_internal"].tolist() == [1, 0]


def test_impacts_to_network_edges(funcstate_network):
    edges = funcstate_network.edges.copy()

    _impacts_to_network(edges, edges.index, [0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])

    assert edges["imp_dir"].tolist() == pytest.approx([0.1, 0.2, 0.8, 0.9])
    assert edges["func_internal"].tolist() == [1, 1, 0, 0]
    assert edges["func_tot"].tolist() == [1, 1, 0, 0]


# ========================================================================
# Network disruption
# ========================================================================


def test_disrupt_network_node_exposure(funcstate_network):
    """Disruption with node exposure updates node func states."""
    exp = exposure_from_nodes(funcstate_network, "healthcare", value=1)

    network_disr, imp_dict = _run_disrupt(
        funcstate_network,
        [exp],
        [_make_impf("healthcare")],
        {"healthcare": 0.3},
        _make_hazard(HAZ_DIAGONAL),
    )

    # Healthcare node at (4,4) has intensity=50 ⟶ impact 0.75 > 0.3
    hc = network_disr.nodes[network_disr.nodes["ci_type"] == "healthcare"]
    assert hc["imp_dir"].iloc[0] == pytest.approx(0.75)
    assert hc["func_internal"].iloc[0] == 0  # above threshold
    assert hc["func_tot"].iloc[0] == 0
    assert "healthcare" in imp_dict
    # Original network is not mutated
    assert all(funcstate_network.nodes["func_tot"] == 1)


def test_disrupt_network_edge_exposure(funcstate_network):
    """Disruption with edge exposure updates edge func states."""
    exp = exposure_from_edges(funcstate_network, "road", res=10_000)

    network_disr, imp_dict = _run_disrupt(
        funcstate_network,
        [exp],
        [_make_impf("road")],
        {"road": 0.3},
        _make_hazard(HAZ_DIAGONAL),
    )

    # Largest damage ratio of the points of each edge: edge 0 (intensities
    # 10 and 20) 0, edge 1 (20 and 30) 0.25, edge 2 (30 and 40) 0.5, edge 3
    # (40 and 50) 0.75. Edges 2 and 3 are above the threshold 0.3.
    assert network_disr.edges["func_internal"].tolist() == [1, 1, 0, 0]
    assert network_disr.edges["imp_dir"].iloc[0] == 0
    assert (network_disr.edges["imp_dir"].iloc[1:] > 0).all()
    assert "road" in imp_dict
    # Original network is not mutated
    assert all(funcstate_network.edges["func_tot"] == 1)


def test_disrupt_network_returns_deepcopy(funcstate_network):
    """disrupt_network returns a deep copy; original is preserved."""
    exp = exposure_from_nodes(funcstate_network, "healthcare", value=1)

    network_disr, _ = _run_disrupt(
        funcstate_network,
        [exp],
        [_make_impf("healthcare")],
        {"healthcare": 0.3},
        _make_hazard(HAZ_DIAGONAL),
    )

    # Disrupted network is a different object
    assert network_disr is not funcstate_network
    assert network_disr.nodes is not funcstate_network.nodes


def test_disrupt_network_multiple_exposures(funcstate_network):
    """Disruption with multiple exposures updates both nodes and edges."""
    exp_nodes = exposure_from_nodes(funcstate_network, "healthcare", value=1)
    exp_edges = exposure_from_edges(funcstate_network, "road", res=10_000)

    network_disr, imp_dict = _run_disrupt(
        funcstate_network,
        [exp_nodes, exp_edges],
        [_make_impf("healthcare"), _make_impf("road")],
        {"healthcare": 0.3, "road": 0.3},
        _make_hazard(HAZ_DIAGONAL),
    )

    assert "healthcare" in imp_dict
    assert "road" in imp_dict
    assert "func_internal" in network_disr.nodes.columns
    assert "func_internal" in network_disr.edges.columns


def test_disrupt_network_empty_exp_list(funcstate_network):
    """Empty exposure list returns unchanged network copy."""
    network_disr, imp_dict = _run_disrupt(
        funcstate_network, [], [], {}, _make_hazard(HAZ_DIAGONAL)
    )

    assert len(imp_dict) == 0
    assert all(network_disr.nodes["func_tot"] == 1)
    assert all(network_disr.edges["func_tot"] == 1)


def test_disrupt_network_edge_imp_dir_is_sum_of_points(funcstate_network):
    """imp_dir of an edge is the sum of the impacts at its points."""
    # intensity 30 everywhere -> 0.25 at every point (value 1)
    exp = exposure_from_edges(funcstate_network, "road", res=10_000)
    n_pnts = exp.gdf.groupby(level=0, sort=False).size().to_numpy()

    network_disr, imp_dict = _run_disrupt(
        funcstate_network,
        [exp],
        [_make_impf("road")],
        {"road": 1e9},
        _make_hazard([[30.0] * 5]),
    )

    assert _ci_values(network_disr.edges, "road", "imp_dir") == pytest.approx(
        (0.25 * n_pnts).tolist()
    )
    assert imp_dict["road"].imp_mat.shape == (1, 4)


def test_disrupt_network_edge_fails_if_one_part_is_damaged(funcstate_network):
    """An edge fails when part of it is fully damaged (a road cut at one place
    is cut). Holds for the current rule (sum of point impacts) and for the
    per-point damage-ratio rule of DI-1; it would not hold for a mean rule."""
    # only centroid 2 (2, 2) is damaging: edges 1 and 2 touch it
    exp = exposure_from_edges(funcstate_network, "road", res=10_000)

    network_disr, _ = _run_disrupt(
        funcstate_network,
        [exp],
        [_make_impf("road")],
        {"road": 0.5},
        _make_hazard([[10.0, 10.0, 60.0, 10.0, 10.0]]),
    )

    assert _ci_values(network_disr.edges, "road", "func_internal") == [1, 0, 0, 1]


def test_disrupt_network_keeps_existing_failures(funcstate_network):
    """func_tot = min(func_internal, func_tot): a component that already failed
    (e.g. in the base-state cascade) stays failed when the hazard spares it."""
    funcstate_network.nodes.loc[
        funcstate_network.nodes["ci_type"] == "healthcare", "func_tot"
    ] = 0
    exp = exposure_from_nodes(funcstate_network, "healthcare", value=1)

    network_disr, _ = _run_disrupt(
        funcstate_network,
        [exp],
        [_make_impf("healthcare")],
        {"healthcare": 0.5},
        _make_hazard([[10.0] * 5]),  # impact 0 everywhere
    )

    assert _ci_values(network_disr.nodes, "healthcare", "func_internal") == [1]
    assert _ci_values(network_disr.nodes, "healthcare", "func_tot") == [0]


def test_disrupt_network_multi_event_raises(funcstate_network):
    """A hazard with several events raises a clear error."""
    exp = exposure_from_nodes(funcstate_network, "healthcare", value=1)
    haz = _make_hazard([[10, 20, 30, 40, 50], [50, 40, 30, 20, 10]])

    with pytest.raises(ValueError, match="one event"):
        _run_disrupt(
            funcstate_network,
            [exp],
            [_make_impf("healthcare")],
            {"healthcare": 0.3},
            haz,
        )


def test_disrupt_network_custom_tag(funcstate_network):
    """The tag selects the impact function and threshold; impacts are written
    to the nodes the exposure was built from."""
    exp = exposure_from_nodes(funcstate_network, "healthcare", value=1, tag="hospital")

    network_disr, imp_dict = _run_disrupt(
        funcstate_network,
        [exp],
        [_make_impf("hospital")],
        {"hospital": 0.3},
        _make_hazard(HAZ_DIAGONAL),
    )

    assert _ci_values(network_disr.nodes, "healthcare", "func_internal") == [0]
    assert "hospital" in imp_dict


def test_disrupt_network_subsets_of_one_ci_type(funcstate_network):
    """Two road exposures (complementary masks), each with its own impact
    function and threshold, update only their own edges; ci_type stays road."""
    main = funcstate_network.edges["osm_id"].isin([100, 101])  # edges 0, 1
    exp_main = exposure_from_edges(
        funcstate_network, "road", res=10_000, mask=main, tag="road_main"
    )
    exp_track = exposure_from_edges(
        funcstate_network, "road", res=10_000, mask=~main, tag="road_track"
    )
    # step functions: main roads damaged from intensity 45, tracks from 25
    impf_main = _make_impf(
        "road_main", intensity=(0, 44.9, 45, 100), mdd=(0, 0, 1, 1)
    )
    impf_track = _make_impf(
        "road_track", intensity=(0, 24.9, 25, 100), mdd=(0, 0, 1, 1)
    )

    network_disr, imp_dict = _run_disrupt(
        funcstate_network,
        [exp_main, exp_track],
        [impf_main, impf_track],
        {"road_main": 0.5, "road_track": 0.5},
        _make_hazard(HAZ_DIAGONAL),
    )

    edges = network_disr.edges
    # edge 1 reaches intensity 30: undamaged as a main road (a track would fail)
    assert edges["func_internal"].tolist() == [1, 1, 0, 0]
    assert edges["imp_dir"].iloc[:2].tolist() == [0, 0]
    assert (edges["ci_type"] == "road").all()
    assert imp_dict["road_main"].imp_mat.shape == (1, 2)
    assert imp_dict["road_track"].imp_mat.shape == (1, 2)


def test_disrupt_network_threshold_inclusive(funcstate_network):
    """An impact equal to the threshold leaves the component functional."""
    exp = exposure_from_nodes(funcstate_network, "healthcare", value=1)

    network_disr, _ = _run_disrupt(
        funcstate_network,
        [exp],
        [_make_impf("healthcare")],
        {"healthcare": 0.75},  # impact at the hospital: 0.75
        _make_hazard(HAZ_DIAGONAL),
    )

    assert _ci_values(network_disr.nodes, "healthcare", "func_internal") == [1]


@pytest.mark.parametrize("threshold", [0.01, 0.99])
@pytest.mark.parametrize("intensity, func_internal", [(49.9, 1), (50.0, 0)])
def test_disrupt_network_step_impact_function(
    funcstate_network, threshold, intensity, func_internal
):
    """With ImpactFunc.from_step_impf, a component fails from the step
    intensity on (included), whatever the threshold between 0 and 1."""
    exp = exposure_from_nodes(funcstate_network, "healthcare", value=1)
    impf = ImpactFunc.from_step_impf(
        (0, 50, 100), haz_type="TC", impf_id="healthcare"
    )

    network_disr, _ = _run_disrupt(
        funcstate_network,
        [exp],
        [impf],
        {"healthcare": threshold},
        _make_hazard([[0, 0, 0, 0, intensity]]),  # hospital at node 4
    )

    assert _ci_values(network_disr.nodes, "healthcare", "func_internal") == [
        func_internal
    ]


def test_disrupt_network_does_not_modify_exposures(funcstate_network):
    """The exposures get no impf_* / centr_* columns."""
    exp_nodes = exposure_from_nodes(funcstate_network, "healthcare", value=1)
    exp_edges = exposure_from_edges(funcstate_network, "road", res=10_000)
    columns = [exp.gdf.columns.tolist() for exp in (exp_nodes, exp_edges)]

    _run_disrupt(
        funcstate_network,
        [exp_nodes, exp_edges],
        [_make_impf("healthcare"), _make_impf("road")],
        {"healthcare": 0.3, "road": 0.3},
        _make_hazard(HAZ_DIAGONAL),
    )

    assert [exp.gdf.columns.tolist() for exp in (exp_nodes, exp_edges)] == columns


def test_disrupt_network_exposure_ci_type_changed_raises(funcstate_network):
    """A row whose ci_type differs between exposure and network raises."""
    exp = exposure_from_nodes(funcstate_network, "healthcare", value=1)
    funcstate_network.nodes.loc[4, "ci_type"] = "school"

    with pytest.raises(ValueError, match="node 4 has ci_type 'school' in the network"):
        _run_disrupt(
            funcstate_network,
            [exp],
            [_make_impf("healthcare")],
            {"healthcare": 0.3},
            _make_hazard(HAZ_DIAGONAL),
        )


def test_disrupt_network_edges_of_any_ci_type(funcstate_network):
    """Line exposures are recognised from the exposure itself, not from the
    hard-coded LINE_EXPOSURES list."""
    funcstate_network.edges["ci_type"] = "pipeline"
    exp = exposure_from_edges(funcstate_network, "pipeline", res=10_000)

    network_disr, _ = _run_disrupt(
        funcstate_network,
        [exp],
        [_make_impf("pipeline")],
        {"pipeline": 0.2},
        _make_hazard(HAZ_DIAGONAL),
    )

    assert _ci_values(network_disr.edges, "pipeline", "func_internal") == [1, 0, 0, 0]
    assert (network_disr.nodes["func_internal"] == 1).all()


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


def test_disrupt_network_missing_impf_raises(funcstate_network):
    """No impact function for the exposure tag -> clear ValueError."""
    exp = exposure_from_nodes(funcstate_network, "healthcare", value=1)

    with pytest.raises(ValueError, match="impact function"):
        _run_disrupt(
            funcstate_network,
            [exp],
            [_make_impf("road")],
            {"healthcare": 0.3},
            _make_hazard(HAZ_DIAGONAL),
        )


def test_disrupt_network_missing_threshold_raises(funcstate_network):
    """No threshold for the exposure tag -> clear ValueError."""
    exp = exposure_from_nodes(funcstate_network, "healthcare", value=1)

    with pytest.raises(ValueError, match="threshold"):
        _run_disrupt(
            funcstate_network,
            [exp],
            [_make_impf("healthcare")],
            {"road": 0.3},
            _make_hazard(HAZ_DIAGONAL),
        )


def test_disrupt_network_duplicate_tags_raises(funcstate_network):
    """Two exposures with the same tag (here the default tag "road") raise."""
    nodes = funcstate_network.nodes
    exp_a = exposure_from_nodes(funcstate_network, "road", mask=nodes["id"] == 1)
    exp_b = exposure_from_nodes(funcstate_network, "road", mask=nodes["id"] == 2)

    with pytest.raises(ValueError, match="Several exposures have the tag"):
        _run_disrupt(
            funcstate_network,
            [exp_a, exp_b],
            [_make_impf("road")],
            {"road": 0.3},
            _make_hazard(HAZ_DIAGONAL),
        )


def test_disrupt_network_overlapping_exposures_raises(funcstate_network):
    """A node in two exposures (overlapping masks) raises."""
    nodes = funcstate_network.nodes
    exp_all = exposure_from_nodes(funcstate_network, "road", tag="road_all")
    exp_2 = exposure_from_nodes(
        funcstate_network, "road", mask=nodes["id"] == 2, tag="road_2"
    )

    with pytest.raises(ValueError, match="'road_all' and 'road_2' both contain node 2"):
        _run_disrupt(
            funcstate_network,
            [exp_all, exp_2],
            [_make_impf("road_all"), _make_impf("road_2")],
            {"road_all": 0.3, "road_2": 0.3},
            _make_hazard(HAZ_DIAGONAL),
        )


def test_disrupt_network_same_label_in_nodes_and_edges(funcstate_network):
    """Node 4 and edge 4 are different rows: no overlap between exposures."""
    label = _with_added_link(funcstate_network)  # edge 4
    exp_nodes = exposure_from_nodes(funcstate_network, "healthcare")  # node 4
    exp_edges = exposure_from_edges(funcstate_network, "road", res=10_000)
    assert label == 4

    _, imp_dict = _run_disrupt(
        funcstate_network,
        [exp_nodes, exp_edges],
        [_make_impf("healthcare"), _make_impf("road")],
        {"healthcare": 0.3, "road": 0.3},
        _make_hazard(HAZ_DIAGONAL),
    )

    assert set(imp_dict) == {"healthcare", "road"}


def test_disrupt_network_exposure_not_from_network_raises(
    funcstate_network, network_with_remote_node_missing_edge
):
    """An exposure with points that are not rows of the network raises."""
    other = network_with_remote_node_missing_edge  # healthcare nodes 4 and 5
    other.initialize_funcstates()
    exp = exposure_from_nodes(other, "healthcare", value=1)

    with pytest.raises(ValueError, match="not in the network"):
        _run_disrupt(
            funcstate_network,
            [exp],
            [_make_impf("healthcare")],
            {"healthcare": 0.3},
            _make_hazard(HAZ_DIAGONAL),
        )


def test_disrupt_network_threshold_on_damage_ratio_nodes(funcstate_network):
    """A fully flooded people cell fails whatever its population count
    (resampled rasters give fractional counts, e.g. 0.3)."""
    funcstate_network.nodes["counts"] = [0.3, np.nan, np.nan, np.nan, np.nan]
    exp = exposure_from_nodes(funcstate_network, "people", value_col="counts")
    people_impf = _make_impf("people", intensity=(0, 5, 6, 100), mdd=(0, 0, 1, 1))

    network_disr, _ = _run_disrupt(
        funcstate_network,
        [exp],
        [people_impf],
        {"people": 0.5},
        _make_hazard(HAZ_DIAGONAL),
    )

    assert _ci_values(network_disr.nodes, "people", "imp_dir") == pytest.approx([0.3])
    assert _ci_values(network_disr.nodes, "people", "func_internal") == [0]


@pytest.mark.parametrize("res", [10_000, 50_000])
def test_disrupt_network_threshold_on_damage_ratio_edges(funcstate_network, res):
    """A damage ratio of 0.25 at every point, below the threshold 0.3, leaves
    every edge functional at any disaggregation resolution."""
    exp = exposure_from_edges(funcstate_network, "road", res=res)

    network_disr, _ = _run_disrupt(
        funcstate_network,
        [exp],
        [_make_impf("road")],
        {"road": 0.3},
        _make_hazard([[30.0] * 5]),
    )

    assert _ci_values(network_disr.edges, "road", "func_internal") == [1, 1, 1, 1]


def test_disrupt_network_damage_ratio_at_value_zero(funcstate_network):
    """A flooded people cell with 0 people fails too: the damage ratio does not
    depend on the exposure value."""
    funcstate_network.nodes["counts"] = [0.0, np.nan, np.nan, np.nan, np.nan]
    exp = exposure_from_nodes(funcstate_network, "people", value_col="counts")
    people_impf = _make_impf("people", intensity=(0, 5, 6, 100), mdd=(0, 0, 1, 1))

    network_disr, _ = _run_disrupt(
        funcstate_network,
        [exp],
        [people_impf],
        {"people": 0.5},
        _make_hazard(HAZ_DIAGONAL),
    )

    assert _ci_values(network_disr.nodes, "people", "imp_dir") == [0]
    assert _ci_values(network_disr.nodes, "people", "func_internal") == [0]


def test_disrupt_network_edge_imp_dir_with_div(funcstate_network):
    """With DIV, imp_dir of an edge is the damage to the edge value; the
    functional state still follows the damage ratio (0.25 <= 0.3)."""
    exp = exposure_from_edges(
        funcstate_network,
        "road",
        res=10_000,
        disagg_col="osm_id",  # 100, 101, 102, 103
        disagg_met=u_lp.DisaggMethod.DIV,
    )

    network_disr, _ = _run_disrupt(
        funcstate_network,
        [exp],
        [_make_impf("road")],
        {"road": 0.3},
        _make_hazard([[30.0] * 5]),  # damage ratio 0.25 at every point
    )

    assert _ci_values(network_disr.edges, "road", "imp_dir") == pytest.approx(
        [25.0, 25.25, 25.5, 25.75]
    )
    assert _ci_values(network_disr.edges, "road", "func_internal") == [1, 1, 1, 1]


# The 4 toy edges are 156.8-156.9 km long: with a damage ratio of 0.25 at
# every point, each edge has 39.2 km damaged
@pytest.mark.parametrize("res", [10_000, 50_000])
@pytest.mark.parametrize(
    "threshold, func_internal", [(39_000, [0, 0, 0, 0]), (40_000, [1, 1, 1, 1])]
)
def test_disrupt_network_damaged_length(
    funcstate_network, res, threshold, func_internal
):
    """With the edge rule "damaged_length", an edge fails if its damaged
    length in metres is above the threshold, whatever res."""
    exp = exposure_from_edges(funcstate_network, "road", res=res)

    network_disr, _ = _run_disrupt(
        funcstate_network,
        [exp],
        [_make_impf("road")],
        {"road": threshold},
        _make_hazard([[30.0] * 5]),  # damage ratio 0.25 at every point
        edge_rule="damaged_length",
    )

    assert _ci_values(network_disr.edges, "road", "func_internal") == func_internal


@pytest.mark.parametrize(
    "threshold, func_internal", [(100_000, [1, 1, 1, 1]), (50_000, [1, 0, 0, 1])]
)
def test_disrupt_network_damaged_length_partly_damaged_edges(
    funcstate_network, threshold, func_internal
):
    """Edges 1 and 2 are fully damaged on half of their length (78.4 km): they
    fail only if the threshold is below that. With the edge rule "max_ratio"
    they fail in any case (test_disrupt_network_edge_fails_if_one_part_is_damaged)."""
    exp = exposure_from_edges(funcstate_network, "road", res=10_000)

    network_disr, _ = _run_disrupt(
        funcstate_network,
        [exp],
        [_make_impf("road")],
        {"road": threshold},
        _make_hazard([[10.0, 10.0, 60.0, 10.0, 10.0]]),  # damaging at node 2 only
        edge_rule="damaged_length",
    )

    assert _ci_values(network_disr.edges, "road", "func_internal") == func_internal


def test_disrupt_network_edge_rule_per_tag(funcstate_network):
    """edge_rule as a dict sets the rule of some edge exposures; the others use
    "max_ratio"."""
    main = funcstate_network.edges["osm_id"].isin([100, 101])  # edges 0, 1
    exp_main = exposure_from_edges(
        funcstate_network, "road", res=10_000, mask=main, tag="road_main"
    )
    exp_track = exposure_from_edges(
        funcstate_network, "road", res=10_000, mask=~main, tag="road_track"
    )

    network_disr, _ = _run_disrupt(
        funcstate_network,
        [exp_main, exp_track],
        [_make_impf("road_main"), _make_impf("road_track")],
        # main roads: 39.2 km damaged <= 40 km; tracks: ratio 0.25 > 0.2
        {"road_main": 40_000, "road_track": 0.2},
        _make_hazard([[30.0] * 5]),  # damage ratio 0.25 at every point
        edge_rule={"road_main": "damaged_length"},
    )

    assert network_disr.edges["func_internal"].tolist() == [1, 1, 0, 0]


def test_disrupt_network_unknown_edge_rule_raises(funcstate_network):
    exp = exposure_from_edges(funcstate_network, "road", res=10_000)

    with pytest.raises(ValueError, match="Unknown edge_rule 'mean'"):
        _run_disrupt(
            funcstate_network,
            [exp],
            [_make_impf("road")],
            {"road": 0.3},
            _make_hazard(HAZ_DIAGONAL),
            edge_rule="mean",
        )


def test_disrupt_network_edge_rule_for_node_exposure_raises(funcstate_network):
    """A dict edge_rule may only name edge exposures."""
    exp = exposure_from_nodes(funcstate_network, "healthcare")

    with pytest.raises(ValueError, match="not tags of edge exposures"):
        _run_disrupt(
            funcstate_network,
            [exp],
            [_make_impf("healthcare")],
            {"healthcare": 0.3},
            _make_hazard(HAZ_DIAGONAL),
            edge_rule={"healthcare": "damaged_length"},
        )


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
