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

Tests for the power functionality of the network module

Power toy network (``_power_network``; EPSG:4326 near the equator, 0.01 degree
is about 1.1 km)::

    cluster A                               cluster B

    N1      E1          E2                         E3
    L1 ---- L2 -------- L3                  L4 --- L5
    P1                                      P2         H

- Plants P1 (generation 100) and P2 (30), linked to L1 and L4 by power lines
  added by the module.
- Electrified people E1 (demand 60), E2 (40), E3 (50) and a hospital H depend
  on their closest power line node; N1 is not electrified (no link).

The expected values follow the original model (``feature/networks``,
``GraphCalcs.powercap_from_clusters`` and ``_propagate_check_fail``, used for
Mühlhofer et al. 2024): in each power cluster, the supply / demand ratio
(capped at 1) is the capacity of its line nodes; dependants receive
``func_tot * capacity`` from their line node and need ``thresh_func``.

Two layers of checks:

- hand-computed cases (``SCENARIOS``);
- a copy of the original algorithm (``_reference_cascade``), itself checked on
  the hand-computed cases, compared with the new code on random power networks.

Tests marked with ``_xfail(step, ...)`` describe behaviour implemented in the
given step of the project doc ``claude/nw-power-review.md`` (strict: the step
must remove the marker).
"""

import copy as cp

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from shapely.geometry import LineString, Point

from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_calcs import NetworkCalcs
from climada_petals.engine.networks.nw_utils import (
    ACCESS_DISRUPTED,
    ACCESS_NO_BASE,
    ACCESS_UNDISRUPTED,
    ADDED_COL,
    _edge_lengths_metres,
)


def _xfail(step, reason):
    """xfail marker for behaviour implemented in the given step of the plan."""
    return pytest.mark.xfail(
        strict=True, reason=f"Implemented in step {step}: {reason}"
    )


# ========================================================================
# Fixtures
# ========================================================================

# Node labels of the power toy network -> node id (= row position)
POWER_IDS = {
    "L1": 0,
    "L2": 1,
    "L3": 2,
    "L4": 3,
    "L5": 4,
    "P1": 5,
    "P2": 6,
    "E1": 7,
    "E2": 8,
    "N1": 9,
    "E3": 10,
    "H": 11,
}

_COORDS = {
    "L1": (0, 0),
    "L2": (0.01, 0),
    "L3": (0.02, 0),
    "L4": (0.10, 0),
    "L5": (0.11, 0),
    "P1": (0, -0.005),
    "P2": (0.10, -0.005),
    "E1": (0.009, 0.003),  # 352 m from L2, about 1 km from L1 and L3
    "E2": (0.02, 0.003),  # 334 m from L3
    "N1": (0, 0.003),
    "E3": (0.11, 0.003),  # 334 m from L5
    "H": (0.112, -0.003),  # 401 m from L5
}

_DEPENDENCY_LINKS = [
    ("L2", "E1", "dependency_power_line_people"),
    ("L3", "E2", "dependency_power_line_people"),
    ("L5", "E3", "dependency_power_line_people"),
    ("L5", "H", "dependency_power_line_healthcare"),
]


def _edges_gdf(links, coords, crs="EPSG:4326"):
    """Edges from ``(from, to, ci_type, added)`` tuples of node labels."""
    edges = gpd.GeoDataFrame(
        {
            "from_id": [POWER_IDS[src] for src, _, _, _ in links],
            "to_id": [POWER_IDS[tgt] for _, tgt, _, _ in links],
            "ci_type": [ci_type for _, _, ci_type, _ in links],
            ADDED_COL: [added for _, _, _, added in links],
            "geometry": [
                LineString([coords[src], coords[tgt]]) for src, tgt, _, _ in links
            ],
        },
        geometry="geometry",
        crs=crs,
    )
    edges["distance"] = _edge_lengths_metres(edges)
    return edges


def _power_network(with_dependencies):
    """Power toy network (see the module docstring).

    With ``with_dependencies``, the dependency edges to the closest line node
    are included (L2->E1, L3->E2, L5->E3, L5->H), as ``setup_dependencies``
    creates them with ``link_condition="closest"`` and
    ``target_filter="electrified"``.
    """
    labels = list(POWER_IDS)
    ci_types = {"L": "power_line", "P": "power_plant", "E": "people", "N": "people"}
    nodes = gpd.GeoDataFrame(
        {
            "id": range(len(labels)),
            "ci_type": [
                "healthcare" if lab == "H" else ci_types[lab[0]] for lab in labels
            ],
            "el_generation": [
                {"P1": 100.0, "P2": 30.0}.get(lab, np.nan) for lab in labels
            ],
            "el_consumption": [
                {"E1": 60.0, "E2": 40.0, "N1": 0.0, "E3": 50.0}.get(lab, np.nan)
                for lab in labels
            ],
            "counts": [
                {"E1": 60.0, "E2": 40.0, "N1": 30.0, "E3": 50.0}.get(lab, np.nan)
                for lab in labels
            ],
            "electrified": [lab in ("E1", "E2", "E3") for lab in labels],
            "geometry": [Point(_COORDS[lab]) for lab in labels],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    links = [
        ("L1", "L2", "power_line", False),
        ("L2", "L3", "power_line", False),
        ("L4", "L5", "power_line", False),
        ("L1", "P1", "power_line", True),
        ("L4", "P2", "power_line", True),
    ]
    if with_dependencies:
        links += [(src, tgt, ci_type, True) for src, tgt, ci_type in _DEPENDENCY_LINKS]
    return Network(edges=_edges_gdf(links, _COORDS), nodes=nodes)


@pytest.fixture
def power_network():
    """Power toy network without dependency edges."""
    return _power_network(with_dependencies=False)


@pytest.fixture
def power_network_with_dependencies():
    """Power toy network with the dependency edges to the closest line node."""
    return _power_network(with_dependencies=True)


def _power_dependency_table(thresh_people=0.6, thresh_healthcare=0.6, **columns):
    """Dependency table of the original power model: electrified people and
    hospitals depend on their closest power line node (straight line).
    ``columns`` replace columns for both rows."""
    table = pd.DataFrame(
        {
            "source": ["power_line", "power_line"],
            "target": ["people", "healthcare"],
            "type_I": ["enduser", "functional"],
            "type_II": ["physical", "physical"],
            "via_link": [None, None],
            "thresh_func": [thresh_people, thresh_healthcare],
            "link_condition": ["closest", "closest"],
            "thresh_dist": [np.inf, np.inf],
            "thresh_dur": [np.nan, np.nan],
            "bidir_link": [False, False],
            "access_cnstr": [False, False],
            "n_links": [1, 1],
            "target_filter": ["electrified", None],
        }
    )
    for col, value in columns.items():
        table[col] = value
    return table


@pytest.fixture
def power_dependency_table():
    """Power dependency table, ``thresh_func`` 0.6 for people and hospitals."""
    return _power_dependency_table()


# ========================================================================
# Helpers
# ========================================================================


def _values(network, labels, col):
    """Values of a node column for toy-network labels."""
    nodes = network.nodes.set_index("id")
    return [nodes.loc[POWER_IDS[lab], col] for lab in labels]


def _pairs(*label_pairs):
    """``{(source id, target id)}`` from label pairs."""
    return {(POWER_IDS[src], POWER_IDS[tgt]) for src, tgt in label_pairs}


def _dependency_pairs(network, ci_type):
    """``{(from_id, to_id)}`` of the edges of a type."""
    edges = network.edges[network.edges["ci_type"] == ci_type]
    return {(int(src), int(tgt)) for src, tgt in zip(edges["from_id"], edges["to_id"])}


def _fail(network, nodes=(), edges=()):
    """Set ``func_internal`` and ``func_tot`` to 0 for toy-network nodes
    (labels) and power-line edges (label pairs, either direction)."""
    ids = [POWER_IDS[lab] for lab in nodes]
    network.nodes.loc[network.nodes["id"].isin(ids), ["func_internal", "func_tot"]] = 0
    for src, tgt in edges:
        a, b = POWER_IDS[src], POWER_IDS[tgt]
        from_id, to_id = network.edges["from_id"], network.edges["to_id"]
        mask = (network.edges["ci_type"] == "power_line") & (
            ((from_id == a) & (to_id == b)) | ((from_id == b) & (to_id == a))
        )
        assert mask.sum() == 1
        network.edges.loc[mask, ["func_internal", "func_tot"]] = 0


def _with_link(network, src, tgt, ci_type):
    """Copy of a toy network with one more edge."""
    coords = {lab: _COORDS[lab] for lab in (src, tgt)}
    new = _edges_gdf([(src, tgt, ci_type, True)], coords, crs=network.crs)
    edges = pd.concat([network.edges.drop(columns="id"), new], ignore_index=True)
    return Network(
        edges=gpd.GeoDataFrame(edges, geometry="geometry", crs=network.crs),
        nodes=network.nodes.copy(),
    )


def _without_link(network, src, tgt):
    """Copy of a toy network without the edge src -> tgt."""
    edges = network.edges
    mask = (edges["from_id"] == POWER_IDS[src]) & (edges["to_id"] == POWER_IDS[tgt])
    assert mask.sum() == 1
    return Network(
        edges=edges[~mask].drop(columns="id").reset_index(drop=True),
        nodes=network.nodes.copy(),
    )


def _base_then_disrupted(network, dep_table, failures, access_check_method):
    """New code: base-state cascade, then the failures, then the cascade again."""
    nc_base = NetworkCalcs(network, dep_table=dep_table)
    nc_base.initialize_base_state()
    nc_base.cascade(access_check_method=access_check_method)
    disrupted = nc_base.network
    _fail(disrupted, **failures)
    nc_disr = NetworkCalcs(disrupted, dep_table=dep_table)
    nc_disr.cascade(access_check_method=access_check_method)
    return nc_disr.network


# ========================================================================
# Reference: the original algorithm (feature/networks)
# ========================================================================
# Copied from feature/networks (GraphCalcs.powercap_from_clusters,
# _propagate_check_fail, cascade), changed only where needed to run with the
# current igraph and the current network:
# - connected_components() instead of the deprecated clusters();
# - subgraph -> graph vertex mapping through a temporary attribute instead of
#   relying on the vertex order of the induced subgraph;
# - power lines are stored once (the original stored both directions and set
#   the reverse of a failed line to 0 too: same clusters);
# - only roads are left out of the internal update (no roads here).


def _ref_subgraph(graph, ci_types):
    """Induced subgraph of the given vertex types, with ``_ref_id`` = graph id."""
    graph.vs["_ref_id"] = list(range(graph.vcount()))
    subgraph = graph.induced_subgraph(graph.vs.select(ci_type_in=ci_types))
    del graph.vs["_ref_id"]
    return subgraph


def _ref_powercap(graph, p_source, p_sink, demand_ci, source_var, demand_var):
    """Original ``powercap_from_clusters``."""
    capacity_vars = [
        var for var in graph.vs.attributes() if f"capacity_{p_sink}_" in var
    ]
    # subgraph spanning all power vertices and the people, functional edges only
    subgraph = _ref_subgraph(graph, ["power_line", p_source, p_sink, demand_ci])
    subgraph.delete_edges(subgraph.es.select(func_tot_lt=0.1))

    for cluster in subgraph.connected_components(mode="weak"):
        sources = subgraph.vs[cluster].select(ci_type=p_source)
        sinks = subgraph.vs[cluster].select(ci_type=p_sink)
        demands = subgraph.vs[cluster].select(ci_type=demand_ci)

        psupply = sum(source[source_var] * source["func_tot"] for source in sources)
        pdemand = sum(demand[demand_var] for demand in demands)
        sd_ratio = min(1, psupply / pdemand) if pdemand else 1

        for var in capacity_vars:
            graph.vs[sinks["_ref_id"]][var] = sd_ratio


def _ref_propagate(graph, source, target, thresh_func):
    """Original ``_propagate_check_fail``: capacities propagated along all
    edges between source and target vertices (adjacency matrix)."""
    subgraph = _ref_subgraph(graph, [source, target])
    adj_sub = subgraph.get_adjacency_sparse()
    func_capa = np.array(subgraph.vs["func_tot"], dtype=float) * np.array(
        subgraph.vs[f"capacity_{source}_{target}"], dtype=float
    )
    capa_rec = np.asarray(
        scipy.sparse.csr_matrix(func_capa).dot(adj_sub).todense()
    ).ravel()
    is_target = np.array(subgraph.vs["ci_type"]) == target
    capa_suff = (capa_rec >= np.where(is_target, thresh_func, 0)).astype(int)
    graph_ids = np.array(subgraph.vs["_ref_id"])
    if target == "people":
        graph.vs[graph_ids[is_target].tolist()][f"actual_supply_{source}_{target}"] = (
            capa_suff[is_target].tolist()
        )
    else:
        func_tot = np.minimum(capa_suff, np.array(subgraph.vs["func_tot"], dtype=float))
        graph.vs[graph_ids.tolist()]["func_tot"] = func_tot.tolist()


def _reference_cascade(
    network,
    dep_table,
    p_source="power_plant",
    p_sink="power_line",
    demand_ci="people",
    source_var="el_generation",
    demand_var="el_consumption",
):
    """Original cascade (power update, functional dependencies until no
    change, then end users with fixed links) on a copy of the network."""
    graph = network.to_graph(directed=True)
    functional = dep_table[dep_table["type_I"] == "functional"]
    enduser = dep_table[dep_table["type_I"] == "enduser"]
    delta = -1
    while delta != 0:
        func_before = sum(graph.vs["func_tot"])
        _ref_powercap(graph, p_source, p_sink, demand_ci, source_var, demand_var)
        for _, row in functional.iterrows():
            _ref_propagate(graph, row["source"], row["target"], row["thresh_func"])
        delta = abs(func_before - sum(graph.vs["func_tot"]))
    for _, row in enduser.iterrows():
        _ref_propagate(graph, row["source"], row["target"], row["thresh_func"])
    return Network.from_graphs(graph, crs=network.crs)


# ========================================================================
# Hand-computed cases
# ========================================================================

_BASE_CAPACITY = {"L1": 1, "L2": 1, "L3": 1, "L4": 0.6, "L5": 0.6}
_BASE_SUPPLY = {"E1": 1, "E2": 1, "N1": 0, "E3": 1}

SCENARIOS = {
    # A: 100 / 100 -> 1; B: 30 / 50 = 0.6, just enough for thresh_func 0.6
    "base": {
        "failures": {},
        "thresh": 0.6,
        "capacity": _BASE_CAPACITY,
        "supply": _BASE_SUPPLY,
        "healthcare": 1,
    },
    # A splits: {P1, L1, L2, E1} 100 / 60 -> 1; {L3, E2} has no plant -> 0
    "line_L2_L3": {
        "failures": {"edges": [("L2", "L3")]},
        "thresh": 0.6,
        "capacity": {**_BASE_CAPACITY, "L3": 0},
        "supply": {**_BASE_SUPPLY, "E2": 0},
        "healthcare": 1,
    },
    # B has no supply: E3 loses power, the hospital fails
    "plant_P2": {
        "failures": {"nodes": ["P2"]},
        "thresh": 0.6,
        "capacity": {**_BASE_CAPACITY, "L4": 0, "L5": 0},
        "supply": {**_BASE_SUPPLY, "E3": 0},
        "healthcare": 0,
    },
    # a failed line node does not split its cluster; it delivers nothing
    "node_L2": {
        "failures": {"nodes": ["L2"]},
        "thresh": 0.6,
        "capacity": _BASE_CAPACITY,
        "supply": {**_BASE_SUPPLY, "E1": 0},
        "healthcare": 1,
    },
    # B covers 60 % of its demand; 70 % are needed
    "thresh_0.7": {
        "failures": {},
        "thresh": 0.7,
        "capacity": _BASE_CAPACITY,
        "supply": {**_BASE_SUPPLY, "E3": 0},
        "healthcare": 0,
    },
}


def _assert_scenario(network, scenario):
    capacity = scenario["capacity"]
    for col in ("capacity_power_line_people", "capacity_power_line_healthcare"):
        np.testing.assert_allclose(
            _values(network, list(capacity), col), list(capacity.values())
        )
    supply = scenario["supply"]
    assert _values(network, list(supply), "actual_supply_power_line_people") == list(
        supply.values()
    )
    assert _values(network, ["H"], "func_tot") == [scenario["healthcare"]]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_reference_reproduces_hand_cases(power_network_with_dependencies, name):
    """The copy of the original algorithm gives the hand-computed values."""
    scenario = SCENARIOS[name]
    dep_table = _power_dependency_table(scenario["thresh"], scenario["thresh"])
    nc = NetworkCalcs(power_network_with_dependencies, dep_table=dep_table)
    nc.initialize_base_state()
    network = nc.network
    _fail(network, **scenario["failures"])

    _assert_scenario(_reference_cascade(network, dep_table), scenario)


def test_reference_merges_clusters_through_people(
    power_network_with_dependencies, power_dependency_table
):
    """Original quirk Q2: people linked to two clusters merge them. E3 also
    linked to L3 and P2 failed: one cluster, 100 / 150 -> 2/3 everywhere, so
    the hospital in cluster B (no working plant) stays functional."""
    network = _with_link(
        power_network_with_dependencies, "L3", "E3", "dependency_power_line_people"
    )
    nc = NetworkCalcs(network, dep_table=power_dependency_table)
    nc.initialize_base_state()
    failed = nc.network
    _fail(failed, nodes=["P2"])

    reference = _reference_cascade(failed, power_dependency_table)

    np.testing.assert_allclose(
        _values(reference, ["L1", "L5"], "capacity_power_line_people"), [2 / 3, 2 / 3]
    )
    assert _values(reference, ["H"], "func_tot") == [1]


# ========================================================================
# Dependency setup: closest links and target filter (P2)
# ========================================================================


def _setup(network, dep_table):
    nc = NetworkCalcs(network, dep_table=dep_table)
    nc.initialize_base_state()
    nc.setup_dependencies()
    return nc.network


def test_setup_dependencies_closest(power_network, power_dependency_table):
    """Each electrified person and the hospital are linked to their closest
    line node (straight line); N1 is not electrified."""
    network = _setup(power_network, power_dependency_table)

    assert _dependency_pairs(network, "dependency_power_line_people") == _pairs(
        ("L2", "E1"), ("L3", "E2"), ("L5", "E3")
    )
    assert _dependency_pairs(network, "dependency_power_line_healthcare") == _pairs(
        ("L5", "H")
    )


def test_setup_dependencies_closest_without_target_filter(power_network):
    """Without a target filter, every person is linked."""
    dep_table = _power_dependency_table(target_filter=None)

    network = _setup(power_network, dep_table)

    assert _dependency_pairs(network, "dependency_power_line_people") == _pairs(
        ("L2", "E1"), ("L3", "E2"), ("L1", "N1"), ("L5", "E3")
    )


def test_setup_dependencies_closest_n_links(power_network):
    """``n_links`` closest line nodes per target."""
    dep_table = _power_dependency_table(n_links=2)

    network = _setup(power_network, dep_table)

    assert _dependency_pairs(network, "dependency_power_line_people") == _pairs(
        ("L2", "E1"),
        ("L1", "E1"),
        ("L3", "E2"),
        ("L2", "E2"),
        ("L5", "E3"),
        ("L4", "E3"),
    )


def test_setup_dependencies_closest_thresh_dist(power_network):
    """Straight-line distance in metres: E2 and E3 are 334 m from their line
    node, E1 352 m and H 401 m."""
    dep_table = _power_dependency_table(thresh_dist=340)

    network = _setup(power_network, dep_table)

    assert _dependency_pairs(network, "dependency_power_line_people") == _pairs(
        ("L3", "E2"), ("L5", "E3")
    )
    assert _dependency_pairs(network, "dependency_power_line_healthcare") == set()


def test_setup_dependencies_target_filter_missing_column_raises(
    power_network, power_dependency_table
):
    power_network.nodes = power_network.nodes.drop(columns="electrified")
    nc = NetworkCalcs(power_network, dep_table=power_dependency_table)
    nc.initialize_base_state()

    with pytest.raises(ValueError, match="electrified"):
        nc.setup_dependencies()


def test_setup_dependencies_cascade_without_warning(network_calcs, nw_caplog):
    """Tutorial order: dependency links created after ``initialize_base_state``
    have no functional state, and the cascade does not report them (B5)."""
    network_calcs.initialize_base_state()
    network_calcs.setup_dependencies()
    edges = network_calcs.network.edges
    dep_edges = edges[edges["ci_type"].str.startswith("dependency_")]
    assert not dep_edges.empty
    assert dep_edges["func_tot"].isna().all()

    network_calcs.cascade(access_check_method="propagation")

    assert "have no functional state" not in nw_caplog.text


# ========================================================================
# Power clusters (P3; access_check_method="propagation")
# ========================================================================


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_power_cascade_hand_cases(power_network_with_dependencies, name):
    scenario = SCENARIOS[name]
    dep_table = _power_dependency_table(scenario["thresh"], scenario["thresh"])

    network = _base_then_disrupted(
        power_network_with_dependencies,
        dep_table,
        scenario["failures"],
        "propagation",
    )

    _assert_scenario(network, scenario)


def test_power_cascade_access_states(
    power_network_with_dependencies, power_dependency_table
):
    """Line L2-L3 fails: E2 loses its supply, N1 never had one."""
    network = _base_then_disrupted(
        power_network_with_dependencies,
        power_dependency_table,
        {"edges": [("L2", "L3")]},
        "propagation",
    )

    assert _values(network, ["E1", "E2", "N1"], "access_state_power_line_people") == [
        ACCESS_UNDISRUPTED,
        ACCESS_DISRUPTED,
        ACCESS_NO_BASE,
    ]


def test_power_cluster_without_demand_has_full_capacity(
    power_network_with_dependencies, power_dependency_table
):
    """No person depends on cluster B: ratio 1 (as in the original)."""
    network = _without_link(power_network_with_dependencies, "L5", "E3")

    network = _base_then_disrupted(network, power_dependency_table, {}, "propagation")

    np.testing.assert_allclose(
        _values(network, ["L4", "L5"], "capacity_power_line_people"), [1, 1]
    )
    assert _values(network, ["H"], "func_tot") == [1]


def test_power_clusters_not_merged_through_people(
    power_network_with_dependencies, power_dependency_table
):
    """Same case as ``test_reference_merges_clusters_through_people``:
    clusters are found on the power network only, so cluster B (failed plant)
    has no supply and the hospital fails. Intended difference to the
    original, only for people with links to several clusters."""
    network = _with_link(
        power_network_with_dependencies, "L3", "E3", "dependency_power_line_people"
    )

    network = _base_then_disrupted(
        network, power_dependency_table, {"nodes": ["P2"]}, "propagation"
    )

    np.testing.assert_allclose(
        _values(network, ["L4", "L5"], "capacity_power_line_people"), [0, 0]
    )
    assert _values(network, ["H"], "func_tot") == [0]


def test_power_cascade_missing_source_var_raises(
    power_network_with_dependencies, power_dependency_table
):
    network = power_network_with_dependencies
    network.nodes = network.nodes.drop(columns="el_generation")
    nc = NetworkCalcs(network, dep_table=power_dependency_table)
    nc.initialize_base_state()

    with pytest.raises(ValueError, match="el_generation"):
        nc.cascade(access_check_method="propagation")


# ========================================================================
# Fixed links without access constraint in routing mode (P4)
# ========================================================================


@_xfail("P4", "rows without access constraint keep their links")
def test_routing_keeps_links_without_access_constraint(
    power_network_with_dependencies, power_dependency_table
):
    """Line node L2 fails: E1 is not re-linked to another line node."""
    network = _base_then_disrupted(
        power_network_with_dependencies,
        power_dependency_table,
        {"nodes": ["L2"]},
        "routing",
    )

    assert _values(network, ["E1"], "actual_supply_power_line_people") == [0]
    assert _dependency_pairs(network, "dependency_power_line_people") == _pairs(
        ("L2", "E1"), ("L3", "E2"), ("L5", "E3")
    )


@_xfail("P4", "rows without access constraint propagate capacity")
def test_routing_uses_cluster_capacity(power_network_with_dependencies):
    """Cluster B covers 60 % of its demand, 70 % are needed."""
    dep_table = _power_dependency_table(0.7, 0.7)

    network = _base_then_disrupted(
        power_network_with_dependencies, dep_table, {}, "routing"
    )

    _assert_scenario(network, SCENARIOS["thresh_0.7"])


@_xfail("P4", "closest links, power clusters and fixed links together")
def test_power_end_to_end(power_network, power_dependency_table):
    """As in the tutorial: setup_dependencies, base-state cascade, then a
    failed line node and the cascade again (default routing mode)."""
    nc = NetworkCalcs(power_network, dep_table=power_dependency_table)
    nc.initialize_base_state()
    nc.setup_dependencies()
    nc.cascade()
    base = nc.network
    _assert_scenario(base, SCENARIOS["base"])

    _fail(base, nodes=["L2"])
    nc_disr = NetworkCalcs(base, dep_table=power_dependency_table)
    nc_disr.cascade()

    _assert_scenario(nc_disr.network, SCENARIOS["node_L2"])


# ========================================================================
# New code vs the original algorithm on random power networks
# ========================================================================

SEEDS = range(8)


def _random_power_network(seed):
    """Random power network with explicit dependency edges and failures.

    12 line nodes forming a random forest (several clusters), 3 plants linked
    to random line nodes, 10 people (80 % electrified, each linked to one
    random line node: k = 1 as in the original runs), 3 hospitals linked to
    random line nodes. Failures: power-line edges (incl. plant links) 20 %,
    line nodes 15 %, plants 25 %. ``thresh_func`` 0.3, 0.6 or 1.

    Returns
    -------
    network, dep_table, failed node ids, failed edge row labels
    """
    rng = np.random.default_rng(seed)
    n_lines, n_plants, n_people, n_health = 12, 3, 10, 3
    ci_types = (
        ["power_line"] * n_lines
        + ["power_plant"] * n_plants
        + ["people"] * n_people
        + ["healthcare"] * n_health
    )
    n_nodes = len(ci_types)
    lines = np.arange(n_lines)
    plants = n_lines + np.arange(n_plants)
    people = n_lines + n_plants + np.arange(n_people)
    health = n_lines + n_plants + n_people + np.arange(n_health)

    xy = rng.uniform(0, 0.2, size=(n_nodes, 2))
    el_generation = np.full(n_nodes, np.nan)
    el_generation[plants] = rng.uniform(10, 100, n_plants)
    electrified = np.zeros(n_nodes, dtype=bool)
    electrified[people] = rng.random(n_people) < 0.8
    el_consumption = np.full(n_nodes, np.nan)
    el_consumption[people] = np.where(
        electrified[people], rng.uniform(5, 40, n_people), 0
    )

    links = []
    for i in range(1, n_lines):
        if rng.random() < 0.75:
            links.append((int(rng.integers(i)), i, "power_line"))
    for plant in plants:
        links.append((int(rng.integers(n_lines)), int(plant), "power_line"))
    for person in people[electrified[people]]:
        links.append(
            (int(rng.integers(n_lines)), int(person), "dependency_power_line_people")
        )
    for hospital in health:
        links.append(
            (
                int(rng.integers(n_lines)),
                int(hospital),
                "dependency_power_line_healthcare",
            )
        )

    nodes = gpd.GeoDataFrame(
        {
            "id": range(n_nodes),
            "ci_type": ci_types,
            "el_generation": el_generation,
            "el_consumption": el_consumption,
            "electrified": electrified,
            "geometry": [Point(x, y) for x, y in xy],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    edges = gpd.GeoDataFrame(
        {
            "from_id": [src for src, _, _ in links],
            "to_id": [tgt for _, tgt, _ in links],
            "ci_type": [ci_type for _, _, ci_type in links],
            "geometry": [LineString([xy[src], xy[tgt]]) for src, tgt, _ in links],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    edges["distance"] = _edge_lengths_metres(edges)

    is_line = (edges["ci_type"] == "power_line").to_numpy()
    failed_edges = edges.index[is_line & (rng.random(len(edges)) < 0.2)].tolist()
    failed_nodes = [int(i) for i in lines if rng.random() < 0.15] + [
        int(i) for i in plants if rng.random() < 0.25
    ]
    thresh_people, thresh_health = rng.choice([0.3, 0.6, 1.0], size=2)
    dep_table = _power_dependency_table(float(thresh_people), float(thresh_health))
    return Network(edges=edges, nodes=nodes), dep_table, failed_nodes, failed_edges


def _random_failed_network(seed):
    """Random network in its initial state with the failures applied."""
    network, dep_table, failed_nodes, failed_edges = _random_power_network(seed)
    nc = NetworkCalcs(network, dep_table=dep_table)
    nc.initialize_base_state()
    failed = nc.network
    failed.nodes.loc[
        failed.nodes["id"].isin(failed_nodes), ["func_internal", "func_tot"]
    ] = 0
    failed.edges.loc[failed_edges, ["func_internal", "func_tot"]] = 0
    return failed, dep_table


def _by_id(network, ci_type, col):
    """Values of a node column for one ci_type, sorted by node id."""
    nodes = network.nodes[network.nodes["ci_type"] == ci_type].sort_values("id")
    return nodes[col].to_numpy(dtype=float)


_COMPARED = [
    ("power_line", "capacity_power_line_people"),
    ("people", "actual_supply_power_line_people"),
    ("healthcare", "func_tot"),
]


def _compare_with_reference(access_check_method):
    """Run the new cascade and the reference on each random network (one
    cascade from the initial state with the failures applied) and assert
    that line capacities, people supply and hospital states are equal; the
    error message lists every mismatch by seed."""
    mismatches = []
    for seed in SEEDS:
        failed, dep_table = _random_failed_network(seed)
        expected = _reference_cascade(cp.deepcopy(failed), dep_table)
        nc = NetworkCalcs(failed, dep_table=dep_table)
        nc.cascade(access_check_method=access_check_method)
        for ci_type, col in _COMPARED:
            got = _by_id(nc.network, ci_type, col)
            exp = _by_id(expected, ci_type, col)
            if not np.allclose(got, exp):
                mismatches.append(f"seed {seed}, {col}: {got} != {exp}")
    assert not mismatches, "\n".join(mismatches)


def test_random_power_networks_cover_the_cases():
    """The random networks exercise the rules: cluster ratios 0, between 0 and
    1 and 1; electrified people with and without power; failed and working
    hospitals (reference algorithm)."""
    ratios, supply, health = [], [], []
    for seed in SEEDS:
        failed, dep_table = _random_failed_network(seed)
        reference = _reference_cascade(failed, dep_table)
        ratios.extend(_by_id(reference, "power_line", "capacity_power_line_people"))
        electrified = reference.nodes["electrified"].astype(bool)
        people = reference.nodes[(reference.nodes["ci_type"] == "people") & electrified]
        supply.extend(people["actual_supply_power_line_people"])
        health.extend(_by_id(reference, "healthcare", "func_tot"))
    ratios = np.array(ratios)

    assert (ratios == 0).any()
    assert ((ratios > 0) & (ratios < 1)).any()
    assert (ratios == 1).any()
    assert {0, 1} <= set(supply)
    assert {0, 1} <= set(health)


def test_power_cascade_matches_reference_propagation():
    _compare_with_reference("propagation")


@_xfail("P4", "rows without access constraint keep their links")
def test_power_cascade_matches_reference_routing():
    _compare_with_reference("routing")
