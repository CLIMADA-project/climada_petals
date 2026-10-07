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

Tests for the NetworkCalcs wrapper (nw_calcs)

All tests use the toy chain network ``network_with_ci_types`` (see conftest):

    people(0) -> road(1) -> road(2) -> road(3) -> healthcare(4)

and the ``dependency_table`` fixture:

    road -> people          enduser,    edge condition
    healthcare -> people    enduser,    shortest path via road, access constraint
    road -> healthcare      functional, edge condition
"""

import copy as cp

import igraph as ig
import numpy as np
import pytest

from climada_petals.engine.networks.graph_calcs import GraphCalcs
from climada_petals.engine.networks.nw_calcs import NetworkCalcs


def people_values(network, col):
    """Values of column ``col`` for all people nodes of ``network``."""
    nodes = network.nodes
    return nodes.loc[nodes["ci_type"] == "people", col].tolist()


def prepared_network_calcs(network_calcs):
    """Initialise base state and dependencies, as done before any cascade."""
    network_calcs.initialize_base_state()
    network_calcs.setup_dependencies()
    return network_calcs


def failed_network_calcs(network_calcs, fail_nodes=None, fail_edges=None):
    """New NetworkCalcs on a copy of the prepared network with failed components.

    Parameters
    ----------
    fail_nodes : dict, optional
        ``{column: value}`` selecting nodes to fail, e.g. ``{"ci_type": "healthcare"}``.
    fail_edges : list of int, optional
        Positional indices of edges to fail.
    """
    network = cp.deepcopy(prepared_network_calcs(network_calcs).network)
    if fail_nodes:
        for col, value in fail_nodes.items():
            network.nodes.loc[network.nodes[col] == value, "func_tot"] = 0
    if fail_edges:
        network.edges.loc[fail_edges, "func_tot"] = 0
    return NetworkCalcs(network=network, dep_table=network_calcs.dep_table)


# ========================================================================
# Initialisation and properties
# ========================================================================


def test_init(network_with_ci_types, dependency_table):
    """The constructor stores the inputs and creates a GraphCalcs."""
    nc = NetworkCalcs(network=network_with_ci_types, dep_table=dependency_table)

    assert nc.network is network_with_ci_types
    assert nc.dep_table is dependency_table
    assert isinstance(nc._graph_calc, GraphCalcs)
    assert nc._graph_calc.network is network_with_ci_types


def test_init_without_dep_table(network_with_ci_types):
    """The dependency table is optional."""
    nc = NetworkCalcs(network=network_with_ci_types)

    assert nc.dep_table is None


def test_graph_property(network_calcs):
    """graph returns the igraph representation of the network."""
    graph = network_calcs.graph

    assert isinstance(graph, ig.Graph)
    assert graph.vcount() == len(network_calcs.network.nodes)
    assert graph.ecount() == len(network_calcs.network.edges)


def test_network_setter_updates_graph_calc(network_calcs, network_with_remote_node):
    """Setting a new network keeps the internal GraphCalcs in sync."""
    network_calcs.network = network_with_remote_node

    assert network_calcs._graph_calc.network is network_with_remote_node


@pytest.mark.parametrize("method", ["initialize_base_state", "setup_dependencies"])
def test_methods_without_dep_table_raise(network_with_ci_types, method):
    """Methods needing a dependency table raise if none was given."""
    nc = NetworkCalcs(network=network_with_ci_types)

    with pytest.raises(ValueError, match="dependency table"):
        getattr(nc, method)()


def test_cascade_without_dep_table_raises(network_with_ci_types):
    """cascade raises if no dependency table was given."""
    nc = NetworkCalcs(network=network_with_ci_types)

    with pytest.raises(ValueError, match="dependency table"):
        nc.cascade()


def test_dep_table_without_bidir_column(network_calcs, dependency_table):
    """The bidir_link column is optional."""
    network_calcs.dep_table = dependency_table.drop(columns="bidir_link")

    prepared_network_calcs(network_calcs)

    dep_edges = network_calcs.graph.es.select(ci_type="dependency_healthcare_people")
    assert [(e.source, e.target) for e in dep_edges] == [(4, 0)]


# ========================================================================
# Base state
# ========================================================================


def test_initialize_base_state(network_calcs):
    """Functional states, capacities and supply columns are initialised."""
    network_calcs.initialize_base_state()
    nodes = network_calcs.network.nodes
    edges = network_calcs.network.edges

    for gdf in (nodes, edges):
        assert (gdf["func_tot"] == 1).all()
        assert (gdf["func_internal"] == 1).all()
    for _, row in network_calcs.dep_table.iterrows():
        assert f"capacity_{row.source}_{row.target}" in nodes.columns
    assert people_values(network_calcs.network, "actual_supply_road_people") == [0]
    assert people_values(network_calcs.network, "access_state_healthcare_people") == [
        "no base access"
    ]


# ========================================================================
# Network construction
# ========================================================================


def test_merge_clusters_connected_network(network_calcs):
    """A connected network is left unchanged."""
    n_edges = len(network_calcs.network.edges)

    network_calcs.merge_clusters(ci_type="road", max_iter=2, dist_thresh=np.inf)

    assert len(network_calcs.network.edges) == n_edges


def test_merge_clusters_links_disconnected_node(network_with_remote_node_missing_edge):
    """A disconnected node is linked to the network with a new road edge."""
    nc = NetworkCalcs(network=network_with_remote_node_missing_edge)
    n_edges = len(nc.network.edges)

    nc.merge_clusters(ci_type="road", max_iter=5, dist_thresh=np.inf)

    new_edges = nc.network.edges.iloc[n_edges:]
    assert len(new_edges) == 1
    assert new_edges["ci_type"].tolist() == ["road"]
    # the remote node 5 is linked to its closest node 4
    assert {new_edges["from_id"].iloc[0], new_edges["to_id"].iloc[0]} == {4, 5}
    assert len(nc.graph.connected_components(mode="weak")) == 1


def test_merge_clusters_respects_dist_thresh(network_with_remote_node_missing_edge):
    """No link is added when the clusters are further apart than dist_thresh."""
    nc = NetworkCalcs(network=network_with_remote_node_missing_edge)
    n_edges = len(nc.network.edges)

    nc.merge_clusters(ci_type="road", max_iter=5, dist_thresh=1000)

    assert len(nc.network.edges) == n_edges
    assert len(nc.graph.connected_components(mode="weak")) == 2


def test_add_physical_links(
    network_calcs, physical_dependencies, expected_physical_links
):
    """Physical links are added with expected count, type and node pairs."""
    n_edges = len(network_calcs.network.edges)

    network_calcs.add_physical_links(physical_dependencies)

    added_edges = network_calcs.network.edges.iloc[n_edges:]
    assert len(added_edges) == expected_physical_links["added_edge_count"]
    pairs = {
        ci_type: {
            (int(e.from_id), int(e.to_id))
            for e in added_edges[added_edges["ci_type"] == ci_type].itertuples()
        }
        for ci_type in ("road", "healthcare")
    }
    assert pairs["road"] == expected_physical_links["road_pairs"]
    assert pairs["healthcare"] == expected_physical_links["healthcare_pairs"]


@pytest.mark.parametrize("bidir_link", [True, None])
def test_add_physical_links_bidir(network_calcs, physical_dependencies, bidir_link):
    """bidir_link=True adds the links in both directions; without the column,
    a single link per pair is added."""
    if bidir_link is None:
        physical_dependencies = physical_dependencies.drop(columns="bidir_link")
        expected = {(1, 0), (4, 0)}
    else:
        physical_dependencies["bidir_link"] = bidir_link
        expected = {(1, 0), (0, 1), (4, 0), (0, 4)}
    n_edges = len(network_calcs.network.edges)

    network_calcs.add_physical_links(physical_dependencies)

    added_edges = network_calcs.network.edges.iloc[n_edges:]
    assert {
        (int(e.from_id), int(e.to_id)) for e in added_edges.itertuples()
    } == expected
    assert len(added_edges) == len(expected)


def test_setup_dependencies(network_calcs, expected_dep_pairs):
    """One dependency edge per dependency, between the expected nodes."""
    prepared_network_calcs(network_calcs)

    for dep_link, (exp_sources, exp_targets) in expected_dep_pairs.items():
        dep_edges = [e for e in network_calcs.graph.es if e["ci_type"] == dep_link]
        assert [e.source for e in dep_edges] == exp_sources
        assert [e.target for e in dep_edges] == exp_targets


# ========================================================================
# Cascades
# ========================================================================


@pytest.mark.parametrize("rerouting", [False, True])
def test_cascade_no_failure(network_calcs, rerouting):
    """Without failures, people get supply and undisrupted access."""
    prepared_network_calcs(network_calcs)
    network = network_calcs.network
    assert people_values(network, "actual_supply_road_people") == [0]
    assert people_values(network, "actual_supply_healthcare_people") == [0]
    assert people_values(network, "access_state_healthcare_people") == [
        "no base access"
    ]

    network_calcs.cascade(friction_surf=None, rerouting=rerouting)

    network = network_calcs.network
    assert people_values(network, "actual_supply_road_people") == [1]
    assert people_values(network, "actual_supply_healthcare_people") == [1]
    assert people_values(network, "access_state_road_people") == ["access undisrupted"]
    assert people_values(network, "access_state_healthcare_people") == [
        "access undisrupted"
    ]


def test_cascade_no_failure_propagation(network_calcs):
    """The 'propagation' access check gives the same result without failures."""
    prepared_network_calcs(network_calcs)

    network_calcs.cascade(access_check_method="propagation")

    network = network_calcs.network
    assert people_values(network, "actual_supply_road_people") == [1]
    assert people_values(network, "actual_supply_healthcare_people") == [1]
    assert people_values(network, "access_state_road_people") == ["access undisrupted"]
    assert people_values(network, "access_state_healthcare_people") == [
        "access undisrupted"
    ]


@pytest.mark.parametrize("rerouting", [False, True])
def test_cascade_source_failure(network_calcs, rerouting):
    """A failed hospital cuts healthcare supply but not road supply."""
    nc = failed_network_calcs(network_calcs, fail_nodes={"ci_type": "healthcare"})

    nc.cascade(friction_surf=None, rerouting=rerouting)

    network = nc.network
    assert people_values(network, "actual_supply_road_people") == [1]
    assert people_values(network, "access_state_road_people") == ["access undisrupted"]
    assert people_values(network, "actual_supply_healthcare_people") == [0]
    assert people_values(network, "access_state_healthcare_people") == [
        "access disrupted source"
    ]


@pytest.mark.parametrize("rerouting", [False, True])
def test_cascade_road_failure_propagates_to_healthcare(network_calcs, rerouting):
    """A failed road edge 3->4 disables road node 3, then the hospital that
    depends on it (functional dependency), then people's healthcare access."""
    nc = failed_network_calcs(network_calcs, fail_edges=[3])

    nc.cascade(friction_surf=None, rerouting=rerouting)

    nodes = nc.network.nodes
    assert nodes.loc[3, "func_tot"] == 0  # road node 3
    assert nodes.loc[4, "func_tot"] == 0  # hospital
    assert people_values(nc.network, "actual_supply_road_people") == [1]
    assert people_values(nc.network, "actual_supply_healthcare_people") == [0]
    assert people_values(nc.network, "access_state_healthcare_people") == [
        "access disrupted source"
    ]


@pytest.mark.parametrize("rerouting", [False, True])
def test_cascade_road_failure_disrupts_access_via(network_calcs, rerouting):
    """A failed road edge 1->2 cuts the path to a still functional hospital."""
    nc = failed_network_calcs(network_calcs, fail_edges=[1])

    nc.cascade(friction_surf=None, rerouting=rerouting)

    nodes = nc.network.nodes
    assert nodes.loc[4, "func_tot"] == 1  # hospital still supplied by road node 3
    assert people_values(nc.network, "actual_supply_healthcare_people") == [0]
    assert people_values(nc.network, "access_state_healthcare_people") == [
        "access disrupted via"
    ]
