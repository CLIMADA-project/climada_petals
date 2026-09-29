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

Tests for the Network data container (nw_base)
"""

import copy as cp

import geopandas as gpd
import igraph as ig
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from climada_petals.engine.networks.nw_base import Network


# ========================================================================
# Initialisation
# ========================================================================


def test_init_empty():
    """An empty network has empty edges/nodes with the required columns."""
    network = Network()

    assert network.edges.empty
    assert network.nodes.empty
    assert {"from_id", "to_id", "id", "orig_id", "geometry"} <= set(
        network.edges.columns
    )
    assert {"id", "orig_id", "geometry"} <= set(network.nodes.columns)
    assert network.crs.to_string() == "EPSG:4326"


def test_init_with_data(edges_gdf, nodes_gdf):
    """Edges and nodes are stored with their CRS."""
    network = Network(edges=edges_gdf, nodes=nodes_gdf)

    assert len(network.nodes) == 5
    assert len(network.edges) == 4
    assert network.crs.to_string() == "EPSG:4326"
    assert network.nodes.crs.to_string() == "EPSG:4326"
    assert network.edges.crs.to_string() == "EPSG:4326"


def test_init_nodes_only_takes_crs_from_nodes(nodes_projected_gdf):
    """With nodes only, the empty edges get the CRS of the nodes."""
    network = Network(nodes=nodes_projected_gdf)

    assert network.edges.empty
    assert network.edges.crs.to_string() == "EPSG:32632"
    assert network.crs.to_string() == "EPSG:32632"


def test_init_adds_missing_id_columns():
    """Missing 'id' and 'orig_id' columns are added as sequential integers."""
    edges = gpd.GeoDataFrame(
        {
            "from_id": [0, 1],
            "to_id": [1, 2],
            "geometry": [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    nodes = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    network = Network(edges=edges, nodes=nodes)

    assert network.edges["id"].tolist() == [0, 1]
    assert network.edges["orig_id"].tolist() == [0, 1]
    assert network.nodes["id"].tolist() == [0, 1, 2]
    assert network.nodes["orig_id"].tolist() == [0, 1, 2]


def test_init_keeps_existing_ids(edges_gdf, nodes_gdf):
    """Existing 'id' and 'orig_id' columns are not overwritten."""
    nodes_gdf["orig_id"] = [10, 11, 12, 13, 14]

    network = Network(edges=edges_gdf, nodes=nodes_gdf)

    assert network.nodes["orig_id"].tolist() == [10, 11, 12, 13, 14]


def test_init_crs_mismatch_raises(edges_gdf, nodes_gdf):
    """Edges and nodes with different CRS raise a ValueError."""
    with pytest.raises(ValueError, match="same CRS"):
        Network(edges=edges_gdf.to_crs("EPSG:3857"), nodes=nodes_gdf)


# ========================================================================
# Reprojection
# ========================================================================


def test_reproject(edges_gdf, nodes_gdf):
    """reproject returns a new network with transformed coordinates."""
    network = Network(edges=edges_gdf, nodes=nodes_gdf)

    network_rp = Network.reproject(network, "EPSG:3857")

    assert network_rp.crs.to_string() == "EPSG:3857"
    assert network_rp.nodes.crs.to_string() == "EPSG:3857"
    assert network_rp.edges.crs.to_string() == "EPSG:3857"
    # (1°E, 1°N) in Web Mercator
    assert network_rp.nodes.geometry.iloc[1].x == pytest.approx(111319.49, rel=1e-6)
    assert network_rp.nodes.geometry.iloc[1].y == pytest.approx(111325.14, rel=1e-6)
    # the original network is unchanged
    assert network.crs.to_string() == "EPSG:4326"
    assert network.nodes.geometry.iloc[1].x == 1


# ========================================================================
# Combining networks
# ========================================================================


def test_from_networks_single_network(edges_gdf, nodes_gdf):
    """Combining a single network returns the same nodes and edges."""
    network = Network(edges=edges_gdf, nodes=nodes_gdf)

    combined = Network.from_networks([network])

    assert len(combined.nodes) == 5
    assert len(combined.edges) == 4
    assert combined.nodes["id"].tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:32632"])
def test_from_networks_offsets_ids(
    crs, edges_gdf, nodes_gdf, edges_projected_gdf, nodes_projected_gdf
):
    """Node ids and edge endpoints of later networks are offset."""
    if crs == "EPSG:4326":
        edges, nodes = edges_gdf, nodes_gdf
    else:
        edges, nodes = edges_projected_gdf, nodes_projected_gdf
    n_nodes, n_edges = len(nodes), len(edges)
    network1 = Network(edges=edges.copy(), nodes=nodes.copy())
    network2 = Network(edges=edges.copy(), nodes=nodes.copy())

    combined = Network.from_networks([network1, network2])

    assert combined.crs.to_string() == crs
    assert combined.nodes.crs.to_string() == crs
    assert combined.edges.crs.to_string() == crs
    assert len(combined.nodes) == 2 * n_nodes
    assert len(combined.edges) == 2 * n_edges
    assert combined.nodes["id"].tolist() == list(range(2 * n_nodes))
    # edges of the second network point to the offset node ids
    second = combined.edges.iloc[n_edges:]
    assert second["from_id"].tolist() == (edges["from_id"] + n_nodes).tolist()
    assert second["to_id"].tolist() == (edges["to_id"] + n_nodes).tolist()


def test_from_networks_nodes_only_network(edges_gdf, nodes_gdf):
    """A nodes-only network (e.g. people) can be combined with a line network."""
    roads = Network(edges=edges_gdf, nodes=nodes_gdf)
    people = Network(nodes=nodes_gdf.copy())

    combined = Network.from_networks([roads, people])

    assert len(combined.nodes) == 10
    assert len(combined.edges) == 4


def test_from_networks_crs_mismatch_raises(
    edges_gdf, nodes_gdf, edges_projected_gdf, nodes_projected_gdf
):
    """Combining networks with different CRS raises a ValueError."""
    network_geo = Network(edges=edges_gdf, nodes=nodes_gdf)
    network_proj = Network(edges=edges_projected_gdf, nodes=nodes_projected_gdf)

    with pytest.raises(ValueError, match="same CRS"):
        Network.from_networks([network_geo, network_proj])
    with pytest.raises(ValueError, match="same CRS"):
        Network.from_networks([network_proj, network_geo])


# ========================================================================
# Conversion to and from igraph
# ========================================================================


@pytest.mark.parametrize("directed", [False, True])
def test_to_graph(edges_gdf, nodes_gdf, directed):
    """to_graph builds a graph with all nodes, edges and their attributes."""
    network = Network(edges=edges_gdf, nodes=nodes_gdf)

    graph = network.to_graph(directed=directed)

    assert isinstance(graph, ig.Graph)
    assert graph.is_directed() == directed
    assert graph.vcount() == 5
    assert graph.ecount() == 4
    assert [(e.source, e.target) for e in graph.es] == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert graph.es["osm_id"] == [100, 101, 102, 103]
    assert graph.vs["orig_id"] == [0, 1, 2, 3, 4]


def test_to_graph_nodes_only(nodes_gdf):
    """A network without edges gives a graph with isolated vertices."""
    network = Network(nodes=nodes_gdf)

    graph = network.to_graph(directed=True)

    assert graph.vcount() == 5
    assert graph.ecount() == 0
    assert graph.vs["id"] == [0, 1, 2, 3, 4]


def test_to_graph_drops_name_column(edges_gdf, nodes_gdf):
    """A 'name' column (reserved by igraph) is not passed to the graph."""
    nodes_gdf["name"] = ["a", "b", "c", "d", "e"]
    network = Network(edges=edges_gdf, nodes=nodes_gdf)

    graph = network.to_graph()

    assert "name" not in graph.vs.attributes()
    # the network itself keeps the column
    assert "name" in network.nodes.columns


def test_from_graphs_roundtrip(network_with_ci_types):
    """to_graph followed by from_graphs keeps nodes, edges and attributes."""
    graph = network_with_ci_types.to_graph(directed=True)

    network = Network.from_graphs(graph, crs=network_with_ci_types.crs)

    assert network.crs.to_string() == "EPSG:4326"
    assert network.nodes["id"].tolist() == [0, 1, 2, 3, 4]
    assert network.nodes["ci_type"].tolist() == network_with_ci_types.nodes[
        "ci_type"
    ].tolist()
    assert network.edges["from_id"].tolist() == [0, 1, 2, 3]
    assert network.edges["to_id"].tolist() == [1, 2, 3, 4]
    assert (network.edges["ci_type"] == "road").all()


def test_from_graphs_with_added_vertex_and_edge(network_with_ci_types):
    """Vertices and edges added to the graph end up in the network."""
    graph = cp.deepcopy(network_with_ci_types).to_graph(directed=True)
    graph.add_vertex(
        id=5, orig_id=5, ci_type="healthcare", func_tot=1, geometry=Point(5, 5)
    )
    graph.add_edge(2, 5, ci_type="road", geometry=LineString([(2, 2), (5, 5)]))

    network = Network.from_graphs(graph, crs=network_with_ci_types.crs)

    assert len(network.nodes) == 6
    assert len(network.edges) == 5
    assert network.nodes["id"].tolist() == [0, 1, 2, 3, 4, 5]
    assert network.nodes.iloc[5]["ci_type"] == "healthcare"
    new_edge = network.edges.iloc[4]
    assert (new_edge["from_id"], new_edge["to_id"]) == (2, 5)
    assert new_edge["ci_type"] == "road"


# ========================================================================
# Saving and loading
# ========================================================================


def test_save_and_load_network_zip(edges_gdf, nodes_gdf, temp_dir):
    """A saved network is loaded back identically."""
    network = Network(edges=edges_gdf, nodes=nodes_gdf)

    zip_path = network.save_network_zip(temp_dir, "test_network")
    loaded = Network.load_network_zip(temp_dir, "test_network")

    assert zip_path.exists()
    assert loaded.crs.to_string() == "EPSG:4326"
    pd.testing.assert_frame_equal(
        loaded.nodes.reset_index(drop=True), network.nodes.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        loaded.edges.reset_index(drop=True), network.edges.reset_index(drop=True)
    )


def test_save_and_load_nodes_only_network(nodes_gdf, temp_dir):
    """A network without edges is saved and loaded with empty edges."""
    network = Network(nodes=nodes_gdf)

    network.save_network_zip(temp_dir, "nodes_only")
    loaded = Network.load_network_zip(temp_dir, "nodes_only")

    assert len(loaded.nodes) == 5
    assert loaded.edges.empty


def test_save_and_load_empty_network(temp_dir):
    """An empty network can be saved and loaded."""
    zip_path = Network().save_network_zip(temp_dir, "empty_network")
    loaded = Network.load_network_zip(temp_dir, "empty_network")

    assert zip_path.exists()
    assert loaded.nodes.empty
    assert loaded.edges.empty


def test_load_network_zip_nonexistent(temp_dir):
    """Loading a missing archive returns an empty network."""
    network = Network.load_network_zip(temp_dir, "nonexistent")

    assert network.nodes.empty
    assert network.edges.empty


# ========================================================================
# Initialisation of functional states, capacities and supply
# ========================================================================


def test_initialize_funcstates(edges_gdf, nodes_gdf):
    """All components start fully functional and without direct impact."""
    network = Network(edges=edges_gdf, nodes=nodes_gdf)

    network.initialize_funcstates()

    for gdf in (network.edges, network.nodes):
        assert (gdf["func_internal"] == 1).all()
        assert (gdf["func_tot"] == 1).all()
        assert (gdf["imp_dir"] == 0).all()


def test_initialize_capacity(network_with_ci_types):
    """Sources get capacity 1, targets -1 and all other nodes 0."""
    dep_table = pd.DataFrame({"source": ["road"], "target": ["healthcare"]})

    network_with_ci_types.initialize_capacity(dep_table)

    # ci_types: people, road, road, road, healthcare
    assert network_with_ci_types.nodes["capacity_road_healthcare"].tolist() == [
        0,
        1,
        1,
        1,
        -1,
    ]


def test_initialize_supply(network_with_ci_types):
    """Enduser dependencies get supply 0 and an initial access state."""
    dep_table = pd.DataFrame(
        {"source": ["healthcare"], "target": ["people"], "type_I": ["enduser"]}
    )

    network_with_ci_types.initialize_supply(dep_table)

    nodes = network_with_ci_types.nodes
    is_people = nodes["ci_type"] == "people"
    assert (nodes["actual_supply_healthcare_people"] == 0).all()
    assert (
        nodes.loc[is_people, "access_state_healthcare_people"] == "no base access"
    ).all()
    assert (nodes.loc[~is_people, "access_state_healthcare_people"] == "undefined").all()


def test_initialize_supply_ignores_functional_dependencies(network_with_ci_types):
    """Only enduser dependencies get supply and access state columns."""
    dep_table = pd.DataFrame(
        {"source": ["road"], "target": ["healthcare"], "type_I": ["functional"]}
    )

    network_with_ci_types.initialize_supply(dep_table)

    assert "actual_supply_road_healthcare" not in network_with_ci_types.nodes.columns
    assert "access_state_road_healthcare" not in network_with_ci_types.nodes.columns
