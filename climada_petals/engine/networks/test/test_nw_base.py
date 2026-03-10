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

Test network modules
"""

import pytest
import tempfile
import shutil
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import igraph as ig
import copy as cp

from climada_petals.engine.networks.test.fixtures_test_networks import *


def test_network_init_empty():
    """Test Network initialization with empty dataframes"""
    network = Network()
    assert network.edges.empty
    assert network.nodes.empty
    assert "from_id" in network.edges.columns
    assert "id" in network.nodes.columns


def test_network_init_with_data(edges_gdf, nodes_gdf):
    """Test Network initialization with data"""
    network = Network(edges=edges_gdf, nodes=nodes_gdf)
    assert len(network.nodes) == 5
    assert len(network.edges) == 4
    assert network.nodes.crs.to_string() == "EPSG:4326"
    assert network.edges.crs.to_string() == "EPSG:4326"


def test_network_init_adds_missing_columns():
    """Test that Network init adds missing required columns"""
    # Create dataframes without orig_id and id columns
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
    assert "orig_id" in network.edges.columns
    assert "id" in network.edges.columns
    assert "orig_id" in network.nodes.columns
    assert "id" in network.nodes.columns


def test_network_init_crs_mismatch(edges_gdf, nodes_gdf):
    """Test that Network init raises error if edges and nodes have different CRS"""
    # Create edges with different CRS
    edges = edges_gdf.to_crs("EPSG:3857")
    with pytest.raises(ValueError):
        Network(edges=edges, nodes=nodes_gdf)


def test_reproject(edges_gdf, nodes_gdf):
    """Test reprojection of network"""
    network = Network(edges=edges_gdf, nodes=nodes_gdf)
    original_crs = network.nodes.crs.to_string()

    network_rp = Network.reproject(network, "EPSG:3857")

    assert network_rp.nodes.crs.to_string() == "EPSG:3857"
    assert network_rp.edges.crs.to_string() == "EPSG:3857"
    assert network_rp.crs.to_string() == "EPSG:3857"
    assert original_crs != network_rp.nodes.crs.to_string()


def test_from_networks_single_network(edges_gdf, nodes_gdf):
    """Test combining a single network"""
    network = Network(edges=edges_gdf, nodes=nodes_gdf)
    combined = Network.from_networks([network])

    assert len(combined.nodes) == 5
    assert len(combined.edges) == 4


def test_from_networks_multiple_networks(edges_gdf, nodes_gdf):
    """Test combining multiple networks"""
    network1 = Network(edges=edges_gdf, nodes=nodes_gdf)
    network2 = Network(edges=edges_gdf, nodes=nodes_gdf)

    combined = Network.from_networks([network1, network2])

    assert len(combined.nodes) == 10  # 5 + 5
    assert len(combined.edges) == 8  # 4 + 4
    # Check that node IDs are properly offset
    assert combined.nodes["id"].max() == 9


def test_to_graph_undirected(edges_gdf, nodes_gdf):
    """Test conversion to undirected igraph"""
    network = Network(edges=edges_gdf, nodes=nodes_gdf)
    graph = network.to_graph(directed=False)

    assert isinstance(graph, ig.Graph)
    assert graph.vcount() == 5
    assert graph.ecount() == 4
    assert not graph.is_directed()


def test_to_graph_directed(edges_gdf, nodes_gdf):
    """Test conversion to directed igraph"""
    network = Network(edges=edges_gdf, nodes=nodes_gdf)
    graph = network.to_graph(directed=True)

    assert isinstance(graph, ig.Graph)
    assert graph.vcount() == 5
    assert graph.ecount() == 4
    assert graph.is_directed()


def test_to_graph_nodes_only(nodes_gdf):
    """Test conversion to graph with no edges"""
    network = Network(nodes=nodes_gdf)
    graph = network.to_graph(directed=True)

    assert graph.vcount() == 5
    assert graph.ecount() == 0


def test_save_and_load_network_zip(edges_gdf, nodes_gdf, temp_dir):
    """Test saving and loading network from zip"""
    network = Network(edges=edges_gdf, nodes=nodes_gdf)

    # Save network
    zip_path = network.save_network_zip(temp_dir, "test_network")
    assert zip_path.exists()

    # Load network
    loaded_network = Network.load_network_zip(temp_dir, "test_network")

    assert len(loaded_network.nodes) == 5
    assert len(loaded_network.edges) == 4
    pd.testing.assert_frame_equal(
        loaded_network.nodes.reset_index(drop=True),
        network.nodes.reset_index(drop=True),
    )


def test_load_network_zip_nonexistent(temp_dir):
    """Test loading from nonexistent zip file"""
    network = Network.load_network_zip(temp_dir, "nonexistent")

    assert network.nodes.empty
    assert network.edges.empty


def test_save_network_zip_empty(temp_dir):
    """Test saving empty network"""
    network = Network()
    zip_path = network.save_network_zip(temp_dir, "empty_network")

    assert zip_path.exists()


def test_initialize_funcstates(edges_gdf, nodes_gdf):
    """Test initialization of functional states"""
    network = Network(edges=edges_gdf, nodes=nodes_gdf)
    network.initialize_funcstates()

    # Check edges functional states
    assert "func_internal" in network.edges.columns
    assert "func_tot" in network.edges.columns
    assert "imp_dir" in network.edges.columns
    assert (network.edges["func_internal"] == 1).all()
    assert (network.edges["func_tot"] == 1).all()
    assert (network.edges["imp_dir"] == 0).all()

    # Check nodes functional states
    assert "func_internal" in network.nodes.columns
    assert "func_tot" in network.nodes.columns
    assert "imp_dir" in network.nodes.columns
    assert (network.nodes["func_internal"] == 1).all()
    assert (network.nodes["func_tot"] == 1).all()
    assert (network.nodes["imp_dir"] == 0).all()


def test_initialize_capacity(network_with_ci_types):
    """Test initialization of capacity"""
    dep_table = pd.DataFrame({"source": ["road"], "target": ["healthcare"]})
    network_with_ci_types.initialize_capacity(dep_table)

    capacity_col = "capacity_road_healthcare"
    assert capacity_col in network_with_ci_types.nodes.columns
    # Road nodes should have capacity 1
    assert (
        network_with_ci_types.nodes.loc[
            network_with_ci_types.nodes["ci_type"] == "road", capacity_col
        ]
        == 1
    ).all()
    # Healthcare nodes should have capacity -1
    assert (
        network_with_ci_types.nodes.loc[
            network_with_ci_types.nodes["ci_type"] == "healthcare", capacity_col
        ]
        == -1
    ).all()


def test_initialize_supply(network_with_ci_types):
    """Test initialization of supply"""

    dep_table = pd.DataFrame(
        {"source": ["healthcare"], "target": ["people"], "type_I": ["enduser"]}
    )

    network_with_ci_types.initialize_supply(dep_table)

    access_col = "access_state_healthcare_people"
    supply_col = "actual_supply_healthcare_people"

    assert access_col in network_with_ci_types.nodes.columns
    assert supply_col in network_with_ci_types.nodes.columns

    # People nodes should have supply 0 and access state "no base access"
    people_nodes = network_with_ci_types.nodes[
        network_with_ci_types.nodes["ci_type"] == "people"
    ]
    assert (people_nodes[supply_col] == 0).all()
    assert (people_nodes[access_col] == "no base access").all()


def test_update_network_from_graphs(network_with_ci_types):
    """Test updating network from graph object"""
    network_update = cp.deepcopy(network_with_ci_types)
    graph = network_update.to_graph(directed=True)
    # modify graph by adding a new node and edge
    graph.add_vertex(name="new_node", id=5, orig_id=5, ci_type="healthcare", func_tot=1)
    graph.add_edge(2, 4)

    # Update network from graph
    network_update = Network.from_graphs(graph)

    assert len(network_update.nodes) == 6
    assert len(network_update.edges) == 5
    assert "from_id" in network_update.edges.columns
    assert "to_id" in network_update.edges.columns
    assert network_update.nodes.iloc[4]["ci_type"] == "healthcare"
