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
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import igraph as ig

from climada_petals.engine.networks.nw_base import Network


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def nodes_gdf():
    """Create simple test nodes"""
    return gpd.GeoDataFrame(
        {
            'id': [0, 1, 2],
            'orig_id': [0, 1, 2],
            'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]
        },
        geometry='geometry',
        crs='EPSG:4326'
    )


@pytest.fixture
def edges_gdf():
    """Create simple test edges"""
    return gpd.GeoDataFrame(
        {
            'from_id': [0, 1],
            'to_id': [1, 2],
            'id': [0, 1],
            'orig_id': [0, 1],
            'osm_id': [100, 101],
            'geometry': [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)])
            ]
        },
        geometry='geometry',
        crs='EPSG:4326'
    )


class TestNetwork:
    """Test cases for the Network class"""

    def test_network_init_empty(self):
        """Test Network initialization with empty dataframes"""
        network = Network()
        assert network.edges.empty
        assert network.nodes.empty
        assert 'from_id' in network.edges.columns
        assert 'id' in network.nodes.columns

    def test_network_init_with_data(self, edges_gdf, nodes_gdf):
        """Test Network initialization with data"""
        network = Network(edges=edges_gdf, nodes=nodes_gdf)
        assert len(network.nodes) == 3
        assert len(network.edges) == 2
        assert network.nodes.crs.to_string() == 'EPSG:4326'
        assert network.edges.crs.to_string() == 'EPSG:4326'

    def test_network_init_adds_missing_columns(self):
        """Test that Network init adds missing required columns"""
        # Create dataframes without orig_id and id columns
        edges = gpd.GeoDataFrame(
            {
                'from_id': [0, 1],
                'to_id': [1, 2],
                'geometry': [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)])
                ]
            },
            geometry='geometry',
            crs='EPSG:4326'
        )
        nodes = gpd.GeoDataFrame(
            {
                'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]
            },
            geometry='geometry',
            crs='EPSG:4326'
        )

        network = Network(edges=edges, nodes=nodes)
        assert 'orig_id' in network.edges.columns
        assert 'id' in network.edges.columns
        assert 'orig_id' in network.nodes.columns
        assert 'id' in network.nodes.columns

    def test_reproject(self, edges_gdf, nodes_gdf):
        """Test reprojection of network"""
        network = Network(edges=edges_gdf, nodes=nodes_gdf)
        original_crs = network.nodes.crs.to_string()

        network.reproject('EPSG:3857')

        assert network.nodes.crs.to_string() == 'EPSG:3857'
        assert network.edges.crs.to_string() == 'EPSG:3857'
        assert original_crs != network.nodes.crs.to_string()

    def test_from_nws_single_network(self, edges_gdf, nodes_gdf):
        """Test combining a single network"""
        network = Network(edges=edges_gdf, nodes=nodes_gdf)
        combined = Network.from_nws([network])

        assert len(combined.nodes) == 3
        assert len(combined.edges) == 2

    def test_from_nws_multiple_networks(self, edges_gdf, nodes_gdf):
        """Test combining multiple networks"""
        network1 = Network(edges=edges_gdf, nodes=nodes_gdf)
        network2 = Network(edges=edges_gdf, nodes=nodes_gdf)

        combined = Network.from_nws([network1, network2])

        assert len(combined.nodes) == 6  # 3 + 3
        assert len(combined.edges) == 4  # 2 + 2
        # Check that node IDs are properly offset
        assert combined.nodes['id'].max() == 5

    def test_to_graph_undirected(self, edges_gdf, nodes_gdf):
        """Test conversion to undirected igraph"""
        network = Network(edges=edges_gdf, nodes=nodes_gdf)
        graph = network.to_graph(directed=False)

        assert isinstance(graph, ig.Graph)
        assert graph.vcount() == 3
        assert graph.ecount() == 2
        assert not graph.is_directed()

    def test_to_graph_directed(self, edges_gdf, nodes_gdf):
        """Test conversion to directed igraph"""
        network = Network(edges=edges_gdf, nodes=nodes_gdf)
        graph = network.to_graph(directed=True)

        assert isinstance(graph, ig.Graph)
        assert graph.vcount() == 3
        assert graph.ecount() == 2
        assert graph.is_directed()

    def test_to_graph_nodes_only(self, nodes_gdf):
        """Test conversion to graph with no edges"""
        network = Network(nodes=nodes_gdf)
        graph = network.to_graph(directed=False)

        assert graph.vcount() == 3
        assert graph.ecount() == 0

    def test_save_and_load_network_zip(self, edges_gdf, nodes_gdf, temp_dir):
        """Test saving and loading network from zip"""
        network = Network(edges=edges_gdf, nodes=nodes_gdf)

        # Save network
        zip_path = network.save_network_zip(temp_dir, 'test_network')
        assert zip_path.exists()

        # Load network
        loaded_network = Network.load_network_zip(temp_dir, 'test_network')

        assert len(loaded_network.nodes) == 3
        assert len(loaded_network.edges) == 2
        pd.testing.assert_frame_equal(
            loaded_network.nodes.reset_index(drop=True),
            network.nodes.reset_index(drop=True)
        )

    def test_load_network_zip_nonexistent(self, temp_dir):
        """Test loading from nonexistent zip file"""
        network = Network.load_network_zip(temp_dir, 'nonexistent')

        assert network.nodes.empty
        assert network.edges.empty

    def test_save_network_zip_empty(self, temp_dir):
        """Test saving empty network"""
        network = Network()
        zip_path = network.save_network_zip(temp_dir, 'empty_network')

        assert zip_path.exists()

    def test_initialize_funcstates(self, edges_gdf, nodes_gdf):
        """Test initialization of functional states"""
        network = Network(edges=edges_gdf, nodes=nodes_gdf)
        network.initialize_funcstates()

        # Check edges functional states
        assert 'func_internal' in network.edges.columns
        assert 'func_tot' in network.edges.columns
        assert 'imp_dir' in network.edges.columns
        assert (network.edges['func_internal'] == 1).all()
        assert (network.edges['func_tot'] == 1).all()
        assert (network.edges['imp_dir'] == 0).all()

        # Check nodes functional states
        assert 'func_internal' in network.nodes.columns
        assert 'func_tot' in network.nodes.columns
        assert 'imp_dir' in network.nodes.columns
        assert (network.nodes['func_internal'] == 1).all()
        assert (network.nodes['func_tot'] == 1).all()
        assert (network.nodes['imp_dir'] == 0).all()

    def test_initialize_capacity(self, edges_gdf, nodes_gdf):
        """Test initialization of capacity"""
        nodes = nodes_gdf.copy()
        nodes['ci_type'] = ['power', 'power', 'demand']
        network = Network(edges=edges_gdf, nodes=nodes)

        network.initialize_capacity('power', 'demand')

        capacity_col = 'capacity_power_demand'
        assert capacity_col in network.nodes.columns
        # Power nodes should have capacity 1
        assert (network.nodes.loc[network.nodes['ci_type'] == 'power', capacity_col] == 1).all()
        # Demand nodes should have capacity -1
        assert (network.nodes.loc[network.nodes['ci_type'] == 'demand', capacity_col] == -1).all()

    def test_initialize_supply(self, edges_gdf, nodes_gdf):
        """Test initialization of supply"""
        nodes = nodes_gdf.copy()
        nodes['ci_type'] = ['power', 'power', 'people']
        network = Network(edges=edges_gdf, nodes=nodes)

        network.initialize_supply('power')

        access_col = 'access_state_power_people'
        supply_col = 'actual_supply_power_people'

        assert access_col in network.nodes.columns
        assert supply_col in network.nodes.columns

        # People nodes should have supply 1 and access state "no base access"
        people_nodes = network.nodes[network.nodes['ci_type'] == 'people']
        assert (people_nodes[supply_col] == 1).all()
        assert (people_nodes[access_col] == 'no base access').all()

    def test_update_network_from_graphs(self, edges_gdf, nodes_gdf):
        """Test updating network from graph object"""
        network = Network(edges=edges_gdf, nodes=nodes_gdf)
        graph = network.to_graph(directed=False)

        # Update network from graph
        network.update_network_from_graphs(graph)

        assert len(network.nodes) == 3
        assert len(network.edges) == 2
        assert 'from_id' in network.edges.columns
        assert 'to_id' in network.edges.columns


