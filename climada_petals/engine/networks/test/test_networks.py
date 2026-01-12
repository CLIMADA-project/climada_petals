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
import copy as cp
from climada_petals.engine.networks.nw_base import Network
from climada.util.constants import ONE_LAT_KM

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
            'id': [0, 1, 2, 3],
            'orig_id': [0, 1, 2, 3],
            'geometry': [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)]
        },
        geometry='geometry',
        crs='EPSG:4326'
    )


@pytest.fixture
def edges_gdf():
    """Create simple test edges"""
    return gpd.GeoDataFrame(
        {
            'from_id': [1, 2],
            'to_id': [2, 3],
            'id': [0, 1],
            'orig_id': [0, 1],
            'osm_id': [100, 101],
            'distance': [157200, 157200],  # approx distances in meters
            'geometry': [
                LineString([(1, 1), (2, 2)]),
                LineString([(2, 2), (3, 3)])
            ]
        },
        geometry='geometry',
        crs='EPSG:4326'
    )

@pytest.fixture
def network_with_ci_types(edges_gdf, nodes_gdf):
    """Create a network with CI type information"""
    nodes = nodes_gdf.copy()
    nodes['ci_type'] = ['people', 'road', 'road','healthcare']
    nodes['func_tot'] = 1
    nodes.loc[1,'func_tot'] = 0  # partial functionality for testing
    edges = edges_gdf.copy()
    edges['ci_type'] = 'road'
    edges['func_tot'] = 1
    return Network(edges=edges, nodes=nodes)

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
        assert len(network.nodes) == 4
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

        assert len(combined.nodes) == 4
        assert len(combined.edges) == 2

    def test_from_nws_multiple_networks(self, edges_gdf, nodes_gdf):
        """Test combining multiple networks"""
        network1 = Network(edges=edges_gdf, nodes=nodes_gdf)
        network2 = Network(edges=edges_gdf, nodes=nodes_gdf)

        combined = Network.from_nws([network1, network2])

        assert len(combined.nodes) == 8  # 4 + 4
        assert len(combined.edges) == 4  # 2 + 2
        # Check that node IDs are properly offset
        assert combined.nodes['id'].max() == 7

    def test_to_graph_undirected(self, edges_gdf, nodes_gdf):
        """Test conversion to undirected igraph"""
        network = Network(edges=edges_gdf, nodes=nodes_gdf)
        graph = network.to_graph(directed=False)

        assert isinstance(graph, ig.Graph)
        assert graph.vcount() == 4
        assert graph.ecount() == 2
        assert not graph.is_directed()

    def test_to_graph_directed(self, edges_gdf, nodes_gdf):
        """Test conversion to directed igraph"""
        network = Network(edges=edges_gdf, nodes=nodes_gdf)
        graph = network.to_graph(directed=True)

        assert isinstance(graph, ig.Graph)
        assert graph.vcount() == 4
        assert graph.ecount() == 2
        assert graph.is_directed()

    def test_to_graph_nodes_only(self, nodes_gdf):
        """Test conversion to graph with no edges"""
        network = Network(nodes=nodes_gdf)
        graph = network.to_graph(directed=False)

        assert graph.vcount() == 4
        assert graph.ecount() == 0

    def test_save_and_load_network_zip(self, edges_gdf, nodes_gdf, temp_dir):
        """Test saving and loading network from zip"""
        network = Network(edges=edges_gdf, nodes=nodes_gdf)

        # Save network
        zip_path = network.save_network_zip(temp_dir, 'test_network')
        assert zip_path.exists()

        # Load network
        loaded_network = Network.load_network_zip(temp_dir, 'test_network')

        assert len(loaded_network.nodes) == 4
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

    def test_initialize_capacity(self, network_with_ci_types):
        """Test initialization of capacity"""


        network_with_ci_types.initialize_capacity('road', 'healthcare')

        capacity_col = 'capacity_road_healthcare'
        assert capacity_col in network_with_ci_types.nodes.columns
        # Road nodes should have capacity 1
        assert (network_with_ci_types.nodes.loc[network_with_ci_types.nodes['ci_type'] == 'road', capacity_col] == 1).all()
        # Healthcare nodes should have capacity -1
        assert (network_with_ci_types.nodes.loc[network_with_ci_types.nodes['ci_type'] == 'healthcare', capacity_col] == -1).all()

    def test_initialize_supply(self, edges_gdf, nodes_gdf):
        """Test initialization of supply"""
        nodes = nodes_gdf.copy()
        nodes['ci_type'] = ['road', 'road', 'road', 'people']
        network = Network(edges=edges_gdf, nodes=nodes)

        network.initialize_supply('road')

        access_col = 'access_state_road_people'
        supply_col = 'actual_supply_road_people'

        assert access_col in network.nodes.columns
        assert supply_col in network.nodes.columns

        # People nodes should have supply 1 and access state "no base access"
        people_nodes = network.nodes[network.nodes['ci_type'] == 'people']
        assert (people_nodes[supply_col] == 1).all()
        assert (people_nodes[access_col] == 'no base access').all()

    def test_update_network_from_graphs(self, network_with_ci_types):
        """Test updating network from graph object"""
        network_update = cp.deepcopy(network_with_ci_types)
        graph = network_update.to_graph(directed=False)
        #modify graph by adding a new node and edge
        graph.add_vertex(name='new_node', id=5, orig_id=5, ci_type='healthcare', func_tot=1)
        graph.add_edge(2, 4)

        # Update network from graph
        network_update.update_network_from_graphs(graph)

        assert len(network_update.nodes) == 5
        assert len(network_update.edges) == 3
        assert 'from_id' in network_update.edges.columns
        assert 'to_id' in network_update.edges.columns
        assert network_update.nodes.iloc[4]['ci_type'] == 'healthcare'

# ========================================================================
# Tests for GraphCalcs class
# ========================================================================

from climada_petals.engine.networks.nw_calcs import GraphCalcs
import numpy as np

@pytest.fixture
def graph_calcs(network_with_ci_types):
    """Create GraphCalcs instance with test network"""
    nw_calcs_mock = type('obj', (object,), {'network': network_with_ci_types})()
    return GraphCalcs(parent=nw_calcs_mock, directed=False)


class TestGraphCalcs:
    """Test cases for the GraphCalcs class"""

    def test_graph_calcs_init(self, network_with_ci_types):
        """Test GraphCalcs initialization"""
        nw_calcs_mock = type('obj', (object,), {'network': network_with_ci_types})()
        gc = GraphCalcs(parent=nw_calcs_mock, directed=False)

        assert gc.parent == nw_calcs_mock
        assert gc.directed is False
        assert gc._graph is None

    def test_graph_calcs_build_graph(self, graph_calcs):
        """Test building graph from network"""
        graph = graph_calcs.build_graph()

        assert isinstance(graph, ig.Graph)
        assert graph.vcount() == 5
        assert graph.ecount() == 2

    def test_graph_calcs_graph_property_lazy_load(self, graph_calcs):
        """Test that graph property lazy loads the graph"""
        assert graph_calcs._graph is None

        graph = graph_calcs.graph

        assert graph is not None
        assert isinstance(graph, ig.Graph)

    def test_graph_calcs_invalidate(self, graph_calcs):
        """Test invalidating cached graph"""
        _ = graph_calcs.graph  # Load graph
        assert graph_calcs._graph is not None

        graph_calcs.invalidate()

        assert graph_calcs._graph is None

    def test_filter_vertices_single_attr(self, graph_calcs):
        """Test filtering vertices by single attribute"""
        graph_calcs.build_graph()

        df_vs = GraphCalcs._filter_vertices(graph_calcs.graph, {'ci_type': 'road'})

        assert len(df_vs) == 3
        assert all(df_vs['ci_type'] == 'road')

    def test_filter_vertices_multiple_attrs(self, graph_calcs):
        """Test filtering vertices by multiple attributes"""
        graph_calcs.build_graph()

        df_vs = GraphCalcs._filter_vertices(
            graph_calcs.graph,
            {'ci_type': 'road', 'func_tot': 1}
        )

        assert len(df_vs) == 2
        assert all(df_vs['ci_type'] == 'road')
        assert all(df_vs['func_tot'] == 1)

    def test_filter_edges_by_ci_type(self, graph_calcs):
        """Test filtering edges by CI type"""
        graph_calcs.build_graph()

        df_es_match = GraphCalcs._filter_edges(graph_calcs.graph, {'ci_type': 'road'})
        df_es_not = GraphCalcs._filter_edges(graph_calcs.graph, {'ci_type': 'river'})

        assert len(df_es_match) == 2
        assert len(df_es_not) == 0

    def test_get_subgraph2graph_vsdict(self, graph_calcs):
        """Test vertex mapping from subgraph to graph"""
        graph_calcs.build_graph()
        graph = graph_calcs.graph

        # Create a subgraph with all vertices
        subgraph = graph.induced_subgraph(range(graph.vcount()))
        subgraph.vs['orig_id'] = [1, 0, 3, 2]

        mapping = GraphCalcs._get_subgraph2graph_vsdict(graph, subgraph)

        assert isinstance(mapping, dict)
        assert mapping == {0: 1, 1: 0, 2: 3, 3: 2}

    def test_get_subgraph2graph_esdict(self, graph_calcs):
        """Test edge mapping from subgraph to graph"""
        graph_calcs.build_graph()
        graph = graph_calcs.graph

        # Create a subgraph with all vertices
        subgraph = graph.induced_subgraph(range(graph.vcount()))
        subgraph.es['orig_id'] = [1, 0]

        mapping = GraphCalcs._get_subgraph2graph_esdict(graph, subgraph)

        assert isinstance(mapping, dict)
        assert mapping == {0: 1, 1: 0}

    def test_select_closest_k_basic(self):
        """Test selecting k nearest neighbors"""
        # Create source and target node GeoDataFrames
        gdf_vs_target = gpd.GeoDataFrame(
            {
                'id': [0, 1],
                'geometry': [Point(0, 0), Point(3, 3)]
            },
            geometry='geometry',
            crs='EPSG:4326'
        )
        gdf_vs_source = gpd.GeoDataFrame(
            {
                'id': [2, 3],
                'geometry': [Point(1, 1), Point(2, 2)]
            },
            geometry='geometry',
            crs='EPSG:4326'
        )

        v_ids_source, v_ids_target = GraphCalcs._select_closest_k(
            gdf_vs_source, gdf_vs_target, dist_thresh=np.inf, bidir=False, k=1
        )

        assert len(v_ids_source) > 0
        assert len(v_ids_source) == len(v_ids_target)
        np.testing.assert_array_equal(v_ids_target, [0, 1])
        np.testing.assert_array_equal(v_ids_source, [2, 3])

    def test_select_closest_k_dist(self):
        """Test selecting k nearest neighbors with distance threshold"""
        # Create source and target node GeoDataFrames
        gdf_vs_target = gpd.GeoDataFrame(
            {
                'id': [0, 1],
                'geometry': [Point(0, 0), Point(6, 6)]
            },
            geometry='geometry',
            crs='EPSG:4326'
        )
        gdf_vs_source = gpd.GeoDataFrame(
            {
                'id': [2, 3],
                'geometry': [Point(1, 1), Point(2, 2)]
            },
            geometry='geometry',
            crs='EPSG:4326'
        )
        dist_th = 2 * (ONE_LAT_KM * 1000)
        v_ids_source, v_ids_target = GraphCalcs._select_closest_k(
            gdf_vs_source, gdf_vs_target, dist_thresh=dist_th, bidir=False, k=1
        )

        assert len(v_ids_source) > 0
        assert len(v_ids_source) == len(v_ids_target)
        np.testing.assert_array_equal(v_ids_target, [0])
        np.testing.assert_array_equal(v_ids_source, [2])

    def test_funcstates_sum(self, graph_calcs):
        """Test summing functional states"""
        graph_calcs.build_graph()

        v_sum, e_sum = graph_calcs._funcstates_sum()

        assert isinstance(v_sum, (int, float))
        assert isinstance(e_sum, (int, float))
        assert v_sum > 0
        assert e_sum > 0

    def test_create_subgraph(self, graph_calcs):
        """Test creating subgraph with filtered vertices"""
        graph_calcs.build_graph()

        source_attrs = {'ci_type': 'road'}
        target_attrs = {'ci_type': 'people'}
        via_attrs = {'ci_type': 'road'}

        subgraph = graph_calcs._create_subgraph(source_attrs, target_attrs, via_attrs)

        assert isinstance(subgraph, ig.Graph)
        assert subgraph.vcount() == 4  # 3 road nodes + 1 people node
        assert subgraph.ecount() == 2  # 2 edges between road nodes

    def test_link_vertices_edgecond(self, graph_calcs):
        """Test linking vertices based on edge conditions"""
        graph_calcs.build_graph()
        initial_edge_count = graph_calcs.graph.ecount()

        # This should add edges based on condition
        graph_calcs.link_vertices_edgecond(
            target_attrs={'ci_type': 'healthcare'},
            edge_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dependency_road_healthcare'}
        )

        # Verify that method completes without error
        assert graph_calcs.graph.ecount() > initial_edge_count
        assert 'dependency_road_healthcare' in graph_calcs.graph.es["ci_type"]



# ========================================================================
# Tests for NetworkCalcs class
# ========================================================================

from climada_petals.engine.networks.nw_calcs import NetworkCalcs


@pytest.fixture
def dependency_table():
    """Create a simple dependency table"""
    return pd.DataFrame({
        'source': ['road', 'healthcare'],
        'target': ['people', 'people'],
        'type_I': ['physical', 'physical'],
        'via_link': ['road', 'road'],
        'link_condition': ['distance', 'distance'],
        'thresh_dist': [np.inf, np.inf],
        'bidir_link': [False, False],
        'access_cnstr': [False, True],
        'n_links': [1, 1]
    })


@pytest.fixture
def network_calcs(network_with_ci_types, dependency_table):
    """Create NetworkCalcs instance"""
    return NetworkCalcs(network=network_with_ci_types, dep_table=dependency_table)


class TestNetworkCalcs:
    """Test cases for the NetworkCalcs class"""

    def test_network_calcs_init(self, network_with_ci_types, dependency_table):
        """Test NetworkCalcs initialization"""
        nc = NetworkCalcs(network=network_with_ci_types, dep_table=dependency_table)

        assert nc.network == network_with_ci_types
        assert nc.dep_table is dependency_table
        assert isinstance(nc.graph_calc, GraphCalcs)

    def test_initialize_base_state(self, network_calcs):
        """Test initialization of base functional state"""
        network_calcs.initialize_base_state()

        assert 'func_internal' in network_calcs.network.nodes.columns
        assert 'func_tot' in network_calcs.network.nodes.columns
        assert 'func_internal' in network_calcs.network.edges.columns
        assert 'func_tot' in network_calcs.network.edges.columns

    def test_merge_clusters(self, network_calcs):
        """Test merging clusters when network is already connected"""

        network_calcs.graph_calc.build_graph()
        init_node_count = len(network_calcs.network.nodes)
        init_edge_count = len(network_calcs.network.edges)

        # Try to merge clusters
        network_calcs.merge_clusters(ci_type='road', max_iter=1, dist_thresh=np.inf)

        # Verify network structure is maintained
        assert len(network_calcs.network.nodes) == init_node_count
        assert len(network_calcs.network.edges) >= init_edge_count #! double edges may be added
        assert len(network_calcs.graph_calc.graph.connected_components()) == 1

    def test_add_physical_links(self, network_calcs):
        """Test adding physical links to network"""
        initial_edge_count = len(network_calcs.network.edges)

        network_calcs.add_physical_links()

        # check that one edge between people and road has been added
        assert len(network_calcs.network.edges) >= initial_edge_count + 1
        assert len(network_calcs.network.nodes) == len(network_calcs.network.nodes)
        assert network_calcs.network.edges.iloc[initial_edge_count]['ci_type'] == 'road'

    def test_setup_dependencies(self, network_calcs):
        """Test setting up dependencies"""

        network_calcs.setup_dependencies()

        # Verify that graph has been updated
        assert "dependency_road_people" in network_calcs.graph_calc.graph.es["ci_type"]
        assert "dependency_healthcare_people" in network_calcs.graph_calc.graph.es["ci_type"]

    def test_network_calcs_graph_property(self, network_calcs):
        """Test graph property of NetworkCalcs"""
        graph = network_calcs.graph

        assert isinstance(graph, ig.Graph)




