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
            'id': [0, 1, 2, 3, 4],
            'orig_id': [0, 1, 2, 3, 4],
            'geometry': [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4)]
        },
        geometry='geometry',
        crs='EPSG:4326'
    )


@pytest.fixture
def edges_gdf():
    """Create simple test edges"""
    return gpd.GeoDataFrame(
        {
            'from_id': [0, 1, 2, 3],
            'to_id': [1, 2, 3, 4],
            'id': [0, 1, 2, 3],
            'orig_id': [0, 1, 2, 3],
            'osm_id': [100, 101, 102, 103],
            'distance': [157200, 157200, 157200, 157200],  # approx distances in meters
            'geometry': [
                LineString([(0, 0), (1, 1)]), #roads need to go from ci to user
                LineString([(1, 1), (2, 2)]),
                LineString([(2, 2), (3, 3)]),
                LineString([(3, 3), (4, 4)])

            ]
        },
        geometry='geometry',
        crs='EPSG:4326'
    )

@pytest.fixture
def network_with_ci_types(edges_gdf, nodes_gdf):
    """Create a network with CI type information"""
    nodes = cp.deepcopy(nodes_gdf)
    nodes['ci_type'] = ['people', 'road', 'road', 'road', 'healthcare']
    edges = cp.deepcopy(edges_gdf)
    edges['ci_type'] = 'road'
    nodes['func_tot'] = 1
    edges['func_tot'] = 1
    return Network(edges=edges, nodes=nodes)

@pytest.fixture
def network_with_remote_node_missing_edge(network_with_ci_types):
    """Create a network with CI type information"""
    network = cp.deepcopy(network_with_ci_types)
    #add far away hospital node
    new_node = gpd.GeoDataFrame(
        {
            'id': [5],
            'orig_id': [5],
            'ci_type': ['healthcare'],
            'func_tot': [1],
            'geometry': [Point(4, 50)]
        },
        geometry='geometry',
        crs='EPSG:4326'
    )
    network.nodes = pd.concat([network.nodes, new_node], ignore_index=True)
    return network

@pytest.fixture
def network_with_remote_node(network_with_remote_node_missing_edge):
    """Create a network with CI type information"""
    network = cp.deepcopy(network_with_remote_node_missing_edge)
    #add edge from last road node to far away hospital node
    new_edge = gpd.GeoDataFrame(
        {
            'from_id': [2],
            'to_id': [5],
            'id': [4],
            'orig_id': [4],
            'osm_id': [104],
            'distance': [7000000],  # approx distances in meters
            'ci_type': ['road'],
            'func_tot': [1],
            'geometry': [LineString([(2, 2), (4, 50)])]
        },
        geometry='geometry',
        crs='EPSG:4326'
    )
    network.edges = pd.concat([network.edges, new_edge], ignore_index=True)
    return network
#@pytest.fixture
#def network_with_missing_edge(network_with_ci_types):
#    """Create a network with CI type information"""
#    network = cp.deepcopy(network_with_ci_types)
#    network.edges = network.edges.drop(index=0).reset_index(drop=True)  # remove one edge for testing
#    return network

@pytest.fixture
def network_with_edge_fail(network_with_remote_node):
    """Create a network with CI type information"""
    network = cp.deepcopy(network_with_remote_node)
    network.edges.loc[3,'func_tot'] = 0  # ci fail for testing
    network.nodes.loc[2, 'func_tot'] = 0  # road node needs to fail too
    return network

@pytest.fixture
def network_with_source_fail(network_with_remote_node):
    """Create a network with a failed source CI"""
    network = cp.deepcopy(network_with_remote_node)
    network.nodes.loc[network.nodes['ci_type']=='healthcare','func_tot'] = 0  # ci fail for testing
    return network


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
        assert len(network.nodes) == 5
        assert len(network.edges) == 4
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

        assert len(combined.nodes) == 5
        assert len(combined.edges) == 4

    def test_from_nws_multiple_networks(self, edges_gdf, nodes_gdf):
        """Test combining multiple networks"""
        network1 = Network(edges=edges_gdf, nodes=nodes_gdf)
        network2 = Network(edges=edges_gdf, nodes=nodes_gdf)

        combined = Network.from_nws([network1, network2])

        assert len(combined.nodes) == 10  # 5 + 5
        assert len(combined.edges) == 8  # 4 + 4
        # Check that node IDs are properly offset
        assert combined.nodes['id'].max() == 9

    def test_to_graph_undirected(self, edges_gdf, nodes_gdf):
        """Test conversion to undirected igraph"""
        network = Network(edges=edges_gdf, nodes=nodes_gdf)
        graph = network.to_graph(directed=False)

        assert isinstance(graph, ig.Graph)
        assert graph.vcount() == 5
        assert graph.ecount() == 4
        assert not graph.is_directed()

    def test_to_graph_directed(self, edges_gdf, nodes_gdf):
        """Test conversion to directed igraph"""
        network = Network(edges=edges_gdf, nodes=nodes_gdf)
        graph = network.to_graph(directed=True)

        assert isinstance(graph, ig.Graph)
        assert graph.vcount() == 5
        assert graph.ecount() == 4
        assert graph.is_directed()

    def test_to_graph_nodes_only(self, nodes_gdf):
        """Test conversion to graph with no edges"""
        network = Network(nodes=nodes_gdf)
        graph = network.to_graph(directed=True)

        assert graph.vcount() == 5
        assert graph.ecount() == 0

    def test_save_and_load_network_zip(self, edges_gdf, nodes_gdf, temp_dir):
        """Test saving and loading network from zip"""
        network = Network(edges=edges_gdf, nodes=nodes_gdf)

        # Save network
        zip_path = network.save_network_zip(temp_dir, 'test_network')
        assert zip_path.exists()

        # Load network
        loaded_network = Network.load_network_zip(temp_dir, 'test_network')

        assert len(loaded_network.nodes) == 5
        assert len(loaded_network.edges) == 4
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
        dep_table = pd.DataFrame({
            'source': ['road'],
            'target': ['healthcare']
        })
        network_with_ci_types.initialize_capacity(dep_table)

        capacity_col = 'capacity_road_healthcare'
        assert capacity_col in network_with_ci_types.nodes.columns
        # Road nodes should have capacity 1
        assert (network_with_ci_types.nodes.loc[network_with_ci_types.nodes['ci_type'] == 'road', capacity_col] == 1).all()
        # Healthcare nodes should have capacity -1
        assert (network_with_ci_types.nodes.loc[network_with_ci_types.nodes['ci_type'] == 'healthcare', capacity_col] == -1).all()

    def test_initialize_supply(self, network_with_ci_types):
        """Test initialization of supply"""

        dep_table = pd.DataFrame({
            'source': ['healthcare'],
            'target': ['people'],
            'type_I': ['enduser']
        })

        network_with_ci_types.initialize_supply(dep_table)

        access_col = 'access_state_healthcare_people'
        supply_col = 'actual_supply_healthcare_people'

        assert access_col in network_with_ci_types.nodes.columns
        assert supply_col in network_with_ci_types.nodes.columns

        # People nodes should have supply 0 and access state "no base access"
        people_nodes = network_with_ci_types.nodes[network_with_ci_types.nodes['ci_type'] == 'people']
        assert (people_nodes[supply_col] == 0).all()
        assert (people_nodes[access_col] == 'no base access').all()

    def test_update_network_from_graphs(self, network_with_ci_types):
        """Test updating network from graph object"""
        network_update = cp.deepcopy(network_with_ci_types)
        graph = network_update.to_graph(directed=True)
        #modify graph by adding a new node and edge
        graph.add_vertex(name='new_node', id=5, orig_id=5, ci_type='healthcare', func_tot=1)
        graph.add_edge(2, 4)

        # Update network from graph
        network_update.update_network_from_graphs(graph)

        assert len(network_update.nodes) == 6
        assert len(network_update.edges) == 5
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
    return GraphCalcs(parent=nw_calcs_mock, directed=True)


@pytest.fixture
def graph_calcs_with_source_fail(network_with_source_fail):
    """Create GraphCalcs instance with test network containing CI failures"""
    nw_calcs_mock = type('obj', (object,), {'network': network_with_source_fail})()
    return GraphCalcs(parent=nw_calcs_mock, directed=True)

@pytest.fixture
def graph_calcs_with_edge_ci_fail(network_with_edge_fail):
    """Create GraphCalcs instance with test network containing edge CI failures"""
    nw_calcs_mock = type('obj', (object,), {'network': network_with_edge_fail})()
    return GraphCalcs(parent=nw_calcs_mock, directed=True)

@pytest.fixture
def graph_calcs_with_remote_node_missing_edge(network_with_remote_node_missing_edge):
    """Create GraphCalcs instance with test network containing missing edge"""
    nw_calcs_mock = type('obj', (object,), {'network': network_with_remote_node_missing_edge})()
    return GraphCalcs(parent=nw_calcs_mock, directed=True)

@pytest.fixture
def graph_calcs_with_remote_node(network_with_remote_node):
    """Create GraphCalcs instance with test network containing remote node"""
    nw_calcs_mock = type('obj', (object,), {'network': network_with_remote_node})()
    return GraphCalcs(parent=nw_calcs_mock, directed=True)

class TestGraphCalcs:
    """Test cases for the GraphCalcs class"""

    def test_graph_calcs_init(self, network_with_ci_types):
        """Test GraphCalcs initialization"""
        nw_calcs_mock = type('obj', (object,), {'network': network_with_ci_types})()
        gc = GraphCalcs(parent=nw_calcs_mock, directed=True)

        assert gc.parent == nw_calcs_mock
        assert gc.directed is True
        assert gc._graph is None

    def test_graph_calcs_build_graph(self, graph_calcs):
        """Test building graph from network"""
        graph = graph_calcs.build_graph()

        assert isinstance(graph, ig.Graph)
        assert graph.vcount() == 5
        assert graph.ecount() == 4

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

    def test_filter_vertices_single_attr(self, graph_calcs_with_remote_node):
        """Test filtering vertices by single attribute"""
        graph_calcs_with_remote_node.build_graph()

        df_vs = GraphCalcs._filter_vertices(graph_calcs_with_remote_node.graph, {'ci_type': 'healthcare'})

        assert len(df_vs) == 2
        assert all(df_vs['ci_type'] == 'healthcare')

    def test_filter_vertices_multiple_attrs(self, graph_calcs_with_remote_node):
        """Test filtering vertices by multiple attributes"""
        graph_calcs_with_remote_node.build_graph()

        df_vs = GraphCalcs._filter_vertices(
            graph_calcs_with_remote_node.graph,
            {'ci_type': 'healthcare', 'func_tot': 1}
        )

        assert len(df_vs) == 2
        assert all(df_vs['ci_type'] == 'healthcare')
        assert all(df_vs['func_tot'] == 1)

    def test_filter_edges_by_ci_type(self, graph_calcs):
        """Test filtering edges by CI type"""
        graph_calcs.build_graph()

        df_es_match = GraphCalcs._filter_edges(graph_calcs.graph, {'ci_type': 'road'})
        df_es_not = GraphCalcs._filter_edges(graph_calcs.graph, {'ci_type': 'river'})

        assert len(df_es_match) == 4
        assert len(df_es_not) == 0

    def test_get_subgraph2graph_vsdict(self, graph_calcs):
        """Test vertex mapping from subgraph to graph"""
        graph_calcs.build_graph()
        graph = graph_calcs.graph

        # Create a subgraph with all vertices
        subgraph = graph.induced_subgraph(range(graph.vcount()))
        subgraph.vs['orig_id'] = [1, 0, 3, 2, 4]

        mapping = GraphCalcs._get_subgraph2graph_vsdict(graph, subgraph)

        assert isinstance(mapping, dict)
        assert mapping == {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}

    def test_get_subgraph2graph_esdict(self, graph_calcs):
        """Test edge mapping from subgraph to graph"""
        graph_calcs.build_graph()
        graph = graph_calcs.graph

        # Create a subgraph with all vertices
        subgraph = graph.induced_subgraph(range(graph.vcount()))
        subgraph.es['orig_id'] = [1, 2, 0, 3]

        mapping = GraphCalcs._get_subgraph2graph_esdict(graph, subgraph)

        assert isinstance(mapping, dict)
        assert mapping == {0: 1, 1: 2, 2: 0, 3: 3}

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

    def test_create_subgraph_filter(self, graph_calcs_with_remote_node):
        """Test creating subgraph with filtered vertices"""
        graph_calcs_with_remote_node.build_graph()

        source_attrs = {'ci_type': 'healthcare'}
        target_attrs = {'ci_type': 'people'}
        via_attrs = {'ci_type': 'road'}

        subgraph = graph_calcs_with_remote_node._create_subgraph(source_attrs, target_attrs, via_attrs)

        assert isinstance(subgraph, ig.Graph)
        assert subgraph.vcount() == 6  # 3 func road node + 1 people node + 2 healthcare nodes
        assert subgraph.ecount() == 5  # 5 edge func between func road nodes and healthcare
        assert set(subgraph.vs['ci_type']).difference({'healthcare', 'road', 'people'}) == set()
        assert set(subgraph.es['ci_type']).difference({'road'}) == set()

    def test_create_subgraph_filter_source(self, graph_calcs_with_source_fail):
        """Test creating subgraph with filtered vertices"""
        graph_calcs_with_source_fail.build_graph()

        source_attrs = {'ci_type': 'healthcare', 'func_tot': 1}
        target_attrs = {'ci_type': 'people'}
        via_attrs = {'ci_type': 'road'}

        subgraph = graph_calcs_with_source_fail._create_subgraph(source_attrs, target_attrs, via_attrs)

        assert isinstance(subgraph, ig.Graph)
        assert subgraph.vcount() == 4  # 3 func road node + 1 people node
        assert subgraph.ecount() == 3  # 3 edge func between func road
        #assert set(subgraph.vs.select(ci_type='healthcare')['func_tot']) == set()
        assert set(subgraph.vs['ci_type']).difference({'road', 'people'}) == set()
        assert set(subgraph.es['ci_type']).difference({'road'}) == set()

    def test_create_subgraph_filter_via(self, graph_calcs_with_edge_ci_fail):
        """Test creating subgraph with filtered vertices"""
        graph_calcs_with_edge_ci_fail.build_graph()

        source_attrs = {'ci_type': 'road'}
        target_attrs = {'ci_type': 'healthcare'}
        via_attrs = {'ci_type': 'road', 'func_tot': 1}

        subgraph = graph_calcs_with_edge_ci_fail._create_subgraph(source_attrs, target_attrs, via_attrs)

        assert isinstance(subgraph, ig.Graph)
        assert subgraph.vcount() == 5  # 3 func road node + 2 healthcare nodes
        assert subgraph.ecount() == 3  # 3 edge func between func road nodes and healthcare
        assert set(subgraph.es.select(ci_type='road')['func_tot']) == {1}
        assert set(subgraph.vs['ci_type']).difference({'healthcare', 'road'}) == set()
        assert set(subgraph.es['ci_type']).difference({'road'}) == set()

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
        dep_edge = graph_calcs.graph.es.select(ci_type='dependency_road_healthcare')
        sources = [e.source for e in dep_edge]
        targets = [e.target for e in dep_edge]
        # Verify that method completes without error
        assert len(dep_edge) > 0
        assert all((src['ci_type'] == 'road' for src in graph_calcs.graph.vs[sources]))
        assert all((tgt['ci_type'] == 'healthcare' for tgt in graph_calcs.graph.vs[targets]))

    def test_link_clusters_no_clusters(self, graph_calcs):
        """Test link_clusters when network is already connected"""
        graph_calcs.build_graph()
        initial_edge_count = graph_calcs.graph.ecount()

        graph_calcs.link_clusters(
            dist_thresh=np.inf,
            link_attrs={'ci_type': 'cluster_link'}
        )

        # Should not add edges if already connected

        assert 'cluster_link' not in graph_calcs.graph.es['ci_type']

    def test_link_clusters_with_threshold_low(self, graph_calcs_with_remote_node):
        """Test link_clusters with distance threshold"""
        graph_calcs_with_remote_node.build_graph()
        initial_edge_count = graph_calcs_with_remote_node.graph.ecount()
        graph_calcs_with_remote_node.link_clusters(
            dist_thresh=1000,
            link_attrs={'ci_type': 'cluster_link'}
        )

        # Verify method completes
        assert graph_calcs_with_remote_node.graph.ecount() == initial_edge_count
        assert 'cluster_link' not in graph_calcs_with_remote_node.graph.es['ci_type']

    def test_link_clusters_with_threshold_high(self, graph_calcs_with_remote_node_missing_edge):
        """Test link_clusters with distance threshold"""
        graph_calcs_with_remote_node_missing_edge.build_graph()
        initial_edge_count = graph_calcs_with_remote_node_missing_edge.graph.ecount()
        graph_calcs_with_remote_node_missing_edge.link_clusters(
            dist_thresh=np.inf,
            link_attrs={'ci_type': 'cluster_link'},
        )

        # Verify method completes
        assert graph_calcs_with_remote_node_missing_edge.graph.ecount() == initial_edge_count + 1
        assert 'cluster_link' in graph_calcs_with_remote_node_missing_edge.graph.es['ci_type']

    def test_link_vertices_closest_k_low_thresh(self, graph_calcs):
        """Test linking vertices by k-nearest neighbors"""
        graph_calcs.build_graph()
        initial_edge_count = graph_calcs.graph.ecount()

        graph_calcs.link_vertices_closest_k(
            source_attrs={'ci_type': 'road'},
            target_attrs={'ci_type': 'people'},
            link_attrs={'ci_type': 'link_road_people'},
            dist_thresh=1000,
            bidir=False,
            k=1
        )

        # Should add at least one edge
        assert graph_calcs.graph.ecount() == initial_edge_count
        assert 'link_road_people' not in graph_calcs.graph.es['ci_type']

    def test_link_vertices_closest_k_high_thresh(self, graph_calcs):
        """Test linking vertices by k-nearest neighbors"""
        graph_calcs.build_graph()
        initial_edge_count = graph_calcs.graph.ecount()

        graph_calcs.link_vertices_closest_k(
            source_attrs={'ci_type': 'road'},
            target_attrs={'ci_type': 'people'},
            link_attrs={'ci_type': 'link_road_people'},
            dist_thresh=np.inf,
            bidir=False,
            k=1
        )

        # Should add at least one edge
        assert graph_calcs.graph.ecount() == initial_edge_count + 1
        assert 'link_road_people' in graph_calcs.graph.es['ci_type']

    def test_link_vertices_closest_k_bidir(self, graph_calcs):
        """Test linking vertices by k-nearest neighbors"""
        graph_calcs.build_graph()
        initial_edge_count = graph_calcs.graph.ecount()

        graph_calcs.link_vertices_closest_k(
            source_attrs={'ci_type': 'road'},
            target_attrs={'ci_type': 'people'},
            link_attrs={'ci_type': 'link_road_people'},
            dist_thresh=np.inf,
            bidir=True,
            k=1
        )

        # Should add at least one edge
        assert graph_calcs.graph.ecount() == initial_edge_count + 2
        assert 'link_road_people' in graph_calcs.graph.es['ci_type']

    def test_link_vertices_shortest_paths_single(self, graph_calcs):
        """Test linking via shortest paths with k=1"""
        graph_calcs.build_graph()
        initial_edge_count = graph_calcs.graph.ecount()

        graph_calcs.link_vertices_shortest_paths(
            source_attrs={'ci_type': 'road'},
            target_attrs={'ci_type': 'people'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'shortest_path_link'},
            dist_thresh=10e6,
            criterion='distance',
            k=1,
            bidir=False
        )

        # Should add edges based on shortest paths
        assert graph_calcs.graph.ecount() == initial_edge_count + 1
        assert 'shortest_path_link' in graph_calcs.graph.es['ci_type']


    def test_link_vertices_shortest_paths_multiple(self, graph_calcs):
        """Test linking via shortest paths with k>1"""
        graph_calcs.build_graph()
        initial_edge_count = graph_calcs.graph.ecount()

        graph_calcs.link_vertices_shortest_paths(
            source_attrs={'ci_type': 'road'},
            target_attrs={'ci_type': 'healthcare'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'multi_path_link'},
            dist_thresh=10e6,
            criterion='distance',
            k=2,
            bidir=False
        )

        assert graph_calcs.graph.ecount() > initial_edge_count

    def test_edges_from_vlists(self, graph_calcs_with_remote_node):
        """Test adding edges from vertex lists"""
        graph_calcs_with_remote_node.build_graph()
        initial_edge_count = graph_calcs_with_remote_node.graph.ecount()

        # Add edges between specific vertices
        v_ids_source = [1, 2, 4]
        v_ids_target = [3, 4, 1]

        graph_calcs_with_remote_node._edges_from_vlists(
            v_ids_source,
            v_ids_target,
            link_attrs={'ci_type': 'test_link'}
        )
        sources = [e.source for e in graph_calcs_with_remote_node.graph.es if e['ci_type'] == 'test_link']
        targets = [e.target for e in graph_calcs_with_remote_node.graph.es if e['ci_type'] == 'test_link']
        # Should have added 3 edges
        assert graph_calcs_with_remote_node.graph.ecount() == initial_edge_count + 3
        assert 'test_link' in graph_calcs_with_remote_node.graph.es['ci_type']
        assert len(sources) == 3
        assert len(targets) == 3
        # Verify correct source-target pairs
        for i in range(len(sources)):
            assert sources[i] == v_ids_source[i]
            assert targets[i] == v_ids_target[i]

    def test_edges_from_vlists_with_distance(self, graph_calcs):
        """Test adding edges with pre-calculated distances"""
        graph_calcs.build_graph()

        v_ids_source = [1]
        v_ids_target = [2]

        graph_calcs._edges_from_vlists(
            v_ids_source,
            v_ids_target,
            link_attrs={'ci_type': 'test_link', 'distance': [1000.0]}
        )

        # Check that distance was preserved
        new_edge = graph_calcs.graph.es[graph_calcs.graph.ecount() - 1]
        assert new_edge['distance'] == 1000.0

    def test_calc_dependencies_distance(self, graph_calcs):
        """Test calculating dependencies with distance criterion"""
        graph_calcs.build_graph()
        initial_edge_count = graph_calcs.graph.ecount()

        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': 'healthcare'},
            target_attrs={'ci_type': 'people'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dep_link'},
            link_condition='distance',
            dist_thresh=1E7,
            bidir_link=False,
        )

        assert 'dep_link' in graph_calcs.graph.es['ci_type']
        assert graph_calcs.graph.ecount() == initial_edge_count + 1

    def test_calc_dependencies_edgecond(self, graph_calcs):
        """Test calculating dependencies with edge condition"""
        graph_calcs.build_graph()
        initial_edge_count = graph_calcs.graph.ecount()

        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': 'road'},
            target_attrs={'ci_type': 'people'},
            via_attrs={},
            link_attrs={'ci_type': 'edge_cond_link'},
            link_condition='edgecond',
            dist_thresh=None,
            bidir_link=False,
        )

        # Should add edges based on edge conditions
        assert graph_calcs.graph.ecount() == initial_edge_count + 1
        assert 'edge_cond_link' in graph_calcs.graph.es['ci_type']

    def test_calc_dependencies_distance_via_fail(self, graph_calcs_with_edge_ci_fail):
        """Test calculating dependencies with distance criterion"""
        graph_calcs_with_edge_ci_fail.build_graph()
        initial_edge_count = graph_calcs_with_edge_ci_fail.graph.ecount()

        graph_calcs_with_edge_ci_fail._calc_dependencies(
            source_attrs={'ci_type': 'healthcare'},
            target_attrs={'ci_type': 'people'},
            via_attrs={'ci_type': 'road', 'func_tot': 1},
            link_attrs={'ci_type': 'dep_link'},
            link_condition='distance',
            dist_thresh=1E7,
            bidir_link=False,
        )

        assert 'dep_link' not in graph_calcs_with_edge_ci_fail.graph.es['ci_type']
        assert graph_calcs_with_edge_ci_fail.graph.ecount() == initial_edge_count

    def test_funcstates_sum_with_failures(self, graph_calcs):
        """Test summing functional states with some failures"""
        graph_calcs.build_graph()

        # Set some vertices to failed state
        graph_calcs.graph.vs[1]['func_tot'] = 0
        graph_calcs.graph.es[0]['func_tot'] = 0

        v_sum, e_sum = graph_calcs._funcstates_sum()

        # Should reflect the failures
        assert v_sum < graph_calcs.graph.vcount()
        assert e_sum < graph_calcs.graph.ecount()

    def test_check_access_basic_undisrupted(self, graph_calcs, dependency_table):
        """Test _check_access basic functionality"""
        graph_calcs.parent.network.initialize_funcstates()
        graph_calcs.parent.network.initialize_capacity(dependency_table)
        graph_calcs.parent.network.initialize_supply(dependency_table)
        graph_calcs.build_graph()

        # Get first dependency row (road -> people)
        for _, row in dependency_table.loc[dependency_table['target'] == 'people'].iterrows():
            graph_calcs._calc_dependencies(
                source_attrs={'ci_type': row.source},
                target_attrs={'ci_type': row.target},
                via_attrs={'ci_type': 'road'},
                link_attrs={'ci_type': f'dependency_{row.source}_{row.target}'},
                link_condition=row['link_condition'],
                dist_thresh=row['thresh_dist'],
                bidir_link=row['bidir_link'],
            )

            # Call _check_access
            graph_calcs._check_access(row, friction_surf=None, rerouting=False)

            # Verify access states are set on people nodes
            people_nodes = graph_calcs.graph.vs.select(ci_type='people')
            for node in people_nodes:
                assert f'access_state_{row.source}_{row.target}' in node.attributes()
                assert f'actual_supply_{row.source}_{row.target}' in node.attributes()
                assert node[f'access_state_{row.source}_{row.target}'] == 'access undisrupted'

    def test_check_access_with_rerouting_undisrupted(self, graph_calcs, dependency_table):
        """Test _check_access with rerouting enabled"""
        graph_calcs.parent.network.initialize_funcstates()
        graph_calcs.parent.network.initialize_capacity(dependency_table)
        graph_calcs.parent.network.initialize_supply(dependency_table)
        graph_calcs.build_graph()

        for _, row in dependency_table.loc[dependency_table['target'] == 'people'].iterrows():
            graph_calcs._calc_dependencies(
                source_attrs={'ci_type': row.source},
                target_attrs={'ci_type': row.target},
                via_attrs={'ci_type': 'road'},
                link_attrs={'ci_type': f'dependency_{row.source}_{row.target}'},
                link_condition=row['link_condition'],
                dist_thresh=row['thresh_dist'],
                bidir_link=row['bidir_link'],
            )

            # Call _check_access
            graph_calcs._check_access(row, friction_surf=None, rerouting=True)

            # Verify access states are set on people nodes
            people_nodes = graph_calcs.graph.vs.select(ci_type='people')
            for node in people_nodes:
                assert f'access_state_{row.source}_{row.target}' in node.attributes()
                assert f'actual_supply_{row.source}_{row.target}' in node.attributes()
                assert node[f'access_state_{row.source}_{row.target}'] == 'access undisrupted'

    def test_check_access_with_rerouting_failed_source(self, graph_calcs_with_remote_node, dependency_table):
        """Test _check_access with rerouting enabled"""
        graph_calcs_with_remote_node.parent.network.initialize_funcstates()
        graph_calcs_with_remote_node.parent.network.initialize_capacity(dependency_table)
        graph_calcs_with_remote_node.parent.network.initialize_supply(dependency_table)
        graph_calcs_with_remote_node.build_graph()

        row = dependency_table.iloc[1]

        graph_calcs_with_remote_node._calc_dependencies(
            source_attrs={'ci_type': row.source},
            target_attrs={'ci_type': row.target},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': f'dependency_{row.source}_{row.target}'},
            link_condition=row['link_condition'],
            dist_thresh=row['thresh_dist'],
            bidir_link=row['bidir_link'],
        )

        #fail ci to test rerouting
        healthcare = graph_calcs_with_remote_node.graph.vs.select(ci_type='healthcare')
        healthcare[0]['func_tot'] = 0

        # Call _check_access
        graph_calcs_with_remote_node._check_access(row, friction_surf=None, rerouting=True)
        # Verify access states are set on people nodes
        people_nodes = graph_calcs_with_remote_node.graph.vs.select(ci_type='people')
        for node in people_nodes:
            assert f'access_state_{row.source}_{row.target}' in node.attributes()
            assert f'actual_supply_{row.source}_{row.target}' in node.attributes()
            assert node[f'access_state_{row.source}_{row.target}'] == 'access new source'
            assert node[f'actual_supply_{row.source}_{row.target}'] == 1

    def test_check_access_with_rerouting_via_disrupted(self, graph_calcs_with_remote_node, dependency_table):
        """Test _check_access with rerouting when via is failing"""
        graph_calcs_with_remote_node.parent.network.initialize_funcstates()
        graph_calcs_with_remote_node.parent.network.initialize_capacity(dependency_table)
        graph_calcs_with_remote_node.parent.network.initialize_supply(dependency_table)
        graph_calcs_with_remote_node.build_graph()

        row = dependency_table.iloc[1]  # Use row 1 (healthcare->people enduser)

        graph_calcs_with_remote_node._calc_dependencies(
            source_attrs={'ci_type': row.source},
            target_attrs={'ci_type': row.target},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': f'dependency_{row.source}_{row.target}'},
            link_condition=row['link_condition'],
            dist_thresh=row['thresh_dist'],
            bidir_link=row['bidir_link'],
        )

        # Fail a road node to test rerouting
        graph_calcs_with_remote_node.graph.vs.select(ci_type='road')[0]['func_tot'] = 0
        graph_calcs_with_remote_node.graph.es.select(ci_type='road')[0]['func_tot'] = 0

        # Call _check_access
        graph_calcs_with_remote_node._check_access(row, friction_surf=None, rerouting=True)
        # Verify access states are set on people nodes
        people_nodes = graph_calcs_with_remote_node.graph.vs.select(ci_type='people')
        for node in people_nodes:
            assert f'access_state_{row.source}_{row.target}' in node.attributes()
            assert f'actual_supply_{row.source}_{row.target}' in node.attributes()
            # With rerouting enabled and just one road failed, access should still be possible
            assert node[f'access_state_{row.source}_{row.target}'] == 'access disrupted via'

    def test_check_access_with_rerouting_new_source(self, graph_calcs_with_remote_node, dependency_table):
        """Test _check_access with rerouting when via is failing"""
        graph_calcs_with_remote_node.parent.network.initialize_funcstates()
        graph_calcs_with_remote_node.parent.network.initialize_capacity(dependency_table)
        graph_calcs_with_remote_node.parent.network.initialize_supply(dependency_table)
        graph_calcs_with_remote_node.build_graph()

        row = dependency_table.iloc[1]  # Use row 1 (healthcare->people enduser)

        graph_calcs_with_remote_node._calc_dependencies(
            source_attrs={'ci_type': row.source},
            target_attrs={'ci_type': row.target},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': f'dependency_{row.source}_{row.target}'},
            link_condition=row['link_condition'],
            dist_thresh=row['thresh_dist'],
            bidir_link=row['bidir_link'],
        )

        # Fail a road node to test rerouting
        graph_calcs_with_remote_node.graph.vs[3]['func_tot'] = 0
        graph_calcs_with_remote_node.graph.es[3]['func_tot'] = 0

        # Call _check_access
        graph_calcs_with_remote_node._check_access(row, friction_surf=None, rerouting=True)
        # Verify access states are set on people nodes
        people_nodes = graph_calcs_with_remote_node.graph.vs.select(ci_type='people')
        for node in people_nodes:
            assert f'access_state_{row.source}_{row.target}' in node.attributes()
            assert f'actual_supply_{row.source}_{row.target}' in node.attributes()
            # With rerouting enabled and just one road failed, access should still be possible
            assert node[f'access_state_{row.source}_{row.target}'] == 'access undisrupted'

    def test_check_access_without_rerouting_via_disrupted(self, graph_calcs_with_remote_node, dependency_table):
        """Test _check_access with rerouting when via is failing"""
        graph_calcs_with_remote_node.parent.network.initialize_funcstates()
        graph_calcs_with_remote_node.parent.network.initialize_capacity(dependency_table)
        graph_calcs_with_remote_node.parent.network.initialize_supply(dependency_table)
        graph_calcs_with_remote_node.build_graph()

        row = dependency_table.iloc[1]  # Use row 1 (healthcare->people enduser)

        graph_calcs_with_remote_node._calc_dependencies(
            source_attrs={'ci_type': row.source},
            target_attrs={'ci_type': row.target},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': f'dependency_{row.source}_{row.target}'},
            link_condition=row['link_condition'],
            dist_thresh=row['thresh_dist'],
            bidir_link=row['bidir_link'],
        )

        # Fail a road node to test rerouting
        graph_calcs_with_remote_node.graph.vs[3]['func_tot'] = 0
        graph_calcs_with_remote_node.graph.es[3]['func_tot'] = 0

        # Call _check_access
        graph_calcs_with_remote_node._check_access(row, friction_surf=None, rerouting=False)
        # Verify access states are set on people nodes
        people_nodes = graph_calcs_with_remote_node.graph.vs.select(ci_type='people')
        for node in people_nodes:
            assert f'access_state_{row.source}_{row.target}' in node.attributes()
            assert f'actual_supply_{row.source}_{row.target}' in node.attributes()
            # Without rerouting enabled, access should be disrupted
            assert node[f'access_state_{row.source}_{row.target}'] == 'access disrupted via'

    def test_check_access_without_rerouting_failed_source(self, graph_calcs_with_remote_node, dependency_table):
        """Test _check_access with rerouting enabled"""
        graph_calcs_with_remote_node.parent.network.initialize_funcstates()
        graph_calcs_with_remote_node.parent.network.initialize_capacity(dependency_table)
        graph_calcs_with_remote_node.parent.network.initialize_supply(dependency_table)
        graph_calcs_with_remote_node.build_graph()

        row = dependency_table.iloc[1]

        graph_calcs_with_remote_node._calc_dependencies(
            source_attrs={'ci_type': row.source},
            target_attrs={'ci_type': row.target},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': f'dependency_{row.source}_{row.target}'},
            link_condition=row['link_condition'],
            dist_thresh=row['thresh_dist'],
            bidir_link=row['bidir_link'],
        )

        #fail ci to test rerouting
        healthcare = graph_calcs_with_remote_node.graph.vs.select(ci_type='healthcare')
        healthcare[0]['func_tot'] = 0

        # Call _check_access
        graph_calcs_with_remote_node._check_access(row, friction_surf=None, rerouting=False)
        # Verify access states are set on people nodes
        people_nodes = graph_calcs_with_remote_node.graph.vs.select(ci_type='people')
        for node in people_nodes:
            assert f'access_state_{row.source}_{row.target}' in node.attributes()
            assert f'actual_supply_{row.source}_{row.target}' in node.attributes()
            assert node[f'access_state_{row.source}_{row.target}'] == 'access disrupted source'
            assert node[f'actual_supply_{row.source}_{row.target}'] == 0

    def test_check_access_no_constraint(self, graph_calcs_with_remote_node, dependency_table):
        """Test _check_access with access constraints disabled"""
        graph_calcs_with_remote_node.parent.network.initialize_funcstates()
        graph_calcs_with_remote_node.parent.network.initialize_capacity(dependency_table)
        graph_calcs_with_remote_node.parent.network.initialize_supply(dependency_table)
        graph_calcs_with_remote_node.build_graph()

        for _, row in dependency_table.loc[dependency_table['target'] == 'people'].iterrows():
            graph_calcs_with_remote_node._calc_dependencies(
                source_attrs={'ci_type': row.source},
                target_attrs={'ci_type': row.target},
                via_attrs={'ci_type': 'road'},
                link_attrs={'ci_type': f'dependency_{row.source}_{row.target}'},
                link_condition=row['link_condition'],
                dist_thresh=row['thresh_dist'],
                bidir_link=row['bidir_link'],
            )
            #deactivate access constraints
            row['access_cnstr'] = False

            # Fail a road node to test access constraint being ignored
            graph_calcs_with_remote_node.graph.vs.select(ci_type='road')[1]['func_tot'] = 0
            graph_calcs_with_remote_node.graph.es.select(ci_type='road')[1]['func_tot'] = 0

            # Call _check_access
            graph_calcs_with_remote_node._check_access(row, friction_surf=None, rerouting=True)

            # Verify access states are set on people nodes
            people_nodes = graph_calcs_with_remote_node.graph.vs.select(ci_type='people')
            for node in people_nodes:
                assert f'access_state_{row.source}_{row.target}' in node.attributes()
                assert f'actual_supply_{row.source}_{row.target}' in node.attributes()
                assert node[f'access_state_{row.source}_{row.target}'] == 'access undisrupted'

    def test_check_access_no_rerouting_constraint_all_functional(self, graph_calcs):
        """Test _check_access with no rerouting, access constraints, all edges functional"""
        graph_calcs.parent.network.initialize_funcstates()
        graph_calcs.build_graph()

        # Create initial dependencies
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': 'healthcare'},
            target_attrs={'ci_type': 'people'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dependency_healthcare_people'},
            link_condition='distance',
            dist_thresh=10E6,
            bidir_link=False
        )

        # Initialize supply
        for v in graph_calcs.graph.vs.select(ci_type='people'):
            v['access_state_healthcare_people'] = 'no base access'
            v['actual_supply_healthcare_people'] = 0

        # Create dependency row
        row = pd.Series({
            'source': 'healthcare',
            'target': 'people',
            'via_link': 'road',
            'link_condition': 'distance',
            'thresh_dist': 10E6,
            'bidir_link': False,
            'access_cnstr': True
        })

        # Check access with no rerouting
        graph_calcs._check_access(row, friction_surf=None, rerouting=False)

        # All via edges are functional, so access should be undisrupted
        people_node = graph_calcs.graph.vs.select(ci_type='people')[0]
        assert people_node['access_state_healthcare_people'] == 'access undisrupted'
        assert people_node['actual_supply_healthcare_people'] == 1

    def test_check_access_no_rerouting_constraint_via_failed(self, graph_calcs):
        """Test _check_access with no rerouting, access constraints, failed via edge"""
        graph_calcs.parent.network.initialize_funcstates()
        graph_calcs.build_graph()

        # Create initial dependencies
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': 'healthcare'},
            target_attrs={'ci_type': 'people'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dependency_healthcare_people'},
            link_condition='distance',
            dist_thresh=10E6,
            bidir_link=False
        )

        # Initialize supply
        for v in graph_calcs.graph.vs.select(ci_type='people'):
            v['access_state_healthcare_people'] = 'no base access'
            v['actual_supply_healthcare_people'] = 0

        # Fail one via edge (break the path)
        graph_calcs.graph.es[1]['func_tot'] = 0

        # Create dependency row
        row = pd.Series({
            'source': 'healthcare',
            'target': 'people',
            'via_link': 'road',
            'link_condition': 'distance',
            'thresh_dist': 10E6,
            'bidir_link': False,
            'access_cnstr': True
        })

        # Check access with no rerouting
        graph_calcs._check_access(row, friction_surf=None, rerouting=False)

        # Via edge failed, so access should be disrupted
        people_node = graph_calcs.graph.vs.select(ci_type='people')[0]
        assert people_node['access_state_healthcare_people'] == 'access disrupted via'
        assert people_node['actual_supply_healthcare_people'] == 0

    def test_check_access_no_rerouting_no_constraint_via_failed(self, graph_calcs):
        """Test _check_access with no rerouting, no access constraints, failed via edge"""
        graph_calcs.parent.network.initialize_funcstates()
        graph_calcs.build_graph()

        # Create initial dependencies
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': 'healthcare'},
            target_attrs={'ci_type': 'people'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dependency_healthcare_people'},
            link_condition='distance',
            dist_thresh=10E6,
            bidir_link=False
        )

        # Initialize supply
        for v in graph_calcs.graph.vs.select(ci_type='people'):
            v['access_state_healthcare_people'] = 'no base access'
            v['actual_supply_healthcare_people'] = 0

        # Fail one via edge
        graph_calcs.graph.es[1]['func_tot'] = 0

        # Create dependency row without access constraints
        row = pd.Series({
            'source': 'healthcare',
            'target': 'people',
            'via_link': 'road',
            'link_condition': 'distance',
            'thresh_dist': 10E6,
            'bidir_link': False,
            'access_cnstr': False
        })

        # Check access with no rerouting
        graph_calcs._check_access(row, friction_surf=None, rerouting=False)

        # No access constraints, so even with failed via edge, access should be undisrupted
        people_node = graph_calcs.graph.vs.select(ci_type='people')[0]
        assert people_node['access_state_healthcare_people'] == 'access undisrupted'
        assert people_node['actual_supply_healthcare_people'] == 1


    def test_propagate_check_fail_people(self, graph_calcs):
        """Test propagate_check_fail setup"""
        graph_calcs.build_graph()

        # Initialize capacity for propagation
        graph_calcs.graph.vs['capacity_road_people'] = 0
        for v in graph_calcs.graph.vs.select(ci_type='road'):
            v['capacity_road_people'] = 1
            v['func_tot'] = 1  # Ensure roads are functional
        for v in graph_calcs.graph.vs.select(ci_type='people'):
            v['actual_supply_road_people'] = 0
            v['capacity_road_people'] = -1  # Need positive capacity to receive supply

        #add dependency edge for propagation
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': 'road'},
            target_attrs={'ci_type': 'people'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dependency_road_people'},
            link_condition='distance',
            dist_thresh=10E6,
            bidir_link=False,
        )

        # Run propagation
        graph_calcs._propagate_check_fail(
            source='road',
            target='people',
            type_I='enduser',
            thresh_func=1
        )

        # Verify propagation completed
        assert 'actual_supply_road_people' in graph_calcs.graph.vs.attributes()
        assert all(v['actual_supply_road_people'] == 1 for v in graph_calcs.graph.vs.select(ci_type='people'))
        assert all(v['access_state_road_people'] == 'access undisrupted' for v in graph_calcs.graph.vs.select(ci_type='people'))

    def test_propagate_check_fail_ci(self, graph_calcs):
        """Test propagate_check_fail setup"""
        graph_calcs.build_graph()

        # Initialize capacity for propagation
        graph_calcs.graph.vs['capacity_road_healthcare'] = 0
        for v in graph_calcs.graph.vs.select(ci_type='road'):
            v['capacity_road_healthcare'] = 1
        for v in graph_calcs.graph.vs.select(ci_type='healthcare'):
            v['actual_supply_road_healthcare'] = 0
            v['capacity_road_healthcare'] = -1

        #add dependency edge for propagation
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': 'road'},
            target_attrs={'ci_type': 'healthcare'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dependency_road_healthcare'},
            link_condition='distance',
            dist_thresh=10E6,
            bidir_link=False,
        )

        # Run propagation
        graph_calcs._propagate_check_fail(
            source='road',
            target='healthcare',
            type_I='functional',
            thresh_func=1
        )

        # Verify propagation completed
        assert 'func_tot' in graph_calcs.graph.vs.attributes()
        assert all(v['func_tot'] ==1 for v in graph_calcs.graph.vs.select(ci_type='healthcare'))
        assert "access_state_healthcare_road" not in graph_calcs.graph.vs.attributes()  # CI propagation should not set access state

    def test_propagate_check_fail_fail_ci(self, graph_calcs):
        """Test propagate_check_fail setup"""
        graph_calcs.build_graph()

        # Initialize capacity for propagation
        graph_calcs.graph.vs['capacity_road_healthcare'] = 0
        for v in graph_calcs.graph.vs.select(ci_type='road'):
            v['func_tot'] = 0
            v['capacity_road_healthcare'] = 1
        for v in graph_calcs.graph.vs.select(ci_type='healthcare'):
            v['actual_supply_road_healthcare'] = 0
            v['capacity_road_healthcare'] = -1

        #add dependency edge for propagation
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': 'road'},
            target_attrs={'ci_type': 'healthcare'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dependency_road_healthcare'},
            link_condition='distance',
            dist_thresh=10E6,
            bidir_link=False,
        )
        # Run propagation
        graph_calcs._propagate_check_fail(
            source='road',
            target='healthcare',
            type_I='functional',
            thresh_func=1
        )

        # Verify propagation completed
        assert 'func_tot' in graph_calcs.graph.vs.attributes()
        assert all(v['func_tot'] ==0 for v in graph_calcs.graph.vs.select(ci_type='healthcare'))

    def test_propagate_check_fail_fail_enduser(self, graph_calcs):
        """Test propagate_check_fail setup"""
        graph_calcs.build_graph()

        # Initialize capacity for propagation
        graph_calcs.graph.vs['capacity_healthcare_people'] = 0
        for v in graph_calcs.graph.vs.select(ci_type='healthcare'):
            v['func_tot'] = 0
            v['capacity_healthcare_people'] = 1
        for v in graph_calcs.graph.vs.select(ci_type='people'):
            v['actual_supply_healthcare_people'] = 1
            v['capacity_healthcare_people'] = -1

        #add dependency edge for propagation
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': 'healthcare'},
            target_attrs={'ci_type': 'people'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dependency_healthcare_people'},
            link_condition='distance',
            dist_thresh=10E6,
            bidir_link=False,
        )

        # Run propagation
        graph_calcs._propagate_check_fail(
            source='healthcare',
            target='people',
            type_I='enduser',
            thresh_func=1
        )

        # Verify propagation completed
        assert 'actual_supply_healthcare_people' in graph_calcs.graph.vs.attributes()
        assert all(v['actual_supply_healthcare_people'] == 0 for v in graph_calcs.graph.vs.select(ci_type='people'))
        assert all(v['access_state_healthcare_people'] == 'access disrupted' for v in graph_calcs.graph.vs.select(ci_type='people'))

    def test_update_internal_dependencies_roads(self, graph_calcs):
        """Test updating internal dependencies for roads"""
        graph_calcs.build_graph()

        # Set some edge functionality
        graph_calcs.graph.es['func_tot'] = 0

        graph_calcs._update_internal_dependencies(
            p_source='powerstation',
            p_sink='powerline',
            source_var='capacity',
            demand_var='demand'

        )

        # Method should complete
        assert all(v['func_tot'] ==0 for v in graph_calcs.graph.vs.select(ci_type='road'))

    def test_update_functional_dependencies(self, graph_calcs):
        """Test updating functional dependencies"""
        graph_calcs.build_graph()

        # Create dependency dataframe
        df_dependencies = pd.DataFrame({
            'source': ['road'],
            'target': ['healthcare'],
            'type_I': ['functional'],
            'type_II': ['logical'],
            'via_link': ['none'],
            'thresh_func': [1.0],
            'thresh_dist': [np.inf],
            'bidir_link': [False],
            'access_cnstr': [False],
        })

        # Initialize capacities
        graph_calcs.graph.vs['capacity_road_healthcare'] = 0
        for v in graph_calcs.graph.vs.select(ci_type='road'):
            v['capacity_road_healthcare'] = 1
        for v in graph_calcs.graph.vs.select(ci_type='healthcare'):
            v['capacity_road_healthcare'] = -1

        graph_calcs._update_functional_dependencies(df_dependencies)
        assert all(v['func_tot'] == 1 for v in graph_calcs.graph.vs.select(ci_type='healthcare'))

        #repeat with road failure
        #set all road nodes as failed
        for v in graph_calcs.graph.vs.select(ci_type='road'):
            v['func_tot'] = 0
        graph_calcs._update_functional_dependencies(df_dependencies)
        assert all(v['func_tot'] == 0 for v in graph_calcs.graph.vs.select(ci_type='healthcare'))

    def test_update_enduser_dependencies_routing(self, graph_calcs_with_source_fail):
        """Test updating end-user dependencies"""
        graph_calcs_with_source_fail.build_graph()

        # Create end-user dependency dataframe
        df_dependencies = pd.DataFrame({
            'source': ['road', 'healthcare'],
            'target': ['people', 'people'],
            'type_I': ['enduser', 'enduser'],
            'access_cnstr': [False, True],
            'via_link': ['none', 'road'],
            'link_condition': ['edgecond', 'distance'],
            'thresh_dist': [10E6, 10E6],
            'bidir_link': [False, False],
            'thresh_func': [1.0, 1.0]
        })

        graph_calcs_with_source_fail._update_enduser_dependencies(
            df_dependencies,
            friction_surf=None,
            rerouting=False,
            access_check_method="routing"
        )

        assert graph_calcs_with_source_fail.graph is not None

    def test_update_enduser_dependencies_propagation(self, graph_calcs_with_source_fail):
         """Test updating end-user dependencies"""

         # Create end-user dependency dataframe
         df_dependencies = pd.DataFrame({
             'source': ['road', 'healthcare'],
             'target': ['people', 'people'],
             'type_I': ['enduser', 'enduser'],
             'access_cnstr': [False, True],
             'via_link': ['none', 'road'],
             'link_condition': ['edgecond', 'distance'],
             'thresh_dist': [10E6, 10E6],
             'bidir_link': [False, False],
             'thresh_func': [1.0, 1.0]
         })
         graph_calcs_with_source_fail.parent.network.initialize_funcstates()
         graph_calcs_with_source_fail.parent.network.initialize_capacity(df_dependencies)
         graph_calcs_with_source_fail.parent.network.initialize_supply(df_dependencies)
         graph_calcs_with_source_fail.build_graph()

         graph_calcs_with_source_fail._update_enduser_dependencies(
             df_dependencies,
             friction_surf=None,
             rerouting=False,
             access_check_method="propagation"
         )

         assert graph_calcs_with_source_fail.graph is not None

    def test_update_enduser_dependencies_unknown(self, graph_calcs_with_source_fail):
         """Test updating end-user dependencies"""
         graph_calcs_with_source_fail.build_graph()

         # Create end-user dependency dataframe
         df_dependencies = pd.DataFrame({
             'source': ['road', 'healthcare'],
             'target': ['people', 'people'],
             'type_I': ['enduser', 'enduser'],
             'access_cnstr': [False, True],
             'via_link': ['none', 'road'],
             'link_condition': ['edgecond', 'distance'],
             'thresh_dist': [10E6, 10E6],
             'bidir_link': [False, False],
             'thresh_func': [1.0, 1.0]
         })

         with pytest.raises(ValueError, match="Invalid access check method specified!"):
            graph_calcs_with_source_fail._update_enduser_dependencies(
                df_dependencies,
                friction_surf=None,
                rerouting=False,
                access_check_method="blabla"
            )

    def test_get_former_access_info(self, graph_calcs):
        """Test _get_former_access_info helper function"""
        graph_calcs.parent.network.initialize_funcstates()
        graph_calcs.build_graph()

        # Create dependencies
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': 'healthcare'},
            target_attrs={'ci_type': 'people'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dependency_healthcare_people'},
            link_condition='distance',
            dist_thresh=10E6,
            bidir_link=False
        )

        # Get former access info
        es_access_base, ppl_former_access, ppl_former_access_source_failed = \
            graph_calcs._get_former_access_info('dependency_healthcare_people')

        assert len(es_access_base) == 1
        assert len(ppl_former_access) == 1
        assert len(ppl_former_access_source_failed) == 0  # No sources failed yet

        # Fail a source
        healthcare_nodes = graph_calcs.graph.vs.select(ci_type='healthcare')
        healthcare_nodes[0]['func_tot'] = 0

        # Get former access info again
        es_access_base, ppl_former_access, ppl_former_access_source_failed = \
            graph_calcs._get_former_access_info('dependency_healthcare_people')

        assert len(ppl_former_access_source_failed) == 1

    def test_recompute_dependencies_with_rerouting(self, graph_calcs):
        """Test _recompute_dependencies_with_rerouting helper function"""
        graph_calcs.parent.network.initialize_funcstates()
        graph_calcs.build_graph()

        # Create initial dependencies
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': 'healthcare'},
            target_attrs={'ci_type': 'people'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dependency_healthcare_people'},
            link_condition='distance',
            dist_thresh=10E6,
            bidir_link=False
        )

        row = pd.Series({
            'source': 'healthcare',
            'target': 'people',
            'via_link': 'road',
            'link_condition': 'distance',
            'thresh_dist': 10E6,
            'bidir_link': False,
            'access_cnstr': True
        })

        # Recompute with rerouting
        ppl_new_access, ppl_access_all_via = \
            graph_calcs._recompute_dependencies_with_rerouting(
                row, 'dependency_healthcare_people'
            )

        assert len(ppl_new_access) == 1
        assert len(ppl_access_all_via) == 1

    def test_recompute_dependencies_with_rerouting_fail_source(self, graph_calcs):
        """Test _recompute_dependencies_with_rerouting helper function with failed source"""
        graph_calcs.parent.network.initialize_funcstates()
        graph_calcs.build_graph()

        # Create initial dependencies
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': 'healthcare'},
            target_attrs={'ci_type': 'people'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dependency_healthcare_people'},
            link_condition='distance',
            dist_thresh=10E6,
            bidir_link=False
        )

        row = pd.Series({
            'source': 'healthcare',
            'target': 'people',
            'via_link': 'road',
            'link_condition': 'distance',
            'thresh_dist': 10E6,
            'bidir_link': False,
            'access_cnstr': True
        })

        # Fail the healthcare source
        healthcare_nodes = graph_calcs.graph.vs.select(ci_type='healthcare')
        healthcare_nodes[0]['func_tot'] = 0

        # Recompute with rerouting
        ppl_new_access, ppl_access_all_via = \
            graph_calcs._recompute_dependencies_with_rerouting(
                row, 'dependency_healthcare_people'
            )

        assert len(ppl_new_access) == 0
        assert len(ppl_access_all_via) == 0

    def test_recompute_dependencies_with_rerouting_fail_via(self, graph_calcs):
        """Test _recompute_dependencies_with_rerouting helper function with failed via link"""
        graph_calcs.parent.network.initialize_funcstates()
        graph_calcs.build_graph()

        # Create initial dependencies
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': 'healthcare'},
            target_attrs={'ci_type': 'people'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dependency_healthcare_people'},
            link_condition='distance',
            dist_thresh=10E6,
            bidir_link=False
        )

        row = pd.Series({
            'source': 'healthcare',
            'target': 'people',
            'via_link': 'road',
            'link_condition': 'distance',
            'thresh_dist': 10E6,
            'bidir_link': False,
            'access_cnstr': True
        })

        # Fail the via link
        road_nodes = graph_calcs.graph.vs.select(ci_type='road')
        road_nodes[0]['func_tot'] = 0
        graph_calcs.graph.es.select(ci_type='road')[0]['func_tot'] = 0

        # Recompute with rerouting
        ppl_new_access, ppl_access_all_via = \
            graph_calcs._recompute_dependencies_with_rerouting(
                row, 'dependency_healthcare_people'
            )

        assert len(ppl_new_access) == 0
        assert len(ppl_access_all_via) == 1

    def test_validate_dependency_paths(self, graph_calcs):
        """Test _validate_dependency_paths helper function"""
        graph_calcs.parent.network.initialize_funcstates()
        graph_calcs.build_graph()

        row = pd.Series({
            'source': 'healthcare',
            'target': 'people',
            'via_link': 'road',
            'link_condition': 'distance',
            'thresh_dist': 10E6,
            'bidir_link': False,
            'access_cnstr': True
        })

        # Create dependencies
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': row['source']},
            target_attrs={'ci_type': row['target']},
            via_attrs={'ci_type': row['via_link']},
            link_attrs={'ci_type': 'dependency_' + row['source'] + '_' + row['target']},
            link_condition=row['link_condition'],
            dist_thresh=row['thresh_dist'],
            bidir_link=False
        )

        es_check = list(graph_calcs.graph.es.select(ci_type='dependency_healthcare_people'))
        edge_pairs = [(edge.source, edge.target) for edge in es_check]

        # Create subgraph
        subgraph = graph_calcs._create_subgraph(
                    source_attrs={'ci_type': row['source'], 'func_tot': 1},
                    target_attrs={'ci_type': row['target']},
                    via_attrs={'ci_type': row['via_link'], 'func_tot': 1}
                )

        row = pd.Series({
            'thresh_dist': 10E6
        })

        # Validate paths
        pairs_to_keep = graph_calcs._validate_dependency_paths(
            edge_pairs, row, subgraph
        )

        # All paths should be valid initially
        assert len(pairs_to_keep) == len(edge_pairs)

    def test_validate_dependency_paths_edgefail(self, graph_calcs):
        """Test _validate_dependency_paths helper function"""
        graph_calcs.parent.network.initialize_funcstates()
        graph_calcs.build_graph()

        row = pd.Series({
            'source': 'healthcare',
            'target': 'people',
            'via_link': 'road',
            'link_condition': 'distance',
            'thresh_dist': 10E6,
            'bidir_link': False,
            'access_cnstr': True
        })

        # Create dependencies
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': row['source']},
            target_attrs={'ci_type': row['target']},
            via_attrs={'ci_type': row['via_link']},
            link_attrs={'ci_type': 'dependency_' + row['source'] + '_' + row['target']},
            link_condition=row['link_condition'],
            dist_thresh=row['thresh_dist'],
            bidir_link=False
        )

        es_check = list(graph_calcs.graph.es.select(ci_type='dependency_healthcare_people'))
        edge_pairs = [(edge.source, edge.target) for edge in es_check]

        #fail one road edge
        graph_calcs.graph.es.select(ci_type='road')[0]['func_tot'] = 0

        # Create subgraph
        subgraph = graph_calcs._create_subgraph(
                    source_attrs={'ci_type': row['source'], 'func_tot': 1},
                    target_attrs={'ci_type': row['target']},
                    via_attrs={'ci_type': row['via_link'], 'func_tot': 1}
                )

        row = pd.Series({
            'thresh_dist': 10E6
        })

        # Validate paths
        pairs_to_keep = graph_calcs._validate_dependency_paths(
            edge_pairs, row, subgraph
        )

        # after only path is failed no remaining paths are left
        assert len(pairs_to_keep) == 0

    def test_validate_dependencies_without_rerouting_with_constraint(self, graph_calcs):
        """Test _validate_dependencies_without_rerouting with access constraints"""
        graph_calcs.parent.network.initialize_funcstates()
        graph_calcs.build_graph()

        row = pd.Series({
            'source': 'healthcare',
            'target': 'people',
            'via_link': 'road',
            'link_condition': 'distance',
            'thresh_dist': 10E6,
            'bidir_link': False,
            'access_cnstr': True
        })

        dependency_name = 'dependency_' + row['source'] + '_' + row['target']

        # Create dependencies
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': row['source']},
            target_attrs={'ci_type': row['target']},
            via_attrs={'ci_type': row['via_link']},
            link_attrs={'ci_type': dependency_name},
            link_condition=row['link_condition'],
            dist_thresh=row['thresh_dist'],
            bidir_link=False
        )

        es_access_base = list(graph_calcs.graph.es.select(ci_type=dependency_name))
        ppl_former_access = [edge.target for edge in es_access_base]


        # Validate without rerouting
        ppl_new_access, ppl_access_all_via = \
            graph_calcs._validate_dependencies_without_rerouting(
                row, dependency_name, es_access_base,
            )

        assert len(ppl_new_access) == 1
        assert ppl_access_all_via == ppl_former_access

    def test_mark_access_states_and_supply(self, graph_calcs):
        """Test _mark_access_states_and_supply helper function"""
        graph_calcs.parent.network.initialize_funcstates()
        graph_calcs.build_graph()

        # Create dependencies
        graph_calcs._calc_dependencies(
            source_attrs={'ci_type': 'healthcare'},
            target_attrs={'ci_type': 'people'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dependency_healthcare_people'},
            link_condition='distance',
            dist_thresh=10E6,
            bidir_link=False
        )

        # Initialize access states
        for v in graph_calcs.graph.vs.select(ci_type='people'):
            v['access_state_healthcare_people'] = 'no base access'
            v['actual_supply_healthcare_people'] = 0

        es_access = list(graph_calcs.graph.es.select(ci_type='dependency_healthcare_people'))
        ppl_former_access = [edge.target for edge in es_access]
        ppl_former_access_source_failed = []
        ppl_access_all_via = ppl_former_access
        ppl_new_access = ppl_former_access

        row = pd.Series({
            'source': 'healthcare',
            'target': 'people'
        })

        # Mark access states
        graph_calcs._mark_access_states_and_supply(
            row, ppl_former_access, ppl_former_access_source_failed,
            ppl_access_all_via, ppl_new_access
        )

        # Check that access states were set
        people_nodes = graph_calcs.graph.vs.select(ci_type='people')
        for node in people_nodes:
            assert 'access_state_healthcare_people' in node.attributes()
            assert 'actual_supply_healthcare_people' in node.attributes()
            assert node['access_state_healthcare_people'] == 'access undisrupted'
            assert node['actual_supply_healthcare_people'] == 1

# ========================================================================
# Tests for NetworkCalcs class
# ========================================================================

from climada_petals.engine.networks.nw_calcs import NetworkCalcs


@pytest.fixture
def dependency_table():
    """Create a simple dependency table"""
    return pd.DataFrame({
        'source': ['road', 'healthcare', 'road'],
        'target': ['people', 'people', 'healthcare'],
        'type_I': ['enduser', 'enduser', 'functional'],
        'type_II': ['logical', 'logical', 'logical'],
        'via_link': ['none', 'road', 'none'],
        'thresh_func': [1, 1, 1],
        'link_condition': ['edgecond', 'distance', 'edgecond'],
        'thresh_dist': [5E6, 10E6, 10E6],
        'bidir_link': [False, False, False],
        'access_cnstr': [False, True, False],
        'n_links': [1, 1, 1]
    })


@pytest.fixture
def network_calcs(network_with_ci_types, dependency_table):
    """Create NetworkCalcs instance"""
    return NetworkCalcs(network=network_with_ci_types, dep_table=dependency_table)

@pytest.fixture
def network_calcs_edge_fail(network_with_edge_fail, dependency_table):
    """Create NetworkCalcs instance"""
    return NetworkCalcs(network=network_with_edge_fail, dep_table=dependency_table)

@pytest.fixture
def network_calcs_source_fail(network_with_source_fail, dependency_table):
    """Create NetworkCalcs instance"""
    return NetworkCalcs(network=network_with_source_fail, dep_table=dependency_table)

@pytest.fixture
def expected_dep_pairs():
    """Expected dependency edges between node pairs."""
    return {
        'dependency_road_people': ([1,], [0,]), #sources targets
        'dependency_healthcare_people': ([4,], [0,]),
        'dependency_road_healthcare': ([3,], [4,]),
    }


#@pytest.fixture
#def forbidden_dep_pairs():
#    """Node pairs that must not carry dependency edges."""
#    return {(0, 2), (1, 3), (2, 3)}
#
#
#def _norm_pair(edge):
#    """Return a sorted (u, v) tuple for an undirected edge."""
#    return tuple(sorted((edge.source, edge.target)))
#
#
#def _assert_dependency_pairs(graph, expected_dep_pairs, forbidden_dep_pairs):
#    """Check expected dependency edges and ensure none exist on forbidden pairs."""
#    dep_names = list(expected_dep_pairs.keys())
#
#    for dep_name, expected_pairs in expected_dep_pairs.items():
#        es = graph.es.select(ci_type=dep_name)
#        assert len(es) > 0
#        found = {_norm_pair(e) for e in es}
#        assert found == expected_pairs
#
#    # No dependency edges on forbidden pairs
#    forbidden_edges = graph.es.select(ci_type_in=dep_names)
#    for edge in forbidden_edges:
#        assert _norm_pair(edge) not in forbidden_dep_pairs

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
        assert len(network_calcs.graph_calc.graph.connected_components(mode='weak')) == 1

    def test_add_physical_links(self, network_calcs):
        """Test adding physical links to network"""
        initial_edge_count = len(network_calcs.network.edges)

        network_calcs.add_physical_links()

        # check that one edge between people and road has been added
        assert len(network_calcs.network.edges) >= initial_edge_count + 1
        assert len(network_calcs.network.nodes) == len(network_calcs.network.nodes)
        assert network_calcs.network.edges.iloc[initial_edge_count]['ci_type'] == 'road'


    def test_setup_dependencies(self, network_calcs, expected_dep_pairs):
        """Ensure dependency edges exist before access checks"""
        network_calcs.initialize_base_state()
        network_calcs.setup_dependencies()


        for dep_link, (exp_sources, exp_targets) in expected_dep_pairs.items():
            sources = [e.source for e in network_calcs.graph.es if e['ci_type'] == dep_link]
            targets = [e.target for e in network_calcs.graph.es if e['ci_type'] == dep_link]

            assert  len(sources) == len(exp_sources)
            assert  len(targets) == len(exp_targets)
            for i in range(len(sources)):
                assert sources[i] == exp_sources[i]
                assert targets[i] == exp_targets[i]

    def test_network_calcs_graph_property(self, network_calcs):
        """Test graph property of NetworkCalcs"""
        graph = network_calcs.graph

        assert isinstance(graph, ig.Graph)

    def test_cascade_initial(self, network_calcs):
        """Test cascade with initial flag"""
        network_calcs.network.initialize_capacity(network_calcs.dep_table)
        network_calcs.network.initialize_supply(network_calcs.dep_table)
        network_calcs.setup_dependencies()

        # Run cascade with initial=True
        network_calcs.cascade(
            initial=True,
            friction_surf=None,
            rerouting=False
        )

        # Verify network state was updated
        assert 'actual_supply_road_people' in network_calcs.network.nodes.columns
        assert 'actual_supply_healthcare_people' in network_calcs.network.nodes.columns
        assert np.all(network_calcs.network.nodes.loc[network_calcs.network.nodes["ci_type"]=="people",'actual_supply_road_people']) == 1
        assert np.all(network_calcs.network.nodes.loc[network_calcs.network.nodes["ci_type"]=="people",'actual_supply_healthcare_people']) == 1

    def test_cascade_simple(self, network_calcs_source_fail):
        """Test simple cascade without friction surface"""
        network_calcs_source_fail.network.initialize_capacity(network_calcs_source_fail.dep_table)
        network_calcs_source_fail.network.initialize_supply(network_calcs_source_fail.dep_table)
        network_calcs_source_fail.setup_dependencies()

        assert np.all(network_calcs_source_fail.network.nodes.loc[network_calcs_source_fail.network.nodes["ci_type"]=="people",'actual_supply_road_people']) == 1
        assert np.all(network_calcs_source_fail.network.nodes.loc[network_calcs_source_fail.network.nodes["ci_type"]=="people",'actual_supply_healthcare_people']) == 1

        network_calcs_source_fail.cascade(
            initial=False,
            friction_surf=None,
            rerouting=False
        )

        # Verify cascade completed
        assert np.all(network_calcs_source_fail.network.nodes.loc[network_calcs_source_fail.network.nodes["ci_type"]=="people",'actual_supply_road_people']) == 1
        assert np.all(network_calcs_source_fail.network.nodes.loc[network_calcs_source_fail.network.nodes["ci_type"]=="people",'actual_supply_healthcare_people']) == 1

class TestNetworkCalcsAdvanced:
    """Advanced test cases for NetworkCalcs"""

    @pytest.mark.skip(reason="powercap_from_clusters method not implemented")
    def test_cascade_with_dependencies(self):
        """Test cascade with complete dependency setup"""
        # Create a more complex network
        nodes = gpd.GeoDataFrame(
            {
                'id': [0, 1, 2, 3, 4],
                'orig_id': [0, 1, 2, 3, 4],
                'ci_type': ['healthcare', 'healthcare', 'road', 'road', 'people'],
                'func_tot': [1, 1, 1, 1, 1],
                'geometry': [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4)]
            },
            geometry='geometry',
            crs='EPSG:4326'
        )
        edges = gpd.GeoDataFrame(
            {
                'from_id': [0, 1, 2, 3],
                'to_id': [1, 2, 3, 4],
                'id': [0, 1, 2, 3],
                'orig_id': [0, 1, 2, 3],
                'osm_id': [100, 101, 102, 103],
                'ci_type': ['healthcare', 'road', 'road', 'road'],
                'func_tot': [1, 1, 1, 1],
                'distance': [157200, 157200, 157200, 157200],
                'geometry': [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                    LineString([(2, 2), (3, 3)]),
                    LineString([(3, 3), (4, 4)])
                ]
            },
            geometry='geometry',
            crs='EPSG:4326'
        )
        network = Network(edges=edges, nodes=nodes)

        dep_table = pd.DataFrame({
            'source': ['healthcare', 'road'],
            'target': ['road', 'people'],
            'type_I': ['functional', 'enduser'],
            'via_link': ['road', 'road'],
            'link_condition': ['distance', 'distance'],
            'thresh_dist': [np.inf, np.inf],
            'bidir_link': [False, False],
            'access_cnstr': [False, False],
            'n_links': [1, 1],
            'thresh_func': [0.5, 0.5]
        })

        nc = NetworkCalcs(network=network, dep_table=dep_table)
        nc.initialize_base_state()

        # Set a failure
        nc.network.nodes.loc[1, 'func_tot'] = 0

        # Run cascade
        nc.cascade(
            p_source='healthcare',
            p_sink='road',
            source_var='capacity',
            demand_var='demand',
            initial=False,
            friction_surf=None,
            rerouting=False
        )

        # Verify cascade completed and network was updated
        assert nc.network is not None
        assert 'func_tot' in nc.network.nodes.columns

    @pytest.mark.skip(reason="powercap_from_clusters method not implemented")
    def test_cascade_multiple_iterations(self):
        """Test cascade that requires multiple iterations"""
        nodes = gpd.GeoDataFrame(
            {
                'id': [0, 1, 2],
                'orig_id': [0, 1, 2],
                'ci_type': ['healthcare', 'road', 'road'],
                'func_tot': [1, 1, 1],
                'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]
            },
            geometry='geometry',
            crs='EPSG:4326'
        )
        edges = gpd.GeoDataFrame(
            {
                'from_id': [0, 1],
                'to_id': [1, 2],
                'id': [0, 1],
                'orig_id': [0, 1],
                'osm_id': [100, 101],
                'ci_type': ['road', 'road'],
                'func_tot': [1, 1],
                'distance': [157200, 157200],
                'geometry': [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)])
                ]
            },
            geometry='geometry',
            crs='EPSG:4326'
        )
        network = Network(edges=edges, nodes=nodes)

        dep_table = pd.DataFrame({
            'source': ['healthcare'],
            'target': ['road'],
            'type_I': ['functional'],
            'via_link': ['road'],
            'link_condition': ['distance'],
            'thresh_dist': [np.inf],
            'bidir_link': [False],
            'access_cnstr': [False],
            'n_links': [1],
            'thresh_func': [0.5]
        })

        nc = NetworkCalcs(network=network, dep_table=dep_table)
        nc.initialize_base_state()

        nc.cascade(
            p_source='healthcare',
            p_sink='road',
            source_var='capacity',
            demand_var='demand',
            initial=True
        )

        assert nc.network is not None

    def test_initial_cascade_with_setup_dependencies(self, network_with_ci_types):
        """Test that initial cascade works correctly even when setup_dependencies is called first"""
        import numpy as np

        # Create a dependency table with enduser dependencies
        dep_table = pd.DataFrame([{
            'Dep': 'dependency_healthcare_people',
            'source': 'healthcare',
            'target': 'people',
            'n_links': 1,
            'access_cnstr': True,
            'type_I': 'enduser',
            'type_II': 'logical',
            'thresh_func': 1,
            'thresh_dist': 50000,
            'thresh_dur': 90,
            'conditions': None,
            'link_condition': 'distance',
            'via_link': 'road',
            'bidir_link': True
        }])

        # Create network calculator
        nw_calc = NetworkCalcs(network_with_ci_types, dep_table=dep_table)

        # Add physical links
        nw_calc.add_physical_links()

        # Initialize base state
        nw_calc.initialize_base_state()

        # Setup dependencies (this creates the dependency edges)
        nw_calc.setup_dependencies()

        # Run initial cascade - this should treat it as initial state, not former access
        nw_calc.cascade(initial=True)

        # Check that people have access undisrupted, not access disrupted source
        people_nodes = nw_calc.graph.vs.select(ci_type='people')
        for node in people_nodes:
            # All people should have undisrupted access in initial state
            # (assuming they're within distance threshold)
            if 'access_state_healthcare_people' in node.attributes():
                assert node['access_state_healthcare_people'] in ['access undisrupted', 'no base access'], \
                    f"Node {node.index} has incorrect access state: {node['access_state_healthcare_people']}"



