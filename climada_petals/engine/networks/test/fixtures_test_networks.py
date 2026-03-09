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
import copy as cp
import numpy as np
from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_calcs import GraphCalcs
from climada_petals.engine.networks.nw_calcs import NetworkCalcs

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

# ========================================================================
# Fixtures for GraphCalcs class
# ========================================================================

@pytest.fixture
def graph_calcs(network_with_ci_types):
    """Create GraphCalcs instance with test network"""
    nw_calcs_mock = type('obj', (object,), {'network': network_with_ci_types})()
    return GraphCalcs(network_calc=nw_calcs_mock, directed=True)


@pytest.fixture
def graph_calcs_with_source_fail(network_with_source_fail):
    """Create GraphCalcs instance with test network containing CI failures"""
    nw_calcs_mock = type('obj', (object,), {'network': network_with_source_fail})()
    return GraphCalcs(network_calc=nw_calcs_mock, directed=True)

@pytest.fixture
def graph_calcs_with_edge_ci_fail(network_with_edge_fail):
    """Create GraphCalcs instance with test network containing edge CI failures"""
    nw_calcs_mock = type('obj', (object,), {'network': network_with_edge_fail})()
    return GraphCalcs(network_calc=nw_calcs_mock, directed=True)

@pytest.fixture
def graph_calcs_with_remote_node_missing_edge(network_with_remote_node_missing_edge):
    """Create GraphCalcs instance with test network containing missing edge"""
    nw_calcs_mock = type('obj', (object,), {'network': network_with_remote_node_missing_edge})()
    return GraphCalcs(network_calc=nw_calcs_mock, directed=True)

@pytest.fixture
def graph_calcs_with_remote_node(network_with_remote_node):
    """Create GraphCalcs instance with test network containing remote node"""
    nw_calcs_mock = type('obj', (object,), {'network': network_with_remote_node})()
    return GraphCalcs(network_calc=nw_calcs_mock, directed=True)

# ========================================================================
# Fixtures for NetworkCalcs class
# ========================================================================

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
        'thresh_dur': [np.inf, np.inf, np.inf],
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

