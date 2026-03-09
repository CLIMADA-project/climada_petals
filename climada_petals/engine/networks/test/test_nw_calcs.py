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
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import igraph as ig
import copy as cp
from climada_petals.engine.networks.test.fixtures_test_networks import *


def test_network_calcs_init(network_with_ci_types, dependency_table):
    """Test NetworkCalcs initialization"""
    nc = NetworkCalcs(network=network_with_ci_types, dep_table=dependency_table)

    assert nc.network == network_with_ci_types
    assert nc.dep_table is dependency_table
    assert isinstance(nc._graph_calc, GraphCalcs)


def test_initialize_base_state(network_calcs):
    """Test initialization of base functional state"""
    network_calcs.initialize_base_state()

    assert "func_internal" in network_calcs.network.nodes.columns
    assert "func_tot" in network_calcs.network.nodes.columns
    assert "func_internal" in network_calcs.network.edges.columns
    assert "func_tot" in network_calcs.network.edges.columns


def test_merge_clusters(network_calcs):
    """Test merging clusters when network is already connected"""

    network_calcs._graph_calc.build_graph()
    init_node_count = len(network_calcs.network.nodes)
    init_edge_count = len(network_calcs.network.edges)

    # Try to merge clusters
    network_calcs.merge_clusters(ci_type="road", max_iter=1, dist_thresh=np.inf)

    # Verify network structure is maintained
    assert len(network_calcs.network.nodes) == init_node_count
    assert (
        len(network_calcs.network.edges) >= init_edge_count
    )  #! double edges may be added
    assert len(network_calcs._graph_calc.graph.connected_components(mode="weak")) == 1


def test_add_physical_links(network_calcs):
    """Test adding physical links to network"""
    initial_edge_count = len(network_calcs.network.edges)

    network_calcs.add_physical_links()

    # check that one edge between people and road has been added
    assert len(network_calcs.network.edges) >= initial_edge_count + 1
    assert len(network_calcs.network.nodes) == len(network_calcs.network.nodes)
    assert network_calcs.network.edges.iloc[initial_edge_count]["ci_type"] == "road"


def test_setup_dependencies(network_calcs, expected_dep_pairs):
    """Ensure dependency edges exist before access checks"""
    network_calcs.initialize_base_state()
    network_calcs.setup_dependencies()

    for dep_link, (exp_sources, exp_targets) in expected_dep_pairs.items():
        sources = [e.source for e in network_calcs.graph.es if e["ci_type"] == dep_link]
        targets = [e.target for e in network_calcs.graph.es if e["ci_type"] == dep_link]

        assert len(sources) == len(exp_sources)
        assert len(targets) == len(exp_targets)
        for i in range(len(sources)):
            assert sources[i] == exp_sources[i]
            assert targets[i] == exp_targets[i]


def test_network_calcs_graph_property(network_calcs):
    """Test graph property of NetworkCalcs"""
    graph = network_calcs.graph

    assert isinstance(graph, ig.Graph)


def test_cascade_initial(network_calcs):
    """Test cascade with initial flag"""
    network_calcs.network.initialize_capacity(network_calcs.dep_table)
    network_calcs.network.initialize_supply(network_calcs.dep_table)
    network_calcs.setup_dependencies()

    # Run cascade with initial=True
    network_calcs.cascade(initial=True, friction_surf=None, rerouting=False)

    # Verify network state was updated
    assert "actual_supply_road_people" in network_calcs.network.nodes.columns
    assert "actual_supply_healthcare_people" in network_calcs.network.nodes.columns
    assert (
        np.all(
            network_calcs.network.nodes.loc[
                network_calcs.network.nodes["ci_type"] == "people",
                "actual_supply_road_people",
            ]
        )
        == 1
    )
    assert (
        np.all(
            network_calcs.network.nodes.loc[
                network_calcs.network.nodes["ci_type"] == "people",
                "actual_supply_healthcare_people",
            ]
        )
        == 1
    )


def test_cascade_simple(network_calcs_source_fail):
    """Test simple cascade without friction surface"""
    network_calcs_source_fail.network.initialize_capacity(
        network_calcs_source_fail.dep_table
    )
    network_calcs_source_fail.network.initialize_supply(
        network_calcs_source_fail.dep_table
    )
    network_calcs_source_fail.setup_dependencies()

    assert (
        np.all(
            network_calcs_source_fail.network.nodes.loc[
                network_calcs_source_fail.network.nodes["ci_type"] == "people",
                "actual_supply_road_people",
            ]
        )
        == 1
    )
    assert (
        np.all(
            network_calcs_source_fail.network.nodes.loc[
                network_calcs_source_fail.network.nodes["ci_type"] == "people",
                "actual_supply_healthcare_people",
            ]
        )
        == 1
    )

    network_calcs_source_fail.cascade(
        initial=False, friction_surf=None, rerouting=False
    )

    # Verify cascade completed
    assert (
        np.all(
            network_calcs_source_fail.network.nodes.loc[
                network_calcs_source_fail.network.nodes["ci_type"] == "people",
                "actual_supply_road_people",
            ]
        )
        == 1
    )
    assert (
        np.all(
            network_calcs_source_fail.network.nodes.loc[
                network_calcs_source_fail.network.nodes["ci_type"] == "people",
                "actual_supply_healthcare_people",
            ]
        )
        == 0
    )


@pytest.mark.skip(reason="powercap_from_clusters method not implemented")
def test_cascade_with_dependencies():
    """Test cascade with complete dependency setup"""
    # Create a more complex network
    nodes = gpd.GeoDataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "orig_id": [0, 1, 2, 3, 4],
            "ci_type": ["healthcare", "healthcare", "road", "road", "people"],
            "func_tot": [1, 1, 1, 1, 1],
            "geometry": [
                Point(0, 0),
                Point(1, 1),
                Point(2, 2),
                Point(3, 3),
                Point(4, 4),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    edges = gpd.GeoDataFrame(
        {
            "from_id": [0, 1, 2, 3],
            "to_id": [1, 2, 3, 4],
            "id": [0, 1, 2, 3],
            "orig_id": [0, 1, 2, 3],
            "osm_id": [100, 101, 102, 103],
            "ci_type": ["healthcare", "road", "road", "road"],
            "func_tot": [1, 1, 1, 1],
            "distance": [157200, 157200, 157200, 157200],
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)]),
                LineString([(2, 2), (3, 3)]),
                LineString([(3, 3), (4, 4)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    network = Network(edges=edges, nodes=nodes)

    dep_table = pd.DataFrame(
        {
            "source": ["healthcare", "road"],
            "target": ["road", "people"],
            "type_I": ["functional", "enduser"],
            "via_link": ["road", "road"],
            "link_condition": ["distance", "distance"],
            "thresh_dist": [np.inf, np.inf],
            "bidir_link": [False, False],
            "access_cnstr": [False, False],
            "n_links": [1, 1],
            "thresh_func": [0.5, 0.5],
        }
    )

    nc = NetworkCalcs(network=network, dep_table=dep_table)
    nc.initialize_base_state()

    # Set a failure
    nc.network.nodes.loc[1, "func_tot"] = 0

    # Run cascade
    nc.cascade(
        p_source="healthcare",
        p_sink="road",
        source_var="capacity",
        demand_var="demand",
        initial=False,
        friction_surf=None,
        rerouting=False,
    )

    # Verify cascade completed and network was updated
    assert nc.network is not None
    assert "func_tot" in nc.network.nodes.columns


@pytest.mark.skip(reason="powercap_from_clusters method not implemented")
def test_cascade_multiple_iterations():
    """Test cascade that requires multiple iterations"""
    nodes = gpd.GeoDataFrame(
        {
            "id": [0, 1, 2],
            "orig_id": [0, 1, 2],
            "ci_type": ["healthcare", "road", "road"],
            "func_tot": [1, 1, 1],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    edges = gpd.GeoDataFrame(
        {
            "from_id": [0, 1],
            "to_id": [1, 2],
            "id": [0, 1],
            "orig_id": [0, 1],
            "osm_id": [100, 101],
            "ci_type": ["road", "road"],
            "func_tot": [1, 1],
            "distance": [157200, 157200],
            "geometry": [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    network = Network(edges=edges, nodes=nodes)

    dep_table = pd.DataFrame(
        {
            "source": ["healthcare"],
            "target": ["road"],
            "type_I": ["functional"],
            "via_link": ["road"],
            "link_condition": ["distance"],
            "thresh_dist": [np.inf],
            "bidir_link": [False],
            "access_cnstr": [False],
            "n_links": [1],
            "thresh_func": [0.5],
        }
    )

    nc = NetworkCalcs(network=network, dep_table=dep_table)
    nc.initialize_base_state()

    nc.cascade(
        p_source="healthcare",
        p_sink="road",
        source_var="capacity",
        demand_var="demand",
        initial=True,
    )

    assert nc.network is not None


def test_initial_cascade_with_setup_dependencies(network_with_ci_types):
    """Test that initial cascade works correctly even when setup_dependencies is called first"""
    import numpy as np

    # Create a dependency table with enduser dependencies
    dep_table = pd.DataFrame(
        [
            {
                "Dep": "dependency_healthcare_people",
                "source": "healthcare",
                "target": "people",
                "n_links": 1,
                "access_cnstr": True,
                "type_I": "enduser",
                "type_II": "logical",
                "thresh_func": 1,
                "thresh_dist": 50000,
                "thresh_dur": 90,
                "conditions": None,
                "link_condition": "distance",
                "via_link": "road",
                "bidir_link": True,
            }
        ]
    )

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
    people_nodes = nw_calc.graph.vs.select(ci_type="people")
    for node in people_nodes:
        # All people should have undisrupted access in initial state
        # (assuming they're within distance threshold)
        if "access_state_healthcare_people" in node.attributes():
            assert node["access_state_healthcare_people"] in [
                "access undisrupted",
                "no base access",
            ], f"Node {node.index} has incorrect access state: {node['access_state_healthcare_people']}"
