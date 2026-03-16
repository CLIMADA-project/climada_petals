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

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import igraph as ig
from climada_petals.engine.networks.test.fixtures_test_networks import *
from climada.util.constants import ONE_LAT_KM


def test_graph_calcs_init(network_with_ci_types):
    """Test GraphCalcs initialization"""
    nw_calcs_mock = type("obj", (object,), {"network": network_with_ci_types})()
    gc = GraphCalcs(network=network_with_ci_types, directed=True)

    assert gc.network == network_with_ci_types
    assert gc.directed is True
    assert gc._graph is None


def test_graph_calcs_build_graph(graph_calcs):
    """Test building graph from network"""
    graph = graph_calcs.build_graph()

    assert isinstance(graph, ig.Graph)
    assert graph.vcount() == 5
    assert graph.ecount() == 4


def test_graph_calcs_graph_property_lazy_load(graph_calcs):
    """Test that graph property lazy loads the graph"""
    assert graph_calcs._graph is None

    graph = graph_calcs.graph

    assert graph is not None
    assert isinstance(graph, ig.Graph)


def test_graph_calcs_full_reset(graph_calcs):
    """Test invalidating cached graph"""
    _ = graph_calcs.graph  # Load graph
    assert graph_calcs._graph is not None

    graph_calcs.full_reset()

    assert graph_calcs._graph is None


def test_filter_vertices_single_attr(graph_calcs_with_remote_node):
    """Test filtering vertices by single attribute"""
    graph_calcs_with_remote_node.build_graph()

    df_vs = GraphCalcs._filter_vertices(
        graph_calcs_with_remote_node.graph, {"ci_type": "healthcare"}
    )

    assert len(df_vs) == 2
    assert all(df_vs["ci_type"] == "healthcare")


def test_filter_vertices_multiple_attrs(graph_calcs_with_remote_node):
    """Test filtering vertices by multiple attributes"""
    graph_calcs_with_remote_node.build_graph()

    df_vs = GraphCalcs._filter_vertices(
        graph_calcs_with_remote_node.graph, {"ci_type": "healthcare", "func_tot": 1}
    )

    assert len(df_vs) == 2
    assert all(df_vs["ci_type"] == "healthcare")
    assert all(df_vs["func_tot"] == 1)


def test_filter_edges_by_ci_type(graph_calcs):
    """Test filtering edges by CI type"""
    graph_calcs.build_graph()

    df_es_match = GraphCalcs._filter_edges(graph_calcs.graph, {"ci_type": "road"})
    df_es_not = GraphCalcs._filter_edges(graph_calcs.graph, {"ci_type": "river"})

    assert len(df_es_match) == 4
    assert len(df_es_not) == 0


def test_get_subgraph2graph_vsdict(graph_calcs):
    """Test vertex mapping from subgraph to graph"""
    graph_calcs.build_graph()
    graph = graph_calcs.graph

    # Create a subgraph with all vertices
    subgraph = graph.induced_subgraph(range(graph.vcount()))
    subgraph.vs["orig_id"] = [1, 0, 3, 2, 4]

    mapping = GraphCalcs._get_subgraph2graph_vsdict(graph, subgraph)

    assert isinstance(mapping, dict)
    assert mapping == {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}


def test_get_subgraph2graph_esdict(graph_calcs):
    """Test edge mapping from subgraph to graph"""
    graph_calcs.build_graph()
    graph = graph_calcs.graph

    # Create a subgraph with all vertices
    subgraph = graph.induced_subgraph(range(graph.vcount()))
    subgraph.es["orig_id"] = [1, 2, 0, 3]

    mapping = GraphCalcs._get_subgraph2graph_esdict(graph, subgraph)

    assert isinstance(mapping, dict)
    assert mapping == {0: 1, 1: 2, 2: 0, 3: 3}


def test_select_closest_k_basic():
    """Test selecting k nearest neighbors"""
    # Create source and target node GeoDataFrames
    gdf_vs_target = gpd.GeoDataFrame(
        {"id": [0, 1], "geometry": [Point(0, 0), Point(3, 3)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    gdf_vs_source = gpd.GeoDataFrame(
        {"id": [2, 3], "geometry": [Point(1, 1), Point(2, 2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    v_ids_source, v_ids_target = GraphCalcs._select_closest_k(
        gdf_vs_source,
        gdf_vs_target,
        dist_thresh=np.inf,
        crs=gdf_vs_source.crs,
        bidir=False,
        k=1,
    )

    assert len(v_ids_source) > 0
    assert len(v_ids_source) == len(v_ids_target)
    np.testing.assert_array_equal(v_ids_target, [0, 1])
    np.testing.assert_array_equal(v_ids_source, [2, 3])


def test_select_closest_k_dist():
    """Test selecting k nearest neighbors with distance threshold"""
    # Create source and target node GeoDataFrames
    gdf_vs_target = gpd.GeoDataFrame(
        {"id": [0, 1], "geometry": [Point(0, 0), Point(6, 6)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    gdf_vs_source = gpd.GeoDataFrame(
        {"id": [2, 3], "geometry": [Point(1, 1), Point(2, 2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    dist_th = 2 * (ONE_LAT_KM * 1000)
    v_ids_source, v_ids_target = GraphCalcs._select_closest_k(
        gdf_vs_source,
        gdf_vs_target,
        dist_thresh=dist_th,
        crs=gdf_vs_source.crs,
        bidir=False,
        k=1,
    )

    assert len(v_ids_source) > 0
    assert len(v_ids_source) == len(v_ids_target)
    np.testing.assert_array_equal(v_ids_target, [0])
    np.testing.assert_array_equal(v_ids_source, [2])


def test_select_closest_k_projected_crs():
    """Test _select_closest_k with projected CRS: threshold in metres, no conversion."""
    gdf_vs_target = gpd.GeoDataFrame(
        {"id": [0], "geometry": [Point(500000, 5000000)]},
        geometry="geometry",
        crs="EPSG:32632",
    )
    gdf_vs_source = gpd.GeoDataFrame(
        {"id": [1, 2], "geometry": [Point(500100, 5000000), Point(501000, 5000000)]},
        geometry="geometry",
        crs="EPSG:32632",
    )
    # 500 m threshold — source 1 at 100 m matches, source 2 at 1000 m does not
    v_ids_source, v_ids_target = GraphCalcs._select_closest_k(
        gdf_vs_source,
        gdf_vs_target,
        dist_thresh=500,
        crs=gdf_vs_source.crs,
        bidir=False,
        k=1,
    )
    assert len(v_ids_source) == 1
    np.testing.assert_array_equal(v_ids_source, [1])
    np.testing.assert_array_equal(v_ids_target, [0])


def test_select_closest_k_projected_crs_below_thresh():
    """Test _select_closest_k with projected CRS when threshold is too tight."""
    gdf_vs_target = gpd.GeoDataFrame(
        {"id": [0], "geometry": [Point(500000, 5000000)]},
        geometry="geometry",
        crs="EPSG:32632",
    )
    gdf_vs_source = gpd.GeoDataFrame(
        {"id": [1, 2], "geometry": [Point(500100, 5000000), Point(501000, 5000000)]},
        geometry="geometry",
        crs="EPSG:32632",
    )
    # 50 m threshold — closest source is 100 m away → no match
    v_ids_source, v_ids_target = GraphCalcs._select_closest_k(
        gdf_vs_source,
        gdf_vs_target,
        dist_thresh=50,
        crs=gdf_vs_source.crs,
        bidir=False,
        k=1,
    )
    assert len(v_ids_source) == 0
    assert len(v_ids_target) == 0


def test_select_closest_k_geographic_no_auto_convert():
    """Test _select_closest_k with geographic CRS and dist_auto_convert=False.

    When auto-conversion is disabled, the threshold is treated as degrees
    (the native unit of the geographic CRS).
    """
    gdf_vs_target = gpd.GeoDataFrame(
        {"id": [0, 1], "geometry": [Point(0, 0), Point(6, 6)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    gdf_vs_source = gpd.GeoDataFrame(
        {"id": [2, 3], "geometry": [Point(1, 1), Point(2, 2)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    # threshold = 2 degrees (no auto-conversion)
    # Target(0,0) → Source(1,1) at Euclidean ~1.41° < 2° → match
    # Target(6,6) → Source(2,2) at Euclidean ~5.66° > 2° → no match
    v_ids_source, v_ids_target = GraphCalcs._select_closest_k(
        gdf_vs_source,
        gdf_vs_target,
        dist_thresh=2,
        crs=gdf_vs_source.crs,
        bidir=False,
        k=1,
        dist_auto_convert=False,
    )
    assert len(v_ids_source) == 1
    np.testing.assert_array_equal(v_ids_source, [2])
    np.testing.assert_array_equal(v_ids_target, [0])


def test_funcstates_sum(graph_calcs):
    """Test summing functional states"""
    graph_calcs.build_graph()

    v_sum, e_sum = graph_calcs.funcstates_sum()

    assert isinstance(v_sum, (int, float))
    assert isinstance(e_sum, (int, float))
    assert v_sum == 5
    assert e_sum == 4


def test_funcstates_sum_with_failures(graph_calcs):
    """Test summing functional states with some failures"""
    graph_calcs.build_graph()

    # Set some vertices to failed state
    graph_calcs.graph.vs[1]["func_tot"] = 0
    graph_calcs.graph.es[0]["func_tot"] = 0

    v_sum, e_sum = graph_calcs.funcstates_sum()

    # Should reflect the failures
    assert v_sum == 4
    assert e_sum == 3


def test_create_subgraph_filter(graph_calcs_with_remote_node):
    """Test creating subgraph with filtered vertices"""
    graph_calcs_with_remote_node.build_graph()

    source_attrs = {"ci_type": "healthcare"}
    target_attrs = {"ci_type": "people"}
    via_attrs = {"ci_type": "road"}

    subgraph = graph_calcs_with_remote_node._create_subgraph(
        source_attrs, target_attrs, via_attrs
    )

    assert isinstance(subgraph, ig.Graph)
    assert (
        subgraph.vcount() == 6
    )  # 3 func road node + 1 people node + 2 healthcare nodes
    assert subgraph.ecount() == 5  # 5 edge func between func road nodes and healthcare
    assert (
        set(subgraph.vs["ci_type"]).difference({"healthcare", "road", "people"})
        == set()
    )
    assert set(subgraph.es["ci_type"]).difference({"road"}) == set()


def test_create_subgraph_filter_source(graph_calcs_with_source_fail):
    """Test creating subgraph with filtered vertices"""
    graph_calcs_with_source_fail.build_graph()

    source_attrs = {"ci_type": "healthcare", "func_tot": 1}
    target_attrs = {"ci_type": "people"}
    via_attrs = {"ci_type": "road"}

    subgraph = graph_calcs_with_source_fail._create_subgraph(
        source_attrs, target_attrs, via_attrs
    )

    assert isinstance(subgraph, ig.Graph)
    assert subgraph.vcount() == 4  # 3 func road node + 1 people node
    assert subgraph.ecount() == 3  # 3 edge func between func road
    # assert set(subgraph.vs.select(ci_type='healthcare')['func_tot']) == set()
    assert set(subgraph.vs["ci_type"]).difference({"road", "people"}) == set()
    assert set(subgraph.es["ci_type"]).difference({"road"}) == set()


def test_create_subgraph_filter_via(graph_calcs_with_edge_ci_fail):
    """Test creating subgraph with filtered vertices"""
    graph_calcs_with_edge_ci_fail.build_graph()

    source_attrs = {"ci_type": "road"}
    target_attrs = {"ci_type": "healthcare"}
    via_attrs = {"ci_type": "road", "func_tot": 1}

    subgraph = graph_calcs_with_edge_ci_fail._create_subgraph(
        source_attrs, target_attrs, via_attrs
    )

    assert isinstance(subgraph, ig.Graph)
    assert subgraph.vcount() == 5  # 3 func road node + 2 healthcare nodes
    assert subgraph.ecount() == 3  # 3 edge func between func road nodes and healthcare
    assert set(subgraph.es.select(ci_type="road")["func_tot"]) == {1}
    assert set(subgraph.vs["ci_type"]).difference({"healthcare", "road"}) == set()
    assert set(subgraph.es["ci_type"]).difference({"road"}) == set()


def test_link_vertices_edgecond(graph_calcs):
    """Test linking vertices based on edge conditions"""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    # This should add edges based on condition
    graph_calcs.link_vertices_edgecond(
        target_attrs={"ci_type": "healthcare"},
        edge_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_road_healthcare"},
    )
    dep_edge = graph_calcs.graph.es.select(ci_type="dependency_road_healthcare")
    sources = [e.source for e in dep_edge]
    targets = [e.target for e in dep_edge]
    # Verify that method completes without error
    assert len(dep_edge) > 0
    assert all((src["ci_type"] == "road" for src in graph_calcs.graph.vs[sources]))
    assert all(
        (tgt["ci_type"] == "healthcare" for tgt in graph_calcs.graph.vs[targets])
    )


def test_link_vertices_edgecond_empty_target(graph_calcs):
    """No edges created when target filter matches no vertices."""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.link_vertices_edgecond(
        target_attrs={"ci_type": "nonexistent_type"},
        edge_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dep_road_none"},
    )

    assert graph_calcs.graph.ecount() == initial_edge_count
    assert "dep_road_none" not in graph_calcs.graph.es["ci_type"]


def test_link_clusters_no_clusters(graph_calcs):
    """Test link_clusters when network is already connected"""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.link_clusters(
        dist_thresh=np.inf, link_attrs={"ci_type": "cluster_link"}
    )

    # Should not add edges if already connected

    assert "cluster_link" not in graph_calcs.graph.es["ci_type"]


def test_link_clusters_with_threshold_low(graph_calcs_with_remote_node):
    """Test link_clusters with distance threshold"""
    graph_calcs_with_remote_node.build_graph()
    initial_edge_count = graph_calcs_with_remote_node.graph.ecount()
    graph_calcs_with_remote_node.link_clusters(
        dist_thresh=1000, link_attrs={"ci_type": "cluster_link"}
    )

    # Verify method completes
    assert graph_calcs_with_remote_node.graph.ecount() == initial_edge_count
    assert "cluster_link" not in graph_calcs_with_remote_node.graph.es["ci_type"]


def test_link_clusters_with_threshold_high(graph_calcs_with_remote_node_missing_edge):
    """Test link_clusters with distance threshold"""
    graph_calcs_with_remote_node_missing_edge.build_graph()
    initial_edge_count = graph_calcs_with_remote_node_missing_edge.graph.ecount()
    graph_calcs_with_remote_node_missing_edge.link_clusters(
        dist_thresh=np.inf,
        link_attrs={"ci_type": "cluster_link"},
    )

    # Verify method completes
    assert (
        graph_calcs_with_remote_node_missing_edge.graph.ecount()
        == initial_edge_count + 1
    )
    assert (
        "cluster_link" in graph_calcs_with_remote_node_missing_edge.graph.es["ci_type"]
    )
    assert graph_calcs_with_remote_node_missing_edge.graph.es.select(
        ci_type="cluster_link"
    )["geometry"][0].bounds[0:2] == (4, 4)
    assert graph_calcs_with_remote_node_missing_edge.graph.es.select(
        ci_type="cluster_link"
    )["geometry"][0].bounds[2:4] == (4, 50)


def test_link_clusters_projected_crs_high_thresh(
    graph_calcs_projected_disconnected,
):
    """Test link_clusters with projected CRS and sufficient threshold.

    Node 3 at (501000, 5000000) is 800 m from node 2 at (500200, 5000000).
    A 1000 m threshold should link the clusters.
    """
    graph_calcs_projected_disconnected.build_graph()
    initial_edge_count = graph_calcs_projected_disconnected.graph.ecount()

    graph_calcs_projected_disconnected.link_clusters(
        dist_thresh=1000,
        link_attrs={"ci_type": "cluster_link"},
    )

    assert graph_calcs_projected_disconnected.graph.ecount() == initial_edge_count + 1
    assert "cluster_link" in graph_calcs_projected_disconnected.graph.es["ci_type"]
    new_edge = graph_calcs_projected_disconnected.graph.es.select(
        ci_type="cluster_link"
    )[0]
    assert new_edge["distance"] == pytest.approx(800, abs=1)


def test_link_clusters_projected_crs_low_thresh(
    graph_calcs_projected_disconnected,
):
    """Test link_clusters with projected CRS and insufficient threshold.

    Closest gap is 800 m; a 500 m threshold should not link.
    """
    graph_calcs_projected_disconnected.build_graph()
    initial_edge_count = graph_calcs_projected_disconnected.graph.ecount()

    graph_calcs_projected_disconnected.link_clusters(
        dist_thresh=500,
        link_attrs={"ci_type": "cluster_link"},
    )

    assert graph_calcs_projected_disconnected.graph.ecount() == initial_edge_count
    assert "cluster_link" not in graph_calcs_projected_disconnected.graph.es["ci_type"]


def test_link_clusters_geographic_no_auto_convert(
    graph_calcs_with_remote_node_missing_edge,
):
    """Test link_clusters with geographic CRS and dist_auto_convert=False.

    Remote node at (4, 50) is ~46 degrees from nearest connected node (4, 4).
    With auto-conversion disabled, threshold is in degrees.
    """
    graph_calcs_with_remote_node_missing_edge.build_graph()
    initial_edge_count = graph_calcs_with_remote_node_missing_edge.graph.ecount()

    # 50 degrees > 46 degrees → should link
    graph_calcs_with_remote_node_missing_edge.link_clusters(
        dist_thresh=50,
        dist_auto_convert=False,
        link_attrs={"ci_type": "cluster_link"},
    )

    assert (
        graph_calcs_with_remote_node_missing_edge.graph.ecount()
        == initial_edge_count + 1
    )
    assert (
        "cluster_link" in graph_calcs_with_remote_node_missing_edge.graph.es["ci_type"]
    )


def test_link_clusters_geographic_no_auto_convert_low_thresh(
    graph_calcs_with_remote_node_missing_edge,
):
    """Test link_clusters with geographic CRS and dist_auto_convert=False.

    10 degrees < 46 degrees → should not link.
    """
    graph_calcs_with_remote_node_missing_edge.build_graph()
    initial_edge_count = graph_calcs_with_remote_node_missing_edge.graph.ecount()

    graph_calcs_with_remote_node_missing_edge.link_clusters(
        dist_thresh=10,
        dist_auto_convert=False,
        link_attrs={"ci_type": "cluster_link"},
    )

    assert (
        graph_calcs_with_remote_node_missing_edge.graph.ecount() == initial_edge_count
    )
    assert (
        "cluster_link"
        not in graph_calcs_with_remote_node_missing_edge.graph.es["ci_type"]
    )


def test_link_vertices_closest_k_projected_crs(
    graph_calcs_projected_disconnected,
):
    """Test link_vertices_closest_k with projected CRS.

    People node at (500000, 5000000), closest road node at (500100, 5000000)
    = 100 m.  A 500 m threshold should link them.
    """
    graph_calcs_projected_disconnected.build_graph()
    initial_edge_count = graph_calcs_projected_disconnected.graph.ecount()

    graph_calcs_projected_disconnected.link_vertices_closest_k(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "people"},
        link_attrs={"ci_type": "link_road_people"},
        dist_thresh=500,
        bidir=False,
        k=1,
    )

    assert graph_calcs_projected_disconnected.graph.ecount() == initial_edge_count + 1
    assert "link_road_people" in graph_calcs_projected_disconnected.graph.es["ci_type"]


def test_link_vertices_closest_k_projected_crs_low_thresh(
    graph_calcs_projected_disconnected,
):
    """Test link_vertices_closest_k with projected CRS and tight threshold.

    Closest road node is 100 m away; 50 m threshold should not link.
    """
    graph_calcs_projected_disconnected.build_graph()
    initial_edge_count = graph_calcs_projected_disconnected.graph.ecount()

    graph_calcs_projected_disconnected.link_vertices_closest_k(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "people"},
        link_attrs={"ci_type": "link_road_people"},
        dist_thresh=50,
        bidir=False,
        k=1,
    )

    assert graph_calcs_projected_disconnected.graph.ecount() == initial_edge_count
    assert (
        "link_road_people" not in graph_calcs_projected_disconnected.graph.es["ci_type"]
    )


def test_link_vertices_closest_k_low_thresh(graph_calcs):
    """Test linking vertices by k-nearest neighbors"""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.link_vertices_closest_k(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "people"},
        link_attrs={"ci_type": "link_road_people"},
        dist_thresh=1000,
        bidir=False,
        k=1,
    )

    # Should not add any edge
    assert graph_calcs.graph.ecount() == initial_edge_count
    assert "link_road_people" not in graph_calcs.graph.es["ci_type"]


def test_link_vertices_closest_k_empty_source(graph_calcs):
    """No edges created when source filter matches no vertices."""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.link_vertices_closest_k(
        source_attrs={"ci_type": "nonexistent_type"},
        target_attrs={"ci_type": "people"},
        link_attrs={"ci_type": "link_none_people"},
        dist_thresh=np.inf,
        bidir=False,
        k=1,
    )

    assert graph_calcs.graph.ecount() == initial_edge_count
    assert "link_none_people" not in graph_calcs.graph.es["ci_type"]


def test_link_vertices_closest_k_empty_target(graph_calcs):
    """No edges created when target filter matches no vertices."""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.link_vertices_closest_k(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "nonexistent_type"},
        link_attrs={"ci_type": "link_road_none"},
        dist_thresh=np.inf,
        bidir=False,
        k=1,
    )

    assert graph_calcs.graph.ecount() == initial_edge_count
    assert "link_road_none" not in graph_calcs.graph.es["ci_type"]


def test_link_vertices_closest_k_empty_both(graph_calcs):
    """No edges created when both source and target filters match nothing."""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.link_vertices_closest_k(
        source_attrs={"ci_type": "fake_source"},
        target_attrs={"ci_type": "fake_target"},
        link_attrs={"ci_type": "link_fake"},
        dist_thresh=np.inf,
        bidir=False,
        k=1,
    )

    assert graph_calcs.graph.ecount() == initial_edge_count
    assert "link_fake" not in graph_calcs.graph.es["ci_type"]


def test_link_vertices_closest_k_high_thresh(graph_calcs):
    """Test linking vertices by k-nearest neighbors"""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.link_vertices_closest_k(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "people"},
        link_attrs={"ci_type": "link_road_people"},
        dist_thresh=np.inf,
        bidir=False,
        k=1,
    )

    # Should add at least one edge
    assert graph_calcs.graph.ecount() == initial_edge_count + 1
    assert "link_road_people" in graph_calcs.graph.es["ci_type"]


def test_link_vertices_closest_k_bidir(graph_calcs):
    """Test linking vertices by k-nearest neighbors"""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.link_vertices_closest_k(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "people"},
        link_attrs={"ci_type": "link_road_people"},
        dist_thresh=np.inf,
        bidir=True,
        k=1,
    )

    # Should add at least one edge
    assert graph_calcs.graph.ecount() == initial_edge_count + 2
    assert "link_road_people" in graph_calcs.graph.es["ci_type"]


def test_link_vertices_shortest_paths_single(graph_calcs):
    """Test linking via shortest paths with k=1"""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.link_vertices_shortest_paths(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "shortest_path_link"},
        dist_thresh=10e6,
        criterion="distance",
        k=1,
        bidir=False,
    )

    # Should add edges based on shortest paths
    assert graph_calcs.graph.ecount() == initial_edge_count + 1
    assert "shortest_path_link" in graph_calcs.graph.es["ci_type"]


def test_link_vertices_shortest_paths_multiple(graph_calcs):
    """Test linking via shortest paths with k>1"""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.link_vertices_shortest_paths(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "healthcare"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "multi_path_link"},
        dist_thresh=10e6,
        criterion="distance",
        k=2,
        bidir=False,
    )

    assert graph_calcs.graph.ecount() > initial_edge_count


def test_link_vertices_shortest_paths_k1_selects_closest(graph_calcs_with_remote_node):
    """k=1 selects only the single closest source per target."""
    graph_calcs_with_remote_node.build_graph()

    graph_calcs_with_remote_node.link_vertices_shortest_paths(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dep_health_people"},
        dist_thresh=10e6,
        criterion="distance",
        k=1,
        bidir=False,
    )

    new_edges = [
        e
        for e in graph_calcs_with_remote_node.graph.es
        if e["ci_type"] == "dep_health_people"
    ]
    assert len(new_edges) == 1
    # closest healthcare (node 4, path ~628800 m) must be chosen
    # over the remote one (node 5, path ~7314400 m)
    assert new_edges[0]["distance"] < 7e6


def test_link_vertices_shortest_paths_k2_selects_two(graph_calcs_with_remote_node):
    """k=2 selects up to 2 closest sources per target."""
    graph_calcs_with_remote_node.build_graph()

    graph_calcs_with_remote_node.link_vertices_shortest_paths(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dep_health_people"},
        dist_thresh=10e6,
        criterion="distance",
        k=2,
        bidir=False,
    )

    new_edges = [
        e
        for e in graph_calcs_with_remote_node.graph.es
        if e["ci_type"] == "dep_health_people"
    ]
    assert len(new_edges) == 2
    assert all(e["distance"] < 10e6 for e in new_edges)


def test_link_vertices_shortest_paths_k_respects_dist_thresh(
    graph_calcs_with_remote_node,
):
    """k=2 but dist_thresh excludes the farther source."""
    graph_calcs_with_remote_node.build_graph()

    graph_calcs_with_remote_node.link_vertices_shortest_paths(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dep_health_people"},
        dist_thresh=1e6,  # excludes remote healthcare (~7.3 M m away)
        criterion="distance",
        k=2,
        bidir=False,
    )

    new_edges = [
        e
        for e in graph_calcs_with_remote_node.graph.es
        if e["ci_type"] == "dep_health_people"
    ]
    assert len(new_edges) == 1
    assert new_edges[0]["distance"] < 1e6


def test_link_vertices_shortest_paths_k_exceeds_sources(
    graph_calcs_with_remote_node,
):
    """k larger than available sources links all sources within threshold."""
    graph_calcs_with_remote_node.build_graph()

    graph_calcs_with_remote_node.link_vertices_shortest_paths(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dep_health_people"},
        dist_thresh=10e6,
        criterion="distance",
        k=10,  # much larger than the 2 available sources
        bidir=False,
    )

    new_edges = [
        e
        for e in graph_calcs_with_remote_node.graph.es
        if e["ci_type"] == "dep_health_people"
    ]
    assert len(new_edges) == 2  # capped at the 2 available healthcare nodes


def test_link_vertices_shortest_paths_empty_source(graph_calcs):
    """No edges created when source filter matches no vertices."""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.link_vertices_shortest_paths(
        source_attrs={"ci_type": "nonexistent_type"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "sp_none_people"},
        dist_thresh=10e6,
        criterion="distance",
        k=1,
        bidir=False,
    )

    assert graph_calcs.graph.ecount() == initial_edge_count
    assert "sp_none_people" not in graph_calcs.graph.es["ci_type"]


def test_link_vertices_shortest_paths_empty_target(graph_calcs):
    """No edges created when target filter matches no vertices."""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.link_vertices_shortest_paths(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "nonexistent_type"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "sp_road_none"},
        dist_thresh=10e6,
        criterion="distance",
        k=1,
        bidir=False,
    )

    assert graph_calcs.graph.ecount() == initial_edge_count
    assert "sp_road_none" not in graph_calcs.graph.es["ci_type"]


def test_link_vertices_shortest_paths_empty_via(graph_calcs):
    """No edges created when via filter yields an empty subgraph."""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.link_vertices_shortest_paths(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "nonexistent_type"},
        link_attrs={"ci_type": "sp_no_via"},
        dist_thresh=10e6,
        criterion="distance",
        k=1,
        bidir=False,
    )

    assert graph_calcs.graph.ecount() == initial_edge_count
    assert "sp_no_via" not in graph_calcs.graph.es["ci_type"]


def test_link_vertices_friction_surf_basic(graph_calcs, monkeypatch):
    """Test friction-surface linking creates edges when duration is below threshold."""
    graph_calcs.build_graph()
    graph_calcs.friction_surf = object()
    initial_edge_count = graph_calcs.graph.ecount()

    monkeypatch.setattr(
        GraphCalcs,
        "_calc_friction",
        staticmethod(lambda edge_geoms, friction_surf: np.zeros(len(edge_geoms))),
    )

    graph_calcs.link_vertices_friction_surf(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "people"},
        link_attrs={"ci_type": "friction_link"},
        dur_thresh=10,
        dist_thresh=np.inf,
        k=1,
        bidir=False,
    )

    assert graph_calcs.graph.ecount() == initial_edge_count + 1
    assert "friction_link" in graph_calcs.graph.es["ci_type"]


def test_link_vertices_friction_surf_bidir(graph_calcs, monkeypatch):
    """Test friction-surface linking with bidir=True adds reverse links."""
    graph_calcs.build_graph()
    graph_calcs.friction_surf = object()
    initial_edge_count = graph_calcs.graph.ecount()

    monkeypatch.setattr(
        GraphCalcs,
        "_calc_friction",
        staticmethod(lambda edge_geoms, friction_surf: np.zeros(len(edge_geoms))),
    )

    graph_calcs.link_vertices_friction_surf(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "people"},
        link_attrs={"ci_type": "friction_link"},
        dur_thresh=10,
        dist_thresh=np.inf,
        k=1,
        bidir=True,
    )

    assert graph_calcs.graph.ecount() == initial_edge_count + 2
    assert "friction_link" in graph_calcs.graph.es["ci_type"]


def test_link_vertices_friction_surf_duration_threshold(graph_calcs, monkeypatch):
    """No links are created when all friction values exceed duration threshold."""
    graph_calcs.build_graph()
    graph_calcs.friction_surf = object()
    initial_edge_count = graph_calcs.graph.ecount()

    monkeypatch.setattr(
        GraphCalcs,
        "_calc_friction",
        staticmethod(lambda edge_geoms, friction_surf: np.full(len(edge_geoms), 1e9)),
    )

    graph_calcs.link_vertices_friction_surf(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "people"},
        link_attrs={"ci_type": "friction_link"},
        dur_thresh=10,
        dist_thresh=np.inf,
        k=1,
        bidir=False,
    )

    assert graph_calcs.graph.ecount() == initial_edge_count
    assert "friction_link" not in graph_calcs.graph.es["ci_type"]


def test_link_vertices_friction_surf_empty_source(graph_calcs, monkeypatch):
    """No edges created when source filter matches no vertices."""
    graph_calcs.build_graph()
    graph_calcs.friction_surf = object()
    initial_edge_count = graph_calcs.graph.ecount()

    monkeypatch.setattr(
        GraphCalcs,
        "_calc_friction",
        staticmethod(lambda edge_geoms, friction_surf: np.zeros(len(edge_geoms))),
    )

    graph_calcs.link_vertices_friction_surf(
        source_attrs={"ci_type": "nonexistent_type"},
        target_attrs={"ci_type": "people"},
        link_attrs={"ci_type": "friction_none_people"},
        dur_thresh=10,
        dist_thresh=np.inf,
        k=1,
        bidir=False,
    )

    assert graph_calcs.graph.ecount() == initial_edge_count
    assert "friction_none_people" not in graph_calcs.graph.es["ci_type"]


def test_link_vertices_friction_surf_empty_target(graph_calcs, monkeypatch):
    """No edges created when target filter matches no vertices."""
    graph_calcs.build_graph()
    graph_calcs.friction_surf = object()
    initial_edge_count = graph_calcs.graph.ecount()

    monkeypatch.setattr(
        GraphCalcs,
        "_calc_friction",
        staticmethod(lambda edge_geoms, friction_surf: np.zeros(len(edge_geoms))),
    )

    graph_calcs.link_vertices_friction_surf(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "nonexistent_type"},
        link_attrs={"ci_type": "friction_road_none"},
        dur_thresh=10,
        dist_thresh=np.inf,
        k=1,
        bidir=False,
    )

    assert graph_calcs.graph.ecount() == initial_edge_count
    assert "friction_road_none" not in graph_calcs.graph.es["ci_type"]


def test_link_vertices_friction_surf_no_friction_surf_raises(graph_calcs):
    """AttributeError is raised when friction_surf is None."""
    graph_calcs.build_graph()
    assert graph_calcs.friction_surf is None

    with pytest.raises(AttributeError):
        graph_calcs.link_vertices_friction_surf(
            source_attrs={"ci_type": "road"},
            target_attrs={"ci_type": "people"},
            link_attrs={"ci_type": "friction_link"},
            dur_thresh=10,
            dist_thresh=np.inf,
            k=1,
            bidir=False,
        )


def test_edges_from_vlists(graph_calcs_with_remote_node):
    """Test adding edges from vertex lists"""
    graph_calcs_with_remote_node.build_graph()
    initial_edge_count = graph_calcs_with_remote_node.graph.ecount()

    # Add edges between specific vertices
    v_ids_source = [1, 2, 4]
    v_ids_target = [3, 4, 1]

    graph_calcs_with_remote_node._edges_from_vlists(
        v_ids_source, v_ids_target, link_attrs={"ci_type": "test_link"}
    )
    sources = [
        e.source
        for e in graph_calcs_with_remote_node.graph.es
        if e["ci_type"] == "test_link"
    ]
    targets = [
        e.target
        for e in graph_calcs_with_remote_node.graph.es
        if e["ci_type"] == "test_link"
    ]
    # Should have added 3 edges
    assert graph_calcs_with_remote_node.graph.ecount() == initial_edge_count + 3
    assert "test_link" in graph_calcs_with_remote_node.graph.es["ci_type"]
    assert len(sources) == 3
    assert len(targets) == 3
    # Verify correct source-target pairs
    for i in range(len(sources)):
        assert sources[i] == v_ids_source[i]
        assert targets[i] == v_ids_target[i]


def test_edges_from_vlists_with_distance(graph_calcs):
    """Test adding edges with pre-calculated distances"""
    graph_calcs.build_graph()

    v_ids_source = [1]
    v_ids_target = [2]

    graph_calcs._edges_from_vlists(
        v_ids_source,
        v_ids_target,
        link_attrs={"ci_type": "test_link", "distance": [1000.0]},
    )

    # Check that distance was preserved
    new_edge = graph_calcs.graph.es[graph_calcs.graph.ecount() - 1]
    assert new_edge["distance"] == 1000.0


def test_calc_dependencies_distance(graph_calcs):
    """Test calculating dependencies with distance criterion"""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dep_link"},
        link_condition="distance",
        dist_thresh=1e7,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    assert "dep_link" in graph_calcs.graph.es["ci_type"]
    assert graph_calcs.graph.ecount() == initial_edge_count + 1


def test_calc_dependencies_edgecond(graph_calcs):
    """Test calculating dependencies with edge condition"""
    graph_calcs.build_graph()
    initial_edge_count = graph_calcs.graph.ecount()

    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "people"},
        via_attrs={},
        link_attrs={"ci_type": "edge_cond_link"},
        link_condition="edgecond",
        dist_thresh=None,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    # Should add edges based on edge conditions
    assert graph_calcs.graph.ecount() == initial_edge_count + 1
    assert "edge_cond_link" in graph_calcs.graph.es["ci_type"]


def test_calc_dependencies_distance_via_fail(graph_calcs_with_edge_ci_fail):
    """Test calculating dependencies with distance criterion"""
    graph_calcs_with_edge_ci_fail.build_graph()
    initial_edge_count = graph_calcs_with_edge_ci_fail.graph.ecount()

    graph_calcs_with_edge_ci_fail.calc_dependencies(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road", "func_tot": 1},
        link_attrs={"ci_type": "dep_link"},
        link_condition="distance",
        dist_thresh=1e7,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    assert "dep_link" not in graph_calcs_with_edge_ci_fail.graph.es["ci_type"]
    assert graph_calcs_with_edge_ci_fail.graph.ecount() == initial_edge_count


def test_check_access_basic_undisrupted(graph_calcs, dependency_table):
    """Test _check_access basic functionality"""
    graph_calcs.network.initialize_funcstates()
    graph_calcs.network.initialize_capacity(dependency_table)
    graph_calcs.network.initialize_supply(dependency_table)
    graph_calcs.build_graph()

    # Get first dependency row (road -> people)
    for _, row in dependency_table.loc[
        dependency_table["target"] == "people"
    ].iterrows():
        graph_calcs.calc_dependencies(
            source_attrs={"ci_type": row.source},
            target_attrs={"ci_type": row.target},
            via_attrs={"ci_type": "road"},
            link_attrs={"ci_type": f"dependency_{row.source}_{row.target}"},
            link_condition=row["link_condition"],
            dist_thresh=row["thresh_dist"],
            dur_thresh=np.inf,
            k=1,
            bidir_link=row["bidir_link"],
        )

        # Call _check_access
        graph_calcs._check_access(row, friction_surf=None, rerouting=False)

        # Verify access states are set on people nodes
        people_nodes = graph_calcs.graph.vs.select(ci_type="people")
        for node in people_nodes:
            assert f"access_state_{row.source}_{row.target}" in node.attributes()
            assert f"actual_supply_{row.source}_{row.target}" in node.attributes()
            assert (
                node[f"access_state_{row.source}_{row.target}"] == "access undisrupted"
            )


def test_check_access_with_rerouting_undisrupted(graph_calcs, dependency_table):
    """Test _check_access with rerouting enabled"""
    graph_calcs.network.initialize_funcstates()
    graph_calcs.network.initialize_capacity(dependency_table)
    graph_calcs.network.initialize_supply(dependency_table)
    graph_calcs.build_graph()

    for _, row in dependency_table.loc[
        dependency_table["target"] == "people"
    ].iterrows():
        graph_calcs.calc_dependencies(
            source_attrs={"ci_type": row.source},
            target_attrs={"ci_type": row.target},
            via_attrs={"ci_type": "road"},
            link_attrs={"ci_type": f"dependency_{row.source}_{row.target}"},
            link_condition=row["link_condition"],
            dist_thresh=row["thresh_dist"],
            dur_thresh=np.inf,
            k=1,
            bidir_link=row["bidir_link"],
        )

        # Call _check_access
        graph_calcs._check_access(row, friction_surf=None, rerouting=True)

        # Verify access states are set on people nodes
        people_nodes = graph_calcs.graph.vs.select(ci_type="people")
        for node in people_nodes:
            assert f"access_state_{row.source}_{row.target}" in node.attributes()
            assert f"actual_supply_{row.source}_{row.target}" in node.attributes()
            assert (
                node[f"access_state_{row.source}_{row.target}"] == "access undisrupted"
            )


def test_check_access_with_rerouting_failed_source(
    graph_calcs_with_remote_node, dependency_table
):
    """Test _check_access with rerouting enabled"""
    graph_calcs_with_remote_node.network.initialize_funcstates()
    graph_calcs_with_remote_node.network.initialize_capacity(dependency_table)
    graph_calcs_with_remote_node.network.initialize_supply(dependency_table)
    graph_calcs_with_remote_node.build_graph()

    row = dependency_table.iloc[1]

    graph_calcs_with_remote_node.calc_dependencies(
        source_attrs={"ci_type": row.source},
        target_attrs={"ci_type": row.target},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": f"dependency_{row.source}_{row.target}"},
        link_condition=row["link_condition"],
        dist_thresh=row["thresh_dist"],
        dur_thresh=np.inf,
        k=1,
        bidir_link=row["bidir_link"],
    )

    # fail ci to test rerouting
    healthcare = graph_calcs_with_remote_node.graph.vs.select(ci_type="healthcare")
    healthcare[0]["func_tot"] = 0

    # Call _check_access
    graph_calcs_with_remote_node._check_access(row, friction_surf=None, rerouting=True)
    # Verify access states are set on people nodes
    people_nodes = graph_calcs_with_remote_node.graph.vs.select(ci_type="people")
    for node in people_nodes:
        assert f"access_state_{row.source}_{row.target}" in node.attributes()
        assert f"actual_supply_{row.source}_{row.target}" in node.attributes()
        assert node[f"access_state_{row.source}_{row.target}"] == "access new source"
        assert node[f"actual_supply_{row.source}_{row.target}"] == 1


def test_check_access_with_rerouting_via_disrupted(
    graph_calcs_with_remote_node, dependency_table
):
    """Test _check_access with rerouting when via is failing"""
    graph_calcs_with_remote_node.network.initialize_funcstates()
    graph_calcs_with_remote_node.network.initialize_capacity(dependency_table)
    graph_calcs_with_remote_node.network.initialize_supply(dependency_table)
    graph_calcs_with_remote_node.build_graph()

    row = dependency_table.iloc[1]  # Use row 1 (healthcare->people enduser)

    graph_calcs_with_remote_node.calc_dependencies(
        source_attrs={"ci_type": row.source},
        target_attrs={"ci_type": row.target},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": f"dependency_{row.source}_{row.target}"},
        link_condition=row["link_condition"],
        dist_thresh=row["thresh_dist"],
        dur_thresh=np.inf,
        k=1,
        bidir_link=row["bidir_link"],
    )

    # Fail a road node to test rerouting
    graph_calcs_with_remote_node.graph.vs.select(ci_type="road")[0]["func_tot"] = 0
    graph_calcs_with_remote_node.graph.es.select(ci_type="road")[0]["func_tot"] = 0

    # Call _check_access
    graph_calcs_with_remote_node._check_access(row, friction_surf=None, rerouting=True)
    # Verify access states are set on people nodes
    people_nodes = graph_calcs_with_remote_node.graph.vs.select(ci_type="people")
    for node in people_nodes:
        assert f"access_state_{row.source}_{row.target}" in node.attributes()
        assert f"actual_supply_{row.source}_{row.target}" in node.attributes()
        # With rerouting enabled and just one road failed, access should still be possible
        assert node[f"access_state_{row.source}_{row.target}"] == "access disrupted via"


def test_check_access_with_rerouting_new_source(
    graph_calcs_with_remote_node, dependency_table
):
    """Test _check_access with rerouting when via is failing"""
    graph_calcs_with_remote_node.network.initialize_funcstates()
    graph_calcs_with_remote_node.network.initialize_capacity(dependency_table)
    graph_calcs_with_remote_node.network.initialize_supply(dependency_table)
    graph_calcs_with_remote_node.build_graph()

    row = dependency_table.iloc[1]  # Use row 1 (healthcare->people enduser)

    graph_calcs_with_remote_node.calc_dependencies(
        source_attrs={"ci_type": row.source},
        target_attrs={"ci_type": row.target},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": f"dependency_{row.source}_{row.target}"},
        link_condition=row["link_condition"],
        dist_thresh=row["thresh_dist"],
        dur_thresh=np.inf,
        k=1,
        bidir_link=row["bidir_link"],
    )

    # Fail a road node to test rerouting
    graph_calcs_with_remote_node.graph.vs[3]["func_tot"] = 0
    graph_calcs_with_remote_node.graph.es[3]["func_tot"] = 0

    # Call _check_access
    graph_calcs_with_remote_node._check_access(row, friction_surf=None, rerouting=True)
    # Verify access states are set on people nodes
    people_nodes = graph_calcs_with_remote_node.graph.vs.select(ci_type="people")
    for node in people_nodes:
        assert f"access_state_{row.source}_{row.target}" in node.attributes()
        assert f"actual_supply_{row.source}_{row.target}" in node.attributes()
        # With rerouting enabled and just one road failed, access should still be possible
        assert node[f"access_state_{row.source}_{row.target}"] == "access undisrupted"


def test_check_access_without_rerouting_via_disrupted(
    graph_calcs_with_remote_node, dependency_table
):
    """Test _check_access with rerouting when via is failing"""
    graph_calcs_with_remote_node.network.initialize_funcstates()
    graph_calcs_with_remote_node.network.initialize_capacity(dependency_table)
    graph_calcs_with_remote_node.network.initialize_supply(dependency_table)
    graph_calcs_with_remote_node.build_graph()

    row = dependency_table.iloc[1]  # Use row 1 (healthcare->people enduser)

    graph_calcs_with_remote_node.calc_dependencies(
        source_attrs={"ci_type": row.source},
        target_attrs={"ci_type": row.target},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": f"dependency_{row.source}_{row.target}"},
        link_condition=row["link_condition"],
        dist_thresh=row["thresh_dist"],
        dur_thresh=np.inf,
        k=1,
        bidir_link=row["bidir_link"],
    )

    # Fail a road node to test rerouting
    graph_calcs_with_remote_node.graph.vs[3]["func_tot"] = 0
    graph_calcs_with_remote_node.graph.es[3]["func_tot"] = 0

    # Call _check_access
    graph_calcs_with_remote_node._check_access(row, friction_surf=None, rerouting=False)
    # Verify access states are set on people nodes
    people_nodes = graph_calcs_with_remote_node.graph.vs.select(ci_type="people")
    for node in people_nodes:
        assert f"access_state_{row.source}_{row.target}" in node.attributes()
        assert f"actual_supply_{row.source}_{row.target}" in node.attributes()
        # Without rerouting enabled, access should be disrupted
        assert node[f"access_state_{row.source}_{row.target}"] == "access disrupted via"


def test_check_access_without_rerouting_failed_source(
    graph_calcs_with_remote_node, dependency_table
):
    """Test _check_access with rerouting enabled"""
    graph_calcs_with_remote_node.network.initialize_funcstates()
    graph_calcs_with_remote_node.network.initialize_capacity(dependency_table)
    graph_calcs_with_remote_node.network.initialize_supply(dependency_table)
    graph_calcs_with_remote_node.build_graph()

    row = dependency_table.iloc[1]

    graph_calcs_with_remote_node.calc_dependencies(
        source_attrs={"ci_type": row.source},
        target_attrs={"ci_type": row.target},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": f"dependency_{row.source}_{row.target}"},
        link_condition=row["link_condition"],
        dist_thresh=row["thresh_dist"],
        dur_thresh=np.inf,
        k=1,
        bidir_link=row["bidir_link"],
    )

    # fail ci to test rerouting
    healthcare = graph_calcs_with_remote_node.graph.vs.select(ci_type="healthcare")
    healthcare[0]["func_tot"] = 0

    # Call _check_access
    graph_calcs_with_remote_node._check_access(row, friction_surf=None, rerouting=False)
    # Verify access states are set on people nodes
    people_nodes = graph_calcs_with_remote_node.graph.vs.select(ci_type="people")
    for node in people_nodes:
        assert f"access_state_{row.source}_{row.target}" in node.attributes()
        assert f"actual_supply_{row.source}_{row.target}" in node.attributes()
        assert (
            node[f"access_state_{row.source}_{row.target}"] == "access disrupted source"
        )
        assert node[f"actual_supply_{row.source}_{row.target}"] == 0


def test_check_access_no_constraint(graph_calcs_with_remote_node, dependency_table):
    """Test _check_access with access constraints disabled"""
    graph_calcs_with_remote_node.network.initialize_funcstates()
    graph_calcs_with_remote_node.network.initialize_capacity(dependency_table)
    graph_calcs_with_remote_node.network.initialize_supply(dependency_table)
    graph_calcs_with_remote_node.build_graph()

    for _, row in dependency_table.loc[
        dependency_table["target"] == "people"
    ].iterrows():
        graph_calcs_with_remote_node.calc_dependencies(
            source_attrs={"ci_type": row.source},
            target_attrs={"ci_type": row.target},
            via_attrs={"ci_type": "road"},
            link_attrs={"ci_type": f"dependency_{row.source}_{row.target}"},
            link_condition=row["link_condition"],
            dist_thresh=row["thresh_dist"],
            dur_thresh=np.inf,
            k=1,
            bidir_link=row["bidir_link"],
        )
        # deactivate access constraints
        row["access_cnstr"] = False

        # Fail a road node to test access constraint being ignored
        graph_calcs_with_remote_node.graph.vs.select(ci_type="road")[1]["func_tot"] = 0
        graph_calcs_with_remote_node.graph.es.select(ci_type="road")[1]["func_tot"] = 0

        # Call _check_access
        graph_calcs_with_remote_node._check_access(
            row, friction_surf=None, rerouting=True
        )

        # Verify access states are set on people nodes
        people_nodes = graph_calcs_with_remote_node.graph.vs.select(ci_type="people")
        for node in people_nodes:
            assert f"access_state_{row.source}_{row.target}" in node.attributes()
            assert f"actual_supply_{row.source}_{row.target}" in node.attributes()
            assert (
                node[f"access_state_{row.source}_{row.target}"] == "access undisrupted"
            )


def test_check_access_no_rerouting_constraint_all_functional(graph_calcs):
    """Test _check_access with no rerouting, access constraints, all edges functional"""
    graph_calcs.network.initialize_funcstates()
    graph_calcs.build_graph()

    # Create initial dependencies
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_healthcare_people"},
        link_condition="distance",
        dist_thresh=10e6,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    # Initialize supply
    for v in graph_calcs.graph.vs.select(ci_type="people"):
        v["access_state_healthcare_people"] = "no base access"
        v["actual_supply_healthcare_people"] = 0

    # Create dependency row
    row = pd.Series(
        {
            "source": "healthcare",
            "target": "people",
            "via_link": "road",
            "link_condition": "distance",
            "thresh_dist": 10e6,
            "bidir_link": False,
            "access_cnstr": True,
            "thresh_dur": np.inf,
            "n_links": 1,
        }
    )

    # Check access with no rerouting
    graph_calcs._check_access(row, friction_surf=None, rerouting=False)

    # All via edges are functional, so access should be undisrupted
    people_node = graph_calcs.graph.vs.select(ci_type="people")[0]
    assert people_node["access_state_healthcare_people"] == "access undisrupted"
    assert people_node["actual_supply_healthcare_people"] == 1


def test_check_access_no_rerouting_constraint_via_failed(graph_calcs):
    """Test _check_access with no rerouting, access constraints, failed via edge"""
    graph_calcs.network.initialize_funcstates()
    graph_calcs.build_graph()

    # Create initial dependencies
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_healthcare_people"},
        link_condition="distance",
        dist_thresh=10e6,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    # Initialize supply
    for v in graph_calcs.graph.vs.select(ci_type="people"):
        v["access_state_healthcare_people"] = "no base access"
        v["actual_supply_healthcare_people"] = 0

    # Fail one via edge (break the path)
    graph_calcs.graph.es[1]["func_tot"] = 0

    # Create dependency row
    row = pd.Series(
        {
            "source": "healthcare",
            "target": "people",
            "via_link": "road",
            "link_condition": "distance",
            "thresh_dist": 10e6,
            "bidir_link": False,
            "access_cnstr": True,
            "thresh_dur": np.inf,
            "n_links": 1,
        }
    )

    # Check access with no rerouting
    graph_calcs._check_access(row, friction_surf=None, rerouting=False)

    # Via edge failed, so access should be disrupted
    people_node = graph_calcs.graph.vs.select(ci_type="people")[0]
    assert people_node["access_state_healthcare_people"] == "access disrupted via"
    assert people_node["actual_supply_healthcare_people"] == 0


def test_check_access_no_rerouting_no_constraint_via_failed(graph_calcs):
    """Test _check_access with no rerouting, no access constraints, failed via edge"""
    graph_calcs.network.initialize_funcstates()
    graph_calcs.build_graph()

    # Create initial dependencies
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_healthcare_people"},
        link_condition="distance",
        dist_thresh=10e6,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    # Initialize supply
    for v in graph_calcs.graph.vs.select(ci_type="people"):
        v["access_state_healthcare_people"] = "no base access"
        v["actual_supply_healthcare_people"] = 0

    # Fail one via edge
    graph_calcs.graph.es[1]["func_tot"] = 0

    # Create dependency row without access constraints
    row = pd.Series(
        {
            "source": "healthcare",
            "target": "people",
            "via_link": "road",
            "link_condition": "distance",
            "thresh_dist": 10e6,
            "bidir_link": False,
            "access_cnstr": False,
            "thresh_dur": np.inf,
            "n_links": 1,
        }
    )

    # Check access with no rerouting
    graph_calcs._check_access(row, friction_surf=None, rerouting=False)

    # No access constraints, so even with failed via edge, access should be undisrupted
    people_node = graph_calcs.graph.vs.select(ci_type="people")[0]
    assert people_node["access_state_healthcare_people"] == "access undisrupted"
    assert people_node["actual_supply_healthcare_people"] == 1


def test_propagate_check_fail_people(graph_calcs):
    """Test propagate_check_fail setup"""
    graph_calcs.build_graph()

    # Initialize capacity for propagation
    graph_calcs.graph.vs["capacity_road_people"] = 0
    for v in graph_calcs.graph.vs.select(ci_type="road"):
        v["capacity_road_people"] = 1
        v["func_tot"] = 1  # Ensure roads are functional
    for v in graph_calcs.graph.vs.select(ci_type="people"):
        v["actual_supply_road_people"] = 0
        v["capacity_road_people"] = -1  # Need positive capacity to receive supply

    # add dependency edge for propagation
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_road_people"},
        link_condition="distance",
        dist_thresh=10e6,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    # Run propagation
    graph_calcs._propagate_check_fail(
        source="road", target="people", type_I="enduser", thresh_func=1
    )

    # Verify propagation completed
    assert "actual_supply_road_people" in graph_calcs.graph.vs.attributes()
    assert all(
        v["actual_supply_road_people"] == 1
        for v in graph_calcs.graph.vs.select(ci_type="people")
    )
    assert all(
        v["access_state_road_people"] == "access undisrupted"
        for v in graph_calcs.graph.vs.select(ci_type="people")
    )


def test_propagate_check_fail_ci(graph_calcs):
    """Test propagate_check_fail setup"""
    graph_calcs.build_graph()

    # Initialize capacity for propagation
    graph_calcs.graph.vs["capacity_road_healthcare"] = 0
    for v in graph_calcs.graph.vs.select(ci_type="road"):
        v["capacity_road_healthcare"] = 1
    for v in graph_calcs.graph.vs.select(ci_type="healthcare"):
        v["actual_supply_road_healthcare"] = 0
        v["capacity_road_healthcare"] = -1

    # add dependency edge for propagation
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "healthcare"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_road_healthcare"},
        link_condition="distance",
        dist_thresh=10e6,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    # Run propagation
    graph_calcs._propagate_check_fail(
        source="road", target="healthcare", type_I="functional", thresh_func=1
    )

    # Verify propagation completed
    assert "func_tot" in graph_calcs.graph.vs.attributes()
    assert all(
        v["func_tot"] == 1 for v in graph_calcs.graph.vs.select(ci_type="healthcare")
    )
    assert (
        "access_state_healthcare_road" not in graph_calcs.graph.vs.attributes()
    )  # CI propagation should not set access state


def test_propagate_check_fail_fail_ci(graph_calcs):
    """Test propagate_check_fail setup"""
    graph_calcs.build_graph()

    # Initialize capacity for propagation
    graph_calcs.graph.vs["capacity_road_healthcare"] = 0
    for v in graph_calcs.graph.vs.select(ci_type="road"):
        v["func_tot"] = 0
        v["capacity_road_healthcare"] = 1
    for v in graph_calcs.graph.vs.select(ci_type="healthcare"):
        v["actual_supply_road_healthcare"] = 0
        v["capacity_road_healthcare"] = -1

    # add dependency edge for propagation
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "healthcare"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_road_healthcare"},
        link_condition="distance",
        dist_thresh=10e6,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )
    # Run propagation
    graph_calcs._propagate_check_fail(
        source="road", target="healthcare", type_I="functional", thresh_func=1
    )

    # Verify propagation completed
    assert "func_tot" in graph_calcs.graph.vs.attributes()
    assert all(
        v["func_tot"] == 0 for v in graph_calcs.graph.vs.select(ci_type="healthcare")
    )
    assert all(
        v["actual_supply_road_healthcare"] == 0
        for v in graph_calcs.graph.vs.select(ci_type="healthcare")
    )


def test_propagate_check_fail_fail_enduser(graph_calcs):
    """Test propagate_check_fail setup"""
    graph_calcs.build_graph()

    # Initialize capacity for propagation
    graph_calcs.graph.vs["capacity_healthcare_people"] = 0
    for v in graph_calcs.graph.vs.select(ci_type="healthcare"):
        v["func_tot"] = 0
        v["capacity_healthcare_people"] = 1
    for v in graph_calcs.graph.vs.select(ci_type="people"):
        v["actual_supply_healthcare_people"] = 1
        v["capacity_healthcare_people"] = -1

    # add dependency edge for propagation
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_healthcare_people"},
        link_condition="distance",
        dist_thresh=10e6,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    # Run propagation
    graph_calcs._propagate_check_fail(
        source="healthcare", target="people", type_I="enduser", thresh_func=1
    )

    # Verify propagation completed
    assert "actual_supply_healthcare_people" in graph_calcs.graph.vs.attributes()
    assert all(
        v["actual_supply_healthcare_people"] == 0
        for v in graph_calcs.graph.vs.select(ci_type="people")
    )
    assert all(
        v["access_state_healthcare_people"] == "access disrupted"
        for v in graph_calcs.graph.vs.select(ci_type="people")
    )


def test_update_internal_dependencies_roads(graph_calcs):
    """Test updating internal dependencies for roads"""
    graph_calcs.build_graph()

    # Set some edge functionality
    graph_calcs.graph.es["func_tot"] = 0

    graph_calcs.update_internal_dependencies(
        p_source="powerstation",
        p_sink="powerline",
        source_var="capacity",
        demand_var="demand",
    )

    # Method should complete
    assert all(v["func_tot"] == 0 for v in graph_calcs.graph.vs.select(ci_type="road"))


def test_update_functional_dependencies(graph_calcs):
    """Test updating functional dependencies"""
    graph_calcs.build_graph()

    # Create dependency dataframe
    df_dependencies = pd.DataFrame(
        {
            "source": ["road"],
            "target": ["healthcare"],
            "type_I": ["functional"],
            "type_II": ["logical"],
            "via_link": ["none"],
            "thresh_func": [1.0],
            "thresh_dist": [np.inf],
            "bidir_link": [False],
            "access_cnstr": [False],
        }
    )

    # Initialize capacities
    graph_calcs.graph.vs["capacity_road_healthcare"] = 0
    for v in graph_calcs.graph.vs.select(ci_type="road"):
        v["capacity_road_healthcare"] = 1
    for v in graph_calcs.graph.vs.select(ci_type="healthcare"):
        v["capacity_road_healthcare"] = -1

    graph_calcs.update_functional_dependencies(df_dependencies)
    assert all(
        v["func_tot"] == 1 for v in graph_calcs.graph.vs.select(ci_type="healthcare")
    )

    # repeat with road failure
    # set all road nodes as failed
    for v in graph_calcs.graph.vs.select(ci_type="road"):
        v["func_tot"] = 0
    graph_calcs.update_functional_dependencies(df_dependencies)
    assert all(
        v["func_tot"] == 0 for v in graph_calcs.graph.vs.select(ci_type="healthcare")
    )


def test_update_enduser_dependencies_routing(graph_calcs_with_source_fail):
    """Test updating end-user dependencies"""
    graph_calcs_with_source_fail.build_graph()

    # Create end-user dependency dataframe
    df_dependencies = pd.DataFrame(
        {
            "source": ["road", "healthcare"],
            "target": ["people", "people"],
            "type_I": ["enduser", "enduser"],
            "access_cnstr": [False, True],
            "via_link": ["none", "road"],
            "link_condition": ["edgecond", "distance"],
            "thresh_dist": [10e6, 10e6],
            "thresh_dur": [np.inf, np.inf],
            "bidir_link": [False, False],
            "thresh_func": [1.0, 1.0],
            "n_links": [1, 1],
        }
    )

    graph_calcs_with_source_fail.update_enduser_dependencies(
        df_dependencies,
        friction_surf=None,
        rerouting=False,
        access_check_method="routing",
    )

    assert graph_calcs_with_source_fail.graph is not None


def test_update_enduser_dependencies_propagation(graph_calcs_with_source_fail):
    """Test updating end-user dependencies"""

    # Create end-user dependency dataframe
    df_dependencies = pd.DataFrame(
        {
            "source": ["road", "healthcare"],
            "target": ["people", "people"],
            "type_I": ["enduser", "enduser"],
            "access_cnstr": [False, True],
            "via_link": ["none", "road"],
            "link_condition": ["edgecond", "distance"],
            "thresh_dist": [10e6, 10e6],
            "thresh_dur": [np.inf, np.inf],
            "bidir_link": [False, False],
            "thresh_func": [1.0, 1.0],
            "n_links": [1, 1],
        }
    )
    graph_calcs_with_source_fail.network.initialize_funcstates()
    graph_calcs_with_source_fail.network.initialize_capacity(df_dependencies)
    graph_calcs_with_source_fail.network.initialize_supply(df_dependencies)
    graph_calcs_with_source_fail.build_graph()

    graph_calcs_with_source_fail.update_enduser_dependencies(
        df_dependencies,
        friction_surf=None,
        rerouting=False,
        access_check_method="propagation",
    )

    assert graph_calcs_with_source_fail.graph is not None


def test_update_enduser_dependencies_unknown(graph_calcs_with_source_fail):
    """Test updating end-user dependencies"""
    graph_calcs_with_source_fail.build_graph()

    # Create end-user dependency dataframe
    df_dependencies = pd.DataFrame(
        {
            "source": ["road", "healthcare"],
            "target": ["people", "people"],
            "type_I": ["enduser", "enduser"],
            "access_cnstr": [False, True],
            "via_link": ["none", "road"],
            "link_condition": ["edgecond", "distance"],
            "thresh_dist": [10e6, 10e6],
            "thresh_dur": [np.inf, np.inf],
            "bidir_link": [False, False],
            "thresh_func": [1.0, 1.0],
            "n_links": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="Invalid access check method specified!"):
        graph_calcs_with_source_fail.update_enduser_dependencies(
            df_dependencies,
            friction_surf=None,
            rerouting=False,
            access_check_method="blabla",
        )


def test_get_former_access_info(graph_calcs):
    """Test _get_former_access_info helper function"""
    graph_calcs.network.initialize_funcstates()
    graph_calcs.build_graph()

    # Create dependencies
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_healthcare_people"},
        link_condition="distance",
        dist_thresh=10e6,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    # Get former access info
    es_access_base, ppl_former_access, ppl_former_access_source_failed = (
        graph_calcs._get_former_access_info("dependency_healthcare_people")
    )

    assert len(es_access_base) == 1
    assert len(ppl_former_access) == 1
    assert len(ppl_former_access_source_failed) == 0  # No sources failed yet

    # Fail a source
    healthcare_nodes = graph_calcs.graph.vs.select(ci_type="healthcare")
    healthcare_nodes[0]["func_tot"] = 0

    # Get former access info again
    es_access_base, ppl_former_access, ppl_former_access_source_failed = (
        graph_calcs._get_former_access_info("dependency_healthcare_people")
    )

    assert len(ppl_former_access_source_failed) == 1


def test_recompute_dependencies_with_rerouting(graph_calcs):
    """Test _recompute_dependencies_with_rerouting helper function"""
    graph_calcs.network.initialize_funcstates()
    graph_calcs.build_graph()

    # Create initial dependencies
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_healthcare_people"},
        link_condition="distance",
        dist_thresh=10e6,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    row = pd.Series(
        {
            "source": "healthcare",
            "target": "people",
            "via_link": "road",
            "link_condition": "distance",
            "thresh_dist": 10e6,
            "bidir_link": False,
            "access_cnstr": True,
            "thresh_dur": np.inf,
            "n_links": 1,
        }
    )

    # Recompute with rerouting
    ppl_new_access, ppl_access_all_via = (
        graph_calcs._recompute_dependencies_with_rerouting(
            row, "dependency_healthcare_people"
        )
    )

    assert len(ppl_new_access) == 1
    assert len(ppl_access_all_via) == 1


def test_recompute_dependencies_with_rerouting_fail_source(graph_calcs):
    """Test _recompute_dependencies_with_rerouting helper function with failed source"""
    graph_calcs.network.initialize_funcstates()
    graph_calcs.build_graph()

    # Create initial dependencies
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_healthcare_people"},
        link_condition="distance",
        dist_thresh=10e6,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    row = pd.Series(
        {
            "source": "healthcare",
            "target": "people",
            "via_link": "road",
            "link_condition": "distance",
            "thresh_dist": 10e6,
            "bidir_link": False,
            "access_cnstr": True,
            "thresh_dur": np.inf,
            "n_links": 1,
        }
    )

    # Fail the healthcare source
    healthcare_nodes = graph_calcs.graph.vs.select(ci_type="healthcare")
    healthcare_nodes[0]["func_tot"] = 0

    # Recompute with rerouting
    ppl_new_access, ppl_access_all_via = (
        graph_calcs._recompute_dependencies_with_rerouting(
            row, "dependency_healthcare_people"
        )
    )

    assert len(ppl_new_access) == 0
    assert len(ppl_access_all_via) == 0


def test_recompute_dependencies_with_rerouting_fail_via(graph_calcs):
    """Test _recompute_dependencies_with_rerouting helper function with failed via link"""
    graph_calcs.network.initialize_funcstates()
    graph_calcs.build_graph()

    # Create initial dependencies
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_healthcare_people"},
        link_condition="distance",
        dist_thresh=10e6,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    row = pd.Series(
        {
            "source": "healthcare",
            "target": "people",
            "via_link": "road",
            "link_condition": "distance",
            "thresh_dist": 10e6,
            "bidir_link": False,
            "access_cnstr": True,
            "thresh_dur": np.inf,
            "n_links": 1,
        }
    )

    # Fail the via link
    road_nodes = graph_calcs.graph.vs.select(ci_type="road")
    road_nodes[0]["func_tot"] = 0
    graph_calcs.graph.es.select(ci_type="road")[0]["func_tot"] = 0

    # Recompute with rerouting
    ppl_new_access, ppl_access_all_via = (
        graph_calcs._recompute_dependencies_with_rerouting(
            row, "dependency_healthcare_people"
        )
    )

    assert len(ppl_new_access) == 0
    assert len(ppl_access_all_via) == 1


def test_validate_dependency_paths(graph_calcs):
    """Test _validate_dependency_paths helper function"""
    graph_calcs.network.initialize_funcstates()
    graph_calcs.build_graph()

    row = pd.Series(
        {
            "source": "healthcare",
            "target": "people",
            "via_link": "road",
            "link_condition": "distance",
            "thresh_dist": 10e6,
            "bidir_link": False,
            "access_cnstr": True,
            "thresh_dur": np.inf,
            "n_links": 1,
        }
    )

    # Create dependencies
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": row["source"]},
        target_attrs={"ci_type": row["target"]},
        via_attrs={"ci_type": row["via_link"]},
        link_attrs={"ci_type": "dependency_" + row["source"] + "_" + row["target"]},
        link_condition=row["link_condition"],
        dist_thresh=row["thresh_dist"],
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    es_check = list(graph_calcs.graph.es.select(ci_type="dependency_healthcare_people"))
    edge_pairs = [(edge.source, edge.target) for edge in es_check]

    # Create subgraph
    subgraph = graph_calcs._create_subgraph(
        source_attrs={"ci_type": row["source"], "func_tot": 1},
        target_attrs={"ci_type": row["target"]},
        via_attrs={"ci_type": row["via_link"], "func_tot": 1},
    )

    row = pd.Series({"thresh_dist": 10e6})

    # Validate paths
    pairs_to_keep = graph_calcs._validate_dependency_paths(edge_pairs, row, subgraph)

    # All paths should be valid initially
    assert len(pairs_to_keep) == len(edge_pairs)


def test_validate_dependency_paths_edgefail(graph_calcs):
    """Test _validate_dependency_paths helper function"""
    graph_calcs.network.initialize_funcstates()
    graph_calcs.build_graph()

    row = pd.Series(
        {
            "source": "healthcare",
            "target": "people",
            "via_link": "road",
            "link_condition": "distance",
            "thresh_dist": 10e6,
            "bidir_link": False,
            "access_cnstr": True,
            "thresh_dur": np.inf,
            "n_links": 1,
        }
    )

    # Create dependencies
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": row["source"]},
        target_attrs={"ci_type": row["target"]},
        via_attrs={"ci_type": row["via_link"]},
        link_attrs={"ci_type": "dependency_" + row["source"] + "_" + row["target"]},
        link_condition=row["link_condition"],
        dist_thresh=row["thresh_dist"],
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    es_check = list(graph_calcs.graph.es.select(ci_type="dependency_healthcare_people"))
    edge_pairs = [(edge.source, edge.target) for edge in es_check]

    # fail one road edge
    graph_calcs.graph.es.select(ci_type="road")[0]["func_tot"] = 0

    # Create subgraph
    subgraph = graph_calcs._create_subgraph(
        source_attrs={"ci_type": row["source"], "func_tot": 1},
        target_attrs={"ci_type": row["target"]},
        via_attrs={"ci_type": row["via_link"], "func_tot": 1},
    )

    row = pd.Series({"thresh_dist": 10e6})

    # Validate paths
    pairs_to_keep = graph_calcs._validate_dependency_paths(edge_pairs, row, subgraph)

    # after only path is failed no remaining paths are left
    assert len(pairs_to_keep) == 0


def test_validate_dependencies_without_rerouting_with_constraint(graph_calcs):
    """Test _validate_dependencies_without_rerouting with access constraints"""
    graph_calcs.network.initialize_funcstates()
    graph_calcs.build_graph()

    row = pd.Series(
        {
            "source": "healthcare",
            "target": "people",
            "via_link": "road",
            "link_condition": "distance",
            "thresh_dist": 10e6,
            "bidir_link": False,
            "access_cnstr": True,
            "thresh_dur": np.inf,
            "n_links": 1,
        }
    )

    dependency_name = "dependency_" + row["source"] + "_" + row["target"]

    # Create dependencies
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": row["source"]},
        target_attrs={"ci_type": row["target"]},
        via_attrs={"ci_type": row["via_link"]},
        link_attrs={"ci_type": dependency_name},
        link_condition=row["link_condition"],
        dist_thresh=row["thresh_dist"],
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    es_access_base = list(graph_calcs.graph.es.select(ci_type=dependency_name))
    ppl_former_access = [edge.target for edge in es_access_base]

    # Validate without rerouting
    ppl_new_access, ppl_access_all_via = (
        graph_calcs._validate_dependencies_without_rerouting(
            row,
            dependency_name,
            es_access_base,
        )
    )

    assert len(ppl_new_access) == 1
    assert ppl_access_all_via == ppl_former_access


def test_mark_access_states_and_supply(graph_calcs):
    """Test _mark_access_states_and_supply helper function"""
    graph_calcs.network.initialize_funcstates()
    graph_calcs.build_graph()

    # Create dependencies
    graph_calcs.calc_dependencies(
        source_attrs={"ci_type": "healthcare"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_healthcare_people"},
        link_condition="distance",
        dist_thresh=10e6,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    # Initialize access states
    for v in graph_calcs.graph.vs.select(ci_type="people"):
        v["access_state_healthcare_people"] = "no base access"
        v["actual_supply_healthcare_people"] = 0

    es_access = list(
        graph_calcs.graph.es.select(ci_type="dependency_healthcare_people")
    )
    ppl_former_access = [edge.target for edge in es_access]
    ppl_former_access_source_failed = []
    ppl_access_all_via = ppl_former_access
    ppl_new_access = ppl_former_access

    row = pd.Series({"source": "healthcare", "target": "people"})

    # Mark access states
    graph_calcs._mark_access_states_and_supply(
        row,
        ppl_former_access,
        ppl_former_access_source_failed,
        ppl_access_all_via,
        ppl_new_access,
    )

    # Check that access states were set
    people_nodes = graph_calcs.graph.vs.select(ci_type="people")
    for node in people_nodes:
        assert "access_state_healthcare_people" in node.attributes()
        assert "actual_supply_healthcare_people" in node.attributes()
        assert node["access_state_healthcare_people"] == "access undisrupted"
        assert node["actual_supply_healthcare_people"] == 1


# ========================================================================
# Auto-Sync Feature Tests
# ========================================================================


def test_auto_sync_disabled_by_default(graph_calcs):
    """Test that auto_sync is disabled by default"""
    assert graph_calcs.auto_sync is False


def test_auto_sync_parameter_initialization(network_with_ci_types):
    """Test that auto_sync parameter can be set during initialization"""
    gc_manual = GraphCalcs(network=network_with_ci_types, auto_sync=False)
    gc_auto = GraphCalcs(network=network_with_ci_types, auto_sync=True)

    assert gc_manual.auto_sync is False
    assert gc_auto.auto_sync is True


def test_sync_method_updates_network(graph_calcs):
    """Test that sync() method updates the network with graph changes"""
    graph_calcs.build_graph()
    initial_network_edges = len(graph_calcs.network.edges)
    initial_graph_edges = graph_calcs.graph.ecount()

    # Add edges to graph
    graph_calcs.link_vertices_closest_k(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "healthcare"},
        k=1,
        dist_thresh=np.inf,
        link_attrs={"ci_type": "test_link"},
    )

    # Graph should have new edges, but network not updated yet
    graph_edges_before_sync = graph_calcs.graph.ecount()
    network_edges_before_sync = len(graph_calcs.network.edges)

    assert graph_edges_before_sync > initial_graph_edges
    assert network_edges_before_sync == initial_network_edges  # Not synced yet
    assert graph_edges_before_sync != network_edges_before_sync  # Different!

    # Manual sync
    graph_calcs.sync()

    # Network should now be updated
    network_edges_after_sync = len(graph_calcs.network.edges)
    graph_edges_after_sync = graph_calcs.graph.ecount()

    assert network_edges_after_sync > initial_network_edges
    assert network_edges_after_sync == graph_edges_after_sync  # Now synced!

    # Verify test_link is present in network
    test_links = graph_calcs.network.edges[
        graph_calcs.network.edges["ci_type"] == "test_link"
    ]
    assert len(test_links) > 0

    # Verify counts match
    graph_test_link_count = len(graph_calcs.graph.es.select(ci_type="test_link"))
    assert len(test_links) == graph_test_link_count


def test_auto_sync_link_clusters(graph_calcs_with_remote_node_missing_edge):
    """Test auto_sync with link_clusters"""
    graph_calcs_with_remote_node_missing_edge.build_graph()
    initial_graph_edges = graph_calcs_with_remote_node_missing_edge.graph.ecount()

    # Create instance with auto_sync=True
    gc_auto = GraphCalcs(
        network=graph_calcs_with_remote_node_missing_edge.network, auto_sync=True
    )
    gc_auto.build_graph()

    # Link clusters
    gc_auto.link_clusters(dist_thresh=np.inf, link_attrs={"ci_type": "cluster_link"})

    # Verify graph and network are synchronized
    graph_edges = gc_auto.graph.ecount()
    network_edges = len(gc_auto.network.edges)

    # Network should reflect graph state
    assert network_edges == graph_edges
    # Both should have no cluster links added (already connected)
    cluster_edges = gc_auto.graph.es.select(ci_type="cluster_link")
    assert len(cluster_edges) == network_edges - initial_graph_edges


def test_auto_sync_link_vertices_closest_k(graph_calcs):
    """Test auto_sync with link_vertices_closest_k"""
    gc_auto = GraphCalcs(network=graph_calcs.network, auto_sync=True)
    gc_auto.build_graph()
    initial_graph_edges = gc_auto.graph.ecount()

    # Link vertices
    gc_auto.link_vertices_closest_k(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "healthcare"},
        k=1,
        dist_thresh=np.inf,
        link_attrs={"ci_type": "test_link"},
    )

    # Network should be automatically synchronized
    # Count edges in both graph and network
    graph_edges = gc_auto.graph.ecount()
    network_edges = len(gc_auto.network.edges)

    # Verify sync: network should match graph
    assert network_edges == graph_edges
    # Verify new edges were added
    assert graph_edges > initial_graph_edges
    # Verify link is in network
    test_links = gc_auto.network.edges[gc_auto.network.edges["ci_type"] == "test_link"]
    assert len(test_links) > 0
    # Verify all test_link edges are present in both
    test_link_count = len(gc_auto.graph.es.select(ci_type="test_link"))
    assert len(test_links) == test_link_count


def test_auto_sync_link_vertices_edgecond(graph_calcs):
    """Test auto_sync with link_vertices_edgecond"""
    gc_auto = GraphCalcs(network=graph_calcs.network, auto_sync=True)
    gc_auto.build_graph()

    # Link vertices by edge condition
    gc_auto.link_vertices_edgecond(
        target_attrs={"ci_type": "healthcare"},
        edge_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "test_edgecond_link"},
    )

    # Verify graph and network are synchronized
    graph_edges = gc_auto.graph.ecount()
    network_edges = len(gc_auto.network.edges)
    assert network_edges == graph_edges

    # Verify new edges were actually added
    edgecond_links = gc_auto.network.edges[
        gc_auto.network.edges["ci_type"] == "test_edgecond_link"
    ]
    assert len(edgecond_links) > 0

    # Verify all edgecond links in graph are in network
    graph_edgecond_count = len(gc_auto.graph.es.select(ci_type="test_edgecond_link"))
    assert len(edgecond_links) == graph_edgecond_count


def test_auto_sync_link_vertices_shortest_paths(graph_calcs):
    """Test auto_sync with link_vertices_shortest_paths"""
    gc_auto = GraphCalcs(network=graph_calcs.network, auto_sync=True)
    gc_auto.build_graph()

    # Link vertices via shortest paths
    gc_auto.link_vertices_shortest_paths(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "healthcare"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "test_shortest_path"},
        dist_thresh=np.inf,
        k=1,
        bidir=False,
    )

    # Verify graph and network are synchronized
    graph_edges = gc_auto.graph.ecount()
    network_edges = len(gc_auto.network.edges)
    assert network_edges == graph_edges

    # Verify new edges were added
    shortest_path_edges = gc_auto.network.edges[
        gc_auto.network.edges["ci_type"] == "test_shortest_path"
    ]
    assert len(shortest_path_edges) > 0

    # Verify count matches between graph and network
    graph_sp_count = len(gc_auto.graph.es.select(ci_type="test_shortest_path"))
    assert len(shortest_path_edges) == graph_sp_count


def test_auto_sync_calc_dependencies(graph_calcs):
    """Test auto_sync with calc_dependencies"""
    gc_auto = GraphCalcs(network=graph_calcs.network, auto_sync=True)
    gc_auto.network.initialize_funcstates()
    gc_auto.build_graph()
    initial_graph_edges = gc_auto.graph.ecount()

    # Create dependencies between road and people via road
    gc_auto.calc_dependencies(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_road_people"},
        link_condition="distance",
        dist_thresh=np.inf,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    # Verify graph and network are synchronized
    graph_edges = gc_auto.graph.ecount()
    network_edges = len(gc_auto.network.edges)
    assert network_edges == graph_edges

    # Check that graph has dependency edges
    dep_edges = gc_auto.graph.es.select(ci_type="dependency_road_people")
    assert len(dep_edges) > 0

    # Verify network has same dependency edges
    network_dep_edges = gc_auto.network.edges[
        gc_auto.network.edges["ci_type"] == "dependency_road_people"
    ]
    assert len(network_dep_edges) == len(dep_edges)


def test_auto_sync_update_functional_dependencies(graph_calcs, dependency_table):
    """Test auto_sync with update_functional_dependencies"""
    gc_auto = GraphCalcs(network=graph_calcs.network, auto_sync=True)
    gc_auto.network.initialize_funcstates()
    gc_auto.build_graph()

    # Set up initial state with capacity attributes
    for v in gc_auto.graph.vs:
        for target_type in gc_auto.graph.vs["ci_type"]:
            v[f"capacity_{v['ci_type']}_{target_type}"] = 1.0

    # Set up initial state
    dep_df = dependency_table[dependency_table["type_I"] == "functional"]

    # Get initial functional state
    initial_graph_edges = gc_auto.graph.ecount()
    initial_network_edges = len(gc_auto.network.edges)

    # Update functional dependencies
    gc_auto.update_functional_dependencies(dep_df)

    # Verify graph and network are synchronized
    final_graph_edges = gc_auto.graph.ecount()
    final_network_edges = len(gc_auto.network.edges)
    assert final_network_edges == final_graph_edges

    # Verify both graph and network have same edges after sync
    assert (
        initial_graph_edges == final_graph_edges
    )  # No new edges added by functional update
    assert initial_network_edges == final_network_edges

    # Check that network was synced (funcstates should be valid)
    final_func_sum = gc_auto.funcstates_sum()[0]
    assert isinstance(final_func_sum, (int, float))
    assert final_func_sum >= 0  # Functional state should be non-negative


def test_auto_sync_update_enduser_dependencies_routing(
    graph_calcs_with_source_fail, dependency_table
):
    """Test auto_sync with update_enduser_dependencies using routing"""
    gc_auto = GraphCalcs(network=graph_calcs_with_source_fail.network, auto_sync=True)
    gc_auto.network.initialize_funcstates()
    gc_auto.build_graph()

    # Create initial dependencies
    gc_auto.calc_dependencies(
        source_attrs={"ci_type": "healthcare", "func_tot": 1},
        target_attrs={"ci_type": "people"},
        via_attrs={"ci_type": "road"},
        link_attrs={"ci_type": "dependency_healthcare_people"},
        link_condition="edgecond",
        dist_thresh=np.inf,
        dur_thresh=np.inf,
        k=1,
        bidir_link=False,
    )

    # Get enduser dependencies
    dep_df = dependency_table[dependency_table["type_I"] == "enduser"]

    # Update with routing
    gc_auto.update_enduser_dependencies(
        dep_df, friction_surf=None, access_check_method="routing", rerouting=True
    )

    # Verify graph and network are synchronized
    graph_edges = gc_auto.graph.ecount()
    network_edges = len(gc_auto.network.edges)
    assert network_edges == graph_edges
    assert network_edges > 0

    # Verify people nodes have access state attributes
    people_nodes = gc_auto.graph.vs.select(ci_type="people")
    assert len(people_nodes) > 0
    for node in people_nodes:
        # Should have access state attributes for dependencies
        assert "func_tot" in node.attributes()
        assert node["func_tot"] >= 0


def test_auto_sync_vs_manual_sync_consistency(graph_calcs, network_with_ci_types):
    """Test that auto_sync and manual sync produce the same results"""
    from copy import deepcopy

    # Manual sync version
    network_manual = deepcopy(network_with_ci_types)
    gc_manual = GraphCalcs(network=network_manual, auto_sync=False)
    gc_manual.build_graph()
    gc_manual.link_vertices_closest_k(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "healthcare"},
        k=1,
        dist_thresh=np.inf,
        link_attrs={"ci_type": "test_link"},
    )
    gc_manual.sync()

    # Auto sync version
    network_auto = deepcopy(network_with_ci_types)
    gc_auto = GraphCalcs(network=network_auto, auto_sync=True)
    gc_auto.build_graph()
    gc_auto.link_vertices_closest_k(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "healthcare"},
        k=1,
        dist_thresh=np.inf,
        link_attrs={"ci_type": "test_link"},
    )

    # Both should have same number of edges
    assert len(gc_manual.network.edges) == len(gc_auto.network.edges)
    assert gc_manual.graph.ecount() == gc_auto.graph.ecount()

    # Both should have the same edge types
    manual_types = set(gc_manual.network.edges["ci_type"].values)
    auto_types = set(gc_auto.network.edges["ci_type"].values)
    assert manual_types == auto_types

    # Both should have same test_link count
    manual_test_links = len(
        gc_manual.network.edges[gc_manual.network.edges["ci_type"] == "test_link"]
    )
    auto_test_links = len(
        gc_auto.network.edges[gc_auto.network.edges["ci_type"] == "test_link"]
    )
    assert manual_test_links == auto_test_links
    assert manual_test_links > 0

    # Verify connectivity is identical
    for test_link in gc_manual.graph.es.select(ci_type="test_link"):
        source_idx = test_link.source
        target_idx = test_link.target
        source_id = gc_manual.graph.vs[source_idx]["orig_id"]
        target_id = gc_manual.graph.vs[target_idx]["orig_id"]

        # Find same link in auto version
        auto_source = gc_auto.graph.vs.select(orig_id=source_id)[0].index
        auto_target = gc_auto.graph.vs.select(orig_id=target_id)[0].index

        # Should have equivalent edge
        edge_exists = False
        for edge in gc_auto.graph.es.select(ci_type="test_link"):
            if edge.source == auto_source and edge.target == auto_target:
                edge_exists = True
                break
        assert edge_exists


def test_no_auto_sync_without_explicit_call(graph_calcs):
    """Test that without auto_sync, manual sync is required"""
    gc_manual = GraphCalcs(network=graph_calcs.network, auto_sync=False)
    gc_manual.build_graph()
    initial_edges = len(gc_manual.network.edges)
    initial_graph_edges = gc_manual.graph.ecount()

    # Add edges to graph
    gc_manual.link_vertices_closest_k(
        source_attrs={"ci_type": "road"},
        target_attrs={"ci_type": "healthcare"},
        k=1,
        dist_thresh=np.inf,
        link_attrs={"ci_type": "test_link"},
    )

    # Graph has new edges, but network hasn't been updated yet
    graph_edges_after = gc_manual.graph.ecount()
    network_edges_after = len(gc_manual.network.edges)

    assert graph_edges_after > initial_graph_edges
    assert network_edges_after == initial_edges  # Network still unchanged

    # Verify test_link exists in graph but not in network
    graph_test_links = gc_manual.graph.es.select(ci_type="test_link")
    assert len(graph_test_links) > 0
    network_test_links = gc_manual.network.edges[
        gc_manual.network.edges["ci_type"] == "test_link"
    ]
    assert len(network_test_links) == 0  # Not synced yet

    # Now manually sync
    gc_manual.sync()

    # Network should now be updated
    final_network_edges = len(gc_manual.network.edges)
    final_graph_edges = gc_manual.graph.ecount()
    assert final_network_edges > initial_edges
    assert final_network_edges == final_graph_edges  # Now synced

    # Verify test_link is now in network
    network_test_links_after = gc_manual.network.edges[
        gc_manual.network.edges["ci_type"] == "test_link"
    ]
    assert len(network_test_links_after) == len(graph_test_links)
