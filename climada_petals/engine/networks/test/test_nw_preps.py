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

Test nw_preps module
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, LineString, MultiLineString

from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks import nw_preps
from climada_petals.engine.networks.test.fixtures_test_networks import *  # noqa: F401,F403


# ========================================================================
# Tests: line_endpoints
# ========================================================================


class TestLineEndpoints:
    def test_returns_start_and_end(self):
        """Start and end points of a simple line."""
        line = LineString([(0, 0), (1, 1), (2, 2)])
        start, end = nw_preps.line_endpoints(line)

        assert shapely.get_x(start) == 0.0
        assert shapely.get_y(start) == 0.0
        assert shapely.get_x(end) == 2.0
        assert shapely.get_y(end) == 2.0

    def test_two_point_line(self):
        """Line with only two vertices."""
        line = LineString([(5, 10), (15, 20)])
        start, end = nw_preps.line_endpoints(line)

        assert shapely.get_x(start) == 5.0
        assert shapely.get_y(start) == 10.0
        assert shapely.get_x(end) == 15.0
        assert shapely.get_y(end) == 20.0

    def test_closed_ring(self):
        """Ring line returns same start and end coordinates."""
        line = LineString([(0, 0), (1, 0), (1, 1), (0, 0)])
        start, end = nw_preps.line_endpoints(line)

        assert shapely.equals(start, end)


# ========================================================================
# Tests: nearest / nearest_node
# ========================================================================


class TestNearest:
    def test_nearest_finds_closest(self):
        """Nearest returns the closest row from the dataframe."""
        # Use short LineStrings so STRtree bbox intersections succeed
        gdf = gpd.GeoDataFrame(
            {
                "id": [0, 1, 2],
                "geometry": [
                    LineString([(0, 0), (0.1, 0.1)]),
                    LineString([(5, 5), (5.1, 5.1)]),
                    LineString([(10, 10), (10.1, 10.1)]),
                ],
            },
            geometry="geometry",
        )
        sindex = shapely.STRtree(gdf.geometry)
        # Query with a small buffer so the bbox intersects
        query_geom = Point(4, 4).buffer(2)

        result = nw_preps.nearest(query_geom, gdf, sindex)

        assert result["id"] == 1

    def test_nearest_node_delegates(self):
        """nearest_node returns the same result as nearest."""
        # Use LineStrings as geometry so STRtree bbox intersections succeed
        nodes = gpd.GeoDataFrame(
            {
                "id": [0, 1, 2],
                "geometry": [
                    LineString([(0, 0), (0.1, 0.1)]),
                    LineString([(3, 3), (3.1, 3.1)]),
                    LineString([(10, 10), (10.1, 10.1)]),
                ],
            },
            geometry="geometry",
        )
        sindex = shapely.STRtree(nodes.geometry)
        query_point = Point(2.5, 2.5).buffer(1)

        result = nw_preps.nearest_node(query_point, nodes, sindex)

        assert result["id"] == 1


# ========================================================================
# Tests: add_ids
# ========================================================================


class TestAddIds:
    def test_ids_sequential(self, simple_network):
        """Node and edge ids are sequential starting from 0."""
        result = nw_preps.add_ids(simple_network)

        np.testing.assert_array_equal(result.nodes["id"].values, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(result.edges["id"].values, [0, 1, 2, 3])

    def test_ids_reset_after_drop(self, simple_network):
        """After dropping a node, ids are still sequential."""
        simple_network.nodes = simple_network.nodes.drop(index=2).reset_index(drop=True)
        result = nw_preps.add_ids(simple_network)

        np.testing.assert_array_equal(result.nodes["id"].values, [0, 1, 2, 3])

    def test_custom_id_col(self, simple_network):
        """Custom id column name is used."""
        result = nw_preps.add_ids(simple_network, id_col="my_id")

        assert "my_id" in result.nodes.columns
        assert "my_id" in result.edges.columns
        np.testing.assert_array_equal(result.nodes["my_id"].values, [0, 1, 2, 3, 4])

    def test_empty_network(self, empty_network):
        """Works on empty network."""
        result = nw_preps.add_ids(empty_network)

        assert len(result.nodes) == 0
        assert len(result.edges) == 0


# ========================================================================
# Tests: add_topology
# ========================================================================


class TestAddTopology:
    def test_from_to_ids_correct(self, simple_network):
        """from_id and to_id are set correctly based on edge geometry."""
        network = nw_preps.add_ids(simple_network)
        result = nw_preps.add_topology(network)

        np.testing.assert_array_equal(result.edges["from_id"].values, [0, 1, 2, 3])
        np.testing.assert_array_equal(result.edges["to_id"].values, [1, 2, 3, 4])


# ========================================================================
# Tests: get_endpoints / add_endpoints
# ========================================================================


class TestEndpoints:
    def test_get_endpoints_count(self, simple_network):
        """Each edge contributes 2 endpoints."""
        endpoints = nw_preps.get_endpoints(simple_network)

        # 4 edges × 2 endpoints = 8, but some overlap
        assert len(endpoints) == 8
        assert isinstance(endpoints, gpd.GeoDataFrame)

    def test_get_endpoints_values(self, simple_network):
        """Endpoint coordinates match edge start/end vertices."""
        endpoints = nw_preps.get_endpoints(simple_network)
        xs = [shapely.get_x(g) for g in endpoints.geometry]
        ys = [shapely.get_y(g) for g in endpoints.geometry]

        # Edge 0: (0,0)-(1,1) → endpoints at x=0,1
        assert 0.0 in xs
        assert 4.0 in xs
        assert 0.0 in ys
        assert 4.0 in ys

    def test_add_endpoints_deduplicates(self, simple_network):
        """add_endpoints adds unique endpoint nodes only."""
        # Start with no nodes
        network_no_nodes = Network(
            edges=simple_network.edges.copy(),
            nodes=gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        )
        result = nw_preps.add_endpoints(network_no_nodes)

        # 4 edges connect 5 unique points (0..4)
        assert len(result.nodes) == 5

    def test_get_endpoints_preserves_geographic_crs(self, simple_network):
        """get_endpoints preserves geographic CRS (EPSG:4326)."""
        assert simple_network.edges.crs.to_string() == "EPSG:4326"

        endpoints = nw_preps.get_endpoints(simple_network)

        assert endpoints.crs.to_string() == "EPSG:4326"

    def test_get_endpoints_preserves_projected_crs(self):
        """get_endpoints preserves projected CRS (EPSG:32632)."""
        edges = gpd.GeoDataFrame(
            {
                "from_id": [0, 1],
                "to_id": [1, 2],
                "id": [0, 1],
                "osm_id": [100, 101],
                "geometry": [
                    LineString([(500000, 5000000), (501000, 5000000)]),
                    LineString([(501000, 5000000), (502000, 5000000)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:32632",
        )
        nodes = gpd.GeoDataFrame(
            {
                "id": [0, 1, 2],
                "geometry": [
                    Point(500000, 5000000),
                    Point(501000, 5000000),
                    Point(502000, 5000000),
                ],
            },
            geometry="geometry",
            crs="EPSG:32632",
        )
        network = Network(edges=edges, nodes=nodes)

        endpoints = nw_preps.get_endpoints(network)

        assert endpoints.crs.to_string() == "EPSG:32632"

    def test_add_endpoints_preserves_projected_crs(self):
        """add_endpoints preserves projected CRS through the full pipeline."""
        edges = gpd.GeoDataFrame(
            {
                "from_id": [0, 1],
                "to_id": [1, 2],
                "id": [0, 1],
                "osm_id": [100, 101],
                "geometry": [
                    LineString([(500000, 5000000), (501000, 5000000)]),
                    LineString([(501000, 5000000), (502000, 5000000)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:32632",
        )
        nodes = gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:32632",
        )
        network = Network(edges=edges, nodes=nodes)

        result = nw_preps.add_endpoints(network)

        assert result.crs.to_string() == "EPSG:32632"
        assert result.edges.crs.to_string() == "EPSG:32632"
        assert result.nodes.crs.to_string() == "EPSG:32632"
        assert len(result.nodes) == 3


# ========================================================================
# Tests: merge_multilinestring / merge_multilinestrings
# ========================================================================


class TestMergeMultiLineString:
    def test_linestring_unchanged(self):
        """A LineString is returned unchanged."""
        ls = LineString([(0, 0), (1, 1)])
        result = nw_preps.merge_multilinestring(ls)

        assert shapely.get_type_id(result) == 1  # LineString
        assert shapely.equals(result, ls)

    def test_multilinestring_merged(self):
        """A contiguous MultiLineString is merged to a LineString."""
        mls = MultiLineString([[(0, 0), (1, 0)], [(1, 0), (2, 0)]])
        result = nw_preps.merge_multilinestring(mls)

        assert shapely.get_type_id(result) == 1  # LineString
        coords = shapely.get_coordinates(result)
        np.testing.assert_array_equal(coords, [[0, 0], [1, 0], [2, 0]])

    def test_non_contiguous_multilinestring_unchanged(self):
        """A non-contiguous MultiLineString stays a MultiLineString."""
        mls = MultiLineString([[(0, 0), (1, 0)], [(5, 5), (6, 6)]])
        result = nw_preps.merge_multilinestring(mls)

        # line_merge cannot merge disjoint lines
        assert shapely.get_type_id(result) == 5  # still MultiLineString

    def test_merge_multilinestrings_network(self, multilinestring_network):
        """Network-level merge converts contiguous MultiLineString edges."""
        result = nw_preps.merge_multilinestrings(multilinestring_network)

        geom = result.edges.geometry.iloc[0]
        assert shapely.get_type_id(geom) == 1  # LineString
        coords = shapely.get_coordinates(geom)
        np.testing.assert_array_equal(coords, [[0, 0], [1, 0], [2, 0]])


# ========================================================================
# Tests: find_roundabouts
# ========================================================================


class TestFindRoundabouts:
    def test_detects_ring(self, roundabout_network):
        """Ring geometries are identified as roundabouts."""
        roundabouts = nw_preps.find_roundabouts(roundabout_network)

        assert len(roundabouts) == 1
        assert shapely.predicates.is_ring(roundabouts[0].geometry)

    def test_no_roundabouts(self, simple_network):
        """A chain network has no roundabouts."""
        roundabouts = nw_preps.find_roundabouts(simple_network)

        assert len(roundabouts) == 0


# ========================================================================
# Tests: calculate_degree / add_degree
# ========================================================================


class TestDegree:
    def test_chain_degree(self, simple_network):
        """In a chain, endpoints have degree 1, inner nodes have degree 2."""
        degree = nw_preps.calculate_degree(simple_network)

        np.testing.assert_array_equal(degree, [1, 2, 2, 2, 1])

    def test_branching_degree(self, branching_network):
        """In a Y-branch, the branch node has degree 4."""
        degree = nw_preps.calculate_degree(branching_network)

        # node 0: edge 0 from → deg 1
        # node 1: edge 0 to, edge 1 from → deg 2
        # node 2: edge 1 to, edge 2 from, edge 3 from, edge 4 from → deg 4
        # node 3: edge 2 to → deg 1
        # node 4: edge 3 to → deg 1
        # node 5: edge 4 to → deg 1
        np.testing.assert_array_equal(degree, [1, 2, 4, 1, 1, 1])

    def test_empty_edges_degree(self, nodes_only_network):
        """Network with no edges returns all-zero degrees."""
        degree = nw_preps.calculate_degree(nodes_only_network)

        assert degree == [0, 0, 0]

    def test_add_degree_column(self, simple_network):
        """add_degree adds a 'degree' column to nodes."""
        result = nw_preps.add_degree(simple_network)

        assert "degree" in result.nodes.columns
        np.testing.assert_array_equal(result.nodes["degree"].values, [1, 2, 2, 2, 1])


# ========================================================================
# Tests: concat_dedup
# ========================================================================


class TestConcatDedup:
    def test_removes_duplicate_geometries(self):
        """Duplicate geometries are removed after concatenation."""
        gdf1 = gpd.GeoDataFrame(
            {"val": [1, 2], "geometry": [Point(0, 0), Point(1, 1)]},
            geometry="geometry",
        )
        gdf2 = gpd.GeoDataFrame(
            {"val": [3, 4], "geometry": [Point(1, 1), Point(2, 2)]},
            geometry="geometry",
        )
        result = nw_preps.concat_dedup([gdf1, gdf2])

        assert len(result) == 3  # Point(1,1) deduplicated
        xs = sorted([g.x for g in result.geometry])
        assert xs == [0.0, 1.0, 2.0]

    def test_no_duplicates(self):
        """Without duplicates, all rows are preserved."""
        gdf1 = gpd.GeoDataFrame(
            {"val": [1], "geometry": [Point(0, 0)]}, geometry="geometry"
        )
        gdf2 = gpd.GeoDataFrame(
            {"val": [2], "geometry": [Point(5, 5)]}, geometry="geometry"
        )
        result = nw_preps.concat_dedup([gdf1, gdf2])

        assert len(result) == 2

    def test_sequential_index(self):
        """Result index is sequential starting from 0."""
        gdf1 = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1)]}, geometry="geometry"
        )
        gdf2 = gpd.GeoDataFrame({"geometry": [Point(2, 2)]}, geometry="geometry")
        result = nw_preps.concat_dedup([gdf1, gdf2])

        np.testing.assert_array_equal(result.index.values, [0, 1, 2])


# ========================================================================
# Tests: drop_duplicate_geometries
# ========================================================================


class TestDropDuplicateGeometries:
    def test_drops_duplicates(self):
        """Duplicate geometries are removed, keeping the first."""
        gdf = gpd.GeoDataFrame(
            {"val": [1, 2, 3], "geometry": [Point(0, 0), Point(0, 0), Point(1, 1)]},
            geometry="geometry",
        )
        result = nw_preps.drop_duplicate_geometries(gdf)

        assert len(result) == 2
        assert result.iloc[0]["val"] == 1

    def test_keep_last(self):
        """With keep='last', the last duplicate is kept."""
        gdf = gpd.GeoDataFrame(
            {"val": [1, 2, 3], "geometry": [Point(0, 0), Point(0, 0), Point(1, 1)]},
            geometry="geometry",
        )
        result = nw_preps.drop_duplicate_geometries(gdf, keep="last")

        assert len(result) == 2
        assert result.iloc[0]["val"] == 2

    def test_no_duplicates(self):
        """Without duplicates, all rows are preserved."""
        gdf = gpd.GeoDataFrame(
            {"val": [1, 2], "geometry": [Point(0, 0), Point(1, 1)]},
            geometry="geometry",
        )
        result = nw_preps.drop_duplicate_geometries(gdf)

        assert len(result) == 2


# ========================================================================
# Tests: find_closest_2_edges
# ========================================================================


class TestFindClosest2Edges:
    def test_finds_two_edges(self, simple_network):
        """Returns the two closest edges to a node."""
        edges = simple_network.edges
        node_geom = Point(1, 1)  # node 1, at junction of edges 0 and 1
        edge_ids = list(range(len(edges)))

        e1, e2 = nw_preps.find_closest_2_edges(edge_ids, edges, node_geom)

        # Edges 0 and 1 touch node (1,1)
        returned_ids = sorted([e1.name, e2.name])
        assert returned_ids == [0, 1]

    def test_finds_correct_edges_at_branch(self, branching_network):
        """At branch node 2, finds two of the three edges."""
        edges = branching_network.edges
        node_geom = Point(2, 0)  # node 2
        # Edges 1,2,3,4 are near node 2
        edge_ids = [1, 2, 3, 4]

        e1, e2 = nw_preps.find_closest_2_edges(edge_ids, edges, node_geom)

        # Should return 2 of the edges incident to node 2
        assert e1.name in [1, 2, 3, 4]
        assert e2.name in [1, 2, 3, 4]
        assert e1.name != e2.name


# ========================================================================
# Tests: node_connectivity_degree
# ========================================================================


class TestNodeConnectivityDegree:
    def test_endpoint_degree(self, simple_network):
        """Endpoint node has connectivity degree 1."""
        deg = nw_preps.node_connectivity_degree(0, simple_network)

        assert deg == 1

    def test_inner_degree(self, simple_network):
        """Inner node has connectivity degree 2."""
        deg = nw_preps.node_connectivity_degree(2, simple_network)

        assert deg == 2

    def test_branch_degree(self, branching_network):
        """Branch node has connectivity degree 4."""
        deg = nw_preps.node_connectivity_degree(2, branching_network)

        assert deg == 4

    def test_isolated_node(self, simple_network):
        """A node id not referenced in any edge has degree 0."""
        deg = nw_preps.node_connectivity_degree(999, simple_network)

        assert deg == 0


# ========================================================================
# Tests: reset_ids
# ========================================================================


class TestResetIds:
    def test_sequential_ids(self, simple_network):
        """Node and edge ids are sequential after reset."""
        result = nw_preps.reset_ids(simple_network)

        np.testing.assert_array_equal(result.nodes["id"].values, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(result.edges["id"].values, [0, 1, 2, 3])

    def test_topology_preserved(self, simple_network):
        """from_id / to_id references remain valid after reset."""
        result = nw_preps.reset_ids(simple_network)

        np.testing.assert_array_equal(result.edges["from_id"].values, [0, 1, 2, 3])
        np.testing.assert_array_equal(result.edges["to_id"].values, [1, 2, 3, 4])

    def test_reset_after_gap(self):
        """Ids are compacted even when original ids have gaps."""
        nodes = gpd.GeoDataFrame(
            {
                "id": [10, 20, 30],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        edges = gpd.GeoDataFrame(
            {
                "from_id": [10, 20],
                "to_id": [20, 30],
                "id": [100, 200],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        network = Network(edges=edges, nodes=nodes)
        result = nw_preps.reset_ids(network)

        np.testing.assert_array_equal(result.nodes["id"].values, [0, 1, 2])
        np.testing.assert_array_equal(result.edges["id"].values, [0, 1])
        np.testing.assert_array_equal(result.edges["from_id"].values, [0, 1])
        np.testing.assert_array_equal(result.edges["to_id"].values, [1, 2])


# ========================================================================
# Tests: _intersects / _intersects_dataframe / intersects /
#        nodes_intersecting
# ========================================================================


class TestIntersects:
    def test_intersects_dataframe(self):
        """_intersects_dataframe returns geometries that intersect."""
        geoms = gpd.GeoSeries([Point(0, 0), Point(1, 1), Point(5, 5)], crs="EPSG:4326")
        sindex = shapely.STRtree(geoms)
        query_line = LineString([(0, 0), (2, 2)])

        result = nw_preps._intersects_dataframe(query_line, geoms, sindex)

        # Points (0,0) and (1,1) are on the line
        assert len(result) >= 2
        coords = [(shapely.get_x(g), shapely.get_y(g)) for g in result.values]
        assert (0.0, 0.0) in coords
        assert (1.0, 1.0) in coords

    def test_intersects_with_tolerance(self):
        """_intersects finds geometries within buffer tolerance."""
        geoms = gpd.GeoSeries([Point(0, 0), Point(10, 10)], crs="EPSG:4326")
        sindex = shapely.STRtree(geoms)
        query_point = Point(0.0000000001, 0)

        result = nw_preps._intersects(query_point, geoms, sindex, tolerance=1e-9)

        assert len(result) >= 1

    def test_nodes_intersecting(self):
        """nodes_intersecting finds nodes along a line."""
        nodes = gpd.GeoSeries([Point(0, 0), Point(1, 1), Point(5, 5)], crs="EPSG:4326")
        sindex = shapely.STRtree(nodes)
        line = LineString([(0, 0), (2, 2)])

        result = nw_preps.nodes_intersecting(line, nodes, sindex)

        assert len(result) >= 2


# ========================================================================
# Tests: add_distances
# ========================================================================


class TestAddDistances:
    def test_adds_distance_column(self, simple_network):
        """A 'distance' column is added to edges."""
        result = nw_preps.add_distances(simple_network)

        assert "distance" in result.edges.columns
        assert len(result.edges) == 4

    def test_distances_positive(self, simple_network):
        """All distances are positive."""
        result = nw_preps.add_distances(simple_network)

        assert (result.edges["distance"] > 0).all()

    def test_distances_approximately_equal(self, simple_network):
        """All edges have same geometry length, so distances are similar."""
        result = nw_preps.add_distances(simple_network)

        distances = result.edges["distance"].values
        # All edges go from (n,n) to (n+1,n+1), distances should be similar
        np.testing.assert_allclose(distances, distances[0], rtol=0.1)

    def test_empty_edges(self, nodes_only_network):
        """Empty edges return the network unchanged."""
        result = nw_preps.add_distances(nodes_only_network)

        assert result.edges.empty

    def test_projected_crs(self):
        """Distances are computed correctly for a projected (metric) CRS."""
        # UTM zone 32N (EPSG:32632), coordinates in metres
        edges = gpd.GeoDataFrame(
            {
                "from_id": [0],
                "to_id": [1],
                "id": [0],
                "geometry": [LineString([(500000, 5000000), (501000, 5000000)])],
            },
            geometry="geometry",
            crs="EPSG:32632",
        )
        nodes = gpd.GeoDataFrame(
            {
                "id": [0, 1],
                "geometry": [Point(500000, 5000000), Point(501000, 5000000)],
            },
            geometry="geometry",
            crs="EPSG:32632",
        )
        network = Network(edges=edges, nodes=nodes)
        result = nw_preps.add_distances(network)

        # 1000 m horizontal line
        np.testing.assert_allclose(result.edges["distance"].values, [1000.0], rtol=0.01)

    def test_geographic_crs_values(self):
        """Geodesic distances are reasonable for a known geographic edge."""
        # One degree of longitude at equator ≈ 111 km
        edges = gpd.GeoDataFrame(
            {
                "from_id": [0],
                "to_id": [1],
                "id": [0],
                "geometry": [LineString([(0, 0), (1, 0)])],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        nodes = gpd.GeoDataFrame(
            {
                "id": [0, 1],
                "geometry": [Point(0, 0), Point(1, 0)],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        network = Network(edges=edges, nodes=nodes)
        result = nw_preps.add_distances(network)

        # 1° longitude at equator ≈ 111 195 m (WGS-84 geodesic)
        np.testing.assert_allclose(result.edges["distance"].values, [111195], rtol=0.01)

    def test_no_crs_treated_as_geographic(self):
        """Edges without CRS are treated as geographic (EPSG:4326)."""
        edges = gpd.GeoDataFrame(
            {
                "from_id": [0],
                "to_id": [1],
                "id": [0],
                "geometry": [LineString([(0, 0), (1, 0)])],
            },
            geometry="geometry",
        )
        nodes = gpd.GeoDataFrame(
            {
                "id": [0, 1],
                "geometry": [Point(0, 0), Point(1, 0)],
            },
            geometry="geometry",
        )
        network = Network(edges=edges, nodes=nodes)
        result = nw_preps.add_distances(network)

        # Should produce the same result as explicit EPSG:4326
        np.testing.assert_allclose(result.edges["distance"].values, [111195], rtol=0.01)


# ========================================================================
# Tests: _ecols_to_graphorder / _vcols_to_graphorder
# ========================================================================


class TestColumnOrdering:
    def test_ecols_order(self, simple_network):
        """from_id and to_id are the first two columns."""
        result = nw_preps._ecols_to_graphorder(simple_network.edges)

        assert list(result.columns[:2]) == ["from_id", "to_id"]

    def test_vcols_order(self, simple_network):
        """id is the first column."""
        result = nw_preps._vcols_to_graphorder(simple_network.nodes)

        assert result.columns[0] == "id"

    def test_ecols_preserves_data(self, simple_network):
        """Reordering preserves all column data."""
        original = simple_network.edges
        result = nw_preps._ecols_to_graphorder(original)

        assert set(result.columns) == set(original.columns)
        np.testing.assert_array_equal(
            result["from_id"].values, original["from_id"].values
        )

    def test_vcols_preserves_data(self, simple_network):
        """Reordering preserves all column data."""
        original = simple_network.nodes
        result = nw_preps._vcols_to_graphorder(original)

        assert set(result.columns) == set(original.columns)
        np.testing.assert_array_equal(result["id"].values, original["id"].values)


# ========================================================================
# Tests: merge_edges
# ========================================================================


class TestMergeEdges:
    def test_empty_network(self, empty_network):
        """Empty network is returned unchanged."""
        result = nw_preps.merge_edges(empty_network)

        assert result.edges.empty

    def test_chain_merges_to_one_edge(self, simple_network):
        """A simple chain 0-1-2-3-4 with degree-2 inner nodes merges to 1 edge."""
        # Add degree column for merge_edges to use
        simple_network.nodes["degree"] = nw_preps.calculate_degree(simple_network)
        result = nw_preps.merge_edges(simple_network)

        # All inner nodes (1,2,3) are degree 2, so chain merges to one edge
        assert len(result.edges) == 1
        # Endpoints should connect the degree-1 nodes (0 and 4)
        edge = result.edges.iloc[0]
        endpoints = {edge["from_id"], edge["to_id"]}
        assert endpoints == {0, 4}

    def test_branch_preserves_junction(self, branching_network):
        """Degree-4 junction is preserved during merge."""
        branching_network.nodes["degree"] = nw_preps.calculate_degree(branching_network)
        result = nw_preps.merge_edges(branching_network)

        # node 1 has degree 2 → merged away
        # node 2 has degree 4 → preserved
        # Expect 4 edges: 0→2, 2→3, 2→4, 2→5 (edge 0-1 merged with 1-2)
        # Actually one edge gets merged (0-1 + 1-2 → 0-2), giving 4 edges
        assert len(result.edges) == 4


# ========================================================================
# Tests: ordered_network
# ========================================================================


class TestOrderedNetwork:
    def test_column_order(self, simple_network):
        """ordered_network reorders columns."""
        result = nw_preps.ordered_network(simple_network)

        assert list(result.edges.columns[:2]) == ["from_id", "to_id"]
        assert result.nodes.columns[0] == "id"

    def test_with_attrs(self, simple_network):
        """Additional attributes are added to edges and nodes."""
        attrs = {"test_attr": 42, "label": "abc"}
        result = nw_preps.ordered_network(simple_network, attrs=attrs)

        assert "test_attr" in result.edges.columns
        assert "test_attr" in result.nodes.columns
        assert "label" in result.edges.columns
        assert "label" in result.nodes.columns
        assert (result.edges["test_attr"] == 42).all()
        assert (result.nodes["test_attr"] == 42).all()
        assert (result.edges["label"] == "abc").all()
        assert (result.nodes["label"] == "abc").all()

    def test_does_not_modify_original(self, simple_network):
        """Original network is not modified."""
        orig_edge_cols = list(simple_network.edges.columns)
        _ = nw_preps.ordered_network(simple_network, attrs={"new": 1})

        assert list(simple_network.edges.columns) == orig_edge_cols


# ========================================================================
# Tests: simplified_network (integration)
# ========================================================================


class TestSimplifiedNetwork:
    def test_simplifies_chain(self, simple_network):
        """A chain network is simplified to fewer edges."""
        result = nw_preps.simplified_network(simple_network)

        # After simplification, degree-2 nodes are merged
        assert len(result.edges) >= 1
        assert len(result.nodes) >= 2
        assert "distance" in result.edges.columns

    def test_ids_sequential(self, simple_network):
        """All ids are sequential after simplification."""
        result = nw_preps.simplified_network(simple_network)

        np.testing.assert_array_equal(
            result.nodes["id"].values, np.arange(len(result.nodes))
        )
        np.testing.assert_array_equal(
            result.edges["id"].values, np.arange(len(result.edges))
        )

    def test_preserves_endpoint_geometries(self, simple_network):
        """Start and end point geometries of the chain are preserved."""
        result = nw_preps.simplified_network(simple_network)

        all_node_coords = [
            (shapely.get_x(g), shapely.get_y(g)) for g in result.nodes.geometry
        ]
        assert (0.0, 0.0) in all_node_coords
        assert (4.0, 4.0) in all_node_coords


# ========================================================================
# Tests: split_edges_at_nodes
# ========================================================================


class TestSplitEdgesAtNodes:
    def _make_crossing_network(self, crs):
        """Build a network where an edge passes through an interior node.

        Edge 0 goes from (0,0) to (2,0) passing through (1,0) which is a node.
        Edge 1 is a short branch from (1,0) to (1,1).
        This lets split_edges_at_nodes split edge 0 at the crossing node (1,0).
        """
        edges = gpd.GeoDataFrame(
            {
                "osm_id": [1, 2],
                "from_id": [0, 2],
                "to_id": [1, 2],
                "id": [0, 1],
                "geometry": [
                    LineString([(0, 0), (1, 0), (2, 0)]),
                    LineString([(1, 0), (1, 1)]),
                ],
            },
            geometry="geometry",
            crs=crs,
        )
        nodes = gpd.GeoDataFrame(
            {
                "id": [0, 1, 2],
                "geometry": [Point(0, 0), Point(2, 0), Point(1, 0)],
            },
            geometry="geometry",
            crs=crs,
        )
        return Network(edges=edges, nodes=nodes)

    def test_preserves_geographic_crs(self):
        """CRS is preserved through split_edges_at_nodes for geographic CRS."""
        network = self._make_crossing_network("EPSG:4326")

        result = nw_preps.split_edges_at_nodes(network)

        assert result.edges.crs.to_string() == "EPSG:4326"
        assert result.nodes.crs.to_string() == "EPSG:4326"

    def test_preserves_projected_crs(self):
        """CRS is preserved through split_edges_at_nodes for projected CRS."""
        network = self._make_crossing_network("EPSG:32632")

        result = nw_preps.split_edges_at_nodes(network)

        assert result.edges.crs.to_string() == "EPSG:32632"
        assert result.nodes.crs.to_string() == "EPSG:32632"

    def test_edges_are_geodataframe(self):
        """Result edges are a GeoDataFrame, not a plain DataFrame."""
        network = self._make_crossing_network("EPSG:4326")

        result = nw_preps.split_edges_at_nodes(network)

        assert isinstance(result.edges, gpd.GeoDataFrame)


# ========================================================================
# Tests: clean_roundabouts
# ========================================================================


class TestCleanRoundabouts:
    def test_no_roundabouts_unchanged(self, simple_network):
        """Network without roundabouts is returned with same edge count."""
        result = nw_preps.clean_roundabouts(simple_network)

        assert len(result.edges) == len(simple_network.edges)

    def test_roundabout_removed(self, roundabout_network):
        """Ring edge is removed and replaced by centroid connections."""
        result = nw_preps.clean_roundabouts(roundabout_network)

        # The ring edge should be removed
        for edge in result.edges.itertuples():
            assert not shapely.predicates.is_ring(edge.geometry)

    def test_preserves_geographic_crs(self, roundabout_network):
        """CRS is preserved through clean_roundabouts for geographic CRS."""
        assert roundabout_network.crs.to_string() == "EPSG:4326"

        result = nw_preps.clean_roundabouts(roundabout_network)

        assert result.crs.to_string() == "EPSG:4326"
        assert result.edges.crs.to_string() == "EPSG:4326"
        assert result.nodes.crs.to_string() == "EPSG:4326"

    def test_preserves_projected_crs(self):
        """CRS is preserved through clean_roundabouts for projected CRS."""
        # Roundabout in UTM zone 32N (EPSG:32632)
        ring = LineString(
            [
                (500000, 5000100),
                (500100, 5000200),
                (500200, 5000100),
                (500100, 5000000),
                (500000, 5000100),
            ]
        )
        spoke_left = LineString([(499800, 5000100), (500000, 5000100)])
        spoke_right = LineString([(500200, 5000100), (500400, 5000100)])

        edges = gpd.GeoDataFrame(
            {
                "osm_id": [1, 2, 3],
                "id": [0, 1, 2],
                "geometry": [ring, spoke_left, spoke_right],
            },
            geometry="geometry",
            crs="EPSG:32632",
        )
        nodes = gpd.GeoDataFrame(
            {
                "id": [0, 1, 2],
                "geometry": [
                    Point(499800, 5000100),
                    Point(500100, 5000100),
                    Point(500400, 5000100),
                ],
            },
            geometry="geometry",
            crs="EPSG:32632",
        )
        network = Network(edges=edges, nodes=nodes)
        result = nw_preps.clean_roundabouts(network)

        assert result.crs.to_string() == "EPSG:32632"
        assert result.edges.crs.to_string() == "EPSG:32632"
        assert result.nodes.crs.to_string() == "EPSG:32632"
        # Ring should be removed
        for edge in result.edges.itertuples():
            assert not shapely.predicates.is_ring(edge.geometry)
