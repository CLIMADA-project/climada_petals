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

import contextlib
import logging

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, LineString, MultiLineString

from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks import nw_preps

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
                    LineString([(5, 5), (5.1, 5.1)]),
                    LineString([(10, 10), (10.1, 10.1)]),
                ],
            },
            geometry="geometry",
        )
        sindex = shapely.STRtree(nodes.geometry)
        query_point = Point(4, 4).buffer(2)

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
# Tests: _get_edge_id_column / _ensure_edge_id_column
# ========================================================================


class TestGetEdgeIdColumn:
    def test_prefers_osm_id(self, simple_network):
        """osm_id is preferred over other id columns."""
        edges = simple_network.edges.copy()
        edges["edge_id"] = range(len(edges))
        edges["id"] = range(100, 100 + len(edges))

        col = nw_preps._get_edge_id_column(edges)

        assert col == "osm_id"

    def test_falls_back_to_osmid(self):
        """Falls back to osmid when osm_id is missing."""
        edges = gpd.GeoDataFrame(
            {
                "osmid": [101, 102, 103],
                "edge_id": [0, 1, 2],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                    LineString([(2, 2), (3, 3)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        col = nw_preps._get_edge_id_column(edges)

        assert col == "osmid"

    def test_falls_back_to_edge_id(self):
        """Falls back to edge_id when osm_id and osmid are missing."""
        edges = gpd.GeoDataFrame(
            {
                "edge_id": [0, 1, 2],
                "id": [100, 101, 102],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                    LineString([(2, 2), (3, 3)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        col = nw_preps._get_edge_id_column(edges)

        assert col == "edge_id"

    def test_falls_back_to_id(self):
        """Falls back to id when only id column exists."""
        edges = gpd.GeoDataFrame(
            {
                "id": [0, 1, 2],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                    LineString([(2, 2), (3, 3)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        col = nw_preps._get_edge_id_column(edges)

        assert col == "id"

    def test_returns_none_when_no_id_column(self):
        """Returns None when no id column exists."""
        edges = gpd.GeoDataFrame(
            {
                "name": ["edge1", "edge2"],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        col = nw_preps._get_edge_id_column(edges)

        assert col is None


class TestEnsureEdgeIdColumn:
    def test_preserves_osm_id(self):
        """osm_id is preserved when present."""
        edges = gpd.GeoDataFrame(
            {
                "osm_id": [10001, 10002, 10003],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                    LineString([(2, 2), (3, 3)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        result_edges, id_col = nw_preps._ensure_edge_id_column(edges)

        assert id_col == "osm_id"
        np.testing.assert_array_equal(
            result_edges["osm_id"].values, [10001, 10002, 10003]
        )

    def test_creates_temporary_column_when_missing(self):
        """Creates __edge_uid temporary column when no id column exists."""
        edges = gpd.GeoDataFrame(
            {
                "name": ["edge1", "edge2"],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        result_edges, id_col = nw_preps._ensure_edge_id_column(edges)

        assert id_col == "__edge_uid"
        assert "__edge_uid" in result_edges.columns
        np.testing.assert_array_equal(result_edges["__edge_uid"].values, [0, 1])

    def test_fills_nan_values_with_index(self):
        """NaN values in id column are filled with index."""
        edges = gpd.GeoDataFrame(
            {
                "osm_id": [10001.0, np.nan, 10003.0],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                    LineString([(2, 2), (3, 3)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        result_edges, id_col = nw_preps._ensure_edge_id_column(edges)

        assert id_col == "osm_id"
        # NaN values should be filled with corresponding index values
        assert not result_edges["osm_id"].isna().any()
        # Value at index 1 should be filled with index value 1
        assert result_edges["osm_id"].iloc[1] == 1.0

    def test_preserves_columns(self):
        """All original columns are preserved."""
        edges = gpd.GeoDataFrame(
            {
                "name": ["edge1", "edge2"],
                "length": [100, 200],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        result_edges, _ = nw_preps._ensure_edge_id_column(edges)

        assert "name" in result_edges.columns
        assert "length" in result_edges.columns
        assert "geometry" in result_edges.columns

    def test_original_dataframe_not_modified(self):
        """Original dataframe is not modified."""
        edges = gpd.GeoDataFrame(
            {
                "osm_id": [10001, 10002],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        original_columns = set(edges.columns)

        result_edges, _ = nw_preps._ensure_edge_id_column(edges)

        assert set(result_edges.columns) == original_columns


class TestEdgeIdColumnIntegration:
    def test_clean_roundabouts_with_osm_id(self, roundabout_network):
        """clean_roundabouts preserves osm_id column."""
        result = nw_preps.clean_roundabouts(roundabout_network)

        assert "osm_id" in result.edges.columns
        # osm_id values should still be present in some form
        assert len(result.edges) > 0

    def test_clean_roundabouts_without_edge_id_creates_temporary(self):
        """clean_roundabouts handles edges without id columns by creating temporary one."""
        # Create a roundabout network and remove osm_id
        ring = LineString([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        line = LineString([(0, 0), (2, 2)])

        edges = gpd.GeoDataFrame(
            {
                "name": ["roundabout", "through"],
                "geometry": [ring, line],
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

        # Should not raise error even without edge id columns
        result = nw_preps.clean_roundabouts(network)

        assert result is not None
        # Temporary __edge_uid column should not be in final output
        assert "__edge_uid" not in result.edges.columns

    def test_split_edges_at_nodes_preserves_osm_id(self):
        """split_edges_at_nodes preserves osm_id column through processing."""
        # Create a simple crossing network
        edges = gpd.GeoDataFrame(
            {
                "osm_id": [100, 101],
                "from_id": [0, 2],
                "to_id": [1, 3],
                "geometry": [
                    LineString([(0, 1), (2, 1)]),  # horizontal
                    LineString([(1, 0), (1, 2)]),  # vertical, crosses horizontal
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        nodes = gpd.GeoDataFrame(
            {
                "id": [0, 1, 2, 3],
                "geometry": [
                    Point(0, 1),
                    Point(2, 1),
                    Point(1, 0),
                    Point(1, 2),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        network = Network(edges=edges, nodes=nodes)

        result = nw_preps.split_edges_at_nodes(network)

        assert "osm_id" in result.edges.columns
        # osm_id values should be preserved/duplicated for split edges
        assert result.edges["osm_id"].notna().all()

    def test_reset_ids_after_edge_id_operations(self):
        """reset_ids works correctly after using ensure_edge_id_column."""
        # Network with osm_id but non-sequential node ids
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
                "osm_id": [100, 101],
                "id": [0, 1],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        network = Network(edges=edges, nodes=nodes)

        # Ensure edge id column doesn't break reset_ids
        result = nw_preps.reset_ids(network)

        np.testing.assert_array_equal(result.nodes["id"].values, [0, 1, 2])
        np.testing.assert_array_equal(result.edges["from_id"].values, [0, 1])
        np.testing.assert_array_equal(result.edges["to_id"].values, [1, 2])


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

    @staticmethod
    def _two_node_network(edge_geometry):
        nodes = gpd.GeoDataFrame(
            {"id": [0, 1], "geometry": [Point(0, 0), Point(1, 0)]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        edges = gpd.GeoDataFrame(
            {"geometry": [edge_geometry]}, geometry="geometry", crs="EPSG:4326"
        )
        return Network(edges=edges, nodes=nodes)

    @staticmethod
    def _run_capturing_logs(caplog, func, *args, **kwargs):
        # the climada_petals logger does not propagate to the root logger, so the
        # capture handler has to be attached to it directly
        logger = logging.getLogger("climada_petals")
        logger.addHandler(caplog.handler)
        try:
            return func(*args, **kwargs)
        finally:
            logger.removeHandler(caplog.handler)

    def test_end_within_tolerance_attached(self):
        """An edge end off its node by a floating-point difference is attached."""
        network = self._two_node_network(LineString([(0, 0), (1 + 1e-12, 0)]))

        result = nw_preps.add_topology(network)

        assert len(result.edges) == 1
        assert result.edges["from_id"].iloc[0] == 0
        assert result.edges["to_id"].iloc[0] == 1

    def test_end_beyond_tolerance_dropped_with_warning(self, caplog):
        """An edge whose end has no node within tolerance is dropped and logged."""
        network = self._two_node_network(LineString([(0, 0), (1.5, 0)]))

        result = self._run_capturing_logs(caplog, nw_preps.add_topology, network)

        assert result.edges.empty
        assert "add_topology dropped 1 of 1 edges" in caplog.text

    def test_larger_tolerance_attaches(self):
        """A larger tolerance attaches ends that are further away."""
        network = self._two_node_network(LineString([(0, 0), (1.1, 0)]))

        result = nw_preps.add_topology(network, tolerance=0.2)

        assert result.edges["to_id"].iloc[0] == 1

    def test_multilinestring_dropped_with_warning(self, caplog):
        """Edges without a LineString geometry cannot be attached and are logged."""
        network = self._two_node_network(
            MultiLineString([[(0, 0), (0.5, 0)], [(0.5, 0), (1, 0)]])
        )

        result = self._run_capturing_logs(caplog, nw_preps.add_topology, network)

        assert result.edges.empty
        assert "add_topology dropped 1 of 1 edges" in caplog.text

    def test_custom_id_col(self, simple_network):
        """from_id / to_id take the values of the given node column."""
        network = nw_preps.add_ids(simple_network)
        network.nodes["node_ref"] = [10, 11, 12, 13, 14]

        result = nw_preps.add_topology(network, id_col="node_ref")

        np.testing.assert_array_equal(result.edges["from_id"].values, [10, 11, 12, 13])
        np.testing.assert_array_equal(result.edges["to_id"].values, [11, 12, 13, 14])


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

    def test_get_endpoints_inherits_ci_type(self, simple_network):
        """Endpoints inherit the ci_type of the edge they belong to."""
        edges = simple_network.edges.copy()
        edges["ci_type"] = ["road", "road", "power_line", "power_line"]
        network = Network(edges=edges, nodes=simple_network.nodes.copy())

        endpoints = nw_preps.get_endpoints(network)

        assert "ci_type" in endpoints.columns
        assert endpoints["ci_type"].tolist() == [
            "road",
            "road",
            "road",
            "road",
            "power_line",
            "power_line",
            "power_line",
            "power_line",
        ]

    def test_get_endpoints_only_propagates_requested_attrs(self, simple_network):
        """Edge-only attributes (distance, ids, topology) are not propagated."""
        edges = simple_network.edges.copy()
        edges["ci_type"] = "road"
        network = Network(edges=edges, nodes=simple_network.nodes.copy())

        endpoints = nw_preps.get_endpoints(network)

        assert set(endpoints.columns) == {"geometry", "ci_type"}

    def test_get_endpoints_without_ci_type_column(self, simple_network):
        """No attribute column is added if the edges do not have it."""
        assert "ci_type" not in simple_network.edges.columns

        endpoints = nw_preps.get_endpoints(simple_network)

        assert list(endpoints.columns) == ["geometry"]

    def test_get_endpoints_no_attrs(self, simple_network):
        """attrs=None or empty disables attribute propagation."""
        edges = simple_network.edges.copy()
        edges["ci_type"] = "road"
        network = Network(edges=edges, nodes=simple_network.nodes.copy())

        assert list(nw_preps.get_endpoints(network, attrs=None).columns) == ["geometry"]
        assert list(nw_preps.get_endpoints(network, attrs=()).columns) == ["geometry"]

    def test_get_endpoints_multilinestring_inherits_ci_type(self):
        """Endpoints of each part of a MultiLineString inherit the edge ci_type."""
        edges = gpd.GeoDataFrame(
            {
                "ci_type": ["road"],
                "geometry": [
                    MultiLineString([[(0, 0), (1, 0)], [(2, 0), (3, 0)]]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        network = Network(edges=edges)

        endpoints = nw_preps.get_endpoints(network)

        assert len(endpoints) == 4
        assert (endpoints["ci_type"] == "road").all()

    def test_add_endpoints_nodes_inherit_ci_type(self, simple_network):
        """Nodes created from edge endpoints carry the edge ci_type."""
        edges = simple_network.edges.copy()
        edges["ci_type"] = "road"
        network = Network(
            edges=edges,
            nodes=gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        )

        result = nw_preps.add_endpoints(network)

        assert len(result.nodes) == 5
        assert result.nodes["ci_type"].notna().all()
        assert (result.nodes["ci_type"] == "road").all()

    def test_add_endpoints_keeps_existing_node_attrs(self, simple_network):
        """An existing node at an endpoint keeps its own ci_type."""
        edges = simple_network.edges.copy()
        edges["ci_type"] = "road"
        nodes = gpd.GeoDataFrame(
            {"ci_type": ["healthcare"], "geometry": [Point(4, 4)]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        network = Network(edges=edges, nodes=nodes)

        result = nw_preps.add_endpoints(network)

        assert len(result.nodes) == 5
        ci_by_coords = {
            (shapely.get_x(geom), shapely.get_y(geom)): ci
            for geom, ci in zip(result.nodes.geometry, result.nodes["ci_type"])
        }
        assert ci_by_coords[(4.0, 4.0)] == "healthcare"
        assert all(
            ci == "road" for coords, ci in ci_by_coords.items() if coords != (4.0, 4.0)
        )

    def test_add_endpoints_custom_inherit_attrs(self, simple_network):
        """inherit_attrs controls which edge columns are copied to new nodes."""
        edges = simple_network.edges.copy()
        edges["ci_type"] = "road"
        edges["operator"] = "utility_a"
        network = Network(
            edges=edges,
            nodes=gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        )

        result = nw_preps.add_endpoints(network, inherit_attrs=("operator",))

        assert (result.nodes["operator"] == "utility_a").all()
        assert "ci_type" not in result.nodes.columns
        assert "distance" not in result.nodes.columns


# ========================================================================
# Tests: merge_multilinestring / merge_multilinestrings
# ========================================================================


class TestExplodeMultiLineStrings:
    def test_parts_become_edges_with_attributes(self):
        mls = MultiLineString([[(0, 0), (1, 0)], [(5, 5), (6, 5)]])
        line = LineString([(10, 0), (11, 0)])
        edges = gpd.GeoDataFrame(
            {"highway": ["primary", "track"], "geometry": [mls, line]},
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = nw_preps.explode_multilinestrings(Network(edges=edges))

        assert len(result.edges) == 3
        assert result.edges["highway"].tolist() == ["primary", "primary", "track"]
        assert result.edges.geometry.iloc[2].equals(line)
        assert result.edges.crs.to_string() == "EPSG:4326"

    def test_no_multilinestrings_unchanged(self, simple_network):
        result = nw_preps.explode_multilinestrings(simple_network)

        assert len(result.edges) == len(simple_network.edges)


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

        assert isinstance(degree, np.ndarray)
        np.testing.assert_array_equal(degree, [0, 0, 0])

    def test_isolated_last_node_degree_zero(self, simple_network):
        """A node without edges at the end of the id range has degree 0."""
        isolated = gpd.GeoDataFrame(
            {"id": [5], "geometry": [Point(10, 10)]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        nodes = gpd.GeoDataFrame(
            pd.concat([simple_network.nodes, isolated], ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )
        network = Network(edges=simple_network.edges.copy(), nodes=nodes)

        degree = nw_preps.calculate_degree(network)

        np.testing.assert_array_equal(degree, [1, 2, 2, 2, 1, 0])

    def test_edge_id_beyond_nodes_raises(self, simple_network):
        """An edge referring to a node id >= number of nodes raises."""
        edges = simple_network.edges.copy()
        edges.loc[3, "to_id"] = 7
        network = Network(edges=edges, nodes=simple_network.nodes.copy())

        with pytest.raises(ValueError, match="to_id"):
            nw_preps.calculate_degree(network)

    def test_integral_float_ids_accepted(self, simple_network):
        """Float node references with integral values are accepted."""
        edges = simple_network.edges.copy()
        edges["from_id"] = edges["from_id"].astype(float)
        network = Network(edges=edges, nodes=simple_network.nodes.copy())

        degree = nw_preps.calculate_degree(network)

        np.testing.assert_array_equal(degree, [1, 2, 2, 2, 1])

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
        assert len(result) == 2
        coords = [(shapely.get_x(g), shapely.get_y(g)) for g in result.values]
        assert (0.0, 0.0) in coords
        assert (1.0, 1.0) in coords

    def test_intersects_with_tolerance(self):
        """_intersects finds geometries within buffer tolerance."""
        geoms = gpd.GeoSeries([Point(0, 0), Point(10, 10)], crs="EPSG:4326")
        sindex = shapely.STRtree(geoms)
        query_point = Point(0.0000000001, 0)

        result = nw_preps._intersects(query_point, geoms, sindex, tolerance=1e-9)

        assert len(result) == 1

    def test_nodes_intersecting(self):
        """nodes_intersecting finds nodes along a line."""
        nodes = gpd.GeoSeries([Point(0, 0), Point(1, 1), Point(5, 5)], crs="EPSG:4326")
        sindex = shapely.STRtree(nodes)
        line = LineString([(0, 0), (2, 2)])

        result = nw_preps.nodes_intersecting(line, nodes, sindex)

        assert len(result) == 2


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

    def test_no_crs_raises(self):
        """Edges without CRS raise instead of being silently treated as degrees."""
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

        with pytest.raises(ValueError, match="no CRS"):
            nw_preps.add_distances(network)


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
        np.testing.assert_array_equal(result["to_id"].values, original["to_id"].values)

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

    def test_keeps_columns_and_values(self, simple_network):
        """Only the column order changes, not the columns or their values."""
        result = nw_preps.ordered_network(simple_network)

        assert set(result.edges.columns) == set(simple_network.edges.columns)
        assert set(result.nodes.columns) == set(simple_network.nodes.columns)
        pd.testing.assert_frame_equal(
            result.edges[simple_network.edges.columns], simple_network.edges
        )
        pd.testing.assert_frame_equal(
            result.nodes[simple_network.nodes.columns], simple_network.nodes
        )

    def test_does_not_modify_original(self, simple_network):
        """Original network is not modified."""
        orig_edge_cols = list(simple_network.edges.columns)
        orig_node_cols = list(simple_network.nodes.columns)
        result = nw_preps.ordered_network(simple_network)

        assert list(result.edges.columns) == orig_edge_cols
        assert list(result.nodes.columns) == orig_node_cols


# ========================================================================
# Tests: add_attributes
# ========================================================================


class TestAddAttributes:
    def test_on_both(self, simple_network):
        """By default, attributes are added to edges and nodes."""
        attrs = {"test_attr": 42, "label": "abc"}
        result = nw_preps.add_attributes(simple_network, attrs)

        for gdf in (result.edges, result.nodes):
            assert (gdf["test_attr"] == 42).all()
            assert (gdf["label"] == "abc").all()

    def test_on_edges(self, simple_network):
        """on='edges' only adds attributes to edges."""
        result = nw_preps.add_attributes(
            simple_network, {"ci_type": "road"}, on="edges"
        )

        assert (result.edges["ci_type"] == "road").all()
        assert "ci_type" not in result.nodes.columns

    def test_on_nodes(self, simple_network):
        """on='nodes' only adds attributes to nodes."""
        result = nw_preps.add_attributes(
            simple_network, {"ci_type": "road"}, on="nodes"
        )

        assert (result.nodes["ci_type"] == "road").all()
        assert "ci_type" not in result.edges.columns

    def test_overwrites_existing_attribute(self, simple_network):
        """An existing column is overwritten with the new value."""
        simple_network.nodes["ci_type"] = "people"
        result = nw_preps.add_attributes(simple_network, {"ci_type": "road"})

        assert (result.nodes["ci_type"] == "road").all()
        assert (result.edges["ci_type"] == "road").all()

    def test_empty_attrs(self, simple_network):
        """An empty dict leaves the network unchanged."""
        orig_edges = simple_network.edges.copy()
        orig_nodes = simple_network.nodes.copy()
        result = nw_preps.add_attributes(simple_network, {})

        pd.testing.assert_frame_equal(result.edges, orig_edges)
        pd.testing.assert_frame_equal(result.nodes, orig_nodes)

    def test_modifies_in_place(self, simple_network):
        """add_attributes modifies and returns the input network."""
        result = nw_preps.add_attributes(simple_network, {"ci_type": "road"})

        assert result is simple_network
        assert (simple_network.edges["ci_type"] == "road").all()

    def test_after_ordered_network_keeps_order(self, simple_network):
        """ordered_network followed by add_attributes keeps igraph column order."""
        result = nw_preps.add_attributes(
            nw_preps.ordered_network(simple_network), {"ci_type": "road"}
        )

        assert list(result.edges.columns[:2]) == ["from_id", "to_id"]
        assert result.nodes.columns[0] == "id"
        assert result.edges.columns[-1] == "ci_type"
        assert result.nodes.columns[-1] == "ci_type"


# ========================================================================
# Tests: simplified_network (integration)
# ========================================================================


class TestSimplifiedNetwork:
    def test_simplifies_chain(self, simple_network):
        """A chain network is simplified to fewer edges."""
        result = nw_preps.simplified_network(simple_network)

        # After simplification, degree-2 nodes are merged
        assert len(result.edges) == 1
        assert len(result.nodes) == 2
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

    def test_nodes_inherit_ci_type(self, edges_gdf):
        """Nodes created by simplifying a line network keep the edge ci_type."""
        edges = edges_gdf.copy()
        edges["ci_type"] = "road"
        network = Network(edges=edges)  # line network without nodes

        result = nw_preps.simplified_network(network)

        assert len(result.nodes) >= 2
        assert result.nodes["ci_type"].notna().all()
        assert (result.nodes["ci_type"] == "road").all()
        assert "distance" not in result.nodes.columns


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

    @staticmethod
    def _network(lines, points, **edge_cols):
        edges = gpd.GeoDataFrame(
            {**edge_cols, "geometry": [LineString(c) for c in lines]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        nodes = gpd.GeoDataFrame(
            {"id": list(range(len(points))), "geometry": [Point(p) for p in points]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        return Network(edges=edges, nodes=nodes)

    @staticmethod
    def _lengths(network):
        return sorted(round(g.length, 9) for g in network.edges.geometry)

    def test_node_between_vertices_splits_by_default(self):
        network = self._network([[(0, 0), (2, 0)]], [(0, 0), (1, 0), (2, 0)])

        result = nw_preps.split_edges_at_nodes(network)

        assert self._lengths(result) == [1.0, 1.0]

    def test_node_between_vertices_ignored_in_vertex_mode(self):
        network = self._network([[(0, 0), (2, 0)]], [(0, 0), (1, 0), (2, 0)])

        result = nw_preps.split_edges_at_nodes(network, mode="vertices")

        assert self._lengths(result) == [2.0]

    def test_vertex_mode_splits_at_shared_vertex(self):
        network = self._network([[(0, 0), (1, 0), (2, 0)]], [(0, 0), (1, 0), (2, 0)])

        result = nw_preps.split_edges_at_nodes(network, mode="vertices")

        assert self._lengths(result) == [1.0, 1.0]

    def test_vertex_mode_no_spurious_split(self):
        """Per point, not per coordinate: vertex (0,3) is not a split point."""
        network = self._network(
            [[(0, 0), (0, 3), (2, 3), (4, 0)]], [(0, 0), (2, 3), (4, 0)]
        )

        result = nw_preps.split_edges_at_nodes(network, mode="vertices")

        assert len(result.edges) == 2

    def test_crossing_without_shared_vertex(self):
        """Two crossing edges are split at the crossing by default only."""
        lines = [[(0, 0), (2, 0)], [(1, -1), (1, 1)]]
        points = [(0, 0), (2, 0), (1, -1), (1, 1)]

        default = nw_preps.split_edges_at_nodes(self._network(lines, points))
        vertex_mode = nw_preps.split_edges_at_nodes(
            self._network(lines, points), mode="vertices"
        )

        assert self._lengths(default) == [1.0, 1.0, 1.0, 1.0]
        assert self._lengths(vertex_mode) == [2.0, 2.0]

    def test_pieces_copy_attributes(self):
        network = self._network(
            [[(0, 0), (2, 0)]],
            [(0, 0), (1, 0), (2, 0)],
            osm_id=[7],
            highway=["primary"],
        )

        result = nw_preps.split_edges_at_nodes(network)

        assert result.edges["osm_id"].tolist() == [7, 7]
        assert result.edges["highway"].tolist() == ["primary", "primary"]

    def test_duplicate_edges_not_split_at_their_vertices(self):
        """An identical second edge does not add split points."""
        line = [(0, 0), (1, 0), (2, 0)]
        network = self._network([line, line], [(0, 0), (2, 0)])

        result = nw_preps.split_edges_at_nodes(network)

        assert self._lengths(result) == [2.0, 2.0]

    def test_partial_overlap_split_at_its_ends_only(self):
        """A shared stretch splits at its start and end, not at inner vertices."""
        long_way = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        short_way = [(1, 0), (2, 0), (3, 0)]  # shares (1,0)-(3,0) with long_way
        network = self._network([long_way, short_way], [(0, 0), (4, 0)])

        result = nw_preps.split_edges_at_nodes(network)

        # long_way: cut at (1,0) and (3,0) -> 1, 2, 1; short_way: not cut -> 2
        assert self._lengths(result) == [1.0, 1.0, 2.0, 2.0]

    def test_invalid_mode_raises(self):
        network = self._network([[(0, 0), (2, 0)]], [(0, 0), (2, 0)])

        with pytest.raises(ValueError, match="mode must be"):
            nw_preps.split_edges_at_nodes(network, mode="between")

    def test_multilinestring_unchanged(self):
        mls = MultiLineString([[(0, 0), (1, 0)], [(1, 0), (2, 0)]])
        edges = gpd.GeoDataFrame(
            {"geometry": [mls]}, geometry="geometry", crs="EPSG:4326"
        )
        nodes = gpd.GeoDataFrame(
            {"id": [0], "geometry": [Point(1, 0)]}, geometry="geometry", crs="EPSG:4326"
        )

        result = nw_preps.split_edges_at_nodes(Network(edges=edges, nodes=nodes))

        assert len(result.edges) == 1
        assert result.edges.geometry.iloc[0].equals(mls)


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

    # diamond-shaped ring around (1, 1) and a second one around (5, 1)
    _RING_A = [(0.5, 1), (1, 1.5), (1.5, 1), (1, 0.5), (0.5, 1)]
    _RING_B = [(4.5, 1), (5, 1.5), (5.5, 1), (5, 0.5), (4.5, 1)]

    @staticmethod
    def _edges_network(lines, crs="EPSG:4326"):
        edges = gpd.GeoDataFrame(
            {"geometry": [LineString(c) for c in lines]}, geometry="geometry", crs=crs
        )
        return Network(edges=edges)

    @staticmethod
    def _end_points(network):
        # rounded: ring centroids are computed and not exact
        return {
            frozenset(
                (
                    tuple(np.round(shapely.get_coordinates(g)[0], 9)),
                    tuple(np.round(shapely.get_coordinates(g)[-1], 9)),
                )
            )
            for g in network.edges.geometry
        }

    def test_edge_between_two_roundabouts_extended_at_both_ends(self):
        network = self._edges_network(
            [self._RING_A, self._RING_B, [(1.5, 1), (4.5, 1)]]
        )

        result = nw_preps.clean_roundabouts(network)

        assert self._end_points(result) == {frozenset(((1.0, 1.0), (5.0, 1.0)))}

    def test_crossing_road_not_connected(self):
        """A road crossing the ring without an end on it is left unchanged."""
        crossing = [(1, 0), (1, 2)]  # crosses the ring at (1, 0.5) and (1, 1.5)
        network = self._edges_network([self._RING_A, crossing])

        result = nw_preps.clean_roundabouts(network)

        assert len(result.edges) == 1
        assert result.edges.geometry.iloc[0].equals(LineString(crossing))

    def test_edge_with_both_ends_on_ring_removed(self):
        """A chord between two points of the ring is removed, spokes are kept."""
        network = self._edges_network(
            [self._RING_A, [(0.5, 1), (1.5, 1)], [(0, 1), (0.5, 1)]]
        )

        result = nw_preps.clean_roundabouts(network)

        assert self._end_points(result) == {frozenset(((0.0, 1.0), (1.0, 1.0)))}

    def test_max_length_keeps_longer_rings(self):
        """Rings longer than max_length (metres) are not roundabouts."""
        # ring of about 566 m circumference in UTM coordinates
        ring = [
            (500000, 5000100),
            (500100, 5000200),
            (500200, 5000100),
            (500100, 5000000),
            (500000, 5000100),
        ]
        spoke = [(499800, 5000100), (500000, 5000100)]

        kept = nw_preps.clean_roundabouts(
            self._edges_network([ring, spoke], crs="EPSG:32632"), max_length=100
        )
        collapsed = nw_preps.clean_roundabouts(
            self._edges_network([ring, spoke], crs="EPSG:32632"), max_length=1000
        )

        assert len(kept.edges) == 2
        assert len(collapsed.edges) == 1

    def test_nodes_on_ring_removed_other_nodes_kept(self):
        """Nodes on a collapsed ring are removed; all other nodes are kept."""
        edges = gpd.GeoDataFrame(
            {"geometry": [LineString(self._RING_A), LineString([(0, 1), (0.5, 1)])]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        nodes = gpd.GeoDataFrame(
            {
                "id": [0, 1, 2, 3],
                "geometry": [
                    Point(0.5, 1),  # on the ring, where the spoke joins
                    Point(1, 1.5),  # ring vertex without a road
                    Point(0, 1),  # spoke end, off the ring
                    Point(10, 10),  # elsewhere
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = nw_preps.clean_roundabouts(Network(edges=edges, nodes=nodes))

        coords = {(g.x, g.y) for g in result.nodes.geometry}
        assert coords == {(0.0, 1.0), (10.0, 10.0)}

    def test_warns_when_topology_becomes_outdated(self, caplog):
        """Existing from_id / to_id trigger a warning when a ring is collapsed."""
        edges = gpd.GeoDataFrame(
            {
                "from_id": [0, 1],
                "to_id": [0, 0],
                "geometry": [
                    LineString(self._RING_A),
                    LineString([(0, 1), (0.5, 1)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        network = Network(edges=edges)

        logger = logging.getLogger("climada_petals")
        logger.addHandler(caplog.handler)
        try:
            nw_preps.clean_roundabouts(network)
        finally:
            logger.removeHandler(caplog.handler)

        assert "from_id / to_id are outdated" in caplog.text

    def test_no_topology_warning_without_roundabouts(self, simple_network, caplog):
        logger = logging.getLogger("climada_petals")
        logger.addHandler(caplog.handler)
        try:
            nw_preps.clean_roundabouts(simple_network)
        finally:
            logger.removeHandler(caplog.handler)

        assert "outdated" not in caplog.text

    def test_max_length_without_crs_raises(self):
        edges = gpd.GeoDataFrame(
            {"geometry": [LineString(self._RING_A)]}, geometry="geometry"
        )

        with pytest.raises(ValueError, match="no CRS"):
            nw_preps.clean_roundabouts(Network(edges=edges), max_length=100)


# ========================================================================
# Known bugs: regression tests (Step 0 of claude/nw-preps-update-plan.md)
# ========================================================================
# Each test below documents a confirmed bug and is marked
# xfail(strict=True): it fails today, and the plan step that fixes the bug
# must remove the marker. A fix that forgets to remove it makes the suite
# fail (strict XPASS), so no fixed bug stays silently marked.


def _xfail(step, reason):
    """xfail marker for a known bug fixed in the given step of the update plan."""
    return pytest.mark.xfail(
        strict=True, reason=f"Known bug, fixed in plan step {step}: {reason}"
    )


def _copy_on_write():
    """Context with pandas copy-on-write enabled (always on from pandas 3)."""
    if int(pd.__version__.split(".")[0]) >= 3:
        return contextlib.nullcontext()
    return pd.option_context("mode.copy_on_write", True)


class TestKnownBugsIndexRobustness:
    def test_intersects_dataframe_non_range_index(self):
        geoms = gpd.GeoSeries(
            [Point(0, 0), Point(5, 5), Point(1, 1)],
            index=[10, 20, 30],
            crs="EPSG:4326",
        )
        sindex = shapely.STRtree(geoms)

        result = nw_preps._intersects_dataframe(
            LineString([(0, 0), (2, 2)]), geoms, sindex
        )

        assert set(result.index) == {10, 30}

    def test_drop_duplicate_geometries_non_range_index(self):
        line_a = LineString([(0, 0), (1, 1)])
        line_b = LineString([(1, 1), (2, 2)])
        gdf = gpd.GeoDataFrame(
            {"geometry": [line_a, line_a, line_b]},
            index=[10, 11, 12],
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = nw_preps.drop_duplicate_geometries(gdf)

        assert list(result.index) == [10, 12]

    def test_clean_roundabouts_non_range_index(self, roundabout_network):
        """clean_roundabouts gives the same geometries for a non-0..n-1 index."""
        edges = roundabout_network.edges.copy()
        edges.index = [10, 20, 30]
        shifted = Network(edges=edges, nodes=roundabout_network.nodes.copy())

        expected = nw_preps.clean_roundabouts(roundabout_network)
        result = nw_preps.clean_roundabouts(shifted)

        assert sorted(shapely.to_wkb(g) for g in result.edges.geometry) == sorted(
            shapely.to_wkb(g) for g in expected.edges.geometry
        )


class TestKnownBugsIds:
    def test_calculate_degree_rejects_unordered_node_ids(self, simple_network):
        nodes = simple_network.nodes.copy()
        nodes["id"] = [4, 3, 2, 1, 0]
        network = Network(edges=simple_network.edges.copy(), nodes=nodes)

        with pytest.raises(ValueError, match="node ids"):
            nw_preps.calculate_degree(network)

    def test_calculate_degree_nan_id_clear_error(self, simple_network):
        edges = simple_network.edges.copy()
        edges["from_id"] = edges["from_id"].astype(float)
        edges.loc[0, "from_id"] = np.nan
        network = Network(edges=edges, nodes=simple_network.nodes.copy())

        with pytest.raises(ValueError, match="from_id"):
            nw_preps.calculate_degree(network)

    def test_reset_ids_dangling_reference_raises(self, simple_network):
        edges = simple_network.edges.copy()
        edges.loc[0, "from_id"] = 99
        network = Network(edges=edges, nodes=simple_network.nodes.copy())

        with pytest.raises(ValueError, match="from_id"):
            nw_preps.reset_ids(network)


class TestKnownBugsTopology:
    def test_nearest_finds_nearest_point(self):
        nodes = gpd.GeoDataFrame(
            {"id": [0, 1], "geometry": [Point(0, 0), Point(5, 5)]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        sindex = shapely.STRtree(nodes.geometry)

        result = nw_preps.nearest(Point(4, 4), nodes, sindex)

        assert result["id"] == 1

    def test_nearest_max_distance_raises(self):
        nodes = gpd.GeoDataFrame(
            {"id": [0, 1], "geometry": [Point(0, 0), Point(5, 5)]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        sindex = shapely.STRtree(nodes.geometry)

        with pytest.raises(ValueError, match="max_distance"):
            nw_preps.nearest(Point(4, 4), nodes, sindex, max_distance=0.5)


class TestKnownBugsMergeEdges:
    def test_merge_edges_keeps_isolated_node(self, edges_gdf, nodes_gdf):
        isolated = gpd.GeoDataFrame(
            {"id": [5], "geometry": [Point(10, 10)]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        nodes = gpd.GeoDataFrame(
            pd.concat([nodes_gdf, isolated], ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )
        network = Network(edges=edges_gdf.copy(), nodes=nodes)
        network.nodes["degree"] = nw_preps.calculate_degree(network)

        result = nw_preps.merge_edges(network)

        assert len(result.edges) == 1
        assert any(geom.equals(Point(10, 10)) for geom in result.nodes.geometry)

    def test_merge_edges_removes_only_inner_nodes(self, simple_network):
        """Inner chain nodes are removed, the chain ends are kept."""
        simple_network.nodes["degree"] = nw_preps.calculate_degree(simple_network)

        result = nw_preps.merge_edges(simple_network)

        coords = {(g.x, g.y) for g in result.nodes.geometry}
        assert coords == {(0.0, 0.0), (4.0, 4.0)}

    def test_merge_edges_does_not_modify_input(self, simple_network):
        """The input network's nodes and degree column are left unchanged."""
        simple_network.nodes["degree"] = nw_preps.calculate_degree(simple_network)
        degree_before = simple_network.nodes["degree"].tolist()

        nw_preps.merge_edges(simple_network)

        assert len(simple_network.nodes) == 5
        assert simple_network.nodes["degree"].tolist() == degree_before

    @staticmethod
    def _network(points, edge_pairs, **edge_cols):
        """Network with point nodes and straight edges between node pairs."""
        nodes = gpd.GeoDataFrame(
            {"id": list(range(len(points))), "geometry": [Point(p) for p in points]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        edges = gpd.GeoDataFrame(
            {
                "from_id": [a for a, _ in edge_pairs],
                "to_id": [b for _, b in edge_pairs],
                "id": list(range(len(edge_pairs))),
                **edge_cols,
                "geometry": [LineString([points[a], points[b]]) for a, b in edge_pairs],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        return Network(edges=edges, nodes=nodes)

    @staticmethod
    def _chain_arrays():
        """J -e0- A -e1- B -e2- C -e3- K as nodes 0..4 and edges 0..3."""
        from_ids = np.array([0, 1, 2, 3])
        to_ids = np.array([1, 2, 3, 4])
        deg = np.array([1, 2, 2, 2, 1])
        incident = [[0], [0, 1], [1, 2], [2, 3], [3]]
        return deg, incident, from_ids, to_ids

    def test_walk_chain_both_sides(self):
        """From B, the two walks end at J and K and pass A and C."""
        deg, incident, from_ids, to_ids = self._chain_arrays()
        visited_nodes = {2}
        visited_edges = {1, 2}

        side_1 = nw_preps._walk_chain(
            1, 1, deg, incident, from_ids, to_ids, visited_nodes, visited_edges
        )
        side_2 = nw_preps._walk_chain(
            3, 2, deg, incident, from_ids, to_ids, visited_nodes, visited_edges
        )

        assert side_1 == (0, [1], [0])
        assert side_2 == (4, [3], [3])
        # nothing passed in is modified
        assert visited_nodes == {2}
        assert visited_edges == {1, 2}
        np.testing.assert_array_equal(deg, [1, 2, 2, 2, 1])

    def test_walk_chain_stops_at_non_degree_2_node(self):
        """A walk starting at a node of degree != 2 returns it immediately."""
        deg, incident, from_ids, to_ids = self._chain_arrays()

        result = nw_preps._walk_chain(
            0, 0, deg, incident, from_ids, to_ids, {1}, {0, 1}
        )

        assert result == (0, [], [])

    @staticmethod
    def _merge_capturing_logs(caplog, network, **kwargs):
        # the climada_petals logger does not propagate to the root logger, so the
        # capture handler has to be attached to it directly
        logger = logging.getLogger("climada_petals")
        logger.addHandler(caplog.handler)
        try:
            return nw_preps.merge_edges(network, **kwargs)
        finally:
            logger.removeHandler(caplog.handler)

    # chain 0 -(length 1)- 1 -(length 3)- 2
    _CHAIN_POINTS = [(0, 0), (1, 0), (4, 0)]
    _CHAIN_PAIRS = [(0, 1), (1, 2)]

    def test_merge_attrs_equal_values_kept_without_warning(self, caplog):
        network = self._network(
            self._CHAIN_POINTS,
            self._CHAIN_PAIRS,
            ci_type=["road", "road"],
            highway=["primary", "primary"],
        )

        result = self._merge_capturing_logs(caplog, network)

        assert len(result.edges) == 1
        assert result.edges["highway"].iloc[0] == "primary"
        assert "attribute values differed" not in caplog.text

    def test_merge_attrs_length_weighted_majority(self, caplog):
        """The value covering most of the length wins, not the alphabetical one."""
        network = self._network(
            self._CHAIN_POINTS,
            self._CHAIN_PAIRS,
            ci_type=["road", "road"],
            highway=["primary", "secondary"],
        )

        result = self._merge_capturing_logs(caplog, network)

        assert result.edges["highway"].iloc[0] == "secondary"
        assert "highway (1 edges)" in caplog.text

    def test_merge_attrs_nan_on_conflict(self, caplog):
        network = self._network(
            self._CHAIN_POINTS,
            self._CHAIN_PAIRS,
            ci_type=["road", "road"],
            highway=["primary", "secondary"],
        )

        result = self._merge_capturing_logs(caplog, network, attr_conflict="nan")

        assert pd.isna(result.edges["highway"].iloc[0])
        assert result.edges["ci_type"].iloc[0] == "road"
        assert "highway (1 edges)" in caplog.text

    def test_merge_attrs_external_id_of_longest_edge(self, caplog):
        network = self._network(self._CHAIN_POINTS, self._CHAIN_PAIRS, osm_id=[10, 20])

        result = self._merge_capturing_logs(caplog, network)

        assert result.edges["osm_id"].iloc[0] == 20
        assert "attribute values differed" not in caplog.text

    def test_merge_attrs_custom_id_cols(self, caplog):
        """A column passed in id_cols is treated as an external id."""
        network = self._network(
            self._CHAIN_POINTS, self._CHAIN_PAIRS, source_id=["a", "b"]
        )

        result = self._merge_capturing_logs(caplog, network, id_cols=("source_id",))

        assert result.edges["source_id"].iloc[0] == "b"
        assert "attribute values differed" not in caplog.text

    def test_merge_attrs_osm_id_as_attribute_without_id_cols(self, caplog):
        """With id_cols=(), osm_id is resolved like any other attribute."""
        network = self._network(self._CHAIN_POINTS, self._CHAIN_PAIRS, osm_id=[10, 20])

        result = self._merge_capturing_logs(caplog, network, id_cols=())

        assert result.edges["osm_id"].iloc[0] == 20  # longer edge = majority
        assert "osm_id (1 edges)" in caplog.text

    def test_merge_attrs_distance_summed(self):
        network = self._network(
            self._CHAIN_POINTS, self._CHAIN_PAIRS, distance=[100.0, 300.0]
        )

        result = nw_preps.merge_edges(network)

        assert result.edges["distance"].iloc[0] == pytest.approx(400.0)

    def test_merge_attrs_invalid_conflict_rule_raises(self, simple_network):
        with pytest.raises(ValueError, match="attr_conflict"):
            nw_preps.merge_edges(simple_network, attr_conflict="mode")

    def test_merge_edges_two_independent_chains(self):
        """Two separate chains are merged independently."""
        points = [(0, 0), (1, 0), (2, 0), (0, 5), (1, 5), (2, 5)]
        network = self._network(
            points, [(0, 1), (1, 2), (3, 4), (4, 5)], ci_type=["road"] * 4
        )

        result = nw_preps.merge_edges(network)

        pairs = {frozenset((e.from_id, e.to_id)) for e in result.edges.itertuples()}
        assert pairs == {frozenset((0, 2)), frozenset((3, 5))}

    def test_merge_edges_closed_loop_unchanged(self):
        """A closed loop made only of degree-2 nodes is left unchanged."""
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        network = self._network(
            points, [(0, 1), (1, 2), (2, 3), (3, 0)], ci_type=["road"] * 4
        )

        result = nw_preps.merge_edges(network)

        assert len(result.edges) == 4
        assert len(result.nodes) == 4

    def test_merge_edges_loop_at_junction(self):
        """A loop hanging at a junction becomes one closed edge at the junction."""
        # junction 0 (degree 3: tail edge + both loop ends), loop 0-1-2-0
        points = [(0, 0), (1, 1), (1, -1), (-1, 0)]
        network = self._network(
            points, [(0, 1), (1, 2), (2, 0), (3, 0)], ci_type=["road"] * 4
        )

        result = nw_preps.merge_edges(network)

        assert len(result.edges) == 2
        loop = [e for e in result.edges.itertuples() if e.from_id == e.to_id]
        assert len(loop) == 1
        assert result.nodes.geometry.iloc[loop[0].from_id].equals(Point(0, 0))

    def test_merge_edges_requires_range_node_ids(self, simple_network):
        """merge_edges raises if node ids are not 0..n-1 in row order."""
        nodes = simple_network.nodes.copy()
        nodes["id"] = [4, 3, 2, 1, 0]
        network = Network(edges=simple_network.edges.copy(), nodes=nodes)

        with pytest.raises(ValueError, match="node ids"):
            nw_preps.merge_edges(network)

    def test_merge_edges_without_degree_column(self, simple_network):
        assert "degree" not in simple_network.nodes.columns

        result = nw_preps.merge_edges(simple_network)

        assert len(result.edges) == 1

    def test_merge_edges_copy_on_write(self, simple_network):
        simple_network.nodes["degree"] = nw_preps.calculate_degree(simple_network)

        with _copy_on_write():
            result = nw_preps.merge_edges(simple_network)

        assert len(result.edges) == 1

    def test_merge_edges_ignores_crossing_edge(self):
        # Node 1 has degree 2 (edges 1 and 2). Edge 0 crosses node 1 without
        # being connected to it and has the lowest id, so it is picked first.
        nodes = gpd.GeoDataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "geometry": [
                    Point(0, 0),
                    Point(1, 0),
                    Point(2, 0),
                    Point(1, -1),
                    Point(1, 1),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        edges = gpd.GeoDataFrame(
            {
                "from_id": [3, 0, 1],
                "to_id": [4, 1, 2],
                "id": [0, 1, 2],
                "osm_id": [30, 10, 11],
                # an attribute column, so the test does not depend on the
                # "no optional columns" bug fixed in Step 6c
                "ci_type": ["road", "road", "road"],
                "geometry": [
                    LineString([(1, -1), (1, 1)]),
                    LineString([(0, 0), (1, 0)]),
                    LineString([(1, 0), (2, 0)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        network = Network(edges=edges, nodes=nodes)
        network.nodes["degree"] = nw_preps.calculate_degree(network)

        result = nw_preps.merge_edges(network)

        assert len(result.edges) == 2
        pairs = {frozenset((e.from_id, e.to_id)) for e in result.edges.itertuples()}
        assert frozenset((0, 2)) in pairs

    def test_merge_edges_without_optional_columns(self, edges_gdf, nodes_gdf):
        edges = edges_gdf[["from_id", "to_id", "id", "geometry"]].copy()
        network = Network(edges=edges, nodes=nodes_gdf.copy())
        network.nodes["degree"] = nw_preps.calculate_degree(network)

        result = nw_preps.merge_edges(network)

        assert len(result.edges) == 1


class TestKnownBugsSplitEdgesAtNodes:
    def test_no_spurious_split(self):
        # Real split points: (0,0), (2,3), (4,0). Vertex (0,3) takes its x
        # from (0,0) and its y from (2,3) and must not be split.
        edges = gpd.GeoDataFrame(
            {"id": [0], "geometry": [LineString([(0, 0), (0, 3), (2, 3), (4, 0)])]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        nodes = gpd.GeoDataFrame(
            {"id": [0, 1, 2], "geometry": [Point(0, 0), Point(2, 3), Point(4, 0)]},
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = nw_preps.split_edges_at_nodes(Network(edges=edges, nodes=nodes))

        assert len(result.edges) == 2

    def test_keeps_dead_end(self):
        edges = gpd.GeoDataFrame(
            {"id": [0], "geometry": [LineString([(0, 0), (1, 0), (2, 0)])]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        nodes = gpd.GeoDataFrame(
            {"id": [0], "geometry": [Point(0, 0)]},
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = nw_preps.split_edges_at_nodes(Network(edges=edges, nodes=nodes))

        assert len(result.edges) == 1
        assert result.edges.geometry.iloc[0].length == pytest.approx(2.0)

    def test_keeps_isolated_edge(self):
        edges = gpd.GeoDataFrame(
            {"id": [0], "geometry": [LineString([(0, 0), (1, 0), (2, 0)])]},
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = nw_preps.split_edges_at_nodes(Network(edges=edges))

        assert len(result.edges) == 1


class TestKnownBugsCleanRoundabouts:
    def test_ring_really_removed(self, roundabout_network):
        result = nw_preps.clean_roundabouts(roundabout_network)

        ring_interior = {(1.0, 1.5), (1.0, 0.5)}
        for geom in result.edges.geometry:
            coords = {tuple(c) for c in shapely.get_coordinates(geom)}
            assert not coords & ring_interior
        assert len(result.edges) == 2

    def test_shared_osm_id_at_two_roundabouts(self):
        # Two roundabouts A (centre (1,1)) and B (centre (5,1)), joined by two
        # segments of the same OSM way (osm_id 7). Each segment must be
        # extended to its own roundabout centre, and both must be kept.
        ring_a = LineString([(0.5, 1), (1, 1.5), (1.5, 1), (1, 0.5), (0.5, 1)])
        ring_b = LineString([(4.5, 1), (5, 1.5), (5.5, 1), (5, 0.5), (4.5, 1)])
        edges = gpd.GeoDataFrame(
            {
                "osm_id": [1, 2, 7, 7],
                "geometry": [
                    ring_a,
                    ring_b,
                    LineString([(1.5, 1), (3, 1)]),
                    LineString([(3, 1), (4.5, 1)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = nw_preps.clean_roundabouts(Network(edges=edges))

        # rounded: the centroid is computed and not exactly (1, 1) / (5, 1)
        endpoints = {
            frozenset(
                (
                    tuple(np.round(shapely.get_coordinates(g)[0], 9)),
                    tuple(np.round(shapely.get_coordinates(g)[-1], 9)),
                )
            )
            for g in result.edges.geometry
        }
        assert frozenset(((1.0, 1.0), (3.0, 1.0))) in endpoints
        assert frozenset(((3.0, 1.0), (5.0, 1.0))) in endpoints


class TestKnownBugsSimplifiedNetwork:
    def test_multilinestring_edge_survives(self, multilinestring_network):
        result = nw_preps.simplified_network(multilinestring_network)

        assert len(result.edges) == 1
        assert shapely.get_type_id(result.edges.geometry.iloc[0]) == 1  # LineString

    def test_mixed_ci_type_raises(self, edges_gdf):
        edges = edges_gdf.copy()
        edges["ci_type"] = ["road", "road", "power_line", "power_line"]

        with pytest.raises(ValueError, match="ci_type"):
            nw_preps.simplified_network(Network(edges=edges))

    def test_node_ci_type_different_from_edges_raises(self, edges_gdf, nodes_gdf):
        """Nodes of another ci_type (e.g. facilities) are rejected."""
        edges = edges_gdf.copy()
        edges["ci_type"] = "road"
        nodes = nodes_gdf.copy()
        nodes["ci_type"] = "healthcare"

        with pytest.raises(ValueError, match="same single ci_type"):
            nw_preps.simplified_network(Network(edges=edges, nodes=nodes))

    def test_removes_isolated_nodes_with_warning(self, edges_gdf, nodes_gdf, caplog):
        """Nodes of the same ci_type that touch no edge are removed and logged."""
        edges = edges_gdf.copy()
        edges["ci_type"] = "road"
        isolated = gpd.GeoDataFrame(
            {"id": [5], "geometry": [Point(10, 10)]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        nodes = gpd.GeoDataFrame(
            pd.concat([nodes_gdf, isolated], ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )
        nodes["ci_type"] = "road"

        logger = logging.getLogger("climada_petals")
        logger.addHandler(caplog.handler)
        try:
            result = nw_preps.simplified_network(Network(edges=edges, nodes=nodes))
        finally:
            logger.removeHandler(caplog.handler)

        assert not any(g.equals(Point(10, 10)) for g in result.nodes.geometry)
        assert "Removed 1 of" in caplog.text

    def test_nodes_only_network_keeps_all_nodes(self, nodes_only_network):
        """A network without edges is returned unchanged, with all its nodes."""
        result = nw_preps.simplified_network(nodes_only_network)

        assert len(result.nodes) == len(nodes_only_network.nodes)
        assert "degree" not in result.nodes.columns

    def test_disconnected_multilinestring_split_into_edges(self):
        """Unconnected parts of a MultiLineString become separate edges."""
        mls = MultiLineString([[(0, 0), (1, 0)], [(5, 5), (6, 5)]])
        edges = gpd.GeoDataFrame(
            {"osm_id": [9], "ci_type": ["road"], "geometry": [mls]},
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = nw_preps.simplified_network(Network(edges=edges))

        assert len(result.edges) == 2
        assert (shapely.get_type_id(result.edges.geometry.to_numpy()) == 1).all()
        assert result.edges["osm_id"].tolist() == [9, 9]

    def test_duplicated_stretch_still_merged(self):
        """Duplicates are removed before the topology, so the chain merges."""
        points = [(0, 0), (1, 0), (2, 0)]
        edges = gpd.GeoDataFrame(
            {
                "ci_type": ["road"] * 3,
                "geometry": [
                    LineString([points[0], points[1]]),
                    LineString([points[0], points[1]]),  # duplicate
                    LineString([points[1], points[2]]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = nw_preps.simplified_network(Network(edges=edges))

        assert len(result.edges) == 1
        assert result.edges.geometry.iloc[0].length == pytest.approx(2.0)

    # ring of about 566 m circumference (UTM) with one road joining it
    _UTM_RING = [
        (500000, 5000100),
        (500100, 5000200),
        (500200, 5000100),
        (500100, 5000000),
        (500000, 5000100),
    ]
    _UTM_SPOKE = [(499800, 5000100), (500000, 5000100)]

    def _ring_network(self):
        edges = gpd.GeoDataFrame(
            {
                "ci_type": ["road", "road"],
                "geometry": [LineString(self._UTM_RING), LineString(self._UTM_SPOKE)],
            },
            geometry="geometry",
            crs="EPSG:32632",
        )
        return Network(edges=edges)

    @staticmethod
    def _has_ring(network):
        return bool(shapely.is_ring(network.edges.geometry.to_numpy()).any())

    def test_unknown_profile_raises(self, simple_network):
        with pytest.raises(ValueError, match="Unknown profile"):
            nw_preps.simplified_network(simple_network, profile="railway")

    def test_road_profile_collapses_roundabouts_generic_does_not(self):
        generic = nw_preps.simplified_network(self._ring_network())
        road = nw_preps.simplified_network(self._ring_network(), profile="road")

        assert self._has_ring(generic)
        assert not self._has_ring(road)

    def test_profile_setting_overridden(self):
        """A keyword argument overrides the profile's setting."""
        result = nw_preps.simplified_network(
            self._ring_network(), profile="road", roundabouts=False
        )

        assert self._has_ring(result)

    def test_roundabout_max_length_overridden(self):
        """A smaller limit than the ring length keeps the ring."""
        result = nw_preps.simplified_network(
            self._ring_network(), profile="road", roundabout_max_length=100
        )

        assert self._has_ring(result)

    def test_split_connects_crossing_edges(self):
        """With split="anywhere", two crossing edges meet in a node of degree 4."""
        edges = gpd.GeoDataFrame(
            {
                "ci_type": ["road", "road"],
                "geometry": [
                    LineString([(0, 0), (2, 0)]),
                    LineString([(1, -1), (1, 1)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        unsplit = nw_preps.simplified_network(Network(edges=edges.copy()))
        split = nw_preps.simplified_network(
            Network(edges=edges.copy()), split="anywhere"
        )

        assert (len(unsplit.edges), len(unsplit.nodes)) == (2, 4)
        assert (len(split.edges), len(split.nodes)) == (4, 5)
        assert max(nw_preps.calculate_degree(split)) == 4

    def test_invalid_split_raises(self, simple_network):
        with pytest.raises(ValueError, match="split must be"):
            nw_preps.simplified_network(simple_network, split=True)

    @pytest.mark.parametrize(
        "profile, expected", [("generic", (2, 4)), ("road", (4, 5)), ("power_line", (4, 5))]
    )
    def test_profiles_split_crossing_edges(self, profile, expected):
        """road and power_line connect crossing edges by default, generic does not."""
        edges = gpd.GeoDataFrame(
            {
                "ci_type": ["x", "x"],
                "geometry": [
                    LineString([(0, 0), (2, 0)]),
                    LineString([(1, -1), (1, 1)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = nw_preps.simplified_network(Network(edges=edges), profile=profile)

        assert (len(result.edges), len(result.nodes)) == expected

    def test_osm_ways_crossing_at_inner_vertex_connected(self):
        """Ways sharing an inner vertex (usual OSM junction) end up connected."""
        edges = gpd.GeoDataFrame(
            {
                "ci_type": ["road", "road"],
                "geometry": [
                    LineString([(0, 0), (1, 0), (2, 0)]),
                    LineString([(1, -1), (1, 0), (1, 1)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        result = nw_preps.simplified_network(Network(edges=edges), profile="road")
        graph = result.to_graph(directed=False)

        assert len(graph.connected_components()) == 1
        assert max(nw_preps.calculate_degree(result)) == 4

    def test_ci_type_argument_sets_type(self, simple_network):
        result = nw_preps.simplified_network(simple_network, ci_type="road")

        assert (result.edges["ci_type"] == "road").all()
        assert (result.nodes["ci_type"] == "road").all()

    def test_ci_type_argument_conflict_raises(self, simple_network):
        simple_network.edges["ci_type"] = "power_line"

        with pytest.raises(ValueError, match="was given"):
            nw_preps.simplified_network(simple_network, ci_type="road")

    def test_ci_type_argument_on_nodes_only_network(self, nodes_only_network):
        result = nw_preps.simplified_network(nodes_only_network, ci_type="people")

        assert (result.nodes["ci_type"] == "people").all()

    def test_summary_logged(self, simple_network, caplog):
        simple_network.edges["ci_type"] = "road"
        logger = logging.getLogger("climada_petals")
        level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(caplog.handler)
        try:
            nw_preps.simplified_network(simple_network)
        finally:
            logger.removeHandler(caplog.handler)
            logger.setLevel(level)

        assert "simplified_network: 4 edges in, 1 out" in caplog.text

    def test_missing_ci_type_warns(self, simple_network, caplog):
        """A warning is logged when the edges have no ci_type."""
        assert "ci_type" not in simple_network.edges.columns

        # the climada_petals logger does not propagate to the root logger, so the
        # capture handler has to be attached to it directly
        logger = logging.getLogger("climada_petals")
        logger.addHandler(caplog.handler)
        try:
            nw_preps.simplified_network(simple_network)
        finally:
            logger.removeHandler(caplog.handler)

        assert "Edges have no ci_type" in caplog.text

    def test_single_ci_type_no_warning(self, simple_network, caplog):
        """No ci_type warning when all edges share one ci_type."""
        simple_network.edges["ci_type"] = "road"

        logger = logging.getLogger("climada_petals")
        logger.addHandler(caplog.handler)
        try:
            nw_preps.simplified_network(simple_network)
        finally:
            logger.removeHandler(caplog.handler)

        assert "ci_type" not in caplog.text
