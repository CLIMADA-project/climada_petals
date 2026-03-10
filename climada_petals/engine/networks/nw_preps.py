"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

-------

clean gdataframes with network data and convert to a nodes & edges structure
compatible for igraph graph calculations
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import shapely
from tqdm import tqdm

from climada.util.coordinates import compute_geodesic_lengths
from climada_petals.engine.networks.nw_base import Network

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel("INFO")


# =============================================================================
# Simplification methods from ElcoK/trails/simplify
# =============================================================================

"""
all functions taken and slight modified from:
https://github.com/ElcoK/trails/blob/main/src/trails/simplify.py
"""


def add_ids(network, id_col="id"):
    """Add or replace an id column with ascending ids.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).
    id_col : str, optional
        Column name for the id. Default: 'id'.

    Returns
    -------
    Network
        Network with reset indices and new id columns.
    """
    nodes = network.nodes.copy()
    edges = network.edges.copy()

    if not nodes.empty:
        nodes = nodes.reset_index(drop=True)

    if not edges.empty:
        edges = edges.reset_index(drop=True)

    nodes[id_col] = range(len(nodes))
    edges[id_col] = range(len(edges))

    return Network(edges, nodes)


def add_topology(network, id_col="id"):
    """Add or replace from_id, to_id columns to edges.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).
    id_col : str, optional
        Column name for the id. Default: 'id'.

    Returns
    -------
    Network
        Network with from_id and to_id columns added to edges.
    """

    from_ids = []
    to_ids = []
    bugs = []

    nodes = network.nodes.copy()
    edges = network.edges.copy()

    sindex = shapely.STRtree(nodes.geometry)
    for edge in tqdm(edges.itertuples(), desc="topology", total=len(edges)):
        start, end = line_endpoints(edge.geometry)
        try:
            start_node = nearest_node(start, nodes, sindex)
            from_ids.append(start_node[id_col])
        except:
            bugs.append(edge.id)
            from_ids.append(-1)
        try:
            end_node = nearest_node(end, nodes, sindex)
            to_ids.append(end_node[id_col])
        except:
            bugs.append(edge.id)
            to_ids.append(-1)

    edges["from_id"] = from_ids
    edges["to_id"] = to_ids
    edges = edges.loc[~(edges.id.isin(list(bugs)))].reset_index(drop=True)

    return Network(edges, nodes)


def line_endpoints(line):
    """Return points at first and last vertex of a line.

    Parameters
    ----------
    line : shapely.LineString
        A line geometry.

    Returns
    -------
    start : shapely.Point
        Point at the start of the line.
    end : shapely.Point
        Point at the end of the line.
    """
    start = shapely.get_point(line, 0)
    end = shapely.get_point(line, -1)
    return start, end


def nearest(geom, dataframe, sindex):
    """Find the element of a GeoDataFrame nearest a geometry.

    Parameters
    ----------
    geom : shapely.Geometry
        Reference geometry to find the nearest element to.
    dataframe : gpd.GeoDataFrame
        GeoDataFrame with geometries to search.
    sindex : shapely.STRtree
        Spatial index of the dataframe geometries.

    Returns
    -------
    pandas.Series
        Row from the dataframe nearest to ``geom``.
    """
    matches_idx = sindex.query(geom)
    nearest_geom = min(
        [dataframe.iloc[match_idx] for match_idx in matches_idx],
        key=lambda match: shapely.measurement.distance(match.geometry, geom),
    )
    return nearest_geom


def nearest_node(point, nodes, sindex):
    """Find the nearest node to a point.

    Parameters
    ----------
    point : shapely.Point
        Reference point geometry.
    nodes : gpd.GeoDataFrame
        GeoDataFrame of network nodes.
    sindex : shapely.STRtree
        Spatial index of the node geometries.

    Returns
    -------
    pandas.Series
        Row from nodes nearest to ``point``.
    """
    return nearest(point, nodes, sindex)


def get_endpoints(network):
    """Get nodes for each edge endpoint.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with point geometries at each edge endpoint.
    """
    endpoints = []
    for edge in tqdm(
        network.edges.itertuples(), desc="endpoints", total=len(network.edges)
    ):
        if edge.geometry is None:
            continue
        # 5 is MULTILINESTRING
        if shapely.get_type_id(edge.geometry) == 5:
            for line in edge.geometry.geoms:
                start, end = line_endpoints(line)
                endpoints.append(start)
                endpoints.append(end)
        else:
            start, end = line_endpoints(edge.geometry)
            endpoints.append(start)
            endpoints.append(end)

    # create dataframe to match the nodes geometry column name
    crs = network.edges.crs if network.edges.crs is not None else "EPSG:4326"
    return gpd.GeoDataFrame(geometry=endpoints, crs=crs)


def add_endpoints(network):
    """Add nodes at line endpoints.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).

    Returns
    -------
    Network
        Network with endpoint nodes added.
    """

    endpoints = get_endpoints(network)

    nodes = network.nodes.copy()
    edges = network.edges.copy()

    nodes = concat_dedup([nodes, endpoints])

    return Network(edges, nodes)


def merge_multilinestrings(network):
    """Merge all MultiLineString geometries into LineString geometries.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).

    Returns
    -------
    Network
        Network with merged edge geometries.
    """
    nodes = network.nodes.copy()
    edges = network.edges.copy()

    edges["geometry"] = edges.geometry.apply(lambda x: merge_multilinestring(x))

    return Network(edges, nodes)


def merge_multilinestring(geom):
    """Merge a MultiLineString to a LineString.

    Parameters
    ----------
    geom : shapely.Geometry
        A shapely geometry, most likely a LineString or a MultiLineString.

    Returns
    -------
    shapely.Geometry
        A shapely LineString geometry if the merge was successful.
        Otherwise, the original geometry is returned unchanged.
    """
    if shapely.get_type_id(geom) == 5:
        geom_inb = shapely.line_merge(geom)
        if geom_inb.is_ring:  # still something to fix if desired
            return geom_inb
        return geom_inb
    return geom


def find_roundabouts(network):
    """Find roundabout edges in the network.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).

    Returns
    -------
    list
        Edges that can be identified as roundabouts (ring geometries).
    """
    roundabouts = []
    for edge in network.edges.itertuples():
        if shapely.predicates.is_ring(edge.geometry):
            roundabouts.append(edge)
    return roundabouts


def clean_roundabouts(network):
    """Clean roundabouts and junctions in the network.

    Should be called before splitting edges at nodes to avoid logic conflicts.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).

    Returns
    -------
    Network
        Network with roundabout geometries replaced by centroid connections.
    """
    # TODO: Is reference to osm_id really necessary? Remove by alternative?

    edges = network.edges.copy()
    nodes = network.nodes.copy()

    sindex = shapely.STRtree(edges["geometry"])
    new_edge = []
    remove_edge = []
    new_edge_id = []
    attributes = [x for x in edges.columns if x not in ["geometry", "osm_id"]]

    roundabouts = find_roundabouts(network)
    for roundabout in roundabouts:
        round_centroid = shapely.constructive.centroid(roundabout.geometry)
        remove_edge.append(roundabout.Index)

        edges_intersect = _intersects(roundabout.geometry, edges["geometry"], sindex)
        # index at e[0] geometry at e[1] of edges that intersect with
        for edg in edges_intersect.items():
            edge = edges.iloc[edg[0]]
            start = shapely.get_point(edg[1], 0)
            end = shapely.get_point(edg[1], -1)
            first_co_is_closer = shapely.measurement.distance(
                end, round_centroid
            ) > shapely.measurement.distance(start, round_centroid)
            co_ords = shapely.coordinates.get_coordinates(edge.geometry)
            centroid_co = shapely.coordinates.get_coordinates(round_centroid)
            if first_co_is_closer:
                new_co = np.concatenate((centroid_co, co_ords))
            else:
                new_co = np.concatenate((co_ords, centroid_co))
            snap_line = shapely.linestrings(new_co)

            snap_line = shapely.linestrings(new_co)

            # an edge should never connect to>  2 roundabouts, if it does this will break
            if edge.osm_id in new_edge_id:
                a = []
                counter = 0
                for x in new_edge:
                    if x[0] == edge.osm_id:
                        a = counter
                        break
                    counter += 1
                double_edge = new_edge.pop(a)
                start = shapely.get_point(double_edge[-1], 0)
                end = shapely.get_point(double_edge[-1], -1)
                first_co_is_closer = shapely.measurement.distance(
                    end, round_centroid
                ) > shapely.measurement.distance(start, round_centroid)
                co_ords = shapely.coordinates.get_coordinates(double_edge[-1])
                if first_co_is_closer:
                    new_co = np.concatenate((centroid_co, co_ords))
                else:
                    new_co = np.concatenate((co_ords, centroid_co))
                snap_line = shapely.linestrings(new_co)
                new_edge.append(
                    [edge.osm_id] + list(edge[list(attributes)]) + [snap_line]
                )

            else:
                new_edge.append(
                    [edge.osm_id] + list(edge[list(attributes)]) + [snap_line]
                )
                new_edge_id.append(edge.osm_id)
            remove_edge.append(edg[0])

    new = gpd.GeoDataFrame(
        new_edge, columns=["osm_id"] + attributes + ["geometry"], crs=edges.crs
    )
    edges = edges.loc[~edges.index.isin(remove_edge)]
    edges = pd.concat([edges, new]).reset_index(drop=True)

    return Network(edges, nodes)


def calculate_degree(network):
    """Calculate the degree of each node from from_id and to_id columns.

    Should not be called after removing nodes or edges without first
    resetting the ids.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).

    Returns
    -------
    np.ndarray
        Array of connectivity degrees for each node.
    """
    if network.edges.empty:
        return [0] * len(network.nodes)
    # the number of nodes(from index) to use as the number of bins
    ndC = len(network.nodes.index)
    if ndC - 1 > max(network.edges.from_id) and ndC - 1 > max(network.edges.to_id):
        print("Calculate_degree possibly unhappy")
    return np.bincount(network.edges["from_id"], None, ndC) + np.bincount(
        network.edges["to_id"], None, ndC
    )


def add_degree(network):
    """Add a degree column to the node GeoDataFrame.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).

    Returns
    -------
    Network
        Network with a 'degree' column added to nodes.
    """
    degree = calculate_degree(network)

    edges = network.edges.copy()
    nodes = network.nodes.copy()
    nodes["degree"] = degree

    return Network(edges, nodes)


def concat_dedup(dataframes):
    """Concatenate a list of GeoDataFrames, dropping duplicate geometries.

    Repeatedly drops indices for deduplication to work.

    Parameters
    ----------
    dataframes : list of gpd.GeoDataFrame
        GeoDataFrames to concatenate.

    Returns
    -------
    gpd.GeoDataFrame
        Concatenated GeoDataFrame with duplicate geometries removed.
    """
    cat = pd.concat(dataframes, axis=0, sort=False)
    cat.reset_index(drop=True, inplace=True)
    cat_dedup = drop_duplicate_geometries(cat)
    cat_dedup.reset_index(drop=True, inplace=True)
    return cat_dedup


def find_closest_2_edges(edgeIDs, edges, nodGeometry):
    """Find the two edges closest to a given node geometry.

    Parameters
    ----------
    edgeIDs : list of int
        Indices of candidate edges. Modified in-place (first match removed).
    edges : gpd.GeoDataFrame
        GeoDataFrame of network edges.
    nodGeometry : shapely.Point
        Geometry of the node to find connected edges for.

    Returns
    -------
    edge_path_1 : pandas.Series
        The closest edge to the node.
    edge_path_2 : pandas.Series
        The second closest edge to the node.
    """
    edge_path_1 = min(
        [edges.iloc[match_idx] for match_idx in edgeIDs],
        key=lambda match: shapely.distance(nodGeometry, match.geometry),
    )
    edgeIDs.remove(edge_path_1.name)
    edge_path_2 = min(
        [edges.iloc[match_idx] for match_idx in edgeIDs],
        key=lambda match: shapely.distance(nodGeometry, match.geometry),
    )
    return edge_path_1, edge_path_2


def merge_edges(network, print_err=False):
    """Remove degree-2 nodes and merge their associated edges.

    Finds nodes of degree 2 and their associated 2 edges, then traverses
    edges and nodes in both directions until a node of degree != 2 is found.
    Resets the geometry and from/to ids for the merged edge, and deletes the
    traversed nodes and edges. Uses the mode of edge attributes for the
    merged edge's column values.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).
    print_err : bool, optional
        Whether to print error messages for failed merges. Default: False.

    Returns
    -------
    Network
        Network with degree-2 nodes removed and edges merged.
    """
    if network.edges.empty:
        return network

    nodes = network.nodes.copy()
    edges = network.edges.copy()

    optional_cols = edges.columns.difference(
        ["osm_id", "geometry", "from_id", "to_id", "id"]
    )
    edg_sindex = shapely.STRtree(edges.geometry)

    if "degree" not in nodes.columns:
        deg = calculate_degree(network)
    else:
        deg = nodes["degree"].to_numpy()
    degree2 = np.where(deg == 2)
    n2 = set((nodes["id"].iloc[degree2]))

    nodGeom = nodes["geometry"]
    eIDtoRemove = []

    # make progressbar with tqdm(total=len(n2))
    while n2:
        newEdge = []
        info_first_edge = []
        possibly_delete = []
        pos_0_deg = []
        nodeID = n2.pop()
        pos_0_deg.append(nodeID)
        # Co-ordinates of current node
        node_geometry = nodGeom[nodeID]
        eID = set(edg_sindex.query(node_geometry, predicate="intersects"))
        # Find the nearest 2 edges, unless there is an error in the dataframe
        # this will return the connected edges using spatial indexing
        if len(eID) > 2:
            edge_path_1, edge_path_2 = find_closest_2_edges(eID, edges, node_geometry)
        elif len(eID) < 2:
            continue
        else:
            edge_path_1 = edges.iloc[eID.pop()]
            edge_path_2 = edges.iloc[eID.pop()]
        # For the two edges found, identify the next 2 nodes in either direction
        next_node_1 = (
            edge_path_1.to_id if edge_path_1.from_id == nodeID else edge_path_1.from_id
        )
        next_node_2 = (
            edge_path_2.to_id if edge_path_2.from_id == nodeID else edge_path_2.from_id
        )
        if next_node_1 == next_node_2:
            continue
        possibly_delete.append(edge_path_2.id)
        # At the moment the first edge information is used for the merged edge
        info_first_edge = edge_path_1.id
        newEdge.append(edge_path_1.geometry)
        newEdge.append(edge_path_2.geometry)
        # While the next node along the path is degree 2 keep traversing
        while deg[next_node_1] == 2:
            if next_node_1 in pos_0_deg:
                break
            next_node_1Geom = nodGeom[next_node_1]
            eID = set(edg_sindex.query(next_node_1Geom, predicate="intersects"))
            eID.discard(edge_path_1.id)
            try:
                edge_path_1 = min(
                    [edges.iloc[match_idx] for match_idx in eID],
                    key=lambda match: shapely.distance(
                        next_node_1Geom, (match.geometry)
                    ),
                )
            except:
                continue
            pos_0_deg.append(next_node_1)
            n2.discard(next_node_1)
            next_node_1 = (
                edge_path_1.to_id
                if edge_path_1.from_id == next_node_1
                else edge_path_1.from_id
            )
            newEdge.append(edge_path_1.geometry)
            possibly_delete.append(edge_path_1.id)

        while deg[next_node_2] == 2:
            if next_node_2 in pos_0_deg:
                break
            next_node_2Geom = nodGeom[next_node_2]
            eID = set(edg_sindex.query(next_node_2Geom, predicate="intersects"))
            eID.discard(edge_path_2.id)
            try:
                edge_path_2 = min(
                    [edges.iloc[match_idx] for match_idx in eID],
                    key=lambda match: shapely.distance(
                        next_node_2Geom, (match.geometry)
                    ),
                )
            except:
                continue
            pos_0_deg.append(next_node_2)
            n2.discard(next_node_2)
            next_node_2 = (
                edge_path_2.to_id
                if edge_path_2.from_id == next_node_2
                else edge_path_2.from_id
            )
            newEdge.append(edge_path_2.geometry)
            possibly_delete.append(edge_path_2.id)
        # Update the information of the first edge
        new_merged_geom = shapely.line_merge(shapely.multilinestrings([newEdge]))
        if shapely.get_type_id(new_merged_geom) == 1:
            edges.at[info_first_edge, "geometry"] = new_merged_geom
            if nodGeom[next_node_1] == shapely.get_point(new_merged_geom, 0):
                edges.at[info_first_edge, "from_id"] = next_node_1
                edges.at[info_first_edge, "to_id"] = next_node_2
            else:
                edges.at[info_first_edge, "from_id"] = next_node_2
                edges.at[info_first_edge, "to_id"] = next_node_1
            eIDtoRemove += possibly_delete
            possibly_delete.append(info_first_edge)
            for x in pos_0_deg:
                deg[x] = 0
            mode_edges = edges.loc[edges.id.isin(possibly_delete)]
            edges.loc[info_first_edge, optional_cols] = (
                mode_edges[optional_cols].mode().iloc[0].values
            )
        else:
            if print_err:
                print(
                    "Line",
                    info_first_edge,
                    "failed to merge, has shapely type ",
                    shapely.get_type_id(edges.at[info_first_edge, "geometry"]),
                )

    edges = edges.loc[~(edges.id.isin(eIDtoRemove))].reset_index(drop=True)

    # We remove all degree 0 nodes, including those found in dropHanging
    nodes = nodes.loc[nodes.degree > 0].reset_index(drop=True)

    return Network(edges, nodes)


def node_connectivity_degree(node, network):
    """Get the degree of connectivity for a node.

    Parameters
    ----------
    node : int
        Node id to query.
    network : Network
        A network composed of nodes (points in space) and edges (lines).

    Returns
    -------
    int
        Number of edges connected to the node.
    """
    return len(
        network.edges[(network.edges.from_id == node) | (network.edges.to_id == node)]
    )


def drop_duplicate_geometries(dataframe, keep="first"):
    """Drop duplicate geometries from a GeoDataFrame.

    Converts geometries to WKB so that ``drop_duplicates`` works correctly,
    as discussed in https://github.com/geopandas/geopandas/issues/521.

    Parameters
    ----------
    dataframe : gpd.GeoDataFrame
        GeoDataFrame from which to remove duplicate geometries.
    keep : str, optional
        Which duplicates to keep. Default: 'first'.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with duplicate geometries removed.
    """

    mask = dataframe.geometry.apply(lambda geom: shapely.to_wkb(geom))
    # use dropped duplicates index to drop from actual dataframe
    return dataframe.iloc[mask.drop_duplicates(keep=keep).index]


def reset_ids(network):
    """Reset the ids of nodes and edges to sequential integers.

    Updates from_id and to_id references in the edge table using
    dictionary mapping.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).

    Returns
    -------
    Network
        Network with sequentially renumbered node and edge ids.
    """
    # Copy nodes and edges to avoid modifying the original data
    nodes = network.nodes.copy()
    edges = network.edges.copy()

    # Generate new node IDs
    new_node_ids = np.arange(len(nodes))

    # Map old node IDs to new IDs using a dictionary
    id_map = dict(zip(nodes["id"], new_node_ids))

    # Efficiently map old IDs to new IDs in 'to_id' and 'from_id' columns
    edges["from_id"] = edges["from_id"].map(id_map)
    edges["to_id"] = edges["to_id"].map(id_map)

    # Update node and edge IDs
    nodes["id"] = new_node_ids
    edges["id"] = np.arange(len(edges))

    # Reset indices to ensure a clean, sequential index
    edges.reset_index(drop=True, inplace=True)
    nodes.reset_index(drop=True, inplace=True)

    return Network(edges, nodes)


def split_edges_at_nodes(network):
    """Split network edges where they intersect node geometries.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).

    Returns
    -------
    Network
        Network with edges split at node intersection points.
    """
    sindex_nodes = shapely.STRtree(network.nodes["geometry"])
    sindex_edges = shapely.STRtree(network.edges["geometry"])
    attributes = [
        x for x in network.edges.columns if x not in ["index", "geometry", "osm_id"]
    ]
    grab_all_edges = []

    # TODO: this takes really long. Rewrite?
    for edge in tqdm(
        network.edges.itertuples(index=False),
        desc="splitting",
        total=len(network.edges),
    ):
        hits_nodes = nodes_intersecting(
            edge.geometry, network.nodes["geometry"], sindex_nodes, tolerance=1e-9
        )
        hits_edges = nodes_intersecting(
            edge.geometry, network.edges["geometry"], sindex_edges, tolerance=1e-9
        )
        hits_edges = shapely.set_operations.intersection(edge.geometry, hits_edges)
        try:
            hits_edges = hits_edges[
                ~(shapely.predicates.covers(hits_edges, edge.geometry))
            ]
            hits_edges = pd.Series(
                [
                    shapely.points(item)
                    for sublist in [shapely.get_coordinates(x) for x in hits_edges]
                    for item in sublist
                ],
                name="geometry",
            )
            hits = [
                shapely.points(x)
                for x in shapely.coordinates.get_coordinates(
                    shapely.constructive.extract_unique_points(
                        shapely.multipoints(pd.concat([hits_nodes, hits_edges]).values)
                    )
                )
            ]
        except TypeError:
            return hits_edges
        hits = pd.DataFrame(hits, columns=["geometry"])
        # get points and geometry as list of coordinates
        split_points = shapely.coordinates.get_coordinates(
            shapely.snap(hits, edge.geometry, tolerance=1e-9)
        )
        coor_geom = shapely.coordinates.get_coordinates(edge.geometry)
        # potentially split to multiple edges
        split_locs = np.argwhere(np.isin(coor_geom, split_points).all(axis=1))[:, 0]
        split_locs = list(zip(split_locs.tolist(), split_locs.tolist()[1:]))
        new_edges = [
            coor_geom[split_loc[0] : split_loc[1] + 1] for split_loc in split_locs
        ]
        grab_all_edges.append(
            [
                [edge.osm_id] * len(new_edges),
                [shapely.linestrings(edge) for edge in new_edges],
                [edge[1:-1]] * len(new_edges),
            ]
        )

    big_list = [list(zip(x[0], x[1], x[2])) for x in grab_all_edges]

    # combine all new edges
    edges = pd.DataFrame(
        [
            [item[0], item[1]] + list(item[2])
            for sublist in big_list
            for item in sublist
        ],
        columns=["osm_id", "geometry"] + attributes,
    )
    nodes = network.nodes.copy()

    return Network(edges, nodes)


def _intersects(geom, dataframe, sindex, tolerance=1e-9):
    """Find elements of a GeoSeries intersecting with a geometry.

    Buffers the geometry by a tolerance before querying the spatial index.

    Parameters
    ----------
    geom : shapely.Geometry
        Geometry to test for intersection.
    dataframe : gpd.GeoSeries
        GeoSeries of geometries to search.
    sindex : shapely.STRtree
        Spatial index of the dataframe geometries.
    tolerance : float, optional
        Buffer distance for intersection test. Default: 1e-9.

    Returns
    -------
    gpd.GeoSeries
        Subset of ``dataframe`` that intersects with ``geom``.
    """
    buffer = shapely.buffer(geom, tolerance)
    if shapely.is_empty(buffer):
        # can have an empty buffer with too small a tolerance, fallback to original geom
        buffer = geom
    try:
        return _intersects_dataframe(buffer, dataframe, sindex)
    except:
        # can exceptionally buffer to an invalid geometry, so try re-buffering
        buffer = shapely.buffer(geom, 0)
        return _intersects_dataframe(buffer, dataframe, sindex)


def _intersects_dataframe(geom, dataframe, sindex):
    """Return elements of a GeoSeries that intersect with a geometry.

    Parameters
    ----------
    geom : shapely.Geometry
        Geometry to test for intersection.
    dataframe : gpd.GeoSeries
        GeoSeries of geometries to search.
    sindex : shapely.STRtree
        Spatial index of the dataframe geometries.

    Returns
    -------
    gpd.GeoSeries
        Subset of ``dataframe`` that intersects with ``geom``.
    """
    return dataframe[sindex.query(geom, "intersects")]


def intersects(geom, dataframe, sindex, tolerance=1e-9):
    """Find the subset of a GeoSeries intersecting with a geometry.

    Parameters
    ----------
    geom : shapely.Geometry
        Geometry to test for intersection.
    dataframe : gpd.GeoSeries
        GeoSeries of geometries to search.
    sindex : shapely.STRtree
        Spatial index of the dataframe geometries.
    tolerance : float, optional
        Buffer distance for intersection test. Default: 1e-9.

    Returns
    -------
    gpd.GeoSeries
        Subset of ``dataframe`` that intersects with ``geom``.
    """
    return _intersects(geom, dataframe, sindex, tolerance)


def nodes_intersecting(line, nodes, sindex, tolerance=1e-9):
    """Find nodes intersecting with a line geometry.

    Parameters
    ----------
    line : shapely.LineString
        Line geometry to test for intersection.
    nodes : gpd.GeoSeries
        GeoSeries of node geometries.
    sindex : shapely.STRtree
        Spatial index of the node geometries.
    tolerance : float, optional
        Buffer distance for intersection test. Default: 1e-9.

    Returns
    -------
    gpd.GeoSeries
        Subset of ``nodes`` that intersects with ``line``.
    """
    return intersects(line, nodes, sindex, tolerance)


def _edge_lengths_metres(edges):
    """Compute edge lengths in metres for any CRS.

    For geographic CRS (coordinates in degrees), geodesic distances are
    computed via :func:`climada.util.coordinates.compute_geodesic_lengths`.
    For projected CRS (coordinates already in linear units such as metres),
    ``shapely.length`` is used directly.

    Parameters
    ----------
    edges : gpd.GeoDataFrame
        Edge GeoDataFrame with a geometry column and a CRS set.

    Returns
    -------
    np.ndarray
        Edge lengths in metres.
    """
    if edges.crs is not None and not edges.crs.is_geographic:
        # Projected CRS – shapely.length gives distances in CRS linear units.
        # Convert to metres if the CRS unit is not already metres.
        unit = edges.crs.axis_info[0].unit_name
        lengths = shapely.length(edges.geometry)
        if unit != "metre":
            factor = edges.crs.axis_info[0].unit_conversion_factor
            lengths = lengths * factor
        return lengths

    # Geographic CRS (or None → assume WGS 84) – use geodesic distances.
    if edges.crs is None:
        edges = edges.set_crs("EPSG:4326")
    return compute_geodesic_lengths(edges).values


def add_distances(network):
    """Add a distance column to edges in metres.

    Handles both geographic CRS (e.g. EPSG:4326, coordinates in degrees)
    and projected CRS (e.g. UTM, coordinates in metres).  For geographic
    CRS, geodesic distances are computed via
    :func:`climada.util.coordinates.compute_geodesic_lengths`.  For
    projected CRS, ``shapely.length`` is used directly.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).

    Returns
    -------
    Network
        Network with a 'distance' column (in metres) added to edges.
    """
    edges = network.edges.copy()
    nodes = network.nodes.copy()

    if edges.empty:
        return Network(edges, nodes)

    edges["distance"] = _edge_lengths_metres(edges)

    return Network(edges, nodes)


def _ecols_to_graphorder(edges):
    """Order edge columns as igraph expects them for building a graph.

    Moves 'from_id' and 'to_id' to the front of the DataFrame.

    Parameters
    ----------
    edges : gpd.GeoDataFrame
        Edge GeoDataFrame to reorder.

    Returns
    -------
    gpd.GeoDataFrame
        Reordered edge GeoDataFrame.
    """
    return edges.reindex(
        ["from_id", "to_id"]
        + [x for x in list(edges) if x not in ["from_id", "to_id"]],
        axis=1,
    )


def _vcols_to_graphorder(nodes):
    """Order node columns as igraph expects them for building a graph.

    Moves 'id' to the front of the DataFrame.

    Parameters
    ----------
    nodes : gpd.GeoDataFrame
        Node GeoDataFrame to reorder.

    Returns
    -------
    gpd.GeoDataFrame
        Reordered node GeoDataFrame.
    """
    return nodes.reindex(["id"] + [x for x in list(nodes) if x not in ["id"]], axis=1)


# =============================================================================
# Simplification wrappers
# =============================================================================


def simplified_network(network):
    """Return a simplified network.

    Applies a series of simplification steps: cleaning roundabouts, adding
    endpoints, adding ids and topology, merging degree-2 edges, removing
    duplicate geometries, resetting ids, adding distances, and merging
    MultiLineStrings.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).

    Returns
    -------
    Network
        Simplified network.
    """

    network_simp = Network(network.edges.copy(), network.nodes.copy())

    network_simp = clean_roundabouts(network_simp)
    network_simp = add_endpoints(network_simp)
    # network = split_edges_at_nodes(network) leave for now - takes too long
    # network = add_endpoints(network)
    network_simp = add_ids(network_simp)
    network_simp = add_topology(network_simp)
    network_simp.nodes["degree"] = calculate_degree(network_simp)
    network_simp = merge_edges(network_simp)
    network_simp.edges = drop_duplicate_geometries(network_simp.edges, keep="first")
    network_simp = reset_ids(network_simp)
    network_simp = add_distances(network_simp)
    network_simp = merge_multilinestrings(network_simp)
    return network_simp


def ordered_network(network, attrs={}):
    """Return a column-ordered network for igraph graph generation.

    Reorders node and edge columns to the format expected by igraph
    and optionally adds additional attributes.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).
    attrs : dict, optional
        Additional attributes to add to both edges and nodes.
        Default: {}.

    Returns
    -------
    Network
        Network with reordered columns ready for igraph graph construction.
    """

    network_ord = Network(network.edges.copy(), network.nodes.copy())

    network_ord.nodes = _vcols_to_graphorder(network_ord.nodes)
    network_ord.edges = _ecols_to_graphorder(network_ord.edges)
    for key, value in attrs.items():
        network_ord.edges[key] = value
        network_ord.nodes[key] = value
    return network_ord
