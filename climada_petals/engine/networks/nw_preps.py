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
import pyproj

from climada_petals.engine.networks.nw_base import Network

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')


# =============================================================================
# Simplification methods from ElcoK/trails/simplify
# =============================================================================

"""
all functions taken and slight modified from:
https://github.com/ElcoK/trails/blob/main/src/trails/simplify.py
"""


def add_ids(network, id_col='id'):
    """
    Add or replace an id column with ascending ids

    Parameters
        network (class): A network composed of nodes (points in space) and
            edges (lines)
        id_col (str, optional): [description]. Defaults to 'id'.

    Returns
    -------
        Network (class):
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


def add_topology(network, id_col='id'):
    """
    Add or replace from_id, to_id to edges

    Parameters
    -------
    network (class): A network composed of nodes (points in space) and
        edges (lines)
    id_col (str, optional): [description]. Defaults to 'id'.

    Returns
    -------
        Network (class): A network composed of nodes (points in space) and
            edges (lines)
    """

    from_ids = []
    to_ids = []
    bugs = []

    nodes = network.nodes.copy()
    edges = network.edges.copy()

    sindex = shapely.STRtree(nodes.geometry)
    for edge in tqdm(
            edges.itertuples(), desc="topology", total=len(edges)):
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

    edges['from_id'] = from_ids
    edges['to_id'] = to_ids
    edges = edges.loc[~(edges.id.isin(list(bugs)))].reset_index(drop=True)

    return Network(edges, nodes)


def line_endpoints(line):
    """
    Return points at first and last vertex of a line

    Parameters
    -------
    line ([type]): [description]

    Returns
    -------
        [type]: [description]
    """
    start = shapely.get_point(line, 0)
    end = shapely.get_point(line, -1)
    return start, end


def nearest(geom, dataframe, sindex):
    """
    Find the element of a DataFrame nearest a geometry

    Parameters
    -------
    geom (shapely.geometry): [description]
    dataframe (pandas.DataFrame): [description]
    sindex ([type]): [description]

    Returns
    -------
        [type]: [description]
    """
    matches_idx = sindex.query(geom)
    nearest_geom = min(
        [dataframe.iloc[match_idx] for match_idx in matches_idx],
        key=lambda match: shapely.measurement.distance(match.geometry, geom)
    )
    return nearest_geom


def nearest_node(point, nodes, sindex):
    """
    Find nearest node to a point

    Parameters
    -------
    point *shapely.geometry): [description]
    nodes (network.nodes): [description]
    sindex ([type]): [description]

    Returns
    -------
        [type]: [description]
    """
    return nearest(point, nodes, sindex)


def get_endpoints(network):
    """
    Get nodes for each edge endpoint

    Parameters
    ----------
    network (class): A network composed of nodes (points in space) and
        edges (lines)

    Returns
    -------
        [type]: [description]
    """
    endpoints = []
    for edge in tqdm(
            network.edges.itertuples(),
            desc="endpoints",
            total=len(network.edges)):
        if edge.geometry is None:
            continue
        # 5 is MULTILINESTRING
        if shapely.get_type_id(edge.geometry) == '5':
            for line in edge.geometry.geoms:
                start, end = line_endpoints(line)
                endpoints.append(start)
                endpoints.append(end)
        else:
            start, end = line_endpoints(edge.geometry)
            endpoints.append(start)
            endpoints.append(end)

    # create dataframe to match the nodes geometry column name
    return gpd.GeoDataFrame(geometry=endpoints, crs='EPSG:4326')


def add_endpoints(network):
    """
    Add nodes at line endpoints

    Parameters
    ----------
    network (class): A network composed of nodes (points in space) and
        edges (lines)

    Returns
    -------
    Network (class): A network composed of nodes (points in space) and
        edges (lines)
    """

    endpoints = get_endpoints(network)

    nodes = network.nodes.copy()
    edges = network.edges.copy()

    nodes = concat_dedup([nodes, endpoints])

    return Network(edges, nodes)


def merge_multilinestrings(network):
    """
    Try to merge all multilinestring geometries into linestring geometries.

    Parameters
    ----------
    network (class): A network composed of nodes (points in space) and
        edges (lines)

    Returns
    -------
    Network (class): A network composed of nodes (points in space) and
        edges (lines)
    """
    nodes = network.nodes.copy()
    edges = network.edges.copy()

    edges['geometry'] = edges.geometry.apply(
        lambda x: merge_multilinestring(x))

    return Network(edges, nodes)


def merge_multilinestring(geom):
    """Merge a MultiLineString to LineString

    Parameters
    ----------
    geom (shapely.geometry): A shapely geometry, most likely a linestring or
        a multilinestring

    Returns
    -------
    geom (shapely.geometry): A shapely linestring geometry if merge was
        succesful. If not, it returns the input.
    """
    if shapely.get_type_id(geom) == '5':
        geom_inb = shapely.line_merge(geom)
        if geom_inb.is_ring:  # still something to fix if desired
            return geom_inb
        return geom_inb
    return geom


def find_roundabouts(network):
    """
    Methods to find roundabouts

    Parameters
    ----------
    network (class): A network composed of nodes (points in space) and edges (lines)

    Returns
    -------
    roundabouts (list): Returns the edges that can be identified as roundabouts
    """
    roundabouts = []
    for edge in network.edges.itertuples():
        if shapely.predicates.is_ring(edge.geometry):
            roundabouts.append(edge)
    return roundabouts


def clean_roundabouts(network):
    """
    Methods to clean roundabouts and junctions should be done before
        splitting edges at nodes to avoid logic conflicts

    Parameters
    ----------
    network (class): A network composed of nodes (points in space) and edges (lines)

    Returns
    -------
    network (class): A network composed of nodes (points in space) and edges (lines)
    """
    # TODO: Is reference to osm_id really necessary? Remove by alternative?

    edges = network.edges.copy()
    nodes = network.nodes.copy()

    sindex = shapely.STRtree(edges['geometry'])
    new_edge = []
    remove_edge = []
    new_edge_id = []
    attributes = [x for x in edges.columns if x not in [
        'geometry', 'osm_id']]

    roundabouts = find_roundabouts(network)
    for roundabout in roundabouts:
        round_centroid = shapely.constructive.centroid(roundabout.geometry)
        remove_edge.append(roundabout.Index)

        edges_intersect = _intersects(
            roundabout.geometry, edges['geometry'], sindex)
        # index at e[0] geometry at e[1] of edges that intersect with
        for edg in edges_intersect.items():
            edge = edges.iloc[edg[0]]
            start = shapely.get_point(edg[1], 0)
            end = shapely.get_point(edg[1], -1)
            first_co_is_closer = \
                shapely.measurement.distance(end, round_centroid) > \
                shapely.measurement.distance(start, round_centroid)
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
                first_co_is_closer = \
                    shapely.measurement.distance(end, round_centroid) > \
                    shapely.measurement.distance(start, round_centroid)
                co_ords = shapely.coordinates.get_coordinates(double_edge[-1])
                if first_co_is_closer:
                    new_co = np.concatenate((centroid_co, co_ords))
                else:
                    new_co = np.concatenate((co_ords, centroid_co))
                snap_line = shapely.linestrings(new_co)
                new_edge.append(
                    [edge.osm_id]+list(edge[list(attributes)])+[snap_line])

            else:
                new_edge.append(
                    [edge.osm_id]+list(edge[list(attributes)])+[snap_line])
                new_edge_id.append(edge.osm_id)
            remove_edge.append(edg[0])

    new = pd.DataFrame(new_edge, columns=['osm_id']+attributes+['geometry'])
    edges = edges.loc[~edges.index.isin(remove_edge)]
    edges = pd.concat([edges, new]).reset_index(drop=True)

    return Network(edges, nodes)


def calculate_degree(network):
    """
    Calculates the degree of the nodes from the from and to ids. It
    is not wise to call this method after removing nodes or edges
    without first resetting the ids

    Parameters
    ----------
    network (class): A network composed of nodes (points in space) and edges (lines)

    Returns
    -------
    Connectivity degree (numpy.array): [description]
    """
    if network.edges.empty:
        return [0]*len(network.nodes)
    # the number of nodes(from index) to use as the number of bins
    ndC = len(network.nodes.index)
    if ndC-1 > max(network.edges.from_id) and ndC-1 > max(network.edges.to_id):
        print("Calculate_degree possibly unhappy")
    return (np.bincount(network.edges['from_id'], None, ndC) +
            np.bincount(network.edges['to_id'], None, ndC))


def add_degree(network):
    """
    Adds a degree column to the node dataframe

    Parameters
    ----------
    network (class): A network composed of nodes (points in space) and edges (lines)

    Returns
    -------
    network (class): A network composed of nodes (points in space) and edges (lines)
    """
    degree = calculate_degree(network)

    edges = network.edges.copy()
    nodes = network.nodes.copy()
    nodes['degree'] = degree

    return Network(edges, nodes)


def concat_dedup(dataframes):
    """
    Concatenate a list of GeoDataFrames, dropping duplicate geometries
    - note: repeatedly drops indexes for deduplication to work

    Parameters
    ----------
        dataframes ([type]): [description]

    Returns
    -------
        [type]: [description]
    """
    cat = pd.concat(dataframes, axis=0, sort=False)
    cat.reset_index(drop=True, inplace=True)
    cat_dedup = drop_duplicate_geometries(cat)
    cat_dedup.reset_index(drop=True, inplace=True)
    return cat_dedup


def find_closest_2_edges(edgeIDs, edges, nodGeometry):
    """
    Returns the 2 edges connected to the current node

    Parameters
    ----------
    edgeIDs ([type]): [description]
    edges ([type]): [description]
    nodGeometry ([type]): [description]

    Returns
    -------
        [type]: [description]
    """
    edge_path_1 = min([edges.iloc[match_idx] for match_idx in edgeIDs],
                      key=lambda match: shapely.distance(nodGeometry, match.geometry))
    edgeIDs.remove(edge_path_1.name)
    edge_path_2 = min([edges.iloc[match_idx] for match_idx in edgeIDs],
                      key=lambda match:  shapely.distance(nodGeometry, match.geometry))
    return edge_path_1, edge_path_2


def merge_edges(network, print_err=False):
    """
    This method removes all degree 2 nodes and merges their associated edges, at
    the moment it arbitrarily uses the first edge's attributes for the new edges
    column attributes, in the future the mean or another measure can be used
    to set these new values. The general strategy is to find a node of degree 2,
    and the associated 2 edges, then traverse edges and nodes in both directions
    until a node of degree !=2 is found, at this point stop in this direction. Reset the
    geometry and from/to ids for this edge, delete the nodes and edges traversed.

    Parameters
    ----------
    network (class): A network composed of nodes (points in space) and edges (lines)
    print_err (bool, optional): [description]. Defaults to False.

    Returns
    -------
    network (class): A network composed of nodes (points in space) and edges (lines)
    """
    if network.edges.empty:
        return network

    nodes = network.nodes.copy()
    edges = network.edges.copy()

    optional_cols = edges.columns.difference(
        ['osm_id', 'geometry', 'from_id', 'to_id', 'id'])
    edg_sindex = shapely.STRtree(edges.geometry)

    if 'degree' not in nodes.columns:
        deg = calculate_degree(network)
    else:
        deg = nodes['degree'].to_numpy()
    degree2 = np.where(deg == 2)
    n2 = set((nodes['id'].iloc[degree2]))

    nodGeom = nodes['geometry']
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
        eID = set(edg_sindex.query(node_geometry, predicate='intersects'))
        # Find the nearest 2 edges, unless there is an error in the dataframe
        # this will return the connected edges using spatial indexing
        if len(eID) > 2:
            edge_path_1, edge_path_2 = find_closest_2_edges(
                eID, edges, node_geometry)
        elif len(eID) < 2:
            continue
        else:
            edge_path_1 = edges.iloc[eID.pop()]
            edge_path_2 = edges.iloc[eID.pop()]
        # For the two edges found, identify the next 2 nodes in either direction
        next_node_1 = edge_path_1.to_id if edge_path_1.from_id == nodeID else edge_path_1.from_id
        next_node_2 = edge_path_2.to_id if edge_path_2.from_id == nodeID else edge_path_2.from_id
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
            eID = set(edg_sindex.query(
                next_node_1Geom, predicate='intersects'))
            eID.discard(edge_path_1.id)
            try:
                edge_path_1 = min([edges.iloc[match_idx] for match_idx in eID],
                                  key=lambda match: shapely.distance(next_node_1Geom, (match.geometry)))
            except:
                continue
            pos_0_deg.append(next_node_1)
            n2.discard(next_node_1)
            next_node_1 = edge_path_1.to_id if edge_path_1.from_id == next_node_1 else edge_path_1.from_id
            newEdge.append(edge_path_1.geometry)
            possibly_delete.append(edge_path_1.id)

        while deg[next_node_2] == 2:
            if next_node_2 in pos_0_deg:
                break
            next_node_2Geom = nodGeom[next_node_2]
            eID = set(edg_sindex.query(
                next_node_2Geom, predicate='intersects'))
            eID.discard(edge_path_2.id)
            try:
                edge_path_2 = min([edges.iloc[match_idx] for match_idx in eID],
                                  key=lambda match: shapely.distance(next_node_2Geom, (match.geometry)))
            except:
                continue
            pos_0_deg.append(next_node_2)
            n2.discard(next_node_2)
            next_node_2 = edge_path_2.to_id if edge_path_2.from_id == next_node_2 else edge_path_2.from_id
            newEdge.append(edge_path_2.geometry)
            possibly_delete.append(edge_path_2.id)
        # Update the information of the first edge
        new_merged_geom = shapely.line_merge(
            shapely.multilinestrings([newEdge]))
        if shapely.get_type_id(new_merged_geom) == 1:
            edges.at[info_first_edge, 'geometry'] = new_merged_geom
            if nodGeom[next_node_1] == shapely.get_point(new_merged_geom, 0):
                edges.at[info_first_edge, 'from_id'] = next_node_1
                edges.at[info_first_edge, 'to_id'] = next_node_2
            else:
                edges.at[info_first_edge, 'from_id'] = next_node_2
                edges.at[info_first_edge, 'to_id'] = next_node_1
            eIDtoRemove += possibly_delete
            possibly_delete.append(info_first_edge)
            for x in pos_0_deg:
                deg[x] = 0
            mode_edges = edges.loc[edges.id.isin(possibly_delete)]
            edges.loc[info_first_edge, optional_cols] = mode_edges[optional_cols].mode(
            ).iloc[0].values
        else:
            if print_err:
                print("Line", info_first_edge,
                      "failed to merge, has shapely type ",
                      shapely.get_type_id(edges.at[info_first_edge, 'geometry']))

    edges = edges.loc[~(edges.id.isin(eIDtoRemove))].reset_index(drop=True)

    # We remove all degree 0 nodes, including those found in dropHanging
    nodes = nodes.loc[nodes.degree > 0].reset_index(drop=True)

    return Network(edges, nodes)


def node_connectivity_degree(node, network):
    """
    Get the degree of connectivity for a node.

    Parameters
    ----------
    node ([type]): [description]
    network (class): A network composed of nodes (points in space) and
        edges (lines)

    Returns
    -------
        type]: [description]
    """
    return len(
        network.edges[
            (network.edges.from_id == node) | (network.edges.to_id == node)
        ]
    )


def drop_duplicate_geometries(dataframe, keep='first'):
    """
    Drop duplicate geometries from a dataframe

    Convert to wkb so drop_duplicates will work as discussed
    in https://github.com/geopandas/geopandas/issues/521

    Parameters
    ----------
        dataframe (pandas.DataFrame): [description]
        keep (str, optional): [description]. Defaults to 'first'.

    Returns
    -------
        [type]: [description]
    """

    mask = dataframe.geometry.apply(lambda geom: shapely.to_wkb(geom))
    # use dropped duplicates index to drop from actual dataframe
    return dataframe.iloc[mask.drop_duplicates(keep).index]


def reset_ids(network):
    """
    Resets the ids of the nodes and edges, editing the refereces in edge table
    using dict masking

    Parameters
    ----------
        network (class): A network composed of nodes (points in space) and
            edges (lines)

    Returns
    -------
        [type]: [description]
    """
    nodes = network.nodes.copy()
    edges = network.edges.copy()

    to_ids = edges['to_id'].to_numpy()
    from_ids = edges['from_id'].to_numpy()
    new_node_ids = range(len(nodes))
    # creates a dictionary of the node ids and the actual indices
    id_dict = dict(zip(nodes.id, new_node_ids))
    nt = np.copy(to_ids)
    nf = np.copy(from_ids)
    # updates all from and to ids, because many nodes are effected, this
    # is quite optimal approach for large dataframes
    for k, v in id_dict.items():
        nt[to_ids == k] = v
        nf[from_ids == k] = v
    edges.drop(labels=['to_id', 'from_id'], axis=1, inplace=True)
    edges['from_id'] = nf
    edges['to_id'] = nt
    nodes.drop(labels=['id'], axis=1, inplace=True)
    nodes['id'] = new_node_ids
    edges['id'] = range(len(edges))
    edges.reset_index(drop=True, inplace=True)
    nodes.reset_index(drop=True, inplace=True)

    return Network(edges, nodes)


def split_edges_at_nodes(network):
    """
    Split network edges where they intersect node geometries
    """
    sindex_nodes = shapely.STRtree(network.nodes['geometry'])
    sindex_edges = shapely.STRtree(network.edges['geometry'])
    attributes = [x for x in network.edges.columns if x not in [
        'index', 'geometry', 'osm_id']]
    grab_all_edges = []

    # TODO: this takes really long. Rewrite?
    for edge in tqdm(network.edges.itertuples(index=False), desc="splitting",
                     total=len(network.edges)):
        hits_nodes = nodes_intersecting(
            edge.geometry, network.nodes['geometry'], sindex_nodes, tolerance=1e-9)
        hits_edges = nodes_intersecting(
            edge.geometry, network.edges['geometry'], sindex_edges, tolerance=1e-9)
        hits_edges = shapely.set_operations.intersection(
            edge.geometry, hits_edges)
        try:
            hits_edges = (
                hits_edges[~(shapely.predicates.covers(hits_edges, edge.geometry))])
            hits_edges = pd.Series([shapely.points(item) for sublist in [shapely.get_coordinates(
                x) for x in hits_edges] for item in sublist], name='geometry')
            hits = [shapely.points(x) for x in
                    shapely.coordinates.get_coordinates(
                        shapely.constructive.extract_unique_points
                        (shapely.multipoints(pd.concat([hits_nodes, hits_edges]
                                                       ).values)))]
        except TypeError:
            return hits_edges
        hits = pd.DataFrame(hits, columns=['geometry'])
        # get points and geometry as list of coordinates
        split_points = shapely.coordinates.get_coordinates(
            shapely.snap(hits, edge.geometry, tolerance=1e-9))
        coor_geom = shapely.coordinates.get_coordinates(edge.geometry)
        # potentially split to multiple edges
        split_locs = np.argwhere(
            np.isin(coor_geom, split_points).all(axis=1))[:, 0]
        split_locs = list(zip(split_locs.tolist(), split_locs.tolist()[1:]))
        new_edges = [coor_geom[split_loc[0]:split_loc[1]+1]
                     for split_loc in split_locs]
        grab_all_edges.append([[edge.osm_id]*len(new_edges), [shapely.linestrings(edge)
                              for edge in new_edges], [edge[1:-1]]*len(new_edges)])

    big_list = [list(zip(x[0], x[1], x[2])) for x in grab_all_edges]

    # combine all new edges
    edges = pd.DataFrame([[item[0], item[1]]+list(item[2]) for sublist in big_list for item in sublist],
                         columns=['osm_id', 'geometry']+attributes)
    nodes = network.nodes.copy()

    return Network(edges, nodes)


def _intersects(geom, dataframe, sindex, tolerance=1e-9):
    """
    [summary]

    Parameters
    ----------
    geom (shapely.geometry): [description]
    dataframe ([type]): [description]
    sindex ([type]): [description]
    tolerance ([type], optional): [description]. Defaults to 1e-9.

    Returns
    -------
        [type]: [description]
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
    """
    [summary]

    Parameters
    ----------
    geom ([type]): [description]
    dataframe ([type]): [description]
    sindex ([type]): [description]

    Returns
    -------
    [type]: [description]
    """
    return dataframe[sindex.query(geom, 'intersects')]


def intersects(geom, dataframe, sindex, tolerance=1e-9):
    """Find the subset of a GeoDataFrame intersecting with a shapely geometry

    Parameters
    ----------
        geom ([type]): [description]
        dataframe ([type]): [description]
        sindex ([type]): [description]
        tolerance ([type], optional): [description]. Defaults to 1e-9.

    Returns
    -------
        [type]: [description]
    """
    return _intersects(geom, dataframe, sindex, tolerance)


def nodes_intersecting(line, nodes, sindex, tolerance=1e-9):
    """
    Find nodes intersecting line

    Parameters
    ----------
    line ([type]): [description]
    nodes ([type]): [description]
    sindex ([type]): [description]
    tolerance ([type], optional): [description]. Defaults to 1e-9.

    Returns
    -------
    [type]: [description]
    """
    return intersects(line, nodes, sindex, tolerance)


def add_distances(network):
    """
    This method adds a distance column using shapely (converted from shapely)
    assuming the new crs from the latitude and longitude of the first node
    distance is in metres

    Parameters
    ----------
    network (class): A network composed of nodes (points in space) and edges (lines)

    Returns
    -------
    network (class): A network composed of nodes (points in space) and edges (lines)
    """
    # TODO: replace by climada-internal func (already exists)
    edges = network.edges.copy()
    nodes = network.nodes.copy()

    if edges.empty:
        return Network(edges, nodes)
    # Find crs of current dataframe and arbitrary point(lat,lon) for new crs
    current_crs = "epsg:4326"
    lat = shapely.get_y(network.nodes['geometry'].iloc[0])
    lon = shapely.get_x(network.nodes['geometry'].iloc[0])
    # formula below based on :https://gis.stackexchange.com/a/190209/80697
    approximate_crs = "epsg:" +\
        str(int(32700-np.round((45+lat)/90, 0)*100+np.round((183+lon)/6, 0)))
    # from shapely/issues/95
    coords = shapely.get_coordinates(edges['geometry'])
    transformer = pyproj.Transformer.from_crs(
        current_crs, approximate_crs, always_xy=True)
    new_coords = transformer.transform(coords[:, 0], coords[:, 1])
    result = shapely.set_coordinates(
        edges['geometry'].copy(), np.array(new_coords).T)
    dist = shapely.length(result)

    edges['distance'] = dist

    return Network(edges, nodes)


def _ecols_to_graphorder(edges):
    """
    order columns as igraph expects them for building a graph

    Parameters
    ----------
    """
    return edges.reindex(['from_id', 'to_id'] +
                         [x for x in list(edges)
                          if x not in ['from_id', 'to_id']], axis=1)


def _vcols_to_graphorder(nodes):
    """
    order columns as igraph expects them for building a graph

    Parameters
    ----------
    """
    return nodes.reindex(['id'] + [x for x in list(nodes)
                         if x not in ['id']], axis=1)


# =============================================================================
# Simplification wrappers
# =============================================================================

def simplified_network(network):
    """
    returns a simplified network

    Parameters
    -----------
    network ([nw_base.Network]): [description]

    Returns
    -------
    network_simp ([nw_base.Network]): simplified network
    """

    network_simp = Network(network.edges.copy(), network.nodes.copy())

    network_simp = clean_roundabouts(network_simp)
    network_simp = add_endpoints(network_simp)
    # network = split_edges_at_nodes(network) leave for now - takes too long
    # network = add_endpoints(network)
    network_simp = add_ids(network_simp)
    network_simp = add_topology(network_simp)
    network_simp.nodes['degree'] = calculate_degree(network_simp)
    network_simp = merge_edges(network_simp)
    network_simp.edges = drop_duplicate_geometries(
        network_simp.edges,  keep='first')
    network_simp = reset_ids(network_simp)
    network_simp = add_distances(network_simp)
    network_simp = merge_multilinestrings(network_simp)
    return network_simp


def ordered_network(network, attrs={}):
    """
    returns an ordered network for igraph graph generation

    Parameters
    -----------
    network ([nw_base.Network]): [description]

    Returns
    -------
    network_ord ([nw_base.Network]): ordered network
    """

    network_ord = Network(network.edges.copy(), network.nodes.copy())

    network_ord.nodes = _vcols_to_graphorder(network_ord.nodes)
    network_ord.edges = _ecols_to_graphorder(network_ord.edges)
    for key, value in attrs.items():
        network_ord.edges[key] = value
        network_ord.nodes[key] = value
    return network_ord
