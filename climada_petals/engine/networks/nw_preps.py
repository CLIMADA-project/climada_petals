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

-------

Clean GeoDataFrames with network data and convert them to a nodes & edges
structure compatible with igraph graph calculations.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import shapely
from shapely.ops import substring

from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_utils import _edge_lengths_metres

LOGGER = logging.getLogger(__name__)

# =============================================================================
# Default settings
# =============================================================================

#: Edge columns treated as external (source-data) ids by default, e.g. OSM
#: way ids. They are carried along for provenance only and never used to
#: identify edges internally. When edges are merged, the merged edge takes the
#: value of the longest edge instead of a majority. Override with the
#: ``id_cols`` argument of :func:`merge_edges` / :func:`simplified_network`.
DEFAULT_ID_COLS = ("osm_id", "osmid", "edge_id")

#: Preprocessing profiles for :func:`simplified_network`: the default settings
#: per infrastructure type. Every setting can be overridden by the keyword
#: argument of the same name.
#:
#: - ``roundabouts``: collapse roundabouts (:func:`clean_roundabouts`);
#: - ``roundabout_max_length``: maximum ring length in metres for a
#:   roundabout (None: no limit);
#: - ``split``: split edges where they meet nodes or other edges
#:   (:func:`split_edges_at_nodes`): ``False`` (no splitting), ``"anywhere"``
#:   (at every meeting point, also between two vertices of an edge) or
#:   ``"vertices"`` (only at the edge's own vertices).
PREPROCESSING_PROFILES = {
    "generic": {
        "roundabouts": False,
        "roundabout_max_length": None,
        "split": False,
    },
    "road": {
        "roundabouts": True,
        "roundabout_max_length": 1000.0,
        "split": "anywhere",
    },
    "power_line": {
        "roundabouts": False,
        "roundabout_max_length": None,
        "split": "anywhere",
    },
}


# =============================================================================
# Simplification methods from ElcoK/trails/simplify
# =============================================================================

# All functions in this section are taken and slightly modified from:
# https://github.com/ElcoK/trails/blob/main/src/trails/simplify.py


def add_ids(network, id_col="id"):
    """Add or replace an id column with ascending ids.

    Nodes and edges get ids ``0..n-1`` in row order and a reset index. Edge
    references (``from_id`` / ``to_id``) are not updated: call this before
    :func:`add_topology`, or use :func:`reset_ids` on a network that already
    has topology.

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


def add_topology(network, id_col="id", tolerance=1e-9):
    """Add or replace from_id, to_id columns to edges.

    Each edge end is attached to the nearest node within ``tolerance`` (with
    several nodes at the same distance, the first in row order). Edges
    with an end that has no node within ``tolerance`` (and edges without a
    LineString geometry, e.g. MultiLineStrings) are dropped, and a warning
    with their number is logged. Run :func:`add_endpoints` first so that every
    edge end has a node.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).
    id_col : str, optional
        Node column whose values are written to ``from_id`` / ``to_id``.
        Default: 'id'.
    tolerance : float, optional
        Maximum distance between an edge end and its node, in CRS units.
        The default only absorbs floating-point differences. Default: 1e-9.

    Returns
    -------
    Network
        Network with from_id and to_id columns added to edges.
    """
    nodes = network.nodes.copy()
    edges = network.edges.copy()

    geoms = edges.geometry.to_numpy()
    # first and last point of every edge; None for missing geometries and
    # non-lines (e.g. MultiLineStrings), which then find no node
    ends = {
        "from_id": shapely.get_point(geoms, 0),
        "to_id": shapely.get_point(geoms, -1),
    }
    sindex = shapely.STRtree(nodes.geometry.to_numpy())
    node_ids = nodes[id_col].to_numpy()
    no_node = len(nodes)  # "no node found": larger than any node position
    matched = np.ones(len(edges), dtype=bool)
    for col, points in ends.items():
        # pairs (end position, node position) of the nearest nodes within
        # tolerance; missing points are left out of the result
        end_pos, node_pos = sindex.query_nearest(points, max_distance=tolerance)
        # several nodes at the same distance: the first one in row order
        nearest_pos = np.full(len(edges), no_node)
        np.minimum.at(nearest_pos, end_pos, node_pos)
        has_node = nearest_pos < no_node
        matched &= has_node
        values = np.full(len(edges), -1, dtype=object)
        values[has_node] = node_ids[nearest_pos[has_node]]
        edges[col] = values
    unmatched = np.flatnonzero(~matched)  # row positions of unattached edges

    if len(unmatched):
        LOGGER.warning(
            "add_topology dropped %d of %d edges: an edge end has no node within "
            "%g (CRS units), or the geometry is not a LineString.",
            len(unmatched),
            len(edges),
            tolerance,
        )
        edges = edges.drop(index=edges.index[unmatched])
    edges = edges.reset_index(drop=True)
    # same dtype as the node ids (the -1 placeholders are dropped)
    for col in ends:
        edges[col] = edges[col].astype(node_ids.dtype)

    return Network(edges, nodes)


def get_endpoints(network, attrs=("ci_type",)):
    """Get nodes for each edge endpoint.

    Each endpoint inherits the values of the edge attributes listed in
    ``attrs`` from the edge it belongs to, so that nodes created from line
    networks keep e.g. their infrastructure type. Only attributes that are
    meaningful for points should be propagated (edge properties such as
    distances, ids or topology columns are not).

    Endpoints come in edge row order, start before end; a MultiLineString
    gets the endpoints of each of its parts. Edges with a missing or empty
    geometry get none.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).
    attrs : list or tuple of str, optional
        Edge columns whose values are copied onto the endpoints. Columns not
        present in the edges are ignored. Default: ``("ci_type",)``.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with point geometries at each edge endpoint and the
        inherited attribute columns.
    """
    geoms = network.edges.geometry.to_numpy()
    valid_pos = np.flatnonzero(~shapely.is_missing(geoms) & ~shapely.is_empty(geoms))
    # one line per LineString and per part of a MultiLineString, in row order
    lines, part_of = shapely.get_parts(geoms[valid_pos], return_index=True)
    line_pos = valid_pos[part_of]  # row position of the edge of each line
    # start and end point of each line, interleaved: s0, e0, s1, e1, ...
    endpoints = np.empty(2 * len(lines), dtype=object)
    endpoints[0::2] = shapely.get_point(lines, 0)
    endpoints[1::2] = shapely.get_point(lines, -1)
    edge_pos = np.repeat(line_pos, 2)  # row position of the edge of each endpoint

    # create dataframe to match the nodes geometry column name
    gdf_endpoints = gpd.GeoDataFrame(geometry=endpoints, crs=network.edges.crs)

    if attrs:
        cols = [col for col in attrs if col in network.edges.columns]
        for col in cols:
            gdf_endpoints[col] = network.edges[col].iloc[edge_pos].to_numpy()

    return gdf_endpoints


def _resolve_endpoint_attrs(endpoints, attrs):
    """Give all endpoints at one location the same inherited attribute values.

    Per location and attribute: the shared value if the edges agree (missing
    values ignored), otherwise missing, with one warning per attribute.

    Parameters
    ----------
    endpoints : gpd.GeoDataFrame
        Output of :func:`get_endpoints` (one row per edge end).
    attrs : sequence of str
        Inherited attribute columns; columns not present are ignored.

    Returns
    -------
    gpd.GeoDataFrame
        ``endpoints`` with resolved attribute values.
    """
    cols = [col for col in attrs if col in endpoints.columns]
    if not cols or endpoints.empty:
        return endpoints
    endpoints = endpoints.copy()
    location = pd.Series(
        shapely.to_wkb(endpoints.geometry.to_numpy()), index=endpoints.index
    )
    for col in cols:
        by_location = endpoints[col].groupby(location)
        n_values = by_location.transform("nunique")
        shared = by_location.transform("first")  # first non-missing value
        conflict = n_values > 1
        endpoints[col] = shared.where(~conflict, np.nan)
        if conflict.any():
            LOGGER.warning(
                "add_endpoints: %s differed between the edges meeting at %d "
                "endpoints; left missing for those nodes.",
                col,
                location[conflict].nunique(),
            )
    return endpoints


def add_endpoints(network, inherit_attrs=("ci_type",)):
    """Add nodes at line endpoints.

    New endpoint nodes inherit the attributes listed in ``inherit_attrs`` from
    the edge they belong to. Where an endpoint coincides with an existing node,
    the existing node (and its attributes) is kept. Where several edges share
    an endpoint:

    - if they agree on an attribute (missing values ignored), the node gets
      that value;
    - if they have different values (e.g. a ``road`` and a ``power_line``
      ending at the same point), the attribute is left missing for that node,
      and one warning per attribute gives the number of such endpoints. No
      value is picked arbitrarily; with ``ci_type`` missing, the node is not
      part of any type selection. (:func:`simplified_network` requires a
      single ``ci_type``, so this does not occur there.)

    New nodes have no ``id`` (missing values) while existing nodes keep
    theirs: call :func:`add_ids` afterwards, then :func:`add_topology`.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).
    inherit_attrs : list or tuple of str, optional
        Edge columns whose values are copied onto new endpoint nodes.
        Default: ``("ci_type",)``.

    Returns
    -------
    Network
        Network with endpoint nodes added.
    """

    endpoints = get_endpoints(network, attrs=inherit_attrs)
    endpoints = _resolve_endpoint_attrs(endpoints, inherit_attrs)

    nodes = network.nodes.copy()
    edges = network.edges.copy()

    nodes = concat_dedup([nodes, endpoints])

    return Network(edges, nodes)


def merge_multilinestrings(network):
    """Merge all MultiLineString geometries into LineString geometries.

    MultiLineStrings whose parts connect become LineStrings; the others stay
    MultiLineStrings (split them with :func:`explode_multilinestrings`).
    Topology columns are not updated: run this before :func:`add_endpoints`.

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


def explode_multilinestrings(network):
    """Split MultiLineString edges into one LineString edge per part.

    Meant for MultiLineStrings whose parts do not connect (run
    :func:`merge_multilinestrings` first): such an edge has no well-defined
    start and end, so it cannot be attached to nodes. Each part becomes its
    own edge with a copy of all attributes (``id``, ``from_id`` and
    ``to_id`` included: run :func:`add_ids` and :func:`add_topology`
    afterwards). Other geometries are left unchanged. Logs the number of
    MultiLineStrings split.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).

    Returns
    -------
    Network
        Network whose edges contain no MultiLineStrings.
    """
    edges = network.edges.copy()
    nodes = network.nodes.copy()
    is_multi = shapely.get_type_id(edges.geometry.to_numpy()) == 5
    if not is_multi.any():
        return Network(edges, nodes)
    n_parts = int(shapely.get_num_geometries(edges.geometry.to_numpy()[is_multi]).sum())
    LOGGER.info(
        "Split %d MultiLineString edges with unconnected parts into %d edges.",
        int(is_multi.sum()),
        n_parts,
    )
    multi = edges.loc[is_multi].explode(index_parts=False)
    # keep the parts at the position of their original edge
    edges = pd.concat([edges.loc[~is_multi], multi]).sort_index(kind="stable")
    return Network(edges.reset_index(drop=True), nodes)


def merge_multilinestring(geom):
    """Merge a MultiLineString to a LineString.

    Parameters
    ----------
    geom : shapely.Geometry
        A shapely geometry, most likely a LineString or a MultiLineString.

    Returns
    -------
    shapely.Geometry
        The merged geometry for a MultiLineString: a LineString if its parts
        are connected, otherwise still a MultiLineString. Any other geometry
        is returned unchanged.
    """
    if shapely.get_type_id(geom) == 5:
        return shapely.line_merge(geom)
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


def _has_topology(edges):
    """Whether the edges have (non-missing) ``from_id`` / ``to_id`` values."""
    return {"from_id", "to_id"}.issubset(edges.columns) and edges[
        ["from_id", "to_id"]
    ].notna().any().any()


def clean_roundabouts(network, max_length=None, tolerance=1e-9):
    """Collapse roundabouts into a single point.

    A roundabout is an edge whose geometry is a closed ring (see
    :func:`find_roundabouts`), optionally no longer than ``max_length``. Each
    roundabout edge is removed, and every edge with an end on the ring is
    extended by a straight segment from that end to the ring's centroid, so
    that all roads meeting the roundabout meet at one point. An edge joining
    two roundabouts is extended at both ends.

    Edges that touch the ring but have no end on it (e.g. a road crossing it
    on a bridge) are left unchanged and stay unconnected to it. Edges with
    both ends on the same ring (e.g. a slip lane between two of its exits)
    are removed, since they would become a loop at the centroid. Other closed
    rings and non-LineString edges are left unchanged. Roundabouts drawn as
    several edges (none of them a closed ring) are not detected.

    Nodes and topology:

    - Nodes lying on a collapsed ring (within ``tolerance``) are removed:
      their role is taken over by the centroid, where the extended edges now
      meet. No node is created at the centroid; :func:`add_endpoints` does
      that. All other nodes are kept.
    - ``from_id`` / ``to_id`` are not updated. If the edges already have them
      and a roundabout is collapsed, they are outdated (an extended edge
      still refers to its old junction node) and a warning is logged.

    So call this function before :func:`add_endpoints`, :func:`add_ids` and
    :func:`add_topology` (as :func:`simplified_network` does), or run them
    again afterwards. Call it before :func:`split_edges_at_nodes`.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).
    max_length : float, optional
        Maximum length of a ring, in metres, for it to count as a roundabout.
        Longer closed rings (e.g. a ring road or a residential loop) are kept.
        Default: None (no limit, every closed ring is a roundabout).
    tolerance : float, optional
        Distance (CRS units) within which an edge end counts as lying on a
        ring. Default: 1e-9.

    Returns
    -------
    Network
        Network with roundabouts replaced by connections to their centroids,
        without the nodes that lay on them.

    Raises
    ------
    ValueError
        If ``max_length`` is given and the edges have no CRS.
    """
    edges = network.edges.copy()
    nodes = network.nodes.copy()
    if edges.empty:
        return Network(edges, nodes)

    original = edges["geometry"].to_numpy()
    geoms = original.copy()  # updated as edges are extended
    is_line = shapely.get_type_id(original) == 1
    is_ring = is_line & shapely.is_ring(original)
    ring_pos = np.flatnonzero(is_ring)
    if max_length is not None and len(ring_pos):
        ring_lengths = _edge_lengths_metres(edges.iloc[ring_pos])
        roundabout_pos = ring_pos[np.asarray(ring_lengths) <= max_length]
    else:
        roundabout_pos = ring_pos

    tree = shapely.STRtree(original)
    remove = set(roundabout_pos.tolist())
    n_extended = 0
    n_inner = 0
    for ring_row in roundabout_pos.tolist():
        ring = original[ring_row]
        centroid_xy = shapely.get_coordinates(shapely.centroid(ring))
        touching = tree.query(ring, predicate="dwithin", distance=tolerance)
        for row in touching.tolist():
            if row in remove or is_ring[row] or not is_line[row]:
                continue
            coords = shapely.get_coordinates(geoms[row])
            start_on_ring = (
                shapely.distance(shapely.points(coords[0]), ring) <= tolerance
            )
            end_on_ring = (
                shapely.distance(shapely.points(coords[-1]), ring) <= tolerance
            )
            if start_on_ring and end_on_ring:
                remove.add(row)
                n_inner += 1
            elif start_on_ring:
                geoms[row] = shapely.linestrings(np.vstack([centroid_xy, coords]))
                n_extended += 1
            elif end_on_ring:
                geoms[row] = shapely.linestrings(np.vstack([coords, centroid_xy]))
                n_extended += 1

    # nodes on collapsed rings: replaced by the centroids
    n_nodes_removed = 0
    if len(roundabout_pos) and not nodes.empty:
        _, node_pos = shapely.STRtree(nodes["geometry"].to_numpy()).query(
            original[roundabout_pos], predicate="dwithin", distance=tolerance
        )
        on_ring = np.unique(node_pos)
        n_nodes_removed = len(on_ring)
        nodes = nodes.drop(index=nodes.index[on_ring]).reset_index(drop=True)

    if len(ring_pos):
        LOGGER.info(
            "clean_roundabouts: collapsed %d roundabouts, extended %d edge ends "
            "to their centroids, removed %d edges with both ends on a roundabout "
            "and %d nodes on roundabouts; kept %d longer closed rings.",
            len(roundabout_pos),
            n_extended,
            n_inner,
            n_nodes_removed,
            len(ring_pos) - len(roundabout_pos),
        )
    if len(roundabout_pos) and _has_topology(edges):
        LOGGER.warning(
            "clean_roundabouts: from_id / to_id are outdated after collapsing "
            "roundabouts; run add_endpoints, add_ids and add_topology afterwards."
        )

    edges["geometry"] = gpd.GeoSeries(geoms, index=edges.index, crs=edges.crs)
    keep = [row for row in range(len(edges)) if row not in remove]
    edges = edges.iloc[keep].reset_index(drop=True)
    return Network(edges, nodes)


def calculate_degree(network):
    """Calculate the degree of each node from from_id and to_id columns.

    Requires node ids ``0..n-1`` in row order (call :func:`reset_ids` after
    removing nodes or edges): the returned degrees are ordered by node id and
    assigned to the nodes by position.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).

    Returns
    -------
    np.ndarray
        Array of connectivity degrees for each node.

    Raises
    ------
    ValueError
        If the node ids are not ``0..n-1`` in row order, or if ``from_id`` /
        ``to_id`` are missing, not integers, or refer to node ids that do not
        exist, or the edges have no ``from_id`` / ``to_id`` columns.
    """
    n_nodes = len(network.nodes)
    if network.edges.empty:
        return np.zeros(n_nodes, dtype=int)
    if not {"from_id", "to_id"}.issubset(network.edges.columns):
        raise ValueError(
            "calculate_degree needs edges with from_id / to_id; run add_ids and "
            "add_topology first."
        )
    if not np.array_equal(network.nodes["id"].to_numpy(), np.arange(n_nodes)):
        raise ValueError(
            "calculate_degree requires node ids 0..n-1 in row order; "
            "call reset_ids first."
        )
    from_ids = _edge_node_ids(network.edges, "from_id", n_nodes)
    to_ids = _edge_node_ids(network.edges, "to_id", n_nodes)
    return np.bincount(from_ids, minlength=n_nodes) + np.bincount(
        to_ids, minlength=n_nodes
    )


def _edge_node_ids(edges, col, n_nodes):
    """Return an edge node-reference column as validated integer node ids.

    Parameters
    ----------
    edges : gpd.GeoDataFrame
        Edge GeoDataFrame.
    col : str
        ``"from_id"`` or ``"to_id"``.
    n_nodes : int
        Number of nodes; valid ids are ``0..n_nodes-1``.

    Returns
    -------
    np.ndarray
        Integer node ids.

    Raises
    ------
    ValueError
        If the column has missing or non-integer values, or ids outside
        ``0..n_nodes-1``.
    """
    values = edges[col]
    if values.isna().any():
        raise ValueError(f"Edge column '{col}' contains missing node ids.")
    values = values.to_numpy()
    if not np.issubdtype(values.dtype, np.integer):
        if not np.issubdtype(values.dtype, np.number) or not np.all(
            np.mod(values, 1) == 0
        ):
            raise ValueError(f"Edge column '{col}' must contain integer node ids.")
        values = values.astype(int)
    if values.min() < 0 or values.max() >= n_nodes:
        raise ValueError(
            f"Edge column '{col}' refers to node ids that do not exist "
            f"(valid: 0..{n_nodes - 1})."
        )
    return values


def add_degree(network):
    """Add a degree column to the node GeoDataFrame.

    The degree is counted from the existing ``from_id`` / ``to_id`` (see
    :func:`calculate_degree`), which requires:

    - edges with ``from_id`` / ``to_id``: if they are missing, run
      :func:`add_ids` and then :func:`add_topology`;
    - node ids ``0..n-1`` in row order, referred to by ``from_id`` /
      ``to_id``: if the network has topology but its ids have gaps (e.g.
      after removing nodes or edges), run :func:`reset_ids`, which renumbers
      the nodes and updates ``from_id`` / ``to_id``.

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


def _is_missing(value):
    """True for None and float NaN (not for lists or other containers)."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def _value_key(value):
    """Hashable key for comparing attribute values (lists, e.g. osmnx osmid)."""
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _same_value(a, b):
    """Compare two attribute values, treating missing values as equal."""
    if _is_missing(a) or _is_missing(b):
        return _is_missing(a) and _is_missing(b)
    return _value_key(a) == _value_key(b)


def _merged_value(values, lengths, external_id, attr_conflict):
    """Value of one attribute for an edge merged from a chain of edges.

    Parameters
    ----------
    values : list
        Attribute values of the chain's edges.
    lengths : np.ndarray
        Lengths of the chain's edges (same order).
    external_id : bool
        Whether the column is an external id (value of the longest edge).
    attr_conflict : {"majority", "nan"}
        Rule for differing values of other columns.

    Returns
    -------
    value
        Value for the merged edge.
    conflict : bool
        Whether the values differed (always False for external ids).
    """
    if all(_same_value(v, values[0]) for v in values):
        return values[0], False
    if external_id:
        return values[int(np.argmax(lengths))], False
    if attr_conflict == "nan":
        return np.nan, True
    totals, longest, first = {}, {}, {}
    for value, length in zip(values, lengths):
        if _is_missing(value):
            continue
        key = _value_key(value)
        totals[key] = totals.get(key, 0.0) + length
        longest[key] = max(longest.get(key, 0.0), length)
        # original value per key: lists are compared by their text form
        # (see _value_key), but the merged edge must get the list itself
        if key not in first:
            first[key] = value
    best = max(totals, key=lambda k: (totals[k], longest[k]))
    return first[best], True


def _other_end(edge, node, from_ids, to_ids):
    """Return the node at the other end of ``edge`` (row position) from ``node``."""
    return to_ids[edge] if from_ids[edge] == node else from_ids[edge]


def _walk_chain(
    node, edge, deg, incident, from_ids, to_ids, visited_nodes, visited_edges
):
    """Walk along a chain of degree-2 nodes.

    Starts at ``node``, reached through ``edge``, and repeatedly follows the
    current node's other edge to that edge's far end, until it reaches a node
    whose degree is not 2. It also stops at a node or edge that is already on
    the chain (closed loop) and at a node whose incident edges do not match
    its degree.

    Parameters
    ----------
    node : int
        Node (row position) where the walk starts.
    edge : int
        Edge (row position) through which ``node`` was reached.
    deg : np.ndarray
        Degree per node.
    incident : list of list of int
        Edges (row positions) incident to each node.
    from_ids, to_ids : np.ndarray
        End nodes (row positions) of each edge.
    visited_nodes : set of int
        Nodes already on the chain (at least the chain's start node).
    visited_edges : set of int
        Edges already on the chain.

    Returns
    -------
    end : int
        Node where the walk stops: an end of the merged edge.
    chain_nodes : list of int
        Degree-2 nodes passed on the way (inner nodes of the merged edge).
    chain_edges : list of int
        Edges taken after ``edge``, in walking order.

    Examples
    --------
    In ``J -e0- A -e1- B -e2- C -e3- K`` with A, B, C of degree 2, starting
    at B: ``_walk_chain(A, e1, ...)`` returns ``(J, [A], [e0])`` and
    ``_walk_chain(C, e2, ...)`` returns ``(K, [C], [e3])``.
    """
    chain_nodes = []
    chain_edges = []
    while deg[node] == 2 and node not in visited_nodes and node not in chain_nodes:
        next_edges = [e for e in incident[node] if e != edge]
        if len(next_edges) != 1:
            break  # incident edges do not match the degree
        next_edge = next_edges[0]
        if next_edge in visited_edges or next_edge in chain_edges:
            break  # back on this chain (closed loop)
        chain_nodes.append(node)
        chain_edges.append(next_edge)
        edge = next_edge
        node = _other_end(edge, node, from_ids, to_ids)
    return node, chain_nodes, chain_edges


def merge_edges(
    network, print_err=False, attr_conflict="majority", id_cols=DEFAULT_ID_COLS
):
    """Remove degree-2 nodes and merge their associated edges.

    Finds nodes of degree 2, then follows the network topology (``from_id`` /
    ``to_id``) in both directions until a node of degree != 2 is reached, and
    merges the edges of this chain into one. The merged edge gets the merged
    geometry and the chain's end nodes as ``from_id`` / ``to_id``; the other
    edges of the chain and its inner nodes are removed. All other nodes,
    including isolated ones (degree 0), are kept. Closed loops made only of
    degree-2 nodes are left unchanged.

    Edges are selected by topology only, so edges that cross a node without
    being connected to it are never merged with it.

    Attributes of the merged edge (all columns except ``geometry``,
    ``from_id``, ``to_id`` and ``id``):

    - a value shared by all edges of the chain is kept;
    - external id columns (``id_cols``, by default ``osm_id``, ``osmid``,
      ``edge_id``) take the value of the longest edge of the chain
      (provenance: the merged edge maps back to that source feature);
    - ``distance`` is the sum of the chain's distances (missing if any is
      missing);
    - other columns whose values differ are resolved according to
      ``attr_conflict``: ``"majority"`` takes the value covering the largest
      length of the chain (missing values are ignored; ties go to the value
      of the longest edge), ``"nan"`` sets the value to missing. One warning
      per call lists the columns and the number of merged edges concerned.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).
        Node ids must be ``0..n-1`` in row order (see :func:`reset_ids`).
    print_err : bool, optional
        Whether to log a warning for each failed merge. Default: False.
    attr_conflict : {"majority", "nan"}, optional
        How to resolve attribute values that differ within a merged chain.
        Default: "majority".
    id_cols : sequence of str, optional
        Columns holding external (source-data) ids, see above. Columns not
        present in the edges are ignored. Default: :data:`DEFAULT_ID_COLS`
        (``("osm_id", "osmid", "edge_id")``).

    Returns
    -------
    Network
        Network with degree-2 nodes removed and edges merged.

    Raises
    ------
    ValueError
        If the node ids are not ``0..n-1`` in row order, ``from_id`` /
        ``to_id`` are invalid (see :func:`calculate_degree`), or
        ``attr_conflict`` is not ``"majority"`` or ``"nan"``.
    """
    if attr_conflict not in ("majority", "nan"):
        raise ValueError(
            f"attr_conflict must be 'majority' or 'nan', got {attr_conflict!r}."
        )
    if network.edges.empty:
        return network

    nodes = network.nodes.copy()
    edges = network.edges.copy()
    n_nodes = len(nodes)
    if not np.array_equal(nodes["id"].to_numpy(), np.arange(n_nodes)):
        raise ValueError(
            "merge_edges requires node ids 0..n-1 in row order; "
            "call reset_ids first."
        )

    attr_cols = [
        col
        for col in edges.columns
        if col not in ("geometry", "from_id", "to_id", "id")
    ]
    attr_values = {col: edges[col].tolist() for col in attr_cols}
    changed_cols = set()
    conflicts = {}  # column -> number of merged edges with differing values
    n_merged = 0
    lengths = shapely.length(edges["geometry"].to_numpy())

    # deg is updated during the merge (merged nodes are set to 0), so it must
    # be a writable copy, not a view of the column (read-only under
    # copy-on-write, the default from pandas 3)
    if "degree" not in nodes.columns:
        deg = calculate_degree(network)
    else:
        deg = nodes["degree"].to_numpy(copy=True)

    # Everything below works on row positions: edge e connects from_ids[e]
    # and to_ids[e]; node ids are row positions (checked above)
    from_ids = _edge_node_ids(edges, "from_id", n_nodes).copy()  # writable
    to_ids = _edge_node_ids(edges, "to_id", n_nodes).copy()
    geoms = edges["geometry"].to_numpy().copy()
    node_geoms = nodes["geometry"].to_numpy()
    incident = [[] for _ in range(n_nodes)]  # node -> positions of its edges
    for pos, (from_id, to_id) in enumerate(zip(from_ids, to_ids)):
        incident[from_id].append(pos)
        incident[to_id].append(pos)

    n2 = set(np.flatnonzero(deg == 2).tolist())
    edges_to_remove = set()
    removed_nodes = set()  # inner nodes of successfully merged chains

    while n2:
        start = n2.pop()
        start_edges = incident[start]
        if len(start_edges) != 2 or start_edges[0] == start_edges[1]:
            continue  # degree column inconsistent with the edges, or self-loop
        edge_1, edge_2 = start_edges
        next_1 = _other_end(edge_1, start, from_ids, to_ids)
        next_2 = _other_end(edge_2, start, from_ids, to_ids)
        if next_1 == next_2:
            continue  # two parallel edges between the same nodes
        # walk both sides of the chain; the merged edge keeps edge_1's row
        end_1, nodes_1, edges_1 = _walk_chain(
            next_1,
            edge_1,
            deg,
            incident,
            from_ids,
            to_ids,
            visited_nodes={start},
            visited_edges={edge_1, edge_2},
        )
        end_2, nodes_2, edges_2 = _walk_chain(
            next_2,
            edge_2,
            deg,
            incident,
            from_ids,
            to_ids,
            visited_nodes={start, *nodes_1},
            visited_edges={edge_1, edge_2, *edges_1},
        )
        chain_nodes = [start, *nodes_1, *nodes_2]
        members = [edge_1, edge_2, *edges_1, *edges_2]
        n2.difference_update(chain_nodes)
        if deg[end_1] == 2 and deg[end_2] == 2:
            # neither side reached a node of degree != 2: a closed loop of
            # degree-2 nodes (both walks end at the same node), or a degree
            # column that does not match the edges -> leave unchanged
            continue

        new_merged_geom = shapely.line_merge(
            shapely.multilinestrings([geoms[e] for e in members])
        )
        if shapely.get_type_id(new_merged_geom) != 1:
            if print_err:
                LOGGER.warning(
                    "Edge %s failed to merge, merged geometry has shapely type %s",
                    edge_1,
                    shapely.get_type_id(new_merged_geom),
                )
            continue

        geoms[edge_1] = new_merged_geom
        start_point = shapely.get_point(new_merged_geom, 0)
        if shapely.distance(node_geoms[end_1], start_point) <= shapely.distance(
            node_geoms[end_2], start_point
        ):
            from_ids[edge_1], to_ids[edge_1] = end_1, end_2
        else:
            from_ids[edge_1], to_ids[edge_1] = end_2, end_1
        edges_to_remove.update(members[1:])
        for x in chain_nodes:
            deg[x] = 0
        # keep the end nodes of the merged edge (relevant for closed loops)
        removed_nodes.update(x for x in chain_nodes if x not in (end_1, end_2))
        n_merged += 1
        for col in attr_cols:
            values = [attr_values[col][e] for e in members]
            if col == "distance":  # lengths add up along the chain
                new_value = (
                    np.nan if any(_is_missing(v) for v in values) else sum(values)
                )
                if not _same_value(new_value, attr_values[col][edge_1]):
                    attr_values[col][edge_1] = new_value
                    changed_cols.add(col)
                continue
            new_value, conflict = _merged_value(
                values,
                lengths[members],
                external_id=col in id_cols,
                attr_conflict=attr_conflict,
            )
            if conflict:
                conflicts[col] = conflicts.get(col, 0) + 1
            if not _same_value(new_value, attr_values[col][edge_1]):
                attr_values[col][edge_1] = new_value
                changed_cols.add(col)

    if conflicts:
        LOGGER.warning(
            "merge_edges: attribute values differed within merged edges and were "
            "resolved by %s (%d edges merged): %s",
            "length-weighted majority" if attr_conflict == "majority" else "NaN",
            n_merged,
            ", ".join(f"{col} ({n} edges)" for col, n in sorted(conflicts.items())),
        )

    for col in changed_cols:
        edges[col] = pd.Series(attr_values[col], index=edges.index)
    edges["from_id"] = from_ids
    edges["to_id"] = to_ids
    edges["geometry"] = gpd.GeoSeries(geoms, index=edges.index, crs=edges.crs)
    keep = [pos for pos in range(len(edges)) if pos not in edges_to_remove]
    edges = edges.iloc[keep].reset_index(drop=True)

    # remove only the inner nodes of merged chains (not isolated nodes)
    nodes = nodes.loc[~nodes["id"].isin(removed_nodes)].reset_index(drop=True)

    return Network(edges, nodes)


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
    # boolean mask: works for any index, not only 0..n-1
    return dataframe[~mask.duplicated(keep=keep)]


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

    Raises
    ------
    ValueError
        If ``from_id`` or ``to_id`` refer to node ids that are not in the nodes.
    """
    # Copy nodes and edges to avoid modifying the original data
    nodes = network.nodes.copy()
    edges = network.edges.copy()

    # Generate new node IDs
    new_node_ids = np.arange(len(nodes))

    # Map old node IDs to new IDs using a dictionary
    id_map = dict(zip(nodes["id"], new_node_ids))

    # Map old IDs to new IDs in 'from_id' and 'to_id'; unknown ids become NaN
    for col in ("from_id", "to_id"):
        mapped = edges[col].map(id_map)
        if mapped.isna().any():
            unknown = sorted(edges.loc[mapped.isna(), col].unique().tolist())
            raise ValueError(
                f"Edge column '{col}' refers to node ids that are not in the "
                f"nodes: {unknown[:10]}{' ...' if len(unknown) > 10 else ''}"
            )
        edges[col] = mapped.astype(int)

    # Update node and edge IDs
    nodes["id"] = new_node_ids
    edges["id"] = np.arange(len(edges))

    # Reset indices to ensure a clean, sequential index
    edges.reset_index(drop=True, inplace=True)
    nodes.reset_index(drop=True, inplace=True)

    return Network(edges, nodes)


def split_edges_at_nodes(network, mode="anywhere", tolerance=1e-9):
    """Split edges where they meet nodes or other edges.

    The split points of an edge are

    - the nodes within ``tolerance`` of the edge, and
    - the points where it meets another edge: crossing or touching points,
      and the start and end of stretches the two edges share (the inner
      vertices of a shared stretch are not split points, so e.g. an identical
      duplicate edge does not cut the edge at all).

    With ``mode="anywhere"`` (default) the edge is cut at every split point,
    also between two of its vertices (the cut point becomes a new vertex).
    With ``mode="vertices"`` it is only cut at its own vertices that lie
    within ``tolerance`` of a split point; split points between two vertices
    are ignored, so e.g. a side road ending in the middle of a segment of a
    main road, without a shared vertex, stays unconnected.

    Both ends of every edge are kept, so no geometry is lost. Each piece
    copies all attributes of its edge, including ``id``, ``from_id``,
    ``to_id`` and external ids such as ``osm_id``. Run :func:`add_ids` and
    :func:`add_topology` afterwards to get unique ids and correct topology.
    Edges that are not LineStrings (e.g. MultiLineStrings, see
    :func:`merge_multilinestrings`) are returned unchanged.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).
    mode : {"anywhere", "vertices"}, optional
        Where edges are cut, see above. Default: "anywhere".
    tolerance : float, optional
        Distance (CRS units) within which a node counts as lying on an edge,
        a vertex as coinciding with a split point, and below which pieces are
        not created. Default: 1e-9.

    Returns
    -------
    Network
        Network with edges split at the split points.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"anywhere"`` or ``"vertices"``.
    """
    if mode not in ("anywhere", "vertices"):
        raise ValueError(f"mode must be 'anywhere' or 'vertices', got {mode!r}.")
    edges = network.edges.copy()
    nodes = network.nodes.copy()
    if edges.empty:
        return Network(edges, nodes)

    geoms = edges["geometry"].to_numpy()
    is_line = shapely.get_type_id(geoms) == 1
    line_pos = np.flatnonzero(is_line)
    hits = {pos: [] for pos in line_pos.tolist()}

    # Spatial queries below return pairs (k, i): the k-th *queried* geometry
    # (a position within geoms[line_pos]) and the i-th geometry of the tree
    # (a row position of nodes or edges). line_pos[k] translates k into the
    # row position of the queried line.

    # nodes within tolerance of each line
    node_geoms = nodes["geometry"].to_numpy()
    if len(node_geoms):
        query_k, node_pos = shapely.STRtree(node_geoms).query(
            geoms[line_pos], predicate="dwithin", distance=tolerance
        )
        line_of_pair = line_pos[query_k]  # row position of the line per pair
        coords, pair = shapely.get_coordinates(node_geoms[node_pos], return_index=True)
        for k, xy in zip(pair, coords):
            hits[int(line_of_pair[k])].append(xy)

    # points where each line meets another edge
    query_k, other_pos = shapely.STRtree(geoms).query(
        geoms[line_pos], predicate="intersects"
    )
    line_of_pair = line_pos[query_k]  # row position of the line per pair
    not_self = line_of_pair != other_pos  # every line intersects itself
    line_of_pair, other_pos = line_of_pair[not_self], other_pos[not_self]
    intersections = shapely.intersection(geoms[line_of_pair], geoms[other_pos])
    for pos, intersection in zip(line_of_pair.tolist(), intersections):
        hits[pos].extend(_meeting_points(intersection))

    rows, pieces = [], []
    for pos in range(len(edges)):
        if not is_line[pos]:
            rows.append(pos)
            pieces.append(geoms[pos])
            continue
        for piece in _split_line(geoms[pos], hits[pos], mode, tolerance):
            rows.append(pos)
            pieces.append(piece)

    edges = edges.iloc[rows].reset_index(drop=True)
    edges["geometry"] = gpd.GeoSeries(pieces, index=edges.index, crs=network.edges.crs)
    return Network(edges, nodes)


def split_long_edges(network, max_length, inherit_attrs=("ci_type",)):
    """Split edges longer than ``max_length`` metres into pieces of equal length.

    Every LineString edge longer than ``max_length`` is cut into
    ``ceil(length / max_length)`` pieces of equal length along the line (the
    geometry, including curves, is kept), with a new node at each cut point.
    Use cases: more localised impacts (an edge fails as a whole, so shorter edges
    make failures more local) and regular points along roads where other
    infrastructure or people can be linked, instead of only the vertices of
    the source data (e.g. OSM).

    Works on a network with topology (``from_id`` / ``to_id``, e.g. the
    output of :func:`simplified_network`) and keeps it consistent:

    - new nodes get ids after the existing ones and copy the
      ``inherit_attrs`` of their edge (e.g. ``ci_type``, so that selections by
      type include them); other node columns are missing (a ``degree`` column
      gets 2);
    - each piece copies all attributes of its edge (e.g. ``osm_id`` for
      provenance) and is connected to its neighbouring pieces; ``from_id`` of
      a piece is the node at the start of its geometry;
    - ``distance`` (if present) is recomputed per piece; edge ids are
      renumbered ``0..n-1``.

    Do not run :func:`merge_edges` or :func:`simplified_network` afterwards:
    the cut points have degree 2 and would be merged again. Lengths are
    geodesic for a geographic CRS; in that case the pieces are equal in
    length along the line in CRS units, so in metres they are only
    approximately equal. Non-LineString edges are left unchanged.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines), with
        ``from_id`` / ``to_id``.
    max_length : float
        Maximum edge length in metres.
    inherit_attrs : sequence of str, optional
        Edge columns copied onto the new nodes. Columns not present in the
        edges are ignored. Default: ``("ci_type",)``.

    Returns
    -------
    Network
        Network whose edges are at most about ``max_length`` metres long.

    Raises
    ------
    ValueError
        If ``max_length`` is not positive, the edges have no ``from_id`` /
        ``to_id``, or no CRS.
    """
    if max_length is None or not max_length > 0:
        raise ValueError(f"max_length must be positive, got {max_length!r}.")
    edges = network.edges.copy()
    nodes = network.nodes.copy()
    if edges.empty:
        return Network(edges, nodes)
    if not _has_topology(edges):
        raise ValueError(
            "split_long_edges needs edges with from_id / to_id; run "
            "simplified_network (or add_endpoints, add_ids, add_topology) first."
        )

    geoms = edges["geometry"].to_numpy()
    is_line = shapely.get_type_id(geoms) == 1
    lengths = np.asarray(_edge_lengths_metres(edges), dtype=float)
    n_pieces = np.ones(len(edges), dtype=int)
    too_long = is_line & (lengths > max_length)
    n_pieces[too_long] = np.ceil(lengths[too_long] / max_length).astype(int)
    if not too_long.any():
        return Network(edges, nodes)

    node_geom_by_id = pd.Series(nodes["geometry"].to_numpy(), index=nodes["id"])
    from_ids = edges["from_id"].to_numpy()
    to_ids = edges["to_id"].to_numpy()
    next_id = int(nodes["id"].max()) + 1 if len(nodes) else 0
    first_new_id = next_id

    rows, piece_geoms, piece_from, piece_to = [], [], [], []
    new_node_geoms, new_node_edge = [], []
    for pos in range(len(edges)):
        # edges that are not LineStrings or are short enough are kept unchanged
        if n_pieces[pos] == 1:
            rows.append(pos)
            piece_geoms.append(geoms[pos])
            piece_from.append(from_ids[pos])
            piece_to.append(to_ids[pos])
            continue
        line = geoms[pos]
        start = shapely.get_point(line, 0)
        start_node, end_node = from_ids[pos], to_ids[pos]
        if shapely.distance(node_geom_by_id[end_node], start) < shapely.distance(
            node_geom_by_id[start_node], start
        ):
            start_node, end_node = end_node, start_node  # from_id at the line's end
        fractions = np.linspace(0.0, 1.0, n_pieces[pos] + 1)
        cut_ids = list(range(next_id, next_id + len(fractions) - 2))
        next_id += len(cut_ids)
        for fraction in fractions[1:-1]:
            new_node_geoms.append(
                shapely.line_interpolate_point(line, fraction, normalized=True)
            )
            new_node_edge.append(pos)
        ends = [start_node, *cut_ids, end_node]
        for a, b, u, v in zip(fractions[:-1], fractions[1:], ends[:-1], ends[1:]):
            rows.append(pos)
            piece_geoms.append(substring(line, a, b, normalized=True))
            piece_from.append(u)
            piece_to.append(v)

    out = edges.iloc[rows].reset_index(drop=True)
    out["geometry"] = gpd.GeoSeries(piece_geoms, index=out.index, crs=edges.crs)
    out["from_id"] = piece_from
    out["to_id"] = piece_to
    out["id"] = np.arange(len(out))
    if "distance" in out.columns:
        out["distance"] = _edge_lengths_metres(out)

    new_nodes = gpd.GeoDataFrame(
        {"id": np.arange(first_new_id, next_id), "geometry": new_node_geoms},
        geometry="geometry",
        crs=nodes.crs if nodes.crs is not None else edges.crs,
    )
    for attr in inherit_attrs:
        if attr in edges.columns:
            new_nodes[attr] = edges[attr].to_numpy()[new_node_edge]
    if "degree" in nodes.columns:
        new_nodes["degree"] = 2
    nodes = pd.concat([nodes, new_nodes], ignore_index=True)

    LOGGER.info(
        "split_long_edges: %d edges longer than %g m split into %d pieces; "
        "%d nodes added.",
        int(too_long.sum()),
        max_length,
        int(n_pieces[too_long].sum()),
        len(new_nodes),
    )
    return Network(out, nodes)


def _meeting_points(intersection):
    """Points where two lines meet, from the geometry of their intersection.

    Point parts (crossing or touching points) are returned as they are. Line
    parts (shared stretches) are first merged, and only the start and end of
    each merged stretch are returned, not its inner vertices.

    Parameters
    ----------
    intersection : shapely.Geometry
        Intersection of two lines (Point, LineString, multi-part or
        GeometryCollection).

    Returns
    -------
    list of np.ndarray
        ``(x, y)`` coordinates of the meeting points.
    """
    parts = shapely.get_parts(intersection)
    while np.any(shapely.get_type_id(parts) >= 4):  # nested multi / collection
        parts = shapely.get_parts(parts)
    type_ids = shapely.get_type_id(parts)
    points = list(shapely.get_coordinates(parts[type_ids == 0]))
    lines = parts[(type_ids == 1) | (type_ids == 2)]
    if len(lines):
        merged = shapely.get_parts(shapely.line_merge(shapely.multilinestrings(lines)))
        for stretch in merged:
            coords = shapely.get_coordinates(stretch)
            points.extend([coords[0], coords[-1]])
    return points


def _split_line(line, points, mode, tolerance):
    """Cut a LineString at the given points.

    Parameters
    ----------
    line : shapely.LineString
        Line to cut.
    points : list of array_like
        ``(x, y)`` split points on or near the line.
    mode : {"anywhere", "vertices"}
        Cut at any point along the line, or only at the line's own vertices
        within ``tolerance`` of a split point.
    tolerance : float
        See :func:`split_edges_at_nodes`.

    Returns
    -------
    list of shapely.LineString
        Pieces in order along the line; ``[line]`` if it is not cut. The first
        piece starts at the line's start and the last ends at its end.
    """
    if len(points) == 0:
        return [line]
    points = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)

    if mode == "anywhere":
        distances = shapely.line_locate_point(line, shapely.points(points))
        # sorted candidate cuts as distances along the line, incl. start and end
        cuts = np.unique(np.concatenate(([0.0], distances, [line.length])))
        # keep a cut only if it is more than `tolerance` beyond the last kept
        # one: drops near-duplicate cuts, i.e. zero-length pieces
        cut_positions = [cuts[0]]
        for cut in cuts[1:]:
            if cut - cut_positions[-1] > tolerance:
                cut_positions.append(cut)
        # if the end itself was dropped as a near-duplicate, move the last kept
        # cut exactly onto the end so that no piece stops short of it
        cut_positions[-1] = line.length
        if len(cut_positions) <= 2:
            return [line]
        return [
            substring(line, a, b) for a, b in zip(cut_positions[:-1], cut_positions[1:])
        ]

    coords = shapely.get_coordinates(line)
    # distance of each vertex to its closest split point
    vertex_dist = np.min(
        np.hypot(
            coords[:, None, 0] - points[None, :, 0],
            coords[:, None, 1] - points[None, :, 1],
        ),
        axis=1,
    )
    cut_idx = sorted(
        set(np.flatnonzero(vertex_dist <= tolerance).tolist()) | {0, len(coords) - 1}
    )
    if len(cut_idx) <= 2:
        return [line]
    return [
        shapely.linestrings(coords[a : b + 1])
        for a, b in zip(cut_idx[:-1], cut_idx[1:])
    ]


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


# =============================================================================
# Simplification wrappers
# =============================================================================


def _check_single_ci_type(network):
    """Check that edges and nodes share one ``ci_type``; warn if edges lack it.

    Parameters
    ----------
    network : Network
        Network to check.

    Raises
    ------
    ValueError
        If the edges have more than one ``ci_type``, or the nodes have a
        ``ci_type`` different from the edges.
    """
    edges, nodes = network.edges, network.nodes
    edge_types = set(edges["ci_type"].dropna()) if "ci_type" in edges.columns else set()
    node_types = set(nodes["ci_type"].dropna()) if "ci_type" in nodes.columns else set()
    hint = (
        "Simplify each ci_type separately and combine the results with "
        "Network.from_networks."
    )
    if len(edge_types) > 1:
        raise ValueError(
            f"Edges must have a single ci_type, got "
            f"{sorted(str(ci) for ci in edge_types)}. {hint}"
        )
    if len(edge_types | node_types) > 1:
        raise ValueError(
            "Edges and nodes must have the same single ci_type, got edges "
            f"{sorted(str(ci) for ci in edge_types)} and nodes "
            f"{sorted(str(ci) for ci in node_types)}. Point infrastructure "
            f"(e.g. facilities) belongs in its own network. {hint}"
        )
    if edges.empty:
        return
    if not edge_types:
        LOGGER.warning(
            "Edges have no ci_type: nodes created at edge endpoints will not "
            "have one either. Set edges['ci_type'] before simplifying."
        )
        return
    n_missing = edges["ci_type"].isna().sum()
    if n_missing:
        LOGGER.warning(
            "%d of %d edges have no ci_type: nodes created at their endpoints "
            "will not have one either.",
            n_missing,
            len(edges),
        )


def _drop_isolated_nodes(network):
    """Remove nodes that are not the end of any edge, logging how many.

    Parameters
    ----------
    network : Network
        Network with ``from_id`` / ``to_id`` referring to node ``id``.

    Returns
    -------
    Network
        Network without isolated nodes.
    """
    edges = network.edges.copy()
    nodes = network.nodes.copy()
    connected = set(edges["from_id"]) | set(edges["to_id"])
    isolated = ~nodes["id"].isin(connected)
    if isolated.any():
        LOGGER.warning(
            "Removed %d of %d nodes that are not connected to any edge.",
            int(isolated.sum()),
            len(nodes),
        )
    return Network(edges, nodes.loc[~isolated].reset_index(drop=True))


def _set_ci_type(gdf, ci_type, name):
    """Copy of ``gdf`` with ``ci_type`` set; raise if it already has another one."""
    gdf = gdf.copy()
    if gdf.empty:
        return gdf
    if "ci_type" in gdf.columns:
        existing = set(gdf["ci_type"].dropna()) - {ci_type}
        if existing:
            raise ValueError(
                f"ci_type={ci_type!r} was given, but the {name} already have "
                f"ci_type {sorted(str(ci) for ci in existing)}."
            )
    gdf["ci_type"] = ci_type
    return gdf


def simplified_network(
    network,
    profile="generic",
    *,
    ci_type=None,
    id_cols=DEFAULT_ID_COLS,
    roundabouts=None,
    roundabout_max_length=None,
    split=None,
):
    """Return a simplified network of a single infrastructure type.

    The steps depend on ``profile`` (see :data:`PREPROCESSING_PROFILES`):
    ``"generic"`` (topological simplification only), ``"road"`` (also
    collapses roundabouts up to 1000 m and splits edges wherever they meet,
    so that e.g. OSM ways crossing at an inner vertex are connected) or
    ``"power_line"`` (splits edges wherever they meet). Every setting of
    the profile can be overridden by the keyword argument of the same name;
    ``None`` keeps the profile's value.

    For a network with edges, applies in this order:

    1. :func:`merge_multilinestrings`: MultiLineStrings whose parts connect
       become LineStrings;
    2. :func:`explode_multilinestrings`: remaining MultiLineStrings are split
       into one LineString edge per part (attributes copied), so no edge is
       lost;
    3. :func:`drop_duplicate_geometries`: identical edges are removed (before
       the topology is built, so they do not distort node degrees);
    4. if ``roundabouts``: :func:`clean_roundabouts` (rings up to
       ``roundabout_max_length`` metres collapse into their centroids; nodes
       on them are removed);
    5. :func:`add_endpoints`; if ``split``: :func:`split_edges_at_nodes` and
       :func:`add_endpoints` again (nodes at the new cut points);
    6. :func:`add_ids`, :func:`add_topology`: ``from_id`` / ``to_id``;
    7. :func:`merge_edges`: chains of degree-2 nodes merge into one edge;
    8. nodes connected to no edge are removed (e.g. endpoints of edges
       dropped by :func:`add_topology`), with a warning giving their number;
    9. :func:`reset_ids` and :func:`add_distances`.

    The settings used and a summary of the changes are logged (INFO). A
    network without edges is returned unchanged, with all its nodes (and
    ``ci_type`` set if given).

    Edges and nodes must all have the same ``ci_type``: simplify each
    infrastructure type separately and combine the results with
    :meth:`Network.from_networks`. Nodes created at edge endpoints inherit
    the ``ci_type`` of their edge (see :func:`add_endpoints`): pass
    ``ci_type`` or set it on the edges beforehand; a warning is logged if it
    is missing.

    Parameters
    ----------
    network : Network
        A network composed of nodes (points in space) and edges (lines).
    profile : str, optional
        Name of the preprocessing profile, a key of
        :data:`PREPROCESSING_PROFILES`. Default: "generic".
    ci_type : str, optional
        If given, set as ``ci_type`` of all edges and nodes. Raises if they
        already have a different one. Default: None (use the data's own).
    id_cols : sequence of str, optional
        Edge columns holding external (source-data) ids, kept for provenance;
        see :func:`merge_edges`. Default: :data:`DEFAULT_ID_COLS`.
    roundabouts : bool, optional
        Collapse roundabouts. Default: from the profile.
    roundabout_max_length : float, optional
        Maximum ring length (metres) for a roundabout. Default: from the
        profile. To remove the profile's limit, override the profile entry
        or call :func:`clean_roundabouts` directly.
    split : {False, "anywhere", "vertices"}, optional
        Split edges where they meet nodes or other edges: not at all, at
        every meeting point, or only at the edges' own vertices (see
        :func:`split_edges_at_nodes`). Default: from the profile.

    Returns
    -------
    Network
        Simplified network.

    Raises
    ------
    ValueError
        If ``profile`` or ``split`` is invalid, ``ci_type`` conflicts with the
        data, edges and nodes do not share a single ``ci_type``, or the edges
        have no CRS.
    """
    if profile not in PREPROCESSING_PROFILES:
        raise ValueError(
            f"Unknown profile {profile!r}; choose one of "
            f"{sorted(PREPROCESSING_PROFILES)}."
        )
    settings = dict(PREPROCESSING_PROFILES[profile])
    overrides = {
        "roundabouts": roundabouts,
        "roundabout_max_length": roundabout_max_length,
        "split": split,
    }
    settings.update({key: val for key, val in overrides.items() if val is not None})
    if settings["split"] not in (False, "anywhere", "vertices"):
        raise ValueError(
            "split must be False, 'anywhere' or 'vertices', "
            f"got {settings['split']!r}."
        )

    edges, nodes = network.edges, network.nodes
    if ci_type is not None:
        edges = _set_ci_type(edges, ci_type, "edges")
        nodes = _set_ci_type(nodes, ci_type, "nodes")
    network_simp = Network(edges.copy(), nodes.copy())
    _check_single_ci_type(network_simp)
    if network_simp.edges.empty:
        return network_simp
    LOGGER.info("simplified_network: profile %r with settings %s", profile, settings)

    counts = {"edges_in": len(network_simp.edges)}
    n_multi = int(
        (shapely.get_type_id(network_simp.edges.geometry.to_numpy()) == 5).sum()
    )

    network_simp = merge_multilinestrings(network_simp)
    n_multi_left = int(
        (shapely.get_type_id(network_simp.edges.geometry.to_numpy()) == 5).sum()
    )
    counts["multi_merged"] = n_multi - n_multi_left
    n_before = len(network_simp.edges)
    network_simp = explode_multilinestrings(network_simp)
    counts["multi_split_added"] = len(network_simp.edges) - n_before

    n_before = len(network_simp.edges)
    network_simp.edges = drop_duplicate_geometries(network_simp.edges, keep="first")
    counts["duplicates"] = n_before - len(network_simp.edges)

    n_before = len(network_simp.edges)
    if settings["roundabouts"]:
        network_simp = clean_roundabouts(
            network_simp, max_length=settings["roundabout_max_length"]
        )
    counts["roundabouts"] = n_before - len(network_simp.edges)

    network_simp = add_endpoints(network_simp)
    n_before = len(network_simp.edges)
    if settings["split"]:
        network_simp = split_edges_at_nodes(network_simp, mode=settings["split"])
        network_simp = add_endpoints(network_simp)
    counts["split_added"] = len(network_simp.edges) - n_before

    network_simp = add_ids(network_simp)
    n_before = len(network_simp.edges)
    network_simp = add_topology(network_simp)
    counts["topology_dropped"] = n_before - len(network_simp.edges)

    network_simp.nodes["degree"] = calculate_degree(network_simp)
    n_before = len(network_simp.edges)
    network_simp = merge_edges(network_simp, id_cols=id_cols)
    counts["merged_away"] = n_before - len(network_simp.edges)

    network_simp = _drop_isolated_nodes(network_simp)
    network_simp = reset_ids(network_simp)
    network_simp = add_distances(network_simp)

    LOGGER.info(
        "simplified_network: %d edges in, %d out; %d MultiLineStrings merged, "
        "%d edges added by splitting MultiLineStrings into parts, %d duplicate "
        "edges removed, %d edges removed with roundabouts, %d edges added by "
        "splitting at nodes, %d edges dropped by add_topology, %d edges merged "
        "into longer ones; %d nodes out.",
        counts["edges_in"],
        len(network_simp.edges),
        counts["multi_merged"],
        counts["multi_split_added"],
        counts["duplicates"],
        counts["roundabouts"],
        counts["split_added"],
        counts["topology_dropped"],
        counts["merged_away"],
        len(network_simp.nodes),
    )
    return network_simp


def add_attributes(network, attrs, on="both"):
    """Return a copy of the network with constant attributes added.

    Parameters
    ----------
    network : Network
        The network to add attributes to. It is not modified.
    attrs : dict
        Column name -> value; each value is set on all rows. Existing columns
        are overwritten.
    on : {"both", "edges", "nodes"}, optional
        Where to add the attributes. Default: "both".

    Returns
    -------
    Network
        Copy of the network with the added attributes.

    Raises
    ------
    ValueError
        If ``on`` is not "both", "edges" or "nodes".
    """
    if on not in ("both", "edges", "nodes"):
        raise ValueError(f"on must be 'both', 'edges' or 'nodes', got {on!r}.")
    edges = network.edges.copy()
    nodes = network.nodes.copy()
    for key, value in attrs.items():
        if on in ("both", "edges"):
            edges[key] = value
        if on in ("both", "nodes"):
            nodes[key] = value
    return Network(edges, nodes)
