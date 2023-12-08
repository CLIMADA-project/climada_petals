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

"""
import logging
import igraph as ig
import numpy as np
import pandas as pd
import pyproj
from tqdm import tqdm
import scipy

from climada_petals.engine.networks.nw_base import Network, Graph
from climada_petals.engine.networks.nw_utils import (make_edge_geometries,
                                                     _ckdnearest,
                                                     _preselect_destinations)
from climada.util.constants import ONE_LAT_KM

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')


# =============================================================================
# Making links
# =============================================================================

def link_clusters(graph, dist_thresh=np.inf, metres=True, link_attrs=None):
    """
    link nodes from different clusters to their nearest nodes in other
    clusters to generate one connected graph.

    Parameters
    ----------
    graph : nw_base.Graph object
    dist_thresh : float
        distance threshold up to where clusters can be linked
    metres : bool
        whether distance is in metres

    Returns
    -------
    graph
    """

    gdf_vs = graph_linked.graph.get_vertex_dataframe()
    gdf_vs['membership'] = graph_linked.graph.connected_components().membership

    v_ids_source = []
    v_ids_target = []

    # very rough conversion from metres to degrees
    dist_thresh /= (ONE_LAT_KM*1000)

    for member in np.unique(gdf_vs['membership']):
        gdf_a = gdf_vs[gdf_vs['membership'] == member]
        gdf_b = gdf_vs[gdf_vs['membership'] != member]
        try:
            dists, ix_match = _ckdnearest(
                gdf_a, gdf_b, dist_thresh=dist_thresh)
            source = gdf_a.iloc[np.where(dists == min(dists))[
                0]].index[0]
            target = gdf_b.loc[ix_match[np.where(dists == min(dists))[
                0]]].index[0]
            v_ids_source.append(source)
            v_ids_target.append(target)
        except (IndexError, KeyError):
            # if no match within given distance
            continue

    if len(v_ids_source) > 0:
        graph_linked = _edges_from_vlists(
            graph_linked, v_ids_source, v_ids_target, link_attrs)

    return graph_linked


def link_vertices_closest_k(graph, source_attrs, target_attrs, link_attrs=None,
                            dist_thresh=np.inf, bidir=False, k=5):
    """
    find k nearest source vertices for each target vertex,
    given distance constraints and identifying attributes

    Parameters
    ----------
    graph : nw_base.Graph object
    source_attrs : dict {attr_name_s1 : attr_val_s1, ..., }
    target_attrs : dict {attr_name_t1 : attr_val_t1, ..., }


    Returns
    -------
    graph
    """

    # select only those for which specified attrs apply
    df_vs_target = _filter_vertices(graph.graph, target_attrs)

    # select only those for which specified attrs apply
    df_vs_source = _filter_vertices(graph.graph, source_attrs)

    v_ids_source, v_ids_target = _select_closest_k(
        df_vs_source, df_vs_target, dist_thresh, bidir, k)

    graph = _edges_from_vlists(graph, v_ids_source, v_ids_target, link_attrs)

    return graph


def link_vertices_edgecond(graph, target_attrs, edge_attrs, link_attrs,
                           bidir=False):
    """
    make a dependency edge between two vertices if another edge with a
    certain attribute (specified in edge_attrs) already exists between those
    two.
    Primarily intended for dependency_road_people, given that a road exists
    directly at people node.

    Parameters
    ----------
    graph : nw_base.Graph object
    target_attrs : dict
    edge_attrs : dict
    link_attrs : dict
    bidir : bool

    Returns
    -------
    graph
    """
    df_vs_target = _filter_vertices(graph.graph, target_attrs)

    vs_target = graph.graph.vs[df_vs_target.index.values]

    pot_edges = [graph.graph.incident(v_target, mode='in')
                 for v_target in vs_target]
    # flatten nested list
    pot_edges = [item for sublist in pot_edges for item in sublist]

    # select those edges which fulfill edge_attrs
    for key, value in edge_attrs.items():
        pot_edges = [graph.graph.es[item] for item in pot_edges
                     if graph.graph.es[item][key] == value]

    _edges_from_vlists(graph, [edge.source for edge in pot_edges],
                       [edge.target for edge in pot_edges], link_attrs)
    if bidir:
        _edges_from_vlists(graph, [edge.target for edge in pot_edges],
                           [edge.source for edge in pot_edges], link_attrs)

    return graph


def link_vertices_shortest_paths(graph, source_attrs, target_attrs, via_attrs,
                                 link_attrs, dist_thresh=10e6, criterion='distance',
                                 single_shortest=True, bidir=False):
    """
    Per target, choose single shortest path to source which is
    below dist_thresh.

    Parameters
    ----------
    graph : nw_base.Graph object
    source_attrs : dict
    target_attrs : dict
    via_attrs : dict
    link_attrs : dict
    single_shortest : bool
        Whether to make a link between all sources and targets for which the
        shortest path is < dist_thresh, or whether to only make a link for the
        shortest of all.
    bidir : bool

    Returns
    -------
    graph
    """

    # subgraph containing only "allowed" elements
    subgraph = _create_subgraph(
        graph, source_attrs, target_attrs, via_attrs)

    # mapping from subgraph to graph indices
    subgraph_graph_vsdict = _get_subgraph2graph_vsdict(graph.graph, subgraph)

    # select only those for which specified attrs apply
    df_vs_target = _filter_vertices(subgraph, target_attrs)

    # select only those for which specified attrs apply
    df_vs_source = _filter_vertices(subgraph, source_attrs)

    path_dists = subgraph.distances(
        source=df_vs_source.index.values, target=df_vs_target.index.values,
        weights=criterion)
    path_dists = np.array(path_dists)  # dim: (#sources, #targets)

    if len(path_dists) == 0:
        return graph

    if single_shortest:
        ix_source, ix_target = np.where(
            ((path_dists == path_dists.min(axis=0)) &
             (path_dists <= dist_thresh)))  # min dist. per target
    else:
        ix_source, ix_target = np.where(path_dists < dist_thresh)

    # re-map sources to original graph
    v_ids_source = [subgraph_graph_vsdict[v_id_source] for v_id_source
                    in df_vs_source.index.values[list(ix_source)]]

    # re-map targets to original graph
    v_ids_target = [subgraph_graph_vsdict[v_id_target] for v_id_target
                    in df_vs_target.index.values[list(ix_target)]]

    link_attrs['distance'] = path_dists[(ix_source, ix_target)]

    _edges_from_vlists(graph, v_ids_source, v_ids_target, link_attrs)

    if bidir:
        _edges_from_vlists(graph, v_ids_target, v_ids_source, link_attrs)

    return graph


# =============================================================================
# Helper funcs for making links
# =============================================================================

def _filter_vertices(graph, attr_dict):
    """
    get vertices of graph to which given attributes apply

    Parameters
    ----------
    graph : igraph.Graph object

    Returns
    -------
    df_vs : pd.Dataframe
    """

    df_vs = graph.get_vertex_dataframe()
    for key, value in attr_dict.items():
        df_vs = df_vs[df_vs[key] == value]
    return df_vs


def _filter_edges(graph, attr_dict):
    """
    get edges of graph to which given attributes apply

    Parameters
    ----------
    graph : igraph.Graph object

    Returns
    -------
    df_es : pd.Dataframe
    """

    df_es = graph.get_edge_dataframe()
    for key, value in attr_dict.items():
        df_es = df_es[df_es[key] == value]
    return df_es


def _edges_from_vlists(graph, v_ids_source, v_ids_target, link_attrs=None):
    """
    add edges to graph given source and target vertex lists
    adds geometries, edge lengths, edge names and func states as attributes

    Parameters
    ----------
    graph : nw_base.Graph object

    Returns
    -------
    graph : nw_base.Graph object
    """

    pairs = list(zip(v_ids_source, v_ids_target))

    link_attrs['geometry'] = make_edge_geometries(
        graph.graph.vs[v_ids_source]['geometry'],
        graph.graph.vs[v_ids_target]['geometry'])

    if 'distance' not in link_attrs.keys():
        link_attrs['distance'] = [
            pyproj.Geod(ellps='WGS84').geometry_length(edge_geom)
            for edge_geom in link_attrs['geometry']
        ]

    graph.graph.add_edges(pairs, attributes=link_attrs)

    return graph


def _select_closest_k(gdf_vs_source, gdf_vs_target, dist_thresh,
                      bidir=False, k=5):
    """
    Parameters
    ----------

    Returns
    -------
    list, list
    """

    # crappy conversion of metres to degrees
    dist_thresh /= (ONE_LAT_KM*1000)

    # index matches, in format (#target vs, k). nans for those without matches
    __, ix_matches = _ckdnearest(gdf_vs_target, gdf_vs_source, k=k,
                                 dist_thresh=dist_thresh)
    # broadcast target indices to same format
    ix_matches = ix_matches.flatten()
    v_ids_target = np.array(np.broadcast_to(
        np.array([gdf_vs_target.id]).T, (len(gdf_vs_target), k)).flatten())
    v_ids_target = v_ids_target[~np.isnan(ix_matches)]
    v_ids_source = np.array(
        gdf_vs_source.loc[ix_matches[~np.isnan(ix_matches)]].id)

    if bidir:
        v_ids_target = np.append(v_ids_target, v_ids_source)
        v_ids_source = np.append(v_ids_source, v_ids_target)

    return list(v_ids_source), list(v_ids_target)


def _create_subgraph(graph, source_attrs, target_attrs, via_attrs):
    """
    Create a subgraph from the original graph. Includes only vertices and edges
    from source, target and via types.

    Parameters
    ----------
    graph : nw_base.Graph object
    source_attrs : dict
    target_attrs : dict
    via_attrs : dict
    link_attrs : dict


    Returns
    -------
    vs_keep : list
        vertex ids of original graph that is kept in subgraph

    subgraph : igraph.Graph
        induced subgraph of graph, given v_seq


    See also
    --------
    link_vertices_shortest_paths(), link_vertices_shortest_path()
    """

    # select only those for which specified attrs apply
    df_vs_source = _filter_vertices(graph.graph, source_attrs)
    df_vs_target = _filter_vertices(graph.graph, target_attrs)
    df_vs_via = _filter_vertices(graph.graph, via_attrs)

    vs_keep = np.concatenate((df_vs_source.index.values,
                              df_vs_target.index.values,
                              df_vs_via.index.values))

    # vs_keep has indexing of original graph, subgraph has new indexing. There
    # is no way of keeping track of the re-ordering, other than to have a named
    # attribute!
    graph.graph.vs['orig_id'] = range(len(graph.graph.vs))
    graph.graph.es['orig_id'] = range(len(graph.graph.es))
    subgraph = graph.graph.induced_subgraph(vs_keep)

    # delete remaining edges that have wrong attributes
    df_es_target = _filter_edges(graph.graph, target_attrs)
    df_es_source = _filter_edges(graph.graph, source_attrs)
    df_es_via = _filter_edges(graph.graph, via_attrs)

    correct_edges = np.concatenate((df_es_target.index.values,
                                   df_es_source.index.values,
                                   df_es_via.index.values))

    wrong_edges = set(range(len(subgraph.es))).difference(set(correct_edges))

    subgraph.delete_edges(wrong_edges)

    return subgraph


def _get_subgraph2graph_vsdict(graph, subgraph):
    """
    Keep track of which vertices in induced subgraph represent which vertices
    in original graph. dict[subgraph_vs_ind] = graph_vs_ind
    Goes via the named attribute 'orig_id' created before making the subgraph.

    Parameters
    ----------
    graph : igraph.Graph
    subgraph : igraph.Graph
        induced subgraph of graph

    Returns
    -------
    dict
        mapping from subgraph to graph indices.
    """
    subgraph_vs_indices = [subvx.index for subvx in subgraph.vs]
    subgraph_orig_ids = subgraph.vs.get_attribute_values('orig_id')
    df_subg = pd.DataFrame(
        subgraph_vs_indices, index=subgraph_orig_ids, columns=['index_sub'])

    graph_vs_indices = [vx.index for vx in graph.vs]
    graph_orig_ids = graph.vs.get_attribute_values('orig_id')
    df_g = pd.DataFrame(
        graph_vs_indices, index=graph_orig_ids,  columns=['index_g'])

    df_conc = pd.concat([df_subg, df_g], axis=1)

    return dict((k, v) for k, v in zip(df_conc['index_sub'], df_conc['index_g']))


# =============================================================================
# Propagation functions
# =============================================================================

def _propagate_check_fail(self, source, target, thresh_func):
    """
    propagate capacities from source vertices to target vertices
    on the subgraph via the adjacency matrix.
    check whether capacity enough.
    fail target if not.
    """
    v_seq = self.graph.vs.select(ci_type_in=[source, target])
    subgraph = self.graph.induced_subgraph(v_seq)
    try:
        adj_sub = subgraph.get_adjacency_sparse()
    except TypeError:
        # treats case where empty adjacency matrix!
        adj_sub = scipy.sparse.csr_matrix(subgraph.get_adjacency().data)
    # Hadamard product func_tot (*) capacity
    func_capa = np.multiply(v_seq['func_tot'],
                            v_seq[f'capacity_{source}_{target}'])
    # propagate capacities down from source --> target along adj
    capa_rec = scipy.sparse.csr_matrix(func_capa).dot(adj_sub)
    # functionality thesholds for recieved capacity

    func_thresh = np.array([thresh_func if vx['ci_type'] == target
                            else 0 for vx in v_seq])

    # boolean vector whether received capacity great enough to supply endusers
    capa_suff = (np.array(capa_rec.todense()).squeeze()
                 >= func_thresh).astype(int)

    # This is under the assumption that subgraph retains the same
    # relative ordering of vertices as in v_seq extracted from graph!
    # This further assumes that any operation on a VertexSeq equally modifies its graph.
    # Both should be the case, but the igraph doc is always a bit ambiguous
    if target == 'people':
        v_seq[f'actual_supply_{source}_{target}'] = capa_suff
    else:
        func_tot = np.minimum(capa_suff, v_seq['func_tot'])
        v_seq['func_tot'] = func_tot


def cascade(graph, df_dependencies, p_source='power_plant',
            p_sink='power_line', source_var='el_generation', demand_var='el_consumption',
            preselect='auto', initial=False, friction_surf=None, dur_thresh=None):
    """
    entire cascade wrapper for internal state update, functional dependency iterations,
    enduser dependency updates. CI-specific. Writing more generically does not
    work atm, as there are too many CI-specific functionality assumptions.
    """
    delta = -1
    cycles = 0
    while delta != 0:
        LOGGER.info(
            f'Updating functional states. Current delta: {delta}')
        func_states_vs, func_states_es = _funcstates_sum(graph)
        _update_internal_dependencies(
            p_source=p_source, p_sink=p_sink, source_var=source_var, demand_var=demand_var)

        self._update_functional_dependencies(df_dependencies)
        func_states_vs2, func_states_es2 = self._funcstates_sum()
        delta = max(abs(func_states_vs-func_states_vs2),
                    abs(func_states_es-func_states_es2))
        cycles += 1

    LOGGER.info('Ended functional state update.' +
                ' Proceeding to end-user update.')
    if (cycles > 1) or initial:
        self._update_enduser_dependencies(
            df_dependencies, preselect, friction_surf, dur_thresh)


def _funcstates_sum(graph):
    """
    return the total funcstate sum func_tot across all vertices and
    edges

    Parameters
    ----------
    graph : nw_base.Graph object

    Returns
    -------
    tuple (int, int) : sum of vertex func_tot, sum of edges func_tot
    """
    return (sum(graph.graph.vs.get_attribute_values('func_tot')),
            sum(graph.graph.es.get_attribute_values('func_tot')))


def _update_internal_dependencies(graph, attr_subtypes, p_source, p_sink, source_var,
                                  demand_var):
    """
    for ci-types with an internally networked structure (e.g. roads and
    power lines which consist in edges + nodes), update those ci networks
    internally
    """

    # specifically for roads: if edge is dysfunctional, render its target vertex dysfunctional
    if {'road'}.issubset(set(self.graph.vs['ci_type'])):
        LOGGER.info('Updating roads')
        targets_dys = [edge.target for edge in self.graph.es.select(
            ci_type='road').select(func_tot_eq=0)]
        self.graph.vs.select(targets_dys).select(
            ci_type='road')['func_tot'] = 0

    # specifically for powerlines: check power clusters
    if {p_source, p_sink}.issubset(set(self.graph.vs['ci_type'])):
        LOGGER.info('Updating power clusters')
        # For another version using pandapower, see nw_utils.py
        # Since powerlines are directed in a directed graph,
        # make sure 'reverse' lines are also down

        edges_dys = self.graph.es.select(ci_type='power_line'
                                         ).select(func_tot_eq=0)
        reverse_edges = [(edge.target, edge.source) for edge in edges_dys]
        eids = self.graph.get_eids(pairs=reverse_edges, path=None,
                                   directed=True, error=True)
        self.graph.es[eids]['func_tot'] = 0
        LOGGER.info(f"""Using updated power line algorithm: dysfunc edges before:
              {len(edges_dys)}, after: {len(self.graph.es.select(ci_type='power_line'
                                         ).select(func_tot_eq=0))}""")
        self.powercap_from_clusters(p_source=p_source, p_sink=p_sink,
                                    demand_ci='people', source_var=source_var, demand_var=demand_var)


def _update_functional_dependencies(self, df_dependencies):

    for __, row in df_dependencies[
            df_dependencies['type_I'] == 'functional'].iterrows():

        if row.access_cnstr:
            # TODO: Implement
            LOGGER.warning(
                'Road access condition for CI-CI deps not yet implemented')

        self._propagate_check_fail(row.source, row.target, row.thresh_func)


def _update_enduser_dependencies(self, df_dependencies, preselect,
                                 friction_surf, dur_thresh):

    for __, row in df_dependencies[
            df_dependencies['type_I'] == 'enduser'].iterrows():

        if row.access_cnstr:
            LOGGER.info(
                f'Re-calculating paths from {row.source} to {row.target}')
            if (row.source == 'road'):  # separate checking algorithm for road access
                self.graph.delete_edges(
                    ci_type=f'dependency_{row.source}_{row.target}')
                self.link_vertices_edgecond(row.source, row.target,
                                            link_name=f'dependency_{row.source}_{row.target}',
                                            edge_type='road')
            elif row.single_link:  # those need to be re-checked on their fixed s-t

                self.recheck_access(row.source, row.target, via_ci='road',
                                    friction_surf=friction_surf, dist_thresh=row.thresh_dist,
                                    dur_thresh=dur_thresh, criterion='distance',
                                    link_name=f'dependency_{row.source}_{row.target}',
                                    bidir=False)

            else:
                # the re-checking takes much longer than checking completely
                # from scratch, hence check from scratch.
                starttime = timeit.default_timer()
                self.graph.delete_edges(
                    ci_type=f'dependency_{row.source}_{row.target}')
                self.link_vertices_friction_surf(row.source, row.target, friction_surf,
                                                 link_name=f'dependency_{row.source}_{row.target}',
                                                 dist_thresh=dur_thresh*83.33,
                                                 k=5, dur_thresh=dur_thresh)
                print(f"Time for recalculating friction from {row.source} to {row.target} :", timeit.default_timer(
                ) - starttime)
                starttime = timeit.default_timer()
                self.link_vertices_shortest_paths(row.source, row.target, via_ci='road',
                                                  dist_thresh=row.thresh_dist, criterion='distance',
                                                  link_name=f'dependency_{row.source}_{row.target}',
                                                  bidir=False, preselect=preselect)
                print(f"Time for recalculating paths from {row.source} to {row.target} :", timeit.default_timer(
                ) - starttime)
        self._propagate_check_fail(row.source, row.target, row.thresh_func)


def recheck_access(self, source_ci, target_ci, via_ci, friction_surf,
                   dist_thresh, dur_thresh, criterion='distance',
                   link_name=None, bidir=False):
    """
    for links with access constraints, re-check those with functional
    sources where paths may however be broken now.
    Those with dysfunctional sources don't need to be checked, since
    dysfunctionality will anyways propagate to target later.
    """
    es_check = self.graph.es.select(
        ci_type=f'dependency_{source_ci}_{target_ci}')

    bools_check = [self.graph.vs[edge.source]['func_tot'] > 0
                   for edge in es_check]

    es_check = [edge for edge, bool_check in zip(es_check, bools_check)
                if bool_check]

    if len(es_check) > 0:

        edge_geoms = [edge['geometry'] for edge in es_check]
        v_ids_target = [edge.target for edge in es_check]
        v_ids_source = [edge.source for edge in es_check]
        v_ids_via = [vs.index for vs in
                     self.graph.vs.select(ci_type=f'{via_ci}')]

        # first check friction
        friction = self._calc_friction(edge_geoms, friction_surf)
        bool_keep = friction < dur_thresh

        # then check shortest paths
        v_seq = self.graph.vs(list(np.unique([*v_ids_target, *v_ids_source,
                                              *v_ids_via])))
        subgraph = self.graph.induced_subgraph(v_seq)
        subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(v_seq)
        graph_subgraph_vsdict = {v: k for k,
                                 v in subgraph_graph_vsdict.items()}
        subgraph.delete_edges(subgraph.es.select(func_tot_lt=1))
        wrong_edges = set(subgraph.es['ci_type']).difference(
            {via_ci})
        subgraph.delete_edges(subgraph.es.select(ci_type_in=wrong_edges))

        for ix, source, target, bool_f in (zip(np.arange(len(bool_keep)),
                                               v_ids_source, v_ids_target,
                                               bool_keep)):
            if not bool_f:
                dist = subgraph.shortest_paths(
                    source=graph_subgraph_vsdict[source],
                    target=graph_subgraph_vsdict[target],
                    weights='distance')
                if dist[0][0] < dist_thresh:
                    bool_keep[ix] = True
                    es_check[ix]['distance'] = dist[0][0]
        self.graph.delete_edges([edge.index for edge, bool_f in
                                 zip(es_check, bool_keep)
                                 if not bool_f])


def powercap_from_clusters(self, p_source, p_sink, demand_ci, source_var,
                           demand_var):

    capacity_vars = [var for var in self.graph.vs.attributes()
                     if f'capacity_{p_sink}_' in var]
    power_vs = self.graph.vs.select(
        ci_type_in=['power_line', p_source, p_sink, demand_ci])
    # make subgraph spanning all nodes, but only functional edges
    # Subgraph operations do not modify original graph.
    power_subgraph = self.graph.induced_subgraph(power_vs)
    power_subgraph.delete_edges(func_tot_lt=0.1)

    subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(power_vs)

    for cluster in power_subgraph.clusters(mode='weak'):

        sources = power_subgraph.vs[cluster].select(ci_type=p_source)
        sinks = power_subgraph.vs[cluster].select(ci_type=p_sink)
        demands = power_subgraph.vs[cluster].select(ci_type=demand_ci)

        psupply = sum([source[source_var]*source['func_tot']
                       for source in sources])
        pdemand = sum([demand[demand_var] for demand in demands])

        try:
            sd_ratio = min(1, psupply/pdemand)
        except ZeroDivisionError:
            sd_ratio = 1

        for var in capacity_vars:
            self.graph.vs[
                [subgraph_graph_vsdict[sink.index] for sink in sinks]
            ].set_attribute_values(var, sd_ratio)
