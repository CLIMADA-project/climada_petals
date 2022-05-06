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
import geopandas as gpd
import pandas as pd
import pyproj
import shapely
from scipy.spatial import cKDTree
from tqdm import tqdm


from climada_petals.engine.networks.base import Network
from climada_petals.engine.networks.nw_flows import PowerCluster
from climada.util.constants import ONE_LAT_KM

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')


class GraphCalcs():

    def __init__(self, graph):
        """
        graph : instance of igraph.Graph
        """
        self.graph = graph

    def _edges_from_vlists(self, v_ids_source, v_ids_target, link_name=None,
                           lengths=None):
        """
        add edges to graph given source and target vertex lists
        adds geometries, edge lengths, edge names and func states as attributes
        """
        pairs = list(zip(v_ids_source, v_ids_target))

        edge_geoms = make_edge_geometries(
            self.graph.vs[v_ids_source]['geometry'],
            self.graph.vs[v_ids_target]['geometry'])

        if lengths is None:
            lengths = [pyproj.Geod(ellps='WGS84').geometry_length(edge_geom) for
                     edge_geom in edge_geoms]

        self.graph.add_edges(pairs, attributes=
                             {'geometry' : edge_geoms, 'ci_type' : [link_name],
                              'distance' : lengths, 'func_internal' : 1,
                              'func_tot' : 1, 'imp_dir' : 0})

    def link_clusters(self):
        """
        link nodes from different clusters to their nearest nodes in other
        clusters to generate one connected graph.
        """

        gdf_vs = self.graph.get_vertex_dataframe()
        gdf_vs['membership'] = self.graph.clusters().membership

        source_ix = []
        target_ix = []

        for member in range(len(self.graph.clusters())):
            gdf_a = gdf_vs[gdf_vs['membership']==member]
            gdf_b = gdf_vs[gdf_vs['membership']!=member]
            dists, ix_match = _ckdnearest(gdf_a, gdf_b)

            source_ix.append(
                gdf_a.iloc[np.where(dists==min(dists))[0]].name.values[0])
            target_ix.append(
                gdf_b.loc[ix_match[np.where(dists==min(dists))[0]]].name.values[0])

        link_name = gdf_vs.ci_type[0]

        self._edges_from_vlists(source_ix, target_ix, link_name)


    def link_vertices_closest_k(self, source_ci, target_ci, link_name=None,
                                dist_thresh=None, bidir=False, k=5):
        """
        find k nearest source_ci vertices for each target_ci vertex,
        given distance constraints
        """
        gdf_vs = self.graph.get_vertex_dataframe()
        gdf_vs_target = gdf_vs[gdf_vs.ci_type==target_ci]
        gdf_vs_source = gdf_vs[gdf_vs.ci_type==source_ci]

        # shape: (#target vs, k)
        dists, ix_matches = _ckdnearest(gdf_vs_target, gdf_vs_source, k=k)

        if dist_thresh is not None:
            # conversion from degrees to m holds only vaguely
            dists_bool = dists.flatten() < (dist_thresh/(ONE_LAT_KM*1000))
        else:
            dists_bool = [True]*len(dists.flatten())

        # broadcast target indices to same format, select based on distance
        # name and vertex ids are the same. also same in gdf_vs
        v_ids_target = list(np.broadcast_to(
            np.array([gdf_vs_target.name]).T,
            (len(gdf_vs_target),k)).flatten()[dists_bool])
        v_ids_source = list(gdf_vs_source.loc[ix_matches.flatten()
                                              ].name[dists_bool])

        if bidir:
            v_ids_target.extend(v_ids_source)
            v_ids_source.extend(v_ids_target)

        if not link_name:
            link_name = f'dependency_{source_ci}_{target_ci}'

        self._edges_from_vlists(v_ids_source, v_ids_target, link_name)


    def link_vertices_shortest_paths(self, source_ci, target_ci, via_ci,
                                    dist_thresh=10e6, criterion='distance',
                                    link_name=None, bidir=False):
        """
        make all links below certain dist_thresh along shortest paths length
        between all possible sources & targets --> doesnt matter whether search
        is done from source to all targets or from target to all sources

        """

        v_seq = self.graph.vs.select(ci_type_in=[source_ci, target_ci, via_ci])
        subgraph = self.graph.induced_subgraph(v_seq)
        subgraph.delete_edges(subgraph.es.select(func_tot_lt=1))

        subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(v_seq)

        vs_source = subgraph.vs.select(ci_type=source_ci)
        vs_target = subgraph.vs.select(ci_type=target_ci)

        # metres to degree conversion assumption is imprecise
        ix_matches = _preselect_destinations(
            vs_target, vs_source, dist_thresh/(ONE_LAT_KM*1000))

        v_ids_target = []
        v_ids_source = []
        lengths = []

        for ix_match, v_target in tqdm(
                zip(ix_matches, vs_target), desc=f'Paths from {source_ci} to {target_ci}',
                total=len(vs_target)):
            path_dists = subgraph.shortest_paths(
                source=vs_source[ix_match], target=v_target, weights=criterion)

            if len(path_dists)>0:
                path_dists = np.hstack(path_dists)
                bool_link = path_dists < dist_thresh
                v_ids_target.extend(
                    [subgraph_graph_vsdict[v_target.index]]*sum(bool_link))
                v_ids_source.extend([subgraph_graph_vsdict[vs_source[ix].index]
                                     for ix in np.array(ix_match)[bool_link]])
                lengths.extend(path_dists[bool_link])

        if bidir:
            v_ids_target.extend(v_ids_source)
            v_ids_source.extend(v_ids_target)
            lengths.extend(lengths)

        if not link_name:
            link_name = f'dependency_{source_ci}_{target_ci}'

        self._edges_from_vlists(v_ids_source, v_ids_target,
                                link_name=link_name,
                                lengths=lengths)

    def link_vertices_shortest_path(self, source_ci, target_ci, via_ci,
                                    dist_thresh=10e6, criterion='distance',
                                    link_name=None, bidir=False):
        """
        Per target, choose single shortest path to source which is
        below dist_thresh.

        TODO: reverse the order and
        """

        v_seq = self.graph.vs.select(ci_type_in=[source_ci, target_ci, via_ci])
        subgraph = self.graph.induced_subgraph(v_seq)
        subgraph.delete_edges(subgraph.es.select(func_tot_lt=1))

        subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(v_seq)

        vs_source = subgraph.vs.select(ci_type=source_ci)
        vs_target = subgraph.vs.select(ci_type=target_ci)

        # metres to degree conversion assumption is imprecise
        ix_matches = _preselect_destinations(
            vs_target, vs_source, dist_thresh/(ONE_LAT_KM*1000))

        v_ids_target = []
        v_ids_source = []
        lengths = []

        for ix_match, v_target in tqdm(
                zip(ix_matches, vs_target), desc=f'Paths from {source_ci} to {target_ci}',
                total=len(vs_target)):
            path_dists = subgraph.shortest_paths(
                source=vs_source[ix_match], target=v_target, weights=criterion)

            if len(path_dists)>0:
                path_dists = np.hstack(path_dists)

                if min(path_dists) < dist_thresh:
                    v_ids_target.append(subgraph_graph_vsdict[v_target.index])
                    ix = ix_match[int(np.where(path_dists==min(path_dists))[0])]
                    v_ids_source.append(subgraph_graph_vsdict[vs_source[ix].index])
                    lengths.append(min(path_dists))

        if bidir:
            v_ids_target.extend(v_ids_source)
            v_ids_source.extend(v_ids_target)
            lengths.extend(lengths)

        if not link_name:
            link_name = f'dependency_{source_ci}_{target_ci}'

        self._edges_from_vlists(v_ids_source, v_ids_target,
                                link_name=link_name,
                                lengths=lengths)

    def place_dependency(self, source, target, single_link=True,
                         access_cnstr=False, dist_thresh=None):
        """
        source : supporting infra
        target : dependent infra / ppl
        single_link : bool
            Whether there is (max.) a single link allowable between a target
            vertice and a source vertice or whether target vertices can have several
            dependencies of the same type.
            Whether the link cis unchangeable between the specific S & T or
            if it could be re-routable upon invalidity at a later stage.
        access_cnstr : bool
            Whether the link requires physical (road)-access to be possible
            between S & T or a direct (logical) connection can be made.
        dist_thresh : float
            Whether there is a maximum link length allowed for a link length
            to be established (in metres!). Default is None.

        """
        LOGGER.info(f'Placing dependency between {source} and {target}')
        dep_name = f'dependency_{source}_{target}'

        # make links
        if not access_cnstr:
            if single_link:
                self.link_vertices_closest_k(source, target, link_name=dep_name,
                                           dist_thresh=dist_thresh, k=1)
            else:
                self.link_vertices_closest_k(source, target, link_name=dep_name,
                                             dist_thresh=dist_thresh, k=5)
        else:
            if single_link:
                self.link_vertices_shortest_path(source, target, via_ci='road',
                                    dist_thresh=dist_thresh, criterion='distance',
                                    link_name=dep_name)
            else:
                self.link_vertices_shortest_paths(source, target, via_ci='road',
                                    dist_thresh=dist_thresh, criterion='distance',
                                    link_name=dep_name)


    def _funcstates_sum(self):
        """
        return the total funcstate sum func_tot across all vertices and
        edges

        Returns
        -------
        tuple (int, int) : sum of vertex func_tot, sum of edges func_tot
        """
        return (sum(self.graph.vs.get_attribute_values('func_tot')),
                sum(self.graph.es.get_attribute_values('func_tot')))

    def _update_internal_dependencies(self, p_source, p_sink, per_cap_cons,
                                      source_var='el_gen_mw'):

        # if an edge is dysfunctional, render its target vertex dysfunctional
        ci_types_nw = (set(self.graph.vs['ci_type']) &
                       set(self.graph.es['ci_type']))
        LOGGER.info(f'Checking networked ci-types {ci_types_nw}')
        targets_dys = [edge.target for edge in self.graph.es.select(
            ci_type_in=ci_types_nw).select(func_tot_eq=0)]
        self.graph.vs.select(targets_dys)['func_tot'] = 0

        # specifically check power clusters
        if {p_source, p_sink}.issubset(set(self.graph.vs['ci_type'])):
            LOGGER.info('Updating power clusters')
            # TODO: This is poorly hard-coded and poorly assigned.
            self = PowerCluster.set_capacity_from_sd_ratio(
                    self, per_cap_cons=per_cap_cons, source_ci=p_source,
                    sink_ci=p_sink, source_var=source_var)



    def _update_functional_dependencies(self, df_dependencies):

        for __, row in df_dependencies[
                df_dependencies['type_I']=='functional'].iterrows():

            if row.access_cnstr:
                # TODO: Implement
                LOGGER.warning('Road access condition for CI-CI deps not yet implemented')

            v_seq = self.graph.vs.select(ci_type_in=[row.source, row.target])
            subgraph = self.graph.induced_subgraph(v_seq)
            adj_sub = np.array(subgraph.get_adjacency().data)
            # Hadamard product func_tot (*) capacity
            func_capa = np.multiply(v_seq['func_tot'],
                                    v_seq[f'capacity_{row.source}_{row.target}'])
            # propagate capacities down from source --> target along adj
            capa_rec = func_capa.dot(adj_sub)
            # functionality thesholds for recieved capacity
            func_thresh = np.array([row.thresh_func if vx['ci_type'] == row.target
                                    else -999 for vx in v_seq])
            # boolean vector whether received capacity great enough to supply endusers
            capa_suff = (capa_rec >= func_thresh).astype(int)
            func_tot = np.minimum(capa_suff, v_seq['func_tot'])

            # TODO: This is under the assumption that subgraph retains the same
            # relative ordering of vertices as in v_seq extracted from graph!!
            # This further assumes that any operation on a VertexSeq equally modifies its graph.
            # Both should be the case, but igraph doc. is always a bit ambiguous
            v_seq['func_tot'] = func_tot


    def _update_enduser_dependencies(self, df_dependencies):

        for __, row in df_dependencies[
                df_dependencies['type_I']=='enduser'].iterrows():

            if row.access_cnstr:
                # the re-checking takes much longer than checking completely
                # from scratch, hence check from scratch.
                LOGGER.info(f'Re-calculating paths from {row.source} to {row.target}')
                self.graph.delete_edges(ci_type=f'dependency_{row.source}_{row.target}')
                if row.single_link:
                    self.link_vertices_shortest_path(row.source, row.target, via_ci='road',
                                    dist_thresh=row.thresh_dist, criterion='distance',
                                    link_name=f'dependency_{row.source}_{row.target}')
                else:
                    self.link_vertices_shortest_paths(row.source, row.target, via_ci='road',
                                    dist_thresh=row.thresh_dist, criterion='distance',
                                    link_name=f'dependency_{row.source}_{row.target}')

            v_seq = self.graph.vs.select(ci_type_in=[row.source, row.target])
            subgraph = self.graph.induced_subgraph(v_seq)
            adj_sub = np.array(subgraph.get_adjacency().data)
            # Hadamard product func_tot (*) capacity
            func_capa = np.multiply(v_seq['func_tot'],
                                    v_seq[f'capacity_{row.source}_{row.target}'])
            # propagate capacities down from source --> target along adj
            capa_rec = func_capa.dot(adj_sub)
            # functionality thesholds for recieved capacity
            func_thresh = np.array([row.thresh_func if vx['ci_type'] == row.target
                                    else 0 for vx in v_seq])
            # boolean vector whether received capacity great enough to supply endusers
            capa_suff = (capa_rec >= func_thresh).astype(int)

            # TODO: This is under the assumption that subgraph retains the same
            # relative ordering of vertices as in v_seq extracted from graph!!
            # This further assumes that any operation on a VertexSeq equally modifies its graph.
            # Both should be the case, but igraph doc. is always a bit ambiguous
            v_seq[f'actual_supply_{row.source}_{row.target}'] = capa_suff


    def _get_subgraph2graph_vsdict(self, vertex_seq):
        return dict((k,v) for k, v in zip(
            [subvx.index for subvx in self.graph.subgraph(vertex_seq).vs],
            [vx.index for vx in vertex_seq]))

    def cascade(self, df_dependencies, p_source='power plant',
                p_sink='substation', per_cap_cons=0.000079,
                source_var='el_gen_mw'):
        """
        wrapper for internal state update, functional dependency iterations,
        enduser dependency updates
        """
        delta = -1
        cycles = 0
        while delta != 0:
            cycles+=1
            LOGGER.info('Updating functional states.'+
                        f' Current func.-state delta : {delta}')
            func_states_vs, func_states_es = self._funcstates_sum()
            self._update_internal_dependencies(p_source=p_source,
                                        p_sink=p_sink, per_cap_cons=per_cap_cons,
                                        source_var=source_var)
            self._update_functional_dependencies(df_dependencies)
            func_states_vs2, func_states_es2 = self._funcstates_sum()
            delta = max(abs(func_states_vs-func_states_vs2),
                        abs(func_states_es-func_states_es2))

        LOGGER.info('Ended functional state update.' +
                    ' Proceeding to end-user update.')
        if cycles > 1:
            self._update_enduser_dependencies(df_dependencies)


    def graph_style(self, ecol='red', vcol='red', vsize=2, ewidth=3):
        visual_style = {}
        visual_style["edge_color"] = ecol
        visual_style["vertex_color"] = vcol
        visual_style["vertex_size"] = vsize
        visual_style["edge_width"] = ewidth
        visual_style["layout"] = self.graph.layout("fruchterman_reingold")
        visual_style["edge_arrow_size"] = 0
        visual_style["edge_curved"] = 1

        return visual_style

    def plot_graph(self, *kwargs):
        return ig.plot(self.graph, **self.graph_style(*kwargs))

    def plot_multigraph(self,layer_dict,layout):
        visual_style = {}
        visual_style["vertex_size"] = [layer_dict['vsize'][attr] for attr in
                                       self.graph.vs["ci_type"]]
        visual_style["edge_arrow_size"] = 1
        visual_style["edge_color"] = [layer_dict['edge_col'][attr] for attr in
                                      self.graph.vs["ci_type"]]
        visual_style["vertex_color"] = [layer_dict['vertex_col'][attr] for attr
                                        in self.graph.vs["ci_type"]]
        if layout == "fruchterman_reingold":
            visual_style["layout"] = self.graph.layout("fruchterman_reingold")
        elif layout == 'sugiyama':
            visual_style["layout"] = self.graph.layout_sugiyama(
                layers=[layer_dict['layers'][attr] for attr in
                        self.graph.vs["ci_type"]])
        visual_style["edge_curved"] = 0.2
        visual_style["edge_width"] = 1

        return ig.plot(self.graph, **visual_style)


    def return_network(self):
        return Network.from_graphs([self.graph])


class Graph(GraphCalcs):
    """
    create an igraph Graph from network components
    """

    def __init__(self, network, directed=False):
        """
        network : instance of networks.base.Network"""
        if network.edges is not None:
            self.graph = self.graph_from_es(
                gdf_edges=network.edges, gdf_nodes=network.nodes,
                directed=directed)
        else:
            self.graph = self.graph_from_vs(
                gdf_nodes=network.nodes, directed=directed)

    @staticmethod
    def graph_from_es(gdf_edges, gdf_nodes=None, directed=False):
        return ig.Graph.DataFrame(
            gdf_edges, vertices=gdf_nodes, directed=directed)

    @staticmethod
    def graph_from_vs(gdf_nodes, directed=False):
        vertex_attrs = gdf_nodes.to_dict('list')
        return ig.Graph(
            n=len(gdf_nodes),vertex_attrs=vertex_attrs, directed=directed)


# =============================================================================
# Spatial analysis util functions
# =============================================================================

def make_edge_geometries(vs_geoms_from, vs_geoms_to):
    """
    create straight shapely LineString geometries between lists of
    from and to nodes, to be added to newly created edges as attributes
    """
    return [shapely.geometry.LineString([geom_from, geom_to]) for
                                         geom_from, geom_to in
                                        zip(vs_geoms_from, vs_geoms_to)]

def _preselect_destinations(vs_assign, vs_base, dist_thresh):
    points_base = np.array([(x.x, x.y) for x in vs_base['geometry']])
    point_tree = cKDTree(points_base)

    points_assign = np.array([(x.x, x.y) for x in vs_assign['geometry']])
    ix_matches = []
    for assign_loc in points_assign:
        ix_matches.append(point_tree.query_ball_point(assign_loc, dist_thresh))
    return ix_matches


def _ckdnearest(vs_assign, gdf_base, k=1):
    """
    see https://gis.stackexchange.com/a/301935

    Parameters
    ----------
    vs_assign : gpd.GeoDataFrame or Point

    gdf_base : gpd.GeoDataFrame

    Returns
    ----------

    """
    # TODO: this should be a util function
    # TODO: this mixed input options (1 vertex vs gdf) is not nicely solved
    if isinstance(vs_assign, (gpd.GeoDataFrame, pd.DataFrame)):
        n_assign = np.array(list(vs_assign.geometry.apply(lambda x: (x.x, x.y))))
    else:
        n_assign = np.array([(vs_assign.geometry.x, vs_assign.geometry.y)])
    n_base = np.array(list(gdf_base.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(n_base)
    dist, idx = btree.query(n_assign, k=k)
    return dist, np.array(gdf_base.iloc[idx.flatten()].index).reshape(dist.shape)

# =============================================================================
# General results analysis util functions
# =============================================================================

def service_dict():
    return {'power':'actual_supply_power line_people',
                    'healthcare': 'actual_supply_health_people',
                    'education':'actual_supply_education_people',
                    'telecom' : 'actual_supply_celltower_people',
                    'mobility' : 'actual_supply_road_people',
                    'water' : 'actual_supply_wastewater_people'}


def number_noservice(service, graph):

    no_service = (1-np.array(graph.graph.vs.select(
        ci_type='people')[service_dict()[service]]))
    pop = np.array(graph.graph.vs.select(
        ci_type='people')['counts'])

    return (no_service*pop).sum()

def number_noservices(graph,
                         services=['power', 'healthcare', 'education', 'telecom', 'mobility', 'water']):

    servstats_dict = {}
    for service in services:
        servstats_dict[service] = number_noservice(service, graph)
    return servstats_dict


def disaster_impact_service_geoseries(service, pre_graph, post_graph):

    no_service_post = (1-np.array(post_graph.graph.vs.select(
        ci_type='people')[service_dict()[service]]))
    no_service_pre = (1-np.array(pre_graph.graph.vs.select(
        ci_type='people')[service_dict()[service]]))

    geom = np.array(post_graph.graph.vs.select(
        ci_type='people')['geom_wkt'])

    return gpd.GeoSeries.from_wkt(
        geom[np.where((no_service_post-no_service_pre)>0)])

def disaster_impact_service(service, pre_graph, post_graph):

    no_service_post = (1-np.array(post_graph.graph.vs.select(
        ci_type='people')[service_dict()[service]]))
    no_service_pre = (1-np.array(pre_graph.graph.vs.select(
        ci_type='people')[service_dict()[service]]))
    pop = np.array(pre_graph.graph.vs.select(
        ci_type='people')['counts'])

    return ((no_service_post-no_service_pre)*pop).sum()

def disaster_impact_allservices(pre_graph, post_graph,
                services=['power', 'healthcare', 'education', 'telecom', 'mobility', 'water']):

    imp_dict = {}

    for service in services:
        imp_dict[service] = disaster_impact_service(
            service, pre_graph, post_graph)

    return imp_dict


def get_graphstats(graph):
    from collections import Counter
    stats_dict = {}
    stats_dict['no_edges'] = len(graph.graph.es)
    stats_dict['no_nodes'] = len(graph.graph.vs)
    stats_dict['edge_types'] = Counter(graph.graph.es['ci_type'])
    stats_dict['node_types'] = Counter(graph.graph.vs['ci_type'])
    return stats_dict
