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
import pyproj
from tqdm import tqdm
import scipy

from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_utils import (make_edge_geometries,
                                                     _ckdnearest,
                                                     _preselect_destinations)
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
                                    link_name=None, bidir=False, preselect='auto'):
        """
        make all links below certain dist_thresh along shortest paths length
        between all possible sources & targets --> doesnt matter whether search
        is done from source to all targets or from target to all sources
        
        
        Parameters
        ----------
        preselect : str, bool
            Whether to do a target - source preselection for shortest path
            search. If False, does an all-to-all path search and selects in 
            hindsight based on distance. If True, pre-selects per target some
            potential candidates, but loops individually through each target.
            True recommended for large road networks (>100k edges).
            Default is auto - algorithm based on # edges.
        """
        print('running updated algorithm')
        v_seq = self.graph.vs.select(
            ci_type_in=[source_ci, target_ci, via_ci]).select(func_tot_gt=0)
        subgraph = self.graph.induced_subgraph(v_seq)
        subgraph.delete_edges(subgraph.es.select(func_tot_lt=1))
        wrong_edges = set(subgraph.es['ci_type']).difference({via_ci})
        subgraph.delete_edges(subgraph.es.select(ci_type_in=wrong_edges))

        subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(v_seq)

        vs_source = subgraph.vs.select(ci_type=source_ci)
        vs_target = subgraph.vs.select(ci_type=target_ci)

        v_ids_target = []
        v_ids_source = []
        lengths = []

        if preselect == 'auto':
            preselect = True if len(subgraph.es)>200000 else False
        
        if preselect:
            # metres to degree conversion assumption is imprecise
            ix_matches = _preselect_destinations(
                vs_target, vs_source, dist_thresh/(ONE_LAT_KM*1000))

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
        else:
            path_dists = subgraph.shortest_paths(
                source=vs_source, target=vs_target, weights='distance')
            path_dists = np.array(path_dists) # dim: (#sources, #targets)
            
            ix_source, ix_target = np.where(path_dists<dist_thresh)
            v_ids_source = [subgraph_graph_vsdict[vs.index] 
                            for vs in vs_source[list(ix_source)]]
            v_ids_target = [subgraph_graph_vsdict[vs.index] 
                            for vs in vs_target[list(ix_target)]]
            lengths = path_dists[(ix_source, ix_target)]

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
                                    link_name=None, bidir=False, preselect='auto'):
        """
        Per target, choose single shortest path to source which is
        below dist_thresh.
        """

        v_seq = self.graph.vs.select(
            ci_type_in=[source_ci, target_ci, via_ci]).select(func_tot_gt=0)
        subgraph = self.graph.induced_subgraph(v_seq)
        subgraph.delete_edges(subgraph.es.select(func_tot_lt=1))
        wrong_edges = set(subgraph.es['ci_type']).difference({via_ci})
        subgraph.delete_edges(subgraph.es.select(ci_type_in=wrong_edges))

        subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(v_seq)

        vs_source = subgraph.vs.select(ci_type=source_ci)
        vs_target = subgraph.vs.select(ci_type=target_ci)

        v_ids_target = []
        v_ids_source = []
        lengths = []
        
        if preselect == 'auto':
            preselect = True if len(subgraph.es)>200000 else False
            
        if preselect:
            # metres to degree conversion assumption is imprecise
            ix_matches = _preselect_destinations(
                vs_target, vs_source, dist_thresh/(ONE_LAT_KM*1000))
            
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
        else:
           path_dists = subgraph.shortest_paths(
               source=vs_source, target=vs_target, weights='distance')
           path_dists = np.array(path_dists) # dim: (#sources, #targets)           
           ix_source, ix_target = np.where(((path_dists == path_dists.min(axis=0)) &
                                           (path_dists<=dist_thresh))) # min dist. per target
           v_ids_source = [subgraph_graph_vsdict[vs.index] 
                           for vs in vs_source[list(ix_source)]]
           v_ids_target = [subgraph_graph_vsdict[vs.index] 
                           for vs in vs_target[list(ix_target)]]
           lengths = path_dists[(ix_source, ix_target)]

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
                         access_cnstr=False, dist_thresh=None, preselect='auto'):
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
                                    link_name=dep_name, preselect=preselect)
            else:
                self.link_vertices_shortest_paths(source, target, via_ci='road',
                                    dist_thresh=dist_thresh, criterion='distance',
                                    link_name=dep_name, preselect=preselect)


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

    def _update_internal_dependencies(self, p_source, p_sink, source_var,
                                      demand_var):

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
            # TODO: This is poorly structured. For another version using pandapower, see nw_flows.py
            self.powercap_from_clusters(p_source=p_source, p_sink=p_sink,
                demand_ci='people', source_var=source_var, demand_var=demand_var)


    def powercap_from_clusters(self, p_source='power plant',
                                   p_sink='substation', demand_ci='people',
                                   source_var='el_generation', demand_var='el_consumption'):

        capacity_vars = [var for var in self.graph.vs.attributes()
                         if f'capacity_{p_sink}_' in var]
        power_vs = self.graph.vs.select(
            ci_type_in=['power line', p_source, p_sink, demand_ci])
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


    def _update_functional_dependencies(self, df_dependencies):

        for __, row in df_dependencies[
                df_dependencies['type_I']=='functional'].iterrows():

            if row.access_cnstr:
                # TODO: Implement
                LOGGER.warning('Road access condition for CI-CI deps not yet implemented')
            
            v_seq = self.graph.vs.select(ci_type_in=[row.source, row.target])
            subgraph = self.graph.induced_subgraph(v_seq)
            adj_sub = subgraph.get_adjacency_sparse()
            # Hadamard product func_tot (*) capacity
            func_capa = np.multiply(v_seq['func_tot'],
                                    v_seq[f'capacity_{row.source}_{row.target}'])
            # propagate capacities down from source --> target along adj
            capa_rec = scipy.sparse.csr_matrix(func_capa).dot(adj_sub)
            # functionality thesholds for recieved capacity
            func_thresh = np.array([row.thresh_func if vx['ci_type'] == row.target
                                    else -999 for vx in v_seq])
            # boolean vector whether received capacity great enough to supply endusers
            capa_suff = (np.array(capa_rec.todense()).squeeze()>=func_thresh).astype(int)
            func_tot = np.minimum(capa_suff, v_seq['func_tot'])

            # TODO: This is under the assumption that subgraph retains the same
            # relative ordering of vertices as in v_seq extracted from graph!!
            # This further assumes that any operation on a VertexSeq equally modifies its graph.
            # Both should be the case, but igraph doc. is always a bit ambiguous
            v_seq['func_tot'] = func_tot


    def _update_enduser_dependencies(self, df_dependencies, preselect):

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
                                    link_name=f'dependency_{row.source}_{row.target}',
                                    bidir=False, preselect=preselect)
                else:
                    self.link_vertices_shortest_paths(row.source, row.target, via_ci='road',
                                    dist_thresh=row.thresh_dist, criterion='distance',
                                    link_name=f'dependency_{row.source}_{row.target}',
                                    bidir=False, preselect=preselect)

            v_seq = self.graph.vs.select(ci_type_in=[row.source, row.target])
            subgraph = self.graph.induced_subgraph(v_seq)
            adj_sub = subgraph.get_adjacency_sparse()
            # Hadamard product func_tot (*) capacity
            func_capa = np.multiply(v_seq['func_tot'],
                                    v_seq[f'capacity_{row.source}_{row.target}'])
            # propagate capacities down from source --> target along adj
            capa_rec = scipy.sparse.csr_matrix(func_capa).dot(adj_sub)
            # functionality thesholds for recieved capacity
            func_thresh = np.array([row.thresh_func if vx['ci_type'] == row.target
                                    else 0 for vx in v_seq])
            # boolean vector whether received capacity great enough to supply endusers
            capa_suff = (np.array(capa_rec.todense()).squeeze()>=func_thresh).astype(int)

            # TODO: This is under the assumption that subgraph retains the same
            # relative ordering of vertices as in v_seq extracted from graph!!
            # This further assumes that any operation on a VertexSeq equally modifies its graph.
            # Both should be the case, but igraph doc. is always a bit ambiguous
            v_seq[f'actual_supply_{row.source}_{row.target}'] = capa_suff


    def _get_subgraph2graph_vsdict(self, vertex_seq):
        return dict((k,v) for k, v in zip(
            [subvx.index for subvx in self.graph.subgraph(vertex_seq).vs],
            [vx.index for vx in vertex_seq]))

    def cascade(self, df_dependencies, p_source='power_plant',
                p_sink='power_line', source_var='el_generation', demand_var='el_consumption', 
                preselect='auto'):
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
                                        p_sink=p_sink, source_var=source_var,
                                        demand_var=demand_var)
            self._update_functional_dependencies(df_dependencies)
            func_states_vs2, func_states_es2 = self._funcstates_sum()
            delta = max(abs(func_states_vs-func_states_vs2),
                        abs(func_states_es-func_states_es2))

        LOGGER.info('Ended functional state update.' +
                    ' Proceeding to end-user update.')
        if cycles > 1:
            self._update_enduser_dependencies(df_dependencies, preselect)


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

