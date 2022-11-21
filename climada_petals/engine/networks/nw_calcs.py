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

    def link_clusters(self, dist_thresh=np.inf):
        """
        link nodes from different clusters to their nearest nodes in other
        clusters to generate one connected graph.
        """

        gdf_vs = self.graph.get_vertex_dataframe()
        gdf_vs['membership'] = self.graph.clusters().membership

        source_ix = []
        target_ix = []

        # very rough conversion from metres to degrees
        dist_thresh = dist_thresh/(ONE_LAT_KM*1000)

        for member in range(len(self.graph.clusters())):
            gdf_a = gdf_vs[gdf_vs['membership']==member]
            gdf_b = gdf_vs[gdf_vs['membership']!=member]
            try:
                dists, ix_match = _ckdnearest(gdf_a, gdf_b, dist_thresh=dist_thresh)
                source = gdf_a.iloc[np.where(dists==min(dists))[0]].name.values[0]
                target = gdf_b.loc[ix_match[np.where(dists==min(dists))[0]]].name.values[0]
                source_ix.append(source)
                target_ix.append(target)
                link_name = gdf_vs.ci_type[0]
            except (IndexError, KeyError):
                # if no match within given distance
                continue

        if len(source_ix)>0:
            self._edges_from_vlists(source_ix, target_ix, link_name)


    def _select_closest_k(self, gdf_vs_source, gdf_vs_target, dist_thresh=None, 
                          bidir=False, k=5):
        
        # crappy conversion of metres to degrees
        dist_thresh/=(ONE_LAT_KM*1000)
        
        # index matches, in format (#target vs, k). nans for those without matches
        __, ix_matches = _ckdnearest(gdf_vs_target, gdf_vs_source, k=k,
                                     dist_thresh=dist_thresh)
        # broadcast target indices to same format
        ix_matches = ix_matches.flatten()
        v_ids_target = np.array(
            np.broadcast_to(np.array([gdf_vs_target.name]).T,
                            (len(gdf_vs_target),k)
                            ).flatten())
        v_ids_target = v_ids_target[~np.isnan(ix_matches)]
        v_ids_source = np.array(
            gdf_vs_source.loc[ix_matches[~np.isnan(ix_matches)]].name)

        if bidir:
            v_ids_target = np.append(v_ids_target,v_ids_source)
            v_ids_source = np.append(v_ids_source,v_ids_target)
            
        return list(v_ids_source), list(v_ids_target)
    
    
    def link_vertices_closest_k_cond(self, source_ci, target_ci, cond, 
                                     link_name=None, dist_thresh=None, 
                                     bidir=False, k=5):
        """
        find k nearest source_ci vertices for each target_ci vertex,
        given distance constraints
        """
        gdf_vs = self.graph.get_vertex_dataframe()
        gdf_vs_target = gdf_vs[(gdf_vs.ci_type==target_ci) &
                               (gdf_vs[cond[0]]==cond[1])]
        gdf_vs_source = gdf_vs[(gdf_vs.ci_type==source_ci) & 
                               (gdf_vs.func_tot==1)]
        del gdf_vs
        
        v_ids_source, v_ids_target = self._select_closest_k(
            gdf_vs_source, gdf_vs_target, dist_thresh, bidir, k)
        
        if not link_name:
            link_name = f'dependency_{source_ci}_{target_ci}'

        self._edges_from_vlists(v_ids_source, v_ids_target, link_name) 


    def link_vertices_closest_k(self, source_ci, target_ci, link_name=None,
                                dist_thresh=None, bidir=False, k=5):
        """
        find k nearest source_ci vertices for each target_ci vertex,
        given distance constraints
        """
        gdf_vs = self.graph.get_vertex_dataframe()
        gdf_vs_target = gdf_vs[gdf_vs.ci_type==target_ci]
        gdf_vs_source = gdf_vs[(gdf_vs.ci_type==source_ci) & 
                               (gdf_vs.func_tot==1)]
        del gdf_vs
        
        v_ids_source, v_ids_target = self._select_closest_k(
            gdf_vs_source, gdf_vs_target, dist_thresh, bidir, k)
        
        if not link_name:
            link_name = f'dependency_{source_ci}_{target_ci}'

        self._edges_from_vlists(v_ids_source, v_ids_target, link_name)

    def _calc_friction(self, edge_geoms, friction_surf):
        
        from climada.entity.exposures.base import Exposures
        from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
        from climada.engine import Impact
        from climada.util import lines_polys_handler as u_lp
        import geopandas as gpd
        
        # define mapping as impact function.
        impf_fric = ImpactFunc() 
        impf_fric.id = 1
        impf_fric.haz_type = ''
        impf_fric.name = 'friction surface mapping'
        impf_fric.intensity_unit = 'min/m'
        impf_fric.intensity = np.linspace(friction_surf.intensity.data.min(),
                                          friction_surf.intensity.data.max(), 
                                          num=500) 
        impf_fric.mdd = np.linspace(friction_surf.intensity.data.min(),
                                    friction_surf.intensity.data.max(), 
                                    num=500)
        impf_fric.paa = np.sort(np.linspace(1, 1, num=500))
        impf_fric.check()
        impf_set = ImpactFuncSet()
        impf_set.append(impf_fric)
        
        # perform impact calc for mapping.
        exp_links = Exposures(gpd.GeoDataFrame({'geometry': edge_geoms}))
        exp_links.gdf['impf_'] = 1

        # step-by-step to avoid 0 duration sections
        exp_pnt = u_lp.exp_geom_to_pnt(
            exp_links, res=100, to_meters=True, 
            disagg_met=u_lp.DisaggMethod.FIX, disagg_val=100)
        
        impact_pnt = Impact()
        impact_pnt.calc(exp_pnt, impf_set, friction_surf, save_mat=True)
        if impact_pnt.imp_mat.size < len(exp_pnt.gdf):
            imp_arry = np.array(impact_pnt.imp_mat.todense()).flatten()
            imp_arry[imp_arry==0] = \
                exp_pnt.gdf.value[imp_arry==0]*friction_surf.intensity.data.min()
            impact_pnt.imp_mat = scipy.sparse.csr_matrix(imp_arry)      
        
        friction = u_lp.impact_pnt_agg(
            impact_pnt, exp_pnt.gdf, u_lp.AggMethod.SUM)
        
        return friction.eai_exp
        
    def link_vertices_friction_surf(self, source_ci, target_ci, friction_surf, 
                                    link_name=None, dist_thresh=None, 
                                    bidir=False, k=5, dur_thresh=None):
        
        gdf_vs = self.graph.get_vertex_dataframe()
        gdf_vs_target = gdf_vs[gdf_vs.ci_type==target_ci]
        gdf_vs_source = gdf_vs[(gdf_vs.ci_type==source_ci) & 
                               (gdf_vs.func_tot==1)]
        del gdf_vs
        
        if not (gdf_vs_source.empty or gdf_vs_target.empty):
            v_ids_source, v_ids_target = self._select_closest_k(
                gdf_vs_source, gdf_vs_target, dist_thresh, bidir, k)
                
            edge_geoms = make_edge_geometries(
                self.graph.vs[v_ids_source]['geometry'],
                self.graph.vs[v_ids_target]['geometry'])
            
            friction = self._calc_friction(edge_geoms, friction_surf)
            v_ids_source = np.array(v_ids_source)[friction<dur_thresh]
            v_ids_target = np.array(v_ids_target)[friction<dur_thresh]
            
            if not link_name:
                link_name = f'dependency_{source_ci}_{target_ci}'
            
            self._edges_from_vlists(
                list(v_ids_source), list(v_ids_target), link_name)
                
    
    def _create_subgraph_paths(self, source_ci, target_ci, via_ci):
        """
        Create a subgraph from the original graph to perform a shortest path
        search on.
        Includes only vertices from source, target and via types.
        Deletes all edges that do not belong to appropriate types or are 
        not fully functional.
        Deletes all vertices that are either not fully functional or already
        have a valid connection.
        To be used in link_vertices_shortest_paths() and 
        link_vertices_shortest_path()
        """
        
        # excluded targets that already have a valid dependency link
        # e.g. from walking assignment
        
        v_all = [vertex.index for vertex in self.graph.vs.select(
            ci_type_in=[source_ci, target_ci, via_ci], func_tot_gt=0)]
        v_exc = [edge.target for edge in self.graph.es.select(
                ci_type=f'dependency_{source_ci}_{target_ci}')]
        v_inc = np.setdiff1d(np.array(v_all), np.array(v_exc)).tolist()
        
        v_seq = self.graph.vs[v_inc]

        subgraph = self.graph.induced_subgraph(v_seq)
        subgraph.delete_edges(subgraph.es.select(func_tot_lt=1))
        wrong_edges = set(subgraph.es['ci_type']).difference(
            {via_ci, f'dependency_{source_ci}_{target_ci}'})
        subgraph.delete_edges(subgraph.es.select(ci_type_in=wrong_edges))
        
        return v_seq, subgraph
        
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
            True recommended for large road networks (>>100k edges).
            Default is auto - algorithm based on # edges.
        """
        v_seq, subgraph = self._create_subgraph_paths(
            source_ci, target_ci, via_ci)

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
            if len(path_dists)>0:
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

    def check_vertices_shortest_path(self, source_ci, target_ci, via_ci,
                                     dist_thresh=10e6, criterion='distance',
                                     link_name=None, bidir=False):
        
        # re-check those where source_ci functional AND distance of link > 5000m
        # since those are the ones with road paths
        es_check = self.graph.es.select(
            ci_type=f'dependency_{source_ci}_{target_ci}', distance_gt=5000)
        bools_check = [self.graph.vs[edge.source]['func_tot'] > 0 
                       for edge in es_check]
        es_check = [edge for edge, bool_check in zip(es_check, bools_check) 
                    if bool_check]
        
        # delete those edges on graph that have to be re-checked
        if len(es_check)>0: 
    
            # make subgraph to perform path search on
            v_ids_target = list(np.unique([edge.target for edge in es_check]))
            v_ids_source = list(np.unique([edge.source for edge in es_check]))
            v_ids_via = [vs.index for vs in 
                         self.graph.vs.select(ci_type=f'{via_ci}')]
            self.graph.delete_edges([edge.index for edge in es_check])
            
            v_seq = self.graph.vs([*v_ids_target, *v_ids_source,*v_ids_via])
            subgraph = self.graph.induced_subgraph(v_seq)
            subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(v_seq)
            subgraph.delete_edges(subgraph.es.select(func_tot_lt=1))
            wrong_edges = set(subgraph.es['ci_type']).difference(
                {via_ci, f'dependency_{source_ci}_{target_ci}'})
            subgraph.delete_edges(subgraph.es.select(ci_type_in=wrong_edges))
            
            lengths = []
    
            path_dists = subgraph.shortest_paths(
                source=subgraph.vs.select(ci_type=source_ci), 
                target=subgraph.vs.select(ci_type=target_ci), 
                weights='distance')
            
            path_dists = np.array(path_dists) # dim: (#sources, #targets)
            
            if len(path_dists)>0:
                ix_source, ix_target = np.where(
                    ((path_dists == path_dists.min(axis=0)) &
                     (path_dists<=dist_thresh))) # min dist. per target
                v_ids_source = [subgraph_graph_vsdict[vs.index]
                                for vs in subgraph.vs.select(
                                        ci_type=source_ci)[list(ix_source)]]
                v_ids_target = [subgraph_graph_vsdict[vs.index]
                                for vs in subgraph.vs.select(
                                        ci_type=target_ci)[list(ix_target)]]
                lengths = path_dists[(ix_source, ix_target)]
    
            if bidir:
                v_ids_target.extend(v_ids_source)
                v_ids_source.extend(v_ids_target)
                lengths.extend(lengths)
    
            if not link_name:
                link_name = f'dependency_{source_ci}_{target_ci}'
    
            if len(v_ids_source) > 0:
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
        
        v_seq, subgraph = self._create_subgraph_paths(
            source_ci, target_ci, via_ci)
        
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
           if len(path_dists)>0:
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
                         access_cnstr=False, dist_thresh=None, preselect='auto',
                         friction_surf=None, dur_thresh=None, cond=None):
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

        # TODO: Terrible spaghetti code. Refactor!
        # make links
        if not access_cnstr:
            if single_link:
                if cond is not None:
                    self.link_vertices_closest_k_cond(source, target, link_name=dep_name,
                                                 cond=cond, dist_thresh=dist_thresh, k=1)
                else:
                    self.link_vertices_closest_k(source, target, link_name=dep_name,
                                                 dist_thresh=dist_thresh, k=1)
            else:
                if cond is not None:
                    self.link_vertices_closest_k_cond(source, target, link_name=dep_name,
                                                 cond=cond, dist_thresh=dist_thresh, k=5)
                else:
                    self.link_vertices_closest_k(source, target, link_name=dep_name,
                                                 dist_thresh=dist_thresh, k=5)
        else:
            self.graph.delete_edges(ci_type=f'dependency_{source}_{target}')
            if single_link:
                self.link_vertices_friction_surf(source, target, friction_surf,
                                                 link_name=dep_name,
                                                 dist_thresh=dur_thresh*83.3,
                                                 k=1, dur_thresh=dur_thresh)
                self.link_vertices_shortest_path(source, target, via_ci='road',
                                    dist_thresh=dist_thresh, criterion='distance',
                                    link_name=dep_name, preselect=preselect)
            else:
                self.link_vertices_friction_surf(source, target, friction_surf,
                                                 link_name=dep_name,
                                                 dist_thresh=dur_thresh*83.3,
                                                 k=5, dur_thresh=dur_thresh)
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
        """
        for ci-types with an internally networked structure (e.g. roads and
        power lines which consist in edges + nodes), update those ci networks
        internally
        """
        # TODO: This function is poorly structured.
        # In a future version, have a more general parent-function,
        # and ci-type specific sub.-functions
        # ci_types_nw = (set(self.graph.vs['ci_type']) &
        #               set(self.graph.es['ci_type']))

        # specifically for roads: if edge is dysfunctional, render its target vertex dysfunctional
        if {'road'}.issubset(set(self.graph.vs['ci_type'])):
            LOGGER.info('Updating roads')
            targets_dys = [edge.target for edge in self.graph.es.select(
                ci_type='road').select(func_tot_eq=0)]
            self.graph.vs.select(targets_dys).select(ci_type='road')['func_tot'] = 0

        # specifically for powerlines: check power clusters
        if {p_source, p_sink}.issubset(set(self.graph.vs['ci_type'])):
            LOGGER.info('Updating power clusters')
            # For another version using pandapower, see nw_utils.py
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
            
            self._propagate_check_fail(row.source, row.target, row.thresh_func)

    def _update_enduser_dependencies(self, df_dependencies, preselect,
                                     friction_surf, dur_thresh):
        import timeit
        
        for __, row in df_dependencies[
                df_dependencies['type_I']=='enduser'].iterrows():

            if row.access_cnstr:
                LOGGER.info(f'Re-calculating paths from {row.source} to {row.target}')
                if row.single_link:
                    starttime = timeit.default_timer()
                    # those need to be re-checked on their fixed s-t pairs
                    self.check_vertices_shortest_path(row.source, row.target, via_ci='road',
                                    dist_thresh=row.thresh_dist, criterion='distance',
                                    link_name=f'dependency_{row.source}_{row.target}',
                                    bidir=False)
                    print(f"Time for recalculating from {row.source} to {row.target} :", timeit.default_timer() - starttime)
                else:
                    # the re-checking takes much longer than checking completely
                    # from scratch, hence check from scratch.
                    starttime = timeit.default_timer()
                    self.graph.delete_edges(ci_type=f'dependency_{row.source}_{row.target}')
                    if row.source != 'road': # for road, friction wasn't included in setup.
                        self.link_vertices_friction_surf(row.source, row.target, friction_surf,
                                                          link_name=f'dependency_{row.source}_{row.target}',
                                                          dist_thresh=dur_thresh*83.33,
                                                          k=5, dur_thresh=dur_thresh)
                        print(f"Time for recalculating friction from {row.source} to {row.target} :", timeit.default_timer() - starttime)
                    starttime = timeit.default_timer()
                    self.link_vertices_shortest_paths(row.source, row.target, via_ci='road',
                                    dist_thresh=row.thresh_dist, criterion='distance',
                                    link_name=f'dependency_{row.source}_{row.target}',
                                    bidir=False, preselect=preselect)
                    print(f"Time for recalculating paths from {row.source} to {row.target} :", timeit.default_timer() - starttime)
            self._propagate_check_fail(row.source, row.target, row.thresh_func)


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
            #treats case where empty adjacency matrix!
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
        capa_suff = (np.array(capa_rec.todense()).squeeze()>=func_thresh).astype(int)
         
        # This is under the assumption that subgraph retains the same
        # relative ordering of vertices as in v_seq extracted from graph!
        # This further assumes that any operation on a VertexSeq equally modifies its graph.
        # Both should be the case, but the igraph doc is always a bit ambiguous
        if target=='people':
            v_seq[f'actual_supply_{source}_{target}'] = capa_suff
        else:
            func_tot = np.minimum(capa_suff, v_seq['func_tot'])
            v_seq['func_tot'] = func_tot


    def _get_subgraph2graph_vsdict(self, vertex_seq):
        return dict((k,v) for k, v in zip(
            [subvx.index for subvx in self.graph.subgraph(vertex_seq).vs],
            [vx.index for vx in vertex_seq]))

    def cascade(self, df_dependencies, p_source='power_plant',
                p_sink='power_line', source_var='el_generation', demand_var='el_consumption',
                preselect='auto', initial=False, friction_surf=None, dur_thresh=None):
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
        if (cycles > 1) or (initial):
            self._update_enduser_dependencies(
                df_dependencies, preselect, friction_surf, dur_thresh)

    def return_network(self):
        return Network.from_graphs([self.graph])



class Graph(GraphCalcs):
    """
    creates a graph object from the
    """

    def __init__(self, network, directed=False):
        """
        network : instance of networks.base.Network
        """
        if network.edges is not None:
            self.graph = self.from_es(
                gdf_edges=network.edges, gdf_nodes=network.nodes,
                directed=directed)
        else:
            self.graph = self.from_vs(
                gdf_nodes=network.nodes, directed=directed)

    def _remove_namecol(self, gdf_nodes):
        if gdf_nodes is not None:
            if hasattr(gdf_nodes, 'name'):
                gdf_nodes = gdf_nodes.drop('name', axis=1)
        return gdf_nodes
    
    def from_es(self, gdf_edges, gdf_nodes=None, directed=False):
        return ig.Graph.DataFrame(
            gdf_edges, vertices=self._remove_namecol(gdf_nodes), directed=directed)

    def from_vs(self, gdf_nodes, directed=False):
        gdf_nodes = self._remove_namecol(gdf_nodes)
        vertex_attrs = gdf_nodes.to_dict('list')
        return ig.Graph(
            n=len(gdf_nodes),vertex_attrs=vertex_attrs, directed=directed)

