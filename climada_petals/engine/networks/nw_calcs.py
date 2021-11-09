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
import igraph as ig
import logging
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from climada_petals.engine.networks.base import Network
from climada.util.constants import ONE_LAT_KM

LOGGER = logging.getLogger(__name__)


class GraphCalcs():
    
    def __init__(self, graph):
        """
        nw_or_graph : instance of networks.base.Network or .MultiNetwork or
            igraph.Graph
        """
        self.graph = graph

    def link_clusters(self):
        """
        select random vs from a cluster and match with closest vs from largest
        cluster
        """        
        subgraph_list = self.graph.clusters().subgraphs()
        gdf_vs_source = subgraph_list[0].get_vertex_dataframe()

        for subgraph in subgraph_list[1:]:
            vs_assign = subgraph.get_vertex_dataframe().iloc[0]

            __, ix_match = self._ckdnearest(vs_assign, gdf_vs_source)

            edge_geom = self.make_edge_geometries(
                [vs_assign.geometry], 
                [gdf_vs_source.loc[ix_match].geometry.values[0]])[0]

            self.graph.add_edge(vs_assign.orig_id, 
                                gdf_vs_source.loc[ix_match].orig_id.values[0], 
                                geometry=edge_geom, ci_type=vs_assign.ci_type,
                                distance = 1)
    
    def link_vertices_closest(self, from_ci, to_ci, 
                              link_name=None, dist_thresh=None,
                              bidir=False, k=1):
        """
        match all vertices of graph_assign to closest vertices in graph_base.
        Updated in vertex attributes (vID of graph_base, geometry & distance)
        
        """
        gdf_vs = self.graph.get_vertex_dataframe()
        gdf_vs_target = gdf_vs[gdf_vs.ci_type == to_ci]
        gdf_vs_source = gdf_vs[gdf_vs.ci_type == from_ci]

        # shape: (target vs, k)
        dists, ix_matches = self._ckdnearest(gdf_vs_target, gdf_vs_source, k=k)
        if k>1:
            dists = dists.flatten()
            ix_matches = ix_matches.flatten()
        # TODO: dist_thresh condition only holds vaguely for m input & lat/lon coordinates (EPSG4326)
        # shape: (target vs, k)
        dists_thresh_bool = [((not dist_thresh) or (dist < (dist_thresh/(ONE_LAT_KM*1000))))
                            for dist in dists]
        
        # shape: (target vs, ..)
        if k > 1:
            gdf_target_old = gdf_vs_target.copy()
            for i in range(k-1):
                gdf_vs_target = gdf_vs_target.append(gdf_target_old)
           
        edge_geoms = self.make_edge_geometries(gdf_vs_target.geometry[dists_thresh_bool],
                                                    gdf_vs_source.loc[ix_matches].geometry[dists_thresh_bool])
        if not link_name:
            link_name = f'dependency_{from_ci}_{to_ci}'

        self.graph.add_edges(zip(ix_matches[dists_thresh_bool],
                                     gdf_vs_target.index[dists_thresh_bool]), 
                                         attributes =
                                          {'geometry' : edge_geoms,
                                           'ci_type' : [link_name],
                                           'distance' : dists[dists_thresh_bool]*(ONE_LAT_KM*1000),
                                           'func_internal' : 1,
                                           'func_tot' : 1,
                                           'imp_dir' : 0})
        if bidir:
            self.graph.add_edges(zip(gdf_vs_target.index[dists_thresh_bool],
                                          ix_matches[dists_thresh_bool]), 
                                         attributes =
                                          {'geometry' : edge_geoms,
                                           'ci_type' : [link_name],
                                           'distance' : dists[dists_thresh_bool]*(ONE_LAT_KM*1000),
                                           'func_internal' : 1,
                                           'func_tot' : 1,
                                           'imp_dir' : 0})

    def link_vertices_shortest_paths(self, from_ci, to_ci, via_ci, 
                                    dist_thresh=100000, criterion='distance',
                                    link_name=None, bidir=False):
        """make all links below certain dist_thresh along shortest paths length"""
        
        vs_from = self.graph.vs.select(ci_type=from_ci)
        vs_to = self.graph.vs.select(ci_type=to_ci)
        
        # TODO: don't hardcode metres to degree conversion assumption
        ix_matches = self._preselect_destinations(vs_from, vs_to, 
                                                  dist_thresh/(ONE_LAT_KM*1000))
        
        # TODO: think about performing this on sub-graph instead of weight.based
        for vx, indices in zip(vs_from, ix_matches):
            weight = self._make_edgeweights(criterion, from_ci, to_ci, 
                                            via_ci)
            paths = self.get_shortest_paths(vx, vs_to[indices], weight, 
                                            mode='out', output='epath')
            
            for index, path in zip(indices, paths):
                if path:
                    dist = weight[path].sum()
                    if dist < dist_thresh:
                        edge_geom = self.make_edge_geometries(
                            [vx['geometry']], [vs_to[index]['geometry']])[0]
                        if not link_name:
                            link_name = f'dependency_{from_ci}_{to_ci}'
                        #TODO: collect all edges to be added and assign in one add_edges call
                        self.graph.add_edge(vx, vs_to[index],geometry=edge_geom,
                                            ci_type=link_name, 
                                            distance=dist,func_internal=1, 
                                            func_tot=1, imp_dir=0)  
                        if bidir:
                            self.graph.add_edge(vs_to[index], vx,geometry=edge_geom,
                                            ci_type=link_name, 
                                            distance=dist,func_internal=1, 
                                            func_tot=1, imp_dir=0)

                        
    def link_vertices_shortest_path(self, from_ci, to_ci, via_ci, 
                                    dist_thresh=100000, criterion='distance',
                                    link_name=None, bidir=False):
        """only single shortest path below dist_thresh is chosen"""
        
        vs_from = self.graph.vs.select(ci_type=from_ci)
        vs_to = self.graph.vs.select(ci_type=to_ci)
        
    
        # TODO: don't hardcode metres to degree conversion assumption
        ix_matches = self._preselect_destinations(vs_from, vs_to, 
                                                  dist_thresh/(ONE_LAT_KM*1000))

        for vx, indices in zip(vs_from, ix_matches):
            weight = self._make_edgeweights(criterion, from_ci, to_ci, 
                                            via_ci)            

            paths = self.get_shortest_paths(vx, vs_to[indices], weight, 
                                            mode='out', output='epath')
            min_dist = dist_thresh
            
            for index, path in zip(indices, paths):
                if path:
                    dist = weight[path].sum()
                    if dist < min_dist:
                        min_dist = dist
                        min_index = index
            
            if min_dist < dist_thresh:
                edge_geom = self.make_edge_geometries([vx['geometry']], 
                                                      [vs_to[min_index]['geometry']])[0]
                if not link_name:
                            link_name = f'dependency_{from_ci}_{to_ci}'
                #TODO: collect all edges to be added and assign in one add_edges call
                self.graph.add_edge(vx, vs_to[min_index], geometry=edge_geom,
                                    ci_type=link_name, distance=min_dist,
                                    func_internal=1, func_tot=1, imp_dir=0)
                if bidir:
                    self.graph.add_edge( vs_to[min_index], vx, geometry=edge_geom,
                                    ci_type=link_name, distance=min_dist,
                                    func_internal=1, func_tot=1, imp_dir=0)


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
        
        dep_name = f'dependency_{source}_{target}'
        
        # make links
        if not access_cnstr:
            if single_link:
                self.link_vertices_closest(source, target, link_name=dep_name,
                                           dist_thresh=dist_thresh, k=1)
            else:
                self.link_vertices_closest(source, target, link_name=dep_name,
                                           dist_thresh=dist_thresh, k=10)
        else:
            if single_link:
                self.link_vertices_shortest_path(source, target, via_ci='road', 
                                    dist_thresh=dist_thresh, criterion='distance',
                                    link_name=dep_name)
            else:
                self.link_vertices_shortest_paths(source, target, via_ci='road', 
                                    dist_thresh=dist_thresh, criterion='distance',
                                    link_name=dep_name)

    
    def place_demands(self, ci_type_demand, ci_type_supply, flow_var):
        """place base demand"""
        
        # initialize demand
        demand_name = f'demand_{ci_type_demand}_{ci_type_supply}'
        self.graph.vs.select(ci_type=ci_type_demand).set_attribute_values(
            demand_name, 0)
        
        # only functional vertices have demands
        target_vs = self.graph.vs.select(ci_type=ci_type_demand).select(func_tot_gt=0)
        demand_vals = 1 if flow_var=='binary' else target_vs.get_attribute_values(flow_var)
        target_vs.set_attribute_values(demand_name, demand_vals)

    def calc_flows(self, ci_type_target, ci_type_source):
        
        flow_name = f'flow_{ci_type_source}_{ci_type_target}'
        dep_name = f'dependency_{ci_type_source}_{ci_type_target}'
        demand_name = f'demand_{ci_type_target}_{ci_type_source}'
        
        # initialize flow
        self.graph.es.select(ci_type=dep_name).set_attribute_values(flow_name, 0)
        
        # consider only functioning dependency edges (requiring functioning 
        # source implicitly) & functioning targets
        dep_es = self.graph.es.select(ci_type=dep_name).select(func_tot_gt=0)
        target_vs = self.graph.vs.select(ci_type=ci_type_target).select(func_tot_gt=0)

        for target_vx in target_vs:
            eindex = [edge.index for edge in dep_es 
                      if edge.target==target_vx.index]
            flow = 0 if len(eindex) < 1 else target_vx[demand_name]/len(eindex)
            for edge in self.graph.es[eindex]:
                edge[flow_name] +=flow
        
    def calc_supplies(self, ci_type_target, ci_type_source):
        
        supply_name = f'supply_{ci_type_source}_{ci_type_target}'
        dep_name = f'dependency_{ci_type_source}_{ci_type_target}'
        flow_name = f'flow_{ci_type_source}_{ci_type_target}'

        # initialize supply
        self.graph.vs.select(ci_type=ci_type_source).set_attribute_values(supply_name, 0)

        # consider only functioning dependency edges
        dep_es = self.graph.es.select(ci_type=dep_name).select(func_tot_gt=0)
        
        for edge in dep_es:
            self.graph.vs[edge.source][supply_name]+=edge[flow_name]
            # place a checker for now TODO: remove once sure that updating works
            if self.graph.vs[edge.source]['func_state'] < edge['func_thresh']:
                LOGGER.warning('Funcstate of source below threshold.'+
                               f'Check updating mechanism. Edge: {edge.index}')
    
    def cluster_nodesums(self, sum_variable):
        df_vs = self.graph.get_vertex_dataframe()
        df_vs['vs_membership'] = self.graph.clusters().membership
        return df_vs.groupby(['vs_membership'])[sum_variable].sum()
    
    def cluster_edgesums(self, sum_variable):
        df_es = self.graph.get_edge_dataframe()
        df_es['es_membership'] = self.get_cluster_affiliation_es()
        return df_es.groupby(['es_membership'])[sum_variable].sum()
    
    def get_cluster_affiliation_es(self):
        vs_membership = self.graph.clusters().membership
        vs_cluster_dict = dict(zip(range(self.graph.vcount()),vs_membership))
        return [vs_cluster_dict[x] for x in 
                self.graph.get_edge_dataframe()['source']]
    
    def update_dependency_funcstate(self):
        
        # all types of dependencies: check that source still functional
        dep_names = [dep_name for dep_name in 
                     np.unique(self.graph.es.get_attribute_values('ci_type')) 
                     if 'dependency' in dep_name]
        es = self.graph.es.select(ci_type_in=dep_names)
        func_list = []
        for edge in es:
            func_list.append(1 if self.graph.vs[edge.source]['func_tot'] 
                             > edge['func_thresh'] else 0)
        es.set_attribute_values('func_internal', func_list)
        es.set_attribute_values('func_tot', func_list)

        # access dependencies: check that path additionally satisfies dist_thresh constraint
        es_access = es.select(access_cnstr=True).select(func_tot_gt=0)
        func_list=[]
        
        for edge in es_access:
            source = self.graph.vs[edge.source]
            target = self.graph.vs[edge.source]
            
            # TODO: don't hard-code via-ci
            weight = self._make_edgeweights('distance', source['ci_type'], 
                                            target['ci_type'], 'road')
            
            paths = self.get_shortest_paths(source, target, weight, 
                                            mode='out', output='epath')
            distance = edge['dist_thresh']
            
            if paths:
                for path in paths:
                    dist = weight[path].sum()
                    if dist < distance:
                        distance = dist
            func_list.append(1 if distance < edge['dist_thresh'] else 0)
            
        es_access.set_attribute_values('func_internal', func_list)
        es_access.set_attribute_values('func_tot', func_list)       

    def update_target_funcstate(self):
        
        for vx in self.graph.vs.select(ci_type_notin='people'):
            dep_edges = [edge for edge in vx.incident(mode='in') if 
                         'dependency_' in edge['ci_type']]
            if dep_edges:
                vx['func_tot'] = min(min([edge['func_tot'] for edge in dep_edges]),
                                     vx['func_internal'])
    
    def _funcstate_iteration(self):
        
        func_es_t0 = sum(self.graph.es.get_attribute_values('func_tot'))   
        func_vs_t0 = sum(self.graph.vs.get_attribute_values('func_tot'))
        
        self.update_dependency_funcstate()
        self.update_target_funcstate()
        
        func_es_t1 = sum(self.graph.es.get_attribute_values('func_tot'))   
        func_vs_t1 = sum(self.graph.vs.get_attribute_values('func_tot'))
        
        return func_es_t0-func_es_t1, func_vs_t0-func_vs_t1
    
    def update_funcstates(self):
        
        diff_func_es, diff_func_vs = self._funcstate_iteration()
        iteration_round=0
        
        while ((diff_func_es!=0) or (diff_func_vs!=0) and iteration_round<100):
            diff_func_es, diff_func_vs = self._funcstate_iteration()
            iteration_round+=1
            print(f'Iteration Round: {iteration_round}')       
            print(f'Func_state diff (vs): {diff_func_vs}, func-state diff (es) : {diff_func_es}')
            
        if iteration_round < 100:
            print('Converged!')
        else:
            print('Cascade did not converge within 100 steps')
        
    def get_shortest_paths(self, from_vs, to_vs, edgeweights, mode='out', 
                           output='epath'):
        
        return self.graph.get_shortest_paths(from_vs, to_vs, edgeweights, 
                                             mode, output)
        
    def _make_edgeweights(self, criterion, from_ci, to_ci, via_ci):
        
        allowed_edges= [f'dependency_{from_ci}_{via_ci}', 
                        f'dependency_{to_ci}_{via_ci}',
                        f'{via_ci}']
        
        return np.array([1*edge[criterion] if (
            (edge['ci_type'] in allowed_edges) & (edge['func_internal']==1))
                        else 10e16 for edge in self.graph.es])
        
    def _preselect_destinations(self, vs_assign, vs_base, dist_thresh):
        points_base = np.array([(x.x, x.y) for x in vs_base['geometry']])
        point_tree = cKDTree(points_base)
        
        points_assign = np.array([(x.x, x.y) for x in vs_assign['geometry']])
        ix_matches = []
        for assign_loc in points_assign:
            ix_matches.append(point_tree.query_ball_point(assign_loc, dist_thresh))
        return ix_matches
    
    def get_path_distance(self, epath):
        return sum(self.graph.es[epath]['distance'])
        
    def graph_style(self, ecol='red', vcol='red', vsize=2, ewidth=3, *kwargs):
        visual_style = {}
        visual_style["edge_color"] = ecol
        visual_style["vertex_color"] = vcol
        visual_style["vertex_size"] = vsize
        visual_style["edge_width"] = ewidth
        visual_style["layout"] = self.graph.layout("fruchterman_reingold")
        visual_style["edge_arrow_size"] = 0
        visual_style["edge_curved"] = 1
        if kwargs:
            pass
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

    def _ckdnearest(self, vs_assign, gdf_base, k=1):
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
        if (isinstance(vs_assign, gpd.GeoDataFrame) 
            or isinstance(vs_assign, pd.DataFrame)):
            n_assign = np.array(list(vs_assign.geometry.apply(lambda x: (x.x, x.y))))
        else:
            n_assign = np.array([(vs_assign.geometry.x, vs_assign.geometry.y)])
        n_base = np.array(list(gdf_base.geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(n_base)
        dist, idx = btree.query(n_assign, k=k)
        return dist, np.array(gdf_base.iloc[idx.flatten()].index).reshape(dist.shape)


    def make_edge_geometries(self, vs_geoms_from, vs_geoms_to):
        """
        create straight shapely LineString geometries between lists of 
        from and to nodes, to be added to newly created edges as attributes
        """
        return [shapely.geometry.LineString([geom_from, geom_to]) for
                                             geom_from, geom_to in
                                            zip(vs_geoms_from, vs_geoms_to)]
    
    def calc_supply_demand_balance(self):
        
        vertex_vars = self.graph.vs.attribute_names()
        
        for vx in self.graph.vs:
            ci_type_vx = vx['ci_type']
            supporting_cis = [dvar.split('_')[-1] for dvar in vertex_vars if 
                              f"demand_{ci_type_vx}" in dvar]
        
            for supporting_ci in supporting_cis:
                demand_var = f"demand_{ci_type_vx}_{supporting_ci}"
                flow_var = f"flow_{supporting_ci}_{ci_type_vx}"
                sd_var = f"sd_balance_{supporting_ci}_{ci_type_vx}"  
                
                supply_act = 0
                supporting_edges = [edge for edge in vx.incident(mode='in') 
                                    if f"dependency_{supporting_ci}" in edge['ci_type']]
                supply_act += sum([edge[flow_var] for edge in supporting_edges])
                vx[sd_var] = 1 + (supply_act - vx[demand_var])/vx[demand_var]
    
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

