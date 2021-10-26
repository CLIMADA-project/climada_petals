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
        gdf_vs_base = subgraph_list[0].get_vertex_dataframe()

        for subgraph in subgraph_list[1:]:
            vs_assign = subgraph.get_vertex_dataframe().iloc[0]

            ix_match = self._ckdnearest(vs_assign, gdf_vs_base)

            edge_geom = self.make_edge_geometries(
                [vs_assign.geometry], 
                [gdf_vs_base.loc[ix_match].geometry.values[0]])[0]

            self.graph.add_edge(vs_assign.orig_id, 
                                gdf_vs_base.loc[ix_match].orig_id.values[0], 
                                geometry=edge_geom, ci_type=vs_assign.ci_type,
                                distance = 1)
    
    def link_vertices_closest(self, ci_type_assign, ci_type_base, 
                              dep_type=None,link_name=None, threshold=None):
        """
        match all vertices of graph_assign to closest vertices in graph_base.
        Updated in vertex attributes (vID of graph_base, geometry & distance)
        
        """
        gdf_vs = self.graph.get_vertex_dataframe()
        gdf_vs_assign = gdf_vs[gdf_vs.ci_type == ci_type_assign]
        gdf_vs_base = gdf_vs[gdf_vs.ci_type == ci_type_base]

        dist, ix_match = self._ckdnearest(gdf_vs_assign, gdf_vs_base)
        
        # TODO: threshold condition only holds vaguely for m input & lat/lon coordinates (EPSG4326)
        threshold_bool = [((not threshold) or (distance < (threshold/(ONE_LAT_KM*1000))))
                          for distance in dist]
        
        edge_geoms = self.make_edge_geometries(gdf_vs_assign.geometry[threshold_bool],
                                               gdf_vs_base.loc[ix_match].geometry[threshold_bool])
        # TODO: update orig_ids for new edges!!
        if not link_name:
            link_name = f'dependency_{ci_type_assign}_{ci_type_base}'
        self.graph.add_edges(zip(gdf_vs_assign.index[threshold_bool], 
                                 ix_match[threshold_bool]), 
                                 attributes =
                                  {'geometry' : edge_geoms,
                                   'ci_type' : [link_name],
                                   'dep_type' : [dep_type],
                                   'distance' : dist[threshold_bool]*(ONE_LAT_KM*1000),
                                   'func_internal' : 1})
    

    def link_vertices_shortest_paths(self, from_ci, to_ci, via_ci, 
                                    threshold=100000, criterion='distance',
                                    link_name=None, dep_type=None):
        """make all links below certain threshold along shortest paths length"""
        
        vs_from = self.select_nodes(from_ci)
        vs_to = self.select_nodes(to_ci)
        
        # TODO: don't hardcode metres to degree conversion assumption
        ix_matches = self._preselect_destinations(vs_from, vs_to, 
                                                  threshold/(ONE_LAT_KM*1000))
        
        # TODO: think about performing this on sub-graph instead of weight.based
        for vx, indices in zip(vs_from, ix_matches):
            weight = self._make_edgeweights(criterion, from_ci, to_ci, 
                                            via_ci)
            paths = self.get_shortest_paths(vx, vs_to[indices], weight, 
                                            mode='out', output='epath')
            
            for index, path in zip(indices, paths):
                if path:
                    dist = weight[path].sum()
                    if dist < threshold:
                        edge_geom = self.make_edge_geometries(
                            [vx['geometry']], [vs_to[index]['geometry']])[0]
                        if not link_name:
                            link_name = f'dependency_{from_ci}_{to_ci}'
                        #TODO: collect all edges to be added and assign in one add_edges call
                        self.graph.add_edge(vx, vs_to[index],geometry=edge_geom,
                                            ci_type=link_name, dep_type=dep_type,
                                            distance=dist,func_internal=1, threshold=threshold)

                        
    def link_vertices_shortest_path(self, from_ci, to_ci, via_ci, 
                                    threshold=100000, criterion='distance',
                                    link_name=None, dep_type=None):
        """only single shortest path below threshold is chosen"""
        
        vs_from = self.select_nodes(from_ci)
        vs_to = self.select_nodes(to_ci)
    
        # TODO: don't hardcode metres to degree conversion assumption
        ix_matches = self._preselect_destinations(vs_from, vs_to, 
                                                  threshold/(ONE_LAT_KM*1000))

        for vx, indices in zip(vs_from, ix_matches):
            weight = self._make_edgeweights(criterion, from_ci, to_ci, 
                                            via_ci)            

            paths = self.get_shortest_paths(vx, vs_to[indices], weight, 
                                            mode='out', output='epath')
            
            min_dist = threshold
            
            for index, path in zip(indices, paths):
                if path:
                    dist = weight[path].sum()
                    if dist < min_dist:
                        min_dist = dist
                        min_index = index
            
            if min_dist < threshold:
                edge_geom = self.make_edge_geometries([vx['geometry']], 
                                                      [vs_to[min_index]['geometry']])[0]
                if not link_name:
                            link_name = f'dependency_{from_ci}_{to_ci}'
                #TODO: collect all edges to be added and assign in one add_edges call
                self.graph.add_edge(vx, vs_to[min_index], geometry=edge_geom,
                                    ci_type=link_name, dep_type=dep_type,
                                    distance=dist,func_internal=1, threshold=threshold)


    def place_dependency(self, source, target, link_type, dep_type, dvar, dist_thresh=None):
        """
        source : supporting infra
        target : dependent infra / ppl
        dep_type: au, as, du, ds, fu, fs
        link_type: shortest, closest
        dvar : demand variable. "binary" or varname in node attrs
        dist_thresh : max. link-distance in metres
        """
        
        demand_name = f'demand_{target}_{source}'
        flow_name = f'flow_{source}_{target}'
        supply_name = f'supply_{source}_{target}'
        dep_name = f'dependency_{source}_{target}'
        
        # make links
        if link_type == 'closest':
            self.link_vertices_closest(target, source, dep_type, dep_name, threshold=dist_thresh)
        elif ((link_type == 'shortest') and (dep_type in ['au', 'du', 'fu'])):
             self.link_vertices_shortest_path(source, target,'road',dist_thresh,
                                              dep_name, dep_type)
        elif ((link_type == 'shortest') and (dep_type in ['as', 'ds', 'fs'])):
             self.link_vertices_shortest_paths(source, target,'road', dist_thresh,
                                              dep_name, dep_type)
        else:
            raise ValueError('Invalid combination of kwargs')
        
        
        # place base demand
        target_vs = self.graph.vs.select(ci_type_in=target)
        
        if dvar == 'binary':
            target_vs.set_attribute_values(demand_name, 1)
        else:
            demand_vals = target_vs.get_attribute_values(dvar)
            target_vs.set_attribute_values(demand_name, demand_vals)
            
        # distribute flows & place supplies
        self.graph.vs.select(ci_type=source).set_attribute_values(supply_name, 0)

        for target_vx in target_vs:
            incident_edges = target_vx.incident(mode='in')    
            if dep_type in ['au', 'du', 'fu']:
                for edge in incident_edges:
                    if edge['ci_type'] == f'dependency_{source}_{target}':
                        edge[flow_name] = target_vx[demand_name]
                        self.graph.vs[edge.source][supply_name]+= edge[flow_name]
            else:
                e_count = 0
                for edge in incident_edges:
                    if edge['ci_type'] == f'dependency_{source}_{target}':
                        e_count+=1
                for edge in incident_edges:
                    if edge['ci_type'] == f'dependency_{source}_{target}':
                        edge[flow_name] = target_vx[demand_name]/e_count
                        self.graph.vs[edge.source][supply_name]+= edge[flow_name]
        
            
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
    
    def select_edges(self, edgename):
        return self.graph.es.select(ci_type=edgename)
    
    def select_nodes(self, nodename):
        return self.graph.vs.select(ci_type=nodename)
    
    def update_dependency_funcstate(self):
        
        # all types of dependencies: check that source still functional
        es = self.graph.es.select(_in=['du', 'ds', 'fu', 'fs','au', 'as'])
        func_list = []
        for edge in es:
            func_list.append(self.graph.vs[edge.source]['func_tot'])
        es.set_attribute_values('func_internal', func_list)
        es.set_attribute_values('func_tot', func_list)

        # access dependencies: check that path additionally satisfies threshold constraint
        es = self.graph.es.select(_in=['au', 'as']).select(func_tot_eq=1)
        func_list=[]
        
        for edge in es:
            source = self.graph.vs[edge.source]
            target = self.graph.vs[edge.source]
            
            # TODO: don't hard-code via-ci
            weight = self._make_edgeweights('distance', source['ci_type'], 
                                            target['ci_type'], 'road')
            
            paths = self.get_shortest_paths(source, target, weight, 
                                            mode='out', output='epath')
            distance = edge['threshold']
            
            if paths:
                for path in paths:
                    dist = weight[path].sum()
                    if dist < distance:
                        distance = dist
            func_list.append(distance < edge['threshold'])
            
        es.set_attribute_values('func_internal', func_list)
        es.set_attribute_values('func_tot', func_list)       

        
    def _get_var_from_parent(self, edge_type, varname):
        
        edges = self.select_edges(edge_type)
        var_parent = []
        for edge in edges:
            var_parent.append(self.graph.vs[edge.source][varname])
        return var_parent

    def _assign_to_child(self, edge_type, varname, varlist=None):
        
        edges = self.select_edges(edge_type)
        # "clean" var to be assigned
        for edge in edges:
            self.graph.vs[edge.target][varname] = 0
            
        if not varlist:
            for edge in edges:
                self.graph.vs[edge.target][varname] += \
                    self.graph.vs[edge.source][varname]
        else:
            for edge, var in zip(edges, varlist):
                self.graph.vs[edge.target][varname] = var
                    
    def _assign_to_parent(self, edge_type, varname, varlist=None):
        
        edges = self.select_edges(edge_type)
        # "clean" var to be assigned
        for edge in edges:
            self.graph.vs[edge.source][varname] = 0
            
        if not varlist:
            for edge in edges:
                self.graph.vs[edge.source][varname] += \
                    self.graph.vs[edge.target][varname]
        else:
            for edge, var in zip(edges, varlist):
                self.graph.vs[edge.source][varname] = var
 

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
        
    def _preselect_destinations(self, vs_assign, vs_base, threshold):
        points_base = np.array([(x.x, x.y) for x in vs_base['geometry']])
        point_tree = cKDTree(points_base)
        
        points_assign = np.array([(x.x, x.y) for x in vs_assign['geometry']])
        ix_matches = []
        for assign_loc in points_assign:
            ix_matches.append(point_tree.query_ball_point(assign_loc, threshold))
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

    def _ckdnearest(self, vs_assign, gdf_base):
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
        dist, idx = btree.query(n_assign, k=1)
        return dist, gdf_base.iloc[idx].index


    def make_edge_geometries(self, vs_geoms_from, vs_geoms_to):
        """
        create straight shapely LineString geometries between lists of 
        from and to nodes, to be added to newly created edges as attributes
        """
        return [shapely.geometry.LineString([geom_from, geom_to]) for
                                             geom_from, geom_to in
                                            zip(vs_geoms_from, vs_geoms_to)]
    
    def return_network(self):
        return Network.from_graphs([self.graph])
    

class Graph(GraphCalcs):
    """
    create an igraph Graph from network components
    """
    
    def __init__(self, network):
        """
        network : instance of networks.base.Network"""
        if network.edges is not None:
            self.graph = self.graph_from_es(gdf_edges=network.edges, 
                                            gdf_nodes=network.nodes)
        else:
            self.graph = self.graph_from_vs(gdf_nodes=network.nodes)
    
    @staticmethod
    def graph_from_es(gdf_edges, gdf_nodes=None, directed=False):
        return ig.Graph.DataFrame(gdf_edges, directed=False,vertices=gdf_nodes)
   
    @staticmethod
    def graph_from_vs(gdf_nodes):
        vertex_attrs = gdf_nodes.to_dict('list')
        return ig.Graph(n=len(gdf_nodes),vertex_attrs=vertex_attrs)

