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

from climada_petals.engine.networks.base import Network, MultiNetwork
from climada.util.constants import ONE_LAT_KM

LOGGER = logging.getLogger(__name__)


class GraphMaker:
    """
    create an igraph Graph from network components
    and define some shared methods for both child classes
        
    initializes a graph on-the-fly and performs reuqired ops
    """
    
    def __init__(self, network):
        """
        network : instance of networks.base.Network or .MultiNetwork"""
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
        __, idx = btree.query(n_assign, k=1)
        return gdf_base.iloc[idx].index


    def make_edge_geometries(self, vs_geoms_from, vs_geoms_to):
        """
        create straight shapely LineString geometries between lists of 
        from and to nodes, to be added to newly created edges as attributes
        """
        return [shapely.geometry.LineString([geom_from, geom_to]) for
                                             geom_from, geom_to in
                                            zip(vs_geoms_from, vs_geoms_to)]

class GraphCalcs(GraphMaker):
    
    def __init__(self, nw_or_graph):
        """
        nw_or_graph : instance of networks.base.Network or .MultiNetwork or
            igraph.Graph
        """
        if isinstance(nw_or_graph, ig.Graph):
            self.graph = nw_or_graph
        else:
            return GraphMaker.__init__(self, nw_or_graph)
        
                      
    def return_network(self):
        edges = self.graph.get_edge_dataframe().rename({'source':'from_id',
                                                        'target':'to_id'},
                                                       axis=1).reset_index(drop=True)
        nodes = self.graph.get_vertex_dataframe().rename({'name':'name_id'},
                                                       axis=1).reset_index(drop=True)
        return Network(edges=edges, nodes=nodes)
    
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
        return self.return_network()
        
    
class MultiGraphCalcs(GraphMaker):
    
    def __init__(self, nw_or_graph):
        """
        nw_or_graph : instance of networks.base.Network or .MultiNetwork or
            igraph.Graph
        """
        if isinstance(nw_or_graph, ig.Graph):
            self.graph = nw_or_graph
        else:
            return GraphMaker.__init__(self, nw_or_graph)

    
    def link_closest_vertices(self, ci_type_assign, ci_type_base, 
                              link_type=None):
        """
        match all vertices of graph_assign to closest vertices in graph_base.
        Updated in vertex attributes (vID of graph_base, geometry & distance)
        
        """
        gdf_vs = self.graph.get_vertex_dataframe()
        gdf_vs_assign = gdf_vs[gdf_vs.ci_type == ci_type_assign]
        gdf_vs_base = gdf_vs[gdf_vs.ci_type == ci_type_base]

        ix_match = self._ckdnearest(gdf_vs_assign, gdf_vs_base)

        edge_geoms = self.make_edge_geometries(gdf_vs_assign.geometry,
                                          gdf_vs_base.loc[ix_match].geometry)
        # TODO: update orig_ids for new edges!!
        # edge_orig_ids = 
        if not link_type:
            link_type = f'dependency_{ci_type_assign}_{ci_type_base}'
        self.graph.add_edges(zip(gdf_vs_assign.index, ix_match), attributes =
                              {'geometry' : edge_geoms,
                               'ci_type' : [link_type],
                               'distance' : 1,
                               'func_level' : 1})
    
    def link_closest_vertices_subgraph(self, ci_type_assign, ci_type_base, 
                                       link_type=None, source_of=None, 
                                       target_of=None):
        """
        match all vertices of ci_type_assign to closest vertices in ci_type_base
        that are also source or target of an edge with ci_type 'source_of' or 
        'target_of', resp.

        Example
        -------
        E.g. assign hospitals only to power nodes that have a people-dependency
        link:
        link_closest_vertices_subgraph(health, power line, 
                                       link_type='dependency_health_power', 
                                       source_of='dependency_health_power')
        
        Parameters
        ----------
        
        Returns
        -------
        Updated in vertex attributes (vID of graph_base, geometry & distance)

        """
        gdf_vs = self.graph.get_vertex_dataframe()        
        gdf_vs_assign = gdf_vs[gdf_vs.ci_type == ci_type_assign]
        
        allowed_base = []
        
        if (source_of is not None) and (not target_of):
            es = self.graph.es.select(ci_type=source_of)
            for edge in es:
                allowed_base.append(edge.source)
        elif (target_of is not None) and (not source_of):
            es = self.graph.es.select(ci_type=source_of)
            for edge in es:
                allowed_base.append(edge.target)
        else:
            es1 = self.graph.es.select(ci_type=source_of)
            sources = np.unique([edge.source for edge in es1])
            es2 = self.graph.es.select(ci_type=target_of)
            targets = np.unique([edge.target for edge in es2])
            
            for source in sources:
                if source in targets:
                    allowed_base.append(source)

        allowed_base = np.unique(allowed_base)
        gdf_vs_base = gdf_vs[(gdf_vs.ci_type == ci_type_base)].loc[allowed_base]

        ix_match = self._ckdnearest(gdf_vs_assign, gdf_vs_base)
        edge_geoms = self.make_edge_geometries(gdf_vs_assign.geometry,
                                          gdf_vs_base.loc[ix_match].geometry)
        # TODO: update orig_ids for new edges!!
        # edge_orig_ids = 
        if not link_type:
            link_type = f'dependency_{ci_type_assign}_{ci_type_base}'
        self.graph.add_edges(zip(gdf_vs_assign.index, ix_match), attributes =
                              {'geometry' : edge_geoms,
                               'ci_type' : [link_type],
                               'distance' : 1,
                               'func_level' : 1})

    def return_multinetwork(self):
        return MultiNetwork(graphs=[self.graph])

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
            (edge['ci_type'] in allowed_edges) & (edge['func_level']==1))
                        else 999999 for edge in self.graph.es])
        
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
        
    def link_shortest_path_vertices(self, from_ci, to_ci, via_ci, 
                                    threshold=100000, criterion='distance'):
        
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
            
            for index, path in zip(indices, paths):
                if path:
                    dist = self.get_path_distance(path)
                    if dist < threshold:
                        edge_geom = self.make_edge_geometries(
                            [vx['geometry']], [vs_to[index]['geometry']])[0]
                        self.graph.add_edge(
                            vx, vs_to[index],
                            geometry = edge_geom,
                            ci_type = f'dependency_{from_ci}_{to_ci}',
                            distance = dist)

#      def update_shortest_path_dist(self, from_ci, to_ci, via_ci, )                 

#     def _shortest_paths_sub(self, subgraph, from_ci, to_ci, weight,
#                             mode='out', output='epath'):
#         """
#         shortest paths from all origin CIs to all destination CIs
#         """
#         vseq_from_ci = subgraph.vs.select(ci_type=from_ci)
#         vseq_to_ci = subgraph.vs.select(ci_type=to_ci)

#         # summary dfs
#         epaths = pd.DataFrame(index=range(vseq_from_ci[0].index,vseq_from_ci[-1].index+1),
#                          columns=range(vseq_to_ci[0].index,vseq_to_ci[-1].index+1))
#         weight_sum = pd.DataFrame(index=range(vseq_from_ci[0].index,vseq_from_ci[-1].index+1),
#                          columns=range(vseq_to_ci[0].index,vseq_to_ci[-1].index+1))
        
#         # actual calcs
#         for vs_from in vseq_from_ci:
#             epaths.loc[vs_from.index] = subgraph.get_shortest_paths(
#                 vs_from, vseq_to_ci, weight, mode, output)
#             weight_sum.loc[vs_from.index] = [weight.loc[epaths.loc[vs_from.index,vs_to.index]].sum()
#                                              for vs_to in vseq_to_ci]
#         return epaths, weight_sum.astype(float).replace(0, np.nan)

#     def _single_shortest_path_sub(self, epaths, weight_sum):
#         """
#         get the single shortest path from all origin CIs to their respective
#         closest destination CI given edge-paths lists and weights from all
#         origins to all destinations
#         """

#         shortest_path = pd.DataFrame(index=epaths.index,
#                                      columns=['to_ID','epath', 'weight'])
#         shortest_path.index.name = 'from_ID'

#         # get columns of respective shortest paths
#         shortest_path.to_ID = weight_sum.idxmin(axis=1)
#         # get rows w/ and w/o paths at all
#         paths = shortest_path[~np.isnan(shortest_path.to_ID)].index

#         # this could be vectorized fully with fancy indexing on numpy arrays
#         # --> get positional index locations
#         shortest_path['epath'][paths] = [epaths.loc[from_id,to_id]
#                                          for from_id, to_id in
#                                          zip(paths,shortest_path.loc[paths].to_ID)]
#         shortest_path['weight'][paths] = [weight_sum.loc[from_id,to_id]
#                                          for from_id, to_id in
#                                          zip(paths,shortest_path.loc[paths].to_ID)]

#         return shortest_path

#     def single_shortest_path(self, from_ci, to_ci, via_ci=None, weights='distance',
#                              mode='out', output='epath'):

#         # computation on subgraph
#         subgraph = self._construct_subgraph_from_vs(from_ci, to_ci, via_ci)
#         # combine weights in case more than 1 attribute given
#         weight = self._edge_weights_sub(subgraph, weights)
#         # all shortest paths
#         epaths, weight_sum = self._shortest_paths_sub(subgraph, from_ci, to_ci,
#                                                       weight, mode, output)
#         # single shortest path
#         shortest_path_sub = self._single_shortest_path_sub(epaths, weight_sum)

#         # mapper of vs IDs and es IDs back to whole graph
#         dict_idxmapper = self._idmapper_subgraph_to_graph(subgraph)

#         shortest_path = pd.DataFrame(np.array([shortest_path_sub.to_ID.map(dict_idxmapper['vertices']),
#                                                [pd.Series(edgelist).map(dict_idxmapper['edges']) 
#                                                 for edgelist in shortest_path_sub.epath],
#                                                shortest_path_sub.weight]).T,
#                                      index=shortest_path_sub.index.map(dict_idxmapper['vertices']),
#                                      columns=shortest_path_sub.columns)
#         return shortest_path
    

