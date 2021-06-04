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

"""Make graph base class"""

import igraph as ig
import logging
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

LOGGER = logging.getLogger(__name__)


class Network():
    def __init__(self, gdf_edges=None, gdf_nodes=None):
        """
        gdf_nodes : id identifier needs to be first column of nodes
        gdf_edges : from_id and to_id need to be first two columns of edges
        """

        self.graph = ig.Graph()
        if gdf_edges is not None:
            self.graph = self.graph_from_es(gdf_edges=gdf_edges, gdf_nodes=gdf_nodes)
        elif gdf_nodes is not None:
            self.graph = self.graph_from_vs(gdf_nodes)

    @staticmethod
    def graph_from_es(gdf_edges, gdf_nodes=None, directed=False):
        return ig.Graph.DataFrame(gdf_edges, directed=False,
                                        vertices=gdf_nodes)
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

    def plot_geodata(self, *kwargs):
        pass

class MultiNetwork():
    def __init__(self, *networks):
        #self.__dict__.update(*networks)

        self.graph = ig.Graph()
        for network in networks:
            self.graph += network.graph


    def make_edge_geometries(self, vs_geoms_from, vs_geoms_to):
        return [shapely.geometry.LineString([geom_from, geom_to]) for
                                            geom_from, geom_to in
                                            zip(vs_geoms_from, vs_geoms_to)]

    def _ckdnearest(self, gdf_assign, gdf_base):
        """
        see https://gis.stackexchange.com/a/301935

        Parameters
        ----------
        gdf_base : gpd.GeoDataFrame

        gdf_assign : gpd.GeoDataFrame

        Returns
        ----------
        gpd.GeoDataFrame
        """
        # TODO: this should be a util function
        n_assign = np.array(list(gdf_assign.geometry.apply(lambda x: (x.x, x.y))))
        n_base = np.array(list(gdf_base.geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(n_base)
        #TODO: this is distance in whatever units geometry is!
        __, idx = btree.query(n_assign, k=1)
        return gdf_base.iloc[idx].index


    def link_closest_vertices(self, ci_type_assign, ci_type_base):
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

        self.graph.add_edges(zip(gdf_vs_assign.index, ix_match), attributes =
                             {'geometry' : edge_geoms,
                              'ci_type' : ['dependency_'+ ci_type_assign + '_' + ci_type_base],
                              'distance' : 1})


    def _construct_subgraph_from_vs(self, from_ci, to_ci, via_ci=None):
        """
        re-construct to speed up computation
        Note: (re-indexes all vs and es!)
        """
        # vertex selection
        vertex_df = self.graph.get_vertex_dataframe()
        vid_from_ci = vertex_df[vertex_df.ci_type == from_ci].index
        vid_to_ci = vertex_df[vertex_df.ci_type == to_ci].index
        vid_via_ci = vertex_df[vertex_df.ci_type == via_ci].index

        subgraph = self.graph.induced_subgraph(
            vid_from_ci.append(vid_to_ci).append(vid_via_ci),
            implementation='copy_and_delete')

        return subgraph

    def _idmapper_subgraph_to_graph(self, subgraph):
        """given a sub-graph, provide a mapping from the vertex IDs and edge
        IDs of that one to those of the original graph.
        """        
        vs_cis_sub = np.unique(subgraph.vs.get_attribute_values('ci_type'))
        es_cis_sub = np.unique(subgraph.es.get_attribute_values('ci_type'))
        
        vidx_dict = dict()
        eidx_dict = dict()
        
        # TODO: try to get indices w/o looping through it. Evtl. Dataframes --> df[ci_Type].index
        # try with timeit.
        
        for ci in vs_cis_sub:
            vidx_dict.update(dict(zip([vs.index for vs in subgraph.vs(ci_type=ci)], 
                                      [vs.index for vs in self.graph.vs(ci_type=ci)])))
        for ci in es_cis_sub:
            eidx_dict.update(dict(zip([es.index for es in subgraph.es(ci_type=ci)], 
                                      [es.index for es in self.graph.es(ci_type=ci)])))
            
        return {'edges' : eidx_dict, 'vertices' : vidx_dict}

    def _edge_weights_sub(self, subgraph, weights):
        """allow for combination of weights into one weight metric """

        if type(weights) != str:
            weight = np.ones(len(subgraph.es))
            for weight_attr in weights:
                weight*= subgraph.es.get_attribute_values(weight_attr)
        else:
            weight = subgraph.es.get_attribute_values(weights)

        return pd.Series(weight)


    def _shortest_paths_sub(self, subgraph, from_ci, to_ci, weight,
                            mode='out', output='epath'):
        """
        shortest paths from all origin CIs to all destination CIs
        """
        vseq_from_ci = subgraph.vs.select(ci_type=from_ci)
        vseq_to_ci = subgraph.vs.select(ci_type=to_ci)

        # summary dfs
        epaths = pd.DataFrame(index=range(vseq_from_ci[0].index,vseq_from_ci[-1].index+1),
                         columns=range(vseq_to_ci[0].index,vseq_to_ci[-1].index+1))
        weight_sum = pd.DataFrame(index=range(vseq_from_ci[0].index,vseq_from_ci[-1].index+1),
                         columns=range(vseq_to_ci[0].index,vseq_to_ci[-1].index+1))
        
        # actual calcs
        for vs_from in vseq_from_ci:
            epaths.loc[vs_from.index] = subgraph.get_shortest_paths(
                vs_from, vseq_to_ci, weight, mode, output)
            weight_sum.loc[vs_from.index] = [weight.loc[epaths.loc[vs_from.index,vs_to.index]].sum()
                                             for vs_to in vseq_to_ci]
        return epaths, weight_sum.astype(float).replace(0, np.nan)

    def _single_shortest_path_sub(self, epaths, weight_sum):
        """
        get the single shortest path from all origin CIs to their respective
        closest destination CI given edge-paths lists and weights from all
        origins to all destinations
        """

        shortest_path = pd.DataFrame(index=epaths.index,
                                     columns=['to_ID','epath', 'weight'])
        shortest_path.index.name = 'from_ID'

        # get columns of respective shortest paths
        shortest_path.to_ID = weight_sum.idxmin(axis=1)
        # get rows w/ and w/o paths at all
        paths = shortest_path[~np.isnan(shortest_path.to_ID)].index

        # this could be vectorized fully with fancy indexing on numpy arrays
        # --> get positional index locations
        shortest_path['epath'][paths] = [epaths.loc[from_id,to_id]
                                         for from_id, to_id in
                                         zip(paths,shortest_path.loc[paths].to_ID)]
        shortest_path['weight'][paths] = [weight_sum.loc[from_id,to_id]
                                         for from_id, to_id in
                                         zip(paths,shortest_path.loc[paths].to_ID)]

        return shortest_path

    def single_shortest_path(self, from_ci, to_ci, via_ci=None, weights='distance',
                             mode='out', output='epath'):

        # computation on subgraph
        subgraph = self._construct_subgraph_from_vs(from_ci, to_ci, via_ci)
        # combine weights in case more than 1 attribute given
        weight = self._edge_weights_sub(subgraph, weights)
        # all shortest paths
        epaths, weight_sum = self._shortest_paths_sub(subgraph, from_ci, to_ci,
                                                      weight, mode, output)
        # single shortest path
        shortest_path_sub = self._single_shortest_path_sub(epaths, weight_sum)

        # mapper of vs IDs and es IDs back to whole graph
        dict_idxmapper = self._idmapper_subgraph_to_graph(subgraph)

        shortest_path = pd.DataFrame(np.array([shortest_path_sub.to_ID.map(dict_idxmapper['vertices']),
                                               [pd.Series(edgelist).map(dict_idxmapper['edges']) 
                                                for edgelist in shortest_path_sub.epath],
                                               shortest_path_sub.weight]).T,
                                     index=shortest_path_sub.index.map(dict_idxmapper['vertices']),
                                     columns=shortest_path_sub.columns)
        return shortest_path
    
    def plot_multigraph(self,layer_dict,layout):    
        visual_style = {}
        visual_style["vertex_size"] = [layer_dict['vsize'][attr] for attr in self.graph.vs["ci_type"]]
        visual_style["edge_arrow_size"] = 1
        visual_style["edge_color"] = [layer_dict['edge_col'][attr] for attr in self.graph.vs["ci_type"]]
        visual_style["vertex_color"] = [layer_dict['vertex_col'][attr] for attr in self.graph.vs["ci_type"]]
        if layout == "fruchterman_reingold":
            visual_style["layout"] = self.graph.layout("fruchterman_reingold")
        elif layout == 'sugiyama':
            visual_style["layout"] = self.graph.layout_sugiyama(layers=[layer_dict['layers'][attr] for attr in self.graph.vs["ci_type"]])#
        visual_style["edge_curved"] = 0.2
        visual_style["edge_width"] = 1
        
        return ig.plot(self.graph, **visual_style) 
