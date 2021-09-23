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

"""Make network base classes (data containers)"""

import logging
import numpy as np
import geopandas as gpd
import igraph as ig
import matplotlib.pyplot as plt
import contextily as ctx

LOGGER = logging.getLogger(__name__)

class Network:
    
    def __init__(self, edges=None, nodes=None, graph=None):
        """
        gdf_nodes : id identifier needs to be first column of nodes
        gdf_edges : from_id and to_id need to be first two columns of edges
        """
        self.edges = gpd.GeoDataFrame(columns=['from_id', 'to_id', 'ci_type'])
        self.nodes = gpd.GeoDataFrame(columns=['name_id', 'ci_type'])
        
        if graph is not None:
            edges, nodes = self._assemble_from_graph(graph)
            
        if edges is not None:
            self.edges = gpd.GeoDataFrame(edges)
        if nodes is not None:
            self.nodes = gpd.GeoDataFrame(nodes)
        
        self._update_func_level()
        self.ci_type = self._update_ci_types()        
    
    def _update_ci_types(self):
        return np.unique(np.unique(self.edges.ci_type).tolist().append(
                         np.unique(self.nodes.ci_type).tolist()))
    
    def _update_func_level(self):
        if not hasattr(self.edges, 'func_level'):
            self.edges['func_level'] = 1
        if not hasattr(self.nodes, 'func_level'):
            self.nodes['func_level'] = 1
        
        self.edges.func_level[np.isnan(self.edges.func_level)] = 1
        self.nodes.func_level[np.isnan(self.nodes.func_level)] = 1
        

    def _assemble_from_graph(self, graph):
        
        edges = gpd.GeoDataFrame(graph.get_edge_dataframe().rename(
            {'source':'from_id', 'target':'to_id'}, axis=1))
        nodes = gpd.GeoDataFrame(graph.get_vertex_dataframe().reset_index(
                ).rename({'vertex ID':'name_id'}, axis=1))
        
        return edges, nodes

class MultiNetwork:
    
    def __init__(self, edges=None, nodes=None, networks=None, graphs=None):
        """
        either edges, nodes or networks=[]
        
        nodes : id identifier needs to be first column of nodes
        edges : from_id and to_id need to be first two columns of edges
        networks : networks.base.Network instances
        graphs : igraph.Graph instances
        """
            
        if networks is not None:
            edges, nodes = self._assemble_from_nws(networks)
        
        if graphs is not None:
            edges, nodes = self._assemble_from_graphs(graphs)
    
        if not hasattr(edges, 'orig_id'):
            edges = self._add_orig_id(edges)
        if not hasattr(nodes, 'orig_id'):
            nodes = self._add_orig_id(nodes)
            
        self.edges = edges
        self.nodes = nodes
        self.ci_type =  self._update_ci_types()
        self._update_func_level()

    
    def _add_orig_id(self, gdf):
        gdf['orig_id'] = range(len(gdf))
              
    def _update_ci_types(self):
        return np.unique(np.unique(self.edges.ci_type).tolist().append(
                         np.unique(self.nodes.ci_type).tolist()))
    def _update_func_level(self):
        if not hasattr(self.edges, 'func_level'):
            self.edges['func_level'] = 1
        if not hasattr(self.nodes, 'func_level'):
            self.nodes['func_level'] = 1
        
        self.edges.func_level[np.isnan(self.edges.func_level)] = 1
        self.nodes.func_level[np.isnan(self.nodes.func_level)] = 1
        
    def _assemble_from_nws(self, networks):
        
        edges = gpd.GeoDataFrame(columns=['from_id', 'to_id', 'ci_type'])
        nodes = gpd.GeoDataFrame(columns=['name_id', 'ci_type'])
        id_counter_nodes = 0
        
        for network in networks:
            edge_gdf = network.edges.reset_index(drop=True)
            node_gdf = network.nodes.reset_index(drop=True)
            edge_gdf['from_id'] = edge_gdf['from_id']  + id_counter_nodes
            edge_gdf['to_id'] = edge_gdf['to_id']  + id_counter_nodes
            node_gdf['name_id'] = range(id_counter_nodes, 
                                        id_counter_nodes+len(node_gdf))
            id_counter_nodes+=len(node_gdf)
            edges = edges.append(edge_gdf)
            nodes = nodes.append(node_gdf)
    
        return edges.reset_index(drop=True), nodes.reset_index(drop=True)
    
    def _assemble_from_graphs(self, graphs):
        
        graph = ig.Graph()
        
        for g in graphs:
            graph += g
        
        edges = gpd.GeoDataFrame(graph.get_edge_dataframe().rename(
            {'source':'from_id', 'target':'to_id'}, axis=1))
        nodes = gpd.GeoDataFrame(graph.get_vertex_dataframe().reset_index(
                ).rename({'vertex ID':'name_id'}, axis=1))
        
        return edges, nodes
    
    def plot_cis(self, ci_types=[], **kwargs):
        
        if not ci_types:
            ci_types = self.ci_type
        
        colors = kwargs.get('colors')
        if not colors:
            colors = ['brown', 'red', 'black', 'green', 'blue', 'orange', 
                      'pink', 'white'][:len(ci_types)]
        labels = kwargs.get('labels')
        if not labels:
            labels=ci_types
            
        ax = self.edges[self.edges.ci_type==ci_types[0]].append(
            self.nodes[self.nodes.ci_type==ci_types[0]]
            ).set_crs(epsg=4326).to_crs(epsg=3857).plot(
                figsize=(15, 15), alpha=1, markersize=40, color='yellow', 
                        edgecolor='yellow', label=labels[0])

        for ci_type, color, label in zip(ci_types[1:], colors, labels[1:]):
            self.edges[self.edges.ci_type==ci_type].append(
            self.nodes[self.nodes.ci_type==ci_type]
            ).set_crs(epsg=4326).to_crs(epsg=3857).plot(
                ax=ax, figsize=(15, 15), alpha=1, markersize=40, color=color, 
                edgecolor=color, label=label)
        
        handles, labels = ax.get_legend_handles_labels()
        # manually define patch for airport s
        # patch = mpatches.Patch(color='pink', label='airport')
        # handles.append(patch) 
        ax.legend(handles=handles, loc='upper left')
        ax.set_title(kwargs.get('title'), fontsize=25)
        ctx.add_basemap(ax)
        
        return plt.show()
    