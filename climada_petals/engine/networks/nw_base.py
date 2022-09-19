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

---

Make network base classes (data containers)
"""

import logging
import numpy as np
import geopandas as gpd
import igraph as ig

LOGGER = logging.getLogger(__name__)

class Network:
    
    def __init__(self, 
                 edges=gpd.GeoDataFrame(), 
                 nodes=gpd.GeoDataFrame(), 
                 ci_type=None):
        """
        initialize a network object given edges and nodes dataframes
        """
        if edges.empty:
            edges = gpd.GeoDataFrame(
                columns=['from_id', 'to_id', 'orig_id', 'ci_type'])
        if not hasattr(edges, 'orig_id'):
            edges['orig_id'] = self._add_orig_id(edges)
    
        if nodes.empty:
            nodes = gpd.GeoDataFrame(
                columns=[['name_id', 'orig_id', 'ci_type']])    
        if not hasattr(nodes, 'orig_id'):
            nodes['orig_id'] = self._add_orig_id(nodes)
    
        if not ci_type:
            ci_type = self._update_ci_types(edges, nodes)
        
        self.edges = edges
        self.nodes = nodes
        self.ci_type = ci_type

          
    @classmethod
    def from_nws(cls, networks):
        """
        make one network object out of several network objects
        """
        edges = gpd.GeoDataFrame(columns=['from_id', 'to_id', 'ci_type'])
        nodes = gpd.GeoDataFrame(columns=['name_id', 'ci_type'])
        
        id_counter_nodes = 0
        
        for nw in networks:
            edge_gdf = nw.edges.reset_index(drop=True)
            node_gdf = nw.nodes.reset_index(drop=True)
            edge_gdf['from_id'] = edge_gdf['from_id']  + id_counter_nodes
            edge_gdf['to_id'] = edge_gdf['to_id']  + id_counter_nodes
            node_gdf['name_id'] = range(id_counter_nodes, 
                                        id_counter_nodes+len(node_gdf))
            id_counter_nodes+=len(node_gdf)
            edges = edges.append(edge_gdf)
            nodes = nodes.append(node_gdf)
    
        return Network(edges=edges.reset_index(drop=True), 
                       nodes=nodes.reset_index(drop=True))

    @classmethod
    def from_graphs(cls, graphs):
        """
        make one network object out of several graph objects
        """
        graph = ig.Graph(directed=graphs[0].is_directed())
        for g in graphs:
            graph += g
        
        edges = gpd.GeoDataFrame(graph.get_edge_dataframe().rename(
            {'source':'from_id', 'target':'to_id'}, axis=1))
        nodes = gpd.GeoDataFrame(graph.get_vertex_dataframe().reset_index(
                ).rename({'vertex ID':'name_id'}, axis=1))           
        
        return Network(edges=edges, nodes=nodes)


    @classmethod
    def _update_ci_types(cls, edges, nodes):
        return np.unique(np.unique(edges.ci_type).tolist().append(
                         np.unique(nodes.ci_type).tolist()))

    @classmethod
    def _add_orig_id(cls, gdf):
        return range(len(gdf))
    
    
    def initialize_funcstates(self):
        """ 
        
        """
        self.edges[['func_internal','func_tot']] = 1
        self.nodes[['func_internal','func_tot']] = 1
        self.edges['imp_dir'] = 0
        self.nodes['imp_dir'] = 0
        
    def initialize_capacity(self, source, target):
        self.nodes[f'capacity_{source}_{target}'] = 0
        self.nodes.loc[self.nodes['ci_type']==f'{source}',f'capacity_{source}_{target}'] = 1
        self.nodes.loc[self.nodes['ci_type']==f'{target}',f'capacity_{source}_{target}'] = -1
        
    def initialize_supply(self, source):
        self.nodes[f'actual_supply_{source}_people'] = 0
        self.nodes.loc[self.nodes['ci_type']=='people',f'actual_supply_{source}_people'] = 1
        
    
