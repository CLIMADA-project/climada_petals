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

LOGGER = logging.getLogger(__name__)

class Network:
    
    def __init__(self, edges=None, nodes=None):
        """
        gdf_nodes : id identifier needs to be first column of nodes
        gdf_edges : from_id and to_id need to be first two columns of edges
        """
        self.edges = gpd.GeoDataFrame(columns=['from_id', 'to_id', 'ci_type'])
        self.nodes = gpd.GeoDataFrame(columns=['name_id', 'ci_type'])
        
        if edges is not None:
            self.edges = edges
        if nodes is not None:
            self.nodes = nodes
        
        self.ci_type = self._update_ci_types()

    def _update_ci_types(self):
        return np.unique(np.unique(self.edges.ci_type).tolist().append(
                         np.unique(self.nodes.ci_type).tolist()))


class MultiNetwork:
    
    def __init__(self, edges=None, nodes=None, networks=None):
        """
        either edges, nodes or networks=[]
        
        nodes : id identifier needs to be first column of nodes
        edges : from_id and to_id need to be first two columns of edges
        networks : 
        """
        # be more specific than **kwargs with self.__dict__.update(kwargs) 
            
        if networks is not None:
            edges, nodes = self._assemble_from_nws(networks)
    
        self.edges = edges
        self.nodes = nodes
        self.ci_type =  self._update_ci_types()

                        
    def _update_ci_types(self):
        return np.unique(np.unique(self.edges.ci_type).tolist().append(
                         np.unique(self.nodes.ci_type).tolist()))
    
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
    
        return edges, nodes
    
    