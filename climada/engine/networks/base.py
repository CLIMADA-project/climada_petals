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
    
    def __init__(self, gdf_edges=None, gdf_nodes=None):
        """
        gdf_nodes : id identifier needs to be first column of nodes
        gdf_edges : from_id and to_id need to be first two columns of edges
        """
        self.edges = gpd.GeoDataFrame()
        self.nodes = gpd.GeoDataFrame()
        
        if gdf_edges is not None:
            self.edges = gdf_edges
            self.ci_type = np.unique(gdf_edges.ci_type).tolist()
        else:
            self.ci_type = np.unique(gdf_nodes.ci_type).tolist()
            
        if gdf_nodes is not None:
            self.nodes = gdf_nodes
        
class MultiNetwork:
    
    def __init__(self, *networks):
        """
        gdf_nodes : id identifier needs to be first column of nodes
        gdf_edges : from_id and to_id need to be first two columns of edges
        """
        self.edges = gpd.GeoDataFrame()
        self.nodes = gpd.GeoDataFrame()
        self.ci_type = []
              
        for network in networks:
            self.edges = self.edges.append(network.edges.reset_index(drop=True))
            self.nodes = self.nodes.append(network.nodes.reset_index(drop=True))
            self.ci_type.append(*network.ci_type)

