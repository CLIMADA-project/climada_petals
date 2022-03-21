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
import geopandas as gpd
import pygeos
import pandas as pd
import shapely
import logging
import sys
import numpy as np

sys.path.insert(1, '/Users/evelynm/trails/src/trails')
import simplify


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')

# =============================================================================
# Topological simplifications
# =============================================================================

class NetworkPreprocess():
    """
    Preprocessing Baseclass
    Takes gdfs, returns pre-processed, formatted gdfs nodes, edges that
    have correct network topology.
    
    Note
    ----
    This network preprocessing class is relying greatly on functionalities
    developed within the GitHub trails project:
    (https://github.com/BenDickens/trails) . It has dependencies that are
    not by default in CLIMADA; plan is to potentially replace those gradually.
    """
    def __init__(self, ci_type):
        self.ci_type = ci_type
            

    @staticmethod
    def pygeos_to_shapely(self, df, colname='geometry'):
        """helper: dataframe conversion"""
        gdf = df.copy()
        gdf[colname] = gdf.apply(lambda row: pygeos.to_shapely(row[colname]),
                                 axis=1)
        return gpd.GeoDataFrame(gdf)

    @staticmethod
    def shapely_to_pygeos(self, gdf, colname='geometry'):
        """helper: dataframe conversion"""
        df = pd.DataFrame(gdf)
        df[colname] = df.apply(
            lambda row: pygeos.from_shapely(row[colname]), axis=1)
        return df

    def _ecols_to_graphorder(self, edges):
        return edges.reindex(['from_id', 'to_id'] +
                             [x for x in list(edges) 
                              if x not in ['from_id', 'to_id']], axis=1)

    def _vcols_to_graphorder(self, nodes):
        return nodes.reindex(['name_id'] + [x for x in list(nodes)
                             if x not in ['name_id']], axis=1)

    def _add_ci_type(self, edges=None, nodes=None):
        if not edges.empty:
            edges['ci_type'] = self.ci_type
        if not nodes.empty:            
            nodes['ci_type'] = self.ci_type
        
        return edges, nodes

        
    def _simplify_network(self, edges=None, nodes=None):
        
        # trails.simplify runs in pygeos, not shapely. convert.
        if not edges.empty:
            edges = self.shapely_to_pygeos(self, edges)
        if not nodes.empty:
            nodes = self.shapely_to_pygeos(self, nodes)

        network = simplify.Network(edges=edges, nodes=nodes)
        if not edges.empty:
            network = simplify.add_endpoints(network)
            network = simplify.split_edges_at_nodes(network)
        network = simplify.add_ids(network)
        if not edges.empty:
            network = simplify.add_topology(network)
            network = simplify.drop_hanging_nodes(network)
            network = simplify.merge_edges(network)
            network = simplify.reset_ids(network)
            network = simplify.add_distances(network)
            network = simplify.merge_multilinestrings(network)
        
        # convert back to shapely
        if not edges.empty:
            edges = self.pygeos_to_shapely(self, network.edges)
        nodes = self.pygeos_to_shapely(self, network.nodes)
            
        return edges, nodes
        
    def preprocess(self, gdf_edges=None, gdf_nodes=None):
        """
        standard wrapper end-to-end. Takes edge and node dataframes,
        simplifies them, adds topology (ids), adds CI attributes
        and puts cols in correct
        order for igraph to read them in as graph.
        
        Parameters
        ----------
        gdf_edges : gpd.GeoDataFrame
        gdf_nodes : gpd.GeoDataFrame
        
        Returns
        -------
        edges : gpd.GeoDataFrame
        nodes : gpd.GeoDataFrame
        """
        
        edges = gpd.GeoDataFrame(columns=['osm_id', 'geometry'])
        nodes = gpd.GeoDataFrame(columns=['osm_id', 'geometry'])
        
        if isinstance(gdf_edges, gpd.GeoDataFrame):
            edges = gdf_edges.copy()
        if isinstance(gdf_nodes, gpd.GeoDataFrame):
            nodes = gdf_nodes.copy()
            
        edges, nodes = self._simplify_network(edges, nodes)
        edges = edges.rename({'id': 'orig_id', 'source':'data_source'}, axis=1)
        nodes = nodes.rename({'id': 'orig_id'}, axis=1)
        nodes['name_id'] = nodes.orig_id
        edges, nodes = self._add_ci_type(edges, nodes)
        edges = self._ecols_to_graphorder(edges)
        nodes = self._vcols_to_graphorder(nodes)

        return edges, nodes

    @staticmethod
    def close_gaps(self, df, tolerance=0.0005):
        """Close gaps in LineString geometry where it should be contiguous.
        Snaps both lines to a centroid of a gap in between.
        0.0005 (° if lat/lon) ca 55m lat
        """
        geom = df.geometry.values.data
        coords = pygeos.get_coordinates(geom)
        indices = pygeos.get_num_coordinates(geom)
    
        # generate a list of start and end coordinates and create point geometries
        edges = [0]
        i = 0
        for ind in indices:
            ix = i + ind
            edges.append(ix - 1)
            edges.append(ix)
            i = ix
        edges = edges[:-1]
        points = pygeos.points(np.unique(coords[edges], axis=0))
    
        buffered = pygeos.buffer(points, tolerance)
    
        dissolved = pygeos.union_all(buffered)
    
        exploded = [
            pygeos.get_geometry(dissolved, i)
            for i in range(pygeos.get_num_geometries(dissolved))
        ]
    
        centroids = pygeos.centroid(exploded)
    
        snapped = pygeos.snap(geom, pygeos.union_all(centroids), tolerance)
    
        return snapped

class RoadPreprocess(NetworkPreprocess):
    
    def __init__(self):
        self.ci_type = 'road'
    
    def _simplify_network(self, edges=None, nodes=None):
        """ overrides _simplify_network() method from parent class """

        if not edges.empty:
            edges = self.shapely_to_pygeos(self, edges)

        if nodes.empty:
            nodes = None #simplify cannot handle empty df for nodes, only None
        else:
            self.shapely_to_pygeos(self, nodes)
        network = simplify.Network(edges=edges, nodes=nodes)
        #network = simplify.clean_roundabouts(network)
        network = simplify.add_endpoints(network)
        network = simplify.split_edges_at_nodes(network)
        network = simplify.add_ids(network)
        network = simplify.add_topology(network)
        network = simplify.drop_hanging_nodes(network)
        network = simplify.merge_edges(network)
        network.edges = simplify.drop_duplicate_geometries(network.edges, keep='first') 
        network = simplify.reset_ids(network)
        network = simplify.add_distances(network)
        network = simplify.merge_multilinestrings(network)

        return self.pygeos_to_shapely(self, network.edges), self.pygeos_to_shapely(self, network.nodes)

class PowerlinePreprocess(NetworkPreprocess):
   
    def __init__(self):
        self.ci_type = 'power line'
    
    def _simplify_network(self, edges=None, nodes=None):
        """ overrides _simplify_network() method from parent class """

        if not edges.empty:
            edges = self.shapely_to_pygeos(self, edges)
        if nodes.empty:
            nodes = None #simplify cannot handle empty df for nodes, only None
        else:
            nodes = self.shapely_to_pygeos(self, nodes)
    
        network = simplify.Network(edges=edges, nodes=nodes)
        network = simplify.add_endpoints(network)
        network = simplify.split_edges_at_nodes(network)
        network = simplify.add_ids(network)
        network = simplify.add_topology(network)
        network = simplify.drop_hanging_nodes(network)
        network = simplify.merge_edges(network)
        network = simplify.reset_ids(network)
        network = simplify.add_distances(network)

        return self.pygeos_to_shapely(self, network.edges), self.pygeos_to_shapely(self, network.nodes)
