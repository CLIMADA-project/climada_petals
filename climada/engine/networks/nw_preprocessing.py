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
import pygeos
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
from collections import OrderedDict
from operator import itemgetter
import logging
import sys

sys.path.insert(1, '/Users/evelynm/trails/src/trails')
import simplify

LOGGER = logging.getLogger(__name__)

class NetworkPreprocess():
    """
    DF operations to add nodes & edges info, other relevant attr info to gdf 
    built to eventually phase out trails repo code
    """
    # not yet implemented:
    # splitting and merging lines where sensible
    # simplifying structures (loops, curves, deg 2 nodes, etc.)
    # dropping hanging nodes
    
    @staticmethod
    def consolidate_ci_attrs(gdf, to_drop=None, to_keep=None):
        if to_drop:
            to_drop = [col for col in to_drop if col in gdf.columns]
            gdf = gdf.drop(to_drop, axis=1)
        if to_keep:
            to_keep = [col for col in to_keep if col in gdf.columns]
            gdf = gdf[to_keep]
        return gdf

    @staticmethod
    def add_endpoints(gdf_edges):
        """
        For a gdf where rows represent spatial lines, retrieve coordinates
        of their endpoints.
        
        Parameters
        ----------
        gdf_edges
        
        Returns
        --------
        gdf_edges
        
        """
        gdf_edges[['coords_from','coords_to']] = pd.DataFrame(gdf_edges.apply(
            lambda row: (row.geometry.coords[0], 
                         row.geometry.coords[-1]), axis=1
            ).tolist(), index=gdf_edges.index)

        return gdf_edges
    
    def _unique_points(gdf_edges):
        if ((not hasattr(gdf_edges, 'coords_from')) or 
            (not hasattr(gdf_edges, 'coords_to'))):
            LOGGER.error('Endpoints are missing. Run add_endpoints() first.')
            return None
        else:
            return gpd.GeoDataFrame(gdf_edges['coords_from'].append(
                gdf_edges['coords_to']), columns=['coords']).drop_duplicates().reset_index(drop=True)
        
    def _add_ci_type(gdf, ci_type):
        gdf['ci_type'] = ci_type
        return gdf
        
    @staticmethod
    def get_nodegdf(gdf_edges):
        """
        get a gdf with all unique nodes from the 
        endpoints of a lines-gdf
        """
        gdf_nodes = NetworkPreprocess._unique_points(gdf_edges)
        gdf_nodes['orig_id'] = gdf_nodes.index
        gdf_nodes['geometry'] = gdf_nodes.apply(
            lambda row: shapely.geometry.Point(row.coords), axis=1)
        gdf_nodes['ci_type'] = np.unique(gdf_edges.ci_type)[0]
        return gdf_nodes

    @staticmethod
    def add_topology(gdf_edges, gdf_nodes):        
        node_dict = OrderedDict(gdf_nodes[['coords','orig_id']].values.tolist())
        gdf_edges['from_id'] = itemgetter(*gdf_edges.coords_from.values.tolist())(node_dict)
        gdf_edges['to_id'] = itemgetter(*gdf_edges.coords_to.values.tolist())(node_dict)
        gdf_edges['orig_id'] = gdf_edges.index
        return gdf_edges

    @staticmethod
    def ecols_to_graphorder(gdf_edges):
        return gdf_edges.reindex(['from_id','to_id'] + 
                                 [x for x in list(gdf_edges) 
                                  if x not in ['from_id','to_id']], axis=1)
    @staticmethod
    def vcols_to_graphorder(gdf_nodes):
        return gdf_nodes.reindex(['orig_id'] + 
                                 [x for x in list(gdf_nodes) 
                                  if x not in ['orig_id']], axis=1)
    
    @staticmethod
    def arrange_gdfs(gdf, type='edges', ci_type=None):
        """wrapper w/o topology"""
        #TODO: don't hard-code attrs to keep
        gdf = NetworkPreprocess._add_ci_type(gdf, ci_type)
        if not hasattr(gdf, 'orig_id'):
            gdf['orig_id'] = gdf.index
        
        if type == 'edges':
            gdf = NetworkPreprocess.consolidate_ci_attrs(
                gdf, to_keep=['geometry', 'ci_type', 'from_id', 'to_id', 
                                    'orig_id','distance', 'name', 'highway', 'power'])
            gdf = NetworkPreprocess.ecols_to_graphorder(gdf)
        
        elif type == 'nodes':
            gdf = NetworkPreprocess.consolidate_ci_attrs(
                gdf, to_keep=['geometry', 'ci_type', 'orig_id', 'power', 'count'])   
            gdf = NetworkPreprocess.vcols_to_graphorder(gdf)
        
        return gdf
    
    @staticmethod
    def preprocess_edges_nodes(gdf_edges, ci_type):
        """complete wrapper"""
        gdf_edges = NetworkPreprocess.add_endpoints(gdf_edges)
        gdf_edges = NetworkPreprocess._add_ci_type(gdf_edges, ci_type)
        gdf_nodes = NetworkPreprocess.get_nodegdf(gdf_edges)
        gdf_edges = NetworkPreprocess.add_topology(gdf_edges, gdf_nodes)
        # TODO: don't hard-code this!
        gdf_edges = NetworkPreprocess.consolidate_ci_attrs(
            gdf_edges, to_keep=['from_id', 'to_id', 'orig_id', 'geometry', 
                                'ci_type',  'distance'])
        gdf_nodes = NetworkPreprocess.consolidate_ci_attrs(
            gdf_nodes, to_keep=['geometry', 'ci_type', 'coords', 'orig_id'])
        gdf_edges = NetworkPreprocess.ecols_to_graphorder(gdf_edges)
        gdf_nodes = NetworkPreprocess.vcols_to_graphorder(gdf_nodes)
        
        return gdf_edges, gdf_nodes

    def add_nodes_to_graph():
        pass
# =============================================================================
# Wrappers from trails repo tailored to CI types
# =============================================================================


class NetworkPreprocessTrails():
    def __init__(self, gdf_edges=None, gdf_nodes=None):
        if gdf_edges is not None:
            edges = gdf_edges.copy()
            self.edges = self.shapely_to_pygeos(self, edges)
        if gdf_nodes is not None:
            nodes = gdf_nodes.copy()
            self.nodes = self.shapely_to_pygeos(self, nodes)
    
    def _pygeos_to_shapely(self, geom):
        """helper: geometry conversion"""
        return shapely.wkt.loads(pygeos.io.to_wkt(geom))
    
    def _shapely_to_pygeos(self, geom):
        """helper: geometry conversion"""
        return pygeos.io.from_wkt(geom.wkt)
    
    @staticmethod
    def pygeos_to_shapely(self, df, colname='geometry'):
        """helper: dataframe conversion"""
        gdf = df.copy()
        shapely_geom = list()
        for geom in gdf[colname]:
            shapely_geom.append(self._pygeos_to_shapely(geom))
        gdf[colname] = shapely_geom
        return gdf
    
    @staticmethod
    def shapely_to_pygeos(self, gdf, colname='geometry'):
        """helper: dataframe conversion"""
        df = pd.DataFrame(gdf)
        df[colname] = df.apply(lambda row: self._shapely_to_pygeos(row[colname]), axis=1)
        return df

    def road_wrapper(self):
        """ individual steps, from flow_model.load_network(): for roads """
        network = simplify.Network(edges=self.edges)
        network = simplify.add_endpoints(network)
        network = simplify.split_edges_at_nodes(network)
        network = simplify.clean_roundabouts(network)
        network = simplify.add_ids(network)
        network = simplify.add_topology(network)
        network = simplify.drop_hanging_nodes(network)
        network = simplify.merge_edges(network)
        network = simplify.reset_ids(network) 
        network = simplify.add_distances(network)
        network = simplify.merge_multilinestrings(network)
        network = simplify.fill_attributes(network)
        network = simplify.add_travel_time(network)
        network.edges = network.edges.rename({'id':'orig_id'},axis=1)
        network.nodes = network.nodes.rename({'id':'orig_id'},axis=1)
        return self.pygeos_to_shapely(self, network.edges), self.pygeos_to_shapely(self, network.nodes)
    
    def power_wrapper(self):
        """ individual steps, from flow_model.load_network(): for power lines  """
        network = simplify.Network(edges=self.edges, nodes=self.nodes)
        network = simplify.add_endpoints(network)
        network = simplify.split_edges_at_nodes(network)
        network = simplify.add_ids(network)
        network = simplify.add_topology(network)
        network = simplify.drop_hanging_nodes(network)
        network = simplify.reset_ids(network) 
        network = simplify.add_distances(network)
        network.edges = network.edges.rename({'id':'orig_id'},axis=1)
        network.nodes = network.nodes.rename({'id':'orig_id'},axis=1)
        return self.pygeos_to_shapely(self,network.edges), self.pygeos_to_shapely(self,network.nodes)

