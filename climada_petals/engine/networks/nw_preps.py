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
#from collections import OrderedDict
#from operator import itemgetter
import logging
import sys
import numpy as np
import rasterio
from rasterio.enums import Resampling

sys.path.insert(1, '/Users/evelynm/trails/src/trails')
import simplify

from climada.util import coordinates as u_coords

LOGGER = logging.getLogger(__name__)

KTOE_TO_MWH = 11630 # conversion factor MWh/ktoe (kilo ton of oil equivalents)
HRS_PER_YEAR = 8760

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
        df[colname] = df.apply(
            lambda row: self._shapely_to_pygeos(row[colname]), axis=1)
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
            # TODO: Check why doesnt work in here
            # nodes['geometry'] = nodes.geometry.apply(lambda geom: geom.centroid)
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
        return self.pygeos_to_shapely(self, network.edges), self.pygeos_to_shapely(self, network.nodes)
        
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
            self.shapely_to_pygeos(self, nodes)
    
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

# =============================================================================
# Functional Data
# =============================================================================

class UtilFunctionalData():
    
    def _resample_res(self, filepath, upscale_factor, nodata):
        
        with rasterio.open(filepath) as dataset:
            # resample data to target shape
            arr = dataset.read(
                out_shape=(dataset.count, int(dataset.height * upscale_factor),
                           int(dataset.width * upscale_factor)), 
                resampling=Resampling.average)
            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / arr.shape[-1]),
                (dataset.height / arr.shape[-2]))

        arr = np.where(arr==nodata, 0, arr)
        arr = arr*(1/upscale_factor)**2
        
        return arr, transform
        
        
    def load_resampled_raster(self, filepath, upscale_factor, nodata=-99999.):

        arr, transform = self._resample_res(filepath, upscale_factor, nodata)
        
        grid = u_coords.raster_to_meshgrid(transform, arr.shape[-1], 
                                           arr.shape[-2])                                               
        gdf = gpd.GeoDataFrame({'counts': arr.squeeze().flatten(), 
                                'geometry': gpd.points_from_xy(
                                    grid[0].flatten(), grid[1].flatten())})
        gdf = gdf[gdf.counts!=0].reset_index(drop=True)
        
        # manual correction for over-estimate after aggregation:
        arr_orig, __ = self._resample_res(filepath, 1, nodata)
        corr_factor = arr_orig.squeeze().flatten().sum() / \
            arr.squeeze().flatten().sum()
        gdf['counts'] = gdf.counts * corr_factor
        
        return gdf



class PowerFunctionalData():
    
    def __init__(self, path_elcons_iea=None, path_elimpexp_iea=None):
        
        self.df_el_cons = pd.DataFrame()
        self.df_el_impexp = pd.DataFrame()
        self.tot_cons_mwh = np.nan
        
        if path_elcons_iea is not None:
            self.df_el_cons = pd.read_csv(path_elcons_iea, skiprows=4) # given in ktoe
        if path_elimpexp_iea is not None:
             self.df_el_impexp = pd.read_csv(path_elimpexp_iea, skiprows=4)

    
    def assign_edemand_iea(self, gdf_pop):
        """Assigns loads (mw) to each people cluster"""
        
        # Country meta-data
        pop_tot = gdf_pop.counts.sum()
        
        per_cap_resid_cons_mwh = self.df_el_cons.iloc[-1]['Residential'] * \
            KTOE_TO_MWH/pop_tot
        per_cap_indust_cons_mwh = self.df_el_cons.iloc[-1]['Industry'] * \
            KTOE_TO_MWH/pop_tot
        per_cap_pubser_cons_mwh = self.df_el_cons.iloc[-1]['Commercial and public services'] * \
            KTOE_TO_MWH/pop_tot
        per_cap_cons_mwh = per_cap_resid_cons_mwh + \
            per_cap_indust_cons_mwh + per_cap_pubser_cons_mwh
        
        # needed for supply calcs
        self.tot_cons_mwh = per_cap_cons_mwh*pop_tot
    
        # add to multinet as loads (MW -> annual demand / hr)
        gdf_pop['el_load_mw'] = \
            gdf_pop.counts*per_cap_cons_mwh/HRS_PER_YEAR
        gdf_pop['el_load_resid_mw'] =  \
            gdf_pop.counts*per_cap_resid_cons_mwh/HRS_PER_YEAR
        gdf_pop['el_load_indust_mw'] =  \
           gdf_pop.counts*per_cap_indust_cons_mwh/HRS_PER_YEAR
        gdf_pop['el_load_pubser_mw'] =  \
            gdf_pop.counts*per_cap_pubser_cons_mwh/HRS_PER_YEAR
        
        return gdf_pop
        
    def assign_esupply_iea(self, gdf_pplants):

        """Assigns generation (mw) to each power plant"""
        # TODO: check for last year in csv (not hardcoded 2017)
        
        # Latest annual Import/Export data from the IEA (2018)
        # imports positive, exports negative sign        
        tot_el_imp_mwh = self.df_el_impexp.iloc[-1]['Imports']*KTOE_TO_MWH
        tot_el_exp_mwh = self.df_el_impexp.iloc[-1]['Exports']*KTOE_TO_MWH
        tot_imp_exp_balance_mwh = tot_el_imp_mwh + tot_el_exp_mwh
        
        # Annual generation (2018): assumed as el. consumption + imp/exp balance
        if np.isnan(self.tot_cons_mwh):
            LOGGER.error('''no total electricity consumption set. 
                         Run assign_edemand_iea() first or provide tot_cons_mwh''')
                         
        tot_el_gen_mwh = self.tot_cons_mwh - tot_imp_exp_balance_mwh
        
        # generation from WRI power plants database (usually incomplete)
        gdf_pplants['estimated_generation_gwh_2017'] = pd.to_numeric(
           gdf_pplants.estimated_generation_gwh_2017, errors='coerce')
        gen_pplants_mwh = gdf_pplants.estimated_generation_gwh_2017*1000
        tot_gen_pplants_mwh = gen_pplants_mwh.sum()
        
        # fill plants with no estimated generation by remainder of country production (2017!)
        gen_unassigned = tot_el_gen_mwh - tot_gen_pplants_mwh
        gen_pplants_mwh[np.isnan(gen_pplants_mwh)] = gen_unassigned/np.isnan(gen_pplants_mwh).sum()
        
        # sanity check
        if gen_pplants_mwh.sum() == tot_el_gen_mwh: 
            LOGGER.info('''estimated annual el. production (IEA) now matches
                        assigned annual el. generation (WRI)''')
        else:
            LOGGER.warning('''estimated el. production from IEA doesn`t match
                           power plant el. generation''')
        
        # add el. generation to network, 
        # add another imp/exp balance node outside of cntry shape
        gdf_pplants['el_gen_mw'] = gen_pplants_mwh/HRS_PER_YEAR
        imp_exp_balance = gpd.GeoDataFrame(
            {'geometry':[shapely.geometry.Point(max(gdf_pplants.geometry.x)+1,
                                                max(gdf_pplants.geometry.y)+1)],
             'name': ['imp_exp_balance'],
             'el_gen_mw': [tot_imp_exp_balance_mwh/HRS_PER_YEAR]})
        
        return gdf_pplants.append(imp_exp_balance, ignore_index=True)

    def assign_linecapa():
        pass

# class NetworkPreprocess():
#     """
#     DF operations to add nodes & edges info, other relevant attr info to gdf
#     built to eventually phase out trails repo code

#     # not yet implemented:
#     # splitting and merging lines where sensible
#     # simplifying structures (loops, curves, deg 2 nodes, etc.)
#     # dropping hanging nodes
#     """

#     @staticmethod
#     def consolidate_ci_attrs(gdf, to_drop=None, to_keep=None):
#         if to_drop:
#             to_drop = [col for col in to_drop if col in gdf.columns]
#             gdf = gdf.drop(to_drop, axis=1)
#         if to_keep:
#             to_keep = [col for col in to_keep if col in gdf.columns]
#             gdf = gdf[to_keep]
#         return gdf

#     @staticmethod
#     def add_endpoints(gdf_edges):
#         """
#         For a gdf where rows represent spatial lines, retrieve coordinates
#         of their endpoints.

#         Parameters
#         ----------
#         gdf_edges

#         Returns
#         --------
#         gdf_edges

#         """
#         gdf_edges[['coords_from','coords_to']] = pd.DataFrame(gdf_edges.apply(
#             lambda row: (row.geometry.coords[0],
#                          row.geometry.coords[-1]), axis=1
#             ).tolist(), index=gdf_edges.index)

#         return gdf_edges

#     def _unique_points(gdf_edges):
#         if ((not hasattr(gdf_edges, 'coords_from')) or
#             (not hasattr(gdf_edges, 'coords_to'))):
#             LOGGER.error('Endpoints are missing. Run add_endpoints() first.')
#             return None
#         else:
#             return gpd.GeoDataFrame(gdf_edges['coords_from'].append(
#                 gdf_edges['coords_to']), columns=['coords']).drop_duplicates().reset_index(drop=True)
#     @staticmethod
#     def _add_ci_type(gdf, ci_type):
#         gdf['ci_type'] = ci_type
#         return gdf

#     @staticmethod
#     def get_nodegdf(gdf_edges):
#         """
#         get a gdf with all unique nodes from the
#         endpoints of a lines-gdf
#         """
#         gdf_nodes = NetworkPreprocess._unique_points(gdf_edges)
#         gdf_nodes['orig_id'] = gdf_nodes.index
#         gdf_nodes['geometry'] = gdf_nodes.apply(
#             lambda row: shapely.geometry.Point(row.coords), axis=1)
#         gdf_nodes['ci_type'] = np.unique(gdf_edges.ci_type)[0]
#         return gdf_nodes

#     @staticmethod
#     def add_topology(gdf_edges, gdf_nodes):
#         node_dict = OrderedDict(gdf_nodes[['coords','orig_id']].values.tolist())
#         gdf_edges['from_id'] = itemgetter(*gdf_edges.coords_from.values.tolist())(node_dict)
#         gdf_edges['to_id'] = itemgetter(*gdf_edges.coords_to.values.tolist())(node_dict)
#         gdf_edges['orig_id'] = gdf_edges.index
#         return gdf_edges

#     @staticmethod
#     def ecols_to_graphorder(gdf_edges):
#         return gdf_edges.reindex(['from_id','to_id'] +
#                                  [x for x in list(gdf_edges)
#                                   if x not in ['from_id','to_id']], axis=1)
#     @staticmethod
#     def vcols_to_graphorder(gdf_nodes):
#         return gdf_nodes.reindex(['name'] +
#                                  [x for x in list(gdf_nodes)
#                                   if x not in ['name']], axis=1)

#     @staticmethod
#     def arrange_gdfs(gdf, type='edges', ci_type=None):
#         """wrapper w/o topology"""
#         #TODO: don't hard-code attrs to keep
#         gdf = NetworkPreprocess._add_ci_type(gdf, ci_type)
#         if not hasattr(gdf, 'orig_id'):
#             gdf['orig_id'] = gdf.index

#         if type == 'edges':
#             gdf = NetworkPreprocess.consolidate_ci_attrs(
#                 gdf, to_keep=['geometry', 'ci_type', 'from_id', 'to_id',
#                                     'orig_id','distance', 'name', 'highway', 'power'])
#             gdf = NetworkPreprocess.ecols_to_graphorder(gdf)

#         elif type == 'nodes':
#             gdf = NetworkPreprocess.consolidate_ci_attrs(
#                 gdf, to_keep=['geometry', 'ci_type', 'orig_id', 'power', 'counts'])
#             gdf['name'] = gdf.orig_id
#             gdf = NetworkPreprocess.vcols_to_graphorder(gdf)

#         return gdf

#     @staticmethod
#     def preprocess_edges_nodes(gdf_edges, ci_type):
#         """complete wrapper"""
#         gdf_edges = NetworkPreprocess.add_endpoints(gdf_edges)
#         gdf_edges = NetworkPreprocess._add_ci_type(gdf_edges, ci_type)
#         gdf_nodes = NetworkPreprocess.get_nodegdf(gdf_edges)
#         gdf_edges = NetworkPreprocess.add_topology(gdf_edges, gdf_nodes)
#         # TODO: don't hard-code this!
#         gdf_edges = NetworkPreprocess.consolidate_ci_attrs(
#             gdf_edges, to_keep=['from_id', 'to_id', 'orig_id', 'geometry',
#                                 'ci_type',  'distance'])
#         gdf_nodes = NetworkPreprocess.consolidate_ci_attrs(
#             gdf_nodes, to_keep=['geometry', 'ci_type', 'coords', 'orig_id'])
#         gdf_edges = NetworkPreprocess.ecols_to_graphorder(gdf_edges)
#         gdf_nodes = NetworkPreprocess.vcols_to_graphorder(gdf_nodes)

#         return gdf_edges, gdf_nodes

#     def add_nodes_to_graph():
#         pass
