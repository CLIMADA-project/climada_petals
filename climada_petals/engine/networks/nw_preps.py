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
    
    def assign_edemand_iea(self, gdf_people, path_elcons_iea):
        """Assigns loads (mw) to each people cluster"""
        
        df_el_cons = pd.read_csv(path_elcons_iea, skiprows=4)
        
        # Country meta-data
        pop_tot = gdf_people.counts.sum()
        
        # convert annual cons. data to loads (MW = annual demand / hr)
        per_cap_resid_mw = df_el_cons.iloc[-1]['Residential'] * \
            KTOE_TO_MWH/pop_tot/HRS_PER_YEAR
        per_cap_indust_mw = df_el_cons.iloc[-1]['Industry'] * \
            KTOE_TO_MWH/pop_tot/HRS_PER_YEAR
        per_cap_pubser_mw = df_el_cons.iloc[-1]['Commercial and public services'] * \
            KTOE_TO_MWH/pop_tot/HRS_PER_YEAR
        per_cap_mw = per_cap_resid_mw + per_cap_indust_mw + per_cap_pubser_mw
        
        for var, var_per_cap in zip(
                ['el_load_mw', 'el_load_resid_mw','el_load_indust_mw', 
                 'el_load_pubser_mw'],
                [per_cap_mw, per_cap_resid_mw, per_cap_indust_mw, 
                 per_cap_pubser_mw]):
            gdf_people[var] = gdf_people.counts * var_per_cap
       
        return gdf_people
        
    def assign_esupply_iea(self, gdf_pplants, path_elimpexp_iea, path_elcons_iea):
        """Assigns generation (mw) to each power plant"""
        
        df_el_impexp = pd.read_csv(path_elimpexp_iea, skiprows=4)
        df_el_cons = pd.read_csv(path_elcons_iea, skiprows=4)
                
        # Latest annual Import/Export data from the IEA (2018)
        # imports positive, exports negative sign        
        tot_el_imp_mwh = df_el_impexp.iloc[-1]['Imports']*KTOE_TO_MWH
        tot_el_exp_mwh = df_el_impexp.iloc[-1]['Exports']*KTOE_TO_MWH
        tot_imp_exp_balance_mwh = tot_el_imp_mwh + tot_el_exp_mwh
        
        # Latest annual consumption data from the IEA (2018)
        tot_cons_mwh = df_el_cons.iloc[-1][
            ['Residential', 'Industry','Commercial and public services']
            ].sum()*KTOE_TO_MWH
        
        # Annual generation (2018): assumed as el. consumption + imp/exp balance
        tot_el_gen_mwh = tot_cons_mwh - tot_imp_exp_balance_mwh
        
        # generation from WRI power plants database (usually incomplete)
        # TODO: check for last year in csv (not hardcoded 2017)
        gdf_pplants.estimated_generation_gwh_2017 = pd.to_numeric(
            gdf_pplants.estimated_generation_gwh_2017, errors='coerce')
            
        gen_pplants_mwh = gdf_pplants.estimated_generation_gwh_2017*1000
        
        # fill plants with no estimated generation by remainder of country production (2017!)
        gen_unassigned = tot_el_gen_mwh - gen_pplants_mwh.sum()
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
        # TODO: split into sub-function
        gdf_pplants['el_gen_mw'] = gen_pplants_mwh/HRS_PER_YEAR
        imp_exp_balance = gpd.GeoDataFrame(
            {'geometry':[shapely.geometry.Point(max(gdf_pplants.geometry.x)+1,
                                                max(gdf_pplants.geometry.y)+1)],
             'name': ['imp_exp_balance'],
             'el_gen_mw': [tot_imp_exp_balance_mwh/HRS_PER_YEAR],
             'ci_type' : 'power plant'
             })
        
        return  gdf_pplants.append(imp_exp_balance, ignore_index=True)

    def assign_linecapa():
        pass

