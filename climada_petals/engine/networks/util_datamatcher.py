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

Helper functions to assign demand, supply data etc. to gdfs

"""

import geopandas as gpd
import pandas as pd
import shapely
import logging
import numpy as np
import rasterio
from rasterio.enums import Resampling

from climada.util import coordinates as u_coords

LOGGER = logging.getLogger(__name__)

KTOE_TO_MWH = 11630 # conversion factor MWh/ktoe (kilo ton of oil equivalents)
TJ_TO_MWH = 277.778 # comversion factor MWh/TJ
HRS_PER_YEAR = 8760

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
        
    def assign_esupply_iea(self, gdf_pplants, path_elimpexp_iea, path_elcons_iea, unit='ktoe'):
        """Assigns generation (mw) to each power plant"""
        
        if unit=='ktoe':
            conv_fact = KTOE_TO_MWH
        elif unit == 'TJ':
            conv_fact = TJ_TO_MWH
        else:
            raise KeyError('Invalid unit entered')
        
        df_el_impexp = pd.read_csv(path_elimpexp_iea, skiprows=4)
        df_el_cons = pd.read_csv(path_elcons_iea, skiprows=4)
                
        # Latest annual Import/Export data from the IEA (2018)
        # imports positive, exports negative sign        
        tot_el_imp_mwh = df_el_impexp.iloc[-1]['Imports']*conv_fact
        tot_el_exp_mwh = df_el_impexp.iloc[-1]['Exports']*conv_fact
        tot_imp_exp_balance_mwh = tot_el_imp_mwh + tot_el_exp_mwh
        
        # Latest annual consumption data from the IEA (2018)
        tot_cons_mwh = df_el_cons.iloc[-1][
            ['Residential', 'Industry','Commercial and public services']
            ].sum()*conv_fact
        
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
    
    
    def balance_el_generation(self, gdf_pplants, per_cap_cons, pop_no):
        
        gdf_pplants.estimated_generation_gwh_2017 = pd.to_numeric(
            gdf_pplants.estimated_generation_gwh_2017, errors='coerce')
        
        tot_cons = per_cap_cons*pop_no
        
        imp_exp_balance = tot_cons - gdf_pplants.estimated_generation_gwh_2017.sum()
        
        gdf_pplants['el_gen_mw'] = gdf_pplants.estimated_generation_gwh_2017*1000/HRS_PER_YEAR
        
        imp_exp_balance = gpd.GeoDataFrame(
            {'geometry':[shapely.geometry.Point(max(gdf_pplants.geometry.x)+1,
                                                max(gdf_pplants.geometry.y)+1)],
             'name': ['imp_exp_balance'],
             'el_gen_mw': [imp_exp_balance/HRS_PER_YEAR],
             'ci_type' : 'power plant'
             })
        return  gdf_pplants.append(imp_exp_balance, ignore_index=True)
    
    def assign_linecapa():
        pass

