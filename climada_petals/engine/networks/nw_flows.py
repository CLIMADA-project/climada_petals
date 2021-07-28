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
import logging
import numpy as np
import pandas as pd
import pandapower as pp
import shapely

from climada.engine.networks.base import MultiNetwork

LOGGER = logging.getLogger(__name__)

# conversion factor MWh/ktoe (kilo ton of oil equivalents)
KTOE_TO_MWH = 11630
HRS_PER_YEAR = 8760

class PowerFunctionalData():
    
    def __init__(self):
        self.tot_cons_mwh = None
        self.tot_gen_mwh = None
        
    def assign_electricity_demand_supply(self, multinet, path_el_cons_iea, 
                                         path_el_impexp_iea ):
        """Wrapper to assign d & s to nodes in one go. """
        
        multinet = self.assign_edemand(multinet, path_el_cons_iea)
        multinet = self.assign_esupply(multinet, path_el_impexp_iea)
        
        return multinet
         
    
    def assign_edemand(self, multinet, path_el_cons_iea):
        """Assigns loads (mw) to each people node in the network"""
        
        # Country meta-data
        bool_pop = multinet.nodes.ci_type=='people'
        pop_tot = multinet.nodes[bool_pop].counts.sum()
        
        # Latest annual consumption data from the IEA (2018)
        df_el_cons = pd.read_csv(path_el_cons_iea, skiprows=4) # given in ktoe
        
        per_cap_resid_cons_mwh = df_el_cons.iloc[-1]['Residential'] * \
            KTOE_TO_MWH/pop_tot
        per_cap_indust_cons_mwh = df_el_cons.iloc[-1]['Industry'] * \
            KTOE_TO_MWH/pop_tot
        per_cap_pubser_cons_mwh = df_el_cons.iloc[-1]['Commercial and public services'] * \
            KTOE_TO_MWH/pop_tot
        per_cap_cons_mwh = per_cap_resid_cons_mwh + \
            per_cap_indust_cons_mwh + per_cap_pubser_cons_mwh
        
        # needed for supply calcs
        self.tot_cons_mwh = per_cap_cons_mwh*pop_tot
    
        # add to multinet as loads (MW -> annual demand / hr)
        multinet.nodes['el_load_mw'] = \
            multinet.nodes.counts*per_cap_cons_mwh/HRS_PER_YEAR
        multinet.nodes['el_load_resid_mw'] =  \
            multinet.nodes.counts*per_cap_resid_cons_mwh/HRS_PER_YEAR
        multinet.nodes['el_load_indust_mw'] =  \
            multinet.nodes.counts*per_cap_indust_cons_mwh/HRS_PER_YEAR
        multinet.nodes['el_load_pubser_mw'] =  \
            multinet.nodes.counts*per_cap_pubser_cons_mwh/HRS_PER_YEAR
        
        return multinet
        
    def assign_esupply(self, multinet, path_el_impexp_iea):

        """Assigns generation (mw) to each power plant node in the network"""
        
        # Latest annual Import/Export data from the IEA (2018)
        # imports positive, exports negative sign
        df_imp_exp = pd.read_csv(path_el_impexp_iea, skiprows=4)
        
        tot_el_imp_mwh = df_imp_exp.iloc[-1]['Imports']*KTOE_TO_MWH
        tot_el_exp_mwh = df_imp_exp.iloc[-1]['Exports']*KTOE_TO_MWH
        tot_imp_exp_balance_mwh = tot_el_imp_mwh + tot_el_exp_mwh
        
        # Annual generation (2018): assumed as el. consumption + imp/exp balance
        if not self.tot_cons_mwh:
            LOGGER.error('''no total electricity consumption set. 
                         Run assign_edemand() first.''')
                         
        self.tot_el_gen_mwh = self.tot_cons_mwh - tot_imp_exp_balance_mwh
        
        # generation from WRI power plants database (usually incomplete)
        bool_pplants = multinet.nodes.ci_type == 'power plant'
        multinet.nodes['estimated_generation_gwh_2017'] = pd.to_numeric(
           multinet.nodes.estimated_generation_gwh_2017, errors='coerce')
        gen_pplants_mwh = multinet.nodes.estimated_generation_gwh_2017*1000
        tot_gen_pplants_mwh = gen_pplants_mwh.sum()
        
        # fill plants with no estimated generation by remainder of country production (2017!)
        gen_unassigned = self.tot_el_gen_mwh - tot_gen_pplants_mwh
        bool_unassigned_plants = np.isnan(gen_pplants_mwh) & bool_pplants
        gen_pplants_mwh[bool_unassigned_plants] = gen_unassigned/bool_unassigned_plants.sum()
        
        # sanity check
        if gen_pplants_mwh.sum() == self.tot_el_gen_mwh: 
            LOGGER.info('''estimated annual el. production (IEA) now matches
                        assigned annual el. generation (WRI)''')
        else:
            LOGGER.warning('''estimated el. production from IEA doesn`t match
                           power plant el. generation''')
        
        # add el. generation to network, 
        # add another imp/exp balance node outside of cntry shape
        multinet.nodes['el_gen_mw'] = gen_pplants_mwh/HRS_PER_YEAR
        imp_exp_balance = gpd.GeoDataFrame({'geometry':[
                                            shapely.geometry.Point(
                                                max(multinet.nodes[bool_pplants].geometry.x)+1,
                                                max(multinet.nodes[bool_pplants].geometry.y)+1)],
                                            'name': ['imp_exp_balance'],
                                            'el_gen': [tot_imp_exp_balance_mwh/HRS_PER_YEAR]})
        multinet.nodes.append(imp_exp_balance)
        
        return multinet

    def assign_linecapa():
        pass
    

class PowerFlow():
    """
    
    Attributes
    ----------
    multinetwork : networks.base.MultiNetwork
    """
    
    def __init__(self): #(self, multinetwork=None, powernet=None):
        pass
        # TODO: perhaps remove again; make a calculation-only class.
        # multinetwork = MultiNetwork(edges=gpd.GeoDataFrame(columns=['osm_id', 'geometry', 'ci_type']),
        #                                  nodes=gpd.GeoDataFrame(columns=['osm_id', 'geometry', 'ci_type']))
        # self.powernet = pp.create_empty_network()
        
        # if isinstance(multinetwork, MultiNetwork):
        #     multinetwork = multinetwork
        # if isinstance(powernet, pp.PandapowerNet):
        #     self.powernet = pp.pandapowerNet
    
    def make_pp_powernet(self, multinet, voltage=110, load_cntrl=False, 
                         gen_cntrl=True, max_load=None, parallel_lines=None):
        """generate a simple pandapower power network from """
        
        power_cond_es = multinet.edges.ci_type == "power line"
        poweredges = multinet.edges[power_cond_es]
        power_cond_vs = multinet.nodes.name_id.isin(
            np.unique([poweredges.from_id, poweredges.to_id]))
        powernodes = multinet.nodes[power_cond_vs]

        pp_net = pp.create_empty_network()

        # create busses: all nodes in power sub-network
        for __, row in powernodes.iterrows():
            pp.create_bus(pp_net, name=f'Bus {row.name_id}', vn_kv=voltage, 
                          type='n')
        
        # all powerlines same voltage
        for __, row in poweredges.reset_index().iterrows():
            from_bus = pp.get_element_index(pp_net, 'bus', 
                                            name=f'Bus {row.from_id}')
            to_bus = pp.get_element_index(pp_net, 'bus', 
                                          name=f'Bus {row.to_id}')
            # TODO: std-type per voltage level --> dict
            pp.create_line(pp_net, from_bus, to_bus, length_km=row.distance/1000, 
                           std_type='184-AL1/30-ST1A 110.0', 
                           name='Power line {row["edge ID"]}')
        if parallel_lines:
            pp_net.line['parallel'] = parallel_lines

        if max_load:
            pp_net.line['max_loading_percent'] = max_load

                
        # generators (= power plants)
        for _, row in powernodes[powernodes.ci_type=='power plant'].iterrows():
            bus_idx = pp.get_element_index(pp_net, "bus", 
                                           name=f'Bus {row.name_id}')
            pp.create_gen(pp_net, bus_idx, p_mw=row.capacity_mw, min_p_mw=0, 
                          max_p_mw=row.capacity_mw,  vm_pu=1.01, 
                          controllable=gen_cntrl)
        # add slack (needed for optimization)
        pp.create_gen(pp_net, 0, p_mw=0, min_p_mw=-10000, 
                      max_p_mw=0.1, vm_pu=1.01, controllable=True, slack=True)
        
        # loads (= electricity demands)
        for _, row in powernodes[~np.isnan(powernodes.demand_mw)].iterrows():
            bus_idx = pp.get_element_index(pp_net, "bus", 
                                           name=f'Bus {row.name_id}')
            pp.create_load(pp_net, bus_idx, p_mw=row.demand_mw, 
                           name=f'{row.ci_Type} {row.name_id}', 
                           controllable=load_cntrl)
        
        return pp_net
    
    
    def calibrate_parallel_lines_baseflow(self, pp_net_base, max_load=120):
        """ 
        pp_net_base : pp.PandapowerNet
            with load_cntl=False, parallel_lines=None, max_load=None
        """
        
        # run unconstrained DC-OPF
        LOGGER.info('''Running optimal power flow without loading constraints
                    to estimate no. of parallel lines''')
        self.run_dc_opf(pp_net_base)
        parallel_lines = [x if x > 0 else 1 for x in 
                          np.ceil(pp_net_base.res_line.loading_percent/100)]

        pp_net_base.line['parallel']= parallel_lines
        pp_net_base.line['max_load_percent']= max_load
        
        LOGGER.info('Estimated no. of parallel lines, re-run constrained OPF to verify convergence')
        try:
            self.run_dc_opf(pp_net_base)
            return pp_net_base
        except:
            LOGGER.error('''DC OPF did not converge. 
                         Consider increasing max_load or manually adjusting 
                         no. of parallel lines''')
            return None
        
    
    def run_dc_opf(self, pp_net, delta=1e-16):
        """run an """
        return pp.rundcopp(pp_net, delta=delta) 
        
        