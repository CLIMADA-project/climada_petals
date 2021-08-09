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

import logging
import numpy as np
import pandapower as pp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)

KTOE_TO_MWH = 11630 # conversion factor MWh/ktoe (kilo ton of oil equivalents)
HRS_PER_YEAR = 8760


class PowerFlow():
    """
    
    Attributes
    ----------
    pp_net : a pandapower PowerNet instance
    """
    def __init__(self, pp_net =None):
        
        self.pp_net = pp.create_empty_network()

        if pp_net is not None:
            self.pp_net = pp_net
    
    def _select_power_es_vs(self, multinet):
        """select all edges and nodes directly linked to power line entries"""
        
        power_cond_es = multinet.edges.ci_type == "power line"
        power_cond_vs = multinet.nodes.name_id.isin(
            np.unique([multinet.edges[power_cond_es].from_id, 
                       multinet.edges[power_cond_es].to_id]))
        return power_cond_es, power_cond_vs
    
    def fill_pp_powernet(self, multinet, voltage=110, load_cntrl=False, 
                         gen_cntrl=True, max_load=120, parallel_lines=False):
        """
        generate a simple pandapower power network from a multinetwork instance
        containing power lines, power plants and loads (identified as 
        all other nodes connected to power lines that have a demand estimate
        in MW).
        """
        
        power_cond_es, power_cond_vs = self._select_power_es_vs(multinet)
        poweredges = multinet.edges[power_cond_es]
        powernodes = multinet.nodes[power_cond_vs]
        
        LOGGER.info('creating busses.. ')
        # create busses: all nodes in power sub-network
        for __, row in powernodes.iterrows():
            pp.create_bus(self.pp_net, name=f'Bus {row.name_id}', 
                          vn_kv=voltage, type='n')
        
        LOGGER.info('adding power lines...')
        # all powerlines same voltage
        for __, row in poweredges.reset_index().iterrows():
            from_bus = pp.get_element_index(self.pp_net, 'bus', 
                                            name=f'Bus {row.from_id}')
            to_bus = pp.get_element_index(self.pp_net, 'bus', 
                                          name=f'Bus {row.to_id}')
            # TODO: std-type per voltage level --> dict
            if parallel_lines:
                pp.create_line(self.pp_net, from_bus, to_bus, 
                               length_km=row.distance/1000, 
                               std_type='184-AL1/30-ST1A 110.0', 
                               name=f'{row.orig_id}', 
                               parallel=row.parallel_lines)
            else:
                pp.create_line(self.pp_net, from_bus, to_bus, 
                               length_km=row.distance/1000, 
                               std_type='184-AL1/30-ST1A 110.0', 
                               name=f'{row.orig_id}')
        if max_load:
            self.pp_net.line['max_loading_percent'] = max_load

        LOGGER.info('adding generators...')
        # generators (= power plants)
        for _, row in powernodes[powernodes.ci_type=='power plant'].iterrows():
            bus_idx = pp.get_element_index(self.pp_net, "bus", 
                                           name=f'Bus {row.name_id}')
            pp.create_gen(self.pp_net, bus_idx, p_mw=row.el_gen_mw, min_p_mw=0, 
                          max_p_mw=row.el_gen_mw,  vm_pu=1.01, 
                          controllable=gen_cntrl, name=f'{row.orig_id}')
        # add slack (needed for optimization)
        pp.create_gen(self.pp_net, 0, p_mw=0, min_p_mw=-10000, 
                      max_p_mw=0.1, vm_pu=1.01, controllable=True, slack=True)
        
        LOGGER.info('adding loads...')
        # loads (= electricity demands)
        for _, row in powernodes[~np.isnan(powernodes.el_load_mw)].iterrows():
            bus_idx = pp.get_element_index(self.pp_net, "bus", 
                                           name=f'Bus {row.name_id}')
            pp.create_load(self.pp_net, bus_idx, p_mw=row.el_load_mw, 
                           name=f'{row.orig_id}', min_p_mw=0, 
                           max_p_mw=row.el_load_mw,
                           controllable=load_cntrl)
            
        LOGGER.info(f'''filled the PowerNet with {len(self.pp_net.bus)} busses:
                    {len(self.pp_net.line)} lines, {len(self.pp_net.load)} loads
                    and {len(self.pp_net.gen)} generators''')
    
    def _estimate_parallel_lines(self):
        """ 
        """   
        return [x if x > 0 else 1 for x in 
                np.ceil(self.pp_net.res_line.loading_percent/100)]
           
    def run_dc_opf(self, delta=1e-10):
        """run an DC-optimal power flow with pandapower"""
        return pp.rundcopp(self.pp_net, delta=delta) 
            
    def calibrate_lines_flows(self, max_load=120):
        
        if self.pp_net.bus.empty:
            LOGGER.error('''Empty PowerNet. Please provide as PowerFlow(pp_net)
                         or run PowerFlow().fill_pp_powernet()''')
        
        # "unconstrained case": let single lines overflow (loads fixed, no capacity constraints)
        self.pp_net.line['max_loading_percent'] = np.nan
        self.pp_net.line['parallel'] = 1
        self.pp_net.gen['controllable'] = True
        self.pp_net.load['controllable'] = False
        
        # "constrained" case: set line estimates and re-run with capacity constraints
        LOGGER.info('Running DC - OPF to estimate number of parallel lines.')
        self.run_dc_opf()
        self.pp_net.line['parallel']= self._estimate_parallel_lines()
        self.pp_net.line['max_loading_percent']= max_load
    
        try: 
            self.run_dc_opf()
            LOGGER.info('''DC-OPF converged with estimated number of lines and
                        given capacity constraints.
                        Returning results of power flow optimization.''')
        except:
            LOGGER.error('''DC-OPF did not converge. 
                         Consider increasing max_load or manually adjusting 
                         no. of parallel lines''')
        
    def assign_pflow_results(self, multinet):
        """
        assign # of parallel lines, line loading &, powerflow (MW) to power lines
        assign actual supply (MW) to power plants
        assign acutal received supply (MW) to demand nodes (people, etc.) 
        """
        
        power_cond_es, power_cond_vs = self._select_power_es_vs(multinet)
        
        # line results
        multinet.edges['parallel_lines'] = 0
        multinet.edges['line_loading'] = 0
        multinet.edges['powerflow_mw'] = 0     
        multinet.edges.parallel_lines.loc[power_cond_es] = self.pp_net.line.parallel.values
        multinet.edges.line_loading.loc[power_cond_es] = self.pp_net.res_line.loading_percent.values
        multinet.edges.powerflow_mw.loc[power_cond_es] = self.pp_net.res_line.p_from_mw.values
        
        # generation results
        cond_pplant = power_cond_vs & (multinet.nodes.ci_type == 'power plant')
        multinet.nodes['actual_supply_mw'] = 0
        multinet.nodes.actual_supply_mw.loc[cond_pplant] = self.pp_net.res_gen.p_mw[:-1].values
        
        # load results (all nodes connected to a power line w/ a el_load_mw entry)
        cond_load = power_cond_vs & ~np.isnan(multinet.nodes.el_load_mw)
        multinet.nodes.actual_supply_mw.loc[cond_load] = self.pp_net.res_load.p_mw.values
        
        return multinet
    
    # def pflow_stats(self, multinet):
    #     multinet.nodes[~np.isnan(multinet.nodes.el_load_mw)].el_load_mw > (multinet.nodes.actual_supply_mw+10e-5)
        
    def plot_opf_results(self, multinet, var='pflow', outline=None, **kwargs):
        
        fig, ax1 = plt.subplots(1, 1, sharex=(True), sharey=True,figsize=(15, 15),)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        if var=='pflow':
            multinet.edges.plot(
                'powerflow_mw', alpha=1, label='Power FLow (MW)', 
                legend=True, cax=cax, ax=ax1, 
                linewidth=np.log([abs(x)+1 for x in multinet.edges['powerflow_mw']]), 
                vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"))
            
        elif var=='line_load':
            multinet.edges.plot(
                'line_loading', alpha=1, label='Line Loadings (%)', 
                legend=True, cax=cax, ax=ax1, 
                linewidth=np.log([abs(x)+1 for x in multinet.edges['line_loading']]),
                vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"))
            
        if outline is not None:
            outline.boundary.plot(linewidth=0.5, ax=ax1, 
                                  label='Country outline', color='black')
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles, loc='upper left')
        ax1.set_title('DC-OPF result', fontsize=20)
        fig.tight_layout()
