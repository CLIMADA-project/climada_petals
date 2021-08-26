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
import igraph as ig

from climada_petals.engine.networks.nw_calcs import MultiGraphCalcs, GraphMaker

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

    
    def fill_pp_powernet(self, multinet, voltage=110, load_cntrl=False, 
                         gen_cntrl=True, max_load=120, parallel_lines=False):
        """
        generate a simple pandapower power network from a multinetwork instance
        containing power lines, power plants and loads (identified as 
        all other nodes connected to power lines that have a demand estimate
        in MW).
        """
        
        power_cond_es = multinet.edges.ci_type == "power line"
        power_cond_vs = (multinet.nodes.ci_type == 'power line') | (multinet.nodes.ci_type == 'power plant')
        
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
                               parallel=row.parallel_lines,
                               in_service=(row.func_level==1))
            else:
                pp.create_line(self.pp_net, from_bus, to_bus, 
                               length_km=row.distance/1000, 
                               std_type='184-AL1/30-ST1A 110.0', 
                               name=f'{row.orig_id}',
                               in_service=(row.func_level==1))
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
        for _, row in powernodes[powernodes.el_load_mw >0].iterrows():
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
    
    
    def create_costfunc_loadmax(self):
        """create a cost function that maximizes the power supply at loads """
        pp.create_poly_cost(self.pp_net, 0, 'load', cp1_eur_per_mw=-1)
        for ix in self.pp_net.load.index:
            pp.create_pwl_cost(
                self.pp_net, ix, "load", 
                [[self.pp_net.load.min_p_mw.at[ix], 
                  self.pp_net.load.max_p_mw.at[ix], -1]])
    
    
    def assign_pflow_results(self, multinet):
        """
        assign # of parallel lines, line loading &, powerflow (MW) to power lines
        assign actual supply (MW) to power plants
        assign acutal received supply (MW) to demand nodes (people, etc.) 
        """
        
        cond_pline = multinet.edges.ci_type == "power line"
        cond_pplant = multinet.nodes.ci_type == 'power plant'
        cond_load = (multinet.nodes.el_load_mw > 0) & (multinet.nodes.ci_type=='power line')

        # add empty vars:
        multinet.edges[['parallel_lines','line_loading','powerflow_mw']] = 0
        multinet.nodes['actual_supply_mw'] = 0
        
        # line results 
        multinet.edges.parallel_lines.loc[cond_pline] = self.pp_net.line.parallel.values
        multinet.edges.powerflow_mw.loc[cond_pline] = self.pp_net.res_line.p_from_mw.values
        # replace NaNs by 0 (happens only where pflow = 0)
        multinet.edges.line_loading.loc[cond_pline] = self.pp_net.res_line.loading_percent.fillna(0)
        # generation results (less the slack variable)
        multinet.nodes.actual_supply_mw.loc[cond_pplant] = self.pp_net.res_gen.p_mw[:-1].values
        
        # load results; also update functionality level (at loads only)
        multinet.nodes.actual_supply_mw.loc[cond_load] = self.pp_net.res_load.p_mw.values
        multinet.nodes.func_level.loc[cond_load] = multinet.nodes.actual_supply_mw.loc[cond_load] / multinet.nodes.el_load_mw.loc[cond_load]

        return multinet
    
    def pflow_stats(self):
        supply_load = self.pp_net.res_load.p_mw.sum()
        demand_load = self.pp_net.load.p_mw.sum()
        supply_gen = self.pp_net.res_gen.p_mw.sum()
        max_loading = max(self.pp_net.res_line.loading_percent)
        
        print(f'''The difference between supplied ({int(supply_load)} MW) and 
              demanded power({int(demand_load)} MW) at loads is {int(supply_load-demand_load)} MW, 
              corresponding to {(supply_load-demand_load)/demand_load*100} % of total demand.''')
        print(f'Supplied power at generators is {int(supply_gen)} MW')
        print(f'Max. line loading is {int(max_loading)} %')
        
        
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


class PowerCascades(MultiGraphCalcs):
    
    def __init__(self, nw_or_graph):
        """
        nw_or_graph : instance of networks.base.Network or .MultiNetwork or
            igraph.Graph
        """
        
        if isinstance(nw_or_graph, ig.Graph):
            self.graph = nw_or_graph
        else:
            return GraphMaker.__init__(self, nw_or_graph)

        
    def assign_edemand_pline(self):
        """
        assign per-capita derived electricity consumption from 
        people nodes to the upstream power node.
        """
        self._assign_to_parent(edge_type = 'dependency_people_power', 
            varname = 'el_load_mw')

    
    def get_power_access_level(self, ci_type):
        """
        for ci_type dependent on power, get the power access level 
        (corresponding to the functionality level of its upstream power node)
        """
        access_level = self._get_var_from_parent(
            edge_type = f'dependency_{ci_type}_power', 
            varname = 'func_level')
        self._assign_to_child(edge_type =f'dependency_{ci_type}_power', 
            varname = 'power_access_level', varlist=access_level)
        
    
    def get_actual_power_supplied(self, ci_type):
                
        func_level_parent = self._get_var_from_parent(
            edge_type = f'dependency_{ci_type}_power', 
            varname = 'func_level')
        
        load_parent = self._get_var_from_parent(
            edge_type = f'dependency_{ci_type}_power', 
            varname = 'el_load_mw')
        
        self._assign_to_child(edge_type = f'dependency_{ci_type}_power', 
            varname = 'actual_supply_mw', 
            varlist=[a*b for a,b in zip(func_level_parent,load_parent)])
 
    
class HealthCascades(MultiGraphCalcs):
    
    def update_health_paths(self):
        es = self.select_edges('dependency_people_health')
        edgeweights = self._make_edgeweights('distance','people', 'health',
                                                'road')
        for edge in es:
            path = self.get_shortest_paths(self.graph.vs[edge.source],
                                           self.graph.vs[edge.target], 
                                           edgeweights, mode='out', 
                                           output='epath')
            edge['distance'] = self.get_path_distance(*path)
    
    def get_health_access_level(self, pow_thresh=0.8, dist_thresh=100000, func_thresh=0.7):
        """
        
        """
        vs_ppl = self.select_nodes('people')
        for vx in vs_ppl:
            edge_list = vx.incident()
            bool_access_health = []
            for edge in edge_list:
                bool_access_health.append(
                    (edge['ci_type'] == 'dependency_people_health') & 
                    (edge['distance'] < dist_thresh) & 
                    (self.graph.vs[edge.target]['func_level'] > func_thresh) & 
                    (self.graph.vs[edge.target]['power_access_level'] > pow_thresh))
            if sum(bool_access_health)>0:
                vx['health_access_level'] = 1
            else:
                vx['health_access_level'] = 0

        