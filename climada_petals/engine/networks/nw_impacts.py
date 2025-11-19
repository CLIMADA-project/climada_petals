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
import pandas as pd
import copy as cp
import climada.util.lines_polys_handler as u_lp
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.engine.impact_calc import ImpactCalc


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')

class NetworkImpactCalc():
    def __init__(self, impf_set, haz, network):
        #super().__init__(exp, impf_set, haz)
        self.impf_set = impf_set
        self.haz = haz
        self.network = network
    ## Exposures preparation
    def gdf_from_network(df_edges_or_nodes, ci_type):
        return df_edges_or_nodes[df_edges_or_nodes['ci_type']==ci_type]

    def exposure_from_nodes(gdf, value=1, tag=None):
        exp_pnt = Exposures(gdf)
        exp_pnt.gdf['value'] = value
        exp_pnt.description = tag if tag is not None else gdf.ci_type.iloc[0]

        exp_pnt.set_lat_lon()
        exp_pnt.check()
        return exp_pnt

    def exposure_from_edges(gdf, res, disagg_met=u_lp.DisaggMethod.FIX, disagg_val=1, tag=None):
        exp_line = Exposures(gdf)
        if not disagg_val:
            disagg_val = res
        exp_pnt = u_lp.exp_geom_to_pnt(exp_line, res=res, to_meters=True,
                                       disagg_met=disagg_met, disagg_val=disagg_val)
        exp_pnt.description = tag if tag is not None else gdf.ci_type.iloc[0]

        exp_pnt.set_lat_lon()
        exp_pnt.check()
        return exp_pnt

    def make_network_exposures(network, ci_types=None, res_orig = 500):
        exp_list = []
        if ci_types is None:
            ci_types = network.nodes.ci_type.unique()
        for ci_type in ci_types:
            if ci_type == 'road':
                gdf_roads = gdf_from_network(network.edges, 'road')
                #assign value field as distance for disaggregation
                gdf_roads['value'] = gdf_roads['distance']
                disagg_val_road = None #use None to use value field as disag value
                #disagg_val_road = res_orig # damage fraction on y-axis
                exp = exposure_from_edges(gdf_from_network(network.edges, 'road'),
                                               res=res_orig, disagg_val=disagg_val_road)
            elif ci_type == 'people':
                gdf_ppl = gdf_from_network(network.nodes, 'people')
                exp = exposure_from_nodes(gdf_ppl, value=gdf_ppl.counts)
            else:
                gdf = gdf_from_network(network.nodes, ci_type)
                exp = exposure_from_nodes(gdf)
            exp_list.append(exp)
        return exp_list

    ## Impact calcs
    def calc_point_impacts(haz, exp, impf_set):
        """Impact calulation for a single point exposure."""
        imp = ImpactCalc(exp, impf_set, haz)
        imp = imp.impact(save_mat=True)
        return imp

    def impacts_to_network(imp, exp_tag, impf_thresh_set, ci_network_disr):
        """Assign impacts to network."""
        func_states = list(
                map(int, imp.imp_mat.toarray().flatten()<=impf_thresh_set.getThresh(exp_tag)))

        if exp_tag == 'road':
            ci_network_disr.edges.loc[ci_network_disr.edges.ci_type=='road',
                                      'func_internal'] = func_states
            ci_network_disr.edges.loc[ci_network_disr.edges.ci_type=='road',
                                      'imp_dir'] = imp.imp_mat.toarray().flatten()

        else:
            ci_network_disr.nodes.loc[
                    ci_network_disr.nodes.ci_type==exp_tag, 'func_internal'] = func_states
            ci_network_disr.nodes.loc[
                    ci_network_disr.nodes.ci_type==exp_tag, 'imp_dir'] = imp.imp_mat.toarray().flatten()

        ci_network_disr.edges['func_tot'] = [np.min([func_internal, func_tot]) for
                                              func_internal, func_tot in zip(
                                                  ci_network_disr.edges.func_internal,
                                                  ci_network_disr.edges.func_tot)]
        ci_network_disr.nodes['func_tot'] = [np.min([func_internal, func_tot]) for
                                             func_internal, func_tot in zip(
                                                 ci_network_disr.nodes.func_internal,
                                                 ci_network_disr.nodes.func_tot)]

        return ci_network_disr


    def disrupt_network(network, haz, impf_thresh_set, ci_types=None, res_disagg=500):
        """wrapper to disrupt network based on hazard and exposure data."""
        network_disr = cp.deepcopy(network)
        exp_list = make_network_exposures(network_disr, ci_types, res_orig=res_disagg)

        for exp in exp_list:
            impf = impf_thresh_set.getImpf(exp.description)
            exp.gdf[f"impf_{haz.haz_type}"] = impf.id
            imp = calc_point_impacts(haz, exp, ImpactFuncSet([impf]))
            if exp.description in ['road']:
                imp = u_lp.impact_pnt_agg(imp, exp.gdf, u_lp.AggMethod.SUM)
            network_disr = impacts_to_network(imp, exp.description, impf_thresh_set, network_disr)
            del imp
            del exp
        gc.collect()
        return network_disr