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
from climada_petals.engine.networks.nw_utils import LINE_EXPOSURES

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel("INFO")


## Exposures preparation
def gdf_from_network(df_edges_or_nodes, ci_type):
    """Filter a nodes or edges GeoDataFrame by CI type.

    Parameters
    ----------
    df_edges_or_nodes : gpd.GeoDataFrame
        Edges or nodes dataframe with a ``ci_type`` column.
    ci_type : str
        Infrastructure type to select.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of the filtered subset.
    """
    return df_edges_or_nodes[df_edges_or_nodes["ci_type"] == ci_type].copy()


def exposure_from_nodes(network, ci_type, value=None, value_col=None, tag=None):
    """Prepare an Exposures object from nodes of a network.

    Parameters
    ----------
    network : Network
        The network to get the nodes from.
    ci_type : str
        The type of the nodes to get.
    value : int or float, optional
        The value to assign to every exposure point. Takes precedence over
        ``value_col`` when both are provided.  If neither ``value`` nor
        ``value_col`` is given, a default value of ``1`` is used.
    value_col : str, optional
        Column name in the nodes dataframe to use as the exposure value.
        Only used when ``value`` is ``None``.
    tag : str, optional
        A tag to assign to the exposure. Default ``None`` yields tag as
        ``ci_type``.

    Returns
    -------
    Exposures
        An Exposures object containing the nodes of the network as points
        with a value.
    """
    gdf = gdf_from_network(network.nodes, ci_type)
    exp_pnt = Exposures(gdf)
    if value is not None:
        exp_pnt.gdf["value"] = value
    elif value_col is not None:
        exp_pnt.gdf["value"] = gdf[value_col]
    else:
        exp_pnt.gdf["value"] = 1
    if tag is None:
        tag = ci_type
    exp_pnt.description = tag
    exp_pnt.set_lat_lon()
    exp_pnt.check()
    return exp_pnt


def exposure_from_edges(
    network,
    ci_type,
    res,
    disagg_val=1,
    disagg_met=u_lp.DisaggMethod.FIX,
    disagg_col=None,
    tag=None,
    **disagg_kwargs,
):
    """
    Prepare an Exposures object from edges of a network.

    Parameters
    ----------
    network : Network
        The network to get the edges from.
    ci_type : str
        The type of the edges to get.
    res : int
        The resolution at which to disagregate the edges.
    disagg_val : int, optional
        The value to assign to each exposure points upon disaggregation. Default is 1.
    disagg_met : DisaggMethod, optional
        The value assignement method to use for disaggregation. Default is DisaggMethod.FIX.
    disagg_col : str, optional
        The column from the edges data frame to use for value assignement. Default None yields no "value" colum originally present to be used.
    tag : str, optional
        A tag to assign to the exposure. Default None yields tag as ci_type.
    **disagg_kwargs : dict
        Additional keyword arguments to pass to the disaggregation function.

    Returns
    -------
    Exposures
        An Exposures object containing the edges of the network as points with a value.

    """
    gdf = gdf_from_network(network.edges, ci_type)
    if disagg_val is None:
        gdf["value"] = gdf[disagg_col]

    exp_line = Exposures(gdf)
    exp_pnt = u_lp.exp_geom_to_pnt(
        exp_line,
        res=res,
        disagg_val=disagg_val,
        disagg_met=disagg_met,
        to_meters=True,
        **disagg_kwargs,
    )
    if tag is None:
        tag = ci_type
    exp_pnt.description = tag
    exp_pnt.set_lat_lon()
    exp_pnt.check()
    return exp_pnt


class NetworkImpactCalc:
    def __init__(self, impf_set, impf_thresh_set, haz, network, exp_list):
        self.impf_set = impf_set
        self.impf_thresh_set = impf_thresh_set
        self.haz = haz
        self.network = network
        self.exp_list = exp_list

    ## Impact calcs
    @staticmethod
    def calc_point_impacts(haz, exp, impf_set):
        """Impact calculation for a single point exposure.

        Parameters
        ----------
        haz : Hazard
            Hazard object.
        exp : Exposures
            Exposures object with ``impf_<haz_type>`` column.
        impf_set : ImpactFuncSet
            Impact function set.

        Returns
        -------
        Impact
            Impact result with ``imp_mat`` stored.
        """
        imp = ImpactCalc(exp, impf_set, haz)
        imp = imp.impact(save_mat=True)
        return imp

    @staticmethod
    def impacts_to_network(gdf_nodes_or_edges, imp, exp_tag, impf_thresh):
        """Assign impacts to network.

        Parameters
        ----------
        gdf_nodes_or_edges : gpd.GeoDataFrame
            Nodes or edges dataframe to update.
        imp : Impact
            Impact result with ``imp_mat``.
        exp_tag : str
            CI type tag identifying which rows to update.
        impf_thresh : float
            Threshold on direct impact: values **above** this render
            infrastructure dysfunctional.

        Returns
        -------
        gpd.GeoDataFrame
            Updated dataframe with ``func_internal``, ``imp_dir``, and
            ``func_tot`` columns.
        """
        func_states = list(map(int, imp.imp_mat.toarray().flatten() <= impf_thresh))

        gdf_nodes_or_edges.loc[
            gdf_nodes_or_edges.ci_type == exp_tag, "func_internal"
        ] = func_states
        gdf_nodes_or_edges.loc[gdf_nodes_or_edges.ci_type == exp_tag, "imp_dir"] = (
            imp.imp_mat.toarray().flatten()
        )

        gdf_nodes_or_edges["func_tot"] = [
            np.min([func_internal, func_tot])
            for func_internal, func_tot in zip(
                gdf_nodes_or_edges.func_internal, gdf_nodes_or_edges.func_tot
            )
        ]

        return gdf_nodes_or_edges

    def disrupt_network(self):
        """wrapper to disrupt network based on hazard and exposure data."""
        network_disr = cp.deepcopy(self.network)  # create new network object
        imp_dict = {}
        for exp in self.exp_list:
            impf = self.impf_set.get_func(
                haz_type=self.haz.haz_type, fun_id=exp.description
            )
            impf_thresh = self.impf_thresh_set.get(exp.description)
            exp.gdf[f"impf_{self.haz.haz_type}"] = impf.id
            imp = NetworkImpactCalc.calc_point_impacts(
                self.haz, exp, ImpactFuncSet([impf])
            )
            # propagate impacts to network
            if exp.description in LINE_EXPOSURES:
                imp = u_lp.impact_pnt_agg(
                    imp, exp.gdf, u_lp.AggMethod.SUM
                )  # reaggregate impacts if originally disaggregated
                network_disr.edges = NetworkImpactCalc.impacts_to_network(
                    network_disr.edges, imp, exp.description, impf_thresh
                )
            else:
                network_disr.nodes = NetworkImpactCalc.impacts_to_network(
                    network_disr.nodes, imp, exp.description, impf_thresh
                )
            imp_dict[exp.description] = imp
        # gc.collect()
        ## TODO eventually concat impact matrices and return them?
        return network_disr, imp_dict
