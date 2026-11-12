"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Direct impacts of a hazard on the components of a network

Network components are turned into CLIMADA exposures (``exposure_from_nodes``,
``exposure_from_edges``), their impacts are computed with CLIMADA's
``ImpactCalc``, and the impacts are written back to the network as functional
states (``NetworkImpactCalc.disrupt_network``). The disrupted network can then
be used as the input of the cascade (``NetworkCalcs.cascade``).
"""

import copy as cp
import logging

import numpy as np

import climada.util.lines_polys_handler as u_lp
from climada.engine.impact_calc import ImpactCalc
from climada.entity import Exposures, ImpactFuncSet
from climada_petals.engine.networks.nw_utils import LINE_EXPOSURES

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Exposures from network components
# =============================================================================


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
    """Prepare an Exposures object from the nodes of a network.

    Each node of type ``ci_type`` becomes one exposure point. The exposure
    keeps the node index and columns (including ``ci_type``).

    Parameters
    ----------
    network : Network
        The network to get the nodes from.
    ci_type : str
        The type of the nodes to get.
    value : int or float, optional
        The value to assign to every exposure point. Takes precedence over
        ``value_col`` when both are provided. If neither ``value`` nor
        ``value_col`` is given, a default value of ``1`` is used.
    value_col : str, optional
        Column name in the nodes dataframe to use as the exposure value
        (e.g. ``"counts"`` for people). Only used when ``value`` is ``None``.
    tag : str, optional
        Tag stored as ``Exposures.description``. ``disrupt_network`` uses it
        to select the impact function (``id``) and the functionality
        threshold. Default ``None`` uses ``ci_type``.

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
):
    """Prepare an Exposures object from the edges of a network.

    Each edge of type ``ci_type`` is disaggregated into points spaced about
    ``res`` metres apart (at least one point per edge), with
    ``climada.util.lines_polys_handler.exp_geom_to_pnt``. The exposure has a
    two-level index: the edge index, then the point number along the edge.

    Parameters
    ----------
    network : Network
        The network to get the edges from.
    ci_type : str
        The type of the edges to get.
    res : float
        Distance between the points of the disaggregation, in metres.
    disagg_val : int or float, optional
        Value assigned to each edge before disaggregation. Default is 1.
        Set it to ``None`` to take the value from ``disagg_col``.
    disagg_met : DisaggMethod, optional
        How the edge value is assigned to its points: ``FIX`` (default)
        gives every point the edge value, ``DIV`` divides it equally over the
        points.
    disagg_col : str, optional
        Column of the edges dataframe holding the edge values. Only used when
        ``disagg_val`` is ``None``.
    tag : str, optional
        Tag stored as ``Exposures.description``. ``disrupt_network`` uses it
        to select the impact function (``id``) and the functionality
        threshold. Default ``None`` uses ``ci_type``.

    Returns
    -------
    Exposures
        An Exposures object containing the edges of the network as points
        with a value.
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
    )
    if tag is None:
        tag = ci_type
    exp_pnt.description = tag
    exp_pnt.check()
    return exp_pnt


# =============================================================================
# Impacts on the network
# =============================================================================


class NetworkImpactCalc:
    """Compute direct hazard impacts on a network

    Computes the impact of a hazard on exposures built from the network
    (``exposure_from_nodes``, ``exposure_from_edges``) and writes it back to
    the network as functional states, ready for a cascade (``NetworkCalcs.cascade``).
    The network given is not modified: ``disrupt_network`` works on a copy.

    Attributes
    ----------
    impf_set : ImpactFuncSet
        Impact functions; one per exposure type. The impact functions' ``id``
        must match the exposure tags in exp_list and the ``haz_type``
        must match the hazard type from the hazard object.
    impf_thresh_set : dict
        Functionality threshold per exposure tag: components whose direct
        impact is above the threshold become dysfunctional.
    haz : Hazard
        Hazard with a single event, in the CRS of the network.
    network : Network
        Network the exposures were built from, with functional states
        (``func_internal``, ``func_tot``, ``imp_dir``), e.g. the network
        after the base-state cascade.
    exp_list : list of Exposures
        Exposures built from ``network`` with ``exposure_from_nodes`` or
        ``exposure_from_edges``.
    """

    def __init__(self, impf_set, impf_thresh_set, haz, network, exp_list):
        """Create a network impact calculator

        Parameters
        ----------
        impf_set : ImpactFuncSet
            Impact functions; one per exposure, with ``id`` equal to the
            exposure tag (``Exposures.description``) and ``haz_type`` equal to
            ``haz.haz_type``.
        impf_thresh_set : dict
            ``{tag: threshold}``: functionality threshold per exposure tag.
        haz : Hazard
            Hazard with a single event, in the CRS of the network.
        network : Network
            Network the exposures were built from.
        exp_list : list of Exposures
            Exposures built from ``network``.
        """
        self.impf_set = impf_set
        self.impf_thresh_set = impf_thresh_set
        self.haz = haz
        self.network = network
        self.exp_list = exp_list

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
            Threshold on direct impact: impact values above this render
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
        """Compute the direct impacts and write them to a copy of the network

        For each exposure of ``exp_list``: the impact is computed with the
        impact function of its tag; for edge exposures, the point impacts are
        summed per edge. The network rows of the exposure get

        - ``imp_dir``: the direct impact,
        - ``func_internal``: 1 if ``imp_dir`` is at most the threshold of the
          tag, else 0,
        - ``func_tot``: ``min(func_internal, func_tot)``, so a component that
          already failed (e.g. in the base-state cascade) stays failed.

        Run the cascade on the returned network to obtain the indirect
        impacts (``NetworkCalcs(network_disr, dep_table).cascade()``).

        Returns
        -------
        network_disr : Network
            Deep copy of ``network`` with the updated functional states.
        imp_dict : dict
            ``{tag: Impact}``: the impact on each exposure (per edge for edge
            exposures).
        """
        network_disr = cp.deepcopy(self.network)
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
            if exp.description in LINE_EXPOSURES:
                # re-aggregate the point impacts per edge
                imp = u_lp.impact_pnt_agg(imp, exp.gdf, u_lp.AggMethod.SUM)
                network_disr.edges = NetworkImpactCalc.impacts_to_network(
                    network_disr.edges, imp, exp.description, impf_thresh
                )
            else:
                network_disr.nodes = NetworkImpactCalc.impacts_to_network(
                    network_disr.nodes, imp, exp.description, impf_thresh
                )
            imp_dict[exp.description] = imp
        return network_disr, imp_dict
