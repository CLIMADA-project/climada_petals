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
import pandas as pd

import climada.util.lines_polys_handler as u_lp
from climada.engine.impact_calc import ImpactCalc
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada_petals.engine.networks.nw_utils import (
    ADDED_COL,
    FUNCSTATE_COLS,
    _added_mask,
    _edge_lengths_metres,
)

LOGGER = logging.getLogger(__name__)

#: rules deciding whether an edge fails (argument edge_rule of
#: NetworkImpactCalc): "max_ratio" compares the largest damage ratio of the
#: points of the edge with the threshold (a ratio between 0 and 1);
#: "damaged_length" compares the damaged length of the edge, in metres, with
#: the threshold (in metres)
EDGE_RULES = ("max_ratio", "damaged_length")


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


def _row_mask(mask, gdf, kind):
    """Check a ``mask`` argument and return it as a boolean Series on ``gdf``.

    Parameters
    ----------
    mask : pd.Series or array-like of bool
        A Series must have the same index as ``gdf``; an array the same length.
    gdf : gpd.GeoDataFrame
        Nodes or edges of the network.
    kind : str
        ``"nodes"`` or ``"edges"``, for the error messages.

    Returns
    -------
    pd.Series
        Boolean Series with the index of ``gdf``.
    """
    if isinstance(mask, pd.Series):
        if not mask.index.equals(gdf.index):
            raise ValueError(
                f"mask must have the same index as the network {kind} "
                f"(e.g. a condition on network.{kind})."
            )
        values = mask.to_numpy()
    else:
        values = np.asarray(mask)
        if values.shape != (len(gdf),):
            raise ValueError(
                f"mask must have one value per row of the network {kind} "
                f"({len(gdf)}), got shape {values.shape}."
            )
    if values.dtype != bool:
        raise ValueError(f"mask must be boolean, got dtype {values.dtype}.")
    return pd.Series(values, index=gdf.index)


def _select_components(gdf, ci_type, kind, mask=None):
    """Rows of ``gdf`` of type ``ci_type``, optionally restricted by ``mask``.

    Raises a ``ValueError`` if no row is selected.
    """
    selected = gdf["ci_type"] == ci_type
    if not selected.any():
        raise ValueError(f"The network has no {kind} of ci_type '{ci_type}'.")
    if mask is not None:
        selected &= _row_mask(mask, gdf, kind)
        if not selected.any():
            raise ValueError(f"mask selects no {kind} of ci_type '{ci_type}'.")
    return gdf[selected].copy()


def _check_column(gdf, col, arg, kind):
    """Raise if the column ``col`` given as argument ``arg`` is not in ``gdf``."""
    if col not in gdf.columns:
        raise ValueError(f"{arg}='{col}' is not a column of the network {kind}.")


def exposure_from_nodes(
    network, ci_type, value=None, value_col=None, tag=None, mask=None
):
    """Prepare an Exposures object from the nodes of a network.

    Each node of type ``ci_type`` (optionally restricted by ``mask``) becomes
    one exposure point. The exposure keeps the node index and columns
    (including ``ci_type``).

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
    mask : pd.Series or array-like of bool, optional
        Selects a subset of the nodes of type ``ci_type``, e.g. to give
        hospitals and clinics (both ``"healthcare"``) different impact
        functions: one exposure per subset, each with its own ``tag``. A
        Series must have the index of ``network.nodes``, an array its length.
        Default ``None`` selects all nodes of type ``ci_type``.

    Returns
    -------
    Exposures
        An Exposures object containing the nodes of the network as points
        with a value.

    Raises
    ------
    ValueError
        If no node is selected, if ``mask`` is invalid, or if ``value_col``
        is not a column of the nodes.
    """
    if value_col is not None:
        _check_column(network.nodes, value_col, "value_col", "nodes")
    gdf = _select_components(network.nodes, ci_type, "nodes", mask)
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
    disagg_val=None,
    disagg_met=u_lp.DisaggMethod.FIX,
    disagg_col=None,
    tag=None,
    mask=None,
    include_added=True,
):
    """Prepare an Exposures object from the edges of a network.

    Each edge of type ``ci_type`` (optionally restricted by ``mask``) is
    disaggregated into points spaced about ``res`` metres apart (at least one
    point per edge), with
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
        Value assigned to each edge before disaggregation. Takes precedence
        over ``disagg_col`` when both are provided. If neither
        ``disagg_val`` nor ``disagg_col`` is given, a value of ``1`` is used.
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
    mask : pd.Series or array-like of bool, optional
        Selects a subset of the edges of type ``ci_type``. A Series must have
        the index of ``network.edges``, an array its length. Default ``None``
        selects all edges of type ``ci_type``.
    include_added : bool, optional
        Whether the edges added by the module (column ``nw_utils.ADDED_COL``:
        links from ``add_physical_links``, ``merge_clusters`` /
        ``link_clusters``) are exposed. Default ``True``: they stand for the
        access between components and can fail like the other edges of
        ``ci_type``. Their geometry is a straight line, not the real path, so
        long links (e.g. built with an infinite ``thresh_dist``) may cross the
        hazard where no road exists. With ``False`` they are left out and keep
        their functional state. ``make_network_stat`` never counts them in
        the infrastructure statistics.

    Returns
    -------
    Exposures
        An Exposures object containing the edges of the network as points
        with a value.

    Raises
    ------
    ValueError
        If no edge is selected, if ``mask`` is invalid, or if ``disagg_col``
        is not a column of the edges.

    Examples
    --------
    Main roads and tracks with different impact functions and thresholds,
    using an OSM ``highway`` column:

    >>> main = network.edges["highway"].isin(["motorway", "trunk", "primary"])
    >>> exp_main = exposure_from_edges(
    ...     network, "road", res=500, mask=main, tag="road_main"
    ... )
    >>> exp_track = exposure_from_edges(
    ...     network, "road", res=500, mask=~main, tag="road_track"
    ... )

    The impact function set then needs impact functions with ids
    ``"road_main"`` and ``"road_track"``, and the thresholds both keys.
    """

    if disagg_col is not None:
        _check_column(network.edges, disagg_col, "disagg_col", "edges")
    gdf = _select_components(network.edges, ci_type, "edges", mask)
    if not include_added:
        gdf = gdf[~_added_mask(gdf)].copy()
        if gdf.empty:
            raise ValueError(
                f"All selected edges of ci_type '{ci_type}' were added by the "
                f"module (column '{ADDED_COL}'). Use include_added=True to "
                "expose them."
            )
    if disagg_val is None:
        if disagg_col is None:
            disagg_val = 1
        else:
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


def _exposure_rows(exp):
    """Network rows an exposure was built from.

    Parameters
    ----------
    exp : Exposures
        Exposure from ``exposure_from_nodes`` or ``exposure_from_edges``.

    Returns
    -------
    kind : str
        ``"edges"`` for exposures from ``exposure_from_edges`` (two-level
        index: edge label, point number), ``"nodes"`` otherwise.
    labels : pd.Index
        Labels of the network rows, in the order of the exposure.
    """
    if isinstance(exp.gdf.index, pd.MultiIndex):
        return "edges", exp.gdf.index.get_level_values(0).unique()
    return "nodes", exp.gdf.index


def _check_exposure_rows(exp, tag, kind, labels, gdf):
    """Raise if the exposure was not built from rows of ``gdf``.

    Every row label of the exposure must be a row of ``gdf`` (network nodes
    or edges) with the same ``ci_type`` as in the exposure.

    Parameters
    ----------
    exp : Exposures
        Exposure from ``exposure_from_nodes`` or ``exposure_from_edges``.
    tag : str
        Tag of the exposure, for the error messages.
    kind : str
        ``"nodes"`` or ``"edges"``.
    labels : pd.Index
        Row labels of the exposure (see ``_exposure_rows``).
    gdf : gpd.GeoDataFrame
        Nodes or edges of the network.
    """
    missing = labels[~labels.isin(gdf.index)]
    if len(missing) > 0:
        raise ValueError(
            f"Exposure '{tag}': {len(missing)} of its {kind} are not in the "
            f"network (e.g. row label {missing[0]} of network.{kind}). Build the "
            "exposures from the network given to NetworkImpactCalc."
        )
    if "ci_type" not in exp.gdf.columns:
        raise ValueError(
            f"Exposure '{tag}' has no ci_type column: build it with "
            f"exposure_from_{kind}."
        )
    exp_ci_type = exp.gdf["ci_type"].groupby(level=0, sort=False).first()
    nw_ci_type = gdf.loc[exp_ci_type.index, "ci_type"]
    differs = nw_ci_type != exp_ci_type
    if differs.any():
        label = exp_ci_type.index[differs][0]
        raise ValueError(
            f"Exposure '{tag}': {kind[:-1]} {label} has ci_type "
            f"'{nw_ci_type.loc[label]}' in the network but "
            f"'{exp_ci_type.loc[label]}' in the exposure. Build the exposures "
            "from the network given to NetworkImpactCalc."
        )


def _damage_ratio(haz, exp, impf_set):
    """Damage ratio at each point of an exposure.

    The damage ratio is the impact for an exposure value of 1: the mean
    damage degree (MDD) times the percentage of affected assets (PAA), both
    read from the impact function at the hazard intensity of the point, times
    the hazard fraction at the point. It is between 0 and 1. It is computed
    with ``ImpactCalc`` on a copy of the exposure whose value is 1 at every
    point, so that it does not depend on the exposure value (and is
    defined at points with value 0).

    Parameters
    ----------
    haz : Hazard
        Hazard with a single event.
    exp : Exposures
        Exposure with its ``impf_<haz_type>`` column set.
    impf_set : ImpactFuncSet
        Impact function set containing the impact function of the exposure.

    Returns
    -------
    pd.Series
        Damage ratio per point, with the index of ``exp.gdf``.
    """
    exp_unit = exp.copy()
    exp_unit.gdf["value"] = 1.0
    imp_unit = ImpactCalc(exp_unit, impf_set, haz).impact(save_mat=True)
    return pd.Series(imp_unit.imp_mat.toarray().ravel(), index=exp.gdf.index)


def _impacts_to_network(gdf, labels, imp_dir, func_internal):
    """Write direct impacts and functional states to rows of ``gdf`` (in place).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Nodes or edges of the network, with the columns ``FUNCSTATE_COLS``.
    labels : pd.Index
        Row labels to update; only these rows are changed.
    imp_dir : array-like
        Direct impact per row, in the order of ``labels``.
    func_internal : array-like
        Functional state per row (1 functional, 0 not), in the order of
        ``labels``. ``func_tot`` becomes ``min(func_internal, func_tot)``.
    """
    col_func_internal, col_func_tot, col_imp_dir = FUNCSTATE_COLS
    func_internal = np.asarray(func_internal)
    gdf.loc[labels, col_imp_dir] = np.asarray(imp_dir)
    gdf.loc[labels, col_func_internal] = func_internal
    gdf.loc[labels, col_func_tot] = np.minimum(
        gdf.loc[labels, col_func_tot].to_numpy(), func_internal
    )


class NetworkImpactCalc:
    """Compute direct hazard impacts on a network

    Computes the impact of a hazard on exposures built from the network
    (``exposure_from_nodes``, ``exposure_from_edges``) and writes it back to
    the network as functional states, ready for a cascade (``NetworkCalcs.cascade``).
    The network given is not modified: ``disrupt_network`` works on a copy.

    Attributes
    ----------
    impf_set : ImpactFuncSet
        Impact functions, one for each exposure of ``exp_list``. The impact
        function of an exposure is the one whose ``id`` equals the tag of the
        exposure (``Exposures.description``) and whose ``haz_type`` equals
        ``haz.haz_type``. To use different impact functions for
        components of the same ``ci_type`` (e.g. main roads and tracks),
        build one exposure per group with the ``mask`` argument of
        ``exposure_from_nodes`` / ``exposure_from_edges``, and give each
        exposure its own ``tag``.
    impf_thresh_set : dict
        ``{tag: threshold}``: functionality threshold of each exposure. For a
        node exposure, and for an edge exposure with the edge rule
        ``"max_ratio"``, the threshold is a damage ratio between 0 and 1. For
        an edge exposure with the edge rule ``"damaged_length"``, it is a
        length in metres. A component above the threshold of its exposure
        becomes dysfunctional (see ``disrupt_network``).
    edge_rule : str or dict
        Rule deciding whether an edge fails, ``"max_ratio"`` or
        ``"damaged_length"`` (see ``disrupt_network``): one rule for all edge
        exposures, or ``{tag: rule}`` for some edge exposures (the others use
        ``"max_ratio"``).
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

    def __init__(
        self, impf_set, impf_thresh_set, haz, network, exp_list, edge_rule="max_ratio"
    ):
        """Create a network impact calculator

        Parameters
        ----------
        impf_set : ImpactFuncSet
            Impact functions, one for each exposure of ``exp_list``. The
            impact function of an exposure is the one whose ``id`` equals the
            tag of the exposure (``Exposures.description``) and whose
            ``haz_type`` equals ``haz.haz_type``.
        impf_thresh_set : dict
            ``{tag: threshold}``: functionality threshold of each exposure: a
            damage ratio between 0 and 1 for node exposures and for edge
            exposures with the edge rule ``"max_ratio"``, a length in metres
            for edge exposures with the edge rule ``"damaged_length"``.
        haz : Hazard
            Hazard with a single event, in the CRS of the network.
        network : Network
            Network the exposures were built from.
        exp_list : list of Exposures
            Exposures built from ``network``.
        edge_rule : str or dict, optional
            Rule deciding whether an edge fails (see ``disrupt_network``):
            ``"max_ratio"`` (default) or ``"damaged_length"``, for all edge
            exposures; or ``{tag: rule}`` to set the rule of some edge
            exposures, the others using ``"max_ratio"`` (e.g.
            ``{"road": "damaged_length"}``).
        """
        self.impf_set = impf_set
        self.impf_thresh_set = impf_thresh_set
        self.haz = haz
        self.network = network
        self.exp_list = exp_list
        self.edge_rule = edge_rule

    def _edge_rule_of(self, tag):
        """Edge rule of the edge exposure with the tag ``tag``."""
        if isinstance(self.edge_rule, dict):
            return self.edge_rule.get(tag, "max_ratio")
        return self.edge_rule

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

    def _check_inputs(self):
        """Check the inputs of ``disrupt_network`` before any computation."""
        if self.haz.size != 1:
            raise ValueError(
                f"disrupt_network needs a hazard with one event, got "
                f"{self.haz.size}. Select one, e.g. haz.select(event_id=[...])."
            )
        for kind, gdf in (
            ("nodes", self.network.nodes),
            ("edges", self.network.edges),
        ):
            missing = [col for col in FUNCSTATE_COLS if col not in gdf.columns]
            if missing:
                raise ValueError(
                    f"The network {kind} have no functional states "
                    f"({', '.join(missing)}). Initialise them first with "
                    "network.initialize_funcstates(), or run the base-state "
                    "cascade."
                )

        tags = [exp.description for exp in self.exp_list]
        duplicated = sorted({tag for tag in tags if tags.count(tag) > 1})
        if duplicated:
            raise ValueError(
                f"Several exposures have the tag(s) {duplicated}: each exposure "
                "needs its own tag (argument tag of exposure_from_nodes / "
                "exposure_from_edges)."
            )

        haz_type = self.haz.haz_type
        rows = {"nodes": [], "edges": []}
        for exp in self.exp_list:
            tag = exp.description
            impf = self.impf_set.get_func(haz_type=haz_type, fun_id=tag)
            if not isinstance(impf, ImpactFunc):
                raise ValueError(
                    f"impf_set has no impact function with id '{tag}' and "
                    f"haz_type '{haz_type}' (exposure tag '{tag}')."
                )
            if tag not in self.impf_thresh_set:
                raise ValueError(
                    "impf_thresh_set has no functionality threshold for the "
                    f"exposure tag '{tag}'."
                )
            kind, labels = _exposure_rows(exp)
            _check_exposure_rows(exp, tag, kind, labels, getattr(self.network, kind))
            rows[kind].append(pd.Series(tag, index=labels))

        # edge_rule must name known rules; as a dict, its keys must be tags of
        # edge exposures
        if isinstance(self.edge_rule, dict):
            rules = self.edge_rule
            edge_tags = {
                exp.description
                for exp in self.exp_list
                if _exposure_rows(exp)[0] == "edges"
            }
            not_edges = sorted(set(rules) - edge_tags)
            if not_edges:
                raise ValueError(
                    f"edge_rule names {not_edges}, which are not tags of edge "
                    "exposures in exp_list."
                )
        else:
            rules = {"all edge exposures": self.edge_rule}
        for name, rule in rules.items():
            if rule not in EDGE_RULES:
                raise ValueError(
                    f"Unknown edge_rule '{rule}' for {name}: use one of "
                    f"{list(EDGE_RULES)}."
                )

        # each node and each edge can be in one exposure only
        for kind, tag_series in rows.items():
            if not tag_series:
                continue
            tag_by_row = pd.concat(tag_series)
            shared = tag_by_row.index.duplicated(keep=False)
            if shared.any():
                label = tag_by_row.index[shared][0]
                first, second = tag_by_row.loc[[label]].iloc[:2]
                item = kind[:-1]  # "node" or "edge"
                raise ValueError(
                    f"The exposures '{first}' and '{second}' both contain {item} "
                    f"{label} (row label in network.{kind}): each {item} can be "
                    "in one exposure only (use complementary masks)."
                )

    def disrupt_network(self):
        """Compute the direct impacts and write them to a copy of the network

        Each exposure of ``exp_list`` gets the impact function and threshold
        of its tag. The network rows the exposure was built from get:

        - ``imp_dir``: the direct impact (for edges, summed over their points),
        - ``func_internal``: 0 if the component is above its threshold, else 1,
        - ``func_tot``: ``min(func_internal, func_tot)``, so components that
          already failed (e.g. in the base-state cascade) stay failed.

        The threshold is compared with the damage ratio of each point, i.e.
        the impact for an exposure value of 1 (MDD x PAA at the hazard
        intensity, times the hazard fraction), between 0 and 1:

        - node: its damage ratio;
        - edge, ``edge_rule="max_ratio"`` (default): the largest damage ratio
          of its points;
        - edge, ``edge_rule="damaged_length"``: the damaged length in metres
          (mean damage ratio of its points x edge length); the threshold is
          then in metres.

        With a step impact function (``ImpactFunc.from_step_impf``) and a
        hazard without fraction, the damage ratio is 0 or 1. With a hazard
        fraction (e.g. river floods), it is the affected share of the cell.

        Run the cascade on the returned network to obtain the indirect
        impacts.

        Returns
        -------
        network_disr : Network
            Deep copy of ``network`` with the updated functional states.
        imp_dict : dict
            ``{tag: Impact}``: the impact on each exposure (per edge for edge
            exposures).

        Raises
        ------
        ValueError
            Before any computation, if the inputs are inconsistent: hazard with
            several events, network without functional states, duplicate tags,
            a node or edge in two exposures, exposure rows not in the network,
            missing impact function or threshold.

        Notes
        -----
        Node thresholds are damage ratios that accounts for hazard fraction,
        not absolute impacts.
        """

        self._check_inputs()
        network_disr = cp.deepcopy(self.network)
        haz_type = self.haz.haz_type
        imp_dict = {}
        for exp in self.exp_list:
            tag = exp.description
            impf = self.impf_set.get_func(haz_type=haz_type, fun_id=tag)
            # ImpactCalc adds the columns impf_<haz_type> and centr_<haz_type>
            # to the exposure: work on a copy so that exp_list is not modified
            exp_calc = exp.copy()
            exp_calc.gdf[f"impf_{haz_type}"] = impf.id
            impf_set = ImpactFuncSet([impf])
            imp = self.calc_point_impacts(self.haz, exp_calc, impf_set)
            ratio = _damage_ratio(self.haz, exp_calc, impf_set)
            kind, labels = _exposure_rows(exp)
            if kind == "edges":
                # sum the impacts of the points of each edge: the result has
                # one column per edge, in the same order as labels
                imp = u_lp.impact_pnt_agg(imp, exp_calc.gdf, u_lp.AggMethod.SUM)
                # value compared with the threshold, per edge and in the same
                # order as labels (the edge rules are described in the
                # docstring of disrupt_network)
                ratio_by_edge = ratio.groupby(level=0, sort=False)
                if self._edge_rule_of(tag) == "max_ratio":
                    # largest damage ratio of the points of the edge
                    compared = ratio_by_edge.max().loc[labels].to_numpy()
                else:
                    # damaged length in metres: each point stands for an equal
                    # share of the edge length, so the damaged length is the
                    # mean damage ratio of the points times the edge length
                    lengths = _edge_lengths_metres(network_disr.edges.loc[labels])
                    mean_ratio = ratio_by_edge.mean().loc[labels].to_numpy()
                    compared = mean_ratio * np.asarray(lengths, dtype=float)
            else:
                compared = ratio.to_numpy()
            # one value per network row (the hazard has a single event,
            # checked in _check_inputs)
            imp_dir = imp.imp_mat.toarray().ravel()
            func_internal = (compared <= self.impf_thresh_set[tag]).astype(int)
            _impacts_to_network(
                getattr(network_disr, kind), labels, imp_dir, func_internal
            )
            imp_dict[tag] = imp
        return network_disr, imp_dict
