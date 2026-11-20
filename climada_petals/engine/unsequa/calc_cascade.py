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

Define the uncertainty class for network cascade calculations
"""

__all__ = ["CalcCascade"]

import inspect
import itertools
import logging
import time
from typing import Union

import pandas as pd

from climada.engine.unsequa.calc_base import (
    Calc,
    _multiprocess_chunksize,
    _sample_parallel_iterator,
)
from climada.engine.unsequa.input_var import InputVar
from climada.hazard import Hazard
from climada.util import log_level

from climada_petals.engine.networks import nw_utils as nwu
from climada_petals.engine.networks.nw_calcs import NetworkCalcs
from climada_petals.engine.networks.nw_impacts import NetworkImpactCalc
from climada_petals.engine.unsequa.unc_output import UncCascadeOutput

LOGGER = logging.getLogger(__name__)

#: arguments of ``NetworkImpactCalc`` that ``CalcCascade`` sets from the input
#: variables (not allowed in ``impact_kwargs``)
IMPACT_ARGS_SET = ("impf_set", "failure_thresh", "haz", "network", "exp_list")

#: arguments of ``NetworkCalcs.cascade`` that ``CalcCascade`` sets from the
#: arguments of ``uncertainty`` (not allowed in ``cascade_kwargs``)
CASCADE_ARGS_SET = ("friction_surf",)

#: unit of the population statistics of ``make_network_stat``, saved as the
#: ``unit`` of ``UncCascadeOutput`` (the infrastructure statistics are in
#: metres or numbers of nodes, see ``UncCascadeOutput.imp_met_units``)
POP_UNIT = "people"


class CalcCascade(Calc):
    """
    Cascade uncertainty calculation class.

    Uncertainty and sensitivity analysis of the service access disruption
    computed with a climada_petals network cascade: for each sample of the
    input parameters, the network is disrupted by the hazard, the failure
    cascade is computed, and the network statistics of
    :py:func:`climada_petals.engine.networks.nw_utils.make_network_stat`
    are stored (e.g. number of people losing access to each ci_type).

    Attributes
    ----------
    nw_input_var : InputVar
        Network uncertainty variable. Its evaluation must return a tuple
        ``(network, dep_table, exp_list)``: the network with its dependencies
        and functional states (e.g. after ``NetworkCalcs.initialize_base_state``,
        ``setup_dependencies`` and a base-state ``cascade``), the dependency
        table used for the cascade, and the exposures built from this network
        with ``exposure_from_nodes`` / ``exposure_from_edges``. The network
        and the exposures are not modified by the calculation, so the same
        objects can be returned for every sample. A function that changes the
        network for a sample (e.g. scales the population) must change a copy,
        run the base-state cascade on it (``NetworkCalcs.cascade``) and build
        the exposures from the result, so that the base state and the
        disrupted state of the sample are computed with the same data.
        ``setup_dependencies`` (costly for large networks) can stay outside the
        function as long as the change does not affect the links.
    impf_input_var : InputVar
        Impact function uncertainty variable. Its evaluation must return a
        tuple ``(impf_set, failure_thresh)``: the impact functions and the
        failure threshold of each exposure, as ``{tag: threshold}`` (see
        ``NetworkImpactCalc``). Sampling the failure thresholds is a way to
        include the uncertainty of the fragility of the components.
    haz_input_var : InputVar
        Hazard uncertainty variable. Its evaluation must return a hazard with
        a single event, in the CRS of the network. To include several events,
        sample the event as a parameter of the hazard input variable (e.g.
        ``haz.select(event_id=[event_id])``).
    ci_types : list(str)
        ci_types for which the network statistics are computed.
    friction_surf : object or None
        Friction surface used in the cascade, set by :py:meth:`uncertainty`.
    impact_kwargs : dict
        Further arguments of ``NetworkImpactCalc``, set by
        :py:meth:`uncertainty`.
    cascade_kwargs : dict
        Further arguments of ``NetworkCalcs.cascade``, set by
        :py:meth:`uncertainty`.
    _input_var_names : tuple(str)
        Names of the required uncertainty input variables
        ('nw_input_var', 'impf_input_var', 'haz_input_var')
    _metric_names : tuple(str)
        Names of the output metrics ('imp_met',)
    """

    _input_var_names = (
        "nw_input_var",
        "impf_input_var",
        "haz_input_var",
    )
    """Names of the required uncertainty variables"""

    _metric_names = ("imp_met",)
    """Names of the cascade output metrics"""

    def __init__(
        self,
        nw_input_var: Union[InputVar, tuple],
        impf_input_var: Union[InputVar, tuple],
        haz_input_var: Union[InputVar, Hazard],
        ci_types: list,
    ):
        """Initialize CalcCascade

        Parameters
        ----------
        nw_input_var : InputVar or tuple
            Network uncertainty variable, or a fixed tuple
            ``(network, dep_table, exp_list)``. See the class attributes.
        impf_input_var : InputVar or tuple
            Impact function uncertainty variable, or a fixed tuple
            ``(impf_set, failure_thresh)``. See the class attributes.
        haz_input_var : InputVar or climada.hazard.Hazard
            Hazard uncertainty variable, or a Hazard with a single event. See
            the class attributes.
        ci_types : list(str)
            ci_types for which the network statistics are computed, e.g.
            ``["healthcare", "education"]``. Each must be a ``ci_type`` of the
            nodes or edges of the network (checked on the first sample).

        Raises
        ------
        TypeError
            If ``ci_types`` is a string instead of a list of strings.
        """

        if isinstance(ci_types, str):
            raise TypeError(
                f"ci_types must be a list of ci_type names, e.g. ['{ci_types}'], "
                f"not the string '{ci_types}'."
            )
        Calc.__init__(self)
        self.nw_input_var = InputVar.var_to_inputvar(nw_input_var)
        self.impf_input_var = InputVar.var_to_inputvar(impf_input_var)
        self.haz_input_var = InputVar.var_to_inputvar(haz_input_var)
        self.ci_types = list(ci_types)
        self.friction_surf = None
        self.impact_kwargs = {}
        self.cascade_kwargs = {}
        self.check_distr()

    def uncertainty(
        self,
        unc_sample,
        friction_surf=None,
        processes=1,
        chunksize=None,
        impact_kwargs=None,
        cascade_kwargs=None,
    ):
        """
        Compute the network statistics for each sample in unc_sample.samples_df.

        This sets the attributes self.friction_surf, self.impact_kwargs and
        self.cascade_kwargs.

        Parameters
        ----------
        unc_sample : climada.engine.unsequa.UncOutput
            Uncertainty data object with the input parameters samples
        friction_surf : object, optional
            Friction surface for duration-based dependencies, passed to
            ``NetworkCalcs`` and ``NetworkCalcs.cascade``. The default is None.
        processes : int, optional
            Number of CPUs to use for parallel computations.
            The default is 1 (not parallel)
        chunksize: int, optional
            Size of the sample chunks for parallel processing.
            Default is equal to the number of samples divided by the
            number of processes.
        impact_kwargs : dict, optional
            Further arguments of ``NetworkImpactCalc``, the same for all
            samples, e.g. ``{"edge_rule": "damaged_length"}`` or
            ``{"combine_exposures": True}``. The arguments set from the input
            variables (``impf_set``, ``failure_thresh``, ``haz``, ``network``,
            ``exp_list``) are not allowed. The default is None (defaults of
            ``NetworkImpactCalc``).
        cascade_kwargs : dict, optional
            Further arguments of ``NetworkCalcs.cascade``, the same for all
            samples, e.g. ``{"rerouting": False}``,
            ``{"access_check_method": "propagation"}`` or the names of the
            power columns (``{"source_var": "generation"}``).
            ``friction_surf`` is not allowed: use the argument of this method.
            The default is None (defaults of ``NetworkCalcs.cascade``).

        Returns
        -------
        unc_output : climada_petals.engine.unsequa.UncCascadeOutput
            Uncertainty data object with the network statistics for each
            sample (``imp_met_unc_df``, one column per statistic), the unit of
            each statistic (``imp_met_unit_df``) and all the sample data
            copied over from unc_sample.

        Raises
        ------
        ValueError
            If ``unc_sample`` has no samples, or if ``impact_kwargs`` /
            ``cascade_kwargs`` contain an argument that is set by
            ``CalcCascade`` or that the called function does not have.
        ValueError
            If one of ``ci_types`` is not a ``ci_type`` of the network.
        TypeError
            If ``nw_input_var`` does not evaluate to a tuple
            ``(network, dep_table, exp_list)``, or ``impf_input_var`` to a
            tuple ``(impf_set, failure_thresh)``.

        The checks on the evaluated input variables and the errors of the
        network calculation (e.g. a hazard with several events, a missing
        failure threshold) are raised by the first sample, which is computed
        before the others to estimate the computation time.

        Notes
        -----
        Parallelization logic is described in the base class
        here :py:class:`~climada.engine.unsequa.calc_base.Calc`
        """

        if unc_sample.samples_df.empty:
            raise ValueError(
                "No sample was found. Please create one first "
                "using CalcCascade.make_sample(N)"
            )

        self.impact_kwargs = _check_kwargs(
            impact_kwargs, NetworkImpactCalc, IMPACT_ARGS_SET, "impact_kwargs"
        )
        self.cascade_kwargs = _check_kwargs(
            cascade_kwargs, NetworkCalcs.cascade, CASCADE_ARGS_SET, "cascade_kwargs"
        )

        # copy may not be needed, but is kept to prevent potential
        # data corruption issues. The computational cost should be
        # minimal as only a list of floats is copied.
        samples_df = unc_sample.samples_df.copy(deep=True)

        if chunksize is None:
            chunksize = _multiprocess_chunksize(samples_df, processes)
        self.friction_surf = friction_surf

        one_sample = samples_df.iloc[0:1]
        start = time.time()
        self._compute_metrics(one_sample, chunksize=1, processes=1)
        elapsed_time = time.time() - start
        self.est_comp_time(unc_sample.n_samples, elapsed_time, processes)

        chunk_results = self._compute_metrics(
            samples_df, chunksize=chunksize, processes=processes
        )
        imp_met_unc_df = pd.concat(
            [stats for stats, _ in chunk_results], ignore_index=True
        )
        units = {}
        for _, chunk_units in chunk_results:
            units.update(chunk_units)
        imp_met_unit_df = pd.DataFrame([units], columns=imp_met_unc_df.columns)

        return UncCascadeOutput(
            samples_df=samples_df,
            unit=POP_UNIT,
            imp_met_unc_df=imp_met_unc_df,
            imp_met_unit_df=imp_met_unit_df,
        )

    def _compute_metrics(self, samples_df, chunksize, processes):
        """Compute the uncertainty metrics

        Parameters
        ----------
        samples_df : pd.DataFrame
            dataframe of input parameter samples
        chunksize : int
            size of the samples chunks
        processes : int
            number of processes to use

        Returns
        -------
        list(tuple)
            one tuple ``(stats, units)`` per chunk of samples: the network
            statistics (pd.DataFrame) and their units (dict), see
            ``_map_cascade_calc``
        """
        with log_level(level="ERROR", name_prefix="climada"):
            p_iterator = _sample_parallel_iterator(
                samples=samples_df,
                chunksize=chunksize,
                nw_input_var=self.nw_input_var,
                impf_input_var=self.impf_input_var,
                haz_input_var=self.haz_input_var,
                ci_types=self.ci_types,
                friction_surf=self.friction_surf,
                impact_kwargs=self.impact_kwargs,
                cascade_kwargs=self.cascade_kwargs,
            )
            if processes > 1:
                # need to use multiprocess instead of multiprocessing
                # to avoid pickling error and get_context to avoid deadlocks
                from multiprocess import get_context

                with get_context("spawn").Pool(processes=processes) as pool:
                    LOGGER.info("Using %s CPUs.", processes)
                    return pool.starmap(_map_cascade_calc, p_iterator)
            # itertools.starmap is lazy: evaluate it here, inside the
            # log_level context and before timing ends
            return list(itertools.starmap(_map_cascade_calc, p_iterator))


def _map_cascade_calc(
    sample_chunks,
    nw_input_var,
    impf_input_var,
    haz_input_var,
    ci_types,
    friction_surf,
    impact_kwargs,
    cascade_kwargs,
):
    """
    Map to compute the cascade for all parameter samples in parallel

    Parameters
    ----------
    sample_chunks : pd.DataFrame
        Dataframe of the parameter samples
    nw_input_var : InputVar
        Network uncertainty variable, evaluating to
        ``(network, dep_table, exp_list)``
    impf_input_var : InputVar
        Impact function uncertainty variable, evaluating to
        ``(impf_set, failure_thresh)``
    haz_input_var: InputVar
        Hazard uncertainty variable
    ci_types : list(str)
        ci_types for which the network statistics are computed
    friction_surf : object or None
        Friction surface used in the cascade
    impact_kwargs : dict
        Further arguments of ``NetworkImpactCalc``
    cascade_kwargs : dict
        Further arguments of ``NetworkCalcs.cascade``

    Returns
    -------
    stats : pd.DataFrame
        Network statistics (columns) for each sample (rows) of the chunk
    units : dict
        Unit of each statistic, ``{column: unit}`` (see ``_statistic_units``)
    """
    uncertainty_values = []
    units = {}
    for _, sample in sample_chunks.iterrows():
        nw_samples = sample[nw_input_var.labels].to_dict()
        impf_samples = sample[impf_input_var.labels].to_dict()
        haz_samples = sample[haz_input_var.labels].to_dict()

        network, dep_table, exp_list = _unpack_input(
            nw_input_var.evaluate(**nw_samples),
            ("network", "dep_table", "exp_list"),
            "nw_input_var",
        )
        impf_set, failure_thresh = _unpack_input(
            impf_input_var.evaluate(**impf_samples),
            ("impf_set", "failure_thresh"),
            "impf_input_var",
        )
        haz = haz_input_var.evaluate(**haz_samples)
        _check_ci_types(network, ci_types)

        # disrupt the network with the direct impacts
        nw_impact_calc = NetworkImpactCalc(
            network=network,
            impf_set=impf_set,
            failure_thresh=failure_thresh,
            haz=haz,
            exp_list=exp_list,
            **impact_kwargs,
        )
        network_disr, _ = nw_impact_calc.disrupt_network()

        # propagate the failures
        nw_calc = NetworkCalcs(
            network=network_disr, dep_table=dep_table, friction_surf=friction_surf
        )
        nw_calc.cascade(
            friction_surf=friction_surf,
            **cascade_kwargs,
        )

        stats = nwu.make_network_stat(nw_calc.network, ci_types=ci_types)
        uncertainty_values.append(stats)
        units.update(_statistic_units(nw_calc.network, ci_types, stats))
    return pd.DataFrame.from_records(uncertainty_values), units


def _check_kwargs(kwargs, func, args_set, name):
    """Check the further arguments passed to a function of the network module

    Parameters
    ----------
    kwargs : dict or None
        Further arguments, e.g. ``impact_kwargs`` of ``CalcCascade.uncertainty``.
    func : callable
        Function or class the arguments are passed to.
    args_set : tuple(str)
        Arguments of ``func`` that ``CalcCascade`` sets itself.
    name : str
        Name of ``kwargs`` in the error messages.

    Returns
    -------
    dict
        A copy of ``kwargs`` (empty if ``kwargs`` is None).

    Raises
    ------
    ValueError
        If ``kwargs`` contains an argument of ``args_set`` or an argument that
        ``func`` does not have.
    """
    kwargs = dict(kwargs or {})
    set_by_calc = sorted(set(kwargs).intersection(args_set))
    if set_by_calc:
        raise ValueError(
            f"{name} cannot contain {set_by_calc}: these arguments are set by "
            "CalcCascade (from the input variables or the arguments of "
            "uncertainty)."
        )
    allowed = set(inspect.signature(func).parameters).difference(
        {"self", *args_set}
    )
    unknown = sorted(set(kwargs).difference(allowed))
    if unknown:
        raise ValueError(
            f"{name} contains {unknown}, which are not arguments of "
            f"{func.__qualname__}. Possible arguments: {sorted(allowed)}."
        )
    return kwargs


def _unpack_input(value, names, var_name):
    """Return the evaluation of an input variable as a tuple of ``len(names)``

    Parameters
    ----------
    value : object
        Evaluation of the input variable.
    names : tuple(str)
        Names of the expected elements, used in the error message.
    var_name : str
        Name of the input variable, used in the error message.

    Returns
    -------
    tuple
        ``value`` as a tuple.

    Raises
    ------
    TypeError
        If ``value`` is not a tuple (or list) with one element per name.
    """
    if isinstance(value, (tuple, list)) and len(value) == len(names):
        return tuple(value)
    got = type(value).__name__
    if isinstance(value, (tuple, list)):
        got = f"{got} of length {len(value)}"
    raise TypeError(
        f"{var_name} must evaluate to a tuple ({', '.join(names)}), got {got}."
    )


def _check_ci_types(network, ci_types):
    """Check that the ci_types of the statistics are in the network

    Parameters
    ----------
    network : Network
        Network of the sample.
    ci_types : list(str)
        ci_types for which the network statistics are computed.

    Raises
    ------
    ValueError
        If one of ``ci_types`` is not a ``ci_type`` of the nodes or edges of
        ``network``. ``make_network_stat`` would raise too, but only after
        the impact and cascade calculation, and without listing the ci_types
        of the network.
    """
    present = set()
    for gdf in (network.nodes, network.edges):
        if "ci_type" in gdf.columns:
            present.update(gdf["ci_type"].dropna().unique())
    missing = [ci_type for ci_type in ci_types if ci_type not in present]
    if missing:
        in_network = sorted(
            ci_type for ci_type in present if not str(ci_type).startswith("dependency_")
        )
        raise ValueError(
            f"ci_types {missing} are not in the network. ci_types of the "
            f"network: {in_network}."
        )


def _statistic_units(network, ci_types, columns, pop_unit=POP_UNIT):
    """Unit of each network statistic of ``make_network_stat``

    Parameters
    ----------
    network : Network
        Network the statistics were computed on.
    ci_types : list(str)
        ci_types of the statistics.
    columns : iterable(str)
        Names of the statistics.
    pop_unit : str, optional
        Unit of the population statistics. The default is ``POP_UNIT``
        (``"people"``).

    Returns
    -------
    dict
        ``{column: unit}``: ``"m"`` for ``{ci_type}_infra_impact`` /
        ``{ci_type}_infra_total`` of line types (ci_types of the edges, whose
        statistics are sums of the edge column ``distance`` in metres),
        ``"nodes"`` for those of point types (numbers of nodes), ``pop_unit``
        for all other statistics (population).
    """
    edge_types = set(network.edges["ci_type"].dropna().unique())
    units = {column: pop_unit for column in columns}
    for ci_type in ci_types:
        unit = "m" if ci_type in edge_types else "nodes"
        for column in (f"{ci_type}_infra_impact", f"{ci_type}_infra_total"):
            if column in units:
                units[column] = unit
    return units
