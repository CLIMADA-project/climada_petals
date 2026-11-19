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
        network for a sample (e.g. scales the population) must change a copy.
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
    value_unit : str
        Unit of the output metrics ('people').
    friction_surf : object or None
        Friction surface used in the cascade, set by :py:meth:`uncertainty`.
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
            ``["healthcare", "education"]``.
        """

        Calc.__init__(self)
        self.nw_input_var = InputVar.var_to_inputvar(nw_input_var)
        self.impf_input_var = InputVar.var_to_inputvar(impf_input_var)
        self.haz_input_var = InputVar.var_to_inputvar(haz_input_var)
        self.ci_types = ci_types
        self.value_unit = "people"
        self.friction_surf = None
        self.rerouting = True
        self.access_check_method = "routing"
        self.check_distr()

    def uncertainty(
        self,
        unc_sample,
        friction_surf=None,
        rerouting=True,
        access_check_method="routing",
        processes=1,
        chunksize=None,
    ):
        """
        Compute the network statistics for each sample in unc_sample.samples_df.

        This sets the attributes self.friction_surf, self.rerouting and
        self.access_check_method.

        Parameters
        ----------
        unc_sample : climada.engine.unsequa.UncOutput
            Uncertainty data object with the input parameters samples
        friction_surf : object, optional
            Friction surface used in the cascade for duration-based
            dependencies. The default is None.
        rerouting : bool, optional
            Passed to ``NetworkCalcs.cascade``: end users of dependencies with
            access constraint can be linked to another source when their
            source or path fails. The default is True.
        access_check_method : str, optional
            Passed to ``NetworkCalcs.cascade``: access check of the
            dependencies with access constraint, ``"routing"`` or
            ``"propagation"``. The default is ``"routing"``.
        processes : int, optional
            Number of CPUs to use for parallel computations.
            The default is 1 (not parallel)
        chunksize: int, optional
            Size of the sample chunks for parallel processing.
            Default is equal to the number of samples divided by the
            number of processes.

        Returns
        -------
        unc_output : climada_petals.engine.unsequa.UncCascadeOutput
            Uncertainty data object with the network statistics for each
            sample (``imp_met_unc_df``, one column per statistic) and all the
            sample data copied over from unc_sample.

        Raises
        ------
        ValueError
            If ``unc_sample`` has no samples. Errors of the network
            calculation (e.g. a hazard with several events, a missing failure
            threshold) are raised by the first sample, which is computed
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

        # copy may not be needed, but is kept to prevent potential
        # data corruption issues. The computational cost should be
        # minimal as only a list of floats is copied.
        samples_df = unc_sample.samples_df.copy(deep=True)

        if chunksize is None:
            chunksize = _multiprocess_chunksize(samples_df, processes)
        self.friction_surf = friction_surf
        self.rerouting = rerouting
        self.access_check_method = access_check_method

        one_sample = samples_df.iloc[0:1]
        start = time.time()
        self._compute_metrics(one_sample, chunksize=1, processes=1)
        elapsed_time = time.time() - start
        self.est_comp_time(unc_sample.n_samples, elapsed_time, processes)

        imp_met_list = self._compute_metrics(
            samples_df, chunksize=chunksize, processes=processes
        )
        imp_met_unc_df = pd.concat(imp_met_list, ignore_index=True)

        return UncCascadeOutput(
            samples_df=samples_df,
            unit=self.value_unit,
            imp_met_unc_df=imp_met_unc_df,
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
        list(pd.DataFrame)
            network statistics, one dataframe per chunk of samples
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
                rerouting=self.rerouting,
                access_check_method=self.access_check_method,
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
    rerouting,
    access_check_method,
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
    rerouting : bool
        Allow rerouting in the cascade
    access_check_method : str
        Access check method of the cascade

    Returns
    -------
    pd.DataFrame
        Network statistics (columns) for each sample (rows) of the chunk
    """
    uncertainty_values = []
    for _, sample in sample_chunks.iterrows():
        nw_samples = sample[nw_input_var.labels].to_dict()
        impf_samples = sample[impf_input_var.labels].to_dict()
        haz_samples = sample[haz_input_var.labels].to_dict()

        network, dep_table, exp_list = nw_input_var.evaluate(**nw_samples)
        impf_set, failure_thresh = impf_input_var.evaluate(**impf_samples)
        haz = haz_input_var.evaluate(**haz_samples)

        # disrupt the network with the direct impacts
        nw_impact_calc = NetworkImpactCalc(
            network=network,
            impf_set=impf_set,
            failure_thresh=failure_thresh,
            haz=haz,
            exp_list=exp_list,
        )
        network_disr, _ = nw_impact_calc.disrupt_network()

        # propagate the failures
        nw_calc = NetworkCalcs(
            network=network_disr, dep_table=dep_table, friction_surf=friction_surf
        )
        nw_calc.cascade(
            friction_surf=friction_surf,
            rerouting=rerouting,
            access_check_method=access_check_method,
        )

        uncertainty_values.append(
            nwu.make_network_stat(nw_calc.network, ci_types=ci_types)
        )
    return pd.DataFrame.from_records(uncertainty_values)
