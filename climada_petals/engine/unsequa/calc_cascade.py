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

Define Uncertainty Impact class
"""

__all__ = ["CalcCascade"]

import copy as cp
import itertools
import logging
import time
from typing import Union

import numpy as np
import pandas as pd
import pathos.multiprocessing as mp

# import network from petals
from climada_petals.engine.networks import nw_utils as nwu
from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_calcs import NetworkCalcs
from climada_petals.engine.networks.nw_impacts import NetworkImpactCalc
from climada.entity import ImpactFuncSet
from climada.engine.unsequa.calc_base import (
    Calc,
    _multiprocess_chunksize,
    _sample_parallel_iterator,
    _transpose_chunked_data,
)
from climada.engine.unsequa.input_var import InputVar
from climada.engine.unsequa.calc_base import Calc
from climada_petals.engine.unsequa.unc_output import UncCascadeOutput
from climada.hazard import Hazard
from climada.util import log_level

# use pathos.multiprocess fork of multiprocessing for compatibility
# wiht notebooks and other environments https://stackoverflow.com/a/65001152/12454103


LOGGER = logging.getLogger(__name__)


class CalcCascade(Calc):
    """
    Cascade uncertainty caclulation class.

    This is the class to perform uncertainty analysis on the outputs of a
    climada_petals network cascade object.

    Attributes
    ----------
    rp : list(int)
        List of the chosen return periods.
    calc_eai_exp : bool
        Compute eai_exp or not
    calc_at_event : bool
        Compute eai_exp or not
    value_unit : str
        Unit of the exposures value
    exp_input_var : InputVar or Exposures
        Exposure uncertainty variable
    impf_input_var : InputVar if ImpactFuncSet
        Impact function set uncertainty variable
    haz_input_var: InputVar or Hazard
        Hazard uncertainty variable
    _input_var_names : tuple(str)
        Names of the required uncertainty input variables
        ('exp_input_var', 'impf_input_var', 'haz_input_var')
    _metric_names : tuple(str)
        Names of the impact output metrics
        ('aai_agg', 'freq_curve', 'at_event', 'eai_exp')
    """

    _input_var_names = (
        "nw_input_var",
        "impf_input_var",
        "haz_input_var",
    )
    """Names of the required uncertainty variables"""

    _metric_names = [
        "imp_met"
    ]  # need to be a list instead of tuple in case of unique metric
    """Names of the cost benefit output metrics"""

    def __init__(
        self,
        nw_input_var: Union[InputVar, Network],
        impf_input_var: Union[InputVar, ImpactFuncSet],
        haz_input_var: Union[InputVar, Hazard],
        ci_types: list,
    ):
        """Initialize UncCalcImpact

        Sets the uncertainty input variables, the impact metric_names, and the
        units.

        Parameters
        ----------
        nw_input_var : climada.engine.uncertainty.input_var.InputVar or network
            Exposure uncertainty variable or network
        impf_input_var : climada.engine.uncertainty.input_var.InputVar or climada.entity.ImpactFuncSet
            Impact function set uncertainty variable or Impact function set
        haz_input_var : climada.engine.uncertainty.input_var.InputVar or climada.hazard.Hazard
            Hazard uncertainty variable or Hazard

        """

        Calc.__init__(self)
        self.nw_input_var = InputVar.var_to_inputvar(nw_input_var)
        self.impf_input_var = InputVar.var_to_inputvar(impf_input_var)
        self.haz_input_var = InputVar.var_to_inputvar(haz_input_var)
        self.ci_types = ci_types
        self.value_unit = "people"
        self.check_distr()

    def uncertainty(
        self,
        unc_sample,
        friction_surf,
        processes=1,
        chunksize=None,
    ):
        """
        Computes the impact for each sample in unc_data.sample_df.

        By default, number of people losing access to each ci_type is computed
        along with the number of people directly affected.

        This sets the attributes self.rp, self.calc_eai_exp,
        self.calc_at_event, self.metrics.

        This sets the attributes:
        unc_output.aai_agg_unc_df,
        unc_output.freq_curve_unc_df
        unc_output.eai_exp_unc_df
        unc_output.at_event_unc_df
        unc_output.unit

        Parameters
        ----------
        unc_sample : climada.engine.uncertainty.unc_output.UncOutput
            Uncertainty data object with the input parameters samples
        rp : list(int), optional
            Return periods in years to be computed.
            The default is [5, 10, 20, 50, 100, 250].
        calc_eai_exp : boolean, optional
            Toggle computation of the impact at each centroid location.
            The default is False.
        calc_at_event : boolean, optional
            Toggle computation of the impact for each event.
            The default is False.
        processes : int, optional
            Number of CPUs to use for parralel computations.
            The default is 1 (not parallel)
        chunksize: int, optional
            Size of the sample chunks for parallel processing.
            Default is equal to the number of samples divided by the
            number of processes.

        Returns
        -------
        unc_output : climada.engine.uncertainty.unc_output.UncImpactOutput
            Uncertainty data object with the impact outputs for each sample
            and all the sample data copied over from unc_sample.

        Raises
        ------
        ValueError:
            If no sampling parameters defined, the distribution cannot
            be computed.

        Notes
        -----
        Parallelization logic is described in the base class
        here :py:class:`~climada.engine.unsequa.calc_base.Calc`

        See Also
        --------
        climada.engine.impact:
            compute impact and risk.

        """

        if unc_sample.samples_df.empty:
            raise ValueError(
                "No sample was found. Please create one first"
                "using UncImpact.make_sample(N)"
            )

        # copy may not be needed, but is kept to prevent potential
        # data corruption issues. The computational cost should be
        # minimal as only a list of floats is copied.'''
        samples_df = unc_sample.samples_df.copy(deep=True)

        if chunksize is None:
            chunksize = _multiprocess_chunksize(samples_df, processes)
        unit = self.value_unit
        self.friction_surf = friction_surf
        one_sample = samples_df.iloc[0:1]
        start = time.time()
        self._compute_metrics(one_sample, chunksize=1, processes=1)
        elapsed_time = time.time() - start
        self.est_comp_time(unc_sample.n_samples, elapsed_time, processes)

        imp_met_list = self._compute_metrics(
            samples_df, chunksize=chunksize, processes=processes
        )
        # Assign computed impact distribution data to self
        imp_met_unc_df = pd.DataFrame(imp_met_list, index=self.ci_types).T
        # freq_curve_unc_df = pd.DataFrame(
        #    freq_curve_list, columns=["rp" + str(n) for n in rp]
        # )
        # eai_exp_unc_df = pd.DataFrame(eai_exp_list)
        ## Note: sparse dataframes are not used as they are not nativel y compatible with .to_hdf5
        # at_event_unc_df = pd.DataFrame(at_event_list)
        #
        # if calc_eai_exp:
        #    exp = self.exp_input_var.evaluate()
        #    coord_df = pd.DataFrame(
        #        dict(latitude=exp.latitude, longitude=exp.longitude)
        #    )
        # else:
        #    coord_df = pd.DataFrame([])
        return UncCascadeOutput(
            samples_df=samples_df,
            unit=unit,
            imp_met_unc_df=imp_met_unc_df,
            # freq_curve_unc_df=freq_curve_unc_df,
            # eai_exp_unc_df=eai_exp_unc_df,
            # at_event_unc_df=at_event_unc_df,
            # coord_df=coord_df,
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
        list
            values of impact metrics per sample
        """
        # Compute impact distributions
        with log_level(level="ERROR", name_prefix="climada"):
            p_iterator = _sample_parallel_iterator(
                samples=samples_df,
                chunksize=chunksize,
                nw_input_var=self.nw_input_var,
                impf_input_var=self.impf_input_var,
                haz_input_var=self.haz_input_var,
                ci_types=self.ci_types,
                friction_surf=self.friction_surf,
            )
            if processes > 1:
                from multiprocess import get_context

                # need to use multiprocess instead of multiprocessing
                # to avoid pickling error and get_context to avoid deadlocks
                with get_context("spawn").Pool(processes=processes) as pool:
                    LOGGER.info("Using %s CPUs.", processes)
                    imp_metrics = pool.starmap(_map_impact_calc, p_iterator)
            else:
                imp_metrics = itertools.starmap(_map_impact_calc, p_iterator)

        # Perform the actual computation
        with log_level(level="ERROR", name_prefix="climada"):
            return _transpose_chunked_data(imp_metrics)
            # return imp_metrics


def _map_impact_calc(
    sample_chunks,
    nw_input_var,
    impf_input_var,
    haz_input_var,
    ci_types,
    friction_surf,
):
    """
    Map to compute impact for all parameter samples in parallel

    Parameters
    ----------
    sample_chunks : pd.DataFrame
        Dataframe of the parameter samples
    nw_input_var : InputVar or Network
        Network uncertainty variable
    impf_input_var : InputVar if ImpactFuncSet
        Impact function set uncertainty variable
    haz_input_var: InputVar or Hazard
        Hazard uncertainty variable
    ci_types : list(str)
        List of the chosen critical infrastructures for which to compute impacts


    Returns
    -------
        : list
        impact metrics list for all samples containing aai_agg, rp_curve,
        eai_exp (np.array([]) if self.calc_eai_exp=False) and at_event
        (np.array([]) if self.calc_at_event=False).

    """
    # uncertainty_values = {k: [] for k in ci_types}
    uncertainty_values = []
    for _, sample in sample_chunks.iterrows():
        nw_samples = sample[nw_input_var.labels].to_dict()
        impf_samples = sample[impf_input_var.labels].to_dict()
        haz_samples = sample[haz_input_var.labels].to_dict()

        nw, df_dep, exp_list = nw_input_var.evaluate(**nw_samples)  # create network
        impf, thresh_func_dict = impf_input_var.evaluate(**impf_samples)
        haz = haz_input_var.evaluate(**haz_samples)

        # disrupt network
        #create nw_impact_object
        nw_impact_calc = NetworkImpactCalc(network=nw,
                                           impf_set=impf,
                                           impf_thresh_set=thresh_func_dict,
                                           haz=haz,
                                           exp_list=exp_list)

        #run impact calculation
        nw_disr, _ = nw_impact_calc.disrupt_network()

        #cascade
        nw_calc = NetworkCalcs(network=nw_disr,
                               dep_table=df_dep,
                               friction_surf=friction_surf)
        nw_calc.cascade(initial=False, rerouting=True, friction_surf=friction_surf)
        nw_disr_casc = nw_calc.network

        # CALC IMPACTSTATS
        imp_dict = nwu.disaster_impact_allservices_df(
            nw.nodes, nw_disr_casc.nodes, services=ci_types
        )
        imp_types = [
            ci_type + "_access" if ci_type != "people" else ci_type
            for ci_type in ci_types
        ]
        #print(imp_types)
        # {uncertainty_values[k].append(v) for k, v in imp_dict.items()}
        imp_list = [imp_dict[imp_type] for imp_type in imp_types]
        uncertainty_values.append(imp_list)
    return list(zip(*uncertainty_values))
