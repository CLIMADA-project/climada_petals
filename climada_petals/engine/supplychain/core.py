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

Define the SupplyChain class.
"""

from __future__ import (
    annotations,
)
from abc import (
    ABC,
    abstractmethod,
)

# Replace by `from typing import Self` when python 3.11+ is the min required version.

__all__ = ["DirectShock", "IndirectCostModel", "BoARIOModel"]

import logging
from typing import Iterable, Literal
import warnings
import pandas as pd
import numpy as np

import pymrio

from boario.extended_models import ARIOPsiModel
from boario.simulation import Simulation
from boario import event as boario_event_module

from climada import CONFIG
from climada.engine import Impact
from climada.entity import Exposures

from climada_petals.engine.supplychain.utils import (
    distribute_reg_impact_to_sectors,
    calc_va,
    calc_x_from_G,
    translate_exp_to_regions,
    translate_exp_to_sectors,
    translate_imp_to_regions,
    _thin_to_wide,
)

from climada_petals.engine.supplychain.mriot_handling import _get_coco_MRIOT_name

LOGGER = logging.getLogger(__name__)

MRIOT_DIRECTORY = CONFIG.engine.supplychain.local_data.mriot.dir()
"""Directory where Multi-Regional Input-Output Tables (MRIOT) are downloaded."""

VA_NAME = "value added"
"""Index name for value added"""

SIMULATION_LENGTH_BUFFER = 365


class DirectShock:
    """DirectShock class.

    The DirectShock class provides methods for 'translating' an Impact and Exposure into
    an economic shock, corresponding to a specific MRIOT table, from which indirect economic
    costs can be computed.

    Attributes
    ----------
    mriot_name: str
            The name of the MRIOT defining the typology of assets.

    mriot_sectors: list[str]
            The list of possible sectors.

    mriot_regions: list[str]
            The list of possible regions.

    name: str, default "unnamed"
            A name to identify the object.

    exposure_assets: pd.Series
            Exposure translated in the region/sector typology of the MRIOT. The index of the Series is a subset
            of the possible (region, sector).

    impacted_assets: pd.DataFrame
            Impact translated in the region/sector typology of the MRIOT. The columns index of the DataFrame is a subset
            of the possible (region, sector), and the row index correspond to event_ids of the Impact.

    """

    def __init__(
        self,
        mriot_name: str,
        mriot_sectors: list[str],
        mriot_regions: list[str],
        exposure_assets: pd.Series,
        impacted_assets: pd.DataFrame,
        event_dates: np.ndarray,
        monetary_factor: int,
        shock_name: str = "unnamed",
    ) -> None:
        self.mriot_sectors = mriot_sectors
        self.mriot_regions = mriot_regions
        self.mriot_industries = pd.MultiIndex.from_product(
            [self.mriot_regions, self.mriot_sectors], names=["region", "sector"]
        )
        self.mriot_name = mriot_name
        self.monetary_factor = monetary_factor
        self.name = shock_name
        self.exposure_assets = _thin_to_wide(exposure_assets, self.mriot_industries)
        self.impacted_assets = _thin_to_wide(
            impacted_assets, impacted_assets.index, self.mriot_industries
        )
        self.event_dates = event_dates

    @classmethod
    def _init_with_mriot(
        cls,
        mriot: pymrio.IOSystem,
        exposed_assets: pd.Series,
        impacted_assets: pd.DataFrame,
        event_dates: np.ndarray,
        shock_name: str,
    ):
        return cls(
            mriot.name,
            mriot.get_sectors(),
            mriot.get_regions(),
            exposed_assets,
            impacted_assets,
            event_dates,
            mriot.monetary_factor,
            shock_name,
        )

    @property
    def relative_impact(self):
        """The ratio of impacted assets over total assets."""
        return self.impacted_assets / self.exposure_assets

    @classmethod
    def from_exp_and_imp(
        cls,
        mriot: pymrio.IOSystem,
        exposure: Exposures,
        impact: Impact,
        shock_name: str,
        affected_sectors: Iterable[str] | dict[str, float] | pd.Series | Literal["all"],
        impact_distribution: dict[str, float] | pd.Series | None,
        exp_value_col: str = "value",
    ):
        """Build a DirectShock from an MRIOT, Exposure and Impact objects.

        This method translates both the given Exposure and Impact objects to the MRIOT typology.
        First, it aggregates exposure values by `region_id` and then maps them to the
        regions of the given mriot. Assets are then distributed to the
        different sectors specified by `affected_sectors`.
        It proceeds with a similar process for the Impact object, using `impact_distribution` to
        distribute the impact across sectors.

        Parameters
        ----------
        mriot : pymrio.IOSystem
            The MRIOT to use for the typology of region and sectors.
        exposure : Exposures
            The Exposures object to use to derive total assets from.
        impact : Impact
            The Impact object to derive the impact on assets from.
        shock_name : str
            An optional name to identify the object.
        affected_sectors : Iterable[str] | dict[str, float] | pd.Series | Literal["all"]
            The sectors of the MRIOT that are impacted. If given as a
            collection of string, or `"all"`, then the total assets of the
            region are distributed proportionally to each sectors gross output.
            A dictionnary `sector:share` can also be passed to specify which
            share of the total regional assets should be distributed to each
            sector.
        impact_distribution : dict[str, float] | pd.Series, optional
            This argument specify how the impact per region should be
            distributed to the impacted sectors. Using `None` will distribute
            proportionally to each sectors gross output in the MRIOT. A
            dictionnary in the form `sector:share` or similarly a `Series` can
            be used to specify a custom distribution.
        exp_value_col : str
            The name of the column of the Exposure data representing the value
            of assets in each centroids.

        """
        exp = translate_exp_to_regions(exposure, mriot_name=mriot.name)
        exposure_assets = translate_exp_to_sectors(
            exp, affected_sectors=affected_sectors, mriot=mriot, value_col=exp_value_col
        )
        return cls.from_assets_and_imp(
            mriot,
            exposure_assets,
            impact,
            shock_name,
            affected_sectors,
            impact_distribution,
            impact.date,
        )

    @classmethod
    def from_assets_and_imp(
        cls,
        mriot: pymrio.IOSystem,
        exposure_assets: pd.Series,
        impact: Impact,
        shock_name: str,
        affected_sectors: Iterable[str] | dict[str, float] | pd.Series | Literal["all"],
        impact_distribution: dict[str, float] | pd.Series | None,
        event_dates: np.ndarray,
    ):
        """Build a DirectShock from an MRIOT, assets Series, and Impact objects.

        This method translates the given Impact object to the MRIOT typology
        (see :py:meth:`from_exp_and_imp`).

        Parameters
        ----------
        mriot : pymrio.IOSystem
            The MRIOT to use for the typology of region and sectors.
        exposure_assets : pd.Series
            A pandas `Series` with (region,sector) as index and assets value.
        impact : Impact
            The Impact object to derive the impact on assets from.
        shock_name : str
            An optional name to identify the object.
        affected_sectors : Iterable[str] | dict[str, float] | Literal["all"]
            The sectors of the MRIOT that are impacted. If given as a
            collection of string, or `"all"`, then the total assets of the
            region are distributed proportionally to each sectors gross output.
            A dictionnary `sector:share` can also be passed to specify which
            share of the total regional assets should be distributed to each
            sector.
        impact_distribution : dict[str, float] | pd.Series, optional
            This argument specify how the impact per region should be
            distributed to the impacted sectors. Using `None` will distribute
            proportionally to each sectors gross output in the MRIOT. A
            dictionnary in the form `sector:share` or similarly a `Series` can
            be used to specify a custom distribution.
        exp_value_col : str
            The name of the column of the Exposure data representing the value
            of assets in each centroids.

        """

        impacted_assets = impact.impact_at_reg()
        impacted_assets = impacted_assets.loc[:, (impacted_assets != 0).any()]
        impacted_assets = translate_imp_to_regions(impacted_assets, mriot)
        if isinstance(affected_sectors, str) and affected_sectors == "all":
            affected_sectors = list(mriot.get_sectors()) # type: ignore

        if impact_distribution is None:
            # Default uses production distribution across sectors, region.
            impact_distribution = (
                mriot.x.loc[
                    pd.IndexSlice[impacted_assets.columns, affected_sectors], "indout"
                ]
                .groupby(level=0)
                .transform(lambda x: x / x.sum())
            )

        if isinstance(impact_distribution, dict):
            impact_distribution = pd.Series(impact_distribution)

        if not isinstance(impact_distribution, pd.Series):
            raise ValueError(f"impact_distribution could not be converted to a Series")

        impacted_assets = distribute_reg_impact_to_sectors(
            impacted_assets, impact_distribution
        )
        impacted_assets.rename_axis(index="event_id", inplace=True)
        return cls._init_with_mriot(
            mriot, exposure_assets, impacted_assets, event_dates, shock_name
        )

    @classmethod
    def combine(
        cls,
        direct_shocks: list[DirectShock],
        kind: Literal["merge", "concat"] = "merge",
        direct_shock_ids: list | None = None,
        combine_name: str = "unnamed_combine",
    ):
        if not len(direct_shocks) > 1:
            return direct_shocks[0]
        # 1. Check that MRIOT name and exposed assets are the same
        combined_exp_assets = cls._combine_exp_assets(direct_shocks)

        imp_assets = [shock.impacted_assets for shock in direct_shocks]
        if kind == "merge":
            LOGGER.info(
                "Merging direct shocks together. The resulting direct shock per event and per each region,sector will be the sum of the different direct shocks."
            )
            # Note that merging shocks on same sector is not possible.
            merged_imp_assets = pd.concat(
                imp_assets, join="outer", sort=True, axis=1, verify_integrity=True
            ).fillna(0)
            return cls(
                direct_shocks[0].mriot_name,
                direct_shocks[0].mriot_sectors,
                direct_shocks[0].mriot_regions,
                combined_exp_assets,
                merged_imp_assets,
                np.array([]),  # To be fixed, need to implement date concatenation
                direct_shocks[0].monetary_factor,
                combine_name,
            )

        if kind == "concat":
            if direct_shock_ids is None:
                direct_shock_ids = list(range(len(direct_shocks)))
            LOGGER.info(
                "Concatenating direct shocks. This assume the different direct shocks are from independent events."
            )
            concatenated_imp_assets = pd.concat(
                imp_assets,
                axis=0,
                keys=direct_shock_ids,
                names=["DirectShock_id", "event_id"],
                sort=True,
            ).fillna(0)
            return cls(
                direct_shocks[0].mriot_name,
                direct_shocks[0].mriot_sectors,
                direct_shocks[0].mriot_regions,
                direct_shocks[0].exposure_assets,
                concatenated_imp_assets,
                np.array([]),  # To be fixed, need to implement date concatenation
                direct_shocks[0].monetary_factor,
                combine_name,
            )

    @staticmethod
    def _combine_exp_assets(direct_shocks: list[DirectShock]) -> pd.Series:
        def _check_mriot_name(mriot_names: list[str]) -> bool:
            return all(name == mriot_names[0] for name in mriot_names)

        mriot_names, assets = zip(
            *[(shock.mriot_name, shock.exposure_assets) for shock in direct_shocks]
        )
        if not _check_mriot_name(mriot_names):
            raise ValueError("DirectShock do not all have the same mriot_name.")

        concat_assets = pd.concat(assets, join="outer")
        duplicated_idx = concat_assets.index[concat_assets.index.duplicated()]
        duplicated_values = concat_assets.loc[duplicated_idx]
        # This checks that all exposure values related to one (region, sector) are the same.
        # Thus multiple different exposure for different sub-sectors belonging to the same
        # MRIOT sector is not possible.
        if not duplicated_values.groupby(duplicated_values.index).nunique().eq(1).all():
            raise ValueError("Assets present in multiple shocks have different values.")
        else:
            return concat_assets.groupby(level=[0, 1]).first()


class IndirectCostModel(ABC):
    def __init__(
        self, mriot: pymrio.IOSystem, direct_shock: DirectShock | None = None
    ) -> None:
        self.mriot = mriot
        self.conversion_factor = 1.0
        self.shock_model_with(direct_shock)

    def shock_model_with(self, direct_shock: DirectShock | list[DirectShock] | None):
        if direct_shock is None:
            self.direct_shock = direct_shock
        else:
            if isinstance(direct_shock, list):
                direct_shock = DirectShock.combine(direct_shock)

            if self.mriot.name != direct_shock.mriot_name:
                raise ValueError(
                    f"""Cannot shock model with a shock from a different MRIOT:
                     Model {self.mriot.name} != Shock {direct_shock.mriot_name}"""
                )

            self.direct_shock = direct_shock


class StaticIOModel(IndirectCostModel):
    def __init__(
        self, mriot: pymrio.IOSystem, direct_shock: DirectShock | None = None
    ) -> None:
        self.mriot = mriot
        self.conversion_factor = 1.0
        self.shock_model_with(direct_shock)

    def calc_indirect_impacts(self) -> pd.DataFrame:
        def create_df_metrics(df, method, indout, abs_shock):
            # Create absolute production change dataframe
            df_abs = df.melt(ignore_index=False)
            df_abs["metric"] = "absolute production change"
            df_abs["method"] = method

            # Create relative production change dataframe
            df_rel = (df / indout).melt(ignore_index=False)
            df_rel["metric"] = "relative production change"
            df_rel["method"] = method

            df_rel_shock = (df / abs_shock).melt(ignore_index=False).fillna(0.0)
            df_rel_shock["metric"] = "production lost to shock ratio"
            df_rel_shock["method"] = method
            return df_rel, df_abs, df_rel_shock

        # Example of using the function with df_leontief and df_ghosh
        # Leontief
        demand = self.mriot.Y.sum(1)
        shock = self.direct_shock.relative_impact.reindex(
            columns=demand.index, fill_value=0.0
        ).fillna(0.0)
        degraded_demand = -shock * demand

        # Ghosh
        value_added = calc_va(self.mriot.Z, self.mriot.x)
        degraded_va = -shock * value_added.loc["value added"]
        #
        # for i in range(n_events)

        res_leontief = [
            (pymrio.calc_x_from_L(self.mriot.L, degraded_demand.loc[event_id]))[
                "indout"
            ]
            for event_id in shock.index
        ]
        res_ghosh = [
            (calc_x_from_G(self.mriot.G, degraded_va.loc[event_id]))["indout"]
            for event_id in shock.index
        ]

        df_leontief = pd.DataFrame(res_leontief, index=shock.index)
        df_ghosh = pd.DataFrame(res_ghosh, index=shock.index)
        df_leontief_rel, df_leontief_abs, df_leontief_shock = create_df_metrics(
            df_leontief,
            "leontief",
            self.mriot.x["indout"],
            self.direct_shock.impacted_assets,
        )
        df_ghosh_rel, df_ghosh_abs, df_ghosh_shock = create_df_metrics(
            df_ghosh, "ghosh", self.mriot.x["indout"], self.direct_shock.impacted_assets
        )
        res = pd.concat(
            [
                df
                for df in [
                    df_leontief_abs,
                    df_leontief_rel,
                    df_leontief_shock,
                    df_ghosh_abs,
                    df_ghosh_rel,
                    df_ghosh_shock,
                ]
            ],
            axis=0,
        )
        res = res.reset_index().sort_values(["event_id","region","sector", "method", "metric"])[["event_id","region","sector", "method", "metric","value"]]
        res.index = res.index.sort_values()
        return res

class BoARIOModel(IndirectCostModel):
    def __init__(
        self,
        mriot: pymrio.IOSystem,
        direct_shock: DirectShock,
        model_kwargs: dict | None = None,
        simulation_kwargs: dict | None = None,
        event_kwargs: dict | None = None,
    ) -> None:
        self.mriot = mriot
        self.conversion_factor = 1.0
        self.shock_model_with(direct_shock)
        boario_model_params = model_kwargs if model_kwargs is not None else {}
        boario_simulation_params = (
            simulation_kwargs if simulation_kwargs is not None else {}
        )
        boario_event_params = (
            event_kwargs
            if event_kwargs is not None
            else {"recovery_tau": 180, "rebuild_tau": 180}
        )
        self.model = ARIOPsiModel(
            mriot,
            productive_capital_vector=direct_shock.exposure_assets,
            **boario_model_params,
        )
        self.sim = Simulation(
            self.model,
            n_temporal_units_to_sim=int(
                direct_shock.event_dates.max()
                - direct_shock.event_dates.min()
                + SIMULATION_LENGTH_BUFFER
            ),
            **boario_simulation_params,
        )
        for i, date in enumerate(direct_shock.event_dates):
            if (
                direct_shock.impacted_assets.iloc[i]
                > boario_event_module.LOW_DEMAND_THRESH / direct_shock.monetary_factor
            ).any():
                ev = boario_event_module.from_series(
                    event_type="recovery",
                    impact=direct_shock.impacted_assets.iloc[i],
                    occurrence=date - direct_shock.event_dates.min() + 1,
                    event_monetary_factor=direct_shock.monetary_factor,
                    **boario_event_params,
                )
                self.sim.add_event(ev)

    def run_sim(self):
        self.sim.loop()
        return self.sim


# class SupplyChain:
#     """SupplyChain class.

#     The SupplyChain class provides methods for loading Multi-Regional Input-Output
#     Tables (MRIOT) and computing direct, indirect and total impacts.

#     Attributes
#     ----------
#     mriot : pymrio.IOSystem
#             An object containing all MRIOT related info (see also pymrio package):
#                 mriot.Z : transaction matrix, or interindustry flows matrix
#                 mriot.Y : final demand
#                 mriot.x : industry or total output
#                 mriot.meta : metadata
#     secs_exp : pd.DataFrame
#             Exposure dataframe of each region/sector in the MRIOT. Columns are the
#             same as the chosen MRIOT.
#     secs_imp : pd.DataFrame
#             Impact dataframe for the directly affected countries/sectors for each event with
#             impacts. Columns are the same as the chosen MRIOT and rows are the hazard events ids.
#     secs_shock : pd.DataFrame
#             Shocks (i.e. impact / exposure) dataframe for the directly affected countries/sectors
#             for each event with impacts. Columns are the same as the chosen MRIOT and rows are the
#             hazard events ids.
#     inverse : dict
#             Dictionary with keys being the chosen approach (ghosh, leontief)
#             and values the Leontief (L, if approach is leontief) or Ghosh (G, if
#             approach is ghosh) inverse matrix.
#     coeffs : dict
#             Dictionary with keys the chosen approach (ghosh, leontief)
#             and values the Technical (A, if approach is leontief) or allocation
#             (B, if approach is ghosh) coefficients matrix.
#     sim: boario.simulation.Simulation
#             Boario's simulation object. Only relevant when io_approach in "boario_aggregated" or
#             "boario_disaggregated". Default is None.
#     events_date: np.array
#             Integer date corresponding to the proleptic Gregorian ordinal, where January 1 of year
#             1 has ordinal 1 (ordinal format of datetime library) of events leading to impact.
#             Deafult is None.
#     supchain_imp : dict
#             Dictionary with keys the chosen approach (ghosh, leontief or boario
#             and its variations) and values dataframes of production losses (ghosh, leontief)
#             or production dynamics (boario and its variations) to countries/sectors for each event.
#             For each dataframe, columns are the same as the chosen MRIOT and rows are the
#             hazard events' ids.
#     """

#     def __init__(self, mriot):
#         """Initialize SupplyChain.

#         Parameters
#         ----------
#         mriot : pymrio.IOSystem
#                 An object containing all MRIOT related info (see also pymrio package):
#                     mriot.Z : transaction matrix, or interindustry flows matrix
#                     mriot.Y : final demand
#                     mriot.x : industry or total output
#                     mriot.meta : metadata
#         """

#         self.mriot = mriot
#         self.secs_exp = None
#         self.secs_imp = None
#         self.secs_shock = None
#         self.sim = None
#         self.events_date = None
#         self.inverse = dict()
#         self.coeffs = dict()
#         self.supchain_imp = dict()

#     def calc_impacts(
#         self,
#         io_approach,
#         exposure=None,
#         impact=None,
#         impacted_secs=None,
#         shock_factor=None,
#         boario_params=dict(),
#         boario_type="recovery",
#         boario_aggregate="agg",
#     ):
#         """Calculate indirect production impacts based on to the
#         chosen input-output approach.

#         Parameters
#         ----------
#         io_approach : str
#             The adopted input-output modeling approach.
#             Possible choices are 'leontief', 'ghosh' or 'boario'
#             'boario_recovery', 'boario_rebuild' and 'boario_shockprod'.
#         exposure : climada.entity.Exposure
#             CLIMADA Exposure object of direct impact calculation. Default is None.
#         impact : climada.engine.Impact
#             CLIMADA Impact object of direct impact calculation. Default is None.
#         impacted_secs : (range, np.ndarray, str, list)
#             Information regarding the impacted sectors. This can be provided
#             as positions of the impacted sectors in the MRIOT (as range or np.ndarray)
#             or as sector names (as string or list). Default is None.
#         shock_factor : np.array
#             It has length equal to the number of sectors. For each sector, it defines to
#             what extent the fraction of indirect losses differs from the one of direct
#             losses (i.e., impact / exposure). Deafult value is None, which means that shock
#             factors for all sectors are equal to 1, i.e., that production and stock losses
#             fractions are the same.
#         boario_params: dict
#             Dictionary containing parameters to instantiate boario's ARIOPsiModel (key 'model'),
#             Simulation (key 'sim') and Event (key 'event') classes. Parameters instantiating
#             each class need to be stored in a dictionary, e.g., {'model': {}, 'sim': {}, 'event': {}}.
#             You can also specify "show_progress=False" to remove the progress bar during simulations.
#             Only meangingful when io_approach='boario'. Default is None.
#         boario_type: str
#             The chosen boario type. Possible choices are 'recovery', 'rebuild' and
#             'production_shock'. Only meaningful when io_approach='boario'. Default 'recovery'.
#         boario_aggregate: str
#             Whether events are aggregated or not. Possible choices are 'agg' or 'sep'.
#             Only meaningful when io_approach='boario'. Default is 'agg'.

#         Notes
#         -----
#            * The Leontief approach assumes the shock to degrade the final demand,
#            and computes the resulting changed production.
#            * The Ghosh approach assumes the shock to impact value added,
#            and computes the resulting production.
#            * The BoARIO approach assumes the shock to incapacitate productive capital
#            (and possibly generate a reconstruction demand with ``boario_type="rebuild"``)
#            and computes the change of production over time with the ARIO model.
#            See the `BoARIO documentation <https://spjuhel.github.io/BoARIO/>`_ for more details
#            (Note that not all features of BoARIO are included yet).


#         References
#         ----------
#         [1] W. W. Leontief, Output, employment, consumption, and investment,
#         The Quarterly Journal of Economics 58, 1944.
#         [2] Ghosh, A., Input-Output Approach in an Allocation System,
#         Economica, New Series, 25, no. 97: 58-64. doi:10.2307/2550694, 1958.
#         [3] Kitzes, J., An Introduction to Environmentally-Extended Input-Output
#         Analysis, Resources, 2, 489-503; doi:10.3390/resources2040489, 2013.
#         """

#             show_progress = boario_params.get("show_progress", True)
#             for boario_param_type in ["model", "sim"]:
#                 if boario_param_type not in boario_params:
#                     warnings.warn(
#                         f"""BoARIO '{boario_param_type}' parameters were not specified and default values are used. This is not recommended and likely undesired."""
#                     )
#                     boario_params.update({f"{boario_param_type}": {}})

#             if "event" not in boario_params:
#                 if boario_type == "recovery":
#                     boario_params.update({"event": {"recovery_time": 60}})
#                     warnings.warn(
#                         f"BoARIO {boario_type} event parameters were not specified."
#                         "This is not recommended. Default value for `recovery_time` is 60."
#                     )
#                 elif boario_type == "rebuild":
#                     raise ValueError(
#                         """Using the ``boario_type=rebuild`` requires you to define the rebuilding sectors in the ``boario_params`` argument:
#                     {"model":{}, "sim":{}, "event":{"rebuilding_sectors={"<sector_name>":reconstruction_share}}}"""
#                     )

#             # call ARIOPsiModel with default params
#             model = ARIOPsiModel(
#                 self.mriot,
#                 # productive capital vector (i.e. exposure) needs to be in
#                 # MRIOT's unit, this is the case as self.secs_exp was rescaled
#                 # with the conversion_factor upon its construction
#                 productive_capital_vector=self.secs_exp,
#                 # model monetary factor equals the MRIOT's unit
#                 monetary_factor=self.conversion_factor(),
#                 **boario_params["model"],
#             )

#             # run simulation up to one year after the last event
#             self.sim = Simulation(
#                 model,
#                 n_temporal_units_to_sim=int(
#                     self.events_date.max() - self.events_date.min() + 365
#                 ),
#                 **boario_params["sim"],
#             )

#             if boario_type == "recovery":

#                 events_list = [
#                     EventKapitalRecover.from_series(
#                         impact=self.secs_imp.iloc[i],
#                         occurrence=int(
#                             self.events_date[i] - self.events_date.min() + 1
#                         ),
#                         # event monetary factor equals the impact units. self.secs_imp
#                         # was rescaled by the conversion_factor upon its construction so
#                         # we pass the conversion_factor as unit
#                         event_monetary_factor=self.conversion_factor(),
#                         **boario_params["event"],
#                     )
#                     for i in range(n_events)
#                 ]

#             elif boario_type == "rebuild":

#                 events_list = [
#                     EventKapitalRebuild.from_series(
#                         impact=self.secs_imp.iloc[i],
#                         occurrence=(self.events_date[i] - self.events_date.min() + 1),
#                         # event monetary factor equal to the impact units. self.secs_imp
#                         # was rescaled by the conversion_factor upon its construction so
#                         # we pass the conversion_factor as unit
#                         event_monetary_factor=self.conversion_factor(),
#                         **boario_params["event"],
#                     )
#                     for i in range(n_events)
#                 ]

#             # Currently not working in BoARIO.
#             # elif boario_type == 'shockprod':
#             #     events_list = [EventArbitraryProd.from_series(
#             #                             impact=self.secs_shock.iloc[i],
#             #                             occurrence = (self.events_date[i]-self.events_date.min()+1),
#             #                             **boario_params['event']
#             #                 ) for i in range(n_events)
#             #     ]

#             else:
#                 raise RuntimeError(f"Unknown boario type : {boario_type}")

#             if boario_aggregate == "agg":
#                 self.sim.add_events(events_list)
#                 self.sim.loop(progress=show_progress)
#                 self.supchain_imp.update(
#                     {
#                         f"{io_approach}_{boario_type}_{boario_aggregate}": self.sim.production_realised.copy()[
#                             self.secs_imp.columns
#                         ]
#                     }
#                 )

#             elif boario_aggregate == "sep":
#                 results = []
#                 for ev in events_list:
#                     self.sim.add_event(ev)
#                     self.sim.loop(progress=show_progress)
#                     results.append(
#                         self.sim.production_realised.copy()[self.secs_imp.columns]
#                     )
#                     self.sim.reset_sim_full()

#                 self.supchain_imp.update(
#                     {f"{io_approach}_{boario_type}_{boario_aggregate}": results}
#                 )

#             else:
#                 raise RuntimeError(
#                     f"Unknown boario aggregation type: {boario_aggregate}"
#                 )

#         # The calc_matrices() call at the top fails before so this is not usefull
#         # else:
#         #    raise RuntimeError(f"Unknown io_approach: {io_approach}")

#     def conversion_factor(self):
#         """Conversion factor based on unit specified in the Multi-Regional Input-Output Table."""

#         unit = None
#         if isinstance(self.mriot.unit, pd.DataFrame):
#             unit = self.mriot.unit.values[0][0]
#         elif isinstance(self.mriot.unit, str):
#             unit = self.mriot.unit
#         if unit in ["M.EUR", "Million USD", "M.USD"]:
#             conversion_factor = 1e6
#         else:
#             conversion_factor = 1
#             warnings.warn(
#                 "No known unit was provided. It is assumed that values do not need to "
#                 "be converted."
#             )
#         return conversion_factor

#     def map_exp_to_mriot(self, exp_regid, mriot_type):
#         """
#         Map regions names in exposure into Input-Output regions names.
#         exp_regid must follow ISO 3166 numeric country codes.
#         """

#         if mriot_type == "EXIOBASE3":
#             mriot_reg_name = u_coord.country_to_iso(exp_regid, "alpha2")

#         elif mriot_type in ["WIOD16", "OECD21"]:
#             mriot_reg_name = u_coord.country_to_iso(exp_regid, "alpha3")

#         else:
#             warnings.warn(
#                 "For a correct calculation the format of regions' names in exposure and "
#                 "the IO table must match."
#             )
#             return exp_regid

#         idx_country = np.where(self.mriot.get_regions() == mriot_reg_name)[0]

#         if not idx_country.size > 0.0:
#             mriot_reg_name = "ROW"

#         return mriot_reg_name
