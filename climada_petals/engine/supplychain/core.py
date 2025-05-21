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

Define the main supply chain classes: DirectShocksSet and models classes.
"""

from __future__ import (
    annotations,
)
from abc import (
    ABC,
)

from climada_petals.engine.supplychain.mriot_handling import lexico_reindex

__all__ = ["DirectShocksSet", "IndirectCostModel", "BoARIOModel"]

import logging
from typing import Any, Dict, Iterable, Literal, Sequence
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
    translate_exp_to_regions,
    translate_exp_to_sectors,
    translate_imp_to_regions,
    _thin_to_wide,
)

LOGGER = logging.getLogger(__name__)

MRIOT_DIRECTORY = CONFIG.engine.supplychain.local_data.mriot.dir()
"""Directory where Multi-Regional Input-Output Tables (MRIOT) are downloaded."""

VA_NAME = "value added"
"""Index name for value added"""

SIMULATION_LENGTH_BUFFER = 365
"""Default buffer for BoARIO simulation length (in days)"""


class DirectShocksSet:
    """DirectShocksSet class

    The DirectShocksSet class provides methods for 'translating' Impact and Exposure objects into
    economic shocks, corresponding to a specific MRIOT table, from which indirect economic
    costs can be computed.

    Attributes
    ----------

    mriot_name: str
            The name of the MRIOT defining the typology of assets.
    mriot_sectors: pd.Index
            The list of possible sectors.
    mriot_regions: pd.Index
            The list of possible regions.
    name: str, default "unnamed"
            A name to identify the object. For convenience only.
    monetary_factor: int
            The monetary factor of the correponding MRIOT.
    exposure_assets: pd.Series
            Exposure translated in the region/sector typology of the MRIOT. The index of the Series
            are the possible (region, sector).
    impacted_assets: pd.DataFrame
            Impact translated in the region/sector typology of the MRIOT. The columns index are the possible (region, sector),
            and the row index correspond to event_ids of the Impact.
    event_dates: pd.Series
            Series of the dates of the events, as ordinals, with event ids as index.

    """

    def __init__(
        self,
        mriot_name: str,
        mriot_sectors: pd.Index,
        mriot_regions: pd.Index,
        exposure_assets: pd.Series,
        impacted_assets: pd.DataFrame,
        event_dates: np.ndarray,
        monetary_factor: int,
        shock_name: str = "unnamed",
    ) -> None:
        # Sanity checks
        if not isinstance(exposure_assets, pd.Series):
            raise ValueError("Exposure assets must be a pandas Series")

        self.mriot_sectors = mriot_sectors
        self.mriot_regions = mriot_regions
        self.mriot_industries = pd.MultiIndex.from_product(
            [self.mriot_regions, self.mriot_sectors], names=["region", "sector"]
        )

        if not (
            not_present := exposure_assets.index.difference(self.mriot_industries)
        ).empty:
            raise ValueError(
                f"Exposure assets indices do not match MRIOT industries.\nNot matching: {not_present}"
            )

        if not isinstance(impacted_assets, pd.DataFrame):
            raise ValueError("Impacted assets must be a pandas DataFrame")

        if not (
            not_present := impacted_assets.columns.difference(self.mriot_industries)
        ).empty:
            raise ValueError(
                f"Impacted assets columns do not match MRIOT industries.\nNot matching: {not_present}"
            )

        self.event_ids = impacted_assets.index
        if not event_dates.size == self.event_ids.size:
            raise ValueError(
                f"Number of events mismatch between dates ({event_dates.size} events) and impacted assets ({self.event_ids.size} events)"
            )

        if not (
            not_present := impacted_assets.columns.difference(exposure_assets.index)
        ).empty:
            warnings.warn(
                f"Some impacted assets do not have a corresponding exposure value ({not_present}). The impact will not be considered."
            )

        self.mriot_name = mriot_name
        self.monetary_factor = monetary_factor
        self.name = shock_name

        self.exposure_assets = _thin_to_wide(exposure_assets, self.mriot_industries)
        self.exposure_assets.sort_index(inplace=True)
        self.impacted_assets = _thin_to_wide(
            impacted_assets, self.event_ids, self.mriot_industries
        )
        self.impacted_assets = self.impacted_assets.reindex(
            sorted(self.impacted_assets.columns), axis=1
        )
        self.event_dates = pd.Series(event_dates, index=self.event_ids)

    @classmethod
    def _init_with_mriot(
        cls,
        mriot: pymrio.IOSystem,
        exposure_assets: pd.Series,
        impacted_assets: pd.DataFrame,
        event_dates: np.ndarray,
        shock_name: str,
    ):
        mriot = lexico_reindex(mriot)
        return cls(
            mriot.name,
            mriot.get_sectors(),
            mriot.get_regions(),
            exposure_assets,
            impacted_assets,
            event_dates,
            mriot.monetary_factor,
            shock_name,
        )

    @classmethod
    def from_exp_and_imp(
        cls,
        mriot: pymrio.IOSystem,
        exposure: Exposures,
        impact: Impact,
        affected_sectors: Iterable[str] | dict[str, float] | pd.Series | Literal["all"],
        impact_distribution: dict[str, float] | pd.Series | None,
        shock_name: str | None = None,
        exp_value_col: str = "value",
        custom_mriot: bool = False,
    ):
        """Build a DirectShocksSet from MRIOT, Exposure and Impact objects.

        This method translates both the given Exposure and Impact objects to the MRIOT typology.pd.Series
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
        shock_name : str | None
            An optional name to identify the object, defaults to "unnamed".
        affected_sectors : Iterable[str] | dict[str, float] | pd.Series | Literal["all"]
            The sectors of the MRIOT that are impacted. If given as a
            collection of string, or `"all"`, then the total assets of the
            region are distributed proportionally to each sectors gross output.
            A dictionary `sector:share` can also be passed to specify which
            share of the total regional assets should be distributed to each
            sector.
        impact_distribution : dict[str, float] | pd.Series, optional
            This argument specify how the impact per region should be
            distributed to the impacted sectors. Using `None` will distribute
            proportionally to each sectors gross output in the MRIOT. A
            dictionary in the form `sector:share` or similarly a `Series` can
            be used to specify a custom distribution.
        exp_value_col : str
            The name of the column of the Exposure data representing the value
            of assets in each centroids.
        custom_mriot : bool
            Whether to consider the MRIOT as a custom one (skips name checking), defaults to False.
            Note that its regions has to be ISO3 countries name for it to work.

        """
        mriot = lexico_reindex(mriot)
        exp = translate_exp_to_regions(
            exposure, mriot_name=mriot.name, custom_mriot=custom_mriot
        )
        exposure_assets = translate_exp_to_sectors(
            exp, affected_sectors=affected_sectors, mriot=mriot, value_col=exp_value_col
        )
        return cls.from_assets_and_imp(
            mriot=mriot,
            exposure_assets=exposure_assets,
            impact=impact,
            shock_name=shock_name,
            affected_sectors=affected_sectors,
            impact_distribution=impact_distribution,
            custom_mriot=custom_mriot,
        )

    @classmethod
    def from_assets_and_imp(
        cls,
        mriot: pymrio.IOSystem,
        exposure_assets: pd.Series,
        impact: Impact,
        shock_name: str | None,
        affected_sectors: Iterable[str] | dict[str, float] | pd.Series | Literal["all"],
        impact_distribution: dict[str, float] | pd.Series | None,
        custom_mriot: bool = False,
    ):
        """Build a DirectShocksSetfrom an MRIOT, assets Series, and Impact objects.

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
        shock_name : str or None
            An optional name to identify the object.
        affected_sectors : Iterable[str] | dict[str, float] | pd.Series | Literal["all"]
            The sectors of the MRIOT that are impacted. If given as a
            collection of string, or `"all"`, then the total assets of the
            region are distributed proportionally to each sectors gross output.
            A dictionary `sector:share` can also be passed to specify which
            share of the total regional assets should be distributed to each
            sector.
        impact_distribution : dict[str, float] | pd.Series, optional
            This argument specify how the impact per region should be
            distributed to the impacted sectors. Using `None` will distribute
            proportionally to each sectors gross output in the MRIOT. A
            dictionary in the form `sector:share` or similarly a `Series` can
            be used to specify a custom distribution.
        custom_mriot : bool
            Whether to consider the MRIOT as a custom one (skips name checking), defaults to False.
            Note that its regions has to be ISO3 countries name for it to work.

        """
        mriot = lexico_reindex(mriot)
        event_dates = impact.date

        # get regional impact in MRIOT format
        impacted_assets = impact.impact_at_reg()
        impacted_assets = impacted_assets.loc[:, (impacted_assets != 0).any()]
        impacted_assets = translate_imp_to_regions(
            impacted_assets, mriot, custom_mriot=custom_mriot
        )

        # Setup distribution toward sectors
        if isinstance(affected_sectors, str) and affected_sectors == "all":
            affected_sectors = list(mriot.get_sectors())  # type: ignore

        if impact_distribution is None:
            # Default uses production distribution across sectors, region.
            impact_distribution = mriot.x.loc[
                pd.IndexSlice[impacted_assets.columns, affected_sectors],
                mriot.x.columns[0],
            ]

        if isinstance(impact_distribution, dict):
            impact_distribution = pd.Series(impact_distribution)

        if not isinstance(impact_distribution, pd.Series):
            raise ValueError(f"Impact_distribution could not be converted to a Series")

        impact_distribution = impact_distribution.sort_index()

        impacted_assets = distribute_reg_impact_to_sectors(
            impacted_assets, impact_distribution
        )
        impacted_assets.rename_axis(index="event_id", inplace=True)
        # Call constructor
        return cls._init_with_mriot(
            mriot=mriot,
            exposure_assets=exposure_assets,
            impacted_assets=impacted_assets,
            event_dates=event_dates,
            shock_name=shock_name,
        )

    @property
    def event_ids_with_impact(self) -> pd.Index:
        """
        Get the IDs of events that have a non-zero impact on assets.

        This property filters the `impacted_assets` DataFrame to retrieve the indices
        of rows where at least one asset has a non-zero impact.

        Returns
        -------
        pandas.Index
            Index of events with at least one impacted asset.
        """
        return self.impacted_assets.loc[self.impacted_assets.ne(0).any(axis=1)].index

    @property
    def impacted_assets_not_null(self) -> pd.DataFrame:
        """
        Get a subset of the `impacted_assets` DataFrame containing only non-null impacts.

        This property filters the `impacted_assets` DataFrame to include only rows and
        columns where at least one non-zero impact exists.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the impacted assets with non-zero values.
        """
        return self.impacted_assets.loc[
            self.impacted_assets.ne(0).any(axis=1),
            self.impacted_assets.ne(0).any(axis=0),
        ]

    @property
    def exposure_assets_not_null(self) -> pd.Series:
        """
        Get a subset of the `exposure_assets` DataFrame containing only non-null exposure.

        This property filters the `exposure_assets` DataFrame to include only rows with non-zero values.

        Returns
        -------
        pandas.Series
            A Series containing the impacted assets with non-zero values.
        """
        return self.exposure_assets.loc[self.exposure_assets.ne(0)]

    @property
    def relative_impact(self) -> pd.DataFrame:
        """The ratio of impacted assets over total assets (0. if total assets are 0.)."""
        return (self.impacted_assets / self.exposure_assets).fillna(0.0).replace(
            [np.inf, -np.inf], 0
        ) * (self.exposure_assets > 0)

    @property
    def relative_impact_not_null(self) -> pd.DataFrame:
        """
        Get a subset of the `relative_impact` DataFrame containing only non-null values.

        This property filters the `relative_impact` DataFrame to include only rows and
        columns where at least one non-zero impact exists.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the impacted assets with non-zero values.
        """
        return self.relative_impact.loc[
            self.relative_impact.ne(0).any(axis=1),
            self.relative_impact.ne(0).any(axis=0),
        ]


    @staticmethod
    def _check_mriot_name(mriot_names: Sequence[str]) -> bool:
        """Checks that a list of MRIOT names are all identical"""
        return all(name == mriot_names[0] for name in mriot_names)

    @staticmethod
    def _combine_event_dates(event_dates: list[pd.Series]) -> pd.Series:
        """Combines multiple series of event dates into one"""
        combined_event_dates = pd.concat(event_dates)
        combined_event_dates = combined_event_dates.groupby(level=0).transform(
            lambda x: x.iloc[0] if x.nunique() == 1 else "conflicting value"
        )
        if "conflicting value" in combined_event_dates.values:
            raise ValueError(
                f"Conflicting values for event dates for events: {combined_event_dates[combined_event_dates=='conflicting value'].index.unique()}"
            )
        combined_event_dates = combined_event_dates.groupby(level=0).first()
        return combined_event_dates

    @staticmethod
    def _combine_exp_assets(assets: Sequence[pd.Series]) -> pd.Series:
        """Combines exposure assets together"""
        if not all(asset.index.equals(assets[0].index) for asset in assets):
            raise ValueError("Mismatching indexes amongst the different assets.")
        assets_no0 = [asset.loc[asset > 0.0] for asset in assets]
        concat_assets = pd.concat(assets_no0, join="outer")
        duplicated_idx = concat_assets.index[concat_assets.index.duplicated()]
        duplicated_values = concat_assets.loc[duplicated_idx]
        # This checks that all exposure values related to one (region, sector) are the same.
        # Thus multiple different exposure for different sub-sectors belonging to the same
        # MRIOT sector is not possible.
        if not duplicated_values.groupby(duplicated_values.index).nunique().eq(1).all():
            raise ValueError("Conflicting values amongst the different assets.")
        else:
            return _thin_to_wide(
                concat_assets.groupby(level=[0, 1]).first(), assets[0].index
            )

    @staticmethod
    def _merge_imp_assets(imp_assets: Sequence[pd.DataFrame]):
        """Merges impacted assets together (summing impacts on same industries)."""
        merged_imp_assets = pd.concat(
            imp_assets,
            join="outer",
            sort=True,
            axis=1,
        ).fillna(0)
        merged_imp_assets = merged_imp_assets.groupby(merged_imp_assets.index).sum()
        merged_imp_assets = (
            merged_imp_assets.T.groupby(merged_imp_assets.columns).sum().T
        )
        return merged_imp_assets

    @classmethod
    def combine(
        cls,
        direct_shocks: list[DirectShocksSet],
        kind: Literal["merge", "concat"] = "merge",
        combine_name: str = "unnamed_combine",
    ):
        """
        Combine multiple `DirectShocksSet` instances into a single instance.

        This method supports two modes of combination: merging and concatenating.
        - In "merge" mode, the direct shocks are summed together for each matching event
          and per region/sector.
        - In "concat" mode, the direct shocks are assumed to originate from
          independent events and are concatenated along the event axis.

        Parameters
        ----------
        direct_shocks : list of DirectShocksSet
            A list of `DirectShocksSet` instances to be combined.
        kind : {"merge", "concat"}, optional (Default: "merge")
            The type of combination to apply.
            - "merge": Sum the shocks across all events.
            - "concat": Concatenate shocks assuming independent events.
        combine_name : str, optional
            Name for the combined `DirectShocksSet`. Default is "unnamed_combine".

        Returns
        -------
        DirectShocksSet
            A new `DirectShocksSet` instance representing the combination of the input shocks.

        Raises
        ------
        ValueError
            If the `DirectShocksSet` instances do not share the same `mriot_name`.

        Notes
        -----
        - All `DirectShocksSet` instances must share the same `mriot_name`.
        - Merging sums the shocks and combines the corresponding impacted and exposed assets.
        - Concatenating assumes independent events and stacks the shocks along the event axis.

        Examples
        --------

        Assuming:
        - `mriot` is an IOSystem object representing a MRIOT with (at least) Agriculture and Manufacture sectors.
        - `exp_agri` contains exposure data for the Agriculture sector
        - `exp_manu` contains exposure data for the for the Manufacture sector.
        - `imp_TC_agri` is an Impact object with impacts on the agriculture sector from different tropical cyclones
        - `imp_TC_manu` is an Impact object with impacts on the manufacture sector from different tropical cyclones
        - `imp_FL_agri` is an Impact object with impacts on the manufacture sector from different floods on the agriculture sector
        - `imp_TC_surges_manu` is an Impact object with impacts from surges that occurred during the same tropical cyclones as defined in `imp_TC_manu`

        >>> shocks_TC_agri = DirectShocksSet.from_assets_and_imp(mriot=mriot, exposure=exp_agri, impact=imp_TC_agri, affected_sectors=["Agriculture"])
        >>> shocks_TC_manu = DirectShocksSet.from_assets_and_imp(mriot=mriot, exposure=exp_manu, impact=imp_TC_manu, affected_sectors=["Manufacture"])
        >>> shocks_FL_agri = DirectShocksSet.from_assets_and_imp(mriot=mriot, exposure=exp_agri, impact=imp_FL_agri, affected_sectors=["Agriculture"])
        >>> shocks_TC_surges_manu = DirectShocksSet.from_assets_and_imp(mriot=mriot, exposure=exp_manu, impact=imp_TC_surges_agri, affected_sectors=["Manufacture"])

        Assume you want to compute the impacts from TCs on both Agriculture and Manufacture sectors, you can "merge" `shocks_TC_agri` and `shocks_TC_manu` for that purpose:

        >>> combined_set = DirectShocksSet.combine([shock_TC_agri, shock_TC_manu], kind="merge")

        If you want to evaluate the perturbations from TCs and floods on the Agriculture sectors, you can "concat" `shocks_TC_agri` and `shocks_FL_agri` for that purpose:

        >>> combined_set = DirectShocksSet.combine([shocks_TC_agri, shocks_FL_agri], kind="concat", combine_name="impact from floods and TCs")

        Finally if you want to look at the shocks of "co-occurence" of surges and TCs for which the impacts were computed independently, you can merge `shocks_TC_manu` and `shocks_TC_surge_manu`.

        Of course you can use more that one sectors at once per individual DirectShocksSet. The purpose of this `combine` method is to allow flexibility with employing different exposure layers and impact computation.
        """
        if not len(direct_shocks) > 1:
            return direct_shocks[0]

        assets = [shock.exposure_assets for shock in direct_shocks]
        combined_exp_assets = cls._combine_exp_assets(assets)
        imp_assets = [shock.impacted_assets for shock in direct_shocks]
        mriot_names = [shock.mriot_name for shock in direct_shocks]
        if not DirectShocksSet._check_mriot_name(mriot_names):
            raise ValueError("DirectShocksSetdo not all have the same mriot_name.")
        if kind == "merge":
            event_dates = cls._combine_event_dates(
                [shock.event_dates for shock in direct_shocks]
            )
            LOGGER.info(
                "Merging direct shocks together. The resulting direct shock per event and per each region,sector will be the sum of the different direct shocks."
            )
            merged_imp_assets = cls._merge_imp_assets(imp_assets)
            return cls(
                direct_shocks[0].mriot_name,
                direct_shocks[0].mriot_sectors,
                direct_shocks[0].mriot_regions,
                combined_exp_assets,
                merged_imp_assets,
                event_dates,
                direct_shocks[0].monetary_factor,
                combine_name,
            )

        if kind == "concat":
            LOGGER.info(
                "Concatenating direct shocks. This assume the different direct shocks are from independent events."
            )

            event_dates = pd.concat([shock.event_dates for shock in direct_shocks])
            concatenated_imp_assets = pd.concat(
                imp_assets,
                axis=0,
                sort=True,
            ).fillna(0)
            return cls(
                direct_shocks[0].mriot_name,
                direct_shocks[0].mriot_sectors,
                direct_shocks[0].mriot_regions,
                direct_shocks[0].exposure_assets,
                concatenated_imp_assets,
                event_dates,
                direct_shocks[0].monetary_factor,
                combine_name,
            )


class IndirectCostModel(ABC):
    """
    Abstract base class for modeling indirect costs using a Multi-Regional Input-Output Table (MRIOT).

    This class provides a structure for incorporating direct shocks into an indirect cost model.

    Parameters
    ----------
    mriot : pymrio.IOSystem
        The Multi-Regional Input-Output System used in the model.
    direct_shocks : DirectShocksSet or None, optional
        The direct shocks to apply to the model. If provided, the model will
        initialize with these shocks. Default is None.

    Attributes
    ----------
    mriot : pymrio.IOSystem
        The Multi-Regional Input-Output System used in the model.
    direct_shocks : DirectShocksSet or None
        The direct shocks applied to the model. None if no shocks have been applied.

    """

    def __init__(
        self,
        mriot: pymrio.IOSystem,
        direct_shocks: DirectShocksSet | None = None,
    ) -> None:
        self.mriot = mriot
        self.direct_shocks = None
        if direct_shocks is not None:
            self.shock_model_with(direct_shocks)

    def shock_model_with(
        self,
        direct_shocks: DirectShocksSet | list[DirectShocksSet],
        combine_mode: Literal["concat"] | Literal["merge"] = "concat",
    ):
        """
        Apply direct shocks to the model, combining them if multiple shocks are provided.

        Parameters
        ----------
        direct_shocks : DirectShocksSet or list of DirectShocksSet
            The direct shocks to apply to the model. If a list is provided, the shocks
            will be combined based on the `combine_mode`.
        combine_mode : {"concat", "merge"}, optional
            The method for combining multiple shocks. Default is "concat".
            - "concat": Concatenate shocks assuming independent events.
            - "merge": Merge shocks by summing their impacts for matching (event,region,sector).

        Raises
        ------
        ValueError
            If the MRIOT name of the provided shocks does not match the model's MRIOT name.

        Notes
        -----
        - The `combine_mode` is only applicable if `direct_shocks` is provided as a list.
        - The shocks must correspond to the same MRIOT used in the model.
        """
        if isinstance(direct_shocks, list):
            direct_shocks = DirectShocksSet.combine(direct_shocks, combine_mode)

        if self.mriot.name != direct_shocks.mriot_name:
            raise ValueError(
                f"""Cannot shock model with a shock from a different MRIOT:
                 Model {self.mriot.name} != Shock {direct_shocks.mriot_name}"""
            )

        self.direct_shocks = direct_shocks


class StaticIOModel(IndirectCostModel):
    """
    Static Input-Output Model for analyzing indirect economic impacts.

    Extends and makes concrete the IndirectCostModel abstract class by providing methods to calculate
    degraded economic metrics and indirect impacts using Leontief and Ghosh models.

    Parameters
    ----------
    mriot : pymrio.IOSystem
        The Multi-Regional Input-Output System used in the model.
    direct_shocks : DirectShocksSet or None, optional
        The direct shocks to apply to the model. Default is None.

    Attributes
    ----------
    mriot : pymrio.IOSystem
        The Multi-Regional Input-Output System used in the model.
    direct_shocks : DirectShocksSet or None
        The direct shocks applied to the model. None if no shocks have been applied.
    """

    def __init__(
        self,
        mriot: pymrio.IOSystem,
        direct_shocks: DirectShocksSet | None = None,
    ) -> None:
        super().__init__(mriot, direct_shocks)

    @property
    def value_added(self) -> pd.Series:
        """
        Calculates and returns the value added from the MRIOT.

        Returns
        -------
        pd.Series
            A series representing the value added by sector.
        """
        return calc_va(self.mriot.Z, self.mriot.x).loc[VA_NAME]

    @property
    def final_demand(self) -> pd.Series:
        """
        Retrieves the (total) final demand addressed to
        each sectors from the MRIOT.

        Returns
        -------
        pd.Series
            A series representing the final demand by sector.
        """
        return self.mriot.Y.sum(1)

    @property
    def degraded_value_added(self) -> pd.DataFrame | pd.Series:
        """
        Calculates and returns the degraded value added considering
        the direct shocks.

        Returns
        -------
        pd.Series
            A series representing the degraded value added by sector.
        """
        if self.direct_shocks:
            return -self.direct_shocks.relative_impact * self.value_added
        else:
            LOGGER.warning("The model currently is not shocked. Returning value added.")
            return self.value_added

    @property
    def degraded_final_demand(self) -> pd.DataFrame | pd.Series:
        """
        Calculates and returns the degraded final demand considering
        the direct shocks.

        Returns
        -------
        pd.Series
            A series representing the degraded final demand by sector.
        """
        if self.direct_shocks:
            return -self.direct_shocks.relative_impact * self.final_demand
        else:
            LOGGER.warning(
                "The model currently is not shocked. Returning final demand."
            )
            return self.final_demand

    def calc_leontief(
        self, event_ids: list[int] | pd.Index | None = None
    ) -> pd.DataFrame:
        """
        Computes indirect impacts using the Leontief model.

        Parameters
        ----------
        event_ids : list[int] or pd.Index or None, optional
            A list of event IDs to calculate impacts for. If None, all events are used.

        Returns
        -------
        pd.DataFrame
            A DataFrame of indirect outputs for the specified events.
        """
        if self.direct_shocks:
            if event_ids is None:
                event_ids = self.direct_shocks.event_ids
            res_leontief = [
                (
                    pymrio.calc_x_from_L(
                        self.mriot.L, self.degraded_final_demand.loc[event_id]
                    )
                )[self.mriot.x.columns[0]]
                for event_id in event_ids
            ]
            return pd.DataFrame(res_leontief, index=event_ids)
        else:
            LOGGER.warning(
                "The model currently is not shocked. Returning empty DataFrame."
            )
            return pd.DataFrame()

    def calc_ghosh(self, event_ids: list[int] | pd.Index | None = None) -> pd.DataFrame:
        """
        Computes indirect impacts using the Ghosh model.

        Parameters
        ----------
        event_ids : list[int] or pd.Index or None, optional
            A list of event IDs to calculate impacts for. If None, all events are used.

        Returns
        -------
        pd.DataFrame
            A DataFrame of indirect outputs for the specified events.
        """
        if self.direct_shocks:
            if event_ids is None:
                event_ids = self.direct_shocks.event_ids
            res_ghosh = [
                (self.degraded_value_added.loc[event_id].dot(self.mriot.G))
                for event_id in event_ids
            ]
            return pd.DataFrame(res_ghosh, index=event_ids)
        else:
            LOGGER.warning(
                "The model currently is not shocked. Returning empty DataFrame."
            )
            return pd.DataFrame()

    def calc_indirect_impacts(
        self,
        event_ids: list[int] | pd.Index | Literal["with_impact"] | None = "with_impact",
    ) -> pd.DataFrame | None:
        """
        Calculates detailed indirect impacts using both Leontief and Ghosh models.

        Parameters
        ----------
        event_ids : "with_impact" or list[int] or pd.Index or None, default "with_impact"
            A list of event IDs to calculate impacts for. Only events with non null impacts by default. If None, all events are used.

        Returns
        -------
        pd.DataFrame or None
            A DataFrame containing indirect impacts with various metrics and methods,
            or None if no shocks are defined for the model.
        """

        def create_df_metrics(event_ids, method, indout, abs_shock):
            if event_ids == "with_impact":
                event_ids = self.direct_shocks.event_ids_with_impact

            if method == "leontief":
                df = self.calc_leontief(event_ids)
            elif method == "ghosh":
                df = self.calc_ghosh(event_ids)
            else:
                raise ValueError(f"Unrecognized methods: {method}")

            if event_ids is not None:
                abs_shock = abs_shock.loc[event_ids]

            # Create absolute production change dataframe
            df_abs = df.melt(ignore_index=False)
            df_abs["metric"] = "absolute production change"
            df_abs["method"] = method

            # Create relative production change dataframe (filtering null production)
            df_rel = ((df / indout) * (indout > 0.0)).melt(ignore_index=False)
            df_rel["metric"] = "relative production change"
            df_rel["method"] = method

            # Create relative to total direct shock dataframe
            df_rel_shock = (
                (df.div(abs_shock.sum(axis=1), axis=0))
                .melt(ignore_index=False)
                .fillna(0.0)
                .replace([np.inf, -np.inf], 0)
            )
            df_rel_shock["metric"] = "production lost to total shock ratio"
            df_rel_shock["method"] = method

            # Create relative to sector direct shock dataframe
            df_rel_sec_shock = (
                (df / abs_shock)
                .melt(ignore_index=False)
                .fillna(0.0)
                .replace([np.inf, -np.inf], 0)
            )
            df_rel_sec_shock["metric"] = "production lost to sector shock ratio"
            df_rel_sec_shock["method"] = method

            return df_rel, df_abs, df_rel_shock, df_rel_sec_shock

        if self.direct_shocks is None:
            LOGGER.warning("The model has no shock defined, returning None.")
            return None

        df_leontief_rel, df_leontief_abs, df_leontief_shock, df_leontief_sec_shock = (
            create_df_metrics(
                event_ids,
                "leontief",
                self.mriot.x.iloc[:, 0],
                self.direct_shocks.impacted_assets,
            )
        )
        df_ghosh_rel, df_ghosh_abs, df_ghosh_shock, df_ghosh_sec_shock = (
            create_df_metrics(
                event_ids,
                "ghosh",
                self.mriot.x.iloc[:, 0],
                self.direct_shocks.impacted_assets,
            )
        )
        res = pd.concat(
            [
                df
                for df in [
                    df_leontief_abs,
                    df_leontief_rel,
                    df_leontief_shock,
                    df_leontief_sec_shock,
                    df_ghosh_abs,
                    df_ghosh_rel,
                    df_ghosh_shock,
                    df_ghosh_sec_shock,
                ]
            ],
            axis=0,
        )
        res = res.reset_index().sort_values(
            ["event_id", "region", "sector", "method", "metric"]
        )[["event_id", "region", "sector", "method", "metric", "value"]]
        res.index = res.index.sort_values()
        return res


class BoARIOModel(IndirectCostModel):
    """
    BoARIO Model for simulating the economic impacts of shocks using the ARIO model.

    Extends and makes concrete the IndirectCostModel abstract class
    by integrating the ARIOPsiModel and Simulation classes
    to simulate the effects of direct shocks over time, with customizable event parameters.

    Parameters
    ----------
    mriot : pymrio.IOSystem
        The Multi-Regional Input-Output System used in the model.
    direct_shocks : DirectShocksSet
        The direct shocks to apply to the model.
    model_kwargs : dict, optional
        A dictionary of model parameters to be passed to the ARIOPsiModel.
    simulation_kwargs : dict, optional
        A dictionary of simulation parameters to be passed to the Simulation class.
    event_kwargs : dict, optional
        A dictionary of event-specific parameters (e.g., recovery time) to be passed
        to the event creation logic.

    Attributes
    ----------
    mriot : pymrio.IOSystem
        The Multi-Regional Input-Output System used in the model.
    model : ARIOPsiModel
        The ARIO model used to represent the economy.
    sim : Simulation
        The simulation object used to run the model.
    direct_shocks : DirectShocksSet or None
        The direct shocks applied to the model.

    Notes
    -----
    We highly recommend users to go through `BoARIO's documentation <https://spjuhel.github.io/BoARIO>`

    """

    def __init__(
        self,
        mriot: pymrio.IOSystem,
        direct_shocks: DirectShocksSet,
        model_kwargs: Dict[str, Any] | None = None,
        simulation_kwargs: Dict[str, Any] | None = None,
        event_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        self.mriot = mriot

        # Set the different parameters from defaults and user given.
        default_event_params = {"recovery_tau": 180, "rebuild_tau": 180}
        event_params = {**default_event_params,**(event_kwargs or {})}
        default_model_params = {}
        model_params = {**default_model_params,**(model_kwargs or {})}
        default_sim_params = {}
        sim_params = {**default_sim_params,**(simulation_kwargs or {})}

        # Instantiate the BoARIO objects
        model = ARIOPsiModel(
            self.mriot,
            productive_capital_vector=direct_shocks.exposure_assets,
            **model_params,
        )
        self.sim = Simulation(
            model,
            n_temporal_units_to_sim=int(
                direct_shocks.event_dates.max()
                - direct_shocks.event_dates.min()
                + SIMULATION_LENGTH_BUFFER
            ),
            **sim_params,
        )

        self.shock_model_with(direct_shocks, event_kwargs=event_params)

    def shock_model_with(
        self,
        direct_shocks: DirectShocksSet | list[DirectShocksSet],
        combine_mode: Literal["concat"] | Literal["merge"] = "concat",
        event_kwargs: Dict[str, Any] | None = None,
    ):
        """
        Apply direct shocks to the model, adding events for recovery if necessary.

        Parameters
        ----------
        direct_shocks : DirectShocksSet or list of DirectShocksSet
            The direct shocks to apply to the model. If a list is provided, the shocks
            will be combined based on the `combine_mode`.
        combine_mode : {"concat", "merge"}, optional
            The method for combining multiple shocks. Default is "concat".
            - "concat": Concatenate shocks assuming independent events.
            - "merge": Merge shocks by summing their impacts.
        event_kwargs : dict, optional
            A dictionary of event-specific parameters to be passed to event creation.
            See BoARIO documentation on `Events <https://spjuhel.github.io/BoARIO/tutorials/boario-events.html>`
        """
        default_boario_event_params = {"recovery_tau": 180, "rebuild_tau": 180}
        event_params = {**default_boario_event_params, **(event_kwargs or {})}
        if direct_shocks is not None:
            super().shock_model_with(direct_shocks, combine_mode)
        if self.direct_shocks is not None:
            skipped = []
            for date, event_id in zip(
                self.direct_shocks.event_dates, self.direct_shocks.event_ids
            ):
                if (
                    self.direct_shocks.impacted_assets.loc[event_id]
                    > boario_event_module.LOW_DEMAND_THRESH
                    / self.direct_shocks.monetary_factor
                ).any():
                    ev = boario_event_module.from_series(
                        event_type="recovery",
                        impact=self.direct_shocks.impacted_assets.loc[event_id],
                        occurrence=date - self.direct_shocks.event_dates.min() + 5,
                        event_monetary_factor=self.direct_shocks.monetary_factor,
                        name=f"Event {event_id}",
                        **event_params,
                    )
                    self.sim.add_event(ev)
                else:
                    skipped.append(event_id)

            LOGGER.warning(
                f"Impact for following events was too small to have an effect, skipping them for efficiency: {skipped}"
            )

        else:
            self.direct_shocks = None

    def run_sim(self):
        """
        Run the simulation for the model.

        Executes the simulation loop and returns the simulation object.

        Returns
        -------
        Simulation
            The simulation object after running the simulation loop.
        """
        self.sim.loop()
        return self.sim
