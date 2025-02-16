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

Define utility functions for the supplychain module.
"""

import copy
from typing import Iterable, Literal, overload
import pandas as pd
import numpy as np
import pymrio
import country_converter as coco

from climada.entity import Exposures
from climada_petals.engine.supplychain.mriot_handling import (
    VA_NAME,
    check_sectors_in_mriot,
    _get_coco_MRIOT_name,
)


def calc_va(
    Z: pd.DataFrame | np.ndarray, x: pd.DataFrame | np.ndarray
) -> pd.DataFrame | np.ndarray:
    """Calculate value added (v) from Z and x

    value added = industry output (x) - inter-industry inputs (sum_rows(Z))

    Parameters
    ----------
    Z : pandas.DataFrame or numpy.array
        Symmetric multi-regional input output table (flows)
    x : pandas.DataFrame or numpy.array
        industry output

    Returns
    -------
    pandas.DataFrame or numpy.array
        Value added va as row vector

    Notes
    -----
    This function adapts pymrio.tools.iomath.calc_x to compute
    value added (va).
    """

    value_added = np.diff(np.vstack((Z.sum(0), x.T)), axis=0)
    if isinstance(Z, pd.DataFrame):
        value_added = pd.DataFrame(value_added, columns=Z.index, index=[VA_NAME])
    if isinstance(value_added, np.ndarray):
        value_added = pd.DataFrame(value_added, index=[VA_NAME])
    return value_added


def calc_B(
    Z: pd.DataFrame | np.ndarray, x: pd.DataFrame | np.ndarray
) -> pd.DataFrame | np.ndarray:
    """Calculate the B matrix (allocation coefficients matrix)
    from Z matrix and x vector

    Parameters
    ----------
    Z : pandas.DataFrame or numpy.array
        Symmetric multi-regional input output table (flows)
    x : pandas.DataFrame, pandas.Series, or numpy.array
        Industry output column vector

    Returns
    -------
    pandas.DataFrame or numpy.array
        Allocation coefficients matrix B.
        Same type as input parameter ``Z``.

    Notes
    -----
    This function adapts pymrio.tools.iomath.calc_A to compute
    the allocation coefficients matrix B.
    """
    # Convert x to a NumPy array
    x = np.asarray(x)

    # Handle zero values in x
    with np.errstate(divide="ignore"):
        recix = np.where(x == 0, 0, 1 / x).reshape((1, -1))

    # Calculate B matrix
    if isinstance(Z, pd.DataFrame):
        return pd.DataFrame(
            np.transpose(Z.values) * recix, index=Z.index, columns=Z.columns
        )
    else:
        return np.transpose(Z) * recix


def calc_G(B: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Calculate the Ghosh inverse matrix G from B

    G = inverse matrix of (I - B) where I is an identity matrix of same shape as B.
    Note that we define G as the transpose of the Ghosh inverse matrix, so that we can apply the factors of
    production intensities from the left-hand-side for both Leontief and Ghosh attribution. In this way the
    multipliers have the same (vector) dimensions and can be added.

    Parameters
    ----------
    B : pandas.DataFrame or numpy.array
        Symmetric input output table (coefficients)

    Returns
    -------
    pandas.DataFrame or numpy.array
        Ghosh input output table G
        The type is determined by the type of B.
        If DataFrame index/columns as B
    """
    id_matrix = np.eye(B.shape[0])
    if isinstance(B, pd.DataFrame):
        return pd.DataFrame(
            np.linalg.inv(id_matrix - B), index=B.index, columns=B.columns
        )
    else:
        return np.linalg.inv(id_matrix - B)  # G = inverse matrix of (I - B)


def calc_x_from_G(
    G: pd.DataFrame | np.ndarray, va: pd.DataFrame | pd.Series | np.ndarray
) -> pd.DataFrame | np.ndarray:
    """Calculate the industry output x from a v vector and G matrix

    x = G . va

    The industry output x is computed from a value-added vector v

    Parameters
    ----------
    va : pandas.DataFrame or pd.Series or numpy.array
        a row vector of the total final added-value
    G : pandas.DataFrame or numpy.array
        **Transpose** of Ghosh inverse matrix

    Returns
    -------
    pandas.DataFrame or numpy.array
        Industry output x as column vector.
        Same type as input parameter ``G``.

    Notes
    -----
    This function adapts the function pymrio.tools.iomath.calc_x_from_L to
    compute total output (x) from the Ghosh inverse.
    """

    x = G.dot(va.T)
    if isinstance(x, pd.Series):
        x = pd.DataFrame(x)
    if isinstance(x, pd.DataFrame):
        x.columns = ["indout"]
    return x


def translate_exp_to_regions(
    exp: Exposures,
    mriot_name: str,
    custom_mriot: bool = False,
) -> Exposures:
    """
    Creates a region column within the GeoDataFrame of the Exposures object which matches regions
    of an MRIOT. Also compute the share of total mriot region value per centroid.

    Parameters
    ----------
    exp : Exposure
        The Exposure object to modify.
    mriot_type : str
        Type of the MRIOT to convert region_id to. Currently available are:
        ["WIOD"]

    Returns
    -------
    Exposures
        A modified copy.
    """
    exp = copy.deepcopy(exp)
    cc = coco.CountryConverter()
    # Find region names
    exp.gdf["region"] = cc.pandas_convert(
        series=exp.gdf["region_id"], to=_get_coco_MRIOT_name(mriot_name, custom_mriot)  # type: ignore
    ).str.upper()

    # compute distribution per region
    exp.gdf["value_ratio"] = exp.gdf.groupby("region")["value"].transform(
        lambda x: x / x.sum()
    )
    return exp


def translate_exp_to_sectors(
    exp: Exposures,
    affected_sectors: Iterable[str] | dict[str, float] | pd.Series | Literal["all"],
    mriot: pymrio.IOSystem,
    value_col: str = "value",
) -> pd.Series:
    """
    Translate exposure data to the Multi-Regional Input-Output Table (MRIOT) context.

    By default the function distribute the exposure value to the sectors according to the production ratios
    of the affected sectors (Their relative share of the gross output (in the MRIOT) of all affected sectors).
    The ratios can also be provided directly using a dict or a Series, and are applied as is.
    The value is also scaled to match the monetary factor of the MRIOT (i.e., divided by).

    Parameters
    ----------
    exp : Exposures
        An instance of the `Exposures` class containing geospatial exposure data for a group of sectors.
        The Exposures GeoDataFrame has to contain a `region` column with regions present in the MRIOTs. Such an Exposures object can be built with :py:func:`translate_exp_to_regions`
    affected_sectors : Iterable[str] | dict[str, float] | pd.Series | Literal["all"]
        Sectors affected by the event. It can either be:
        - A list or pd.Index of sector names (as str) if sectors distribution at each point is to be proportional
        to their production share in the MRIOT.
        - "all", in which case all sectors of the MRIOT are considered affected.
        - A dictionary/Series where keys/index are sector names (str) and values are predefined shares of the relative
        presence of each sectors within each coordinates (float).
    mriot : pym.IOSystem
        An instance of `pymrio.IOSystem`, representing the multi-regional input-output model.

    Returns
    -------
    pd.Series
        A pandas Series with a MultiIndex of regions and sectors, containing the 'translated' exposure values.

    """
    if "region" not in exp.gdf.columns:
        exp = translate_exp_to_regions(exp, mriot_name=mriot.name)  # type: ignore
    if isinstance(affected_sectors, str) and affected_sectors == "all":
        affected_sectors = list(mriot.get_sectors())  # type: ignore

    if isinstance(affected_sectors, (list, pd.Index)):
        # logger.info("Calculating production ratios using MRIOT data for the provided sectors.")
        check_sectors_in_mriot(affected_sectors, mriot)
        prod_ratio = mriot.x.loc[  # type: ignore
            pd.IndexSlice[exp.gdf["region"].unique(), affected_sectors],
            mriot.x.columns[0],
        ]
        prod_ratio = prod_ratio.groupby("region").transform(lambda x: x / x.sum())
    elif isinstance(affected_sectors, (dict, pd.Series)):
        # logger.info("Using predefined production ratios from affected_sectors dictionary.")
        check_sectors_in_mriot(affected_sectors.keys(), mriot)
        prod_ratio = pd.Series(affected_sectors)
        repeated = pd.concat([prod_ratio] * exp.gdf["region"].nunique())
        multi_index = pd.MultiIndex.from_product(
            [exp.gdf["region"].unique(), prod_ratio.index], names=["region", "sector"]
        )
        prod_ratio = pd.Series(repeated.values, index=multi_index)
    else:
        raise TypeError(
            f"Wrong type for `affected_sectors` (expects Iterable[str] | dict[str, float] | pd.Series | Literal['all'] not {type(affected_sectors)})."
        )

    if (prod_ratio.groupby("region").sum().round(6) != 1.0).any():
        raise ValueError(
            f"The distribution share do not sum to 1. for all regions. Sum = {prod_ratio.groupby('region').sum()}"
        )
    exposed_assets = exp.gdf.groupby("region")[value_col].sum() * prod_ratio
    exposed_assets /= mriot.monetary_factor
    return exposed_assets


def translate_imp_to_regions(
    reg_impact: pd.DataFrame,
    mriot: pymrio.IOSystem,
    custom_mriot: bool = False,
) -> pd.DataFrame:
    """
    Translate regional impact data to MRIOT regions.

    Parameters
    ----------
    reg_impact : pd.DataFrame
        DataFrame with regional impact data. Index should represent event IDs,
        and columns should represent ISO3 regions.
    mriot_type : str
        The target MRIOT region type for conversion (e.g., 'ISO3', 'ISO2').
    custom_mriot : bool, default False
        If True, consider the given MRIOT as custom (MRIOT regions require to be in ISO3 format).

    Returns
    -------
    pd.DataFrame
        A DataFrame with event IDs as index and MRIOT regions as columns, with summed impact
        values if multiple regions map to the same MRIOT region.

    Raises
    ------
    ValueError
        If `reg_impact` does not have the expected format or `mriot_type` is invalid.
    """
    if not isinstance(reg_impact, pd.DataFrame):
        raise ValueError("`reg_impact` must be a pandas DataFrame.")

    cc = coco.CountryConverter()
    valid_iso3_regions = set(cc.ISO3["ISO3"])
    invalid_regions = set(reg_impact.columns) - valid_iso3_regions
    if invalid_regions:
        raise ValueError(
            f"`reg_impact` contains regions that are not valid ISO3 regions: {', '.join(invalid_regions)}"
        )

    reg_impact = reg_impact.copy()
    reg_impact /= mriot.monetary_factor

    reg_impact = reg_impact.rename_axis(index="event_id", columns="region")
    reg_impact = reg_impact.melt(
        ignore_index=False, var_name="region", value_name="value"
    )
    reg_impact["region_mriot"] = cc.pandas_convert(
        reg_impact["region"], to=_get_coco_MRIOT_name(mriot.name, custom_mriot)
    ).str.upper()
    ret = reg_impact.set_index("region_mriot", append=True)["value"]

    # Multiple ISO3 regions can end up in same MRIOT region (ROW for instance)
    # so we need to groupby-sum these before unstacking
    ret = ret.groupby(level=[0, 1]).sum().unstack()
    return ret


def distribute_reg_impact_to_sectors(
    reg_impact: pd.DataFrame,
    distributor: pd.Series,
) -> pd.DataFrame:
    """
    Distribute regional impact data across sectors based on a distributor.

    Parameters
    ----------
    reg_impact : pd.DataFrame
        DataFrame containing regional impact data.
        Columns represent regions, and index represents events.
    distributor : pd.Series
        Series used to distribute the impact across sectors.
        Can have a MultiIndex (region, sector) or a single index (sector).

    Returns
    -------
    pd.DataFrame
        A DataFrame with a MultiIndex for columns (region, sector) and distributed impact values.

    Raises
    ------
    ValueError
        If any region or sector in `distributor` are not represented in the reg_impact DataFrame.
    """

    # Input validation
    if not isinstance(reg_impact, pd.DataFrame):
        raise ValueError("`reg_impact` must be a pandas DataFrame.")

    if not isinstance(distributor, pd.Series):
        raise ValueError("`distributor` must be a pandas Series.")

    # Create a MultiIndex for the resulting columns (region, sector)
    if isinstance(distributor.index, pd.MultiIndex):
        multi_index = pd.MultiIndex.from_product(
            [reg_impact.columns, distributor.index.get_level_values("sector").unique()],
            names=["region", "sector"],
        )
    else:
        multi_index = pd.MultiIndex.from_product(
            [reg_impact.columns, distributor.index], names=["region", "sector"]
        )

    if not isinstance(distributor.index, pd.MultiIndex):
        distributor = distributor.reindex(multi_index.get_level_values("sector"))
        distributor.index = multi_index

    missing_regions = set(multi_index.get_level_values("region")) - set(
        distributor.index.get_level_values("region")
    )
    if missing_regions:
        raise ValueError(
            f"The following regions are missing in the distributor: {', '.join(missing_regions)}"
        )

    missing_sectors = set(multi_index.get_level_values("sector")) - set(
        distributor.index.get_level_values("sector")
    )
    if missing_sectors:
        raise ValueError(
            f"The following regions are missing in the distributor: {', '.join(missing_sectors)}"
        )

    # Normalize values by regions
    distributor = distributor.groupby(level="region").transform(lambda x: x / x.sum())
    # Expand regional_impact to have matching multi-level columns
    sector_count = len(multi_index.get_level_values("sector").unique())
    reg_impact_exp = reg_impact.loc[
        :,
        reg_impact.columns.repeat(sector_count),
    ]
    reg_impact_exp.columns = multi_index

    # Multiply element-wise
    return reg_impact_exp.mul(distributor.values, level="sector")


@overload
def _thin_to_wide(
    thin: pd.Series, long_index: pd.Index, long_columns: None = None
) -> pd.Series: ...


@overload
def _thin_to_wide(
    thin: pd.DataFrame,
    long_index: pd.Index,
    long_columns: pd.Index,
) -> pd.DataFrame: ...


def _thin_to_wide(
    thin: pd.Series | pd.DataFrame,
    long_index: pd.Index,
    long_columns: pd.Index | None = None,
) -> pd.Series | pd.DataFrame:
    if isinstance(thin, pd.Series):
        wide = pd.Series(index=long_index, dtype=thin.dtype)
    elif isinstance(thin, pd.DataFrame):
        if long_columns is None:
            raise ValueError(
                "long_columns argument cannot be None when widening a DataFrame."
            )
        wide = pd.DataFrame(
            index=long_index, columns=long_columns, dtype=thin.dtypes.iloc[0]
        )
    wide.fillna(0, inplace=True)
    if isinstance(thin, pd.DataFrame):
        wide.loc[thin.index, thin.columns] = thin.values
    else:
        wide.loc[thin.index] = thin.values
    return wide
