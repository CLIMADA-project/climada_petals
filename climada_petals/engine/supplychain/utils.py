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
from pathlib import Path
from typing import Iterable, Literal, overload
import warnings
import zipfile
import pandas as pd
import numpy as np

import pymrio
import country_converter as coco

from climada.util import files_handler as u_fh
from climada import CONFIG
from climada.entity import Exposures

WIOD_FILE_LINK = CONFIG.engine.supplychain.resources.wiod16.str()
"""Link to the 2016 release of the WIOD tables."""

VA_NAME = "value added"
"""Index name for value added"""


def calc_va(Z, x):
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


def calc_B(Z, x):
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


def calc_G(B):
    """Calculate the Ghosh inverse matrix G either from B
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
    I = np.eye(B.shape[0])
    if isinstance(B, pd.DataFrame):
        return pd.DataFrame(np.linalg.inv(I - B), index=B.index, columns=B.columns)
    else:
        return np.linalg.inv(I - B)  # G = inverse matrix of (I - B)


def calc_x_from_G(G, va):
    """Calculate the industry output x from a v vector and G matrix

    x = Gva

    The industry output x is computed from a value-added vector v

    Parameters
    ----------
    v : pandas.DataFrame or numpy.array
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


def parse_mriot_from_df(
    mriot_df, col_iso3, col_sectors, rows_data, cols_data, row_fd_cats=None
):
    """Build multi-index dataframes of the transaction matrix, final demand and total
       production from a Multi-Regional Input-Output Table dataframe.

    Parameters
    ----------
    mriot_df : pandas.DataFrame
        The Multi-Regional Input-Output Table
    col_iso3 : int
        Column's position of regions' iso names
    col_sectors : int
        Column's position of sectors' names
    rows_data : (int, int)
        Tuple of integers with positions of rows
        containing the MRIOT data for intermediate demand
        matrix.
        Final demand matrix is assumed to be the remaining columns
        of the DataFrame except the last one (which generally holds
        total output).
    cols_data : (int, int)
        Tuple of integers with positions of columns
        containing the MRIOT data
    row_fd_cats : int
        Integer index of the row containing the
        final demand categories.
    """

    start_row, end_row = rows_data
    start_col, end_col = cols_data

    sectors = mriot_df.iloc[start_row:end_row, col_sectors].unique()
    regions = mriot_df.iloc[start_row:end_row, col_iso3].unique()
    if row_fd_cats is None:
        n_fd_cat = (mriot_df.shape[1] - (end_col + 1)) // len(regions)
        fd_cats = [f"fd_cat_{i}" for i in range(n_fd_cat)]
    else:
        fd_cats = mriot_df.iloc[row_fd_cats, end_col:-1].unique()

    multiindex = pd.MultiIndex.from_product(
        [regions, sectors], names=["region", "sector"]
    )

    multiindex_final_demand = pd.MultiIndex.from_product(
        [regions, fd_cats], names=["region", "category"]
    )

    Z = mriot_df.iloc[start_row:end_row, start_col:end_col].values.astype(float)
    Z = pd.DataFrame(data=Z, index=multiindex, columns=multiindex)

    Y = mriot_df.iloc[start_row:end_row, end_col:-1].values.astype(float)
    Y = pd.DataFrame(data=Y, index=multiindex, columns=multiindex_final_demand)

    x = mriot_df.iloc[start_row:end_row, -1].values.astype(float)
    x = pd.DataFrame(data=x, index=multiindex, columns=["indout"])

    return Z, Y, x


def mriot_file_name(mriot_type, mriot_year):
    """Retrieve the original EXIOBASE3, WIOD16 or OECD21 MRIOT file name

    Parameters
    ----------
    mriot_type : str
    mriot_year : int

    Returns
    -------
    str
        name of MRIOT file
    """

    if mriot_type == "EXIOBASE3":
        return f"IOT_{mriot_year}_ixi.zip"

    if mriot_type == "WIOD16":
        return f"WIOT{mriot_year}_Nov16_ROW.xlsb"

    if mriot_type == "OECD21":
        return f"ICIO2021_{mriot_year}.csv"

    raise ValueError("Unknown MRIOT type")


def download_mriot(mriot_type, mriot_year, download_dir):
    """Download EXIOBASE3, WIOD16 or OECD21 Multi-Regional Input Output Tables
    for specific years.

    Parameters
    ----------
    mriot_type : str
    mriot_year : int
    download_dir : pathlib.PosixPath

    Notes
    -----
    The download of EXIOBASE3 and OECD21 tables makes use of pymrio functions.
    The download of WIOD16 tables requires ad-hoc functions, since the
    related pymrio functions were broken at the time of implementation
    of this function.
    """

    if mriot_type == "EXIOBASE3":
        pymrio.download_exiobase3(
            storage_folder=download_dir, system="ixi", years=[mriot_year]
        )

    elif mriot_type == "WIOD16":
        download_dir.mkdir(parents=True, exist_ok=True)
        downloaded_file_name = u_fh.download_file(
            WIOD_FILE_LINK,
            download_dir=download_dir,
        )
        downloaded_file_zip_path = Path(downloaded_file_name + ".zip")
        Path(downloaded_file_name).rename(downloaded_file_zip_path)

        with zipfile.ZipFile(downloaded_file_zip_path, "r") as zip_ref:
            zip_ref.extractall(download_dir)

    elif mriot_type == "OECD21":
        years_groups = ["1995-1999", "2000-2004", "2005-2009", "2010-2014", "2015-2018"]
        year_group = years_groups[int(np.floor((mriot_year - 1995) / 5))]

        pymrio.download_oecd(storage_folder=download_dir, years=year_group)


def parse_mriot(mriot_type, downloaded_file):
    """Parse EXIOBASE3, WIOD16 or OECD21 MRIOT for specific years

    Parameters
    ----------
    mriot_type : str
    downloaded_file : pathlib.PosixPath

    Notes
    -----
    The parsing of EXIOBASE3 and OECD21 tables makes use of pymrio functions.
    The parsing of WIOD16 tables requires ad-hoc functions, since the
    related pymrio functions were broken at the time of implementation
    of this function.
    """

    if mriot_type == "EXIOBASE3":
        mriot = pymrio.parse_exiobase3(path=downloaded_file)
        # no need to store A
        mriot.A = None

    elif mriot_type == "WIOD16":
        mriot_df = pd.read_excel(downloaded_file, engine="pyxlsb")

        Z, Y, x = parse_mriot_from_df(
            mriot_df,
            col_iso3=2,
            col_sectors=1,
            row_fd_cats=2,
            rows_data=(5, 2469),
            cols_data=(4, 2468),
        )

        mriot = pymrio.IOSystem(Z=Z, Y=Y, x=x)
        multiindex_unit = pd.MultiIndex.from_product(
            [mriot.get_regions(), mriot.get_sectors()], names=["region", "sector"]
        )
        mriot.unit = pd.DataFrame(
            data=np.repeat(["M.USD"], len(multiindex_unit)),
            index=multiindex_unit,
            columns=["unit"],
        )

        # This should include the year as well
        mriot.name = "WIOD16"

    elif mriot_type == "OECD21":
        mriot = pymrio.parse_oecd(path=downloaded_file)
        mriot.x = pymrio.calc_x(mriot.Z, mriot.Y)

    else:
        raise RuntimeError(f"Unknown mriot_type: {mriot_type}")
    return mriot

def check_sectors_in_mriot(sectors: Iterable[str], mriot: pymrio.IOSystem) -> None:
    """
    Check whether the given list of sectors exists within the MRIOT data.

    Parameters
    ----------
    sectors : list of str
        List of sector names to check.
    mriot : pym.IOSystem
        An instance of `pymrio.IOSystem`, representing the multi-regional input-output model.

    Raises
    ------
    ValueError
        If any of the sectors in the list are not found in the MRIOT data.
    """
    # Retrieve all available sectors from the MRIOT data
    available_sectors = set(mriot.get_sectors())

    # Identify missing sectors
    missing_sectors = set(sectors) - available_sectors

    # Raise an error if any sectors are missing
    if missing_sectors:
        raise ValueError(
            f"The following sectors are missing in the MRIOT data: {missing_sectors}"
        )


def translate_exp_to_regions(
    exp: Exposures,
    mriot_type: str,
) -> Exposures:
    """
    Creates a region column within the GeoDataFrame of the Exposures object which matches region
    of an MRIOT. Also compute the share of total mriot region value per centroid.

    Parameters
    ----------
    exp : Exposure
        The Exposure object to modify.
    mriot_type : str
        Type of the MRIOT to convert region_id to. Currently available are:
        ["WIOD"]
    copy : bool, optional
        If True returns a modified copy of the Exposures object instead of modifying in place.

    Returns
    -------
    None or Exposures
        Modify inplace and returns None by default, or a modified copy if `return_copy=True`.
    """
    exp = copy.deepcopy(exp)
    cc = coco.CountryConverter()
    # Find region names
    exp.gdf["region"] = cc.pandas_convert(
        series=exp.gdf["region_id"], to=mriot_type
    ).str.upper()

    # compute distribution per region
    exp.gdf["value_ratio"] = exp.gdf.groupby("region")["value"].transform(
        lambda x: x / x.sum()
    )
    return exp

def translate_exp_to_sectors(
    exp: Exposures,
    affected_sectors: Iterable[str] | dict[str, float]|Literal["all"],
    mriot: pymrio.IOSystem,
    value_col: str = "value",
) -> pd.Series:
    """
    Translate exposure data to the Multi-Regional Input-Output Table (MRIOT) context.

    Parameters
    ----------
    exp : Exposures
        An instance of the `Exposures` class containing geospatial exposure data for a group of sectors.
    affected_sectors : list of str or dict of {str: float} or None
        Sectors affected by the event. Can be either:
        - A list of sector names (str) if sectors distribution at each point is to be proportional
        to their production share in the MRIOT.
        - A dictionary where keys are sector names (str) and values are predefined shares of the relative
        presence of each sectors within each coordinates (float).
        - "all", in which case all sectors of the MRIOT are considered affected.
    mriot : pym.IOSystem
        An instance of `pymrio.IOSystem`, representing the multi-regional input-output model.

    Returns
    -------
    pd.Series
        A pandas Series with a MultiIndex of regions and sectors, containing the translated exposure values.

    Notes
    -----
    The function adjusts the exposure values according to the production ratios of the affected sectors.
    If the production ratios are provided directly, they are applied as is. If a list of affected sectors
    is provided, the production ratios are calculated using the MRIOT data.
    """

    ## If production ratios are provided as a dictionary, use them directly
    if affected_sectors == "all":
        affected_sectors = mriot.get_sectors()

    if isinstance(affected_sectors, dict):
        # logger.info("Using predefined production ratios from affected_sectors dictionary.")

        # Ensure that all sectors in the dictionary exist
        check_sectors_in_mriot(affected_sectors.keys(), mriot)
        prod_ratio = pd.Series(affected_sectors, name="sector")

        # Sum the exposure values by region
        exposed_assets = exp.gdf.groupby("region")[value_col].sum()

        # Create a MultiIndex to match regions with each sector
        multi_index = pd.MultiIndex.from_product(
            [exposed_assets.index, prod_ratio.index], names=["region", "sector"]
        )

        # Repeat the exposure values for each sector within each region
        exposed_assets = exposed_assets.repeat(len(prod_ratio))
        exposed_assets.index = multi_index

        # Multiply the exposure values by the corresponding production ratios
        exposed_assets = exposed_assets.mul(prod_ratio, level="sector")

    elif isinstance(affected_sectors, list):
        # logger.info("Calculating production ratios using MRIOT data for the provided sectors.")

        # Check if all sectors are present in the MRIOT data
        check_sectors_in_mriot(affected_sectors, mriot)

        # Extract and normalize production ratios for the affected sectors
        prod_ratio = mriot.x.loc[
            pd.IndexSlice[exp.gdf["region"].unique(), affected_sectors], "indout"
        ]

        # Normalize the production ratios by region
        prod_ratio = prod_ratio.groupby("region").transform(lambda x: x / x.sum())

        # Apply the production ratios to the summed exposure values
        exposed_assets = exp.gdf.groupby("region")[value_col].sum() * prod_ratio

    else:
        raise ValueError(
            "`affected_sectors` must be either a list of sector names or a dictionary with sector ratios."
        )

    return exposed_assets

def get_mriot_type(mriot:pymrio.IOSystem)->str:
    if "WIOD" in mriot.name:
        return "WIOD"
    else:
        raise NotImplementedError("This MRIOT is not yet implemented (or its name is not set properly).")


def translate_reg_impact_to_mriot_regions(
    reg_impact: pd.DataFrame,
    mriot_type: str,
) -> pd.DataFrame:
    """
    Translate regional impact data to MRIOT regions.

    Parameters
    ----------
    reg_impact : pd.DataFrame
        DataFrame with regional impact data. Index should represent event IDs, and columns should represent ISO3 regions.
    mriot_type : str
        The target MRIOT region type for conversion (e.g., 'ISO3', 'ISO2').
    inplace : bool, optional
        If True, modifies the original `reg_impact` DataFrame. If False, returns a modified copy.

    Returns
    -------
    pd.DataFrame
        A DataFrame with event IDs as index and MRIOT regions as columns, with summed impact values if multiple regions map to the same MRIOT region.

    Raises
    ------
    ValueError
        If `reg_impact` does not have the expected format or `mriot_type` is invalid.
    """
    if not isinstance(reg_impact, pd.DataFrame):
        raise ValueError("`reg_impact` must be a pandas DataFrame.")

    if not isinstance(mriot_type, str):
        raise ValueError("`mriot_type` must be a string.")

    cc = coco.CountryConverter()
    valid_iso3_regions = set(cc.ISO3["ISO3"])
    invalid_regions = set(reg_impact.columns) - valid_iso3_regions
    if invalid_regions:
        raise ValueError(f"`reg_impact` contains regions that are not valid ISO3 regions: {', '.join(invalid_regions)}")

    reg_impact = reg_impact.copy()

    reg_impact = reg_impact.rename_axis(index="event_id", columns="region")
    reg_impact = reg_impact.melt(ignore_index=False, var_name="region", value_name="value")
    reg_impact["region_mriot"] = cc.pandas_convert(
        reg_impact["region"], to=mriot_type
    ).str.upper()
    reg_impact = reg_impact.set_index("region_mriot", append=True)[["value"]]

    # Multiple ISO3 regions can end up in same MRIOT region (ROW for instance)
    # so we need to groupby-sum these before unstacking
    reg_impact = reg_impact["value"].groupby(level=[0, 1]).sum().unstack()
    return reg_impact


def distribute_reg_impact_to_sectors(
    reg_impact: pd.DataFrame,
    distributor: pd.Series,
) -> pd.DataFrame:
    """
    Distribute regional impact data across sectors based on a distributor.

    Parameters
    ----------
    reg_impact : pd.DataFrame
        DataFrame containing regional impact data. Columns represent regions, and index represents events.
    distributor : pd.Series
        Series used to distribute the impact across sectors. Can have a MultiIndex (region, sector) or a single index (sector).
    inplace : bool, optional
        If True, modifies the original `reg_impact` DataFrame. If False, returns a modified copy.

    Returns
    -------
    pd.DataFrame
        A DataFrame with a MultiIndex for columns (region, sector) and distributed impact values.

    Raises
    ------
    ValueError
        If the distributor's index is neither a MultiIndex nor a single index of sectors.
    ValueError
        If any sectors in `distributor` are not represented in the reg_impact DataFrame.
    """

    # Input validation
    if not isinstance(reg_impact, pd.DataFrame):
        raise ValueError("`reg_impact` must be a pandas DataFrame.")

    if not isinstance(distributor, pd.Series):
        raise ValueError("`distributor` must be a pandas Series.")

    if not isinstance(distributor.index, pd.MultiIndex) and not isinstance(distributor.index, pd.Index):
        raise ValueError("`distributor` index must be a pandas MultiIndex or Index.")

    # Create a MultiIndex for the resulting columns (region, sector)
    if isinstance(distributor.index, pd.MultiIndex):
        multi_index = distributor.index
    else:
        multi_index = pd.MultiIndex.from_product(
            [reg_impact.columns, distributor.index], names=["region", "sector"]
        )

    if not isinstance(distributor.index, pd.MultiIndex):
        distributor = distributor.reindex(multi_index.get_level_values("sector"))

    missing_sectors = set(distributor.index.get_level_values('sector')) - set(multi_index.get_level_values('sector'))
    if missing_sectors:
        raise ValueError(f"The following sectors are missing in the distributor: {', '.join(missing_sectors)}")


    # Expand regional_impact to have matching multi-level columns
    sector_count = len(multi_index.get_level_values("sector").unique())
    reg_impact_exp = reg_impact.loc[
        :,
        reg_impact.columns.repeat(sector_count),
    ]
    reg_impact_exp.columns = multi_index

    # Multiply element-wise
    return reg_impact_exp.mul(distributor.values, level="sector")
