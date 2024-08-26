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

from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

import pymrio

from climada.util import files_handler as u_fh
from climada import CONFIG

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

    elif mriot_type == "OECD21":
        mriot = pymrio.parse_oecd(path=downloaded_file)
        mriot.x = pymrio.calc_x(mriot.Z, mriot.Y)

    else:
        raise RuntimeError(f"Unknown mriot_type: {mriot_type}")
    return mriot
