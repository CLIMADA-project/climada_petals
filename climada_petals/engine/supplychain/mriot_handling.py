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

MRIOT handler (download, parsing, IO).
"""

from collections.abc import Iterable
import copy
from typing import cast, Literal
import warnings
import numpy as np
import pathlib
import pymrio
import pandas as pd
import zipfile
import re

from climada import CONFIG
from climada.util import files_handler as u_fh
from climada_petals.engine.supplychain import LOGGER

MRIOT_DIRECTORY = CONFIG.engine.supplychain.local_data.mriot.dir()
"""Directory where Multi-Regional Input-Output Tables (MRIOT) are downloaded."""

MRIOT_TYPE_REGEX = r"(?P<mrio_type>OECD23|EXIOBASE3|EORA26|WIOD16)"
MRIOT_YEAR_REGEX = r"(?P<mrio_year>\d{4})"
MRIOT_FULLNAME_REGEX = re.compile("{}-{}".format(MRIOT_TYPE_REGEX, MRIOT_YEAR_REGEX))

MRIOT_DEFAULT_FILENAME = {
    "EXIOBASE3": lambda year: f"IOT_{year}_ixi.zip",
    "WIOD16": lambda year: f"WIOT{year}_Nov16_ROW.xlsb",
    "OECD23": lambda year: f"ICIO2023_{year}.csv",
}

MRIOT_MONETARY_FACTOR = {
    "EXIOBASE3": 1000000,
    "EORA26": 1000,
    "WIOD16": 1000000,
    "OECD23": 1000000,
    "EUREGIO": 1000000,
}

MRIOT_BASENAME = {
    "EXIOBASE3": "exiobase3_ixi",
    "EORA26": "eora26",
    "WIOD16": "wiod_v2016",
    "OECD23": "icio_v2023",
}


MRIOT_COUNTRY_CONVERTER_CORR = {
    "EXIOBASE3": "EXIO3",
    "WIOD16": "WIOD",
    "EORA26": "Eora",
    "OECD23": "ISO3",
}

WIOD_FILE_LINK = CONFIG.engine.supplychain.resources.wiod16.str()
"""Link to the 2016 release of the WIOD tables."""

VA_NAME = "value added"
"""Index name for value added"""

_ATTR_LIST = [
    "Z",
    "Y",
    "x",
    "A",
    "As",
    "G",
    "L",
    "unit",
    "population",
    "meta",
    "__non_agg_attributes__",
    "__coefficients__",
    "__basic__",
]

ICIO23_sectors_mapping = {
    "A01_02": "Agriculture, hunting, forestry",
    "A03": "Fishing and aquaculture",
    "B05_06": "Mining and quarrying, energy producing products",
    "B07_08": "Mining and quarrying, non-energy producing products",
    "B09": "Mining support service activities",
    "C10T12": "Food products, beverages and tobacco",
    "C13T15": "Textiles, textile products, leather and footwear",
    "C16": "Wood and products of wood and cork",
    "C17_18": "Paper products and printing",
    "C19": "Coke and refined petroleum products",
    "C20": "Chemical and chemical products",
    "C21": "Pharmaceuticals, medicinal chemical and botanical products",
    "C22": "Rubber and plastics products",
    "C23": "Other non-metallic mineral products",
    "C24": "Basic metals",
    "C25": "Fabricated metal products",
    "C26": "Computer, electronic and optical equipment",
    "C27": "Electrical equipment",
    "C28": "Machinery and equipment, nec ",
    "C29": "Motor vehicles, trailers and semi-trailers",
    "C30": "Other transport equipment",
    "C31T33": "Manufacturing nec; repair and installation of machinery and equipment",
    "D": "Electricity, gas, steam and air conditioning supply",
    "E": "Water supply; sewerage, waste management and remediation activities",
    "F": "Construction",
    "G": "Wholesale and retail trade; repair of motor vehicles",
    "H49": "Land transport and transport via pipelines",
    "H50": "Water transport",
    "H51": "Air transport",
    "H52": "Warehousing and support activities for transportation",
    "H53": "Postal and courier activities",
    "I": "Accommodation and food service activities",
    "J58T60": "Publishing, audiovisual and broadcasting activities",
    "J61": "Telecommunications",
    "J62_63": "IT and other information services",
    "K": "Financial and insurance activities",
    "L": "Real estate activities",
    "M": "Professional, scientific and technical activities",
    "N": "Administrative and support services",
    "O": "Public administration and defence; compulsory social security",
    "P": "Education",
    "Q": "Human health and social work activities",
    "R": "Arts, entertainment and recreation",
    "S": "Other service activities",
    "T": "Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use",
}


def lexico_reindex(mriot: pymrio.IOSystem) -> pymrio.IOSystem:
    """Re-index IOSystem lexicographically.

    Sort indexes and columns of the dataframe of a pymrio.IOSystem by lexical order.

    Parameters
    ----------
    mriot : pymrio.IOSystem
        The IOSystem to sort.

    Returns
    -------
    pymrio.IOSystem
        A sorted copy of the IOSystem.
    """
    mriot = copy.deepcopy(mriot)
    if getattr("mriot", "_sorted", None):
        return mriot
    else:
        for matrix_name in ["Z", "Y", "x", "A", "As", "G", "L"]:
            matrix = getattr(mriot, matrix_name, None)
            if matrix is not None:
                setattr(
                    mriot,
                    matrix_name,
                    matrix.reindex(sorted(matrix.index)).sort_index(axis=1),
                )
        mriot._sorted = True
        return mriot


def build_exio3_from_zip(
    mrio_zip: str | pathlib.Path,
    remove_attributes: bool = True,
    aggregate_ROW: bool = True,
) -> pymrio.IOSystem:
    """Creates an EXIOBASE3 IOSystem from a zip file.

    This function mainly relies on `pymrio.parse_exiobase3()`. Optionaly and by
    default, it also removes attributes that can be memory expensive and not
    used within Climada context, and aggregates the EXIOBASE3 world regions to
    one unique ROW region.
    It also:
    - adds several new attributes to the object such as `monetary_factor` and `year` for internal mechanics
    - sorts the indexes of the MRIOT.
    - computes missing parts or the MRIOT

    Parameters
    ----------
    mrio_zip : str | pathlib.Path
        The path to the EXIOBASE3 zip file.
    remove_attributes : bool, default True.
        Whether to remove unnecessary attribute which can be memory expensive.
    aggregate_ROW : bool, default True.
        Whether to aggregate `["WA", "WE", "WF", "WL", "WM"]` to `"ROW"`.

    Returns
    -------
    pymrio.IOSystem
        An IOSystem corresponding to the EXIOBASE3 MRIOT.

    """

    mrio_path = pathlib.Path(mrio_zip)
    mrio_pym = pymrio.parse_exiobase3(path=mrio_path)
    mrio_pym = cast(pymrio.IOSystem, mrio_pym)  # Just for the LSP
    if remove_attributes:
        LOGGER.info("Removing unnecessary IOSystem attributes")
        attr = _ATTR_LIST
        tmp = list(mrio_pym.__dict__.keys())
        for at in tmp:
            if at not in attr:
                delattr(mrio_pym, at)
        LOGGER.info("Done")

    if aggregate_ROW:
        LOGGER.info("Aggregating the different ROWs regions together")
        agg_regions = pd.DataFrame(
            {
                "original": mrio_pym.get_regions()[
                    ~mrio_pym.get_regions().isin(["WA", "WE", "WF", "WL", "WM"])
                ].tolist()
                + ["WA", "WE", "WF", "WL", "WM"],
                "aggregated": mrio_pym.get_regions()[
                    ~mrio_pym.get_regions().isin(["WA", "WE", "WF", "WL", "WM"])
                ].tolist()
                + ["ROW"] * 5,
            }
        )
        mrio_pym = mrio_pym.aggregate(region_agg=agg_regions)

    LOGGER.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    LOGGER.info("Done")

    LOGGER.info("Computing the missing IO components")
    mrio_pym.calc_all()
    LOGGER.info("Done")

    setattr(mrio_pym, "monetary_factor", MRIOT_MONETARY_FACTOR["EXIOBASE3"])
    setattr(mrio_pym, "basename", MRIOT_BASENAME["EXIOBASE3"])
    setattr(mrio_pym, "year", mrio_pym.meta.description[-4:])
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    # Also put it in meta for saving
    mrio_pym.meta.change_meta("monetary_factor", MRIOT_MONETARY_FACTOR["EXIOBASE3"])
    mrio_pym.meta.change_meta("year", mrio_pym.meta.description[-4:])
    mrio_pym.meta.change_meta("basename", MRIOT_BASENAME["EXIOBASE3"])
    mrio_pym.meta.change_meta("sectors_agg", "full_sectors")
    mrio_pym.meta.change_meta("regions_agg", "full_regions")
    return mrio_pym


def build_eora_from_zip(
    mrio_zip: str,
    reexport_treatment: bool = False,
    inv_treatment: bool = True,
    remove_attributes: bool = True,
) -> pymrio.IOSystem:
    """Creates an EORA26 IOSystem from a zip file.

    This function mainly relies on `pymrio.parse_eora26()`. Optionaly and by
    default, it also removes attributes that can be memory expensive and not
    used within Climada context. It also allows to apply some "fixes" to the EORA26 table,
    such as merging the "reexport" sector or removing possible negative final demands.
    Further, it also:
    - adds several new attributes to the object such as `monetary_factor` and `year` for internal mechanics
    - sorts the indexes of the MRIOT.
    - computes missing parts or the MRIOT

    Parameters
    ----------

    mrio_zip : str
        The path to the eora26 zip file.
    reexport_treatment : bool, default False
        Whether to merge reexport sector with the "other" sector.
    inv_treatment : bool, default True
        Whether to clip negative final demand to 0 (due to Change in inventories).
    remove_attributes : bool, default True
        Whether to remove unnecessary attribute which can be memory expensive.
    Returns
    -------
    pymrio.IOSystem
        An IOSystem corresponding to the EORA26 MRIOT.

    """

    mrio_path = pathlib.Path(mrio_zip)
    mrio_pym = pymrio.parse_eora26(path=mrio_path)
    LOGGER.info("Removing unnecessary IOSystem attributes")
    if remove_attributes:
        attr = _ATTR_LIST
        tmp = list(mrio_pym.__dict__.keys())
        for at in tmp:
            if at not in attr:
                delattr(mrio_pym, at)
    LOGGER.info("Done")

    setattr(mrio_pym, "monetary_factor", MRIOT_MONETARY_FACTOR["EORA26"])
    setattr(mrio_pym, "basename", MRIOT_BASENAME["EORA26"])
    setattr(mrio_pym, "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"])
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    # Also put it in meta for saving
    mrio_pym.meta.change_meta("monetary_factor", MRIOT_MONETARY_FACTOR["EORA26"])
    mrio_pym.meta.change_meta(
        "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"]
    )
    mrio_pym.meta.change_meta("basename", MRIOT_BASENAME["EORA26"])
    mrio_pym.meta.change_meta("sectors_agg", "full_sectors")
    mrio_pym.meta.change_meta("regions_agg", "full_regions")

    if reexport_treatment:
        LOGGER.info(
            "EORA26 has the re-import/re-export sector which other mrio often don't have (ie EXIOBASE), we put it in 'Other'."
        )
        mrio_pym.rename_sectors({"Re-export & Re-import": "Others"})
        mrio_pym.aggregate_duplicates()
        setattr(mrio_pym, "sectors_agg", "full_no_reexport_sectors")

    if inv_treatment:
        LOGGER.info(
            "EORA26 has negative values in its final demand which can cause problems. We set them to 0."
        )
        if mrio_pym.Y is not None:
            mrio_pym.Y = mrio_pym.Y.clip(lower=0)
        else:
            raise AttributeError("Y attribute is not set")

    LOGGER.info("Computing the missing IO components")
    mrio_pym.calc_all()
    LOGGER.info("Done")

    LOGGER.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    LOGGER.info("Done")

    return mrio_pym


def build_oecd_from_csv(
    mrio_csv: str, year: int | None = None, remove_attributes: bool = True
) -> pymrio.IOSystem:
    """This parsing function is put on hold while https://github.com/IndEcol/pymrio/issues/157 is not addressed"""
    mrio_path = pathlib.Path(mrio_csv)
    mrio_pym = pymrio.parse_oecd(path=mrio_path, year=year)
    LOGGER.info("Removing unnecessary IOSystem attributes")
    if remove_attributes:
        attr = _ATTR_LIST
        tmp = list(mrio_pym.__dict__.keys())
        for at in tmp:
            if at not in attr:
                delattr(mrio_pym, at)
    LOGGER.info("Done")
    setattr(mrio_pym, "monetary_factor", MRIOT_MONETARY_FACTOR["OECD23"])
    setattr(mrio_pym, "basename", MRIOT_BASENAME["OECD23"])
    setattr(mrio_pym, "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"])
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    # Also put it in meta for saving
    mrio_pym.meta.change_meta("monetary_factor", MRIOT_MONETARY_FACTOR["OECD23"])
    mrio_pym.meta.change_meta(
        "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"]
    )
    mrio_pym.meta.change_meta("basename", MRIOT_BASENAME["OECD23"])
    mrio_pym.meta.change_meta("sectors_agg", "full_sectors")
    mrio_pym.meta.change_meta("regions_agg", "full_regions")

    LOGGER.info("Computing the missing IO components")
    mrio_pym.calc_all()
    LOGGER.info("Done")
    LOGGER.info("Renaming sectors")
    mrio_pym.rename_sectors(ICIO23_sectors_mapping)
    LOGGER.info("Done")
    LOGGER.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    LOGGER.info("Done")
    return mrio_pym


def parse_wiod_v2016(mrio_xlsb: str):
    """Parse the WIOD 2016 Multi-Regional Input-Output Table from an Excel binary file.

    Parameters
    ----------
    mrio_xlsb : str
        The path to the Excel binary file containing the WIOD 2016 data.

    Returns
    -------
    pymrio.IOSystem
        An IOSystem object corresponding to the parsed WIOD 2016 data.
    """
    mrio_path = pathlib.Path(mrio_xlsb)
    mriot_df = pd.read_excel(mrio_xlsb, engine="pyxlsb")
    Z, Y, x = parse_mriot_from_df(
        mriot_df,
        col_iso3=2,
        col_sectors=1,
        row_fd_cats=2,
        rows_data=(5, 2469),
        cols_data=(4, 2468),
    )
    mrio_pym = pymrio.IOSystem(Z=Z, Y=Y, x=x)
    multiindex_unit = pd.MultiIndex.from_product(
        [mrio_pym.get_regions(), mrio_pym.get_sectors()], names=["region", "sector"]  # type: ignore
    )
    mrio_pym.unit = pd.DataFrame(
        data=np.repeat(["M.USD"], len(multiindex_unit)),
        index=multiindex_unit,
        columns=["unit"],
    )

    setattr(mrio_pym, "monetary_factor", MRIOT_MONETARY_FACTOR["WIOD16"])
    setattr(mrio_pym, "basename", MRIOT_BASENAME["WIOD16"])
    setattr(mrio_pym, "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"])
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    # Also put it in meta for saving
    mrio_pym.meta.change_meta("monetary_factor", MRIOT_MONETARY_FACTOR["WIOD16"])
    mrio_pym.meta.change_meta(
        "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"]
    )
    mrio_pym.meta.change_meta("basename", MRIOT_BASENAME["WIOD16"])
    mrio_pym.meta.change_meta("sectors_agg", "full_sectors")
    mrio_pym.meta.change_meta("regions_agg", "full_regions")

    LOGGER.info("Computing the missing IO components")
    mrio_pym.calc_all()
    LOGGER.info("Done")

    LOGGER.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    LOGGER.info("Done")
    return mrio_pym


def parse_mriot_from_df(
    mriot_df: pd.DataFrame,
    col_iso3: int,
    col_sectors: int,
    rows_data: tuple[int, int],
    cols_data: tuple[int, int],
    row_fd_cats: int | None = None,
):
    """Build multi-index dataframes of the transaction matrix, final demand and total
       production from a Multi-Regional Input-Output Table dataframe.

    Parameters
    ----------
    mriot_df : pandas.DataFrame
        The Multi-Regional Input-Output Table as a DataFrame
    col_iso3 : int
        Column's position of regions' ISO3 names
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

    Returns
    -------
    (Z, Y, x): tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]
        Tuple of 3 Dataframe, respectively containing the intermediate demand matrix Z,
        the final demand matrix Y and the gross output x.
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


def download_mriot(
    mriot_type: Literal["EXIOBASE3"] | Literal["WIOD16"] | Literal["OECD23"],
    mriot_year: int,
    download_dir: pathlib.Path,
):
    """Download EXIOBASE3, WIOD16 or OECD23 Multi-Regional Input Output Tables
    for specific years.

    Parameters
    ----------
    mriot_type : str
        The type of MRIOT to download.
    mriot_year : int
        The specific year to download. Must be available with the specified type.
    download_dir : pathlib.PosixPath
        Where to download the files.

    Notes
    -----
    The download of EXIOBASE3 and OECD23 tables makes use of pymrio functions.
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
        downloaded_file_zip_path = pathlib.Path(downloaded_file_name + ".zip")
        pathlib.Path(downloaded_file_name).rename(downloaded_file_zip_path)

        with zipfile.ZipFile(downloaded_file_zip_path, "r") as zip_ref:
            zip_ref.extractall(download_dir)

    elif mriot_type == "OECD23":

        # years_groups = ["1995-2000", "2001-2005", "2006-2010", "2011-2015", "2016-2020"]
        if mriot_year <= 2000:
            year_group = "1995-2000"
        elif mriot_year <= 2020:
            year_group = f"{2001 + (mriot_year - 2001) // 5 * 5}-{2001 + (mriot_year - 2001) // 5 * 5 + 4}"
        else:
            raise ValueError(f"{mriot_year} is not a valid OECD23 ICIO mriot year.")

        _fix_oecd_download_problem(download_dir)

        pymrio.download_oecd(storage_folder=download_dir, years=year_group)


def parse_mriot(
    mriot_type: str, downloaded_file: pathlib.Path, mriot_year: int, **kwargs
) -> pymrio.IOSystem:
    """Parse EXIOBASE3, WIOD16 or OECD23 MRIOT for specific years

    Parameters
    ----------
    mriot_type : str
        The type of MRIOT to parse.
    downloaded_file : pathlib.PosixPath
        The path to the files to parse.
    year: int
        The year of the MRIOT to parse.

    Returns
    -------
    pymrio.IOSystem
        An IOSystem representing the MRIOT.

    Notes
    -----
    The parsing of EXIOBASE3 and OECD23 tables makes use of pymrio functions.
    The parsing of WIOD16 tables requires ad-hoc functions, since the
    related pymrio functions were broken at the time of implementation
    of this function.

    Some metadata is rewrote or added to the objects for consistency
    in usage, internals (name, monetary factor, year).
    """

    if mriot_type == "EXIOBASE3":
        mriot = build_exio3_from_zip(mrio_zip=downloaded_file, **kwargs)
    elif mriot_type == "WIOD16":
        mriot = parse_wiod_v2016(mrio_xlsb=downloaded_file)
    elif mriot_type == "OECD23":
        mriot = build_oecd_from_csv(mrio_csv=downloaded_file, year=mriot_year)
    elif mriot_type == "EORA26":
        mriot = build_eora_from_zip(mrio_zip=downloaded_file, **kwargs)
    else:
        raise RuntimeError(f"Unknown mriot_type: {mriot_type}")

    mriot.meta.change_meta(
        "description", "Metadata for pymrio Multi Regional Input-Output Table"
    )
    mriot.meta.change_meta("name", f"{mriot_type}-{mriot_year}")

    # Check if negative demand - this happens when the
    # "Changes in Inventory (CII)" demand category is
    # larger than the sum of all other categories
    if (mriot.Y.sum(axis=1) < 0).any():
        warnings.warn(
            "Found negatives values in total final demand, "
            "setting them to 0 and recomputing production vector"
        )
        mriot.Y.loc[mriot.Y.sum(axis=1) < 0] = mriot.Y.loc[
            mriot.Y.sum(axis=1) < 0
        ].clip(lower=0)
        mriot.x = pymrio.calc_x(mriot.Z, mriot.Y)
        mriot.A = pymrio.calc_A(mriot.Z, mriot.x)

    return mriot


def get_mriot(
    mriot_type: str, mriot_year: int, redownload: bool = False, save: bool = True
):
    """Retrieves and optionally saves the Multi-Regional Input-Output Table (MRIOT)
    for a specified type and year. It handles downloading, parsing, and managing
    the directory structure for MRIOT files.

    Files are stored in the climada data folder, within a "MRIOT" folder by default.

    Parameters
    ----------
    mriot_type : str
        The type of MRIOT to retrieve (e.g., "OECD23").
    mriot_year : int
        The specific year of the MRIOT to retrieve.
    redownload : bool, optional
        Indicates whether to force redownload (and parsing) of the data (default is False).
    save : bool, optional
        Indicates whether to save the processed MRIOT data (default is True).

    Returns
    -------
    pymrio.IOSystem
        The loaded or newly parsed MRIOT represented as an IOSystem.

    Notes
    -----
    The function manages both the downloading of the MRIOT data and the parsing
    of the data into an appropriate format. If the data for the specified year and
    type is already available, it will load the existing data unless redownload
    is set to True. If the data must be parsed, it can optionally be saved
    to the designated location.
    """
    downloads_dir = MRIOT_DIRECTORY / mriot_type / "downloads"

    if mriot_type == "OECD23":
        year_group = f"{2001 + (mriot_year - 2001) // 5 * 5}-{2001 + (mriot_year - 2001) // 5 * 5 + 4}"
        downloads_dir = downloads_dir / year_group

    downloaded_file = downloads_dir / MRIOT_DEFAULT_FILENAME[mriot_type](mriot_year)
    # parsed data directory
    parsed_data_dir = MRIOT_DIRECTORY / mriot_type / str(mriot_year)

    if redownload and downloaded_file.exists():
        for fil in downloads_dir.iterdir():
            fil.unlink()
        downloads_dir.rmdir()
    if redownload and parsed_data_dir.exists():
        for fil in parsed_data_dir.iterdir():
            fil.unlink()
        parsed_data_dir.rmdir()

    if not downloaded_file.exists():
        download_mriot(mriot_type, mriot_year, downloads_dir)
    if not parsed_data_dir.exists():
        mriot = parse_mriot(mriot_type, downloaded_file, mriot_year)
        if save:
            mriot.save(parsed_data_dir, table_format="parquet")
    else:
        mriot = pymrio.load(path=parsed_data_dir)
        # Not too dirty trick to keep pymrio's saver/loader but have additional attributes.
        setattr(mriot, "monetary_factor", mriot.meta._content["monetary_factor"])
        setattr(mriot, "basename", mriot.meta._content["basename"])
        setattr(mriot, "year", mriot.meta._content["year"])
        setattr(mriot, "sectors_agg", mriot.meta._content["sectors_agg"])
        setattr(mriot, "regions_agg", mriot.meta._content["regions_agg"])

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


def _get_coco_MRIOT_name(mriot_name, is_custom=False):
    if is_custom:
        return "ISO3"

    match = MRIOT_FULLNAME_REGEX.match(mriot_name)
    if not match:
        raise ValueError(
            f"Input string '{mriot_name}' is not in the correct format '<MRIOT-name>_<year>' or not recognized."
        )

    mriot_type = match.group("mrio_type")
    return MRIOT_COUNTRY_CONVERTER_CORR[mriot_type]


def _fix_oecd_download_problem(download_dir):
    """Quick fix for https://github.com/IndEcol/pymrio/issues/156"""
    if download_dir.exists():
        for fil in download_dir.iterdir():
            fil.unlink()
        download_dir.rmdir()
