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
from typing import cast
import warnings
import numpy as np
import pathlib
from climada_petals.engine.supplychain import LOGGER
import pymrio
import pandas as pd
import zipfile

from climada import CONFIG
from climada.util import files_handler as u_fh
import re


MRIOT_DIRECTORY = CONFIG.engine.supplychain.local_data.mriot.dir()
"""Directory where Multi-Regional Input-Output Tables (MRIOT) are downloaded."""

MRIOT_TYPE_REGEX = (
    r"(?P<mrio_type>OECD23|EXIOBASE3|EORA26|WIOD16)"
 )
MRIOT_YEAR_REGEX = r"(?P<mrio_year>\d{4})"
MRIOT_FULLNAME_REGEX = re.compile("{}-{}".format(MRIOT_TYPE_REGEX,MRIOT_YEAR_REGEX))

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

MRIOT_COUNTRY_CONVERTER_CORR = {
    "EXIOBASE3" : "EXIO3",
    "WIOD16" : "WIOD",
    "EORA26" : "Eora",
    "OECD23" : "ISO3"
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

ICIO23_sectors_mapping = {'A01_02': 'Agriculture, hunting, forestry',
 'A03': 'Fishing and aquaculture',
 'B05_06': 'Mining and quarrying, energy producing products',
 'B07_08': 'Mining and quarrying, non-energy producing products',
 'B09': 'Mining support service activities',
 'C10T12': 'Food products, beverages and tobacco',
 'C13T15': 'Textiles, textile products, leather and footwear',
 'C16': 'Wood and products of wood and cork',
 'C17_18': 'Paper products and printing',
 'C19': 'Coke and refined petroleum products',
 'C20': 'Chemical and chemical products',
 'C21': 'Pharmaceuticals, medicinal chemical and botanical products',
 'C22': 'Rubber and plastics products',
 'C23': 'Other non-metallic mineral products',
 'C24': 'Basic metals',
 'C25': 'Fabricated metal products',
 'C26': 'Computer, electronic and optical equipment',
 'C27': 'Electrical equipment',
 'C28': 'Machinery and equipment, nec ',
 'C29': 'Motor vehicles, trailers and semi-trailers',
 'C30': 'Other transport equipment',
 'C31T33': 'Manufacturing nec; repair and installation of machinery and equipment',
 'D': 'Electricity, gas, steam and air conditioning supply',
 'E': 'Water supply; sewerage, waste management and remediation activities',
 'F': 'Construction',
 'G': 'Wholesale and retail trade; repair of motor vehicles',
 'H49': 'Land transport and transport via pipelines',
 'H50': 'Water transport',
 'H51': 'Air transport',
 'H52': 'Warehousing and support activities for transportation',
 'H53': 'Postal and courier activities',
 'I': 'Accommodation and food service activities',
 'J58T60': 'Publishing, audiovisual and broadcasting activities',
 'J61': 'Telecommunications',
 'J62_63': 'IT and other information services',
 'K': 'Financial and insurance activities',
 'L': 'Real estate activities',
 'M': 'Professional, scientific and technical activities',
 'N': 'Administrative and support services',
 'O': 'Public administration and defence; compulsory social security',
 'P': 'Education',
 'Q': 'Human health and social work activities',
 'R': 'Arts, entertainment and recreation',
 'S': 'Other service activities',
 'T': 'Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use'}

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
        The sorted IOSystem.
    """

    for matrix_name in ["Z", "Y", "x", "A", "As", "G", "L"]:
        matrix = getattr(mriot, matrix_name)
        if matrix is not None:
            setattr(
                mriot, matrix_name, matrix.reindex(sorted(matrix.index)).sort_index(axis=1)
            )

    return mriot


def build_exio3_from_zip(mrio_zip: str, remove_attributes=True, aggregate_ROW=True):
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

    mrio_pym.meta.change_meta("name", "EXIOBASE3")

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
    setattr(mrio_pym, "basename", "exiobase3_ixi")
    setattr(mrio_pym, "year", mrio_pym.meta.description[-4:])
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    # Also put it in meta for saving
    mrio_pym.meta.change_meta("monetary_factor", MRIOT_MONETARY_FACTOR["EXIOBASE3"])
    mrio_pym.meta.change_meta("year", mrio_pym.meta.description[-4:])
    mrio_pym.meta.change_meta("basename", "exiobase3_ixi")
    mrio_pym.meta.change_meta("sectors_agg", "full_sectors")
    mrio_pym.meta.change_meta("regions_agg", "full_regions")
    return mrio_pym


def build_eora_from_zip(
    mrio_zip: str,
    reexport_treatment=False,
    inv_treatment=True,
    remove_attributes=True,
):
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
    setattr(mrio_pym, "basename", "eora26")
    setattr(mrio_pym, "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"])
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    # Also put it in meta for saving
    mrio_pym.meta.change_meta("monetary_factor", MRIOT_MONETARY_FACTOR["EORA26"])
    mrio_pym.meta.change_meta(
        "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"]
    )
    mrio_pym.meta.change_meta("basename", "eora26")
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

    LOGGER.info("Re-indexing lexicographicaly")
    mrio_pym = lexico_reindex(mrio_pym)
    LOGGER.info("Done")

    LOGGER.info("Computing the missing IO components")
    mrio_pym.calc_all()
    LOGGER.info("Done")

    return mrio_pym


def build_oecd_from_csv(
    mrio_csv: str, year: int | None = None, remove_attributes: bool = True
):
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
    setattr(mrio_pym, "basename", "icio_v2023")
    setattr(mrio_pym, "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"])
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    # Also put it in meta for saving
    mrio_pym.meta.change_meta("monetary_factor", MRIOT_MONETARY_FACTOR["OECD23"])
    mrio_pym.meta.change_meta(
        "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"]
    )
    mrio_pym.meta.change_meta("basename", "icio_v2023")
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
    setattr(mrio_pym, "basename", "wiod_v2016")
    setattr(mrio_pym, "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"])
    setattr(mrio_pym, "sectors_agg", "full_sectors")
    setattr(mrio_pym, "regions_agg", "full_regions")
    # Also put it in meta for saving
    mrio_pym.meta.change_meta("monetary_factor", MRIOT_MONETARY_FACTOR["WIOD16"])
    mrio_pym.meta.change_meta(
        "year", re.search(MRIOT_YEAR_REGEX, mrio_path.name)["mrio_year"]
    )
    mrio_pym.meta.change_meta("basename", "wiod_v2016")
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


def download_mriot(mriot_type, mriot_year, download_dir):
    """Download EXIOBASE3, WIOD16 or OECD23 Multi-Regional Input Output Tables
    for specific years.

    Parameters
    ----------
    mriot_type : str
    mriot_year : int
    download_dir : pathlib.PosixPath

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
        years_groups = ["1995-2000", "2001-2005", "2006-2010", "2011-2015", "2016-2020"]
        year_group = years_groups[int(np.floor((mriot_year - 1995) / 5))-1]

        pymrio.download_oecd(storage_folder=download_dir, years=year_group)


def parse_mriot(mriot_type, downloaded_file, mriot_year, **kwargs):
    """Parse EXIOBASE3, WIOD16 or OECD23 MRIOT for specific years

    Parameters
    ----------
    mriot_type : str
    downloaded_file : pathlib.PosixPath

    Notes
    -----
    The parsing of EXIOBASE3 and OECD23 tables makes use of pymrio functions.
    The parsing of WIOD16 tables requires ad-hoc functions, since the
    related pymrio functions were broken at the time of implementation
    of this function.

    Some metadata is rewrote or added to the objects for consistency in usage (name, monetary factor, year).
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


    return mriot


def get_mriot(mriot_type, mriot_year, redownload=False, save=True):
    # if data were parsed and saved: load them
    downloads_dir = MRIOT_DIRECTORY / mriot_type / "downloads"
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

def _get_coco_MRIOT_name(mriot_name):
    match = MRIOT_FULLNAME_REGEX.match(mriot_name)
    if not match:
        raise ValueError(f"Input string '{mriot_name}' is not in the correct format '<MRIOT-name>_<year>' or not recognized.")
    mriot_type = match.group("mrio_type")
    return MRIOT_COUNTRY_CONVERTER_CORR[mriot_type]
