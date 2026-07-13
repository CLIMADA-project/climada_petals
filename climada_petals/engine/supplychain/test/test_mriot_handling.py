"""This file is part of CLIMADA.

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

Test Supplychain class utils.
"""

import pathlib
import re
import unittest
from unittest.mock import MagicMock, call, patch
import pytest
import pymrio
import pandas as pd
import numpy as np

from climada_petals.engine.supplychain.mriot_handling import (
    MRIOT_MONETARY_FACTOR,
    MRIOT_YEAR_REGEX,
    WIOD_FILE_LINK,
    build_eora_from_zip,
    build_exio3_from_zip,
    build_oecd_from_csv,
    download_mriot,
    get_mriot,
    lexico_reindex,
    parse_mriot,
    parse_mriot_from_df,
    parse_wiod_v2016,
)

class TestLexicoReindex(unittest.TestCase):
    def test_lexico_reindex(self):
        example_mriot = pymrio.load_test().calc_all()
        sorted_mriot = lexico_reindex(example_mriot)

        # Check if matrices Z, Y, x, and A are sorted
        pd.testing.assert_frame_equal(
            sorted_mriot.Z, example_mriot.Z.sort_index(axis=0).sort_index(axis=1)
        )
        pd.testing.assert_frame_equal(
            sorted_mriot.Y, example_mriot.Y.sort_index(axis=0).sort_index(axis=1)
        )
        pd.testing.assert_frame_equal(
            sorted_mriot.x, example_mriot.x.sort_index(axis=0).sort_index(axis=1)
        )
        pd.testing.assert_frame_equal(
            sorted_mriot.A, example_mriot.A.sort_index(axis=0).sort_index(axis=1)
        )


class TestBuildExio3FromZip(unittest.TestCase):

    def setUp(self):
        # Common setup for all tests
        self.mock_io_system = MagicMock()
        self.mock_io_system.meta = MagicMock()
        self.mock_io_system.meta.description = "Description - year = 2021"
        self.mock_io_system.calc_all = MagicMock()
        self.mock_io_system.get_regions = MagicMock(
            return_value=pd.Index(
                ["Region A", "Region B", "WA", "WE", "WF", "WL", "WM", "ZA"]
            )
        )
        self.mock_io_system.aggregate = MagicMock(return_value=self.mock_io_system)

        # Mocking return values for parse_exiobase3
        patcher_parse = patch(
            "pymrio.parse_exiobase3", return_value=self.mock_io_system
        )
        self.addCleanup(patcher_parse.stop)
        self.mock_parse_exiobase3 = patcher_parse.start()

        patcher_reindex = patch(
            "climada_petals.engine.supplychain.mriot_handling.lexico_reindex",
            return_value=self.mock_io_system,
        )
        self.addCleanup(patcher_reindex.stop)
        self.mock_lexico_reindex = patcher_reindex.start()

        # Dummy zip path
        self.mock_zip_path = "dummy_exio3.zip"

    def tearDown(self):
        # Stop all patches to clean up after each test
        patch.stopall()

    def test_build_exio3_from_zip(self):
        # Call the function under test
        mrio = build_exio3_from_zip(
            self.mock_zip_path, remove_attributes=False, aggregate_ROW=True
        )

        # Ensure parse_exiobase3 was called correctly
        self.mock_parse_exiobase3.assert_called_once_with(
            path=pathlib.Path(self.mock_zip_path)
        )

        # Check unnecessary attributes are removed
        self.mock_io_system.calc_all.assert_called_once()

        # Check lexicographical reindexing
        self.mock_lexico_reindex.assert_called_once_with(self.mock_io_system)

        # Check if aggregation for ROW regions is done
        self.mock_io_system.aggregate.assert_called_once()

        # Check if the attributes are set correctly
        self.assertEqual(mrio.monetary_factor, MRIOT_MONETARY_FACTOR["EXIOBASE3"])
        self.assertEqual(mrio.basename, "exiobase3_ixi")
        self.assertEqual(mrio.year, self.mock_io_system.meta.description[-4:])
        self.assertEqual(mrio.sectors_agg, "full_sectors")
        self.assertEqual(mrio.regions_agg, "full_regions")

        # Ensure the meta attributes are updated
        self.mock_io_system.meta.change_meta.assert_any_call(
            "monetary_factor", MRIOT_MONETARY_FACTOR["EXIOBASE3"]
        )
        self.mock_io_system.meta.change_meta.assert_any_call(
            "year", self.mock_io_system.meta.description[-4:]
        )
        self.mock_io_system.meta.change_meta.assert_any_call(
            "basename", "exiobase3_ixi"
        )
        self.mock_io_system.meta.change_meta.assert_any_call(
            "sectors_agg", "full_sectors"
        )
        self.mock_io_system.meta.change_meta.assert_any_call(
            "regions_agg", "full_regions"
        )

    def test_build_exio3_without_removing_attributes(self):
        # Call the function with remove_attributes=False
        build_exio3_from_zip(
            self.mock_zip_path, remove_attributes=False, aggregate_ROW=False
        )

        # Ensure lexicographical reindexing and calc_all are still called
        self.mock_io_system.calc_all.assert_called_once()
        self.mock_lexico_reindex.assert_called_once_with(self.mock_io_system)

        # Ensure aggregation for ROW is skipped
        self.mock_io_system.aggregate.assert_not_called()


class TestBuildEoraFromZip(unittest.TestCase):

    def setUp(self):
        # Common setup for all tests
        self.mock_io_system = MagicMock()
        self.mock_io_system.meta = MagicMock()
        self.mock_io_system.calc_all = MagicMock()
        self.mock_io_system.rename_sectors = MagicMock()
        self.mock_io_system.aggregate_duplicates = MagicMock()

        # Mock final demand Y attribute to simulate negative values
        self.mock_io_system.Y = pd.DataFrame(
            {"Sector A": [100, -50, 200], "Sector B": [-30, 150, -10]},
            index=["Region 1", "Region 2", "Region 3"],
        )

        # Mocking return values for parse_eora26
        patcher_parse = patch("pymrio.parse_eora26", return_value=self.mock_io_system)
        self.addCleanup(patcher_parse.stop)
        self.mock_parse_eora26 = patcher_parse.start()

        patcher_reindex = patch(
            "climada_petals.engine.supplychain.mriot_handling.lexico_reindex",
            return_value=self.mock_io_system,
        )
        self.addCleanup(patcher_reindex.stop)
        self.mock_lexico_reindex = patcher_reindex.start()

        # Dummy zip path
        self.mock_zip_path = "dummy_eora_2021.zip"
        self.mrio_year_match = {"mrio_year": "2021"}

        patcher_regex = patch("re.search", return_value=self.mrio_year_match)
        self.addCleanup(patcher_regex.stop)
        self.mock_re_search = patcher_regex.start()

    def tearDown(self):
        # Stop all patches to clean up after each test
        patch.stopall()

    def test_build_eora_from_zip_default(self):
        # Call the function under test with default parameters
        mrio = build_eora_from_zip(self.mock_zip_path, remove_attributes=False)

        # Ensure parse_eora26 was called correctly
        self.mock_parse_eora26.assert_called_once_with(
            path=pathlib.Path(self.mock_zip_path)
        )

        # Ensure unnecessary attributes are removed
        self.mock_io_system.calc_all.assert_called_once()

        # Ensure lexicographical reindexing
        self.mock_lexico_reindex.assert_called_once_with(self.mock_io_system)

        # Check if attributes are set correctly
        self.assertEqual(mrio.monetary_factor, MRIOT_MONETARY_FACTOR["EORA26"])
        self.assertEqual(mrio.basename, "eora26")
        self.assertEqual(
            mrio.year, re.search(MRIOT_YEAR_REGEX, self.mock_zip_path)["mrio_year"]
        )
        self.assertEqual(mrio.sectors_agg, "full_sectors")
        self.assertEqual(mrio.regions_agg, "full_regions")

        # Ensure the meta attributes are updated
        self.mock_io_system.meta.change_meta.assert_any_call(
            "monetary_factor", MRIOT_MONETARY_FACTOR["EORA26"]
        )
        self.mock_io_system.meta.change_meta.assert_any_call(
            "year", self.mrio_year_match["mrio_year"]
        )
        self.mock_io_system.meta.change_meta.assert_any_call("basename", "eora26")
        self.mock_io_system.meta.change_meta.assert_any_call(
            "sectors_agg", "full_sectors"
        )
        self.mock_io_system.meta.change_meta.assert_any_call(
            "regions_agg", "full_regions"
        )

        # Ensure re-export treatment was not applied
        self.mock_io_system.rename_sectors.assert_not_called()
        self.mock_io_system.aggregate_duplicates.assert_not_called()

    def test_build_eora_with_reexport_treatment(self):
        # Call the function with reexport_treatment=True
        mrio = build_eora_from_zip(
            self.mock_zip_path, reexport_treatment=True, remove_attributes=False
        )

        # Ensure the re-export treatment is applied
        self.mock_io_system.rename_sectors.assert_called_once_with(
            {"Re-export & Re-import": "Others"}
        )
        self.mock_io_system.aggregate_duplicates.assert_called_once()

        # Ensure the sectors_agg attribute is updated correctly
        self.assertEqual(mrio.sectors_agg, "full_no_reexport_sectors")

    def test_build_eora_with_inventory_treatment(self):
        # Mock final demand Y attribute to simulate negative values
        # Call the function with inv_treatment=True
        mriot = build_eora_from_zip(
            self.mock_zip_path, inv_treatment=True, remove_attributes=False
        )

        pd.testing.assert_frame_equal(
            self.mock_io_system.Y,
            pd.DataFrame(
                {
                    "Sector A": [100, 0, 200],  # Negative values clipped to 0
                    "Sector B": [0, 150, 0],
                },
                index=["Region 1", "Region 2", "Region 3"],
            ),
        )

    def test_build_eora_without_inventory_treatment(self):
        # Call the function with inv_treatment=False
        build_eora_from_zip(
            self.mock_zip_path, inv_treatment=False, remove_attributes=False
        )

        # Ensure no modification to Y occurs when inv_treatment is False
        pd.testing.assert_frame_equal(
            self.mock_io_system.Y,
            pd.DataFrame(
                {
                    "Sector A": [100, -50, 200],  # No clipping, negative values remain
                    "Sector B": [-30, 150, -10],
                },
                index=["Region 1", "Region 2", "Region 3"],
            ),
        )


class TestBuildOECDFromCSV(unittest.TestCase):

    def setUp(self):
        # Mock IOSystem and its methods/attributes
        self.mock_io_system = MagicMock()
        self.mock_io_system.meta = MagicMock()
        self.mock_io_system.calc_all = MagicMock()
        self.mock_io_system.get_regions = MagicMock(
            return_value=pd.Index(["Region A", "Region B", "Region C"])
        )
        self.mock_io_system.aggregate = MagicMock(return_value=self.mock_io_system)

        # Mocking return values for parse_oecd
        patcher_parse = patch("pymrio.parse_oecd", return_value=self.mock_io_system)
        self.addCleanup(patcher_parse.stop)
        self.mock_parse_oecd = patcher_parse.start()

        patcher_reindex = patch(
            "climada_petals.engine.supplychain.mriot_handling.lexico_reindex",
            return_value=self.mock_io_system,
        )
        self.addCleanup(patcher_reindex.stop)
        self.mock_lexico_reindex = patcher_reindex.start()

        # Dummy csv path
        self.mock_csv_path = "dummy_oecd_2020.csv"

    def tearDown(self):
        # Stop all patches to clean up after each test
        patch.stopall()

    def test_build_oecd_from_csv(self):
        # Call the function under test with year and remove_attributes=True
        mrio = build_oecd_from_csv(
            self.mock_csv_path, year=2020, remove_attributes=False
        )

        # Ensure parse_oecd was called correctly
        self.mock_parse_oecd.assert_called_once_with(
            path=pathlib.Path(self.mock_csv_path), year=2020
        )

        self.mock_io_system.calc_all.assert_called_once()

        # Check if lexicographical reindexing is done
        self.mock_lexico_reindex.assert_called_once_with(self.mock_io_system)

        # Check if the attributes are set correctly
        self.assertEqual(mrio.monetary_factor, MRIOT_MONETARY_FACTOR["OECD23"])
        self.assertEqual(mrio.basename, "icio_v2023")
        self.assertEqual(
            mrio.year, re.search(MRIOT_YEAR_REGEX, self.mock_csv_path)["mrio_year"]
        )
        self.assertEqual(mrio.sectors_agg, "full_sectors")
        self.assertEqual(mrio.regions_agg, "full_regions")

        # Ensure the meta attributes are updated
        self.mock_io_system.meta.change_meta.assert_any_call(
            "monetary_factor", MRIOT_MONETARY_FACTOR["OECD23"]
        )
        self.mock_io_system.meta.change_meta.assert_any_call(
            "year", re.search(MRIOT_YEAR_REGEX, self.mock_csv_path)["mrio_year"]
        )
        self.mock_io_system.meta.change_meta.assert_any_call("basename", "icio_v2023")
        self.mock_io_system.meta.change_meta.assert_any_call(
            "sectors_agg", "full_sectors"
        )
        self.mock_io_system.meta.change_meta.assert_any_call(
            "regions_agg", "full_regions"
        )

    def test_build_oecd_from_csv_without_removing_attributes(self):
        # Call the function with remove_attributes=False
        mrio = build_oecd_from_csv(
            self.mock_csv_path, year=2020, remove_attributes=False
        )

        # Ensure no attributes are removed
        self.mock_io_system.calc_all.assert_called_once()

        # Ensure lexicographical reindexing is still done
        self.mock_lexico_reindex.assert_called_once_with(self.mock_io_system)

    def test_build_oecd_with_invalid_year_in_filename(self):
        # Mock invalid year in file name and ensure it handles it correctly
        invalid_filename = "dummy_oecd_invalid_year.csv"
        with self.assertRaises(TypeError):
            build_oecd_from_csv(invalid_filename, remove_attributes=False)


class TestParseWIODv2016(unittest.TestCase):

    def setUp(self):
        # Mock IOSystem and its methods/attributes
        self.mock_io_system = MagicMock()
        self.mock_io_system.meta = MagicMock()
        self.mock_io_system.calc_all = MagicMock()
        self.mock_io_system.get_regions = MagicMock(
            return_value=pd.Index(["Region A", "Region B", "Region C"])
        )
        self.mock_io_system.get_sectors = MagicMock(
            return_value=pd.Index(["Sector 1", "Sector 2", "Sector 3"])
        )

        # Mocking parse_mriot_from_df return values
        self.mock_parse_mriot_from_df = patch(
            "climada_petals.engine.supplychain.mriot_handling.parse_mriot_from_df",
            return_value=(MagicMock(), MagicMock(), MagicMock()),  # Z, Y, x matrices
        ).start()

        # Mocking return values for read_excel
        patcher_read_excel = patch("pandas.read_excel", return_value=MagicMock())
        self.addCleanup(patcher_read_excel.stop)
        self.mock_read_excel = patcher_read_excel.start()

        # Mocking IOSystem initialization
        patcher_iosystem = patch("pymrio.IOSystem", return_value=self.mock_io_system)
        self.addCleanup(patcher_iosystem.stop)
        self.mock_iosystem = patcher_iosystem.start()

        # Mocking reindex function
        patcher_reindex = patch(
            "climada_petals.engine.supplychain.mriot_handling.lexico_reindex",
            return_value=self.mock_io_system,
        )
        self.addCleanup(patcher_reindex.stop)
        self.mock_lexico_reindex = patcher_reindex.start()

        # Dummy xlsb path
        self.mock_xlsb_path = "dummy_wiod_2010.xlsb"

    def tearDown(self):
        # Stop all patches to clean up after each test
        patch.stopall()

    def test_parse_wiod_v2016(self):
        # Call the function under test
        mrio = parse_wiod_v2016(self.mock_xlsb_path)

        # Ensure read_excel was called with the correct arguments
        self.mock_read_excel.assert_called_once_with(
            self.mock_xlsb_path, engine="pyxlsb"
        )

        # Ensure parse_mriot_from_df was called with the expected parameters
        self.mock_parse_mriot_from_df.assert_called_once_with(
            self.mock_read_excel.return_value,
            col_iso3=2,
            col_sectors=1,
            row_fd_cats=2,
            rows_data=(5, 2469),
            cols_data=(4, 2468),
        )

        # Ensure IOSystem is initialized with the correct Z, Y, and x matrices
        self.mock_iosystem.assert_called_once_with(
            Z=self.mock_parse_mriot_from_df.return_value[0],
            Y=self.mock_parse_mriot_from_df.return_value[1],
            x=self.mock_parse_mriot_from_df.return_value[2],
        )

        # Ensure MultiIndex is created correctly for 'unit'
        expected_multiindex = pd.MultiIndex.from_product(
            [
                self.mock_io_system.get_regions.return_value,
                self.mock_io_system.get_sectors.return_value,
            ],
            names=["region", "sector"],
        )
        pd.testing.assert_frame_equal(
            self.mock_io_system.unit,
            pd.DataFrame(
                data=np.repeat(["M.USD"], len(expected_multiindex)),
                index=expected_multiindex,
                columns=["unit"],
            ),
        )

        # Check if the attributes are set correctly
        self.assertEqual(mrio.monetary_factor, MRIOT_MONETARY_FACTOR["WIOD16"])
        self.assertEqual(mrio.basename, "wiod_v2016")
        self.assertEqual(
            mrio.year, re.search(MRIOT_YEAR_REGEX, self.mock_xlsb_path)["mrio_year"]
        )
        self.assertEqual(mrio.sectors_agg, "full_sectors")
        self.assertEqual(mrio.regions_agg, "full_regions")

        # Ensure the meta attributes are updated
        self.mock_io_system.meta.change_meta.assert_any_call(
            "monetary_factor", MRIOT_MONETARY_FACTOR["WIOD16"]
        )
        self.mock_io_system.meta.change_meta.assert_any_call(
            "year", re.search(MRIOT_YEAR_REGEX, self.mock_xlsb_path)["mrio_year"]
        )
        self.mock_io_system.meta.change_meta.assert_any_call("basename", "wiod_v2016")
        self.mock_io_system.meta.change_meta.assert_any_call(
            "sectors_agg", "full_sectors"
        )
        self.mock_io_system.meta.change_meta.assert_any_call(
            "regions_agg", "full_regions"
        )

        # Ensure calc_all and reindex are called
        self.mock_io_system.calc_all.assert_called_once()
        self.mock_lexico_reindex.assert_called_once_with(self.mock_io_system)


class TestParseFromDF(unittest.TestCase):
    def dummy_mriot_df(self):
        "Generate dummy DataFrame containing MRIOT data"
        data = {
            "iso3": [None, "USA", "USA", "CHN", "CHN"],  # Region codes
            "sector": [
                None,
                "Agriculture",
                "Industry",
                "Agriculture",
                "Industry",
            ],  # Sector names
            # Intermediate demand data (Z matrix)
            "USA_Agr": [None, 100, 50, 30, 20],
            "USA_Ind": [None, 40, 150, 25, 60],
            "CHN_Agr": [None, 30, 20, 100, 50],
            "CHN_Ind": [None, 10, 40, 50, 200],
            # Final demand (Y matrix) and total output (x)
            "USA_fd1": ["FD_1", 300, 400, 500, 600],  # Final demand for USA
            "USA_fd2": ["FD_2", 100, 200, 300, 400],  # Final demand for CHN
            "CHN_fd1": ["FD_1", 300, 400, 500, 600],  # Final demand for USA
            "CHN_fd2": ["FD_2", 100, 200, 300, 400],  # Final demand for CHN
            "total_output": [None, 800, 900, 1000, 1100],  # Total production (x)
        }

        # Convert the dictionary into a pandas dataframe

        mriot_df = pd.DataFrame(data)
        col_iso3 = 0  # The column index where the region names are stored
        col_sectors = 1  # The column index where the sector names are stored
        rows_data = (1, 5)  # Row range containing the intermediate demand matrix
        cols_data = (2, 6)  # Column range containing the intermediate demand matrix (Z)
        row_fd_cats = 0
        return mriot_df, col_iso3, col_sectors, rows_data, cols_data, row_fd_cats

    def test_parse_mriot_from_df(self):
        mriot_df, col_iso3, col_sectors, rows_data, cols_data, row_fd_cats = (
            self.dummy_mriot_df()
        )
        Z, Y, x = parse_mriot_from_df(
            mriot_df, col_iso3, col_sectors, rows_data, cols_data, row_fd_cats
        )
        expected_Z = pd.DataFrame.from_dict(
            {
                "index": [
                    ("USA", "Agriculture"),
                    ("USA", "Industry"),
                    ("CHN", "Agriculture"),
                    ("CHN", "Industry"),
                ],
                "columns": [
                    ("USA", "Agriculture"),
                    ("USA", "Industry"),
                    ("CHN", "Agriculture"),
                    ("CHN", "Industry"),
                ],
                "data": [
                    [100.0, 40.0, 30.0, 10.0],
                    [50.0, 150.0, 20.0, 40.0],
                    [30.0, 25.0, 100.0, 50.0],
                    [20.0, 60.0, 50.0, 200.0],
                ],
                "index_names": ["region", "sector"],
                "column_names": ["region", "sector"],
            },
            "tight",
        )

        expected_Y = pd.DataFrame.from_dict(
            {
                "index": [
                    ("USA", "Agriculture"),
                    ("USA", "Industry"),
                    ("CHN", "Agriculture"),
                    ("CHN", "Industry"),
                ],
                "columns": [
                    ("USA", "FD_1"),
                    ("USA", "FD_2"),
                    ("CHN", "FD_1"),
                    ("CHN", "FD_2"),
                ],
                "data": [
                    [300.0, 100.0, 300.0, 100.0],
                    [400.0, 200.0, 400.0, 200.0],
                    [500.0, 300.0, 500.0, 300.0],
                    [600.0, 400.0, 600.0, 400.0],
                ],
                "index_names": ["region", "sector"],
                "column_names": ["region", "category"],
            },
            "tight",
        )

        expected_x = pd.DataFrame.from_dict(
            {
                "index": [
                    ("USA", "Agriculture"),
                    ("USA", "Industry"),
                    ("CHN", "Agriculture"),
                    ("CHN", "Industry"),
                ],
                "columns": ["indout"],
                "data": [[800.0], [900.0], [1000.0], [1100.0]],
                "index_names": ["region", "sector"],
                "column_names": [None],
            },
            "tight",
        )

        pd.testing.assert_frame_equal(Z, expected_Z)
        pd.testing.assert_frame_equal(Y, expected_Y)
        pd.testing.assert_frame_equal(x, expected_x)

    def test_parse_mriot_from_df_no_fd_cat(self):
        mriot_df, col_iso3, col_sectors, rows_data, cols_data, _ = self.dummy_mriot_df()
        Z, Y, x = parse_mriot_from_df(
            mriot_df, col_iso3, col_sectors, rows_data, cols_data, None
        )
        expected_Z = pd.DataFrame.from_dict(
            {
                "index": [
                    ("USA", "Agriculture"),
                    ("USA", "Industry"),
                    ("CHN", "Agriculture"),
                    ("CHN", "Industry"),
                ],
                "columns": [
                    ("USA", "Agriculture"),
                    ("USA", "Industry"),
                    ("CHN", "Agriculture"),
                    ("CHN", "Industry"),
                ],
                "data": [
                    [100.0, 40.0, 30.0, 10.0],
                    [50.0, 150.0, 20.0, 40.0],
                    [30.0, 25.0, 100.0, 50.0],
                    [20.0, 60.0, 50.0, 200.0],
                ],
                "index_names": ["region", "sector"],
                "column_names": ["region", "sector"],
            },
            "tight",
        )

        expected_Y = pd.DataFrame.from_dict(
            {
                "index": [
                    ("USA", "Agriculture"),
                    ("USA", "Industry"),
                    ("CHN", "Agriculture"),
                    ("CHN", "Industry"),
                ],
                "columns": [
                    ("USA", "fd_cat_0"),
                    ("USA", "fd_cat_1"),
                    ("CHN", "fd_cat_0"),
                    ("CHN", "fd_cat_1"),
                ],
                "data": [
                    [300.0, 100.0, 300.0, 100.0],
                    [400.0, 200.0, 400.0, 200.0],
                    [500.0, 300.0, 500.0, 300.0],
                    [600.0, 400.0, 600.0, 400.0],
                ],
                "index_names": ["region", "sector"],
                "column_names": ["region", "category"],
            },
            "tight",
        )

        expected_x = pd.DataFrame.from_dict(
            {
                "index": [
                    ("USA", "Agriculture"),
                    ("USA", "Industry"),
                    ("CHN", "Agriculture"),
                    ("CHN", "Industry"),
                ],
                "columns": ["indout"],
                "data": [[800.0], [900.0], [1000.0], [1100.0]],
                "index_names": ["region", "sector"],
                "column_names": [None],
            },
            "tight",
        )

        pd.testing.assert_frame_equal(Z, expected_Z)
        pd.testing.assert_frame_equal(Y, expected_Y)
        pd.testing.assert_frame_equal(x, expected_x)


class TestDownloadMRIOT(unittest.TestCase):
    @patch("pymrio.download_exiobase3")  # Mock pymrio's download function for EXIOBASE3
    def test_download_exiobase3(self, mock_download_exiobase3):
        # Mock parameters
        mriot_type = "EXIOBASE3"
        mriot_year = 2010
        download_dir = pathlib.Path("/fake/dir")

        # Call the function
        download_mriot(mriot_type, mriot_year, download_dir)

        # Assert that pymrio.download_exiobase3 was called with the correct arguments
        mock_download_exiobase3.assert_called_once_with(
            storage_folder=download_dir, system="ixi", years=[mriot_year]
        )

    @patch("pathlib.Path.rename")
    @patch("pathlib.Path.mkdir")
    @patch("zipfile.ZipFile")  # Mock zipfile extraction
    @patch(
        "climada.util.files_handler.download_file"
    )  # Mock custom download function for WIOD16
    def test_download_wiod16(
        self, mock_download_file, mock_zipfile, mock_mkdir, mock_rename
    ):
        # Mock parameters
        mriot_type = "WIOD16"
        mriot_year = 2016
        download_dir = pathlib.Path("/fake/dir")

        # Mock the download behavior
        mock_download_file.return_value = "fake_wiod_file"

        # Call the function
        download_mriot(mriot_type, mriot_year, download_dir)

        # Assert that u_fh.download_file was called once with the correct arguments
        mock_download_file.assert_called_once_with(
            WIOD_FILE_LINK,  # Assuming 'WIOD_FILE_LINK' is some constant defined in the function
            download_dir=download_dir,
        )
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_rename.assert_called_once_with(pathlib.Path("fake_wiod_file.zip"))

        # Check that the ZipFile mock was called to handle extraction
        mock_zipfile.assert_called_once()

    @patch("pymrio.download_oecd")  # Mock pymrio's download function for OECD21
    def test_download_oecd23(self, mock_download_oecd):
        # Mock parameters
        mriot_type = "OECD23"
        mriot_year = 2007
        download_dir = pathlib.Path("/fake/dir")

        # Call the function
        download_mriot(mriot_type, mriot_year, download_dir)

        # Assert that pymrio.download_oecd was called with the correct year group
        mock_download_oecd.assert_called_once_with(
            storage_folder=download_dir, years="2006-2010"
        )


class TestParseMriot(unittest.TestCase):
    @patch("climada_petals.engine.supplychain.mriot_handling.build_exio3_from_zip")
    def test_parse_mriot_exiobase3(self, mock_build_exio3):
        # Mock return object for pymrio.parse_exiobase3
        mock_mriot = MagicMock()
        mock_mriot.Y = pd.DataFrame(
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            index=pd.MultiIndex.from_tuples(
                [("reg1", "sec1"), ("reg1", "sec2"), ("reg2", "sec1"), ("reg2", "sec2")]
            ),
            columns=pd.MultiIndex.from_tuples(
                [("reg1", "fd1"), ("reg1", "fd2"), ("reg2", "fd1"), ("reg2", "fd2")]
            ),
        )
        mock_build_exio3.return_value = mock_mriot

        downloaded_file = pathlib.Path("dummy_path")
        mriot = parse_mriot("EXIOBASE3", downloaded_file, 2010, testkwarg="testkwarg")

        # Assert pymrio.parse_exiobase3 was called correctly
        mock_build_exio3.assert_called_once_with(
            mrio_zip=downloaded_file, testkwarg="testkwarg"
        )
        mock_mriot.meta.change_meta.assert_has_calls(
            [ call("name", "EXIOBASE3-2010") ], any_order=True
        )

    @patch("climada_petals.engine.supplychain.mriot_handling.parse_wiod_v2016")
    def test_parse_mriot_wiod16(self, mock_parse_wiod):
        mock_mriot = MagicMock()
        mock_mriot.Y = pd.DataFrame(
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            index=pd.MultiIndex.from_tuples(
                [("reg1", "sec1"), ("reg1", "sec2"), ("reg2", "sec1"), ("reg2", "sec2")]
            ),
            columns=pd.MultiIndex.from_tuples(
                [("reg1", "fd1"), ("reg1", "fd2"), ("reg2", "fd1"), ("reg2", "fd2")]
            ),
        )
        mock_parse_wiod.return_value = mock_mriot

        downloaded_file = pathlib.Path("dummy_path")
        mriot = parse_mriot("WIOD16", downloaded_file, 2010, testkwarg="testkwarg")

        # Assert pymrio.parse_exiobase3 was called correctly
        mock_parse_wiod.assert_called_once_with(mrio_xlsb=downloaded_file)
        mock_mriot.meta.change_meta.assert_has_calls(
            [ call("name", "WIOD16-2010") ], any_order=True
        )

    @patch("climada_petals.engine.supplychain.mriot_handling.build_oecd_from_csv")
    def test_parse_mriot_oecd23(self, mock_build_oecd):
        # Mock return object for pymrio.parse_oecd
        mock_mriot = MagicMock()
        mock_mriot.Y = pd.DataFrame(
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            index=pd.MultiIndex.from_tuples(
                [("reg1", "sec1"), ("reg1", "sec2"), ("reg2", "sec1"), ("reg2", "sec2")]
            ),
            columns=pd.MultiIndex.from_tuples(
                [("reg1", "fd1"), ("reg1", "fd2"), ("reg2", "fd1"), ("reg2", "fd2")]
            ),
        )
        mock_build_oecd.return_value = mock_mriot

        downloaded_file = pathlib.Path("dummy_path")
        parse_mriot("OECD23", downloaded_file, 2010)

        # Assert pymrio.parse_oecd was called correctly
        mock_build_oecd.assert_called_once_with(mrio_csv=downloaded_file, year=2010)
        mock_mriot.meta.change_meta.assert_has_calls(
            [ call("name", "OECD23-2010") ], any_order=True
        )

    @patch("climada_petals.engine.supplychain.mriot_handling.build_eora_from_zip")
    def test_parse_mriot_eora26(self, mock_build_eora):
        # Mock return object for pymrio.parse_oecd
        mock_mriot = MagicMock()

        # Also check for negative values in Y (which requires Z to be defined)
        mock_mriot.Z = pd.DataFrame(
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            index=pd.MultiIndex.from_tuples(
                [("reg1", "sec1"), ("reg1", "sec2"), ("reg2", "sec1"), ("reg2", "sec2")]
            ),
            columns=pd.MultiIndex.from_tuples(
                [("reg1", "sec1"), ("reg1", "sec2"), ("reg2", "sec1"), ("reg2", "sec2")]
            ),
        )
        mock_mriot.Y = pd.DataFrame(
            [[0, -2, -3, -4], [0, -2, -3, -4], [0, 2, 3, 4], [-1, 2, 3, 4]],
            index=pd.MultiIndex.from_tuples(
                [("reg1", "sec1"), ("reg1", "sec2"), ("reg2", "sec1"), ("reg2", "sec2")]
            ),
            columns=pd.MultiIndex.from_tuples(
                [("reg1", "fd1"), ("reg1", "fd2"), ("reg2", "fd1"), ("reg2", "fd2")]
            ),
        )
        mock_build_eora.return_value = mock_mriot

        downloaded_file = pathlib.Path("dummy_path")
        parse_mriot("EORA26", downloaded_file, 2010)

        # Assert pymrio.parse_oecd was called correctly
        mock_build_eora.assert_called_once_with(mrio_zip=downloaded_file)
        mock_mriot.meta.change_meta.assert_has_calls(
            calls=[ call("name", "EORA26-2010") ], any_order=True
        )

    def test_parse_mriot_unknown_type(self):
        downloaded_file = pathlib.Path("dummy_path")
        with self.assertRaises(RuntimeError):
            parse_mriot("UNKNOWN", downloaded_file, 2010)
