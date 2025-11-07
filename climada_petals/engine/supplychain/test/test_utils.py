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

import unittest
from unittest.mock import Mock, patch, MagicMock
from climada.engine.impact import Impact
from climada.entity.exposures import Exposures
from climada.util import DEF_CRS
from geopandas import gpd
import numpy as np
import pandas as pd
from scipy import sparse
from climada_petals.engine.supplychain.utils import (
    calc_G,
    calc_va,
    calc_B,
    VA_NAME,
    calc_x_from_G,
    check_sectors_in_mriot,
    distribute_reg_impact_to_sectors,
    translate_exp_to_regions,
    translate_exp_to_sectors,
    translate_imp_to_regions,
)


from climada.util import files_handler as u_fh
import pathlib
import unittest
import warnings
import numpy as np
import pandas as pd
import pymrio


def dummy_mriot_df():
    "Generate dummy DataFrame containing MRIOT data"
    data = {
        "iso3": [None, "USA", "USA", "CHN", "CHN"],  # Region codes
        "sector": [ None,
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


def dummy_exp_imp():
    "Generate dummy exposure and impacts"
    lat = np.array([1, 3])
    lon = np.array([1.5, 3.5])
    exp = Exposures(crs=DEF_CRS)
    exp.gdf["longitude"] = lon
    exp.gdf["latitude"] = lat
    exp.gdf["value"] = np.array([150.0, 80.0])
    exp.gdf["region_id"] = [840, 608]  # USA, PHL (ROW)

    imp = Impact(
        event_id=np.arange(2) + 10,
        event_name=np.arange(2),
        date=np.arange(2),
        coord_exp=np.vstack([lon, lat]).T,
        crs=DEF_CRS,
        unit="USD",
        eai_exp=np.array([6, 4.33]),
        at_event=np.array([55, 35]),
        frequency=np.array([1 / 6, 1 / 30]),
        frequency_unit="1/month",
        aai_agg=10.34,
        imp_mat=sparse.csr_matrix(np.array([[30, 25], [30, 5]])),
    )

    return exp, imp


MRIOT_INDEX = pd.MultiIndex.from_tuples(
    [
        ("USA", "Sector A"),
        ("USA", "Sector B"),
        ("ROW", "Sector A"),
        ("ROW", "Sector B"),
    ],
    names=["region", "sector"],
)

MRIOT_INDEX_FD = pd.MultiIndex.from_tuples(
    [
        ("USA", "FD 1"),
        ("USA", "FD 2"),
        ("ROW", "FD 1"),
        ("ROW", "FD 2"),
    ],
    names=["region", "sector"],
)


def dummy_mriot():
    mriot = pymrio.IOSystem()
    A = pd.DataFrame(
        [
            [0.24, 0.1, 0.02, 0.04],
            [0.18, 0.20, 0.12, 0.1],
            [0.1, 0.02, 0.20, 0.12],
            [0.12, 0.04, 0.24, 0.20],
        ],
        index=MRIOT_INDEX,
        columns=MRIOT_INDEX,
    )

    Y = pd.DataFrame(
        [
            [100.0, 60.0, 20.0, 20.0],
            [80.0, 40.0, 100.0, 184.0],
            [60.0, 20.0, 24.0, 12.0],
            [80.0, 40.0, 48.0, 36.0],
        ],
        index=MRIOT_INDEX,
        columns=MRIOT_INDEX_FD,
    )
    # This makes sure the table is balanced
    mriot.A = A
    mriot.Y = Y
    mriot.L = pymrio.calc_L(A)
    mriot.x = pymrio.calc_x_from_L(mriot.L, mriot.Y.sum(axis=1))
    mriot.Z = pymrio.calc_Z(A, mriot.x)


class TestUtils(unittest.TestCase):

    def test_calc_va_with_dataframe(self):
        Z = pd.DataFrame(
            [[0.0, 1.0], [1.0, 0.0]],
            index=["Industry A", "Industry B"],
            columns=["Industry A", "Industry B"],
        )
        x = pd.DataFrame(
            [10.0, 20.0], index=["Industry A", "Industry B"], columns=["indout"]
        )

        expected_va = pd.DataFrame(
            [[9.0, 19.0]], index=[VA_NAME], columns=["Industry A", "Industry B"]
        )

        result = calc_va(Z, x)
        pd.testing.assert_frame_equal(result, expected_va)

    def test_calc_va_with_numpy_array(self):
        Z = np.array([[0.0, 1.0], [1.0, 0.0]])
        x = np.array([10.0, 20.0])

        expected_va = np.array([[9.0, 19.0]])

        result = calc_va(Z, x)
        np.testing.assert_array_equal(result, expected_va)

    def test_calc_va_invalid_input(self):
        Z = "invalid input"
        x = pd.DataFrame(
            [10.0, 20.0], index=["Industry A", "Industry B"], columns=["indout"]
        )

        with self.assertRaises((TypeError, AttributeError)):
            calc_va(Z, x)  # type: ignore

    def test_calc_B_with_dataframe(self):
        Z = pd.DataFrame(
            [[0.0, 2.0], [3.0, 0.0]],
            index=["Industry A", "Industry B"],
            columns=["Industry A", "Industry B"],
        )
        x = pd.DataFrame(
            [10.0, 20.0], index=["Industry A", "Industry B"], columns=["indout"]
        )

        expected_B = pd.DataFrame(
            [[0.0, 0.15], [0.2, 0.0]],
            index=["Industry A", "Industry B"],
            columns=["Industry A", "Industry B"],
        )

        result = calc_B(Z, x)
        pd.testing.assert_frame_equal(result, expected_B)

    def test_calc_B_with_numpy_array(self):
        Z = np.array([[0.0, 2.0], [3.0, 0.0]])
        x = np.array([10.0, 20.0])

        expected_B = np.array([[0.0, 0.15], [0.2, 0.0]])

        result = calc_B(Z, x)
        np.testing.assert_array_almost_equal(result, expected_B)

    def test_calc_B_single_industry(self):
        Z = pd.DataFrame([[0.0]], index=["Industry A"], columns=["Industry A"])
        x = pd.DataFrame([10.0], index=["Industry A"], columns=["indout"])

        expected_B = pd.DataFrame([[0.0]], index=["Industry A"], columns=["Industry A"])

        result = calc_B(Z, x)
        pd.testing.assert_frame_equal(result, expected_B)

    def test_calc_B_invalid_input(self):
        Z = "invalid input"
        x = pd.DataFrame(
            [10.0, 20.0], index=["Industry A", "Industry B"], columns=["indout"]
        )

        with self.assertRaises(TypeError):
            calc_B(Z, x)  # type: ignore

    def test_calc_G_with_dataframe(self):
        B = pd.DataFrame([[0.0, 0.5], [0.5, 0.0]], index=["Industry A", "Industry B"])
        expected_G = pd.DataFrame(
            [[1.333333333, 0.666666667], [0.6666666667, 1.3333333333]],
            index=["Industry A", "Industry B"],
        )

        result = calc_G(B)
        pd.testing.assert_frame_equal(result, expected_G, check_exact=False)

    def test_calc_G_with_numpy_array(self):
        B = np.array([[0.0, 0.5], [0.5, 0.0]])
        expected_G = np.array(
            [[1.3333333333, 0.666666667], [0.666666667, 1.3333333333]]
        )

        result = calc_G(B)
        np.testing.assert_array_almost_equal(result, expected_G)

    def test_calc_G_single_industry(self):
        B = pd.DataFrame([[0.0]], index=["Industry A"])
        expected_G = pd.DataFrame([[1.0]], index=["Industry A"])

        result = calc_G(B)
        pd.testing.assert_frame_equal(result, expected_G)

    def test_calc_G_invalid_input(self):
        B = "invalid input"

        with self.assertRaises((TypeError, AttributeError)):
            calc_G(B)  # type: ignore

    def test_calc_x_from_G_with_numpy_arrays(self):
        # Test using numpy arrays
        G = np.array([[1.2, 0.3], [0.1, 1.5]])
        va = np.array([[100.0, 200.0]])
        expected_x = np.array([[180.0], [310.0]])

        result = calc_x_from_G(G, va)
        np.testing.assert_array_almost_equal(result, expected_x)

    def test_calc_x_from_G_with_pandas_dataframes(self):
        # Test using pandas DataFrames
        G = pd.DataFrame(
            [[1.2, 0.3], [0.1, 1.5]],
            index=["Sector A", "Sector B"],
            columns=["Sector A", "Sector B"],
        )
        va = pd.DataFrame([[100.0, 200.0]], columns=["Sector A", "Sector B"])
        expected_x = pd.DataFrame(
            [[180.0], [310.0]], index=["Sector A", "Sector B"], columns=["indout"]
        )

        result = calc_x_from_G(G, va)
        pd.testing.assert_frame_equal(result, expected_x)

    def test_calc_x_from_G_with_va_series(self):
        # Test using pandas DataFrames
        G = pd.DataFrame(
            [[1.2, 0.3], [0.1, 1.5]],
            index=["Sector A", "Sector B"],
            columns=["Sector A", "Sector B"],
        )
        va = pd.Series([100.0, 200.0], index=["Sector A", "Sector B"])
        expected_x = pd.DataFrame(
            [[180.0], [310.0]], index=["Sector A", "Sector B"], columns=["indout"]
        )

        result = calc_x_from_G(G, va)
        pd.testing.assert_frame_equal(result, expected_x)

    def test_calc_x_from_G_with_mixed_types(self):
        # Test using a mix of pandas and numpy
        G = pd.DataFrame(
            [[1.2, 0.3], [0.1, 1.5]],
            index=["Sector A", "Sector B"],
            columns=["Sector A", "Sector B"],
        )
        va = np.array([[100.0, 200.0]])
        expected_x = pd.DataFrame(
            [[180.0], [310.0]], index=["Sector A", "Sector B"], columns=["indout"]
        )

        result = calc_x_from_G(G, va)
        pd.testing.assert_frame_equal(result, expected_x)
        self.assertIsInstance(result, pd.DataFrame)

    def test_calc_x_from_G_result_is_ndarray(self):
        # Test if the result is a numpy ndarray when input is an ndarray
        G = np.array([[1.2, 0.3], [0.1, 1.5]])
        va = np.array([[100, 200]])

        result = calc_x_from_G(G, va)
        self.assertIsInstance(result, np.ndarray)



    def test_check_sectors_in_mriot_all_sectors_exist(self):
        # Mock MRIOT system
        mock_mriot = MagicMock()

        # Mock get_sectors method to return a list of sectors
        mock_mriot.get_sectors.return_value = [
            "Agriculture",
            "Manufacturing",
            "Services",
        ]

        # Call the function with sectors that exist in the mocked MRIOT data
        try:
            check_sectors_in_mriot(["Agriculture", "Services"], mock_mriot)
        except ValueError:
            self.fail("check_sectors_in_mriot raised ValueError unexpectedly!")

        # Ensure get_sectors was called once
        mock_mriot.get_sectors.assert_called_once()

    def test_check_sectors_in_mriot_some_sectors_missing(self):
        # Mock MRIOT system
        mock_mriot = MagicMock()

        # Mock get_sectors method to return a list of sectors
        mock_mriot.get_sectors.return_value = ["Agriculture", "Manufacturing"]

        # Check that a ValueError is raised when a missing sector is provided
        with self.assertRaises(ValueError) as context:
            check_sectors_in_mriot(["Agriculture", "Technology"], mock_mriot)

        # Check that the error message contains the missing sector
        self.assertIn(
            "The following sectors are missing in the MRIOT data: {'Technology'}",
            str(context.exception),
        )

        # Ensure get_sectors was called once
        mock_mriot.get_sectors.assert_called_once()

    def test_check_sectors_in_mriot_empty_sectors_list(self):
        # Mock MRIOT system
        mock_mriot = MagicMock()

        # Mock get_sectors method to return an empty list of sectors
        mock_mriot.get_sectors.return_value = []

        # Call the function with an empty sector list, should not raise an error
        try:
            check_sectors_in_mriot([], mock_mriot)
        except ValueError:
            self.fail("check_sectors_in_mriot raised ValueError unexpectedly!")

        # Ensure get_sectors was called once
        mock_mriot.get_sectors.assert_called_once()

    def test_translate_exp_to_regions_wiod(self):
        # Generate dummy exposures and impacts
        mock_exp = Mock(
            gdf=gpd.GeoDataFrame({"region_id": [840, 608], "value": [100, 100]})
        )

        # Call the function to translate exposures to regions
        translated_exp = translate_exp_to_regions(mock_exp, "WIOD16-2010")

        # Check that the region field has been added and converted correctly
        expected_regions = ["USA", "ROW"]  # Based on region_id [840, 608]
        self.assertListEqual(list(translated_exp.gdf["region"]), expected_regions)

        # Check that the value_ratio column is correctly calculated
        usa_value_sum = translated_exp.gdf.loc[
            translated_exp.gdf["region"] == "USA", "value"
        ].sum()
        row_value_sum = translated_exp.gdf.loc[
            translated_exp.gdf["region"] == "ROW", "value"
        ].sum()
        usa_expected_ratios = (
            translated_exp.gdf.loc[translated_exp.gdf["region"] == "USA", "value"]
            / usa_value_sum
        )
        row_expected_ratios = (
            translated_exp.gdf.loc[translated_exp.gdf["region"] == "ROW", "value"]
            / row_value_sum
        )

        # Assert the value_ratio is computed correctly for both regions
        np.testing.assert_array_almost_equal(
            translated_exp.gdf.loc[
                translated_exp.gdf["region"] == "USA", "value_ratio"
            ].values,
            usa_expected_ratios.values,
        )
        np.testing.assert_array_almost_equal(
            translated_exp.gdf.loc[
                translated_exp.gdf["region"] == "ROW", "value_ratio"
            ].values,
            row_expected_ratios.values,
        )

    def test_translate_exp_to_regions_invalid_mriot_type(self):
        # Generate dummy exposures and impacts
        mock_exp = MagicMock()

        # Check that an invalid mriot_type raises an appropriate exception
        with self.assertRaises(ValueError):
            translate_exp_to_regions(mock_exp, "INVALID_MRIOT_TYPE")

    def test_translate_exp_to_regions(self):
        mock_exp = Mock(
            gdf=gpd.GeoDataFrame(
                {"region_id": [840, 840, 608, 608], "value": [80, 20, 120, 80]}
            )
        )
        translated_exp = translate_exp_to_regions(mock_exp, "WIOD16-2010")
        region_groups = translated_exp.gdf.groupby("region")
        for _, group in region_groups:
            total_ratio = group["value_ratio"].sum()
            self.assertAlmostEqual(total_ratio, 1.0, places=5)

    def test_translate_exp_to_sectors_with_all_affected_sectors(self):
        # Mock Exposures object with required structure
        mock_translated_exp = Mock(
            gdf=gpd.GeoDataFrame(
                {"region": ["USA", "USA", "ROW", "ROW"], "value": [80, 20, 120, 80]}
            )
        )
        mock_mriot = Mock(
            x=pd.DataFrame.from_dict(
                {
                    "index": [
                        ("USA", "Sector A"),
                        ("USA", "Sector B"),
                        ("ROW", "Sector A"),
                        ("ROW", "Sector B"),
                    ],
                    "columns": ["indout"],
                    "data": [[600], [400], [700], [300]],
                    "index_names": ["region", "sector"],
                    "column_names": [None],
                },
                orient="tight",
            ),
            monetary_factor = 1.
        )

        mock_mriot.get_sectors.return_value = pd.Index(["Sector A", "Sector B"])
        expected_output = pd.Series(
            [0.6 * 100, 0.4 * 100, 0.7 * 200, 0.3 * 200], index=MRIOT_INDEX
        )  # Define expected output
        result = translate_exp_to_sectors(mock_translated_exp, "all", mock_mriot)
        pd.testing.assert_series_equal(result, expected_output)

    def test_translate_exp_to_sectors_with_list_of_affected_sectors(self):
        mock_translated_exp = Mock(
            gdf=gpd.GeoDataFrame(
                {"region": ["USA", "USA", "ROW", "ROW"], "value": [80, 20, 120, 80]}
            )
        )
        mock_mriot = Mock(
            x=pd.DataFrame.from_dict(
                {
                    "index": [
                        ("USA", "Sector A"),
                        ("USA", "Sector B"),
                        ("ROW", "Sector A"),
                        ("ROW", "Sector B"),
                    ],
                    "columns": ["indout"],
                    "data": [[600], [400], [700], [300]],
                    "index_names": ["region", "sector"],
                    "column_names": [None],
                },
                orient="tight",
            ),
            monetary_factor=1.
        )

        mock_mriot.get_sectors.return_value = pd.Index(["Sector A", "Sector B"])
        expected_output = pd.Series(
            [100.0, 200.0],
            index=pd.MultiIndex.from_tuples(
                [("USA", "Sector A"), ("ROW", "Sector A")], names=["region", "sector"]
            ),
        )  # Define expected output

        affected_sectors = ["Sector A"]

        result = translate_exp_to_sectors(
            mock_translated_exp, affected_sectors, mock_mriot
        )
        pd.testing.assert_series_equal(result, expected_output)

    def test_translate_exp_to_sectors_with_dict_of_affected_sectors(self):
        mock_translated_exp = Mock(
            gdf=gpd.GeoDataFrame(
                {"region": ["USA", "USA", "ROW", "ROW"], "value": [80, 20, 120, 80]}
            )
        )
        mock_mriot = Mock(
            x=pd.DataFrame.from_dict(
                {
                    "index": [
                        ("USA", "Sector A"),
                        ("USA", "Sector B"),
                        ("ROW", "Sector A"),
                        ("ROW", "Sector B"),
                    ],
                    "columns": ["indout"],
                    "data": [[600], [400], [700], [300]],
                    "index_names": ["region", "sector"],
                    "column_names": [None],
                },
                orient="tight",
            ),
            monetary_factor=1.
        )

        mock_mriot.get_sectors.return_value = pd.Index(["Sector A", "Sector B"])
        expected_output = pd.Series(
            [0.6 * 100, 0.4 * 100, 0.6 * 200, 0.4 * 200], index=MRIOT_INDEX
        )  # Define expected output

        affected_sectors = {"Sector A": 0.6, "Sector B": 0.4}
        result = translate_exp_to_sectors(
            mock_translated_exp, affected_sectors, mock_mriot
        )
        pd.testing.assert_series_equal(result, expected_output)

    def test_translate_exp_to_sectors_with_series_of_affected_sectors(self):
        mock_translated_exp = Mock(
            gdf=gpd.GeoDataFrame(
                {"region": ["USA", "USA", "ROW", "ROW"], "value": [80, 20, 120, 80]}
            )
        )
        mock_mriot = Mock(
            x=pd.DataFrame.from_dict(
                {
                    "index": [
                        ("USA", "Sector A"),
                        ("USA", "Sector B"),
                        ("ROW", "Sector A"),
                        ("ROW", "Sector B"),
                    ],
                    "columns": ["indout"],
                    "data": [[600], [400], [700], [300]],
                    "index_names": ["region", "sector"],
                    "column_names": [None],
                },
                orient="tight",
            ),
            monetary_factor=1.
        )

        mock_mriot.get_sectors.return_value = pd.Index(["Sector A", "Sector B"])
        expected_output = pd.Series(
            [0.7 * 100, 0.3 * 100, 0.7 * 200, 0.3 * 200], index=MRIOT_INDEX
        )  # Define expected output

        affected_sectors = pd.Series({"Sector A": 0.7, "Sector B": 0.3})
        result = translate_exp_to_sectors(
            mock_translated_exp, affected_sectors, mock_mriot
        )
        pd.testing.assert_series_equal(result, expected_output)

    def test_translate_exp_to_sectors_invalid_affected_sectors_type(self):
        mock_exp = Mock()
        mock_mriot = Mock()  # Function to create a mock pymrio.IOSystem
        with self.assertRaises(TypeError):
            translate_exp_to_sectors(mock_exp, 123, mock_mriot)

    def test_translate_exp_to_sectors_invalid_distribution_share_sum(self):
        mock_translated_exp = Mock(
            gdf=gpd.GeoDataFrame(
                {"region": ["USA", "USA", "ROW", "ROW"], "value": [80, 20, 120, 80]}
            )
        )
        mock_mriot = Mock(
            x=pd.DataFrame.from_dict(
                {
                    "index": [
                        ("USA", "Sector A"),
                        ("USA", "Sector B"),
                        ("ROW", "Sector A"),
                        ("ROW", "Sector B"),
                    ],
                    "columns": ["indout"],
                    "data": [[600], [400], [700], [300]],
                    "index_names": ["region", "sector"],
                    "column_names": [None],
                },
                orient="tight",
            )
        )

        mock_mriot.get_sectors.return_value = pd.Index(["Sector A", "Sector B"])

        affected_sectors = {"Sector A": 0.5, "Sector B": 0.3}  # Does not sum to 1
        with self.assertRaises(ValueError):
            translate_exp_to_sectors(mock_translated_exp, affected_sectors, mock_mriot)


    def test_translate_imp_to_region_wrongtype(self):
        mock_mriot = MagicMock()
        with self.assertRaises(ValueError):
            translate_imp_to_regions("wrongtype", mriot=mock_mriot)

    def test_translate_imp_to_region_wrong_region(self):
        mock_mriot = MagicMock()
        reg_impact = pd.DataFrame([[100,20,3]],index=[1], columns=["FRA","USA", "WRONG"])
        with self.assertRaises(ValueError):
            translate_imp_to_regions(reg_impact=reg_impact, mriot=mock_mriot)

    def test_translate_imp_to_region(self):
        mock_mriot = Mock(
            monetary_factor=10.,
        )
        mock_mriot.name = "EXIOBASE3-2010"
        reg_impact = pd.DataFrame([[100,20]],index=[1], columns=["FRA","USA"])
        ret = translate_imp_to_regions(reg_impact=reg_impact, mriot=mock_mriot)

        expected = pd.DataFrame(
            [[10.,2.]],
            index=pd.Index([1],name="event_id"),
            columns=pd.Index(["FR","US"],name="region_mriot")
        )
        pd.testing.assert_frame_equal(expected, ret)

    def test_distribute_reg_impact_to_sectors_wrongtype(self):
        reg_impact = pd.DataFrame(
            [[10.,2.]],
            index=pd.Index([1],name="event_id"),
            columns=pd.Index(["FR","US"],name="region_mriot")
        )

        distributor = pd.Series([0.4,0.6], index=pd.Index(["sec1","sec2"], name="sector"))

        with self.assertRaises(ValueError):
            distribute_reg_impact_to_sectors("wrongtype", "wrongtype")

        with self.assertRaises(ValueError):
            distribute_reg_impact_to_sectors(reg_impact, "wrongtype")

        with self.assertRaises(ValueError):
            distribute_reg_impact_to_sectors("wrongtype", distributor)

    def test_distribute_reg_impact_to_sectors_index(self):
        reg_impact = pd.DataFrame(
            [[10.,2.]],
            index=pd.Index([1],name="event_id"),
            columns=pd.Index(["FR","US"],name="region_mriot")
        )

        distributor = pd.Series([4,6], index=pd.Index(["sec1","sec2"], name="sector"))
        ret = distribute_reg_impact_to_sectors(reg_impact, distributor)

        expected = pd.DataFrame( [[4.,6.,0.8,1.2]],
            index=pd.Index([1],name="event_id"),
            columns=pd.MultiIndex.from_tuples([( "FR","sec1" ),( "FR","sec2" ),( "US", "sec1" ), ( "US", "sec2")],names=[ "region","sector" ])
        )
        pd.testing.assert_frame_equal(ret, expected)

    def test_distribute_reg_impact_to_sectors_multiindex_missing_regions(self):
        reg_impact = pd.DataFrame(
            [[10.,2.]],
            index=pd.Index([1],name="event_id"),
            columns=pd.Index(["FR","US"],name="region_mriot")
        )

        distributor = pd.Series([4,6], index=pd.MultiIndex.from_tuples([( "FR","sec1" ),( "FR","sec2" )],names=[ "region","sector" ]))
        with self.assertRaises(ValueError):
            ret = distribute_reg_impact_to_sectors(reg_impact, distributor)


    def test_distribute_reg_impact_to_sectors_multiindex(self):
        reg_impact = pd.DataFrame(
            [[10.,2.]],
            index=pd.Index([1],name="event_id"),
            columns=pd.Index(["FR","US"],name="region_mriot")
        )

        distributor = pd.Series([4,6,5,5], index=pd.MultiIndex.from_tuples([( "FR","sec1" ),( "FR","sec2" ),( "US", "sec1" ), ( "US", "sec2")],names=[ "region","sector" ]))
        ret = distribute_reg_impact_to_sectors(reg_impact, distributor)
        expected = pd.DataFrame( [[4.,6.,1.,1.]],
            index=pd.Index([1],name="event_id"),
            columns=pd.MultiIndex.from_tuples([( "FR","sec1" ),( "FR","sec2" ),( "US", "sec1" ), ( "US", "sec2")],names=[ "region","sector" ])
        )
        pd.testing.assert_frame_equal(ret, expected)
