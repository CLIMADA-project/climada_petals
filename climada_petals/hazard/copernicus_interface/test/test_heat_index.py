import unittest
import numpy as np
import xarray as xr
import numpy.testing as npt
import pandas as pd
from climada_petals.hazard.copernicus_interface.heat_index import (
    calculate_relative_humidity,
    calculate_heat_index_simplified,
    calculate_heat_index_adjusted,
    calculate_humidex,
    calculate_apparent_temperature,
    calculate_wbgt_simple,
    calculate_tx30,
    calculate_tr,
    calculate_hw,
)


class TestSeasonalForecastCalculations(unittest.TestCase):
    """Unit tests for functions in the seasonal_forecast module."""

    def setUp(self):
        """Data for testing"""
        data = {
            "t2m": [310, 300, 277.2389906],
            "td": [280, 290, 273.1714606],
            "va": [2, 0.02, 0.593496491],
        }

        # Convert to a Pandas DataFrame
        df = pd.DataFrame(data)

        # Convert DataFrame to an Xarray Dataset
        self.ds = xr.Dataset(
            {key: ("time", df[key].values) for key in df.columns},
            coords={"time": pd.date_range("2023-01-01", periods=len(df))},
        )

        # Assign values for specific tests
        self.t2m = self.ds["t2m"].values
        self.td = self.ds["td"].values
        self.va = self.ds["va"].values

    def test_calculate_relative_humidity(self):
        result = calculate_relative_humidity(self.t2m, self.td)
        self.expected_rh = np.array(
            [15.93, 54.31, 74.75]
        )  # Expected values are obtained from the thermofeel.py implementation
        self.assertEqual(result.shape, self.expected_rh.shape)
        npt.assert_allclose(result, self.expected_rh, atol=1.0, rtol=0.01)

    def test_calculate_heat_index_simplified(self):
        result = calculate_heat_index_simplified(self.t2m, self.td)
        result_k = result + 273.15
        expected_his = np.array(
            [307.73, 300.68, 277.23]
        )  # Expected values are obtained from the thermofeel.py implementation
        self.assertEqual(result_k.shape, self.t2m.shape)
        npt.assert_allclose(result_k, expected_his, atol=1.0, rtol=0.01)

    def test_calculate_heat_index_adjusted(self):
        result = calculate_heat_index_adjusted(self.t2m, self.td)
        result_k = result + 273.15
        expected_hia = np.array(
            [307.73, 300.68, 275.65]
        )  # Expected values are obtained from the thermofeel.py implementation
        self.assertEqual(result_k.shape, expected_hia.shape)
        npt.assert_almost_equal(result_k, expected_hia, decimal=2)

    def test_calculate_wbgt_simple(self):
        result = calculate_wbgt_simple(self.t2m, self.td)
        result_k = result + 273.15
        expected_wbgt = np.array(
            [297.04, 297.12, 276.34]
        )  # Expected values are obtained from the thermofeel.py implementation
        self.assertEqual(result_k.shape, expected_wbgt.shape)
        self.assertTrue(np.all(result_k > 0))
        npt.assert_allclose(result_k, expected_wbgt, atol=3.0, rtol=0.01)

    def test_calculate_humidex(self):
        result = calculate_humidex(self.t2m, self.td)
        result_k = result + 273.15
        expected_humidex = np.array(
            [309.95, 305.18, 275.08]
        )  # Expected values are obtained from the thermofeel.py implementation
        self.assertEqual(result_k.shape, expected_humidex.shape)
        npt.assert_allclose(result_k, expected_humidex, atol=3.0, rtol=0.01)

    def test_calculate_apparent_temperature(self):
        wind_speed = self.va
        u10 = wind_speed
        v10 = np.zeros_like(u10)
        result = calculate_apparent_temperature(self.t2m, u10, v10, self.td)
        result_k = result + 273.15  # Convert Celsius to Kelvin
        expected_at = np.array(
            [307.85, 302.30, 274.84]
        )  # Expected values are obtained from the thermofeel.py implementation
        self.assertEqual(result_k.shape, expected_at.shape)
        npt.assert_almost_equal(result_k, expected_at, decimal=2)

    def test_calculate_hw(self):
        temperatures = np.array([26, 27, 28, 26, 28, 29, 30])
        threshold = 27
        min_duration = 2
        max_gap = 0
        result = calculate_hw(temperatures, threshold, min_duration, max_gap)
        expected = np.array([0, 1, 1, 0, 1, 1, 1])
        print("\nExpected HW  :", expected)
        print("Computed HW  :", result)
        npt.assert_array_equal(result, expected)

    def test_calculate_tr(self):
        temperature_data = xr.DataArray(
            data=[18, 21, 19, 22],
            dims=["time"],
            coords={"time": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]},
        )
        result = calculate_tr(temperature_data, tr_threshold=20)
        expected = xr.DataArray(
            data=[False, True, False, True],
            dims=["time"],
            coords={"time": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]},
        )
        xr.testing.assert_equal(result, expected)

    def test_calculate_tx30(self):
        temperature_data = xr.DataArray(
            data=[29, 30, 31],
            dims=["time"],
            coords={"time": ["2023-01-01", "2023-01-02", "2023-01-03"]},
        )
        result = calculate_tx30(temperature_data, threshold=30)
        expected = xr.DataArray(
            data=[False, False, True],
            dims=["time"],
            coords={"time": ["2023-01-01", "2023-01-02", "2023-01-03"]},
        )
        xr.testing.assert_equal(result, expected)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(
        TestSeasonalForecastCalculations
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
