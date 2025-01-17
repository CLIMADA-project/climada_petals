import unittest
import numpy as np
import xarray as xr
import numpy.testing as npt
import pandas as pd
from climada_petals.hazard.copernicus_interface.heat_index import (
    calculate_relative_humidity,
    calculate_heat_index_simplified,
    calculate_heat_index_adjusted,
    calculate_heat_index,
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
        """Load test data from CSV files before each test."""
        self.testcases_file = "testcases.csv"
        self.at_file = "at.csv"
        self.humidex_file = "humidex.csv"
        self.wbgt_file = "wbgt.csv"  
        self.rh_file = "rh.csv"  

        # Load test input data
        t = np.genfromtxt(self.testcases_file, delimiter=",", names=True)

        self.t2m = t["t2m"]
        self.va = t["va"]
        self.td = t["td"]

        # Load expected results from CSV files
        self.expected_at = np.loadtxt(self.at_file)  # Expected apparent temperature
        self.expected_humidex = np.loadtxt(self.humidex_file)  # Expected humidex
        self.expected_wbgt = np.loadtxt(self.wbgt_file)  # Expected WBGT
        self.expected_rh = np.loadtxt(self.rh_file) 

    def test_calculate_relative_humidity(self):
        """Test calculation of relative humidity percentage."""
        result = calculate_relative_humidity(self.t2m, self.td)
        # Ensure expected data shape matches result shape
        self.assertEqual(result.shape, self.expected_rh.shape)
        npt.assert_allclose(result, self.expected_rh, atol=1.0, rtol=0.01)

    def test_calculate_heat_index_simplified(self):
        """Test simplified heat index calculation."""
        t2k = np.array([308.15, 310.15])  # Kelvin
        tdk = np.array([303.15, 305.15])  # Kelvin
        result = calculate_heat_index_simplified(t2k, tdk)
        self.assertEqual(result.shape, t2k.shape)  # please check shape consistency

    def test_calculate_heat_index_adjusted(self):
        """Test adjusted heat index calculation."""
        t2k = np.array([308.15, 310.15])  # Kelvin
        tdk = np.array([303.15, 305.15])  # Kelvin
        result = calculate_heat_index_adjusted(t2k, tdk)
        self.assertEqual(result.shape, t2k.shape)  

    def test_calculate_heat_index(self):
        """Test calculation of heat index (simplified and adjusted)."""
        da_t2k = xr.DataArray(
            data=np.array([[308.15, 310.15], [300.15, 303.15]]),
            dims=["latitude", "longitude"],
            coords={"latitude": [10, 20], "longitude": [30, 40]},
        )
        da_tdk = xr.DataArray(
            data=np.array([[303.15, 305.15], [295.15, 298.15]]),
            dims=["latitude", "longitude"],
            coords={"latitude": [10, 20], "longitude": [30, 40]},
        )

        # Simplified heat index
        da_his = calculate_heat_index(da_t2k, da_tdk, "HIS")
        self.assertIsInstance(da_his, xr.DataArray)
        self.assertEqual(da_his.attrs["description"], "heat_index_simplified")
        self.assertEqual(da_his.attrs["units"], "degC")

        # Adjusted heat index
        da_hia = calculate_heat_index(da_t2k, da_tdk, "HIA")
        self.assertIsInstance(da_hia, xr.DataArray)
        self.assertEqual(da_hia.attrs["description"], "heat_index_adjusted")
        self.assertEqual(da_hia.attrs["units"], "degC")

    def test_calculate_wbgt_simple(self):
        """Test Wet Bulb Globe Temperature (WBGT) calculation using real test case data."""
        result = calculate_wbgt_simple(self.t2m , self.td)
        result_k = result + 273.15
        self.assertEqual(result_k.shape, self.expected_wbgt.shape)
        self.assertTrue(np.all(result_k > 0))
        npt.assert_allclose(result_k, self.expected_wbgt, atol=3.0, rtol=0.01)

    def test_calculate_humidex(self):
        """Test Humidex calculation."""
        result = calculate_humidex(self.t2m, self.td)
        result_k = result + 273.15
        self.assertEqual(result_k.shape, self.expected_humidex.shape)
        npt.assert_allclose(result_k, self.expected_humidex, atol=3.0, rtol=0.01)

    def test_calculate_apparent_temperature(self):
        """Test apparent temperature calculation."""
        wind_speed = self.va  
        u10 = wind_speed  
        v10 = np.zeros_like(u10)  
        result = calculate_apparent_temperature(self.t2m, u10, v10, self.td )
        result_k = result + 273.15  # Convert Celsius to Kelvin
        self.assertEqual(result_k.shape, self.expected_at.shape)
        npt.assert_almost_equal(result_k, self.expected_at, decimal=2)


    def test_calculate_hw(self):
        temperatures = np.array([26, 27, 28, 26, 28, 29, 30])
        threshold = 27
        min_duration = 2
        max_gap = 1
        result = calculate_hw(temperatures, threshold, min_duration, max_gap)
        expected = np.array([0, 1, 1, 1, 1, 1, 1])  
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
    unittest.main()
