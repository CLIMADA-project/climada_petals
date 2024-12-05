import unittest
import numpy as np
import xarray as xr
import numpy.testing as npt
from climada_petals.hazard.copernicus_interface.heat_index import (
    calculate_relative_humidity_percent,
    calculate_heat_index_simplified,
    calculate_heat_index_adjusted,
    calculate_heat_index,
    calculate_relative_humidity,
    calculate_humidex,
    calculate_wind_speed,
    calculate_apparent_temperature,
    calculate_wbgt_simple,
    calculate_tx30,
    calculate_tr,
    calculate_hw,
)


class TestSeasonalForecastCalculations(unittest.TestCase):
    """Unit tests for functions in the seasonal_forecast module."""

    def test_calculate_relative_humidity_percent(self):
        """Test calculation of relative humidity percentage."""
        t2k = np.array([300.15, 303.15])  # Kelvin
        tdk = np.array([295.15, 298.15])  # Kelvin

        expected = np.array([74.15727719, 74.65854214])  # Relative humidity without clipping
        expected_clipped = np.clip(expected, 0, 100)

        result = calculate_relative_humidity_percent(t2k, tdk)

        # Validate results
        npt.assert_almost_equal(result, expected_clipped, decimal=2)


    def test_calculate_heat_index_simplified(self):
        """Test simplified heat index calculation."""
        t2k = np.array([308.15, 310.15])  # Kelvin
        tdk = np.array([303.15, 305.15])  # Kelvin
        result = calculate_heat_index_simplified(t2k, tdk)
        self.assertEqual(result.shape, t2k.shape)  # please check shape consistency

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
        """Test Wet Bulb Globe Temperature (WBGT) calculation."""
        t2k = np.array([300.15, 303.15])  # Kelvin
        tdk = np.array([295.15, 298.15])  # Kelvin
        result = calculate_wbgt_simple(t2k, tdk)
        self.assertEqual(result.shape, t2k.shape)
        self.assertTrue(np.all(result > 0))  # WBGT should be positive

    
    def test_calculate_relative_humidity(self):
        """Test calculation of relative humidity."""
        # Test inputs
        t2_k = np.array([300.15, 303.15])  # Kelvin 
        td_k = np.array([295.15, 298.15])  # Kelvin 

        expected_rh = np.array([74.15727719, 74.65854214])  # Approximate relative humidity in %

        # Call the function
        result = calculate_relative_humidity(t2_k, td_k)

        # Validate results
        npt.assert_almost_equal(result, expected_rh, decimal=2)

    def test_calculate_heat_index_adjusted(self):
        """Test adjusted heat index calculation."""
        t2k = np.array([308.15, 310.15])  # Kelvin
        tdk = np.array([303.15, 305.15])  # Kelvin
        result = calculate_heat_index_adjusted(t2k, tdk)
        self.assertEqual(result.shape, t2k.shape)  

    def test_calculate_humidex(self):
        """Test Humidex calculation."""
        t2k = np.array([303.15, 305.15])  # Kelvin
        tdk = np.array([298.15, 300.15])  # Kelvin
        result = calculate_humidex(t2k, tdk)
        self.assertEqual(result.shape, t2k.shape)

    def test_calculate_wind_speed(self):
        """Test wind speed calculation."""
        u10 = np.array([3, 4])  # m/s
        v10 = np.array([4, 3])  # m/s
        expected = np.sqrt(u10**2 + v10**2)
        result = calculate_wind_speed(u10, v10)
        npt.assert_array_equal(result, expected)

    def test_calculate_apparent_temperature(self):
        """Test apparent temperature calculation."""
        t2k = np.array([305.15, 307.15])  # Kelvin
        u10 = np.array([2, 3])  # m/s
        v10 = np.array([2, 3])  # m/s
        d2m_k = np.array([300.15, 302.15])  # Kelvin
        result = calculate_apparent_temperature(t2k, u10, v10, d2m_k)
        self.assertEqual(result.shape, t2k.shape)

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
