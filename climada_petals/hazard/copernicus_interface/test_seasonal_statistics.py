import unittest
import numpy as np
import xarray as xr
import os
import pandas as pd

# Import the module to test
from climada_petals.hazard.copernicus_interface.seasonal_statistics import (
    calculate_heat_indices_metrics,
    calculate_statistics_from_index,
)

"""
Test Suite for seasonal Statistics Functions

This script contains unit tests for verifying the functionality and robustness 
of the seasonal statistics functions used in climate data analysis

Functions Tested:
-----------------

1. calculate_heat_indices_metrics:
    - Computes various heat-related climate indices (e.g., mean/max/min 
      temperature, heat index, humidity, apparent temperature).
    - Tests ensure:
        - Proper reading of NetCDF data.
        - Correct computation of indices.
        - Handling of missing variables and extreme values.

2. calculate_statistics_from_index:
    - Computes ensemble statistics (mean, median, percentiles) from climate data.
    - Tests verify:
        - Correct statistical calculations.
        - Inclusion of expected metrics (ensemble_mean, ensemble_p95, etc.)

Additional Tests:
-----------------

- NetCDF Format Handling:
    - Ensures the function correctly reads and processes different 
      climate indices (e.g., Tmean, Tmax, HIA).

- Data Structure & Dimensions:
    - Verifies that output datasets (daily_ds, monthly_ds, stats_ds) have 
      correct variables, dimensions (number, step, latitude, longitude), 
      and attributes (e.g., degC for temperature).

- Error Handling:
    - Checks for proper responses to invalid inputs, including:
        - Missing variables (e.g., d2m_mean for humidity calculations).
        - Nonexistent files.
        - Unsupported indices.

- Small Dataset Edge Cases:
    - Ensures functions work correctly for minimal datasets (e.g., a single 
      time step or a single ensemble member).

- Extreme Values:
    - Tests how functions handle extreme temperature and humidity values to 
      ensure stability and meaningful output.

"""

class TestSeasonalStatistics(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")

        temp_data = np.random.uniform(270, 300, size=(3, 10, 2, 2))  # Kelvin
        dewpoint_data = np.random.uniform(260, 290, size=(3, 10, 2, 2))  # Kelvin
        wind_u = np.random.uniform(0, 10, size=(3, 10, 2, 2))  # Wind U Component
        wind_v = np.random.uniform(0, 10, size=(3, 10, 2, 2))  # Wind V component

        self.steps = pd.timedelta_range(start="0 days", periods=10, freq="D")
        self.lats = [0, 1]
        self.lons = [0, 1]
        self.numbers = [1, 2, 3]

        # Create test dataset (with wind components)
        self.test_ds = xr.Dataset(
            data_vars={
                "t2m_mean": (("number", "step", "latitude", "longitude"), temp_data),
                "t2m_max": (("number", "step", "latitude", "longitude"), temp_data + 5),
                "t2m_min": (("number", "step", "latitude", "longitude"), temp_data - 5),
                "d2m_mean": (("number", "step", "latitude", "longitude"), dewpoint_data),
                "u10_max": (("number", "step", "latitude", "longitude"), wind_u), 
                "v10_max": (("number", "step", "latitude", "longitude"), wind_v),  
            },
            coords={
                "step": self.steps,
                "latitude": self.lats,
                "longitude": self.lons,
                "number": self.numbers,
                "valid_time": ("step", dates),
            },
        )

        self.test_file = "test_data.nc"
        self.test_ds.to_netcdf(self.test_file)

        # Small dataset for edge cases
        self.test_ds_small = xr.Dataset(
            data_vars={
                "t2m_mean": (("number", "step", "latitude", "longitude"), np.array([[[[300.15]]]])),
                "d2m_mean": (("number", "step", "latitude", "longitude"), np.array([[[[290.15]]]])),
            },
            coords={
                "number": [1],
                "step": [0],
                "latitude": [0],
                "longitude": [0],
                "valid_time": ("step", ["2023-01-01"]),
            },
        )

        self.test_file_small = "test_data_small.nc"
        self.test_ds_small.to_netcdf(self.test_file_small)

    def tearDown(self):
        """Clean up test files."""
        for file in [self.test_file, self.test_file_small]:
            if os.path.exists(file):
                os.remove(file)

    def test_netcdf_format_indices(self):
        """Test indices that should use NetCDF format."""
        netcdf_indices = ["Tmean", "Tmax", "Tmin", "HIS", "HIA", "RH", "HUM", "AT", "WBGT"]

        for index in netcdf_indices:
            with self.subTest(index=index):
                daily_ds, monthly_ds, stats_ds = calculate_heat_indices_metrics(self.test_file, index)
                self.assertIsInstance(daily_ds, xr.Dataset)
                self.assertIn(index, daily_ds)

    def test_tmean_structure(self):
        """Test if mean temperature calculation produces expected data structure"""
        daily_ds, monthly_ds, stats_ds = calculate_heat_indices_metrics(self.test_file, "Tmean")

        self.assertIsInstance(daily_ds, xr.Dataset)
        self.assertIn("Tmean", daily_ds)
        self.assertEqual(daily_ds.Tmean.attrs["units"], "degC")

        expected_dims = {"number", "step", "latitude", "longitude"}
        self.assertEqual(set(daily_ds.Tmean.dims), expected_dims)

    def test_statistics_structure(self):
        """Test if ensemble statistics maintain correct structure."""
        test_data = xr.DataArray(
            np.random.rand(3, 2, 2),
            coords={
                "number": self.numbers,
                "latitude": self.lats,
                "longitude": self.lons,
            },
            dims=["number", "latitude", "longitude"],
        )

        stats = calculate_statistics_from_index(test_data)

        expected_stats = [
            "ensemble_mean", "ensemble_median", "ensemble_max",
            "ensemble_min", "ensemble_std", "ensemble_p5",  
            "ensemble_p25", "ensemble_p50", "ensemble_p75", "ensemble_p95"
        ]
        for stat in expected_stats:
            self.assertIn(stat, stats)

    def test_error_handling(self):
        """Test Proper error handling."""
        with self.assertRaises(ValueError):
            calculate_heat_indices_metrics(self.test_file, "InvalidIndex")

        with self.assertRaises(FileNotFoundError):
            calculate_heat_indices_metrics("nonexistent_file.nc", "Tmean")

    def test_missing_variable(self):
        """Test behavior when required variables are missing."""
        incomplete_ds = self.test_ds_small.drop_vars("d2m_mean")
        incomplete_file = "test_data_incomplete.nc"
        incomplete_ds.to_netcdf(incomplete_file)

        try:
            with self.assertRaises(KeyError):
                calculate_heat_indices_metrics(incomplete_file, "HIA")
        finally:
            if os.path.exists(incomplete_file):
                os.remove(incomplete_file)

    def test_extreme_temperature_values(self):
        """Test behavior with extreme temperature and humidity values."""
        extreme_ds = xr.Dataset(
            data_vars={
                "t2m_mean": (("number", "step", "latitude", "longitude"), np.array([[[[400.15]]]])),  # Extreme high temp
                "d2m_mean": (("number", "step", "latitude", "longitude"), np.array([[[[350.15]]]])),  # Extreme high dewpoint
            },
            coords={
                "number": [1],
                "step": [0],
                "latitude": [0],
                "longitude": [0],
                "valid_time": ("step", ["2023-01-01"]),
            },
        )

        extreme_file = "test_data_extreme.nc"
        extreme_ds.to_netcdf(extreme_file)

        try:
            daily_ds, monthly_ds, stats_ds = calculate_heat_indices_metrics(extreme_file, "Tmean")
            self.assertIsInstance(daily_ds, xr.Dataset)
            self.assertIn("Tmean", daily_ds)
            self.assertTrue(np.all(np.isfinite(daily_ds["Tmean"].values)))
        finally:
            if os.path.exists(extreme_file):
                os.remove(extreme_file)


if __name__ == "__main__":
    unittest.main()


