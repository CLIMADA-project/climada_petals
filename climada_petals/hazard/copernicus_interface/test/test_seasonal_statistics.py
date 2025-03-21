import unittest
import numpy as np
import xarray as xr
import os
import pandas as pd

# Import the module to test
from climada_petals.hazard.copernicus_interface.seasonal_statistics import (
    calculate_heat_indices_metrics,
    calculate_statistics_from_index,
    calculate_monthly_dataset,
    monthly_periods_from_valid_times,
)


"""
Test Suite for seasonal Statistics Functions

This script contains unit tests for verifying the functionality and robustness 
of the seasonal statistics functions used in climate data analysis.

"""


class TestSeasonalStatistics(unittest.TestCase):
    def setUp(self):
        """Set up test datasets for seasonal statistics testing."""

        # -------------------------- #
        # 1. Main Test Dataset
        # -------------------------- #
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")

        # Define fixed temperature values (instead of random values)
        temp_data = np.array([
            [[[20, 22], [23, 25]],  # Day 1
            [[21, 23], [24, 26]],  # Day 2
            [[22, 24], [25, 27]]], # Day 3

            [[[21, 23], [24, 26]],  
            [[22, 24], [25, 27]],  
            [[23, 25], [26, 28]]],

            [[[22, 24], [25, 27]],  
            [[23, 25], [26, 28]],  
            [[24, 26], [27, 29]]]
        ]) + 273.15  # Convert to Kelvin # Shape (3, 3, 2, 2) → (number, step, latitude, longitude)

        # Define other fixed meteorological variables
        dewpoint_data = temp_data - 2  # Dewpoint temperature slightly lower than actual temp
        wind_u = np.full((3, 3, 2, 2), 5)  # Constant wind speed (U component)
        wind_v = np.full((3, 3, 2, 2), 3)  # Constant wind speed (V component) # Wind V Component

        self.steps = pd.timedelta_range(start="0 days", periods=3, freq="D")
        self.lats = [0, 1]
        self.lons = [0, 1]
        self.numbers = [1, 2, 3]

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

        # -------------------------- #
        # 2. Small Dataset (Edge Case)
        # -------------------------- #
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

        # -------------------------- #
        # 3. Monthly Dataset (Continuous Temperature Data)
        # -------------------------- #
        temp_values = np.arange(10, 30, 2).reshape(1, 10, 1, 1)  # °C
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        self.da_index = xr.DataArray(
            temp_values,
            coords={
                "number": [1],  # Single ensemble member
                "step": dates,
                "latitude": [0],
                "longitude": [0],
                "forecast_month": ("step", dates.to_period("M").astype(str)),
            },
            dims=["number", "step", "latitude", "longitude"],
        )

        # -------------------------- #
        # 4. Binary Dataset for "Count" Method (Binary Event Days)
        # -------------------------- #
        binary_values = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1]).reshape(1, 10, 1, 1)  # 1 = event occurred

        self.da_index_count = xr.DataArray(
            binary_values,
            coords={
                "number": [1],
                "step": dates,
                "latitude": [0],
                "longitude": [0],
                "forecast_month": ("step", dates.to_period("M").astype(str)),
            },
            dims=["number", "step", "latitude", "longitude"],
        )

        # -------------------------- #
        # 5. Generate Steps & Time Deltas
        # -------------------------- #
        valid_times = pd.date_range(start="2019-02-01", periods=3)
        steps = pd.timedelta_range(start="0 days", periods=3, freq="30D")

        self.ds_valid_time = xr.Dataset(
            {
                "valid_time": (["step"], valid_times),
                "Tmax": (["step", "latitude", "longitude"], np.random.rand(3, 2, 2)),
            },
            coords={"step": steps, "latitude": [10, 20], "longitude": [30, 40]},
        )

    def tearDown(self):
        """Clean up test files."""
        for file in [self.test_file, self.test_file_small]:
            if os.path.exists(file):
                os.remove(file)



    ### test calculate_heat_indices_metrics ###

    def test_tmean_calculation(self):
        """Test if 'Tmean' index is computed correctly."""
        ds_daily, _, _ = calculate_heat_indices_metrics(self.test_file, "Tmean")
        computed_tmean = ds_daily["Tmean"].mean(dim=["number", "latitude", "longitude"]).values
        expected_tmean = np.array([23.5, 24.5, 25.5])
        np.testing.assert_allclose(computed_tmean, expected_tmean, atol=1e-4)
        self.assertEqual(ds_daily["Tmean"].attrs.get("units"), "degC")

    def test_tmin_calculation(self):
        """Test if 'Tmin' index is computed correctly."""
        ds_daily, _, _ = calculate_heat_indices_metrics(self.test_file, "Tmin")
        computed_tmin = ds_daily["Tmin"].mean(dim=["number", "latitude", "longitude"]).values
        expected_tmin = np.array([18.5, 19.5, 20.5]) 
        np.testing.assert_allclose(computed_tmin, expected_tmin, atol=1e-4)
        self.assertEqual(ds_daily["Tmin"].attrs.get("units"), "degC")

    def test_tmax_calculation(self):
        """Test if 'Tmax' index is computed correctly."""
        ds_daily, _, _ = calculate_heat_indices_metrics(self.test_file, "Tmax")
        computed_tmax = ds_daily["Tmax"].mean(dim=["number", "latitude", "longitude"]).values
        expected_tmax = np.array([28.5, 29.5, 30.5])  
        np.testing.assert_allclose(computed_tmax, expected_tmax, atol=1e-4)
        self.assertEqual(ds_daily["Tmax"].attrs.get("units"), "degC")

    def test_invalid_index(self):
        """Test if the function raises ValueError for unsupported indices."""
        with self.assertRaises(ValueError):
            calculate_heat_indices_metrics(self.test_file, "INVALID_INDEX")

    def test_missing_file(self):
        """Test if the function raises FileNotFoundError for a nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            calculate_heat_indices_metrics("nonexistent.nc", "Tmean")


    ### monthly_periods_from_valid_times test ###
    def test_monthly_periods_from_valid_times(self):
        """Test if the function correctly assing forescat month date."""
        result = monthly_periods_from_valid_times(self.ds_valid_time)
        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(tuple(result.dims), ("step",))
        expected_months = ["2019-02"] * 3
        np.testing.assert_array_equal(result.values, expected_months)

    ### calculate_monthly_dataset test ###
    def test_monthly_mean(self):
        """Test if the function correctly computes the monthly mean."""
        expected_jan_mean = 19  # Expected mean for January (first 10 days)
        ds_monthly = calculate_monthly_dataset(self.da_index, "Tmean", "mean")
        self.assertIsInstance(
            ds_monthly, xr.Dataset
        )  # check if the function returns an xarray.Dataset
        computed_mean = ds_monthly["Tmean"].sel(step="2023-01").values
        np.testing.assert_almost_equal(computed_mean, expected_jan_mean, decimal=5)

    def test_monthly_count(self):
        """Test if the function correctly computes the count of daily values per month for "TR", "TX30", "HW" """
        expected_jan_count = 7
        ds_monthly = calculate_monthly_dataset(self.da_index_count, "Tcount", "count")
        self.assertIsInstance(
            ds_monthly, xr.Dataset
        )  # Check if the function returns an xarray.Dataset
        computed_count = ds_monthly["Tcount"].sel(step="2023-01").values
        np.testing.assert_equal(computed_count, expected_jan_count)


    ### calculate_statistics_from_index test ###
    def test_calculate_statistics_from_index(self):
        """Test calculate_statistics_from_index function with easy numbers"""

        test_index_data = np.array([
            [10., 20, 30], # first location
            [25., 25, 25], # second location
            [10., 10, 40], # third location
            [-10, 0, 40] # fourth location
        ])
        test_index_data = xr.DataArray(
                    np.moveaxis(test_index_data.reshape((2,2,3)), -1, 0),
                    coords={
                        "number": [1, 2, 3],
                        "latitude": [0,1],
                        "longitude": [0,1],
                    },
                    dims=["number", "latitude", "longitude"],
                )
        results = calculate_statistics_from_index(test_index_data).data_vars

        np.testing.assert_almost_equal(results["ensemble_mean"].values, [[20, 25], [20, 10]])
        np.testing.assert_almost_equal(results["ensemble_median"].values, [[20, 25], [10, 0]])
        np.testing.assert_almost_equal(results["ensemble_max"].values, [[30, 25], [40, 40]])
        np.testing.assert_almost_equal(results["ensemble_min"].values, [[10, 25], [10, -10]])
        np.testing.assert_almost_equal(results["ensemble_std"].values, [[8.1649, 0], [14.1421, 21.6024]], decimal=3)
        np.testing.assert_almost_equal(results["ensemble_p5"].values, 0.9 * np.array([[10,25],[10,-10]]) + 0.1 * np.array([[20,25],[10,0]]))
        np.testing.assert_almost_equal(results["ensemble_p25"].values, 0.5 * np.array([[10,25],[10,-10]]) + 0.5 * np.array([[20,25],[10,0]]))
        np.testing.assert_almost_equal(results["ensemble_p50"].values, [[20, 25], [10, 0]])
        np.testing.assert_almost_equal(results["ensemble_p75"].values, 0.5 * np.array([[20,25],[10,0]]) + 0.5 * np.array([[30,25],[40,40]]))
        np.testing.assert_almost_equal(results["ensemble_p95"].values, 0.1 * np.array([[20,25],[10,0]]) + 0.9 * np.array([[30,25],[40,40]]))


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSeasonalStatistics)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
