import unittest
import os
import xarray as xr
import pandas as pd
import numpy as np
import numpy.testing as npt
from pathlib import Path
from climada.hazard import Hazard

from climada_petals.hazard.copernicus_interface.create_seasonal_forecast_hazard import (
    calculate_leadtimes,
    month_name_to_number,
    _convert_to_hazard,
    _process_data,
)


class TestCalculateSeasonalForescastHazard(unittest.TestCase):

    ##Unit tests for the month_name_to_number function##

    def test_valid_inputs(self):
        """
        Check that valid inputs (ints, month names, abbreviations)
        return the correct numeric month values.
        """
        inputs = ["January", "February", 3, "Apr", 12, "dec", "JuN", "MAR"]
        expected_values = [1, 2, 3, 4, 12, 12, 6, 3]
        computed_values = [month_name_to_number(inp) for inp in inputs]
        self.assertEqual(computed_values, expected_values)

    def test_invalid_inputs(self):
        """
        Check that out-of-range or nonsense values raise ValueError.
        We don't print expected vs. computed because we expect an error.
        """
        invalid_inputs = [
            0,  # Below valid integer range
            13,  # Above valid integer range
            999,  # Far above range
            -1,  # Negative integer
            "Hello",  # Not a month name or abbreviation
            "Aprill",  # Misspelled month
            "",  # Empty string
        ]
        for val in invalid_inputs:
            with self.assertRaises(ValueError, msg=f"Expected ValueError for {val}"):
                month_name_to_number(val)

    ##Unit tests for the calculate_leadtimes function##

    def test_calculate_leadtimes_dec_to_feb(self):
        """Test lead times for a forecast initiated in December 2022 with a valid period from January to February 2023."""
        year = 2022
        initiation_month = "December"
        valid_period = ["January", "February"]
        # From Jan 1, 2023, to Feb 28, 2023, in 6-hour steps. Start_offset—31 * 24 = 744, then
        # end_offset_inclusive—2154 + 6 = 2160, so the final 6-hour mark (2154).
        expected_leadtimes = list(range(744, 2154 + 6, 6))
        computed_leadtimes = calculate_leadtimes(year, initiation_month, valid_period)
        self.assertEqual(computed_leadtimes, expected_leadtimes)

    def test_calculate_leadtimes_single_month(self):
        """Test lead times for a single-month forecast (e.g., March to March)."""
        year = 2023
        initiation_month = "March"
        valid_period = ["March", "March"]
        # The function calculates from Mar 1 to Mar 31 => 30 days => 744 hours but exclude 744
        # in 6-hour intervals:
        expected_leadtimes = list(range(0, 744, 6))
        computed_leadtimes = calculate_leadtimes(year, initiation_month, valid_period)
        self.assertEqual(computed_leadtimes, expected_leadtimes)

    def test_calculate_leadtimes_reverse_period_explicit(self):
        """
        Test a reversed valid_period, months (April, March),it raises a ValueError immediately, indicating the input is invalid.
        """
        year = 2023
        initiation_month = "January"
        valid_period = ["April", "March"]  # reversed

        # A ValueError is expected, so we use self.assertRaises
        with self.assertRaises(ValueError):
            calculate_leadtimes(year, initiation_month, valid_period)

    def test_calculate_leadtimes_invalid_month(self):
        """Test invalid month handling."""
        with self.assertRaises(ValueError):
            calculate_leadtimes(2023, "InvalidMonth", ["January", "February"])



    ##Unit tests for the convert_to_hazard function##
    def setUp(self):
        self.test_dir = Path("./test_data_hazard")
        self.test_dir.mkdir(exist_ok=True)

        self.input_file = self.test_dir / "test_fixed_monthly_data.nc"
        self.output_file = self.test_dir / "test_fixed_hazard.hdf5"

        n_members = 5 # members
        step_vals = ["2018-02", "2018-03"]  # 2 forecast steps (months)
        lat_vals = np.array([-20.0, -25.0, -30.0])  # 3 latitude points
        lon_vals = np.array([100.0, 105.0, 110.0])  # 3 longitude points

        # Temperature values for testing
        manual_temps = np.array(
            [
                [
                    [[-5, 0, 5], [10, 15, 20], [25, 30, 35]],  # First month
                    [[-4, 1, 6], [11, 16, 21], [26, 31, 36]],
                ],  # Second month
                [
                    [[-6, -1, 4], [9, 14, 19], [24, 29, 34]],
                    [[-3, 2, 7], [12, 17, 22], [27, 32, 37]],
                ],
                [
                    [[-7, -2, 3], [8, 13, 18], [23, 28, 33]],
                    [[-2, 3, 8], [13, 18, 23], [28, 33, 38]],
                ],
                [
                    [[-8, -3, 2], [7, 12, 17], [22, 27, 32]],
                    [[-1, 4, 9], [14, 19, 24], [29, 34, 39]],
                ],
                [
                    [[-9, -4, 1], [6, 11, 16], [21, 26, 31]],
                    [[0, 5, 10], [15, 20, 25], [30, 35, 40]],
                ],
            ]
        )

        # Convert manual data to proper shape (5 members, 2 steps, 3 lat, 3 lon)
        data = np.array(manual_temps)

        # Create an xarray Dataset
        ds = xr.Dataset(
            data_vars=dict(
                Tmax=(["number", "step", "latitude", "longitude"], data),
            ),
            coords=dict(
                number=("number", np.arange(n_members)),  # 0 to 4
                step=("step", step_vals),  # '2018-02', '2018-03'
                latitude=("latitude", lat_vals),
                longitude=("longitude", lon_vals),
            ),
        )

        ds = xr.Dataset(
            data_vars=dict(
                Tmax=(
                    ["number", "step", "latitude", "longitude"],
                    data,
                ),  
            ),
            coords=dict(
                number=("number", np.arange(n_members)),
                step=("step", step_vals),
                latitude=("latitude", lat_vals),
                longitude=("longitude", lon_vals),
            ),
        )

        # Save dataset to NetCDF
        ds.to_netcdf(self.input_file)

    def test_convert_to_hazard(self):
        index_metric = "Tmax"

        # Call the function
        result_file = _convert_to_hazard(
            output_file_name=self.output_file,
            overwrite=True,
            input_file_name=self.input_file,
            index_metric=index_metric,
        )

        self.assertEqual(
            result_file, self.output_file
        )  # Confirm the function returns the expected path
        self.assertTrue(
            self.output_file.exists(), "Hazard file not created."
        )  # verify hazard file was created

        hazard = Hazard.from_hdf5(str(self.output_file))  # Load hazard from HDF5

        # TEST 1: Check the number of events
        expected_number = 10  # 5 members * 2 steps = 10 expected events
        computed_number = len(hazard.event_name)
        assert (
            computed_number == expected_number
        ), f"Expected {expected_number} events, but got {computed_number}."

        # TEST 2: Check Expected Dates
        expected_dates = np.array([736726, 736754] * 5)  # 5 times [Feb 2018, Mar 2018]
        computed_dates = hazard.date
        assert np.array_equal(
            computed_dates, expected_dates
        ), f"Expected dates {expected_dates}, but got {computed_dates}"

        # TEST 3: Check Flattened Intensity Values
        expected_intensity_values = np.array(
            [
                -5,
                0,
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                -4,
                1,
                6,
                11,
                16,
                21,
                26,
                31,
                36,
                -6,
                -1,
                4,
                9,
                14,
                19,
                24,
                29,
                34,
                -3,
                2,
                7,
                12,
                17,
                22,
                27,
                32,
                37,
                -7,
                -2,
                3,
                8,
                13,
                18,
                23,
                28,
                33,
                -2,
                3,
                8,
                13,
                18,
                23,
                28,
                33,
                38,
                -8,
                -3,
                2,
                7,
                12,
                17,
            ]
        )

        computed_intensity_values = hazard.intensity.toarray().flatten()[
            : len(expected_intensity_values)
        ]
        assert np.allclose(
            computed_intensity_values, expected_intensity_values, atol=1e-3
        ), f"Expected intensity {expected_intensity_values}, but got {computed_intensity_values}"

    def tearDown(self):
        """Clean up test files."""
        if self.input_file.exists():
            os.remove(self.input_file)
        if self.output_file.exists():
            os.remove(self.output_file)
        if self.test_dir.exists():
            os.rmdir(self.test_dir)



    ## Unit tests for the _process_data function ##
    ## To avoid dataset conflicts, a separate setUp_process_data is provided here. 
    ## This ensures test isolation, as each test requires unique data preparation. ##

    def setUp_process_data(self):
        """Create mock GRIB-like input data for testing."""
        self.test_dir = Path("./test_data_process")
        self.test_dir.mkdir(exist_ok=True)

        self.input_file = self.test_dir / "test_sample_grib.grib"
        self.output_file = self.test_dir / "test_output.nc"

        # Define fixed grid dimensions
        n_members = 2  # 2 ensemble members
        step_vals = np.array(
            [
                "2025-01-01T00:00",
                "2025-01-01T06:00",
                "2025-01-01T12:00",
                "2025-01-01T18:00",
                "2025-01-02T00:00",
                "2025-01-02T06:00",
                "2025-01-02T12:00",
                "2025-01-02T18:00",
            ],
            dtype="datetime64[ns]",
        )  # 2 days, 4 steps per day

        lat_vals = np.array([46.5])  # 1 latitude point
        lon_vals = np.array([12.3])  # 1 longitude point

        # Fixed temperature values (in Celsius)
        fixed_temps = np.array(
            [
                [[[-5 + step]] for step in range(len(step_vals))],  # Member 1
                [[[-4 + step * 0.5]] for step in range(len(step_vals))],  # Member 2
            ]
        )

        # Create an xarray Dataset
        ds_p = xr.Dataset(
            data_vars={
                "t2m": (
                    ["number", "step", "latitude", "longitude"],
                    fixed_temps,
                )  # Temperature in Celsius
            },
            coords={
                "number": np.arange(n_members),
                "step": step_vals,
                "latitude": lat_vals,
                "longitude": lon_vals,
            },
        )

        # Save dataset
        ds_p.to_netcdf(self.input_file)

    def test_process_data(self):
        """Test processing of the input file and verify calculations."""
        self.setUp_process_data()
        _process_data(
            output_file_name=self.output_file,
            overwrite=True,
            input_file_name=self.input_file,
            variables=["t2m"],
            format="netcdf",
        )

        # Verify the output file exists
        self.assertTrue(self.output_file.exists(), "Processed file was not created.")

        # Read processed file and check values
        ds_out = xr.open_dataset(self.output_file)

        # Expected statistics
        expected_min = np.array(
            [
                [-5.0, -1.0],  # Member 1: Day 1 min = -5, Day 2 min = -1
                [-4.0, -2.0],  # Member 2: Day 1 min = -4, Day 2 min = -2
            ]
        )

        expected_mean = np.array(
            [
                [-3.5, 0.5],  # Member 1: Mean values for Day 1 & 2
                [-3.25, -1.25],  # Member 2: Mean values for Day 1 & 2
            ]
        )

        expected_max = np.array(
            [
                [-2.0, 2.0],  # Member 1: Max values for Day 1 & 2
                [-2.5, -0.5],  # Member 2: Max values for Day 1 & 2
            ]
        )

        # Validate computed values
        np.testing.assert_array_equal(
            ds_out["t2m_mean"].values[:, :, 0, 0], expected_mean
        )
        np.testing.assert_array_equal(
            ds_out["t2m_max"].values[:, :, 0, 0], expected_max
        )
        np.testing.assert_array_equal(
            ds_out["t2m_min"].values[:, :, 0, 0], expected_min
        )

        ds_out.close()
        # Clean up after the test
        self.tearDown_process_data()

    def tearDown_process_data(self):
        """Clean up test files and directories."""
        if self.input_file.exists():
            self.input_file.unlink()
        if self.output_file.exists():
            self.output_file.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(
        TestCalculateSeasonalForescastHazard
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
