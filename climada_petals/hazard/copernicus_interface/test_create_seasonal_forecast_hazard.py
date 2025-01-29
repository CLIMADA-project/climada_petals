import unittest
import os
import xarray as xr
import pandas as pd
import numpy as np
import numpy.testing as npt
from pathlib import Path
from climada.hazard import Hazard

from climada_petals.hazard.copernicus_interface.create_seasonal_forecast_hazard import (calculate_leadtimes)

class TestCalculateLeadtimes(unittest.TestCase):












    
    """Unit tests for the calculate_leadtimes function."""

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





# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(
        TestSeasonalForecastProcessing
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)