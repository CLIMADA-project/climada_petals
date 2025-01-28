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
        # From Jan 1, 2023, to Feb 28, 2023, in 6-hour steps
        expected_leadtimes = list(range(31 * 24, (31 + 28) * 24, 6))
        computed_leadtimes = calculate_leadtimes(year, initiation_month, valid_period)
        self.assertEqual(computed_leadtimes, expected_leadtimes)
        print(f"Expected elements: {len(expected_leadtimes)}")
        print(f"Computed elements: {len(computed_leadtimes)}")
        print(f"Expected range: {expected_leadtimes[0]} to {expected_leadtimes[-1]}")
        print(f"Computed range: {computed_leadtimes[0]} to {computed_leadtimes[-1]}")



# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(
        TestSeasonalForecastProcessing
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)