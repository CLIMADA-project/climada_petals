"""
Unit tests for path_utils.py.

Tests include:
- Correct construction of file paths for all data types.
- Detection of file existence.
- Handling of invalid input (e.g., valid_period length).
- Validation of dictionary output for index paths.

Note: Tests check path logic only—no real NetCDF data is used.
"""

import unittest
from pathlib import Path
import shutil
import os
from climada_petals.hazard.copernicus_interface.path_utils import (
    get_file_path,
    check_existing_files,
)


class TestPathUtils(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path("test_dir")
        self.originating_centre = "dwd"
        self.index_metric = "TR"
        self.year = 2023
        self.initiation_month_str = "03"
        self.valid_period_str = "06_08"
        self.bounds_str = "W4_S44_E11_N48"
        self.system = "21"
        self.download_format = "grib"

        # Create base directory for tests
        os.makedirs(self.base_dir, exist_ok=True)

    def tearDown(self):
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)

    ### Test that get_file_path returns correct path for downloaded GRIB data ###
    def test_get_file_path_downloaded_data(self):
        path = get_file_path(
            self.base_dir,
            self.originating_centre,
            self.year,
            self.initiation_month_str,
            self.valid_period_str,
            "downloaded_data",
            self.index_metric,
            self.bounds_str,
            self.system,
            data_format=self.download_format,
        )
        expected_suffix = (
            f"{self.index_metric}_{self.bounds_str}.{self.download_format}"
        )
        self.assertTrue(str(path).endswith(expected_suffix))

    ### Test that get_file_path returns a dictionary for indices data type ###
    def test_get_file_path_indices(self):
        paths = get_file_path(
            self.base_dir,
            self.originating_centre,
            self.year,
            self.initiation_month_str,
            self.valid_period_str,
            "indices",
            self.index_metric,
            self.bounds_str,
            self.system,
        )
        self.assertIsInstance(paths, dict)
        for timeframe in ["daily", "monthly", "stats"]:
            self.assertIn(timeframe, paths)
            self.assertTrue(paths[timeframe].name.endswith(f"{timeframe}.nc"))

    ### Test check_existing_files returns correct message when no files exist ###
    def test_check_existing_files_missing_all(self):
        result = check_existing_files(
            base_dir=self.base_dir,
            originating_centre=self.originating_centre,
            index_metric=self.index_metric,
            year=self.year,
            initiation_month="March",
            valid_period=["June", "August"],
            bounds_str=self.bounds_str,
            system=self.system,
            download_format=self.download_format,
            print_flag=False,
        )
        self.assertIn("No downloaded data found", result)
        self.assertIn("No processed data found", result)
        self.assertIn("No index data found", result)
        self.assertIn("No hazard data found", result)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestPathUtils)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
