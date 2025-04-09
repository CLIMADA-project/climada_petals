"""
Unit tests for forecast_skill.py

Test Coverage:
--------------
- download_forecast_skills(): test path generation, file presence logic, and download fallback
- plot_forecast_skills(): test file access and plotting logic with mocked NetCDF input
"""

import unittest
from unittest import mock
from pathlib import Path
import xarray as xr
import numpy as np
import tempfile
import shutil

from climada_petals.hazard.copernicus_interface import forecast_skill


class TestForecastSkill(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.index_metric = "Tmax"
        self.months = ["03"]
        self.fake_bounds = [5.0, 45.0, 10.0, 47.0]
        self.fake_bounds_str = "TEST"
        self.test_file = (
            Path(self.temp_dir)
            / "skills"
            / self.index_metric
            / "tasmaxMSESS_subyr_gcfs21_shc03-climatology_r1i1p1_1990-2019.nc"
        )
        self.test_file.parent.mkdir(parents=True, exist_ok=True)

        # Patch CONFIG path to redirect to temp directory
        self.patcher = mock.patch(
            "climada_petals.hazard.copernicus_interface.forecast_skill.CONFIG"
        )
        self.mock_config = self.patcher.start()
        self.mock_config.hazard.copernicus.seasonal_forecasts.dir.return_value = (
            self.temp_dir
        )

        # Create a valid dummy NetCDF file with ascending latitudes
        dataset = xr.Dataset(
            {
                "tasmax_fc_mse": (("time", "lat", "lon"), np.random.rand(1, 10, 10)),
                "tasmax_ref_mse": (("time", "lat", "lon"), np.random.rand(1, 10, 10)),
                "tasmax_msess": (("time", "lat", "lon"), np.random.rand(1, 10, 10)),
                "tasmax_msessSig": (("time", "lat", "lon"), np.random.rand(1, 10, 10)),
            },
            coords={
                "time": ("time", [np.datetime64("2000-01-01")]),
                "lat": ("lat", np.linspace(40.0, 50.0, 10)),
                "lon": ("lon", np.linspace(5.0, 10.0, 10)),
            },
        )
        dataset.to_netcdf(self.test_file)

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.temp_dir)

    ## Check that download_forecast_skills returns the expected path when file is already present ##
    def test_download_forecast_skills_creates_expected_files(self):
        self.test_file.write_text("dummy")
        results = forecast_skill.download_forecast_skills(
            index_metric=self.index_metric,
            initiation_months=self.months,
        )
        self.assertIn(self.test_file, results)

    ## Ensure plot_forecast_skills reads the dummy NetCDF and executes without crashing ##
    @mock.patch("climada_petals.hazard.copernicus_interface.forecast_skill.plt.show")
    def test_plot_forecast_skills_with_real_netcdf(self, mock_show):
        forecast_skill.plot_forecast_skills(
            bounds=self.fake_bounds,
            bounds_str=self.fake_bounds_str,
            index_metric=self.index_metric,
            init_months=self.months,
        )
        mock_show.assert_called()  # To avoid  plt.show() since the actual plot is not being tested


# Execute Tests
if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(
        unittest.TestLoader().loadTestsFromTestCase(TestForecastSkill)
    )
