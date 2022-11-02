import tempfile
import unittest
from unittest.mock import patch, MagicMock
from copy import deepcopy
from pathlib import Path

import numpy as np
import numpy.testing as npt
from numpy.random import default_rng

import xarray as xr

from climada.util.constants import SYSTEM_DIR
from climada_petals.hazard.rf_glofas import (
    download_glofas_discharge,
    return_period,
    interpolate_space,
    flood_depth,
)


def cdf_mock(dis, loc, scale):
    """A mock for the gumbel_r.cdf method. Return zeros if inputs are the same, else ones"""
    if np.array_equal(dis, loc) and np.array_equal(loc, scale):
        return np.zeros_like(dis)

    return np.ones_like(dis)


class TestDantroOpsGloFAS(unittest.TestCase):
    """Test case for 'download_glofas_discharge' operation"""

    def setUp(self):
        """Create temporary directory in case we download data"""
        self.tempdir = tempfile.TemporaryDirectory()
        self.rng = default_rng(1)

    def tearDown(self):
        """Clean up the temporary directory"""
        self.tempdir.cleanup()

    # NOTE: Need to patch the object where it is imported and used
    @patch("climada_petals.hazard.rf_glofas.glofas_request", autospec=True)
    def test_download_glofas_discharge(self, glofas_request_mock):
        """Test case for 'download_glofas_discharge' operation"""
        # Store some dummy data
        xr.DataArray(
            data=[0, 1, 2], dims=["x"], coords=dict(x=[0, 1, 2], time=0)
        ).rename("dis24").to_netcdf(self.tempdir.name + "/file-1.nc")
        xr.DataArray(
            data=[10, 11, 12], dims=["x"], coords=dict(x=[0, 1, 2], time=1)
        ).rename("dis24").to_netcdf(self.tempdir.name + "/file-2.nc")
        glofas_request_mock.return_value = [
            Path(self.tempdir.name, f"file-{num}.nc") for num in range(1, 3)
        ]

        out_dir = Path(self.tempdir.name, "bla")
        ds = download_glofas_discharge(
            "forecast",
            "2022-01-01",
            None,
            42,
            out_dir,
            some_kwarg="foo",
        )

        # Check directory
        self.assertTrue(out_dir.exists())

        # Check call
        glofas_request_mock.assert_called_once_with(
            product="forecast",
            date_from="2022-01-01",
            date_to=None,
            num_proc=42,
            output_dir=out_dir,
            request_kw=dict(some_kwarg="foo"),
        )

        # Check return value
        npt.assert_array_equal(ds["time"].data, [0, 1])
        npt.assert_array_equal(ds["x"].data, [0, 1, 2])
        npt.assert_array_equal(ds.data, [[0, 1, 2], [10, 11, 12]])

    # @patch.object(gumbel_r, "cdf", new=cdf_mock)
    @patch("climada_petals.hazard.rf_glofas.gumbel_r.cdf", new=cdf_mock)
    def test_return_period(self):
        """Test 'return_period' operation"""
        x = np.arange(10)
        # Distort the coordinates to test the reindexing
        x_var = x + self.rng.uniform(low=-1e-7, high=1e-7, size=x.shape)
        x_var_big = x + self.rng.uniform(low=-1e-2, high=1e-2, size=x.shape)
        y = np.arange(10, 20)
        values = np.outer(x, y)

        def create_data_array(x, y, values, name):
            return xr.DataArray(
                data=values, dims=["x", "y"], coords=dict(x=x, y=y)
            ).rename(name)

        # Wrong x coordinates should cause an error
        discharge = create_data_array(x_var_big, y, values, "discharge")
        loc = create_data_array(x, y, values, "loc")
        self.assertFalse(discharge.equals(loc))
        with self.assertRaises(ValueError) as cm:
            return_period(discharge, loc, loc)
        self.assertIn(
            "Coordinates of discharge and GEV fits do not match!", str(cm.exception)
        )

        # Mock a DataArray
        da_mock = MagicMock(spec_set=xr.DataArray)
        da_mock.reindex_like.return_value = discharge  # Return without reindexing
        da_mock.count.return_value = 0  # Mock the count

        # Small deviations should cause an error if reindexing does not work
        discharge = create_data_array(x_var, y, values, "discharge")
        self.assertFalse(discharge.equals(loc))
        with self.assertRaises(ValueError) as cm:
            return_period(da_mock, loc, loc)
        self.assertIn("dimension 'x' are not equal", str(cm.exception))

        # Call the function again, reindexing should work as expected
        result = return_period(discharge, loc, loc)
        self.assertEqual(result.name, "Return Period")

        # NaNs would be a sign that indexing does not work
        self.assertTrue(result.notnull().all())
        npt.assert_array_equal(result.values, np.ones_like(result.values))
        npt.assert_array_equal(result["x"].values, x)
        npt.assert_array_equal(result["y"].values, y)

    def test_interpolate_space(self):
        """Test 'interpolate_space' operation"""
        x = np.arange(10)
        x_diff = x + self.rng.uniform(-0.5, 0.5, size=x.shape)
        xx, yy = np.meshgrid(x, x, indexing="xy")
        values = xx + yy

        da_values = xr.DataArray(
            data=values,
            dims=["longitude", "latitude"],
            coords=dict(longitude=x, latitude=x),
        )
        da_coords = xr.DataArray(
            data=values,
            dims=["longitude", "latitude"],
            coords=dict(longitude=x_diff, latitude=x_diff),
        )

        xx_diff, yy_diff = np.meshgrid(x_diff, x_diff, indexing="xy")
        expected_values = xx_diff + yy_diff

        da_result = interpolate_space(da_values, da_coords)
        # Interpolation causes some noise, so "almost" is enough here
        npt.assert_array_almost_equal(da_result.values, expected_values, 10)
        npt.assert_array_equal(da_result["longitude"], x_diff)
        npt.assert_array_equal(da_result["latitude"], x_diff)

    def test_flood_depth(self):
        """Test 'flood_depth' operation"""
        x = np.arange(10)
        values = np.array([range(100)]).reshape((10, 10)) + self.rng.uniform(
            -0.1, 0.1, size=(10, 10)
        )

        ones = np.ones((10, 10))
        da_flood_maps = xr.DataArray(
            data=[ones, ones * 10, ones * 100],
            dims=["return_period", "longitude", "latitude"],
            coords=dict(return_period=[1, 10, 100], longitude=x, latitude=x),
        )
        da_return_period = xr.DataArray(
            data=values,
            dims=["longitude", "latitude"],
            coords=dict(longitude=x, latitude=x),
        )

        da_result = flood_depth(da_return_period, da_flood_maps)
        self.assertEqual(da_result.name, "Flood Depth")
        # NOTE: Single point precision, so reduce the decimal accuracy
        npt.assert_array_almost_equal(da_result.values, values, 5)


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDantroOpsGloFAS)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
