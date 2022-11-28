import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import numpy as np
import numpy.testing as npt
from numpy.random import default_rng

import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon

from climada_petals.hazard.rf_glofas import (
    download_glofas_discharge,
    return_period,
    interpolate_space,
    flood_depth,
    reindex,
    sel_lon_lat_slice,
    max_from_isel,
)


def cdf_mock(dis, loc, scale):
    """A mock for the gumbel_r.cdf method. Return zeros if inputs are the same, else ones"""
    if np.array_equal(dis, loc) and np.array_equal(loc, scale):
        return np.zeros_like(dis)

    return np.ones_like(dis)


def create_data_array(x, y, values, name):
    return xr.DataArray(
        data=values,
        dims=["longitude", "latitude"],
        coords=dict(longitude=x, latitude=y),
    ).rename(name)


class TestDantroOpsGloFASDownload(unittest.TestCase):
    """Test case for 'download_glofas_discharge' operation"""

    def setUp(self):
        """Create temporary directory in case we download data"""
        self.tempdir = tempfile.TemporaryDirectory()

        # Store some dummy data
        xr.DataArray(
            data=[0, 1, 2], dims=["x"], coords=dict(x=[0, 1, 2], time=0)
        ).rename("dis24").to_netcdf(self.tempdir.name + "/file-1.nc")
        xr.DataArray(
            data=[10, 11, 12], dims=["x"], coords=dict(x=[0, 1, 2], time=1)
        ).rename("dis24").to_netcdf(self.tempdir.name + "/file-2.nc")

        # Mock the 'glofas_request' function
        # NOTE: Need to patch the object where it is imported and used
        self.patch_glofas_request = patch(
            "climada_petals.hazard.rf_glofas.glofas_request", autospec=True
        )
        self.glofas_request_mock = self.patch_glofas_request.start()
        self.glofas_request_mock.return_value = [
            Path(self.tempdir.name, f"file-{num}.nc") for num in range(1, 3)
        ]

    def tearDown(self):
        """Clean up the temporary directory"""
        self.tempdir.cleanup()
        self.patch_glofas_request.stop()

    def test_basic(self):
        """Basic case for 'download_glofas_discharge' operation"""
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
        self.glofas_request_mock.assert_called_once_with(
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

    @patch("climada_petals.hazard.rf_glofas.get_country_geometries", autospec=True)
    def test_countries_area(self, get_country_geometries_mock):
        """Check behavior of 'countries' and 'area' kwargs"""
        get_country_geometries_mock.return_value = gpd.GeoDataFrame(
            dict(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
        )

        # Default: countries=None
        download_glofas_discharge(
            "forecast",
            "2022-01-01",
            None,
            42,
            self.tempdir.name,
        )
        get_country_geometries_mock.assert_not_called()

        # Assert that 'area' was not passed
        self.assertEqual(self.glofas_request_mock.call_args.kwargs["request_kw"], {})

        # Set only countries
        download_glofas_discharge(
            "forecast", "2022-01-01", None, 42, self.tempdir.name, "Switzerland"
        )
        get_country_geometries_mock.assert_called_once_with("CHE", extent=None)
        npt.assert_array_equal(
            self.glofas_request_mock.call_args.kwargs["request_kw"]["area"],
            [1.0, 0.0, 0.0, 1.0],
        )

        # Set both countries and area
        download_glofas_discharge(
            "forecast",
            "2022-01-01",
            None,
            42,
            self.tempdir.name,
            countries=["Switzerland", "DEU"],
            area=[0, 1, 2, 3],
        )

        # Check country code translation and area order
        get_country_geometries_mock.assert_called_with(
            ["CHE", "DEU"], extent=[1, 3, 2, 0]
        )
        self.assertEqual(self.glofas_request_mock.call_count, 3)
        npt.assert_array_equal(
            self.glofas_request_mock.call_args.kwargs["request_kw"]["area"],
            [1.0, 0.0, 0.0, 1.0],
        )


class TestDantroOpsGloFAS(unittest.TestCase):
    """Test case for other dantro operations"""

    def setUp(self):
        """Set up a random number generator"""
        self.rng = default_rng(1)

    def test_max_from_isel(self):
        """Test the 'max_from_isel' operation"""
        # NOTE: Use timedelta to check support for this data type
        #       (we typically compute a maximum over multiple time steps)
        da = xr.DataArray(
            data=[[0], [1], [2], [3]],
            coords=dict(step=[np.timedelta64(i, "D") for i in range(4)], x=[0]),
        )

        # Test how it's regularly called
        res = max_from_isel(da, "step", [slice(0, 2), [0, 3, 2]])
        npt.assert_array_equal(res["x"].values, [0])
        # npt.assert_array_equal(
        #     res["step"].values, [np.timedelta64(1, "D"), np.timedelta64(3, "D")]
        # )
        npt.assert_array_equal(res["select"].values, list(range(2)))
        # NOTE: slicing with .isel is NOT inclusive (as opposed to .sel)!
        npt.assert_array_equal(res.values, [[1], [3]])

        # Check errors
        with self.assertRaises(TypeError) as cm:
            max_from_isel(da, "step", [1])
        self.assertIn(
            "This function only works with iterables or slices as selection",
            str(cm.exception),
        )

    # @patch.object(gumbel_r, "cdf", new=cdf_mock)
    @patch("climada_petals.hazard.rf_glofas.gumbel_r.cdf", new=cdf_mock)
    def test_return_period(self):
        """Test 'return_period' operation"""
        x = np.arange(10)
        # Distort the coordinates to test the reindexing
        x_var = x + self.rng.uniform(low=-1e-7, high=1e-7, size=x.shape)
        x_var_big = x + self.rng.uniform(low=-1e-2, high=1e-2, size=x.shape)
        y = np.arange(20, 10, -1)
        values = np.outer(x, y)

        def return_arg(target, *args, **kwargs):
            """A dummy that returns the first argument"""
            return target

        # Wrong x coordinates should cause an error
        discharge = create_data_array(x_var_big, y, values, "discharge")
        loc = create_data_array(x, y, values, "loc")
        self.assertFalse(discharge.equals(loc))
        with self.assertRaises(ValueError) as cm:
            return_period(discharge, loc, loc)
        self.assertIn(
            "Reindexing 'loc' to 'discharge' exceeds tolerance!", str(cm.exception)
        )

        # Small deviations should cause an error if reindexing does not work
        discharge = create_data_array(x_var, y, values, "discharge")
        self.assertFalse(discharge.equals(loc))

        # Mock a DataArray
        da_mock = MagicMock(spec_set=xr.DataArray)
        da_mock.reindex_like.return_value = loc  # Return without reindexing
        da_mock.count.return_value = 0  # Mock the count

        # Patch the reindexing
        with patch("climada_petals.hazard.rf_glofas.reindex", new=return_arg):
            with self.assertRaises(ValueError) as cm:
                return_period(discharge, da_mock, loc)
            self.assertIn("cannot align objects", str(cm.exception))
            self.assertIn("longitude", str(cm.exception))

        # Call the function again, slicing and reindexing should work as expected
        x_loc = np.arange(11)
        y_loc = np.arange(25, 5, -1)
        values_loc = np.outer(x_loc, y_loc)
        loc = create_data_array(x_loc, y_loc, values_loc, "loc")
        result = return_period(discharge, loc, loc)
        self.assertEqual(result.name, "Return Period")

        # "-1" would be a sign that indexing does not work
        self.assertFalse((result == -1).any())

        # NOTE: This checks if slicing works through 'cdf_mock'
        # NOTE: Needs 'allclose' because of float32 dtype
        npt.assert_allclose(result.values, np.ones_like(result.values))
        npt.assert_allclose(result["longitude"].values, x, atol=1e-8)
        npt.assert_allclose(result["latitude"].values, y, atol=1e-8)

    def test_interpolate_space(self):
        """Test 'interpolate_space' operation"""
        x = np.arange(4)
        y = np.flip(x)
        x_diff = x * 0.9
        y_diff = y * 0.8
        xx, yy = np.meshgrid(x, y, indexing="xy")
        values = xx + yy

        da_values = xr.DataArray(
            data=values,
            dims=["latitude", "longitude"],
            coords=dict(longitude=x, latitude=y),
        )
        da_coords = xr.DataArray(
            data=values,
            dims=["latitude", "longitude"],
            coords=dict(longitude=x_diff, latitude=y_diff),
        )

        xx_diff, yy_diff = np.meshgrid(x_diff, y_diff, indexing="xy")
        expected_values = xx_diff + yy_diff

        da_result = interpolate_space(da_values, da_coords)
        npt.assert_array_equal(da_result["longitude"], x_diff)
        npt.assert_array_equal(da_result["latitude"], y_diff)
        # Interpolation causes some noise, so "allclost" is enough here
        npt.assert_allclose(da_result.values, expected_values, rtol=1e-10)

    def test_flood_depth(self):
        """Test 'flood_depth' operation"""
        # Create dummy datasets
        ones = np.ones((12, 13))
        da_flood_maps = xr.DataArray(
            data=[ones, ones * 10, ones * 100],
            dims=["return_period", "longitude", "latitude"],
            coords=dict(
                return_period=[1, 10, 100],
                longitude=np.arange(12),
                latitude=np.arange(13),
            ),
        )

        x = np.arange(10)
        core_dim = np.arange(2)
        values = np.array([range(200)]).reshape((10, 10, 2)) + self.rng.uniform(
            -0.1, 0.1, size=(10, 10, 2)
        )
        da_return_period = xr.DataArray(
            data=values,
            dims=["longitude", "latitude", "core_dim"],
            coords=dict(longitude=x, latitude=x, core_dim=core_dim),
        )

        da_result = flood_depth(da_return_period, da_flood_maps)
        self.assertEqual(da_result.name, "Flood Depth")
        # NOTE: Single point precision, so reduce the decimal accuracy
        npt.assert_allclose(da_result.values, values)

        # Check NaN shortcut
        da_flood_maps = xr.DataArray(
            data=[np.full_like(ones, np.nan)] * 3,
            dims=["return_period", "longitude", "latitude"],
            coords=dict(
                return_period=[1, 10, 100],
                longitude=np.arange(12),
                latitude=np.arange(13),
            ),
        )
        with patch("climada_petals.hazard.rf_glofas.np.full_like") as full_like_mock:
            full_like_mock.side_effect = lambda x, _ : np.full(x.shape, 0.0)
            flood_depth(da_return_period, da_flood_maps)
            # Should have been called 100 times, one time for each lon/lat coordinate
            self.assertEqual(full_like_mock.call_count, 100)

        # Check NaN sanitizer
        da_flood_maps = xr.DataArray(
            data=[np.full_like(ones, np.nan), ones, ones * 10],
            dims=["return_period", "longitude", "latitude"],
            coords=dict(
                return_period=[1, 10, 100],
                longitude=np.arange(12),
                latitude=np.arange(13),
            ),
        )
        with patch("climada_petals.hazard.rf_glofas.interp1d") as interp1d_mock:
            interp1d_mock.return_value = lambda x: np.full_like(x, 0.0)
            flood_depth(da_return_period, da_flood_maps)
            hazard_args = np.array([call[0][1] for call in interp1d_mock.call_args_list])
            self.assertFalse(np.any(np.isnan(hazard_args)))

        # Check that DataArrays have to be aligned
        x_diff = x + self.rng.uniform(-1e-3, 1e-3, size=x.shape)
        da_return_period = xr.DataArray(
            data=values,
            dims=["longitude", "latitude", "core_dim"],
            coords=dict(longitude=x_diff, latitude=x_diff, core_dim=core_dim),
        )
        with self.assertRaises(ValueError) as cm:
            flood_depth(da_return_period, da_flood_maps)
        self.assertIn("cannot align objects", str(cm.exception))

    def test_reindex(self):
        """Test the custom reindex function"""
        # Define target
        x = np.arange(10)
        y = np.arange(10, 20)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        values = xx + yy
        # print(values)
        target = xr.DataArray(values, dims=["y", "x"], coords=dict(x=x, y=y))

        # Define source
        x_diff = x + self.rng.uniform(-0.1, 0.1, size=x.shape)
        y_diff = y + self.rng.uniform(-0.1, 0.1, size=y.shape)
        source = xr.DataArray(
            np.zeros_like(values), dims=["y", "x"], coords=dict(x=x_diff, y=y_diff)
        )

        # Default values
        res = reindex(target, source)
        npt.assert_array_equal(res["x"], x_diff)
        npt.assert_array_equal(res["y"], y_diff)
        npt.assert_array_equal(res.values, values)

        # Add tolerance, we should have some NaNs then
        res = reindex(target, source, tolerance=1e-3)
        self.assertTrue(res.isnull().any())

        # Change fill value
        res = reindex(target, source, tolerance=1e-2, fill_value=-10)
        self.assertTrue((res == -10).any())

        # Check raise error
        with self.assertRaises(ValueError) as cm:
            reindex(target, source, tolerance=1e-3, assert_no_fill_value=True)
        self.assertIn("exceeds tolerance", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            reindex(target, source, fill_value=11, assert_no_fill_value=True)
        self.assertIn("does already contain reindex fill value", str(cm.exception))

    def test_sel_lon_lat_slice(self):
        """Test selection of lon/lat slices"""
        x = np.arange(10)
        target = create_data_array(x, x, np.zeros((10, 10)), "target")
        x_new = np.linspace(0, 6.5, 10)
        source = create_data_array(x_new, x_new, np.zeros((10, 10)), "source")

        res = sel_lon_lat_slice(target, source)
        self.assertEqual(res["latitude"][0], 0)
        self.assertEqual(res["latitude"][-1], 6)
        self.assertEqual(res["longitude"][0], 0)
        self.assertEqual(res["longitude"][-1], 6)

        # Flip coordinates
        target = create_data_array(x, np.flip(x), np.zeros((10, 10)), "target")
        source = create_data_array(x_new, np.flip(x_new), np.zeros((10, 10)), "source")
        res = sel_lon_lat_slice(target, source)
        self.assertEqual(res["latitude"][0], 6)
        self.assertEqual(res["latitude"][-1], 0)
        self.assertEqual(res["longitude"][0], 0)
        self.assertEqual(res["longitude"][-1], 6)


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDantroOpsGloFAS)
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestDantroOpsGloFASDownload)
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
