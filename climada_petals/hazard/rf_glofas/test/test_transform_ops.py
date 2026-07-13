import tempfile
import unittest
from unittest.mock import patch, MagicMock, DEFAULT
from pathlib import Path

import numpy as np
import numpy.testing as npt
from numpy.random import default_rng

import xarray as xr
import xarray.testing as xrt
import geopandas as gpd
from shapely.geometry import Polygon

from climada_petals.hazard.rf_glofas.transform_ops import (
    download_glofas_discharge,
    return_period,
    return_period_resample,
    interpolate_space,
    regrid,
    flood_depth,
    reindex,
    sel_lon_lat_slice,
    max_from_isel,
    apply_flopros,
    fit_gumbel_r,
    save_file,
)


def cdf_mock(dis, loc, scale):
    """A mock for the gumbel_r.cdf method. Return zeros if inputs are the same, else ones"""
    if np.array_equal(dis, loc) and np.array_equal(loc, scale):
        return np.zeros_like(dis)

    return np.ones_like(dis)


def fit_mock(series, method):
    """A mock for gumbel_r.fit method. Returns min and max of the series"""
    return np.amin(series), np.amax(series)


def create_data_array(x, y, values, name):
    return xr.DataArray(
        data=values,
        dims=["longitude", "latitude"],
        coords=dict(longitude=x, latitude=y),
    ).rename(name)


class TestGlofasDownloadOps(unittest.TestCase):
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
            "climada_petals.hazard.rf_glofas.transform_ops.glofas_request",
            autospec=True,
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

    @patch(
        "climada_petals.hazard.rf_glofas.transform_ops.get_country_geometries",
        autospec=True,
    )
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

    def test_preprocess(self):
        """Test the capabilities of the preprocessing"""
        # Simple addition
        ds = download_glofas_discharge(
            "forecast", "2022-01-01", None, 1, preprocess=lambda x: x + 1
        )
        npt.assert_array_equal(ds["time"].data, [0, 1])
        npt.assert_array_equal(ds["x"].data, [0, 1, 2])
        npt.assert_array_equal(ds.data, [[1, 2, 3], [11, 12, 13]])

        # Maximum new concat dim
        ds = download_glofas_discharge(
            "forecast",
            "2022-01-01",
            None,
            1,
            preprocess=lambda x: x.max(dim="x").rename(time="year"),
            open_mfdataset_kw=dict(concat_dim="year"),
        )
        self.assertIn("year", ds.dims)
        self.assertNotIn("time", ds.dims)
        self.assertNotIn("x", ds.dims)
        npt.assert_array_equal(ds["year"].data, [0, 1])
        npt.assert_array_equal(ds.data, [2, 12])


class TestTransformOps(unittest.TestCase):
    """Test case for other dantro operations"""

    def setUp(self):
        """Set up a random number generator"""
        self.rng = default_rng(1)

    @patch("climada_petals.hazard.rf_glofas.transform_ops.gumbel_r.fit", new=fit_mock)
    def test_fit_gumbel_r(self):
        """Test the 'fit_gumbel_r' operation"""
        # Dummy data
        input_data = xr.DataArray(
            data=[[0, 1, 2], [np.nan, 2, 3], [np.nan, np.nan, 1]],
            coords=dict(x=[1, 2, 3], year=[2000, 2001, 2002]),
        )

        # Check result
        # NOTE: The mock will return min for 'loc' and max for 'scale'
        res = fit_gumbel_r(input_data)
        npt.assert_array_equal(res["loc"].values, [0, 2, np.nan])
        npt.assert_array_equal(res["scale"].values, [2, 3, np.nan])
        npt.assert_array_equal(res["samples"].values, [3, 2, 0])

    def test_max_from_isel(self):
        """Test the 'max_from_isel' operation"""
        # NOTE: Use timedelta to check support for this data type
        #       (we typically compute a maximum over multiple time steps)
        da = xr.DataArray(
            data=[[0], [1], [2], [3]],
            coords=dict(
                step=np.arange(np.timedelta64(4, "D")).astype("timedelta64[ns]"), x=[0]
            ),
        )

        # Test how it's regularly called
        res = max_from_isel(da, "step", [slice(0, 2), [0, 3, 2]])
        npt.assert_array_equal(res["x"].values, [0])
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
    @patch("climada_petals.hazard.rf_glofas.transform_ops.gumbel_r.cdf", new=cdf_mock)
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
        with patch(
            "climada_petals.hazard.rf_glofas.transform_ops.reindex", new=return_arg
        ):
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

    def test_return_period_resample(self):
        """Test 'return_period_resample' operation"""
        # Make more than 2 dims to check handling of 'core_dims'
        x = np.arange(10)
        y = np.arange(20, 10, -1)
        z = np.linspace(0, 5, 11)

        values = (x[:, None, None] * y[None, :, None] * z[None, None, :]).astype("float")
        discharge = xr.DataArray(values, coords=dict(longitude=x, latitude=y, time=z))
        gev = xr.DataArray(
            np.outer(x, y).astype("float"), coords=dict(longitude=x, latitude=y)
        )

        # Test special values
        samples = gev.copy().astype("int")
        samples[0, 1] = 0
        gev[1, 0] = np.inf
        gev[1, 1] = np.nan
        discharge[0, 0, 1] = np.nan

        # Check result
        max_return_period = 10
        result = return_period_resample(
            discharge, gev, gev, samples, 5, max_return_period=max_return_period
        )
        self.assertIn("sample", result.dims)
        self.assertEqual(result.sizes["sample"], 5)

        # Check if new dimension is ordered last
        self.assertListEqual(
            list(result.sizes.keys()), list(discharge.sizes.keys()) + ["sample"]
        )

        # Results should be NaN if there are no samples or non-finite value
        npt.assert_array_equal(result.values[0, 1, :], np.full((11, 5), np.nan))
        npt.assert_array_equal(result.values[1, 0, :], np.full((11, 5), np.nan))
        npt.assert_array_equal(result.values[1, 1, :], np.full((11, 5), np.nan))

        # Result should be NaN if discharge is NaN
        npt.assert_array_equal(result.values[0, 0, 1], [np.nan] * 5)
        mask_nan = np.isnan((result.values))
        self.assertTrue(np.any(~mask_nan))
        self.assertTrue(np.all(result.values[~mask_nan] <= max_return_period))
        self.assertTrue(np.all(result.values[~mask_nan] >= 1))

        # Checks calls to 'fit' and 'rvs'
        with patch.multiple(
            "climada_petals.hazard.rf_glofas.transform_ops.gumbel_r",
            fit=DEFAULT,
            rvs=DEFAULT,
        ) as mocks:
            mocks["fit"].return_value = (1, 1)
            bootstrap_samples = 2
            return_period_resample(discharge, gev, gev, samples, bootstrap_samples)

            # Test number of calls
            expected_calls = np.count_nonzero(
                np.isfinite(gev.values) & (samples.values > 0)
            )
            self.assertEqual(
                len(mocks["fit"].call_args_list), expected_calls * bootstrap_samples
            )

            # Test that kwargs align
            kwargs = np.array(
                [
                    (
                        call_args.kwargs["loc"],
                        call_args.kwargs["scale"],
                        call_args.kwargs["size"],
                    )
                    for call_args in mocks["rvs"].call_args_list
                ]
            )
            loc, scale, size = np.vsplit(kwargs.T, 3)
            npt.assert_array_equal(loc, size)
            npt.assert_array_equal(scale, size)

    def test_interpolate_space(self):
        """Test 'interpolate_space' and 'regrid' operations"""

        def _assert_result(da_result, da_expected_values, **kwargs):
            """Check if result is as expected"""
            npt.assert_array_equal(
                da_result["longitude"], da_expected_values["longitude"]
            )
            npt.assert_array_equal(
                da_result["latitude"], da_expected_values["latitude"]
            )
            # Interpolation causes some noise, so "allclose" is enough here
            xrt.assert_allclose(da_result, da_expected_values, **kwargs)

        x = np.arange(4.0)
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
        da_expected = xr.DataArray(
            data=expected_values,
            dims=["latitude", "longitude"],
            coords=dict(longitude=x_diff, latitude=y_diff),
        )

        # 'interpolate_space'
        da_result = interpolate_space(da_values, da_coords)
        _assert_result(da_result, da_expected)

        # 'regrid'
        da_values[2:, 2:] = np.nan
        da_expected[2:, 2:] = [[2, 5], [1, 1]]  # Nearest neighbor extrapolation

        da_result = regrid(da_values, da_coords)
        _assert_result(
            da_result,
            xr.DataArray(
                data=expected_values,
                dims=["latitude", "longitude"],
                coords=dict(longitude=x_diff, latitude=y_diff),
            ),
            rtol=2e5,
        )  # Regridding has lower accuracy

    def test_apply_flopros(self):
        """Test 'apply_flopros' operation"""
        # Create dummy data
        return_period = create_data_array(
            [0, 1, 2], [10, 11], [[1, 2], [1.5, 3], [1.5, 1.5]], "return_period"
        )
        polygons = [
            Polygon([(0, 0), (1.5, 0), (1.5, 10.5), (0, 10.5)]),
            Polygon([(1.5, 0), (3, 0), (3, 12), (0, 12), (0, 10.5), (1.5, 10.5)]),
        ]
        flopros_data = gpd.GeoDataFrame(
            {"MerL_Riv": [1, 2]}, geometry=polygons, crs="EPSG:4326"
        )

        # Call the function
        res = apply_flopros(flopros_data, return_period)
        npt.assert_array_equal(
            res.values, [[np.nan, np.nan], [1.5, 3], [np.nan, np.nan]]
        )

    def test_flood_depth(self):
        """Test 'flood_depth' operation"""
        # Create dummy datasets
        ones = np.ones((4, 3), dtype="float")
        da_flood_maps = xr.DataArray(
            data=[ones, ones * 10, ones * 100],
            dims=["return_period", "longitude", "latitude"],
            coords=dict(
                return_period=[1, 10, 100],
                longitude=np.arange(4),
                latitude=np.arange(3),
            ),
        )

        x = np.arange(4)
        y = np.arange(3)
        core_dim_1 = np.arange(3)
        core_dim_2 = np.arange(2)
        shape = (x.size, y.size, core_dim_1.size, core_dim_2.size)
        values = np.array(
            list(range(x.size * y.size * core_dim_1.size * core_dim_2.size)),
            dtype="float",
        )
        values = values.reshape(shape) + self.rng.uniform(-0.1, 0.1, size=shape)
        values.flat[0] = 101  # Above max
        values.flat[1] = 0.1  # Below min
        da_return_period = xr.DataArray(
            data=values,
            dims=["longitude", "latitude", "core_dim_1", "core_dim_2"],
            coords=dict(
                longitude=x, latitude=y, core_dim_1=core_dim_1, core_dim_2=core_dim_2
            ),
        ).astype(np.float32)

        da_result = flood_depth(da_return_period, da_flood_maps)
        self.assertEqual(da_result.name, "Flood Depth")
        # NOTE: Single point precision, so reduce the decimal accuracy
        xrt.assert_allclose(da_result, da_return_period.clip(1, 100))

        # Check NaN shortcut
        da_flood_maps = xr.DataArray(
            data=[np.full_like(ones, np.nan)] * 3,
            dims=["return_period", "longitude", "latitude"],
            coords=dict(
                return_period=[1, 10, 100],
                longitude=np.arange(4),
                latitude=np.arange(3),
            ),
        )
        da_result = flood_depth(da_return_period, da_flood_maps)
        self.assertTrue(da_result.isnull().all())

        # Check NaN sanitizer
        da_flood_maps = xr.DataArray(
            data=[np.full_like(ones, np.nan), ones * 9, ones * 99],
            dims=["return_period", "longitude", "latitude"],
            coords=dict(
                return_period=[1, 10, 100],
                longitude=np.arange(4),
                latitude=np.arange(3),
            ),
        )
        da_return_period[...] = 1 + self.rng.uniform(-0.1, 0.1, size=shape)
        da_result = flood_depth(da_return_period, da_flood_maps)
        xrt.assert_allclose(da_result, (da_return_period - 1).clip(min=0))

        # Check that DataArrays have to be aligned
        x_diff = x + self.rng.uniform(-1e-3, 1e-3, size=x.shape)
        y_diff = y + self.rng.uniform(-1e-3, 1e-3, size=y.shape)
        da_return_period = xr.DataArray(
            data=values,
            dims=["longitude", "latitude", "core_dim_1", "core_dim_2"],
            coords=dict(
                longitude=x_diff,
                latitude=y_diff,
                core_dim_1=core_dim_1,
                core_dim_2=core_dim_2,
            ),
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
        target = xr.DataArray(
            values.astype("float"), dims=["y", "x"], coords=dict(x=x, y=y)
        )

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

    def test_save_file(self):
        """Test the file saving wrapper for the dantro pipeline"""
        # Mock the dataset
        ds = MagicMock(xr.Dataset)
        ds.data_vars = ["foo", "bar"]

        # Call the function
        outpath = Path(tempfile.gettempdir()) / "outpath"
        encoding = dict(bar=dict(dtype="float64", some_setting=True))
        encoding_defaults = dict(zlib=True, other_setting=False)

        # Assert calls
        save_file(ds, outpath, encoding, **encoding_defaults)
        ds.to_netcdf.assert_called_once_with(
            outpath.with_suffix(".nc"),
            encoding=dict(
                foo=dict(dtype="float32", zlib=True, complevel=4, other_setting=False),
                bar=dict(
                    dtype="float64",
                    zlib=True,
                    complevel=4,
                    other_setting=False,
                    some_setting=True,
                ),
            ),
            engine="netcdf4",
        )
        ds.to_netcdf.reset_mock()

        # Any suffix will be forwarded
        outpath = outpath.with_suffix(".suffix")
        defaults = dict(dtype="float32", zlib=False, complevel=4)
        save_file(ds, outpath)
        ds.to_netcdf.assert_called_once_with(
            outpath, encoding=dict(foo=defaults, bar=defaults), engine="netcdf4"
        )

        # KeyError for data_vars that do not exist
        with self.assertRaises(KeyError) as cm:
            save_file(ds, outpath, dict(baz=dict(stuff=True)))
        self.assertIn("baz", str(cm.exception))
        ds.to_netcdf.reset_mock()

        # DataArray must be promoted
        da = MagicMock(xr.DataArray)
        da.to_dataset.return_value = ds
        save_file(da, outpath)
        da.to_dataset.assert_called_once_with()
        ds.to_netcdf.assert_called_once_with(
            outpath, encoding=dict(foo=defaults, bar=defaults), engine="netcdf4"
        )


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestGlofasDownloadOps)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTransformOps))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
