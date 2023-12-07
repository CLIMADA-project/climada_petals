import unittest
from unittest.mock import patch, MagicMock, DEFAULT
from tempfile import TemporaryDirectory
from pathlib import Path

import xarray as xr
import xarray.testing as xrt
import geopandas as gpd
import pandas.testing as pdt

from climada.util.constants import DEF_CRS
from climada_petals.hazard.rf_glofas.river_flood_computation import (
    _maybe_open_dataarray,
    RiverFloodInundation,
)


class TestMaybeOpenDataArray(unittest.TestCase):
    """Check the maybe_open_dataarray context manager"""

    def setUp(self):
        """Create the input array"""
        self.tempdir = TemporaryDirectory()
        self.arr_1 = xr.DataArray(data=[0], coords={"dim": [0]})
        self.arr_2 = xr.DataArray(data=[1], coords={"dim": [1]})
        self.filename = Path(self.tempdir.name) / "file.nc"
        self.arr_2.to_netcdf(self.filename)

    def tearDown(self):
        """Clean up temporary directory"""
        self.tempdir.cleanup()

    @patch("climada_petals.hazard.rf_glofas.river_flood_computation.xr.open_dataarray")
    def test_with_arr(self, open_dataarray_mock):
        """Check behavior if array is given"""
        with _maybe_open_dataarray(self.arr_1, self.filename) as da:
            xrt.assert_identical(self.arr_1, da)
        open_dataarray_mock.assert_not_called()

    def test_without_arr(self):
        """Check if file is correctly opened with no array input"""
        with _maybe_open_dataarray(None, self.filename) as da:
            xrt.assert_identical(self.arr_2, da)

    @patch("climada_petals.hazard.rf_glofas.river_flood_computation.xr.open_dataarray")
    def test_file_is_closed(self, open_dataarray_mock):
        """Check if the dataset is correctly close after release"""
        arr = MagicMock(xr.DataArray)
        open_dataarray_mock.return_value = arr

        with _maybe_open_dataarray(None, self.filename) as da:
            open_dataarray_mock.assert_called_once_with(self.filename)
            da.close.assert_not_called()

        da.close.assert_called_once()

    @patch("climada_petals.hazard.rf_glofas.river_flood_computation.xr.open_dataarray")
    def test_kwargs(self, open_dataarray_mock):
        """Check if kwargs are passed correctly"""
        kwargs = {"chunks": "auto", "foo": "bar"}
        with _maybe_open_dataarray(None, self.filename, **kwargs) as _:
            open_dataarray_mock.assert_called_once_with(self.filename, **kwargs)


@patch.multiple(
    "climada_petals.hazard.rf_glofas.river_flood_computation",
    download_glofas_discharge=DEFAULT,
    return_period=DEFAULT,
    return_period_resample=DEFAULT,
    regrid=DEFAULT,
    apply_flopros=DEFAULT,
    flood_depth=DEFAULT,
)
class TestRiverFloodInundation(unittest.TestCase):
    """Check the RiverFloodInundation class"""

    @classmethod
    def setUpClass(cls):
        """Create fake data"""
        cls.tempdir = TemporaryDirectory()
        cls.temppath = Path(cls.tempdir.name)

        cls.flood_maps = xr.DataArray(
            [[[0]]],
            coords={"return_period": [0], "longitude": [0], "latitude": [0]},
            name="flood_maps",
        )
        cls.flood_maps.to_netcdf(cls.temppath / "flood-maps.nc")

        arr = xr.DataArray([[0]], coords={"longitude": [0], "latitude": [0]})
        cls.gumbel_fits = xr.Dataset({"loc": arr, "scale": arr, "samples": arr})
        cls.gumbel_fits.to_netcdf(cls.temppath / "gumbel-fit.nc")

        cls.flopros = gpd.GeoDataFrame(
            data={"data": [0]}, geometry=gpd.points_from_xy([0], [0], crs=DEF_CRS)
        )
        Path(cls.temppath / "FLOPROS_shp_V1").mkdir()
        cls.flopros.to_file(cls.temppath / "FLOPROS_shp_V1/FLOPROS_shp_V1.shp")

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory"""
        cls.tempdir.cleanup()

    def setUp(self):
        """Initialize the class instance"""
        self.cache_dir = self.temppath / "cache"
        self.rf = RiverFloodInundation(data_dir=self.temppath, cache_dir=self.cache_dir)

    def test_init(self, **_):
        """Test object initialization"""
        # Load files
        xrt.assert_identical(self.rf.flood_maps, self.flood_maps)
        xrt.assert_identical(self.rf.gumbel_fits, self.gumbel_fits)
        pdt.assert_frame_equal(self.rf.flopros, self.flopros)

        # Create cache dir
        self.assertTrue(self.cache_dir.is_dir())
        self.assertTrue(Path(self.rf._tempdir.name).is_dir())
        for path in self.rf.cache_paths._asdict().values():
            self.assertIn(self.temppath, path.parents)

        # Check that data_dir must exist
        with self.assertRaises(FileNotFoundError) as cm:
            RiverFloodInundation(data_dir="some_dir")
        self.assertIn("'data_dir' does not exist", str(cm.exception))

    def test_clear_cache(self, **_):
        """Check if cache directory is correctly removed"""
        self.assertTrue(self.cache_dir.is_dir())
        first_cache = Path(self.rf._tempdir.name)
        self.assertTrue(first_cache.is_dir())

        # Create new cache dir
        self.rf.clear_cache()
        second_cache = Path(self.rf._tempdir.name)
        self.assertFalse(first_cache.is_dir())
        self.assertTrue(second_cache.is_dir())
        for path in self.rf.cache_paths._asdict().values():
            self.assertIn(second_cache, path.parents)

    def _assert_store_intermediates(
        self, rf, func_name, arr_compare, cache_name, *args, **kwargs
    ):
        """Check if store_intermediates behaves correctly"""
        func = getattr(rf, func_name)
        filename = getattr(rf.cache_paths, cache_name)
        filename.unlink(missing_ok=True)

        rf.store_intermediates = False
        result = func(*args, **kwargs)
        xrt.assert_identical(result, arr_compare)
        self.assertFalse(filename.is_file())

        rf.store_intermediates = True
        result = func(*args, **kwargs)
        xrt.assert_identical(result, arr_compare)
        self.assertTrue(filename.is_file())
        with xr.open_dataarray(filename) as arr:
            xrt.assert_identical(arr, arr_compare)

    def test_download_forecast(self, download_glofas_discharge: MagicMock, **_):
        """Check if download_forecast passes parameters correctly"""
        download_glofas_discharge.return_value = self.flood_maps

        preprocess = lambda x: x
        self._assert_store_intermediates(
            self.rf,
            "download_forecast",
            self.flood_maps,
            "discharge",
            "ABC",
            "2000-01-01",
            lead_time_days=2,
            preprocess=preprocess,
            foo="bar",
        )
        download_glofas_discharge.assert_called_with(
            product="forecast",
            date_from="2000-01-01",
            date_to=None,
            countries="ABC",
            preprocess=preprocess,
            leadtime_hour=["24", "48"],
            foo="bar",
        )

    def test_download_reanalysis(self, download_glofas_discharge: MagicMock, **_):
        """Check if download_reanalysis passes parameters correctly"""
        download_glofas_discharge.return_value = self.flood_maps
        preprocess = lambda x: x
        self._assert_store_intermediates(
            self.rf,
            "download_reanalysis",
            self.flood_maps,
            "discharge",
            "ABC",
            2000,
            preprocess=preprocess,
            foo="bar",
        )
        download_glofas_discharge.assert_called_with(
            product="historical",
            date_from="2000",
            date_to=None,
            countries="ABC",
            preprocess=preprocess,
            foo="bar",
        )

    def test_return_period(self, return_period, **_):
        """Check if return_period passes parameters correctly"""
        return_period.return_value = self.flood_maps
        self._assert_store_intermediates(
            self.rf,
            "return_period",
            self.flood_maps,
            "return_period",
            self.flood_maps,
        )
        return_period.assert_called_with(
            self.flood_maps, self.gumbel_fits["loc"], self.gumbel_fits["scale"]
        )

    @patch("climada_petals.hazard.rf_glofas.river_flood_computation.dask_client")
    def test_return_period_resample(self, dask_client, return_period_resample, **_):
        """Check if return_period_resample passes parameters correctly"""
        return_period_resample.return_value = self.flood_maps
        self._assert_store_intermediates(
            self.rf,
            "return_period_resample",
            self.flood_maps,
            "return_period",
            10,
            self.flood_maps,
        )
        expected_kwargs = dict(
            discharge=self.flood_maps,
            gev_loc=self.gumbel_fits["loc"],
            gev_scale=self.gumbel_fits["scale"],
            gev_samples=self.gumbel_fits["samples"],
            bootstrap_samples=10,
            fit_method="MM",
        )
        return_period_resample.assert_called_with(**expected_kwargs)
        dask_client.assert_not_called()

        # Parallel
        self._assert_store_intermediates(
            self.rf,
            "return_period_resample",
            self.flood_maps,
            "return_period",
            10,
            self.flood_maps,
            num_workers=4,
        )
        return_period_resample.assert_called_with(**expected_kwargs)
        dask_client.assert_any_call(4, 1, "2G")

    def test_regrid(self, regrid, **_):
        """Check if regrid passes parameters correctly"""
        regrid.return_value = self.flood_maps, "regridder"
        self._assert_store_intermediates(
            self.rf,
            "regrid",
            self.flood_maps,
            "return_period_regrid",
            self.flood_maps,
            reuse_regridder=False,
        )
        regrid.assert_called_with(
            self.flood_maps,
            self.flood_maps,
            method="bilinear",
            regridder=None,
            return_regridder=True,
        )
        self.assertIsNotNone(self.rf.regridder)

        self._assert_store_intermediates(
            self.rf,
            "regrid",
            self.flood_maps,
            "return_period_regrid",
            self.flood_maps,
            reuse_regridder=True,
        )
        regrid.assert_called_with(
            self.flood_maps,
            self.flood_maps,
            method="bilinear",
            regridder="regridder",  # Reused
            return_regridder=True,
        )

    def test_apply_protection(self, apply_flopros, **_):
        """Check if apply_protection passes parameters correctly"""
        apply_flopros.return_value = self.flood_maps
        self._assert_store_intermediates(
            self.rf,
            "apply_protection",
            self.flood_maps,
            "return_period_regrid_protect",
            self.flood_maps,
        )

        # Clumsy check because the dataframe does not support equal comparison
        call_args = apply_flopros.call_args.args
        pdt.assert_frame_equal(call_args[0], self.flopros)
        xrt.assert_identical(call_args[1], self.flood_maps)

    def test_flood_depth(self, flood_depth, **_):
        """Check if flood_depth passes parameters correctly"""
        flood_depth.return_value = self.flood_maps

        # Default, use argument
        self.rf.flood_depth(self.flood_maps)
        flood_depth.assert_called_with(
            self.flood_maps,
            self.flood_maps,
        )

        # Store regrid
        self.flood_maps.to_netcdf(self.rf.cache_paths.return_period_regrid)
        self.rf.flood_depth(None)
        flood_depth.assert_called_with(
            self.flood_maps,
            self.flood_maps,
        )
        self.rf.cache_paths.return_period_regrid.unlink()

        # Store regrid protect
        self.flood_maps.to_netcdf(self.rf.cache_paths.return_period_regrid_protect)
        self.rf.flood_depth(None)
        flood_depth.assert_called_with(
            self.flood_maps,
            self.flood_maps,
        )

    def test_compute_default(
        self, return_period, return_period_resample, regrid, flood_depth, **_
    ):
        """Test compute algorithm with defaults"""
        return_period.return_value = self.flood_maps
        return_period_resample.return_value = self.flood_maps
        regrid.return_value = self.flood_maps, "regridder"
        flood_depth.return_value = self.flood_maps

        # No data
        with self.assertRaises(RuntimeError) as cm:
            self.rf.compute(None)
        self.assertIn("No discharge data", str(cm.exception))

        # Default
        ds_result = self.rf.compute(self.flood_maps)
        return_period.assert_called_once()
        return_period_resample.assert_not_called()
        regrid.assert_called_once()
        self.assertEqual(flood_depth.call_count, 2)
        xrt.assert_equal(ds_result["flood_depth"], self.flood_maps)
        xrt.assert_equal(ds_result["flood_depth_flopros"], self.flood_maps)

        # Reset mocks
        for mock in (return_period, return_period_resample, regrid, flood_depth):
            mock.reset_mock()

        # More arguments
        ds_result = self.rf.compute(
            self.flood_maps,
            apply_protection=True,
            resample_kws=dict(num_bootstrap_samples=10),
            regrid_kws=dict(reuse_regridder=True),
        )
        return_period.assert_not_called()
        return_period_resample.assert_called_once()
        regrid.assert_called_once()
        flood_depth.assert_called_once()
        self.assertNotIn("flood_depth", ds_result.data_vars.keys())
        xrt.assert_equal(ds_result["flood_depth_flopros"], self.flood_maps)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestMaybeOpenDataArray)
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestRiverFloodInundation)
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
