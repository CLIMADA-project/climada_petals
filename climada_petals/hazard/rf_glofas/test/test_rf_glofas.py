import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import xarray as xr
import numpy as np
import numpy.testing as npt
from dask.distributed import Client

from climada.hazard import Hazard
from climada_petals.hazard.rf_glofas import hazard_series_from_dataset
from climada_petals.hazard.rf_glofas.rf_glofas import dask_client


class TestDaskClient(unittest.TestCase):
    """Test the custom dask client context manager"""

    def test_dask_client(self):
        """Check if context manager behaves as expected"""
        client = MagicMock(spec=Client)
        with patch(
            "climada_petals.hazard.rf_glofas.rf_glofas.Client", return_value=client
        ) as pt:
            # Call the manager
            with dask_client(2, 3, "1G", "foo", foo="bar") as cm:
                self.assertIs(cm, client)
                pt.assert_called_once_with(
                    "foo",
                    n_workers=2,
                    threads_per_worker=3,
                    memory_limit="1G",
                    foo="bar",
                )

        client.close.assert_called_once_with()


class TestHazardSeriesFromDataset(unittest.TestCase):
    """Test case for contents of the `rf_glofas` submodule"""

    def test_single_hazard(self):
        """Test hazard_series_from_dataset resulting in single hazard"""
        ds = xr.Dataset(
            data_vars=dict(
                intensity=(["time", "latitude", "longitude"], np.zeros((4, 2, 3)))
            ),
            coords=dict(
                time=(
                    "time",
                    np.array(
                        [np.datetime64(f"2000-01-{i:02d}") for i in range(1, 5)]
                    ).astype("datetime64[ns]"),
                ),
                latitude=("latitude", np.arange(2)),
                longitude=("longitude", np.arange(3)),
            ),
        )

        # Use time as event
        haz = hazard_series_from_dataset(ds, "intensity", "time")
        self.assertIsInstance(haz, Hazard)

        # Check hazard
        num_events = ds.sizes["time"]
        self.assertEqual(haz.size, num_events)
        num_centroids = ds.sizes["latitude"] * ds.sizes["longitude"]
        self.assertEqual(haz.centroids.size, num_centroids)
        self.assertTupleEqual(haz.intensity.shape, (num_events, num_centroids))

        # Check data
        npt.assert_array_equal(
            haz.date, [pd.to_datetime(x).toordinal() for x in ds["time"].values]
        )

    def _check_series(self, series, length, num_events, num_centroids):
        """Check the value within a hazard series"""
        self.assertEqual(series.size, length)
        npt.assert_array_equal([haz.size for haz in series], [num_events] * length)
        npt.assert_array_equal(
            [haz.centroids.size for haz in series],
            [num_centroids] * length,
        )
        npt.assert_array_equal(
            [haz.intensity.shape for haz in series],
            [(num_events, num_centroids)] * length,
        )

    def test_single_dim(self):
        """Test hazard_series_from_dataset resulting in single level series"""
        ds = xr.Dataset(
            data_vars=dict(
                intensity=(
                    ["number", "time", "latitude", "longitude"],
                    np.zeros((5, 4, 2, 3)),
                )
            ),
            coords=dict(
                number=("number", np.arange(5)),
                time=(
                    "time",
                    np.array(
                        [np.datetime64(f"2000-01-{i:02d}") for i in range(1, 5)]
                    ).astype("datetime64[ns]"),
                ),
                latitude=("latitude", np.arange(2)),
                longitude=("longitude", np.arange(3)),
            ),
        )

        # Use time as event
        haz_series = hazard_series_from_dataset(ds, "intensity", "time")
        self.assertIsInstance(haz_series, pd.Series)

        # Check index
        index = haz_series.index
        self.assertEqual(index.nlevels, 1)
        self.assertSetEqual(set(index.names), {"number"})
        npt.assert_array_equal(index.get_level_values("number"), ds["number"].values)

        # Check series
        self._check_series(
            haz_series,
            length=ds.sizes["number"],
            num_events=ds.sizes["time"],
            num_centroids=ds.sizes["latitude"] * ds.sizes["longitude"],
        )

    def test_multi_dims(self):
        """Test hazard_series_from_dataset resulting in multiindex series"""
        ds = xr.Dataset(
            data_vars=dict(
                intensity=(
                    ["another_number", "number", "time", "latitude", "longitude"],
                    np.zeros((6, 5, 4, 2, 3)),
                )
            ),
            coords=dict(
                another_number=("another_number", np.arange(6)),
                number=("number", np.arange(5)),
                time=(
                    "time",
                    np.array(
                        [np.datetime64(f"2000-01-{i:02d}") for i in range(1, 5)]
                    ).astype("datetime64[ns]"),
                ),
                latitude=("latitude", np.arange(2)),
                longitude=("longitude", np.arange(3)),
            ),
        )

        # Use time as event
        haz_series = hazard_series_from_dataset(ds, "intensity", "time")
        self.assertIsInstance(haz_series, pd.Series)

        # Check index
        index = haz_series.index
        self.assertEqual(index.nlevels, 2)
        self.assertSetEqual(
            set(index.names), {"number", "another_number"}
        )  # NOTE: Order not defined
        npt.assert_array_equal(
            index.get_level_values("number").unique(), ds["number"].values
        )
        npt.assert_array_equal(
            index.get_level_values("another_number").unique(),
            ds["another_number"].values,
        )

        # Check series
        self._check_series(
            haz_series,
            length=ds.sizes["number"] * ds.sizes["another_number"],
            num_events=ds.sizes["time"],
            num_centroids=ds.sizes["latitude"] * ds.sizes["longitude"],
        )


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDaskClient)
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestHazardSeriesFromDataset)
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
