"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test TCRain class
"""

import datetime as dt
import unittest

import numpy as np
from scipy import sparse
import xarray as xr

from climada import CONFIG
from climada.hazard import Centroids, TCTracks
import climada.hazard.test as hazard_test
from climada_petals.hazard.tc_rainfield import (
    TCRain,
    compute_rain,
    KN_TO_MS,
    MODEL_RAIN,
    _qs_from_t_diff_level,
    _r_from_t_same_level,
    _track_to_si_with_q_and_shear,
)
from climada.util.api_client import Client
from climada.util.constants import SYSTEM_DIR


def getTestData():
    client = Client()
    centr_ds = client.get_dataset_info(name='tc_rainfield_test', status='test_dataset')
    _, [centr_test_hdf5, track, track_short, haz_hdf5] = client.download_dataset(centr_ds)
    return Centroids.from_hdf5(centr_test_hdf5), track, track_short, haz_hdf5


CENTR_TEST_BRB, TEST_TRACK, TEST_TRACK_SHORT, HAZ_TEST_HDF5 = getTestData()


def tcrain_examples():
    client = Client()
    dsi = client.get_dataset_info(name='tcrain_examples', status='package-data')
    _, [drag_tif] = client.download_dataset(dsi)
    return drag_tif


class TestReader(unittest.TestCase):
    """Test loading funcions from the TCRain class"""

    def test_set_one_pass(self):
        """Test from_tracks constructor with a single track."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_haz = TCRain.from_tracks(tc_track, centroids=CENTR_TEST_BRB)

        self.assertEqual(tc_haz.haz_type, 'TR')
        self.assertEqual(tc_haz.units, 'mm')
        self.assertEqual(tc_haz.centroids.size, 296)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.date.size, 1)
        self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).year, 1951)
        self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).month, 8)
        self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).day, 27)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr_matrix))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))
        self.assertIsNone(tc_haz._get_fraction())

        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 296)
        self.assertAlmostEqual(tc_haz.intensity[0, 100], 98.61962462510677, 6)
        self.assertAlmostEqual(tc_haz.intensity[0, 260], 27.702589231065055)

    def test_tcr(self):
        """Test from_tracks constructor with model TCR."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()

        tc_haz = TCRain.from_tracks(tc_track, model="TCR", centroids=CENTR_TEST_BRB)
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 296)
        self.assertAlmostEqual(tc_haz.intensity[0, 100], 71.36902257609432)
        self.assertAlmostEqual(tc_haz.intensity[0, 260], 8.720894289710138)

        # For testing, fill in the mean temperature over the storm life time (from ERA5).
        # This increases the results by more than 70% because the default value for saturation
        # specific humidity corresponds to a temperature of only ~267 K.
        tc_track.data[0]["t600"] = xr.full_like(tc_track.data[0]["central_pressure"], 275.0)
        tc_haz = TCRain.from_tracks(tc_track, model="TCR", centroids=CENTR_TEST_BRB)
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 296)
        self.assertAlmostEqual(tc_haz.intensity[0, 100], 123.55255892009247)
        self.assertAlmostEqual(tc_haz.intensity[0, 260], 15.148539942329757)

    @unittest.skipUnless(SYSTEM_DIR.joinpath("IBTrACS.ALL.v04r00.nc").is_file(),
                         "IBTrACS file is missing, no download in unitttests")
    def test_cross_antimeridian(self):
        # Two locations on the island Taveuni (Fiji), one west and one east of 180° longitude.
        # We list the second point twice, with different lon-normalization:
        cen = Centroids.from_lat_lon([-16.95, -16.8, -16.8], [179.9, 180.1, -179.9])

        # Cyclone YASA (2020) passed directly over Fiji
        tr = TCTracks.from_ibtracs_netcdf(storm_id=["2020346S13168"])

        inten = TCRain.from_tracks(tr, centroids=cen).intensity.toarray()[0, :]

        # Centroids 1 and 2 are identical, they just use a different normalization for lon. This
        # should not affect the result at all:
        self.assertEqual(inten[1], inten[2])

        # All locations should be clearly affected by rain of appx. 135 mm. The exact
        # values are not so important for this test:
        np.testing.assert_allclose(inten, 135, atol=10)

    def test_from_file_pass(self):
        """Test from_tracks constructor with one input."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_haz = TCRain.from_tracks(tc_track, centroids=CENTR_TEST_BRB)
        tc_haz.check()

        self.assertEqual(tc_haz.haz_type, 'TR')
        self.assertEqual(tc_haz.units, 'mm')
        self.assertEqual(tc_haz.centroids.size, 296)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertEqual(tc_haz.category, tc_track.data[0].category)
        self.assertEqual(tc_haz.basin[0], "NA")
        self.assertIsInstance(tc_haz.basin, list)
        self.assertIsInstance(tc_haz.category, np.ndarray)
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr_matrix))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))
        self.assertIsNone(tc_haz._get_fraction())
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 0)

    def test_two_files_pass(self):
        """Test from_tracks constructor with two ibtracs."""
        tc_track = TCTracks.from_processed_ibtracs_csv([TEST_TRACK_SHORT, TEST_TRACK_SHORT])
        tc_haz = TCRain.from_tracks(tc_track, centroids=CENTR_TEST_BRB)
        tc_haz.remove_duplicates()
        tc_haz.check()

        self.assertEqual(tc_haz.haz_type, 'TR')
        self.assertEqual(tc_haz.units, 'mm')
        self.assertEqual(tc_haz.centroids.size, 296)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(np.array_equal(tc_haz.orig, np.array([True])))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr_matrix))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))
        self.assertIsNone(tc_haz._get_fraction())
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 0)

class TestModel(unittest.TestCase):
    """Test modelling of rainfall"""

    def test_compute_rain_pass(self):
        """Test _compute_rain function. Compare to MATLAB reference."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        track_ds = tc_track.data[0]
        si_track = _track_to_si_with_q_and_shear(track_ds)
        centroids = CENTR_TEST_BRB
        ncentroids = centroids.size
        rainfall = np.zeros((1, ncentroids))
        rainrates, reachable_centr_idx = compute_rain(
            si_track, centroids.coord, MODEL_RAIN["R-CLIPER"],
        )
        rainfall[0, reachable_centr_idx] = (
            (rainrates * track_ds["time_step"].values[:, None]).sum(axis=0)
        )

        rainfall = np.round(rainfall, decimals=9)

        self.assertAlmostEqual(rainfall[0, 0], 65.114948501)
        self.assertAlmostEqual(rainfall[0, 130], 39.584947656)
        self.assertAlmostEqual(rainfall[0, 200], 73.792450959)

    def test_rainfield_diff_time_steps(self):
        """Check that the results do not depend too much on the track's time step sizes."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)

        train_org = TCRain.from_tracks(tc_track)

        tc_track.equal_timestep(time_step_h=1)
        train_1h = TCRain.from_tracks(tc_track)

        tc_track.equal_timestep(time_step_h=0.5)
        train_05h = TCRain.from_tracks(tc_track)

        for train in [train_1h, train_05h]:
            np.testing.assert_allclose(
                train_org.intensity.sum(),
                train.intensity.sum(),
                rtol=1e-1,
            )

    def test_r_from_t_same_level(self):
        """Test the derivative of _r_from_t_same_level"""
        for tetens_coeffs in ["Alduchov1996", "Buck1981", "Bolton1980", "Murray1967"]:
            t0 = 270.0
            pref = 950
            kwargs = dict(tetens_coeffs=tetens_coeffs)

            # With h going to zero, the error of the Taylor approximation should go to zero
            # at the order of h^2.
            hs = np.array([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
            ts = t0 + hs

            [r0], [dr0] = _r_from_t_same_level(pref, np.array([t0]), gradient=True, **kwargs)
            r_mix, _ = _r_from_t_same_level(pref, ts, gradient=False, **kwargs)
            diffs_rel = np.abs(r0 + dr0 * hs - r_mix) / hs**2
            diffs_rel_mean = diffs_rel.mean()
            orders = np.abs(diffs_rel - diffs_rel_mean) / diffs_rel_mean
            np.testing.assert_array_less(orders, 0.1)

            # The same doesn't work if the approximative form of the derivative is used
            # with `use_cc_derivative=True`:
            kwargs = dict(use_cc_derivative=True, **kwargs)

            [r0], [dr0] = _r_from_t_same_level(
                pref, np.array([t0]), gradient=True, **kwargs
            )
            r_mix, _ = _r_from_t_same_level(pref, ts, gradient=False, **kwargs)
            diffs_rel = np.abs(r0 + dr0 * hs - r_mix) / hs**2
            diffs_rel_mean = diffs_rel.mean()
            orders = np.abs(diffs_rel - diffs_rel_mean) / diffs_rel_mean
            self.assertGreater(orders.max(), 1)

    def test_qs_from_t_diff_level(self):
        tracks = TCTracks.from_hdf5(tcrain_examples())
        track_ds = tracks.data[0]
        temps_in = track_ds["t600"].values.copy()
        temps_in[3] = -9999.0  # test fill value
        q_out_ref = np.array([
            0.016311, 0.016263, 0.016144, 0.000000, 0.016077, 0.016549, 0.015868, 0.016880,
            0.016642, 0.017160, 0.016925, 0.019063, 0.020451, 0.019922, 0.019091, 0.018822,
            0.018885, 0.019480, 0.020169, 0.019509, 0.019416, 0.018797, 0.020152, 0.020099,
            0.019116, 0.018708, 0.019619, 0.021115, 0.021223, 0.021212, 0.022113, 0.022574,
            0.023665, 0.024241, 0.023977,
        ])
        vmax = track_ds["max_sustained_wind"].values * KN_TO_MS
        pres_in, pres_out = 600, 950
        q_out = _qs_from_t_diff_level(temps_in, vmax, pres_in, pres_out)
        np.testing.assert_allclose(q_out, q_out_ref, rtol=1e-4)

    def test_track_to_si(self):
        tracks = TCTracks.from_hdf5(tcrain_examples())
        track_ds = tracks.data[0]
        si_track = _track_to_si_with_q_and_shear(track_ds)
        self.assertIn("q950", si_track.variables)
        self.assertIn("v850", si_track.variables)
        self.assertEqual(si_track["v850"].shape, (track_ds.sizes["time"], 2))
        # check that the meridional (v) component is listed first
        np.testing.assert_array_equal(si_track["v850"].values[:, 0], track_ds["v850"].values)
        np.testing.assert_array_equal(si_track["v850"].values[:, 1], track_ds["u850"].values)

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModel))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
