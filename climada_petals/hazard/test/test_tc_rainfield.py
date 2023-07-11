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
from pathlib import Path
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
from climada.util.constants import DEMO_DIR


def getTestData():
    client = Client()
    centr_ds = client.get_dataset_info(name='test_tc_rainfield', status='test_dataset')
    _, [centr_test_mat, track, track_short, haz_mat] = client.download_dataset(centr_ds)
    return Centroids.from_mat(centr_test_mat), track, track_short, haz_mat


CENTR_TEST_BRB, TEST_TRACK, TEST_TRACK_SHORT, HAZ_TEST_MAT = getTestData()


class TestReader(unittest.TestCase):
    """Test loading funcions from the TCRain class"""

    def test_set_one_pass(self):
        """Test from_tracks constructor with a single track."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_haz = TCRain.from_tracks(tc_track, centroids=CENTR_TEST_BRB)

        self.assertEqual(tc_haz.haz_type, 'TR')
        self.assertEqual(tc_haz.tag.description, [])
        self.assertEqual(tc_haz.tag.file_name, ['Name: 1951239N12334'])
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
        self.assertAlmostEqual(tc_haz.intensity[0, 100], 128.46424063696978)
        self.assertAlmostEqual(tc_haz.intensity[0, 260], 15.697609721478253)

        # For testing, fill in the mean temperature over the storm life time (from ERA5).
        # This increases the results by more than 50% because the default value for saturation
        # specific humidity corresponds to a temperature of only ~267 K.
        tc_track.data[0]["t600"] = xr.full_like(tc_track.data[0]["central_pressure"], 275.0)
        tc_haz = TCRain.from_tracks(tc_track, model="TCR", centroids=CENTR_TEST_BRB)
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 296)
        self.assertAlmostEqual(tc_haz.intensity[0, 100], 208.0608895225061)
        self.assertAlmostEqual(tc_haz.intensity[0, 260], 25.514027006851833)

    def test_from_file_pass(self):
        """Test from_tracks constructor with one input."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_haz = TCRain.from_tracks(tc_track, centroids=CENTR_TEST_BRB)
        tc_haz.check()

        self.assertEqual(tc_haz.haz_type, 'TR')
        self.assertEqual(tc_haz.tag.description, [])
        self.assertEqual(tc_haz.tag.file_name, ['Name: 1951239N12334'])
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
        self.assertEqual(tc_haz.tag.description, [])
        self.assertEqual(tc_haz.tag.file_name, ['Name: 1951239N12334'])
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
        t0 = 270.0
        pref = 900

        # With h going to zero, the error of the Taylor approximation should go to zero
        # at the order of h^2.
        hs = np.array([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        ts = t0 + hs

        [r0], [dr0] = _r_from_t_same_level(pref, np.array([t0]), gradient=True)
        r_mix, _ = _r_from_t_same_level(pref, ts, gradient=False)
        diffs_rel = np.abs(r0 + dr0 * hs - r_mix) / hs**2
        diffs_rel_mean = diffs_rel.mean()
        orders = np.abs(diffs_rel - diffs_rel_mean) / diffs_rel_mean
        np.testing.assert_array_less(orders, 0.1)

        # Because of a bug in the reference MATLAB implementation,
        # the same doesn't work for `matlab_ref_mode=True`.
        [r0], [dr0] = _r_from_t_same_level(
            pref, np.array([t0]), gradient=True, matlab_ref_mode=True,
        )
        r_mix, _ = _r_from_t_same_level(pref, ts, gradient=False, matlab_ref_mode=True)
        diffs_rel = np.abs(r0 + dr0 * hs - r_mix) / hs**2
        diffs_rel_mean = diffs_rel.mean()
        orders = np.abs(diffs_rel - diffs_rel_mean) / diffs_rel_mean
        self.assertGreater(orders.max(), 1)

    def test_qs_from_t_diff_level(self):
        tracks = TCTracks.from_hdf5(DEMO_DIR / "tcrain_examples.nc")
        track_ds = tracks.data[0]
        temps_in = track_ds["t600"].values.copy()
        temps_in[3] = -9999.0  # test fill value
        vmax = track_ds["max_sustained_wind"].values * KN_TO_MS
        pres_in, pres_out = 600, 900
        q_out = _qs_from_t_diff_level(temps_in, vmax, pres_in, pres_out)
        np.testing.assert_allclose(q_out, [
            0.015216, 0.015169, 0.015053, 0.000000, 0.014987, 0.015450, 0.014783, 0.015774,
            0.015540, 0.016048, 0.015818, 0.017915, 0.019280, 0.018759, 0.017942, 0.017678,
            0.017740, 0.018325, 0.019002, 0.018353, 0.018261, 0.017653, 0.018986, 0.018933,
            0.017967, 0.017566, 0.018461, 0.019933, 0.020039, 0.020028, 0.020916, 0.021371,
            0.022447, 0.023015, 0.022754,
        ], rtol=1e-4)

    def test_track_to_si(self):
        tracks = TCTracks.from_hdf5(DEMO_DIR / "tcrain_examples.nc")
        track_ds = tracks.data[0]
        si_track = _track_to_si_with_q_and_shear(track_ds)
        self.assertIn("q900", si_track.variables)
        self.assertIn("v850", si_track.variables)
        self.assertEqual(si_track["v850"].shape, (track_ds.sizes["time"], 2))
        # check that the meridional (v) component is listed first
        np.testing.assert_array_equal(si_track["v850"].values[:, 0], track_ds["v850"].values)
        np.testing.assert_array_equal(si_track["v850"].values[:, 1], track_ds["u850"].values)

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModel))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
