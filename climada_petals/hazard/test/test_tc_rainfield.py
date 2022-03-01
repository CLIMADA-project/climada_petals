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

import unittest
import datetime as dt
from pathlib import Path
import numpy as np
from scipy import sparse

from climada import CONFIG
from climada.hazard.tc_tracks import TCTracks
from climada_petals.hazard.tc_rainfield import TCRain, rainfield_from_track
from climada.hazard.centroids.centr import Centroids
from climada.util.api_client import Client
import climada.hazard.test as hazard_test


def getTestData():
    client = Client()
    centr_ds = client.get_dataset_info(name='test_tc_rainfield', status='test_dataset')
    _, [centr_test_mat, track, track_short, haz_mat] = client.download_dataset(centr_ds)
    return Centroids.from_mat(centr_test_mat), track, track_short, haz_mat


CENTR_TEST_BRB, TEST_TRACK, TEST_TRACK_SHORT, HAZ_TEST_MAT = getTestData()


class TestReader(unittest.TestCase):
    """Test loading funcions from the TCRain class"""

    def test_set_one_pass(self):
        """Test _set_from_track function."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_haz = TCRain._from_track(tc_track.data[0], CENTR_TEST_BRB)

        self.assertEqual(tc_haz.tag.haz_type, 'TR')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, 'IBTrACS: 1951239N12334')
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
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))

        self.assertAlmostEqual(tc_haz.intensity[0, 100], 99.7160586771286, 6)
        self.assertAlmostEqual(tc_haz.intensity[0, 260], 33.2087621869295)
        self.assertEqual(tc_haz.fraction[0, 100], 1)
        self.assertEqual(tc_haz.fraction[0, 260], 1)

        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 296)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 296)

    def test_from_file_pass(self):
        """Test from_tracks constructor with one input."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_haz = TCRain.from_tracks(tc_track, CENTR_TEST_BRB)
        tc_haz.check()

        self.assertEqual(tc_haz.tag.haz_type, 'TR')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, 'IBTrACS: 1951239N12334')
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
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))

        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 0)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 0)

    def test_two_files_pass(self):
        """Test from_tracks constructor with two ibtracs."""
        tc_track = TCTracks.from_processed_ibtracs_csv([TEST_TRACK_SHORT, TEST_TRACK_SHORT])
        tc_haz = TCRain.from_tracks(tc_track, CENTR_TEST_BRB)
        tc_haz.remove_duplicates()
        tc_haz.check()

        self.assertEqual(tc_haz.tag.haz_type, 'TR')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, ['IBTrACS: 1951239N12334',
                                                'IBTrACS: 1951239N12334'])
        self.assertEqual(tc_haz.units, 'mm')
        self.assertEqual(tc_haz.centroids.size, 296)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(np.array_equal(tc_haz.orig, np.array([True])))
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))

        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 0)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 0)

class TestModel(unittest.TestCase):
    """Test modelling of rainfall"""

    def test_rainfield_from_track_pass(self):
        """Test _rainfield_from_track function. Compare to MATLAB reference."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        rainfall = rainfield_from_track(tc_track.data[0], CENTR_TEST_BRB)

        rainfall = np.round(rainfall, decimals=9)

        self.assertAlmostEqual(rainfall[0, 0], 66.801702386)
        self.assertAlmostEqual(rainfall[0, 130], 43.290917792)
        self.assertAlmostEqual(rainfall[0, 200], 76.315923838)

    def test_rainfield_diff_time_steps(self):
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)

        train_org = TCRain.from_tracks(tc_track)

        tc_track.equal_timestep(time_step_h=1)
        train_1h = TCRain.from_tracks(tc_track)

        tc_track.equal_timestep(time_step_h=0.5)
        train_05h = TCRain.from_tracks(tc_track)

        np.testing.assert_allclose(train_org.intensity.sum(),
                                   train_1h.intensity.sum(), rtol=1e-1)
        np.testing.assert_allclose(train_org.intensity.sum(),
                                   train_05h.intensity.sum(), rtol=1e-1)
        np.testing.assert_allclose(train_05h.intensity.sum(),
                                   train_1h.intensity.sum(), rtol=1e-1)

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModel))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
