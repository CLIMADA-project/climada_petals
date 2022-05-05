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

Test Wild fire class
"""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from climada_petals.hazard import WildFire

DATA_DIR = (Path(__file__).parent).joinpath('../test/data')
TEST_LC = Path.joinpath(DATA_DIR, "WF_LC.tif")
TEST_POP = Path.joinpath(DATA_DIR, "WF_POP.tif")
TEST_FIRMS = pd.read_csv(Path.joinpath(DATA_DIR, "WF_FIRMS.csv"))

description = ''

class TestWildFire(unittest.TestCase):
    """Test loading functions from the WildFire class"""

    def test_hist_fire_firms_pass(self):
        """ Test set_hist_events """
        wf = WildFire()
        wf.set_hist_fire_FIRMS(TEST_FIRMS, land_path=TEST_LC)
        wf.check()

        self.assertEqual(wf.tag.haz_type, 'WFsingle')
        self.assertEqual(wf.units, 'K')
        self.assertTrue(np.allclose(wf.event_id, np.arange(1, 4)))
        self.assertTrue(np.allclose(wf.date,
            np.array([736184, 736185, 736184])))
        self.assertTrue(np.allclose(wf.orig, np.ones(3, bool)))
        self.assertEqual(wf.event_name, ['1', '2', '3'])
        self.assertTrue(np.allclose(wf.frequency, np.ones(3)))
        self.assertEqual(wf.intensity.shape, (3, 14904))
        self.assertEqual(wf.fraction.shape, (3, 14904))
        self.assertEqual(wf.intensity[0, :].nonzero()[1][0], 4493)
        self.assertEqual(wf.intensity[1, :].nonzero()[1][3], 4907)
        self.assertEqual(wf.intensity[2, :].nonzero()[1][10], 212)
        self.assertAlmostEqual(wf.intensity[0, 4493], 336.6)
        self.assertAlmostEqual(wf.intensity[1, 4907], 311.8)
        self.assertAlmostEqual(wf.intensity[2, 212], 309.8)
        self.assertAlmostEqual(wf.intensity[0, 8000], 0.0)
        self.assertAlmostEqual(wf.fraction.max(), 1.0)
        self.assertAlmostEqual(wf.fraction.min(), 0.0)

    def test_from_hist_fire_firms_pass(self):
        """ Test set_hist_events """
        wf = WildFire()
        wf = wf.from_hist_fire_FIRMS(TEST_FIRMS, land_path=TEST_LC)
        wf.check()

        self.assertEqual(wf.tag.haz_type, 'WFsingle')
        self.assertEqual(wf.units, 'K')
        self.assertTrue(np.allclose(wf.event_id, np.arange(1, 4)))
        self.assertTrue(np.allclose(wf.date,
            np.array([736184, 736185, 736184])))
        self.assertTrue(np.allclose(wf.orig, np.ones(3, bool)))
        self.assertEqual(wf.event_name, ['1', '2', '3'])
        self.assertTrue(np.allclose(wf.frequency, np.ones(3)))
        self.assertEqual(wf.intensity.shape, (3, 14904))
        self.assertEqual(wf.fraction.shape, (3, 14904))
        self.assertEqual(wf.intensity[0, :].nonzero()[1][0], 4493)
        self.assertEqual(wf.intensity[1, :].nonzero()[1][3], 4907)
        self.assertEqual(wf.intensity[2, :].nonzero()[1][10], 212)
        self.assertAlmostEqual(wf.intensity[0, 4493], 336.6)
        self.assertAlmostEqual(wf.intensity[1, 4907], 311.8)
        self.assertAlmostEqual(wf.intensity[2, 212], 309.8)
        self.assertAlmostEqual(wf.intensity[0, 8000], 0.0)
        self.assertAlmostEqual(wf.fraction.max(), 1.0)
        self.assertAlmostEqual(wf.fraction.min(), 0.0)    

    def test_hist_fire_season_firms_pass(self):
        """ Test set_hist_event_year_set """
        wf = WildFire()
        wf.set_hist_fire_seasons_FIRMS(TEST_FIRMS, land_path=TEST_LC)

        self.assertEqual(wf.tag.haz_type, 'WFseason')
        self.assertEqual(wf.units, 'K')
        self.assertTrue(np.allclose(wf.event_id, np.arange(1, 2)))
        self.assertTrue(np.allclose(wf.date, np.array([735964])))
        self.assertTrue(np.allclose(wf.orig, np.ones(1, bool)))
        self.assertEqual(wf.event_name, ['2016'])
        self.assertTrue(np.allclose(wf.frequency, np.ones(1)))
        self.assertEqual(wf.intensity.shape, (1, 14904))
        self.assertEqual(wf.fraction.shape, (1, 14904))
        self.assertEqual(wf.intensity[0, :].nonzero()[1][0], 20)
        self.assertEqual(wf.intensity[0, :].nonzero()[1][20], 346)
        self.assertEqual(wf.intensity[0, :].nonzero()[1][100], 1659)
        self.assertAlmostEqual(wf.intensity[0, 20], 326.6)
        self.assertAlmostEqual(wf.intensity[0, 346], 312.6)
        self.assertAlmostEqual(wf.intensity[0, 1668], 338.3)
        self.assertAlmostEqual(wf.intensity[0, 8000], 0.0)
        self.assertAlmostEqual(wf.fraction.max(), 1.0)
        self.assertAlmostEqual(wf.fraction.min(), 0.0)
        
    def test_from_hist_fire_season_firms_pass(self):
        """ Test set_hist_event_year_set """
        wf = WildFire()
        wf = wf.from_hist_fire_seasons_FIRMS(TEST_FIRMS, land_path=TEST_LC)

        self.assertEqual(wf.tag.haz_type, 'WFseason')
        self.assertEqual(wf.units, 'K')
        self.assertTrue(np.allclose(wf.event_id, np.arange(1, 2)))
        self.assertTrue(np.allclose(wf.date, np.array([735964])))
        self.assertTrue(np.allclose(wf.orig, np.ones(1, bool)))
        self.assertEqual(wf.event_name, ['2016'])
        self.assertTrue(np.allclose(wf.frequency, np.ones(1)))
        self.assertEqual(wf.intensity.shape, (1, 14904))
        self.assertEqual(wf.fraction.shape, (1, 14904))
        self.assertEqual(wf.intensity[0, :].nonzero()[1][0], 20)
        self.assertEqual(wf.intensity[0, :].nonzero()[1][20], 346)
        self.assertEqual(wf.intensity[0, :].nonzero()[1][100], 1659)
        self.assertAlmostEqual(wf.intensity[0, 20], 326.6)
        self.assertAlmostEqual(wf.intensity[0, 346], 312.6)
        self.assertAlmostEqual(wf.intensity[0, 1668], 338.3)
        self.assertAlmostEqual(wf.intensity[0, 8000], 0.0)
        self.assertAlmostEqual(wf.fraction.max(), 1.0)
        self.assertAlmostEqual(wf.fraction.min(), 0.0)

    def test_proba_fire_season_pass(self):
        """ Test probabilistic set_probabilistic_event_year_set """
        wf = WildFire()
        wf = WildFire.from_hist_fire_seasons_FIRMS(TEST_FIRMS, land_path=TEST_LC)
        wf.set_proba_fire_seasons(1,[3,4], land_path=TEST_LC,
                                  pop_path=TEST_POP, reproduce=True)

        self.assertEqual(wf.size, 2)
        orig = np.zeros(2, bool)
        orig[0] = True
        self.assertTrue(np.allclose(wf.orig, orig))
        self.assertEqual(len(wf.event_name), 2)
        self.assertEqual(wf.intensity.shape, (2, 14904))
        self.assertEqual(wf.fraction.shape, (2, 14904))
        self.assertEqual(wf.n_fires[1], 3)
        self.assertAlmostEqual(round(wf.centroids.fire_propa_matrix[1,20],2), 0.8)
        self.assertAlmostEqual(round(wf.centroids.fire_propa_matrix[10,30],2), 0.89)
        self.assertAlmostEqual(round(wf.centroids.ignition_weights_matrix[1,20],2), 4.74)
        self.assertAlmostEqual(round(wf.centroids.ignition_weights_matrix[10,30],2), 4.33)
        self.assertEqual(wf.intensity[1, :].nonzero()[1][0], 6246)
        self.assertEqual(wf.intensity[1, :].nonzero()[1][4], 6429)
        self.assertEqual(wf.intensity[0, :].nonzero()[1][11], 213)
        self.assertAlmostEqual(wf.intensity[1, 6246], 315.4)
        self.assertAlmostEqual(wf.intensity[1, 6429], 381.8)
        self.assertAlmostEqual(wf.intensity[0, 213], 339.4)
        self.assertAlmostEqual(wf.fraction.max(), 1.0)
        self.assertAlmostEqual(wf.fraction.min(), 0.0)

    def test_summarize_fires_to_seasons_pass(self):
        """ Test probabilistic set_probabilistic_event_year_set """
        wf = WildFire()
        wf = WildFire.from_hist_fire_FIRMS(TEST_FIRMS, land_path=TEST_LC)
        wf.summarize_fires_to_seasons()

        self.assertEqual(wf.tag.haz_type, 'WFseason')
        self.assertEqual(wf.units, 'K')
        self.assertTrue(np.allclose(wf.event_id, np.arange(1, 2)))
        self.assertTrue(np.allclose(wf.date, np.array([735964])))
        self.assertTrue(np.allclose(wf.orig, np.ones(1, bool)))
        self.assertEqual(wf.event_name, ['2016'])
        self.assertTrue(np.allclose(wf.frequency, np.ones(1)))
        self.assertEqual(wf.intensity.shape, (1, 14904))
        self.assertEqual(wf.fraction.shape, (1, 14904))
        self.assertEqual(wf.intensity[0, :].nonzero()[1][0], 20)
        self.assertEqual(wf.intensity[0, :].nonzero()[1][20], 346)
        self.assertEqual(wf.intensity[0, :].nonzero()[1][100], 1659)
        self.assertAlmostEqual(wf.intensity[0, 20], 326.6)
        self.assertAlmostEqual(wf.intensity[0, 346], 312.6)
        self.assertAlmostEqual(wf.intensity[0, 1668], 338.3)
        self.assertAlmostEqual(wf.intensity[0, 8000], 0.0)
        self.assertAlmostEqual(wf.fraction.max(), 1.0)
        self.assertAlmostEqual(wf.fraction.min(), 0.0)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestWildFire)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
