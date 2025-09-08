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

Test TC forecast module.
"""

import unittest
import numpy as np
import xarray as xr
from scipy import sparse

from climada import CONFIG
from climada.hazard import TropCyclone, Centroids
from climada_petals.hazard.tc_tracks_forecast import TCForecast

DATA_DIR = CONFIG.test_data.dir()
TEST_BUFR_FILE = DATA_DIR.joinpath('tracks_21W_ELOISE_2021091000_eps_bufr4.bin')

N_ENS_TRACKS = 52

"""TC tracks in four BUFR formats as provided by ECMWF. Sourced from
https://confluence.ecmwf.int/display/FCST/New+Tropical+Cyclone+Wind+Radii+product
"""

TEST_CENTROIDS = Centroids.from_pnt_bounds([118, 21, 124, 26], 0.125)

class TestTCForecast(unittest.TestCase):
    """Integration test for the TCForecast module"""

    def test_data_extraction_from_ecmwf(self):
        """Test realtime TC tracks data retrieval from ECMWF"""
        tr_forecast = TCForecast()
        tr_forecast.fetch_ecmwf()

        # Test data format for each track
        self.assertIsInstance(tr_forecast.data[0], xr.Dataset)

        # Test data format for coordinates
        self.assertIsInstance(tr_forecast.data[0]['lat'].values[0], np.float64)
        self.assertIsInstance(tr_forecast.data[1]['lon'].values[0], np.float64)
        self.assertIsInstance(tr_forecast.data[0]['time'].values[0], np.datetime64)

        # Test data format for data variables
        self.assertIsInstance(tr_forecast.data[1]['max_sustained_wind'].values[0], np.float64)
        self.assertIsInstance(tr_forecast.data[0]['central_pressure'].values[0], np.float64)
        self.assertIsInstance(tr_forecast.data[0]['time_step'].values[0], np.float64)
        self.assertIsInstance(tr_forecast.data[2]['radius_max_wind'].values[0], np.float64)
        self.assertIsInstance(tr_forecast.data[1]['environmental_pressure'].values[0], np.float64)
        self.assertIsInstance(tr_forecast.data[0]['basin'].values[0], str)

        # Test data format for attributes
        self.assertEqual(tr_forecast.data[0].max_sustained_wind_unit, 'm/s')
        self.assertEqual(tr_forecast.data[1].central_pressure_unit, 'mb')
        self.assertIsInstance(tr_forecast.data[0].name, str)
        self.assertIsInstance(tr_forecast.data[0].sid, str)
        self.assertIsInstance(tr_forecast.data[1].orig_event_flag, bool)
        self.assertEqual(tr_forecast.data[1].data_provider, 'ECMWF')
        self.assertIsInstance(tr_forecast.data[0].id_no, float)
        self.assertIsInstance(tr_forecast.data[0].ensemble_number, np.integer)
        self.assertIsInstance(tr_forecast.data[1].is_ensemble, np.bool_)
        self.assertIsInstance(tr_forecast.data[0].run_datetime, np.datetime64)
        self.assertIsInstance(tr_forecast.data[0].category, str)

    def test_compute_TC_windfield_from_ecmwf(self):
        """Test computation of TC windfield from ECMWF tracks"""
        tr_forecast = TCForecast()
        tr_forecast.fetch_ecmwf(TEST_BUFR_FILE)
        # compute TC windfield
        tc_forecast = TropCyclone.from_tracks(tr_forecast, centroids=TEST_CENTROIDS)

        self.assertEqual(tc_forecast.haz_type, 'TC')
        self.assertEqual(tc_forecast.units, 'm/s')
        self.assertEqual(tc_forecast.centroids.size, 2009)
        self.assertEqual(tc_forecast.event_id.size, 52)
        self.assertEqual(tc_forecast.event_id[0], 1)
        self.assertEqual(tc_forecast.event_name[1], '21W')
        self.assertTrue(np.array_equal(tc_forecast.frequency[0], 1.))
        self.assertTrue(np.array_equal(tc_forecast.orig[0], False))
        self.assertTrue(isinstance(tc_forecast.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_forecast.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_forecast.intensity.shape, (52, 2009))
        self.assertEqual(tc_forecast.fraction.shape, (52, 2009))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestTCForecast)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
