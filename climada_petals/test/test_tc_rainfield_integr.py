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

import numpy as np

from climada import CONFIG
from climada.hazard import TCTracks
from climada.util.api_client import Client

from climada_petals.hazard.tc_rainfield import TCRain


def getTestData():
    client = Client()
    centr_ds = client.get_dataset_info(name='tc_rainfield_test', status='test_dataset')
    _, [centr_test_mat, track, track_short, haz_hdf5] = client.download_dataset(centr_ds)
    return track


TEST_TRACK = getTestData()


class TestModel(unittest.TestCase):
    """Test modelling of rainfall"""

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


if __name__ == "__main__":
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModel))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
