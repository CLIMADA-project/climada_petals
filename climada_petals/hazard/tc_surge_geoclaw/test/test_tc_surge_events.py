"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test tc_surge_events module
"""

import unittest

import numpy as np
import xarray as xr

from climada_petals.hazard.tc_surge_geoclaw.tc_surge_events import (
    _boxcover_points_along_axis,
    _round_bounds_enlarge,
    TCSurgeEvents,
)


class TestSurgeEvents(unittest.TestCase):
    """Test main TCSurgeEvents class and helper functions"""

    def test_boxcover(self):
        """Test boxcovering function"""
        nsplits = 4
        # sorted list of 1d-points
        points = np.array([-3., -1.3, 1.5, 1.7, 4.6, 5.4, 6.2, 6.8, 7.])
        # shuffle list of points
        points = points[[4, 7, 3, 1, 2, 5, 8, 1, 6, 0]].reshape(-1, 1)
        # this is easy to see from the sorted list of points
        boxes_correct = [(-3.0, -1.3), (1.5, 1.7), (4.6, 7.0)]
        boxes, size = _boxcover_points_along_axis(points, nsplits)
        self.assertEqual(boxes, boxes_correct)
        self.assertEqual(size, sum(b[1] - b[0] for b in boxes))

        nsplits = 3
        points = np.array([
            [0.0, 0.2], [1.3, 0.1], [2.5, 0.0],
            [3.0, 1.5], [0.2, 1.2],
            [0.4, 2.3], [0.5, 3.0],
        ])
        boxes_correct = [
            (0.0, 0.0, 2.5, 0.2),
            (0.2, 1.2, 3.0, 1.5),
            (0.4, 2.3, 0.5, 3.0),
        ]
        boxes, size = _boxcover_points_along_axis(points, nsplits)
        self.assertEqual(boxes, boxes_correct)
        self.assertEqual(size, sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes))

        # exchange x and y coordinate (order of dimensions should not matter)
        boxes, size = _boxcover_points_along_axis(points[:, ::-1], nsplits)
        self.assertEqual(boxes, [(b[1], b[0], b[3], b[2]) for b in boxes_correct])


    def test_round_bounds(self):
        """Test bounds rounding function"""
        bounds = (14.3, -0.3, 29.0, 4.99)
        np.testing.assert_allclose(
            _round_bounds_enlarge(*bounds, precision=5), (10, -5, 30, 5),
        )
        np.testing.assert_allclose(
            _round_bounds_enlarge(*bounds, precision=1), (14, -1, 29, 5),
        )
        np.testing.assert_allclose(
            _round_bounds_enlarge(*bounds, precision=0.2), (14.2, -0.4, 29, 5),
        )


    def test_surge_events(self):
        """Test TCSurgeEvents object"""
        # Artificial track with two "landfall" events
        radii = np.array([40, 40, 40, 30, 30, 30, 40, 30, 30, 30,
                          40, 40, 40, 30, 30, 30, 30, 30, 40, 40])
        track = xr.Dataset({
            'radius_max_wind': ('time', radii),
            'radius_oci': ('time', 4.1 * radii),
            'max_sustained_wind': ('time', [10, 10, 40, 40, 40, 40, 10, 40, 40, 40,
                                            10, 10, 10, 40, 40, 40, 40, 40, 10, 10]),
            'central_pressure': ('time', np.full((20,), 970)),
            'time_step': ('time', np.full((20,), 6, dtype=np.float64)),
        }, coords={
            'time': np.arange(
                '2000-01-01', '2000-01-06', np.timedelta64(6, 'h'), dtype='datetime64[ns]',
            ),
            'lat': ('time', np.linspace(5, 30, 20)),
            'lon': ('time', np.zeros(20)),
        })

        # centroids clearly too far away from track
        centroids = np.array([ar.ravel() for ar in np.meshgrid(np.linspace(-20, -10, 10),
                                                               np.linspace(50, 80, 10))]).T
        s_events = TCSurgeEvents(track, centroids)
        self.assertEqual(s_events.nevents, 0)

        # one comapct set of centroids in the middle of the track
        centroids = np.array([ar.ravel() for ar in np.meshgrid(np.linspace(15, 17, 100),
                                                               np.linspace(-2, 2, 100))]).T
        s_events = TCSurgeEvents(track, centroids)
        self.assertEqual(s_events.nevents, 1)
        np.testing.assert_array_equal(s_events.time_mask_buffered[0][s_events.time_mask[0]], True)
        np.testing.assert_array_equal(s_events.time_mask[0][:7], False)
        np.testing.assert_array_equal(s_events.time_mask[0][7:10], True)
        np.testing.assert_array_equal(s_events.time_mask[0][10:], False)

        # three sets of centroids
        centroids = np.concatenate([
            # first half and close to the track
            [ar.ravel() for ar in np.meshgrid(np.linspace(6, 8, 50), np.linspace(-2, -0.5, 50))],
            # second half on both sides of the track
            [ar.ravel() for ar in np.meshgrid(np.linspace(19, 22, 50), np.linspace(0.5, 1.5, 50))],
            [ar.ravel() for ar in np.meshgrid(np.linspace(25, 26, 50), np.linspace(-1.0, 0.3, 50))],
            # at the end, where storm is too weak to create surge
            [ar.ravel() for ar in np.meshgrid(np.linspace(29, 32, 50), np.linspace(0, 1, 50))],
        ], axis=1).T
        s_events = TCSurgeEvents(track, centroids)
        self.assertEqual(s_events.nevents, 2)
        self.assertEqual(len(list(s_events)), 2)
        for i in range(2):
            np.testing.assert_array_equal(
                s_events.time_mask_buffered[i][s_events.time_mask[i]], True)
        np.testing.assert_array_equal(s_events.time_mask_buffered[0][:6], True)
        np.testing.assert_array_equal(s_events.time_mask_buffered[0][6:], False)
        np.testing.assert_array_equal(s_events.time_mask_buffered[1][:11], False)
        np.testing.assert_array_equal(s_events.time_mask_buffered[1][11:19], True)
        np.testing.assert_array_equal(s_events.time_mask_buffered[1][19:], False)
        # for the double set in second half, it's advantageous to split surge area in two:
        self.assertEqual(len(s_events.surge_areas[1]), 2)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSurgeEvents)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
