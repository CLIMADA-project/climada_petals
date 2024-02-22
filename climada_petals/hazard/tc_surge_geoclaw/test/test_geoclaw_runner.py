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

Test geoclaw_runner module
"""

import datetime as dt
import unittest

import numpy as np

from climada.util.api_client import Client
from climada_petals.hazard.tc_surge_geoclaw.geoclaw_runner import (
    _bounds_to_str,
    _dt64_to_pydt,
    _load_topography,
)


def test_bathymetry_tif():
    """Topo-Bathymetry (combined land surface and ocean floor) raster data for testing

    SRTM15+V2.3 data of Tubuai island enlarged by factor 10.
    """
    client = Client()
    _, [bathymetry_tif] = client.download_dataset(
        client.get_dataset_info(name='test_bathymetry_tubuaix10', status='test_dataset')
    )
    return bathymetry_tif


class TestFuncs(unittest.TestCase):
    """Test helper functions"""

    def test_bounds_to_str(self):
        """Test conversion from lon-lat-bounds tuple to lat-lon string"""
        bounds_str = [
            [(-4.2, 1.0, -3.05, 2.125), '1N-2.125N_4.2W-3.05W'],
            [(106.9, -7, 111.6875, 25.1), '7S-25.1N_106.9E-111.7E'],
            [(-6.9, -7.8334, 11, 25.1), '7.833S-25.1N_6.9W-11E'],
        ]
        for bounds, string in bounds_str:
            str_out = _bounds_to_str(bounds)
            self.assertEqual(str_out, string)


    def test_dt64_to_pydt(self):
        """Test conversion from datetime64 to python datetime objects"""
        # generate test data
        dt64 = np.array([
            '1865-03-07T20:41:02.000000',
            '2008-02-29T00:05:30.000000',
            '2013-12-02T00:00:00.000000',
        ], dtype='datetime64[us]')
        pydt = [
            dt.datetime(1865, 3, 7, 20, 41, 2),
            dt.datetime(2008, 2, 29, 0, 5, 30),
            dt.datetime(2013, 12, 2),
        ]

        # test conversion of numpy array of dates
        pydt_conv = _dt64_to_pydt(dt64)
        self.assertIsInstance(pydt_conv, list)
        self.assertEqual(len(pydt_conv), dt64.size)
        self.assertEqual(pydt_conv, pydt)

        # test conversion of single object
        pydt_conv = _dt64_to_pydt(dt64[2])
        self.assertEqual(pydt_conv, pydt[2])


    def test_load_topography(self):
        """Test _load_topography function"""
        topo_path = test_bathymetry_tif()
        resolutions = [15, 30, 41, 90, 300]
        bounds = [
            (-153.62, -28.79, -144.75, -18.44),
            (-153, -20, -150, -19),
            (-152, -28.5, -145, -27.5),
            (-150.0, -23.3, -149.6, -23.0)
        ]
        zvalues = []
        for res_as in resolutions:
            for bnd in bounds:
                topo_bounds, topo = _load_topography(topo_path, bnd, res_as)
                self.assertLessEqual(topo_bounds[0], bnd[0])
                self.assertLessEqual(topo_bounds[1], bnd[1])
                self.assertGreaterEqual(topo_bounds[2], bnd[2])
                self.assertGreaterEqual(topo_bounds[3], bnd[3])
                xcoords, ycoords = topo.x, topo.y
                np.testing.assert_array_equal(
                    True, (xcoords >= topo_bounds[0]) & (xcoords <= topo_bounds[2])
                )
                np.testing.assert_array_equal(
                    True, (ycoords >= topo_bounds[1]) & (ycoords <= topo_bounds[3])
                )
                zvalues.append(topo.Z)

            # all but last row are positive
            # this also checks that the orientation is correct
            np.testing.assert_array_less(0, zvalues[-1][:-1,:])
            self.assertLess(zvalues[-1].min(), 0)
            self.assertLess(150, zvalues[-1].max())

            # regions off shore:
            np.testing.assert_array_less(zvalues[-2], -1000)
            np.testing.assert_array_less(zvalues[-3], -1000)

            # sanity check: size of grid is increasing with resolution
            if len(zvalues) > 4:
                self.assertLess(zvalues[-1].size, zvalues[-5].size)
                self.assertLess(zvalues[-2].size, zvalues[-6].size)
                self.assertLess(zvalues[-3].size, zvalues[-7].size)
                self.assertLess(zvalues[-4].size, zvalues[-8].size)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFuncs)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
