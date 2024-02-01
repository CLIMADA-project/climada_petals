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

Test sea_level_funs module
"""

import unittest

import numpy as np

from climada.util.api_client import Client
from climada_petals.hazard.tc_surge_geoclaw.sea_level_funs import (
    area_sea_level_from_monthly_nc,
    sea_level_from_nc,
)


def test_altimetry_nc():
    """Altimetry (ocean surface) raster data for testing

    Sample of monthly Copernicus satellite altimetry for year 2010.
    """
    client = Client()
    _, [altimetry_nc] = client.download_dataset(
        client.get_dataset_info(name='test_altimetry_tubuai', status='test_dataset')
    )
    return altimetry_nc


class TestSeaLevelFuns(unittest.TestCase):
    """Test sea level function factories"""

    def test_load_sea_level(self):
        """Test functions to get sea level from NetCDF files"""
        zos_path = test_altimetry_nc()
        periods = [
            # one period in January, one in February, and one close to Jan/Feb boundary
            (np.datetime64("2010-01-10"), np.datetime64("2010-01-14")),
            (np.datetime64("2010-02-08"), np.datetime64("2010-02-20")),
            (np.datetime64("2010-01-20"), np.datetime64("2010-01-28")),
        ]
        bounds = [
            # each region contains the previous (stacked)
            (-153.62, -28.79, -144.75, -18.44),
            (-153, -20, -150, -19),
            (-152, -28.5, -145, -27.5),
        ]

        sea_level = []
        sea_level_fun = area_sea_level_from_monthly_nc(zos_path)
        for per in periods:
            for bnd in bounds:
                sea_level.append(sea_level_fun(bnd, per))
        sea_level = np.array(sea_level).reshape(len(periods), len(bounds))

        # for the period at the month boundary, the mean of Jan and Feb should be used
        np.testing.assert_allclose(sea_level[:-1].mean(axis=0), sea_level[-1])

        # larger regions have larger maximum
        np.testing.assert_array_less(sea_level[:, 1], sea_level[:, 0])
        np.testing.assert_array_less(sea_level[:, 2], sea_level[:, 0])

        # check some individual values
        self.assertAlmostEqual(sea_level[0, 0], 1.752, places=3)
        self.assertAlmostEqual(sea_level[1, 0], 1.839, places=3)
        self.assertAlmostEqual(sea_level[2, 0], 1.795, places=3)
        np.testing.assert_allclose(sea_level[:, 1], 1.39, atol=1e-2)
        np.testing.assert_array_less(sea_level[:, 2], 1.29)
        np.testing.assert_array_less(1.25, sea_level[:, 2])

        step = 0.25
        bounds = [
            # three areas for which the same grid cell is selected:
            (-153.25, -20, -150, -19.25),
            (-152.25, -20.5, -151, -18.75),
            (-153.25, -20 + 0.3 * step, -150, -19.25 + 0.2 * step),
            # neighboring grid cell is selected:
            (-153.25, -20 + step, -150, -19.25 + step),
        ]

        sea_level = []
        sea_level_fun = sea_level_from_nc(zos_path, t_pad=np.timedelta64(4, "D"))
        for per in periods:
            for bnd in bounds:
                sea_level.append(sea_level_fun(bnd, per))
        sea_level = np.array(sea_level).reshape(len(periods), len(bounds))

        # for the period at the month boundary, the mean of Jan and Feb should be used
        np.testing.assert_allclose(sea_level[:-1].mean(axis=0), sea_level[-1])

        # same grid cell, same value:
        np.testing.assert_array_equal(sea_level[:, 0], sea_level[:, 1])
        np.testing.assert_array_equal(sea_level[:, 0], sea_level[:, 2])

        # check some individual values
        self.assertAlmostEqual(sea_level[0, 0], 1.102, places=3)
        self.assertAlmostEqual(sea_level[1, 0], 1.070, places=3)
        self.assertAlmostEqual(sea_level[2, 0], 1.086, places=3)

        # for the neighboring grid cell, sea level should be different, but not too different:
        dist = np.abs(sea_level[:, 0] - sea_level[:, 3])
        np.testing.assert_array_less(0, dist)
        np.testing.assert_array_less(dist, 0.4)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSeaLevelFuns)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
