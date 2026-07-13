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

Test IFFlood class.
"""

import unittest
import numpy as np
from itertools import product
from climada_petals.entity.impact_funcs import river_flood as fl

IMPF_REGIONS = {1: 'Africa', 2: 'Asia', 3: 'Europe', 4: 'North America', 5: 'Oceania', 6: 'South America'}
IMPF_SECTORS = {1: 'residential', 2: 'commercial', 3: 'industrial', 4: 'transport', 5: 'infrastructure',
                6: 'agriculture'}

IMPF_MDD = {
    11: np.array([0., 0., 0.22, 0.378, 0.531, 0.636, 0.817, 0.903, 0.957, 1., 1.]),
    21: np.array([0., 0., 0.327, 0.494, 0.617, 0.721, 0.87, 0.931, 0.984, 1., 1.]),
    31: np.array([0., 0., 0.25, 0.4, 0.5, 0.6, 0.75, 0.85, 0.95, 1., 1.]),
    41: np.array([0., 0.202, 0.443, 0.583, 0.683, 0.784, 0.854, 0.924, 0.959, 1., 1.]),
    51: np.array([0., 0., 0.475, 0.640, 0.715, 0.788, 0.929, 0.967, 0.983, 1., 1.]),
    61: np.array([0., 0., 0.491, 0.711, 0.842, 0.949, 0.984, 1., 1., 1., 1.]),
    12: np.array([]),
    22: np.array([0., 0., 0.377, 0.538, 0.659, 0.763, 0.883, 0.942, 0.981, 1., 1.]),
    32: np.array([0., 0., 0.15, 0.3, 0.45, 0.55, 0.75, 0.9, 1., 1., 1.]),
    42: np.array([0., 0.018, 0.239, 0.374, 0.466, 0.552, 0.687, 0.822, 0.908, 1., 1.]),
    52: np.array([0., 0., 0.239, 0.481, 0.674, 0.865, 1., 1., 1., 1., 1.]),
    62: np.array([0., 0., 0.611, 0.84 , 0.924, 0.992, 1., 1., 1., 1., 1.]),
    13: np.array([0., 0., 0.063, 0.247, 0.403, 0.494, 0.685, 0.919, 1., 1., 1.]),
    23: np.array([0., 0., 0.283, 0.482, 0.629, 0.717, 0.857, 0.909, 0.955, 1., 1.]),
    33: np.array([0., 0., 0.15, 0.27, 0.4, 0.52, 0.7, 0.85, 1., 1., 1.]),
    43: np.array([0., 0.026, 0.323, 0.511, 0.637, 0.74, 0.86, 0.937, 0.98, 1., 1.]),
    53: np.array([]),
    63: np.array([0., 0., 0.667, 0.889, 0.947, 1., 1., 1., 1., 1., 1.]),
    14: np.array([]),
    24: np.array([0., 0., 0.358, 0.572, 0.733, 0.847, 1., 1., 1., 1., 1.]),
    34: np.array([0., 0., 0.317, 0.542, 0.702, 0.832, 1., 1., 1., 1., 1.]),
    44: np.array([]),
    54: np.array([]),
    64: np.array([0., 0., 0.088, 0.175, 0.596, 0.842, 1., 1., 1., 1., 1.]),
    15: np.array([]),
    25: np.array([0., 0., 0.214, 0.373, 0.604, 0.71 , 0.808, 0.887, 0.969, 1., 1.]),
    35: np.array([0., 0., 0.25, 0.42, 0.55, 0.65, 0.8, 0.9, 1., 1., 1.]),
    45: np.array([]),
    55: np.array([]),
    65: np.array([]),
    16: np.array([0., 0., 0.243, 0.472, 0.741, 0.917, 1., 1., 1., 1., 1.]),
    26: np.array([0., 0., 0.135, 0.37 , 0.524, 0.558, 0.66, 0.834, 0.988, 1., 1.]),
    36: np.array([0., 0., 0.3, 0.55, 0.65, 0.75, 0.85, 0.95, 1., 1., 1.]),
    46: np.array([0., 0.019, 0.268, 0.474, 0.551, 0.602, 0.76, 0.874, 0.954, 1., 1.]),
    56: np.array([]),
    66: np.array([])
}

class TestIFRiverFlood(unittest.TestCase):
    """Impact function test"""
    def test_flood_imp_func_set(self):
        test_set = fl.flood_imp_func_set()
        self.assertTrue(np.array_equal(test_set.get_hazard_types(),
                        np.array(['RF'])))
        self.assertEqual(test_set.size(), 6)

    def test_flood_imp_func_parameters(self):
        for reg_id, sec_id in product(range(1,7), range(1,7)):
            region, sector = IMPF_REGIONS[reg_id], IMPF_SECTORS[sec_id]
            impf_id = int(f"{reg_id}{sec_id}")
            impf_mdd = IMPF_MDD[impf_id]

            if impf_mdd.size == 0:
                with self.assertRaises(ValueError):
                    fl.ImpfRiverFlood.from_jrc_region_sector(region, sector)
                continue

            impf = fl.ImpfRiverFlood.from_jrc_region_sector(region, sector)
            self.assertEqual(impf.continent, region)
            self.assertEqual(
                impf.name, f'Flood {region} JRC {sector.capitalize()} noPAA'
                )
            self.assertEqual(impf.haz_type, 'RF')
            self.assertEqual(impf.intensity_unit, 'm')

            self.assertEqual(impf.id, impf_id)
            np.testing.assert_array_almost_equal(
                impf.intensity,
                np.array([0., 0.05, 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])
                )
            np.testing.assert_array_almost_equal(
                impf.mdd,
                impf_mdd
                )
            np.testing.assert_array_almost_equal(
                impf.paa,
                np.ones_like(impf.intensity)
                )

    def test_flood_imp_func_invalid_inputs(self):
        with self.assertRaises(ValueError):
            fl.ImpfRiverFlood.from_jrc_region_sector('unknown country', 'residential')
        with self.assertRaises(ValueError):
            fl.ImpfRiverFlood.from_jrc_region_sector('Africa', 'unknown sector')

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestIFRiverFlood)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
