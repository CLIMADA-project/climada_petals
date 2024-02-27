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

from climada_petals.entity.impact_funcs import river_flood as fl


class TestIFRiverFlood(unittest.TestCase):
    """Impact function test"""
    def test_flood_imp_func_set(self):
        test_set = fl.flood_imp_func_set()
        self.assertTrue(np.array_equal(test_set.get_hazard_types(),
                        np.array(['RF'])))
        self.assertEqual(test_set.size(), 6)

    def test_region_Africa(self):

        impf_1 = fl.ImpfRiverFlood.from_jrc_region_sector("Africa")

        self.assertEqual(impf_1.continent, 'Africa')
        self.assertEqual(impf_1.name, 'Flood Africa JRC Residential noPAA')
        self.assertEqual(impf_1.haz_type, 'RF')
        self.assertEqual(impf_1.id, 11)
        self.assertEqual(impf_1.intensity_unit, 'm')
        self.assertTrue(np.array_equal(impf_1.intensity,
                        np.array([0., 0.05, 0.5, 1., 1.5,
                                  2., 3., 4., 5., 6., 12.])))
        self.assertTrue(np.allclose(impf_1.mdd,
                        np.array([0., 0., 0.2199, 0.3782,
                                  0.5306, 0.6356, 0.8169,
                                  0.9034, 0.9572, 1., 1.]), atol=1e-03))
        self.assertTrue(np.allclose(impf_1.paa,
                        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))

    def test_region_Asia(self):

        impf_2 = fl.ImpfRiverFlood.from_jrc_region_sector("Asia")

        self.assertEqual(impf_2.continent, 'Asia')
        self.assertEqual(impf_2.name, 'Flood Asia JRC Residential noPAA')
        self.assertEqual(impf_2.haz_type, 'RF')
        self.assertEqual(impf_2.id, 21)
        self.assertEqual(impf_2.intensity_unit, 'm')
        self.assertTrue(np.array_equal(impf_2.intensity,
                        np.array([0., 0.05, 0.5, 1., 1.5,
                                  2., 3., 4., 5., 6., 12.])))
        self.assertTrue(np.allclose(impf_2.mdd,
                        np.array([0.000, 0.000, 0.3266, 0.4941, 0.6166, 0.7207,
                                  0.8695, 0.9315, 0.9836, 1.0000, 1.0000]), atol=1e-03))
        self.assertTrue(np.allclose(impf_2.paa,
                        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))

    def test_region_Europe(self):

        impf_3 = fl.ImpfRiverFlood.from_jrc_region_sector("Europe")

        self.assertEqual(impf_3.continent, 'Europe')
        self.assertEqual(impf_3.name, 'Flood Europe JRC Residential noPAA')
        self.assertEqual(impf_3.haz_type, 'RF')
        self.assertEqual(impf_3.id, 31)
        self.assertEqual(impf_3.intensity_unit, 'm')
        self.assertTrue(np.array_equal(impf_3.intensity,
                        np.array([0., 0.05, 0.5, 1., 1.5,
                                  2., 3., 4., 5., 6., 12.])))
        self.assertTrue(np.allclose(impf_3.mdd,
                        np.array([0.00, 0.00, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85,
                                  0.95, 1.00, 1.00]), atol=1e-03))
        self.assertTrue(np.allclose(impf_3.paa,
                        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))

    def test_region_NorthAmerica(self):

        impf_4 = fl.ImpfRiverFlood.from_jrc_region_sector("North America")

        self.assertEqual(impf_4.continent, 'North America')
        self.assertEqual(impf_4.name,
                         'Flood North America JRC Residential noPAA')
        self.assertEqual(impf_4.haz_type, 'RF')
        self.assertEqual(impf_4.id, 41)
        self.assertEqual(impf_4.intensity_unit, 'm')
        self.assertTrue(np.array_equal(impf_4.intensity,
                        np.array([0., 0.05, 0.5, 1., 1.5, 2., 3., 4., 5.,
                                  6., 12.])))
        self.assertTrue(np.allclose(impf_4.mdd,
                        np.array([0.0000, 0.2018, 0.4433, 0.5828, 0.6825,
                                  0.7840, 0.8543, 0.9237, 0.9585, 1.0000,
                                  1.0000]), atol=1e-03))
        self.assertTrue(np.allclose(impf_4.paa,
                        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))

    def test_region_Oceania(self):

        impf_5 = fl.ImpfRiverFlood.from_jrc_region_sector("Oceania")

        self.assertEqual(impf_5.continent, 'Oceania')
        self.assertEqual(impf_5.name, 'Flood Oceania JRC Residential noPAA')
        self.assertEqual(impf_5.haz_type, 'RF')
        self.assertEqual(impf_5.id, 51)
        self.assertEqual(impf_5.intensity_unit, 'm')
        self.assertTrue(np.array_equal(impf_5.intensity,
                        np.array([0., 0.05, 0.5, 1., 1.5,
                                  2., 3., 4., 5., 6., 12.])))
        self.assertTrue(np.allclose(impf_5.mdd,
                        np.array([0.00, 0.00, 0.475, 0.64, 0.715, 0.788, 0.929, 0.967,
                                  0.983, 1.00, 1.00]), atol=1e-03))
        self.assertTrue(np.allclose(impf_5.paa,
                        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))

    def test_region_SouthAmerica(self):

        impf_6 = fl.ImpfRiverFlood.from_jrc_region_sector("South America")
        self.assertEqual(impf_6.continent, 'South America')
        self.assertEqual(impf_6.name,
                         'Flood South America JRC Residential noPAA')
        self.assertEqual(impf_6.haz_type, 'RF')
        self.assertEqual(impf_6.id, 61)
        self.assertEqual(impf_6.intensity_unit, 'm')
        self.assertTrue(np.array_equal(impf_6.intensity,
                        np.array([0., 0.05, 0.5, 1., 1.5,
                                  2., 3., 4., 5., 6., 12.])))
        self.assertTrue(np.allclose(impf_6.mdd,
                        np.array([0.0000, 0.0000, 0.4908, 0.7112, 0.8420, 0.9494,
                                  0.9836, 1.0000, 1.0000, 1.0000, 1.0000]), atol=1e-03))
        self.assertTrue(np.allclose(impf_6.paa,
                        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestIFRiverFlood)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
