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

Unit Tests on GDP2Asset exposures.
"""
import numpy as np
import unittest
import pandas as pd
from climada_petals.entity.exposures import gdp_asset as ga
from climada.util.constants import RIVER_FLOOD_REGIONS_CSV
from climada_petals.util.constants import DEMO_GDP2ASSET


class TestGDP2AssetClass(unittest.TestCase):
    """Unit tests for the LitPop exposure class"""
    def test_wrong_iso3_fail(self):
        """Wrong ISO3 code"""
        testGDP2A = ga.GDP2Asset()

        with self.assertRaises(NameError):
            testGDP2A.set_countries(countries=['CHE'], ref_year=2000)
        with self.assertRaises(NameError):
            testGDP2A.set_countries(countries=['CHE'], ref_year=2000,
                                    path='non/existent/test')
        with self.assertRaises(LookupError):
            testGDP2A.set_countries(countries=['OYY'], path=DEMO_GDP2ASSET)
        with self.assertRaises(LookupError):
            testGDP2A.set_countries(countries=['DEU'], ref_year=2600,
                                    path=DEMO_GDP2ASSET)
        with self.assertRaises(ValueError):
            testGDP2A.set_countries(path=DEMO_GDP2ASSET)


class TestGDP2AssetFunctions(unittest.TestCase):
    """Test LitPop Class methods"""

    def test_set_one_country(self):
        with self.assertRaises(KeyError):
            ga.GDP2Asset._set_one_country('LIE', 2001, path=DEMO_GDP2ASSET)

        exp_test = ga.GDP2Asset._set_one_country('LIE', 2000, path=DEMO_GDP2ASSET)

        np.testing.assert_allclose(exp_test.latitude, np.array(
            [47.0622474, 47.0622474, 47.0622474, 47.103914, 47.103914, 47.103914, 47.1455806,
             47.1455806, 47.1455806, 47.1872472, 47.1872472, 47.2289138, 47.2289138]
        ))
        np.testing.assert_allclose(exp_test.longitude, np.array(
            [9.5206968, 9.5623634, 9.60403, 9.5206968, 9.5623634, 9.60403, 9.5206968,
             9.5623634, 9.60403, 9.5206968, 9.5623634, 9.5206968, 9.5623634]
        ))
        
        np.testing.assert_allclose(exp_test.value, np.array(
            [174032107.65846416, 20386409.991937194, 2465206.6989314994,
             0.0, 12003959.733058406, 97119771.42771776,
             0.0, 4137081.3646739507, 27411196.308422357,
             0.0, 4125847.312198318, 88557558.43543366, 191881403.05181965]
        ))

        self.assertTrue((exp_test.gdf.impf_RF == 3).all())
        self.assertTrue((exp_test.gdf.region_id ==11).all())

    def test_fast_impf_mapping(self):

        testIDs = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)
        self.assertAlmostEqual(ga._fast_impf_mapping(36, testIDs)[0], 11.0)
        self.assertAlmostEqual(ga._fast_impf_mapping(36, testIDs)[1], 3.0)

        self.assertAlmostEqual(ga._fast_impf_mapping(118, testIDs)[0], 11.0)
        self.assertAlmostEqual(ga._fast_impf_mapping(118, testIDs)[1], 3.0)

        self.assertAlmostEqual(ga._fast_impf_mapping(124, testIDs)[0], 0.0)
        self.assertAlmostEqual(ga._fast_impf_mapping(124, testIDs)[1], 2.0)

    def test_read_GDP(self):

        exp_test = ga.GDP2Asset._set_one_country('LIE', 2000, DEMO_GDP2ASSET)
        coordinates = np.zeros((exp_test.gdf.shape[0], 2))
        coordinates[:, 0] = exp_test.latitude
        coordinates[:, 1] = exp_test.longitude

        with self.assertRaises(KeyError):
            ga._read_GDP(coordinates, ref_year=2600, path=DEMO_GDP2ASSET)

        testAssets = ga._read_GDP(coordinates, ref_year=2000,
                                  path=DEMO_GDP2ASSET)

        self.assertAlmostEqual(testAssets[0], 174032107.65846416)
        self.assertAlmostEqual(testAssets[1], 20386409.991937194)
        self.assertAlmostEqual(testAssets[2], 2465206.6989314994)
        self.assertAlmostEqual(testAssets[3], 0.0)
        self.assertAlmostEqual(testAssets[4], 12003959.733058406)
        self.assertAlmostEqual(testAssets[5], 97119771.42771776)
        self.assertAlmostEqual(testAssets[6], 0.0)
        self.assertAlmostEqual(testAssets[7], 4137081.3646739507)
        self.assertAlmostEqual(testAssets[8], 27411196.308422357)
        self.assertAlmostEqual(testAssets[9], 0.0)
        self.assertAlmostEqual(testAssets[10], 4125847.312198318)
        self.assertAlmostEqual(testAssets[11], 88557558.43543366)
        self.assertAlmostEqual(testAssets[12], 191881403.05181965)



if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestGDP2AssetFunctions)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(
            TestGDP2AssetClass))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
