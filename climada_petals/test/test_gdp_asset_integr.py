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

Tests on GDP2Asset.
"""
import unittest
from climada_petals.entity.exposures import gdp_asset as ga
from climada_petals.util.constants import DEMO_GDP2ASSET

class TestGDP2AssetClassCountries(unittest.TestCase):
    """Unit tests for the GDP2Asset exposure class"""
    def test_wrong_iso3_fail(self):
        """Wrong ISO3 code"""
        testGDP2A = ga.GDP2Asset()

        with self.assertRaises(LookupError):
            testGDP2A.set_countries(countries=['OYY'], path=DEMO_GDP2ASSET)
        with self.assertRaises(LookupError):
            testGDP2A.set_countries(countries=['DEU'], ref_year=2600, path=DEMO_GDP2ASSET)
        with self.assertRaises(LookupError):
            testGDP2A.set_countries(countries=['DEU'], ref_year=2600, path=DEMO_GDP2ASSET)
        with self.assertRaises(ValueError):
            testGDP2A.set_countries(path=DEMO_GDP2ASSET)
        with self.assertRaises(IOError):
            testGDP2A.set_countries(countries=['MEX'], path=DEMO_GDP2ASSET)

    def test_one_set_countries(self):
        testGDP2A_LIE = ga.GDP2Asset()
        testGDP2A_LIE.set_countries(countries=['LIE'], path=DEMO_GDP2ASSET)

        self.assertAlmostEqual(testGDP2A_LIE.longitude[0], 9.5206968)
        self.assertAlmostEqual(testGDP2A_LIE.longitude[1], 9.5623634)
        self.assertAlmostEqual(testGDP2A_LIE.longitude[2], 9.60403)
        self.assertAlmostEqual(testGDP2A_LIE.longitude[3], 9.5206968)
        self.assertAlmostEqual(testGDP2A_LIE.longitude[4], 9.5623634)
        self.assertAlmostEqual(testGDP2A_LIE.longitude[5], 9.60403)
        self.assertAlmostEqual(testGDP2A_LIE.longitude[6], 9.5206968)
        self.assertAlmostEqual(testGDP2A_LIE.longitude[7], 9.5623634)
        self.assertAlmostEqual(testGDP2A_LIE.longitude[8], 9.60403)
        self.assertAlmostEqual(testGDP2A_LIE.longitude[9], 9.5206968)
        self.assertAlmostEqual(testGDP2A_LIE.longitude[10], 9.5623634)
        self.assertAlmostEqual(testGDP2A_LIE.longitude[11], 9.5206968)
        self.assertAlmostEqual(testGDP2A_LIE.longitude[12], 9.5623634)

        self.assertAlmostEqual(testGDP2A_LIE.latitude[0], 47.0622474)
        self.assertAlmostEqual(testGDP2A_LIE.latitude[1], 47.0622474)
        self.assertAlmostEqual(testGDP2A_LIE.latitude[2], 47.0622474)
        self.assertAlmostEqual(testGDP2A_LIE.latitude[3], 47.103914)
        self.assertAlmostEqual(testGDP2A_LIE.latitude[4], 47.103914)
        self.assertAlmostEqual(testGDP2A_LIE.latitude[5], 47.103914)
        self.assertAlmostEqual(testGDP2A_LIE.latitude[6], 47.1455806)
        self.assertAlmostEqual(testGDP2A_LIE.latitude[7], 47.1455806)
        self.assertAlmostEqual(testGDP2A_LIE.latitude[8], 47.1455806)
        self.assertAlmostEqual(testGDP2A_LIE.latitude[9], 47.1872472)
        self.assertAlmostEqual(testGDP2A_LIE.latitude[10], 47.1872472)
        self.assertAlmostEqual(testGDP2A_LIE.latitude[11], 47.2289138)
        self.assertAlmostEqual(testGDP2A_LIE.latitude[12], 47.2289138)

        self.assertAlmostEqual(testGDP2A_LIE.value[0], 174032107.65846416)
        self.assertAlmostEqual(testGDP2A_LIE.value[1], 20386409.991937194)
        self.assertAlmostEqual(testGDP2A_LIE.value[2], 2465206.6989314994)
        self.assertAlmostEqual(testGDP2A_LIE.value[3], 0.0)
        self.assertAlmostEqual(testGDP2A_LIE.value[4], 12003959.733058406)
        self.assertAlmostEqual(testGDP2A_LIE.value[5], 97119771.42771776)
        self.assertAlmostEqual(testGDP2A_LIE.value[6], 0.0)
        self.assertAlmostEqual(testGDP2A_LIE.value[7], 4137081.3646739507)
        self.assertAlmostEqual(testGDP2A_LIE.value[8], 27411196.308422357)
        self.assertAlmostEqual(testGDP2A_LIE.value[9], 0.0)
        self.assertAlmostEqual(testGDP2A_LIE.value[10], 4125847.312198318)
        self.assertAlmostEqual(testGDP2A_LIE.value[11], 88557558.43543366)
        self.assertAlmostEqual(testGDP2A_LIE.value[12], 191881403.05181965)

        self.assertAlmostEqual(testGDP2A_LIE.gdf.impf_RF[0], 3.0)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.impf_RF[12], 3.0)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.region_id[0], 11.0)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.region_id[12], 11.0)

    def test_two_countries(self):
        testGDP2A_LIE_CHE = ga.GDP2Asset()
        testGDP2A_LIE_CHE.set_countries(countries=['LIE', 'CHE'], path=DEMO_GDP2ASSET)

        self.assertAlmostEqual(testGDP2A_LIE_CHE.longitude[0], 9.520696799999968, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.longitude[45], 7.39570019999996, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.longitude[1000], 9.604029999999966, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.longitude[2500], 9.395696999999984, 4)

        self.assertAlmostEqual(testGDP2A_LIE_CHE.latitude[0], 47.062247399999976, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.latitude[45], 45.978915799999996, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.latitude[1000], 46.6039148, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.latitude[2500], 47.3955802, 4)

        self.assertAlmostEqual(testGDP2A_LIE_CHE.value[0], 174032107.65846416, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.value[45], 11682292.467251074, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.value[1000], 508470546.39168245, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.value[2500], 949321115.5175464, 4)

        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.impf_RF[0], 3.0)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.impf_RF[12], 3.0)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.region_id[0], 11.0)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.region_id[2500], 11.0)


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(
            TestGDP2AssetClassCountries)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
