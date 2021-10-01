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

Test openstreetmap modules.
"""


import unittest

from climada_petals.entity.exposures.openstreetmap import osm_dataloader as osm_dl

class TestOSMRaw(unittest.TestCase):
    """Test OSMRaw class"""

    def test_create_gf_download_url(self):
        """ test methods of osmraw"""
        OSMRawTest = osm_dl.OSMRaw()
        url_shp = OSMRawTest._create_gf_download_url('DEU', 'shp')
        url_pbf = OSMRawTest._create_gf_download_url('ESP', 'pbf')
        
        self.assertEqual('https://download.geofabrik.de/europe/germany-latest-free.shp.zip',
        url_shp)
        self.assertEqual('https://download.geofabrik.de/europe/spain-latest.osm.pbf',
        url_pbf)
        self.assertRaises(KeyError,OSMRawTest._create_gf_download_url('RUS', 'pbf'))
        self.assertRaises(KeyError,OSMRawTest._create_gf_download_url('XYZ', 'pbf'))

    def test_get_data_geofabrik(self):
        """test methods of osmraw" """
        pass
    
    def test_get_data_planet(self):
        """test methods of osmraw" """
        pass

    def test_get_data_fileextract(self):
        """test methods of osmraw"""
        pass
        
class TestOSMFileQuery(unittest.TestCase):
    """Test OSMFileQuery class"""

    def test_osmfileq_f1(self):
        """ test methods of osmfq"""
        pass

    def test_osmfileq_f2(self):
        """test methods of osmfq" """
        pass
    
class TestOSMApiQuery(unittest.TestCase):
    """Test OSMApiQuery class"""

    def test_osmapiq_f1(self):
        """ test methods of osmaq"""
        pass

    def test_osmapiq_f2(self):
        """test methods of osmaq" """
        pass
    
# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestOSMRaw)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOSMFileQuery))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOSMApiQuery))
    unittest.TextTestRunner(verbosity=2).run(TESTS)