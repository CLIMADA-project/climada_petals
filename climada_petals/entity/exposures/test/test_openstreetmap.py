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
from pathlib import Path
import shapely
from random import randint

from climada_petals.entity.exposures.openstreetmap import osm_dataloader as osm_dl
from climada import CONFIG

DATA_DIR = CONFIG.exposures.test_data.dir()

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
        self.assertRaises(KeyError,
                             OSMRawTest._create_gf_download_url, 'RUS', 'pbf')
        self.assertRaises(KeyError,
                             OSMRawTest._create_gf_download_url,'XYZ', 'pbf')

    def test_get_data_geofabrik(self):
        """test methods of osmraw" """

        OSMRawTest = osm_dl.OSMRaw()
        OSMRawTest.get_data_geofabrik('PCN', file_format='pbf',
                                      save_path=DATA_DIR)
        OSMRawTest.get_data_geofabrik('PCN', file_format='shp',
                                      save_path=DATA_DIR)
        self.assertTrue(Path(DATA_DIR,'pitcairn-islands-latest.osm.pbf').is_file())
        self.assertTrue(Path(DATA_DIR,'pitcairn-islands-latest-free.shp.zip').is_file())

        Path.unlink(Path(DATA_DIR,'pitcairn-islands-latest-free.shp.zip'))
        Path.unlink(Path(DATA_DIR,'pitcairn-islands-latest.osm.pbf'))

    def test_osmosis_extract(self):
        """test _osmosis_extract"""
        pass

    def test_get_data_planet(self):
        """test methods of osmraw" """
        pass

    def get_data_planetextract(self):
        """test get_data_planetextract """
        pass

    def test_get_data_fileextract(self):
        """test get_data_fileextract"""
        pass

class TestOSMFileQuery(unittest.TestCase):
    """Test OSMFileQuery class"""

    def test_OSMFileQuery(self):
        """ test OSMFileQuery"""
        OSM_FQ = osm_dl.OSMFileQuery(Path(DATA_DIR, 'test_piece.osm.pbf'))
        self.assertTrue(hasattr(OSM_FQ, 'osm_path'))

    def test_query_builder(self):
        """test _query_builder"""
        OSM_FQ = osm_dl.OSMFileQuery(Path(DATA_DIR, 'test_piece.osm.pbf'))
        constraint_dict1 = {'osm_keys' : ['highway'],
                           'osm_query' : """highway='primary'"""}
        q1 = OSM_FQ._query_builder('points', constraint_dict1)
        self.assertEqual(q1, "SELECT osm_id,highway FROM points WHERE highway='primary'")

    def test_retrieve(self):
        OSM_FQ = osm_dl.OSMFileQuery(Path(DATA_DIR, 'test_piece.osm.pbf'))
        gdf = OSM_FQ.retrieve('lines', ['highway'], 'highway')

        self.assertEqual(list(gdf.columns.values), ['osm_id', 'highway', 'geometry'])


    def test_retrieve_cis(self):
        OSM_FQ = osm_dl.OSMFileQuery(Path(DATA_DIR, 'test_piece.osm.pbf'))
        gdf = OSM_FQ.retrieve_cis('road')
        self.assertEqual(len(gdf),191)
        self.assertEqual(list(gdf.columns.values), ['osm_id', 'highway', 'man_made', 'public_transport', 'bus', 'name',
       'geometry'])


class TestOSMApiQuery(unittest.TestCase):
    """Test OSMApiQuery class"""

    def test_osmqpiquery(self):
        """test OSMAPIQuery instance"""
        area = (8.5327506, 47.368260, 8.5486078, 47.376877)
        cond = '["building"]'
        TestAPIQuery =  osm_dl.OSMApiQuery(area,cond)
        self.assertTrue(hasattr(TestAPIQuery, 'area'))
        self.assertTrue(hasattr(TestAPIQuery, 'condition'))

    def test_area_to_queryformat(self):
        """ test methods of OSMApiQuery"""

        area_bbox = (8.5327506, 47.368260, 8.5486078, 47.376877)
        area_poly = shapely.geometry.Polygon([(8.5327506, 47.368260), (8.5486078, 47.376877), (8.5486078, 47.39)])

        # Two examples for query conditions:
        cond = '["amenity"="place_of_worship"]'

        area_test_bb =  osm_dl.OSMApiQuery(area_bbox,cond)._area_to_queryformat(area_bbox)
        area_test_poly =  osm_dl.OSMApiQuery(area_poly,cond)._area_to_queryformat(area_poly)

        self.assertEqual(area_test_bb, (47.36826, 8.5327506, 47.376877, 8.5486078))
        self.assertEqual(
            area_test_poly, '(poly:"47.36826 8.5327506 47.376877 8.5486078 47.39 8.5486078 47.36826 8.5327506")')

    def test_insistent_osm_api_query(self):
        """test methods of OSMApiQuery" """
        pass

    def test_overpass_query_string(self):
        """test methods of OSMApiQuery" """
        area_bbox = (8.5327506, 47.368260, 8.5486078, 47.376877)
        area_poly = shapely.geometry.Polygon([(8.5327506, 47.368260), (8.5486078, 47.376877), (8.5486078, 47.39)])
        condition_building = '["building"]'
        condition_church = '["amenity"="place_of_worship"]'

        q1 = osm_dl.OSMApiQuery(area_bbox,condition_building)._overpass_query_string()
        q2 = osm_dl.OSMApiQuery(area_poly,condition_church)._overpass_query_string()

        self.assertEqual(
            q1, '[out:json][timeout:180];(nwr["building"](47.36826, 8.5327506, 47.376877, 8.5486078);(._;>;););out;')
        self.assertEqual(
            q2, '[out:json][timeout:180];(nwr["amenity"="place_of_worship"](poly:"47.36826 8.5327506 47.376877 8.5486078 47.39 8.5486078 47.36826 8.5327506");(._;>;););out;')

    def test_assemble_from_relations(self):
        """test methods of OSMApiQuery" """
        pass

    def test_assemble_from_ways(self):
        """test methods of OSMApiQuery" """
        pass

    def test_assemble_from_nodes(self):
        """test methods of OSMApiQuery" """
        pass

    def test_update_availability(self):
        """test methods of OSMApiQuery" """
        pass

    def test_assemble_results(self):
        """test methods of OSMApiQuery" """
        pass

    def test_osm_geoms_to_gis(self):
        """test methods of OSMApiQuery" """
        pass

    def skip_test_get_data_overpass(self):
        """test methods of OSMApiQuery"
        skipping: test causes a segfault on Jenkins"""
        area_bbox = (8.5327506, 47.368260, 8.5486078, 47.376877)
        area_poly = shapely.geometry.Polygon([(8.5327506, 47.368260), (8.5486078, 47.376877), (8.5486078, 47.39)])
        condition_building = '["building"]'
        condition_church = '["amenity"="place_of_worship"]'
        gdf1 = osm_dl.OSMApiQuery(area_bbox,condition_building).get_data_overpass()
        gdf2 = osm_dl.OSMApiQuery(area_poly,condition_church).get_data_overpass()

        self.assertTrue(len(gdf1)>0)
        self.assertTrue(len(gdf2)>0)
        self.assertTrue('building' in gdf1.iloc[randint(0,len(gdf1)-1)].tags.keys())
        self.assertTrue('amenity' in gdf2.iloc[randint(0,len(gdf2)-1)].tags.keys())
        self.assertTrue('place_of_worship' in gdf2.iloc[randint(0,len(gdf2)-1)].tags['amenity'])

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestOSMRaw)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOSMFileQuery))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOSMApiQuery))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
