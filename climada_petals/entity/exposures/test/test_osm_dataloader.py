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

Test openstreetmap module.
"""

import unittest
import shapely
from random import randint

from climada_petals.entity.exposures import osm_dataloader as osm_dl
from climada import CONFIG

DATA_DIR = CONFIG.exposures.test_data.dir()


class TestOSMApiQuery(unittest.TestCase):
    """Test OSMApiQuery class"""

    def test_osmqpiquery(self):
        """test OSMAPIQuery instance"""
        area = (8.5327506, 47.368260, 8.5486078, 47.376877)
        cond = '["building"]'
        TestAPIQuery = osm_dl.OSMApiQuery.from_bounding_box(area, cond)
        self.assertTrue(hasattr(TestAPIQuery, 'area'))
        self.assertTrue(hasattr(TestAPIQuery, 'condition'))

    def test_area_to_queryformat(self):
        """ test methods of OSMApiQuery"""

        area_bbox = (8.5327506, 47.368260, 8.5486078, 47.376877)
        area_poly = shapely.geometry.Polygon(
            [(8.5327506, 47.368260),
             (8.5486078, 47.376877),
             (8.5486078, 47.39)])

        # Two examples for query conditions:
        cond = '["amenity"="place_of_worship"]'

        osm_qu_bb = osm_dl.OSMApiQuery.from_bounding_box(area_bbox, cond)
        osm_qu_py = osm_dl.OSMApiQuery.from_polygon(area_poly, cond)

        self.assertEqual(
            osm_qu_bb.area, (47.36826, 8.5327506, 47.376877, 8.5486078))
        self.assertEqual(
            osm_qu_py.area,
            '(poly:"47.36826 8.5327506 47.376877 8.5486078 47.39 8.5486078 47.36826 8.5327506")')

    def test_insistent_osm_api_query(self):
        """test methods of OSMApiQuery" """
        pass

    def test_overpass_query_string(self):
        """test methods of OSMApiQuery" """
        area_bbox = (8.5327506, 47.368260, 8.5486078, 47.376877)
        area_poly = shapely.geometry.Polygon(
            [(8.5327506, 47.368260),
             (8.5486078, 47.376877),
             (8.5486078, 47.39)])
        condition_building = '["building"]'
        condition_church = '["amenity"="place_of_worship"]'

        q1 = osm_dl.OSMApiQuery.from_bounding_box(
            area_bbox, condition_building)._overpass_query_string()
        q2 = osm_dl.OSMApiQuery.from_polygon(
            area_poly, condition_church)._overpass_query_string()

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

    def test_get_data_overpass(self):
        """test methods of OSMApiQuery"""

        area_bbox = (8.5327506, 47.368260, 8.5486078, 47.376877)
        area_poly = shapely.geometry.Polygon(
            [(8.5327506, 47.368260),
             (8.5486078, 47.376877),
             (8.5486078, 47.39)])
        condition_building = '["building"]'
        condition_church = '["amenity"="place_of_worship"]'
        gdf1 = osm_dl.OSMApiQuery.from_bounding_box(
            area_bbox, condition_building).get_data_overpass()
        gdf2 = osm_dl.OSMApiQuery.from_polygon(
            area_poly, condition_church).get_data_overpass()

        self.assertTrue(len(gdf1) > 0)
        self.assertTrue(len(gdf2) > 0)
        self.assertTrue(
            'building' in gdf1.iloc[randint(0, len(gdf1)-1)].tags.keys())
        self.assertTrue(
            'amenity' in gdf2.iloc[randint(0, len(gdf2)-1)].tags.keys())
        self.assertTrue('place_of_worship' in gdf2.iloc[randint(
            0, len(gdf2)-1)].tags['amenity'])


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestOSMApiQuery)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
