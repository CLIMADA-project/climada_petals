import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon
from climada_petals.engine.cat_bonds import subareas

from climada_petals.util.config import LOGGER
import unittest
from unittest.mock import MagicMock


class DummyExposure:
    """Simple container to mimic the expected `exposure` object"""
    def __init__(self, gdf):
        self.gdf = gdf

class TestSubarea(unittest.TestCase):

    def setUp(self):
        geometry = [Point(x, y) for x in range(5) for y in range(4)]
        geometry = geometry[:20]
        gdf = gpd.GeoDataFrame(
            {"value": [1] * 8 + [0] * 4 + [1] * 8}, 
            geometry=geometry,
            crs="EPSG:4326"
        )

        self.exposure = MagicMock()
        self.exposure.gdf = gdf
        result = subareas._create_exp_gdf(self.exposure)
        self.exposure.result = result

    def test_create_exp_gdf_returns_single_polygon(self):
        # --- Assert --------------------------------------------------------------------
        # 1. Should contain exactly one merged polygon
        assert len(self.exposure.result.geometry) == 2

        # 2. All geometries should be of type Polygon and not empty
        for geom in self.exposure.result.geometry:
            assert isinstance(geom, Polygon) or isinstance(geom, MultiPolygon)
            assert not geom.is_empty

        # 3. Check it is within the bounding box of the points
        minx, miny, maxx, maxy = self.exposure.gdf.total_bounds
        res_minx, res_miny, res_maxx, res_maxy = geom.bounds

        assert res_minx >= minx - 1e-6
        assert res_miny >= miny - 1e-6
        assert res_maxx <= maxx + 1e-6
        assert res_maxy <= maxy + 1e-6


    def test_crop_grid_cells_to_polygon(self):
        resolution = 1.0
        subareas_gdf = subareas._crop_grid_cells_to_polygon(resolution, self.exposure.result, self.exposure)

        assert not subareas_gdf.empty, "Subareas GeoDataFrame should not be empty."
        subareas_gdf.plot()
        assert len(subareas_gdf) == 16, "There should be 16 subareas created."
        subareas_union = subareas_gdf.unary_union
        assert all(
            subareas_union.contains(geom) for geom in self.exposure.result.geometry
        ), "Exposure should be within the exposure perimeter polygon."

    def test_merge_overlapping_grids(self):
        polygon_over = [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(4, 4), (5, 4), (5, 5), (4, 5)])
        ]
        gdf_over = gpd.GeoDataFrame(geometry=polygon_over, crs="EPSG:4326")
        merged_gdf = subareas._merge_overlapping_grids(gdf_over)
        assert len(merged_gdf) == 2, "There should be 2 merged polygons."
        assert merged_gdf.unary_union.equals(gdf_over.unary_union), "The merged geometries should cover the same area as the original."

        polygon_not_over = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            Polygon([(4, 4), (5, 4), (5, 5), (4, 5)])
        ]
        gdf_not_over = gpd.GeoDataFrame(geometry=polygon_not_over, crs="EPSG:4326")
        merged_gdf_not_over = subareas._merge_overlapping_grids(gdf_not_over)
        assert len(merged_gdf_not_over) == 3, "There should be 3 polygons as there are no overlaps."
        assert merged_gdf_not_over.equals(gdf_not_over), "The merged GeoDataFrame should be identical to the input."

        polygon_within = [
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
            Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            Polygon([(3, 3), (3.5, 3), (3.5, 3.5), (3, 3.5)])
        ]
        gdf_within = gpd.GeoDataFrame(geometry=polygon_within, crs="EPSG:4326")
        merged_gdf_within = subareas._merge_overlapping_grids(gdf_within)
        assert len(merged_gdf_within) == 1, "There should be 1 merged polygon."
        assert merged_gdf_within.unary_union.equals(gdf_within.unary_union), "The merged geometries should cover the same area as the original."



if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSubarea)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
