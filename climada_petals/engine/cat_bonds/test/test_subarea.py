import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon
import logging
from climada_petals.engine.cat_bonds import subareas

logging.basicConfig(
     format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%Y-%m-%d %H:%M",
     level=logging.INFO,
 )
LOGGER = logging.getLogger(__name__)


class DummyExposure:
    """Simple container to mimic the expected `exposure` object"""
    def __init__(self, gdf):
        self.gdf = gdf


def test_create_exp_gdf_returns_single_polygon():
    # --- Arrange -------------------------------------------------------------------
    # Create a small GeoDataFrame with two points that have non-zero "value"
    geometry = [Point(x, y) for x in range(5) for y in range(4)]
    geometry = geometry[:20]
    gdf = gpd.GeoDataFrame(
        {"value": [1] * 8 + [0] * 4 + [1] * 8}, 
        geometry=geometry,
        crs="EPSG:4326"
    )

    exposure = DummyExposure(gdf)

    # --- Act -----------------------------------------------------------------------
    result = subareas._create_exp_gdf(exposure)
    LOGGER.info(f"Resulting GeoDataFrame:\n{result}") 

    # --- Assert --------------------------------------------------------------------
    # 1. Should contain exactly one merged polygon
    assert len(result.geometry) == 2

    # 2. All geometries should be of type Polygon and not empty
    for geom in result.geometry:
        assert isinstance(geom, Polygon) or isinstance(geom, MultiPolygon)
        assert not geom.is_empty

    # 3. Check it is within the bounding box of the points
    minx, miny, maxx, maxy = gdf.total_bounds
    res_minx, res_miny, res_maxx, res_maxy = geom.bounds

    assert res_minx >= minx - 1e-6
    assert res_miny >= miny - 1e-6
    assert res_maxx <= maxx + 1e-6
    assert res_maxy <= maxy + 1e-6

    return exposure, result

def _crop_grid_cells_to_polygon(exp_gdf, exposure):
    resolution = 1.0
    subareas_gdf = subareas._crop_grid_cells_to_polygon(resolution, exp_gdf, exposure)

    assert not subareas_gdf.empty, "Subareas GeoDataFrame should not be empty."
    subareas_gdf.plot()
    assert len(subareas_gdf) == 20, "There should be 20 subareas created."
    subareas_union = subareas_gdf.unary_union
    assert all(
        subareas_union.contains(geom) for geom in exp_gdf.geometry
    ), "Exposure should be within the exposure perimeter polygon."


if __name__ == "__main__":
    exposure, exp_gdf = test_create_exp_gdf_returns_single_polygon()
    _crop_grid_cells_to_polygon(exp_gdf, exposure)
