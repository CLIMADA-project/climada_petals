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

Tests for the network utilities (nw_utils)

``impact_network`` comes from ``conftest.py``.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

from climada_petals.engine.networks import nw_utils as nwu


# ========================================================================
# Fixtures
# ========================================================================

# 4 x 4 raster, cell size 1, covering x 0..4, y 0..4; row 0 is the top row
RASTER_VALUES = np.array(
    [
        [1, 2, 0, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ],
    dtype="float32",
)
NODATA = -99999.0


def _write_raster(path, values, crs="EPSG:4326"):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(0, values.shape[0], 1, 1),
        nodata=NODATA,
    ) as dst:
        dst.write(values, 1)
    return path


@pytest.fixture
def raster_path(tmp_path):
    return _write_raster(tmp_path / "pop.tif", RASTER_VALUES)


def _counts_by_xy(gdf):
    """{(x, y): counts} of a loaded raster."""
    return {
        (round(p.x, 6), round(p.y, 6)): c for p, c in zip(gdf.geometry, gdf["counts"])
    }


# ========================================================================
# Constants
# ========================================================================


def test_line_exposures_include_power_line():
    assert "power_line" in nwu.LINE_EXPOSURES


def test_dependency_name():
    """Naming convention of dependency edges"""
    assert nwu._dependency_name("road", "people") == "dependency_road_people"


# ========================================================================
# Spatial helpers
# ========================================================================


def test_make_edge_geometries():
    """One straight line per from-to pair."""
    lines = nwu.make_edge_geometries(
        [Point(0, 0), Point(1, 1)], [Point(2, 0), Point(1, 3)]
    )

    assert [list(line.coords) for line in lines] == [
        [(0, 0), (2, 0)],
        [(1, 1), (1, 3)],
    ]


def test_ckdnearest_labels_and_threshold():
    """Nearest base point by index label; beyond dist_thresh gives NaN."""
    gdf_base = gpd.GeoDataFrame(
        geometry=[Point(0, 0), Point(5, 0), Point(10, 0)], index=[10, 11, 12]
    )
    gdf_assign = gpd.GeoDataFrame(geometry=[Point(1, 0), Point(9, 0), Point(100, 0)])

    dist, labels = nwu._ckdnearest(gdf_assign, gdf_base, dist_thresh=3)

    np.testing.assert_allclose(dist[:2], [1, 1])
    assert np.isinf(dist[2])
    np.testing.assert_array_equal(labels[:2], [10, 12])
    assert np.isnan(labels[2])


def test_ckdnearest_single_row():
    """A single row (Series) can be assigned as well."""
    gdf_base = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(5, 0)])
    row = gpd.GeoDataFrame(geometry=[Point(4, 0)]).iloc[0]

    _, labels = nwu._ckdnearest(row, gdf_base)

    np.testing.assert_array_equal(labels, [1])


# ========================================================================
# Raster loading
# ========================================================================


def test_load_resampled_raster_cell_centres(raster_path):
    """Without resampling: one point per non-zero cell, at the cell centre."""
    gdf = nwu.load_resampled_raster(raster_path)

    counts = _counts_by_xy(gdf)
    assert len(gdf) == 15  # one zero cell
    assert counts[(0.5, 3.5)] == pytest.approx(1)
    assert counts[(3.5, 0.5)] == pytest.approx(16)
    assert (2.5, 3.5) not in counts  # the zero cell
    assert gdf["counts"].sum() == pytest.approx(RASTER_VALUES.sum())
    assert gdf.crs.to_string() == "EPSG:4326"


def test_load_resampled_raster_extent(raster_path):
    """An extent polygon selects the cells inside it."""
    gdf = nwu.load_resampled_raster(raster_path, extent=box(1, 1, 3, 3))

    assert _counts_by_xy(gdf) == pytest.approx(
        {(1.5, 2.5): 6, (2.5, 2.5): 7, (1.5, 1.5): 10, (2.5, 1.5): 11}
    )


def test_load_resampled_raster_aggregation_keeps_block_sums(raster_path):
    """Aggregating 2 x 2 cells into one gives the sum of the 4 cells."""
    gdf = nwu.load_resampled_raster(raster_path, upscale_factor=0.5)

    assert _counts_by_xy(gdf) == pytest.approx(
        {(1, 3): 14, (3, 3): 19, (1, 1): 46, (3, 1): 54}
    )


def test_load_resampled_raster_aggregation_with_nodata(tmp_path):
    """Nodata cells count as 0 when aggregating, other blocks are unchanged."""
    values = RASTER_VALUES.copy()
    values[0, 2] = NODATA  # inside the top-right block (valid cells 4, 7, 8)
    path = _write_raster(tmp_path / "pop_nodata.tif", values)

    gdf = nwu.load_resampled_raster(path, upscale_factor=0.5)

    assert _counts_by_xy(gdf) == pytest.approx(
        {(1, 3): 14, (3, 3): 19, (1, 1): 46, (3, 1): 54}
    )


def test_load_resampled_raster_uses_raster_crs(tmp_path):
    path = _write_raster(tmp_path / "pop_utm.tif", RASTER_VALUES, crs="EPSG:32632")

    gdf = nwu.load_resampled_raster(path)

    assert gdf.crs.to_string() == "EPSG:32632"


def test_load_resampled_raster_extent_as_bounds(raster_path):
    gdf = nwu.load_resampled_raster(raster_path, extent=(1, 1, 3, 3))

    assert len(gdf) == 4


def test_load_resampled_raster_extent_rounded_to_cells(raster_path):
    """Cells overlapping the extent are kept, and stay on the raster grid."""
    gdf = nwu.load_resampled_raster(raster_path, extent=box(1.2, 1.2, 2.8, 2.8))

    assert _counts_by_xy(gdf) == pytest.approx(
        {(1.5, 2.5): 6, (2.5, 2.5): 7, (1.5, 1.5): 10, (2.5, 1.5): 11}
    )


def test_load_resampled_raster_extent_outside_raises(raster_path):
    with pytest.raises(ValueError, match="does not overlap"):
        nwu.load_resampled_raster(raster_path, extent=box(10, 10, 12, 12))


def test_load_resampled_raster_finer_cells(raster_path):
    """upscale_factor=2: each cell is split into 4 cells with a quarter each."""
    gdf = nwu.load_resampled_raster(raster_path, upscale_factor=2)

    counts = _counts_by_xy(gdf)
    assert len(gdf) == 60  # the 4 parts of the zero cell are dropped
    assert counts[(0.25, 3.75)] == pytest.approx(0.25)
    assert counts[(3.75, 0.25)] == pytest.approx(4)
    assert gdf["counts"].sum() == pytest.approx(RASTER_VALUES.sum())


def test_load_resampled_raster_invalid_factor_raises(raster_path):
    with pytest.raises(ValueError, match="upscale_factor"):
        nwu.load_resampled_raster(raster_path, upscale_factor=0)


def test_load_resampled_raster_crs_override(raster_path):
    gdf = nwu.load_resampled_raster(raster_path, crs="EPSG:3857")

    assert gdf.crs.to_string() == "EPSG:3857"


def test_load_resampled_raster_without_crs_raises(tmp_path):
    path = _write_raster(tmp_path / "pop_no_crs.tif", RASTER_VALUES, crs=None)

    with pytest.raises(ValueError, match="no CRS"):
        nwu.load_resampled_raster(path)


# ========================================================================
# Population cutoff
# ========================================================================


def test_get_pop_cutoff():
    """Left edge of the 10-wide bin where the cumulative share reaches cutoff."""
    gdf = gpd.GeoDataFrame({"counts": [5.0, 15.0, 25.0, 35.0, 120.0]})

    # shares per bin: (0,10] 2.5 %, (10,20] 7.5 %, (20,30] 12.5 %, ...
    assert nwu.get_pop_cutoff(gdf, 0.2) == 20


def test_get_pop_cutoff_includes_largest_cell():
    gdf = gpd.GeoDataFrame({"counts": [5.0, 5.0, 90.0]})

    assert nwu.get_pop_cutoff(gdf, 0.5) == 80


def test_get_pop_cutoff_small_counts():
    gdf = gpd.GeoDataFrame({"counts": [1.0, 2.0, 3.0]})

    assert nwu.get_pop_cutoff(gdf, 0.5) == 0


def test_get_pop_cutoff_zero_population():
    gdf = gpd.GeoDataFrame({"counts": [0.0, 0.0]})

    assert nwu.get_pop_cutoff(gdf, 0.5) == 0


@pytest.mark.parametrize("cutoff", [-0.1, 1.5])
def test_get_pop_cutoff_invalid_cutoff_raises(cutoff):
    gdf = gpd.GeoDataFrame({"counts": [5.0, 15.0]})

    with pytest.raises(ValueError, match="cutoff"):
        nwu.get_pop_cutoff(gdf, cutoff)


# ========================================================================
# WorldPop download
# ========================================================================


@pytest.fixture
def fake_urlretrieve(monkeypatch):
    """Record downloads instead of fetching; returns the list of (url, path)."""
    calls = []

    def _fake(url, path):
        calls.append((url, path))
        open(path, "w").close()

    monkeypatch.setattr(nwu.urllib.request, "urlretrieve", _fake)
    return calls


def test_get_worldpop_data_country_1km(tmp_path, fake_urlretrieve):
    nwu.get_worldpop_data(tmp_path, iso3="SDN", year=2024, res=1000)

    (url, path), = fake_urlretrieve
    assert url.endswith("/2024/SDN/v1/1km_ua/constrained/sdn_pop_2024_CN_1km_R2025A_UA_v1.tif")
    assert path.parent == tmp_path
    assert path.name == "sdn_pop_2024_CN_1km_R2025A_UA_v1.tif"


def test_get_worldpop_data_creates_directory(tmp_path, fake_urlretrieve):
    path = nwu.get_worldpop_data(tmp_path / "new", iso3="SDN", res=1000)

    assert path.is_file()
    assert path.parent == tmp_path / "new"


def test_get_worldpop_data_skips_existing_file(tmp_path, fake_urlretrieve):
    (tmp_path / "sdn_pop_2024_CN_1km_R2025A_UA_v1.tif").touch()

    nwu.get_worldpop_data(tmp_path, iso3="SDN", year=2024, res=1000)

    assert fake_urlretrieve == []


def test_get_worldpop_data_100m_needs_iso3(tmp_path, fake_urlretrieve):
    with pytest.raises(ValueError, match="ISO3"):
        nwu.get_worldpop_data(tmp_path, res=100)


def test_get_worldpop_data_returns_path(tmp_path, fake_urlretrieve):
    path = nwu.get_worldpop_data(tmp_path, iso3="SDN", year=2024, res=1000)

    assert path == tmp_path / "sdn_pop_2024_CN_1km_R2025A_UA_v1.tif"


def test_get_worldpop_data_unsupported_res_raises(tmp_path, fake_urlretrieve):
    with pytest.raises(ValueError, match="res"):
        nwu.get_worldpop_data(tmp_path, iso3="SDN", res=500)


# ========================================================================
# Travel distance thresholds
# ========================================================================


def _dependency_table():
    return pd.DataFrame(
        {
            "source": ["healthcare", "education", "education", "road", "healthcare"],
            "target": ["people", "people", "road", "people", "road"],
            "thresh_dist": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_set_travel_distance_threshs():
    """sources -> people: speed (km/h) x 1000 x hours; other rows unchanged."""
    result = nwu.set_travel_distance_threshs(
        _dependency_table(), "SDN", sources=["healthcare", "education"], hrs_max=2
    )

    # DICT_SPEEDS["SDN"] = 72 km/h -> 72 x 1000 x 2 h = 144 000 m
    assert result["thresh_dist"].tolist() == [
        144000,  # healthcare -> people: set
        144000,  # education -> people: set
        1.0,  # education -> road: other target, unchanged
        1.0,  # road -> people: other source, unchanged
        1.0,  # healthcare -> road: other target, unchanged (operator precedence)
    ]


def test_set_travel_distance_threshs_other_source_single_str():
    """Any source type can be set; a single source can be given as a string."""
    result = nwu.set_travel_distance_threshs(
        _dependency_table(), "SDN", sources="road", hrs_max=1
    )

    assert result["thresh_dist"].tolist() == [1.0, 1.0, 1.0, 72000, 1.0]


def test_set_travel_distance_threshs_other_target():
    """target selects the end-user type (default: DEF_END_USER)."""
    result = nwu.set_travel_distance_threshs(
        _dependency_table(), "SDN", sources=["education"], target="road"
    )

    assert result["thresh_dist"].tolist() == [1.0, 1.0, 72000, 1.0, 1.0]
    assert nwu.DEF_END_USER == "people"


def test_set_travel_distance_threshs_unknown_country():
    result = nwu.set_travel_distance_threshs(
        _dependency_table(), "XXX", sources=["education"], hrs_max=1
    )

    # DICT_SPEEDS["other"] = 73 km/h
    assert result["thresh_dist"].iloc[1] == 73000


def test_set_travel_distance_threshs_does_not_modify_input():
    table = _dependency_table()

    nwu.set_travel_distance_threshs(table, "SDN", sources=["healthcare", "education"])

    assert table["thresh_dist"].tolist() == [1.0] * 5


# ========================================================================
# Network statistics
# ========================================================================


def test_make_network_stat(impact_network):
    stats = nwu.make_network_stat(impact_network)

    expected = {
        "total_population": 170,
        "directly_affected": 50,
        "indirectly_affected": 70,  # people 1 (healthcare) and 2 (road)
        "total_affected": 70,  # people 1 is directly and indirectly affected
        "healthcare_access_loss": 50,
        "healthcare_no_base_access": 20,
        "healthcare_access_undisrupted": 100,
        "healthcare_infra_impact": 1,  # number of failed hospitals
        "healthcare_infra_total": 1,
        "road_access_loss": 20,
        "road_no_base_access": 0,
        "road_access_undisrupted": 150,
        "road_infra_impact": 1000,  # metres of failed road
        "road_infra_total": 1500,
    }
    assert stats == pytest.approx(expected)


def test_make_network_stat_counts_column(impact_network):
    impact_network.nodes = impact_network.nodes.drop(columns="value")

    stats = nwu.make_network_stat(impact_network)

    assert stats["total_population"] == 170


def test_make_network_stat_ci_type_without_enduser_dependency(impact_network):
    """A type without access column gets infrastructure statistics only."""
    plant = gpd.GeoDataFrame(
        {"ci_type": ["power_plant"], "func_tot": [1], "geometry": [Point(5, 5)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    impact_network.nodes = pd.concat([impact_network.nodes, plant], ignore_index=True)

    stats = nwu.make_network_stat(impact_network)

    assert stats["total_population"] == 170
    assert stats["power_plant_infra_impact"] == 0
    assert stats["power_plant_infra_total"] == 1
    assert "power_plant_access_loss" not in stats
    assert stats["indirectly_affected"] == 70  # unchanged


def test_make_network_stat_custom_pop_col(impact_network):
    """pop_col selects another population column."""
    impact_network.nodes["value"] = impact_network.nodes["counts"] * 2

    stats = nwu.make_network_stat(impact_network, pop_col="value")

    assert stats["total_population"] == 340
    assert stats["healthcare_access_loss"] == 100


def test_make_network_stat_ignores_added_links(impact_network):
    """Road links added by the module are not counted as road length."""
    link = gpd.GeoDataFrame(
        {
            "from_id": [4],
            "to_id": [0],
            "ci_type": ["road"],
            "distance": [2000.0],
            "func_tot": [0],
            nwu.ADDED_COL: [True],
            "geometry": [LineString([(0, 1), (0, 0)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    impact_network.edges = pd.concat([impact_network.edges, link], ignore_index=True)

    stats = nwu.make_network_stat(impact_network)

    assert stats["road_infra_impact"] == 1000
    assert stats["road_infra_total"] == 1500


def test_make_network_stat_without_access_loss(impact_network):
    """Nobody losing access: indirectly affected 0, total = directly affected."""
    nodes = impact_network.nodes
    for col in ["access_state_healthcare_people", "access_state_road_people"]:
        nodes.loc[nodes["ci_type"] == "people", col] = "access undisrupted"

    stats = nwu.make_network_stat(impact_network)

    assert stats["indirectly_affected"] == 0
    assert stats["total_affected"] == 50
