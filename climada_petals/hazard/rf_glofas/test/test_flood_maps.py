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

Tests for flood_maps.py
"""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import geopandas as gpd

from climada_petals.hazard.rf_glofas.flood_maps import (
    download_flood_maps_extents,
    open_flood_maps_extents,
    download_flood_map_tiles,
    open_flood_map_tiles,
    JRC_FLOOD_HAZARD_MAP_RPS,
)


class TestFloodMapTilesExtents(unittest.TestCase):
    def setUp(self):
        """Create tempdir"""
        self._tempdir = TemporaryDirectory()
        self.tempdir = Path(self._tempdir.name)

    def tearDown(self):
        """Cleanup tempdir"""
        self._tempdir.cleanup()

    def test_download_flood_maps_extents(self):
        """Check if extents file is correctly downloaded"""
        filepath = self.tempdir / "extents.geojson"
        download_flood_maps_extents(filepath)
        self.assertTrue(filepath.is_file())

    def test_open_flood_maps_extents(self):
        """Test opening the extents file"""
        gdf = open_flood_maps_extents(self.tempdir / "extents.geojson")
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertSequenceEqual(gdf.columns.to_list(), ["id", "name", "geometry"])


class TestFloodMapTiles(unittest.TestCase):
    def setUp(self):
        """Create tempdir"""
        self._tempdir = TemporaryDirectory()
        self.tempdir = Path(self._tempdir.name)
        tiles = open_flood_maps_extents(self.tempdir / "extents.geojson")
        self.tiles_select = tiles.loc[
            tiles["name"].isin(["N60_W170", "N60_W160"])
        ]  # Very little data

    def tearDown(self):
        """Cleanup tempdir"""
        self._tempdir.cleanup()

    def test_download_flood_map_tiles(self):
        """Test downloading tiles (using very small ones)"""
        download_flood_map_tiles(
            output_dir=self.tempdir / "tiles", tiles=self.tiles_select
        )
        for rp in JRC_FLOOD_HAZARD_MAP_RPS:
            with self.subTest(return_period=rp):
                rp_dir = self.tempdir / "tiles" / f"RP{rp}"
                self.assertTrue(rp_dir.is_dir())
                filepaths = sorted(rp_dir.iterdir())
                self.assertListEqual(
                    [
                        f"ID3_N60_W170_RP{rp}_depth.tif",
                        f"ID6_N60_W160_RP{rp}_depth.tif",
                    ],
                    [path.name for path in filepaths],
                )

    def test_open_flood_map_tiles(self):
        """Test optning the tiles (using very little data)"""
        tiledir = self.tempdir / "tiles"
        download_flood_map_tiles(output_dir=tiledir, tiles=self.tiles_select)
        da = open_flood_map_tiles(flood_maps_dir=tiledir, tiles=self.tiles_select)
        self.assertSetEqual(
            {"longitude", "latitude", "return_period"}, set(da.sizes.keys())
        )
        self.assertListEqual(
            da["return_period"].to_numpy().tolist(), [1] + JRC_FLOOD_HAZARD_MAP_RPS
        )
        self.assertEqual(
            np.count_nonzero(~np.isnan(da.sel(return_period=1).to_numpy())), 0
        )


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFloodMapTilesExtents)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFloodMapTiles))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
