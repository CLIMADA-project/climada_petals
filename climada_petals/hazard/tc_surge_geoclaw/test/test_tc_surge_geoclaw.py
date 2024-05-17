"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test tc_surge_geoclaw module
"""

import sys
import unittest

import numpy as np
import xarray as xr

from climada.hazard import Centroids, TCTracks
from climada.util.api_client import Client
from climada_petals.hazard.tc_surge_geoclaw import TCSurgeGeoClaw


def _test_bathymetry_tif():
    """Topo-Bathymetry (combined land surface and ocean floor) raster data for testing

    SRTM15+V2.3 data of Tubuai island enlarged by factor 10.
    """
    client = Client()
    _, [bathymetry_tif] = client.download_dataset(
        client.get_dataset_info(name='test_bathymetry_tubuaix10', status='test_dataset')
    )
    return bathymetry_tif


class TestHazardInit(unittest.TestCase):
    """Test init and properties of TCSurgeGeoClaw class"""

    @unittest.skipIf(sys.platform.startswith("win"), "does not run on Windows")
    def test_init(self):
        """Test TCSurgeGeoClaw basic object properties"""
        # use dummy track that is too weak to actually produce surge
        track = xr.Dataset({
            'radius_max_wind': ('time', np.full((8,), 20.)),
            'radius_oci': ('time', np.full((8,), 200.)),
            'max_sustained_wind': ('time', np.full((8,), 30.)),
            'central_pressure': ('time', np.full((8,), 990.)),
            'time_step': ('time', np.full((8,), 3, dtype=np.float64)),
            'basin': ('time', np.full((8,), "SPW"))
        }, coords={
            'time': np.arange(
                '2010-02-05', '2010-02-06', np.timedelta64(3, 'h'), dtype='datetime64[ns]',
            ),
            'lat': ('time', np.linspace(-26.33, -21.5, 8)),
            'lon': ('time', np.linspace(-147.27, -150.56, 8)),
        }, attrs={
            'sid': '2010029S12177_test_dummy',
            'name': 'Dummy',
            'orig_event_flag': True,
            'category': 0,
        })
        tracks = TCTracks()
        tracks.data = [track, track]
        topo_path = _test_bathymetry_tif()

        # first run, with automatic centroids
        centroids = tracks.generate_centroids(res_deg=30 / (60 * 60), buffer_deg=5.5)
        haz = TCSurgeGeoClaw.from_tc_tracks(tracks, centroids, topo_path)
        self.assertIsInstance(haz, TCSurgeGeoClaw)
        self.assertEqual(haz.intensity.shape[0], 2)
        np.testing.assert_array_equal(haz.intensity.toarray(), 0)

        # second run, with explicit centroids
        coord = np.array([
            # points along coastline:
            [-23.44084378, -149.45562336], [-23.43322580, -149.44678650],
            [-23.42347479, -149.42088538], [-23.42377951, -149.41418156],
            [-23.41494266, -149.39742201], [-23.41494266, -149.38919460],
            [-23.38233772, -149.38949932],
            # points inland at higher altitude:
            [-23.37505943, -149.46882493], [-23.36615826, -149.45798872],
        ])
        centroids = Centroids()
        centroids.set_lat_lon(coord[:, 0], coord[:, 1])
        haz = TCSurgeGeoClaw.from_tc_tracks(tracks, centroids, topo_path)
        self.assertIsInstance(haz, TCSurgeGeoClaw)
        self.assertEqual(haz.intensity.shape, (2, coord.shape[0]))
        np.testing.assert_array_equal(haz.intensity.toarray(), 0)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestHazardInit)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
