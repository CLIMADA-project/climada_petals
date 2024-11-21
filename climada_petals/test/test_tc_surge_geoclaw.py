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

Test non-trivial runs of TCSurgeGeoClaw class
"""

import unittest

import numpy as np
import xarray as xr

from climada.util.api_client import Client
from climada_petals.hazard.tc_surge_geoclaw import (
    area_sea_level_from_monthly_nc,
    setup_clawpack,
)
from climada_petals.hazard.tc_surge_geoclaw.tc_surge_geoclaw import (
    _geoclaw_surge_from_track
)


def _test_altimetry_nc():
    """Altimetry (ocean surface) raster data for testing

    Sample of monthly Copernicus satellite altimetry for year 2010.
    """
    client = Client()
    _, [altimetry_nc] = client.download_dataset(
        client.get_dataset_info(name='test_altimetry_tubuai', status='test_dataset')
    )
    return altimetry_nc


def _test_bathymetry_tif():
    """Topo-Bathymetry (combined land surface and ocean floor) raster data for testing

    SRTM15+V2.3 data of Tubuai island enlarged by factor 10.
    """
    client = Client()
    _, [bathymetry_tif] = client.download_dataset(
        client.get_dataset_info(name='test_bathymetry_tubuaix10', status='test_dataset')
    )
    return bathymetry_tif


class TestGeoclawRun(unittest.TestCase):
    """Test functions that set up and run GeoClaw instances"""

    def test_surge_from_track(self):
        """Test _geoclaw_surge_from_track function (~30 seconds on a notebook)"""
        # similar to IBTrACS 2010029S12177 (OLI, 2010) hitting Tubuai
        track = xr.Dataset({
            'radius_max_wind': ('time', [15., 15, 15, 15, 15, 17, 20, 20]),
            'radius_oci': ('time', [202., 202, 202, 202, 202, 202, 202, 202]),
            'max_sustained_wind': ('time', [105., 97, 90, 85, 80, 72, 65, 66]),
            'central_pressure': ('time', [944., 950, 956, 959, 963, 968, 974, 975]),
            'time_step': ('time', np.full((8,), 3, dtype=np.float64)),
        }, coords={
            'time': np.arange('2010-02-05T09:00', '2010-02-06T09:00',
                              np.timedelta64(3, 'h'), dtype='datetime64[ns]'),
            'lat': ('time', [-26.33, -25.54, -24.79, -24.05,
                             -23.35, -22.7, -22.07, -21.50]),
            'lon': ('time', [-147.27, -148.0, -148.51, -148.95,
                             -149.41, -149.85, -150.27, -150.56]),
        }, attrs={
            'sid': '2010029S12177_test',
        })
        centroids = np.array([
            # points along coastline:
            [-23.8908, -149.8048], [-23.8628, -149.7431],
            [-23.7032, -149.3850], [-23.7183, -149.2211],
            [-23.5781, -149.1434], [-23.5889, -148.8824],
            # points inland at higher altitude:
            [-23.2351, -149.9070], [-23.2049, -149.7927],
        ])
        gauges = [
            (-23.9059, -149.6248),  # offshore
            (-23.8062, -149.2160),  # coastal
            (-23.2394, -149.8574),  # inland
        ]
        setup_clawpack()
        sea_level_fun = area_sea_level_from_monthly_nc(_test_altimetry_nc())
        intensity, gauge_data = _geoclaw_surge_from_track(
            track,
            centroids,
            _test_bathymetry_tif(),
            geoclaw_kwargs=dict(
                topo_res_as=300,
                gauges=gauges,
                sea_level=sea_level_fun,
                outer_pad_deg=0,
            ),
        )

        self.assertEqual(intensity.shape, (centroids.shape[0],))
        self.assertTrue(np.all(intensity[:6] > 0))
        self.assertTrue(np.all(intensity[6:] == 0))
        for gdata in gauge_data:
            self.assertTrue((gdata['time'][0][0] - track.time[0]) / np.timedelta64(1, 'h') >= 0)
            self.assertTrue((track.time[-1] - gdata['time'][0][-1]) / np.timedelta64(1, 'h') >= 0)
            self.assertAlmostEqual(gdata['base_sea_level'][0], 1.7, places=1)
        self.assertLess(gauge_data[0]['topo_height'][0], 0)
        self.assertTrue(0 <= gauge_data[1]['topo_height'][0] <= 10)
        self.assertGreater(gauge_data[2]['topo_height'][0], 10)

        # surge anomaly of at least 0.4 m
        offshore_h = gauge_data[0]['height_above_geoid'][0]
        self.assertGreater(offshore_h.max() - offshore_h.min(), 0.4)

        # the inland gauge should not be affected by the storm, but maybe by the AMR level
        inland_h = gauge_data[2]['height_above_geoid'][0]
        amr_levels = gauge_data[2]['amr_level'][0]
        for lvl in np.unique(amr_levels):
            self.assertEqual(np.unique(inland_h[amr_levels == lvl]).size, 1)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestGeoclawRun)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
