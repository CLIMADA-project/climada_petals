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

Unit test landslide module.
"""
import unittest
import geopandas as gpd
import numpy as np
import shapely

from climada import CONFIG
from climada_petals.hazard.landslide import Landslide, sample_event_from_probs, sample_events
import climada.util.coordinates as u_coord

DATA_DIR = CONFIG.hazard.test_data.dir()
LS_HIST_FILE = DATA_DIR / 'test_ls_hist.shp'
LS_PROB_FILE = DATA_DIR / 'test_ls_prob.tif'


class TestLandslideModule(unittest.TestCase):

    # TODO: there is a probabilistic element in most of these tests, which is unsatisfying.

    def test_from_hist(self):
        """ Test function from_hist()"""
        LS_hist = Landslide.from_hist(bbox=(20,40,23,46),
                                  input_gdf=LS_HIST_FILE)
        self.assertEqual(LS_hist.size, 272)
        self.assertEqual(LS_hist.haz_type, 'LS')
        self.assertEqual(np.unique(LS_hist.intensity.data),np.array([1]))
        self.assertEqual(np.unique(LS_hist.fraction.data),np.array([1]))
        self.assertTrue((LS_hist.frequency<=1).all())

        input_gdf = gpd.read_file(LS_HIST_FILE)
        LS_hist = Landslide()
        LS_hist.set_ls_hist(bbox=(20,40,23,46),
                                  input_gdf=input_gdf)
        self.assertEqual(LS_hist.size, 272)
        self.assertEqual(LS_hist.haz_type, 'LS')
        self.assertEqual(np.unique(LS_hist.intensity.data),np.array([1]))
        self.assertEqual(np.unique(LS_hist.fraction.data),np.array([1]))
        self.assertTrue((LS_hist.frequency<=1).all())

    def test_set_ls_hist(self):
        """ Test deprecated function set_ls_hist()"""
        LS_hist = Landslide()
        LS_hist.set_ls_hist(bbox=(20,40,23,46),
                                  input_gdf=LS_HIST_FILE)
        self.assertEqual(LS_hist.size, 272)
        self.assertEqual(LS_hist.haz_type, 'LS')
        self.assertEqual(np.unique(LS_hist.intensity.data),np.array([1]))
        self.assertEqual(np.unique(LS_hist.fraction.data),np.array([1]))
        self.assertTrue((LS_hist.frequency<=1).all())

        input_gdf = gpd.read_file(LS_HIST_FILE)
        LS_hist = Landslide()
        LS_hist.set_ls_hist(bbox=(20,40,23,46),
                                  input_gdf=input_gdf)
        self.assertEqual(LS_hist.size, 272)
        self.assertEqual(LS_hist.haz_type, 'LS')
        self.assertEqual(np.unique(LS_hist.intensity.data),np.array([1]))
        self.assertEqual(np.unique(LS_hist.fraction.data),np.array([1]))
        self.assertTrue((LS_hist.frequency<=1).all())

    def test_from_prob(self):
        """ Test the function from_prob()"""

        n_years=1000
        LS_prob = Landslide.from_prob(bbox=(8,45,11,46),
                            path_sourcefile=LS_PROB_FILE, n_years=n_years,
                            dist='binom')

        self.assertEqual(LS_prob.haz_type, 'LS')
        self.assertEqual(LS_prob.intensity.shape,(n_years, 43200))
        self.assertEqual(LS_prob.fraction.shape,(n_years, 43200))
        if LS_prob.intensity.size:  # no non-zero value in the Landslide
            self.assertEqual(max(LS_prob.intensity.data),1)
            self.assertEqual(min(LS_prob.intensity.data),1)
            self.assertEqual(max(LS_prob.fraction.data),1)
            self.assertEqual(min(LS_prob.fraction.data),1)
        self.assertEqual(LS_prob.frequency.shape, (n_years,))
        self.assertEqual(LS_prob.frequency[0],1/n_years)
        self.assertEqual(LS_prob.centroids.crs.to_epsg(), 4326)
        self.assertTrue(LS_prob.centroids.coord.max() <= 46)
        self.assertTrue(LS_prob.centroids.coord.min() >= 8)

        n_years=300
        LS_prob = Landslide.from_prob(bbox=(8,45,11,46),
                            path_sourcefile=LS_PROB_FILE,
                            dist='poisson', n_years=n_years)
        self.assertEqual(LS_prob.haz_type, 'LS')
        self.assertEqual(LS_prob.intensity.shape,(n_years, 43200))
        self.assertEqual(LS_prob.fraction.shape,(n_years, 43200))
        if LS_prob.intensity.size:  # no non-zero value in the Landslide
            self.assertEqual(max(LS_prob.intensity.data),1)
            self.assertEqual(min(LS_prob.intensity.data),1)
            self.assertEqual(max(LS_prob.fraction.data),1)
            self.assertEqual(min(LS_prob.fraction.data),1)
        self.assertEqual(LS_prob.frequency.shape, (n_years,))
        self.assertEqual(LS_prob.frequency[0],1/n_years)
        self.assertEqual(LS_prob.centroids.crs.to_epsg(), 4326)
        self.assertTrue(LS_prob.centroids.coord.max() <= 46)
        self.assertTrue(LS_prob.centroids.coord.min() >= 8)

        corr_fact = 1.8*10e6
        n_years=500
        LS_prob = Landslide.from_prob(bbox=(8,45,11,46),
                            path_sourcefile=LS_PROB_FILE,n_years=n_years,
                            dist='poisson', corr_fact=corr_fact)
        self.assertEqual(LS_prob.haz_type, 'LS')
        self.assertEqual(LS_prob.intensity.shape,(n_years, 43200))
        self.assertEqual(LS_prob.fraction.shape,(n_years, 43200))
        if LS_prob.intensity.size:  # no non-zero value in the Landslide
            self.assertEqual(max(LS_prob.intensity.data),1)
            self.assertEqual(min(LS_prob.intensity.data),1)
            self.assertEqual(max(LS_prob.fraction.data),1)
            self.assertEqual(min(LS_prob.fraction.data),1)
        self.assertEqual(LS_prob.frequency.shape, (n_years,))
        self.assertEqual(LS_prob.frequency[0],1/n_years)
        self.assertEqual(LS_prob.centroids.crs.to_epsg(), 4326)
        self.assertTrue(LS_prob.centroids.coord.max() <= 46)
        self.assertTrue(LS_prob.centroids.coord.min() >= 8)

    def test_set_ls_prob(self):
        """ Test deprecated function set_ls_prob()"""
        LS_prob = Landslide()
        n_years=1000
        LS_prob.set_ls_prob(bbox=(8,45,11,46),
                            path_sourcefile=LS_PROB_FILE, n_years=n_years,
                            dist='binom')

        self.assertEqual(LS_prob.haz_type, 'LS')
        self.assertEqual(LS_prob.intensity.shape,(n_years, 43200))
        self.assertEqual(LS_prob.fraction.shape,(n_years, 43200))
        if LS_prob.intensity.size:  # no non-zero value in the Landslide
            self.assertEqual(max(LS_prob.intensity.data),1)
            self.assertEqual(min(LS_prob.intensity.data),1)
            self.assertEqual(max(LS_prob.fraction.data),1)
            self.assertEqual(min(LS_prob.fraction.data),1)
        self.assertEqual(LS_prob.frequency.shape, (n_years,))
        self.assertEqual(LS_prob.frequency[0],1/n_years)
        self.assertEqual(LS_prob.centroids.crs.to_epsg(), 4326)
        self.assertTrue(LS_prob.centroids.coord.max() <= 46)
        self.assertTrue(LS_prob.centroids.coord.min() >= 8)

        LS_prob = Landslide()
        n_years=300
        LS_prob.set_ls_prob(bbox=(8,45,11,46),
                            path_sourcefile=LS_PROB_FILE,
                            dist='poisson', n_years=n_years)
        self.assertEqual(LS_prob.haz_type, 'LS')
        self.assertEqual(LS_prob.intensity.shape,(n_years, 43200))
        self.assertEqual(LS_prob.fraction.shape,(n_years, 43200))
        if LS_prob.intensity.size:  # no non-zero value in the Landslide
            self.assertEqual(max(LS_prob.intensity.data),1)
            self.assertEqual(min(LS_prob.intensity.data),1)
            self.assertEqual(max(LS_prob.fraction.data),1)
            self.assertEqual(min(LS_prob.fraction.data),1)
        self.assertEqual(LS_prob.frequency.shape, (n_years,))
        self.assertEqual(LS_prob.frequency[0],1/n_years)
        self.assertEqual(LS_prob.centroids.crs.to_epsg(), 4326)
        self.assertTrue(LS_prob.centroids.coord.max() <= 46)
        self.assertTrue(LS_prob.centroids.coord.min() >= 8)

        LS_prob = Landslide()
        corr_fact = 1.8e7
        n_years = 500
        LS_prob.set_ls_prob(bbox=(8, 45, 11, 46),
                            path_sourcefile=LS_PROB_FILE, n_years=n_years,
                            dist='poisson', corr_fact=corr_fact)
        self.assertEqual(LS_prob.haz_type, 'LS')
        self.assertEqual(LS_prob.intensity.shape, (n_years, 43200))
        self.assertEqual(LS_prob.fraction.shape, (n_years, 43200))
        if LS_prob.intensity.size:  # no non-zero value in the Landslide
            self.assertEqual(max(LS_prob.intensity.data), 1)
            self.assertEqual(min(LS_prob.intensity.data), 1)
            self.assertEqual(max(LS_prob.fraction.data), 1)
            self.assertEqual(min(LS_prob.fraction.data), 1)
        self.assertEqual(LS_prob.frequency.shape, (n_years,))
        self.assertEqual(LS_prob.frequency[0], 1/n_years)
        self.assertEqual(LS_prob.centroids.crs.to_epsg(), 4326)
        self.assertTrue(LS_prob.centroids.coord.max() <= 46)
        self.assertTrue(LS_prob.centroids.coord.min() >= 8)

    def test_sample_event_from_probs(self):

        bbox = (8,45,11,46)
        corr_fact = 10e6
        n_samples = 400

        __, prob_matrix = u_coord.read_raster(
            LS_PROB_FILE, geometry=[shapely.geometry.box(*bbox, ccw=True)])
        prob_matrix = prob_matrix.squeeze()/corr_fact

        ev_matrix = sample_event_from_probs(prob_matrix, n_samples, dist='binom')
        self.assertTrue(max(ev_matrix) <= n_samples)
        self.assertEqual(min(ev_matrix), 0)
        self.assertTrue(ev_matrix.shape==prob_matrix.shape)

        ev_matrix = sample_event_from_probs(prob_matrix, n_samples, dist='poisson')
        self.assertTrue(max(ev_matrix) <= n_samples)
        self.assertEqual(min(ev_matrix), 0)
        self.assertTrue(ev_matrix.shape==prob_matrix.shape)

    def test_sample_events(self):
        bbox = (8,45,11,46)
        corr_fact = 10e6
        n_years = 400
        __, prob_matrix = u_coord.read_raster(
            LS_PROB_FILE, geometry=[shapely.geometry.box(*bbox, ccw=True)])
        prob_matrix = prob_matrix.squeeze()/corr_fact

        for i in range(5):
            events = sample_events(prob_matrix, n_years, dist='binom')
            if events.nonzero()[0].size:  # skip (most) all-zero-events
                break
        self.assertTrue(events[events.nonzero()].max() <= 1)  # fails for all-zero-events
        self.assertEqual(events.shape, (n_years, prob_matrix.shape[0]))

        for i in range(5):
            events = sample_events(prob_matrix, n_years, dist='poisson')
            if events.nonzero()[0].size:  # skip (most) all-zero-events
                break
        self.assertTrue(events[events.nonzero()].max() <= 1)  # fails for all-zero-events
        self.assertEqual(events.shape, (n_years, prob_matrix.shape[0]))


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLandslideModule)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
