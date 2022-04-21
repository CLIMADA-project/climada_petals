import unittest
import numpy as np

from climada_petals.engine.warn import Warn, FilterData


class TestWarn(unittest.TestCase):
    def test_threshold_data(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[2, 2] = -2
        wind_matrix[3, 3] = 20
        wind_matrix[4, 4] = 40
        wind_matrix[5, 5] = 50
        wind_matrix[6, 6] = 60
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([0.0, 20, 50])
        filter_data = FilterData(warn_levels, operations=[], sizes=[], gradual_decr=False, change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)

        self.assertEqual(warn.warning[2, 2], 0)
        self.assertEqual(warn.warning[3, 3], 1)
        self.assertEqual(warn.warning[4, 4], 1)
        self.assertEqual(warn.warning[5, 5], 2)
        self.assertEqual(warn.warning[6, 6], 2)
        self.assertEqual(np.sum(warn.warning), 6)

    def test_filter_algorithm_functionality(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([0.0, 20, 50])

        filter_data = FilterData(warn_levels,  operations=['DILATION'], sizes=[1], gradual_decr=False, change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 5)  # 5, because of round filter shape
        self.assertEqual(np.max(warn.warning), 1)  # 5, because of round filter shape

        filter_data = FilterData(warn_levels, operations=['EROSION'], sizes=[1], gradual_decr=False, change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.max(warn.warning), 0)

        filter_data = FilterData(warn_levels, operations=['MEDIANFILTERING'], sizes=[2], gradual_decr=False, change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.max(warn.warning), 0)

    def test_filter_algorithm_multiple_levels(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        wind_matrix[7, 7] = 70
        wind_matrix[1, 2] = 85
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([0.0, 20, 50, 80, 90])

        filter_data = FilterData(warn_levels, operations=['DILATION'], sizes=[1], gradual_decr=False, change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.max(warn.warning), 3)  # max shouldn't be reduced

        filter_data = FilterData(warn_levels, operations=['EROSION'], sizes=[1], gradual_decr=False, change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.max(warn.warning), 0)  # single points reduced to level 0

    def test_filter_algorithm_combination(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([0.0, 20, 50])

        # neutral operations
        filter_data = FilterData(warn_levels, operations=['DILATION', 'EROSION', 'MEDIANFILTERING'], sizes=[0, 0, 1],
                                 gradual_decr=False, change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 1)

        # check dilation and erosion lead to same no change in output
        filter_data = FilterData(warn_levels, operations=['DILATION', 'EROSION', 'MEDIANFILTERING'], sizes=[1, 1, 1],
                                 gradual_decr=False, change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 1)

    def test_filter_algorithm_gradual_decr(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[6, 6] = 40
        wind_matrix[7, 7] = 70
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([0.0, 20, 50, 80])

        filter_data = FilterData(warn_levels, operations=['DILATION'], sizes=[1], gradual_decr=False, change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)

        filter_data = FilterData(warn_levels, operations=['DILATION'], sizes=[1], gradual_decr=True, change_sm=False, size_sm=0)
        warn_exp = Warn.from_map(wind_matrix, coords, filter_data)

        # level 2 expand has no impact
        self.assertCountEqual(warn_exp.warning[warn_exp.warning == 2], warn.warning[warn.warning == 2])
        # expanding in level 1 is larger
        self.assertGreater(np.count_nonzero(warn_exp.warning[warn_exp.warning == 1]),
                           np.count_nonzero(warn.warning[warn.warning == 1]))
        # therefore, less in level 0 for expansion
        self.assertLess(np.count_nonzero(warn_exp.warning == 0), np.count_nonzero(warn.warning == 0))

    def test_change_small_regions(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([0.0, 20, 50])

        # check removal of to small region
        filter_data = FilterData(warn_levels, operations=[], sizes=[], gradual_decr=False, change_sm=True, size_sm=2)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 0)
        self.assertEqual(np.max(warn.warning), 0)

        # check keeping regions
        wind_matrix[4, 4] = 40
        wind_matrix[4, 5] = 40
        wind_matrix[4, 6] = 40
        filter_data = FilterData(warn_levels, operations=[], sizes=[], gradual_decr=False, change_sm=True, size_sm=2)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 3)
        self.assertEqual(np.max(warn.warning), 1)


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestWarn)
    unittest.TextTestRunner(vaerbosity=2).run(TESTS)
