import unittest
import numpy as np

from climada_petals.engine.warn import Warn, FilterData


class TestWarn(unittest.TestCase):
    def test_threshold_data(self):
        thresholds = np.array([0.0, 20, 50])
        wind_matrix = np.zeros((10, 10))
        wind_matrix[2, 2] = -2
        wind_matrix[3, 3] = 20
        wind_matrix[4, 4] = 40
        wind_matrix[5, 5] = 50
        wind_matrix[6, 6] = 60

        warn = Warn.from_np_array(wind_matrix, thresholds, operations=[], sizes=[])

        self.assertEqual(warn.warning[2, 2], 0)
        self.assertEqual(warn.warning[3, 3], 0)
        self.assertEqual(warn.warning[4, 4], 1)
        self.assertEqual(warn.warning[5, 5], 1)
        self.assertEqual(warn.warning[6, 6], 2)
        self.assertEqual(np.sum(warn.warning), 4)

    def test_filter_algorithm_functionality(self):
        thresholds = np.array([0.0, 20, 50])
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40

        warn = Warn.from_np_array(wind_matrix, thresholds, operations=['DILATION'], sizes=[1])
        self.assertEqual(np.sum(warn.warning), 5)  # 5, because of round filter shape
        self.assertEqual(np.max(warn.warning), 1)  # 5, because of round filter shape

        warn = Warn.from_np_array(wind_matrix, thresholds, operations=['EROSION'], sizes=[1])
        self.assertEqual(np.max(warn.warning), 0)  # 5, because of round filter shape

        warn = Warn.from_np_array(wind_matrix, thresholds, operations=['MEDIANFILTERING'], sizes=[2])
        self.assertEqual(np.max(warn.warning), 0)  # 5, because of round filter shape

    def test_filter_algorithm_multiple_levels(self):
        thresholds = np.array([0.0, 20, 50, 80, 90])
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        wind_matrix[7, 7] = 70
        wind_matrix[1, 2] = 85

        warn = Warn.from_np_array(wind_matrix, thresholds, operations=['DILATION'], sizes=[1])
        self.assertEqual(np.max(warn.warning), 3)  # max shouldn't be reduced

        warn = Warn.from_np_array(wind_matrix, thresholds, operations=['EROSION'], sizes=[1])
        self.assertEqual(np.max(warn.warning), 0)  # single points reduced to level 0

    def test_filter_algorithm_expand(self):
        thresholds = np.array([0.0, 20, 50, 80])
        wind_matrix = np.zeros((10, 10))
        wind_matrix[6, 6] = 40
        wind_matrix[7, 7] = 70

        warn = Warn.from_np_array(wind_matrix, thresholds, expand=False, operations=['DILATION'], sizes=[1])
        warn_exp = Warn.from_np_array(wind_matrix, thresholds, expand=True, operations=['DILATION'], sizes=[1])

        # level 2 expand has no impact
        self.assertCountEqual(warn_exp.warning[warn_exp.warning == 2], warn.warning[warn.warning == 2])
        # expanding in level 1 is larger
        self.assertGreater(np.count_nonzero(warn_exp.warning[warn_exp.warning == 1]),
                           np.count_nonzero(warn.warning[warn.warning == 1]))
        # therefore, less in level 0 for expansion
        self.assertLess(np.count_nonzero(warn_exp.warning == 0), np.count_nonzero(warn.warning == 0))

    def test_filter_algorithm_combination(self):
        thresholds = np.array([0.0, 20, 50])
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40

        # neutral operations
        warn = Warn.from_np_array(wind_matrix, thresholds, expand=False,
                                  operations=['DILATION', 'EROSION', 'MEDIANFILTERING'], sizes=[0, 0, 1])
        self.assertEqual(np.sum(warn.warning), 1)

        # check dilation and erosion lead to same no change in output
        warn = Warn.from_np_array(wind_matrix, thresholds, expand=False,
                                  operations=['DILATION', 'EROSION', 'MEDIANFILTERING'], sizes=[1, 1, 1])
        self.assertEqual(np.sum(warn.warning), 1)

    def test_remove_small_regions(self):
        thresholds = np.array([0.0, 20, 50])
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40

        # check removal of to small region
        warn = Warn.from_np_array(wind_matrix, thresholds, expand=False, operations=[], sizes=[])
        warn_removed = Warn.remove_small_regions(warn.warning, 2)
        self.assertEqual(np.sum(warn_removed), 0)
        self.assertEqual(np.max(warn_removed), 0)

        # check keeping regions
        wind_matrix[4, 4] = 40
        wind_matrix[4, 5] = 40
        wind_matrix[4, 6] = 40
        warn = Warn.from_np_array(wind_matrix, thresholds, expand=False, operations=[], sizes=[])
        warn_kept = Warn.remove_small_regions(warn.warning, 2)
        self.assertEqual(np.sum(warn_kept), 3)
        self.assertEqual(np.max(warn_kept), 1)


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestWarn)
    unittest.TextTestRunner(vaerbosity=2).run(TESTS)
