import unittest
import numpy as np
import matplotlib.pyplot as plt

from climada_petals.engine.warn import Warn


class TestWarn(unittest.TestCase):
    def test_bin_map(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[2, 2] = -2
        wind_matrix[3, 3] = 20
        wind_matrix[4, 4] = 40
        wind_matrix[5, 5] = 50
        wind_matrix[6, 6] = 60
        coords = np.random.randint(0, 100, wind_matrix.shape)

        # no operations = only binning of map
        warn_levels = np.array([0.0, 20, 50])
        filter_data = FilterData(warn_levels, operations=[], sizes=[], gradual_decr=False, change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)

        self.assertEqual(warn.warning[2, 2], 0)
        self.assertEqual(warn.warning[3, 3], 1)
        self.assertEqual(warn.warning[4, 4], 1)
        self.assertEqual(warn.warning[5, 5], 2)
        self.assertEqual(warn.warning[6, 6], 2)
        self.assertEqual(np.sum(warn.warning), 6)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_bin_map_negative_levels(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([-5, 20, 50])

        filter_data = FilterData(warn_levels, operations=[], sizes=[], gradual_decr=False, change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 1)

    def test_bin_map_non_increasing(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([5, -20, 50])

        filter_data = FilterData(warn_levels, operations=[], sizes=[], gradual_decr=False, change_sm=False, size_sm=0)
        self.assertRaises(ValueError, Warn.from_map, wind_matrix, coords, filter_data)

    def test_filtering(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40

        warn_levels = np.array([0, 20, 50])

        filter_data = FilterData(warn_levels, operations=['DILATION'], sizes=[1], gradual_decr=False, change_sm=False,
                                 size_sm=0)
        warn = Warn.filtering(wind_matrix, filter_data)
        self.assertEqual(np.sum(warn), 5 * 40)
        self.assertEqual(warn.shape, wind_matrix.shape)

        filter_data = FilterData(warn_levels, operations=['EROSION'], sizes=[1], gradual_decr=False, change_sm=False,
                                 size_sm=0)
        warn = Warn.filtering(warn, filter_data)
        self.assertEqual(np.sum(warn), 1 * 40)
        self.assertEqual(warn.shape, wind_matrix.shape)

        filter_data = FilterData(warn_levels, operations=['MEDIANFILTERING'], sizes=[1], gradual_decr=False,
                                 change_sm=False, size_sm=0)
        warn = Warn.filtering(warn, filter_data)
        self.assertEqual(np.sum(warn), 1 * 40)
        self.assertEqual(warn.shape, wind_matrix.shape)

        # test not allowed operation
        filter_data = FilterData(warn_levels, operations=['BLA'], sizes=[1], gradual_decr=False,
                                 change_sm=False, size_sm=0)
        warn = Warn.filtering(warn, filter_data)
        self.assertEqual(np.sum(warn), 1 * 40)
        self.assertEqual(warn.shape, wind_matrix.shape)
        self.assertRaises(TypeError, Warn.reset_levels, wind_matrix, 2)

    def test_generate_warn_map_functionality(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([0.0, 20, 50])

        filter_data = FilterData(warn_levels,  operations=['DILATION'], sizes=[1], gradual_decr=False, change_sm=False,
                                 size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 5)  # 5, because of round filter shape
        self.assertEqual(np.max(warn.warning), 1)  # 5, because of round filter shape

        filter_data = FilterData(warn_levels, operations=['EROSION'], sizes=[1], gradual_decr=False, change_sm=False,
                                 size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.max(warn.warning), 0)

        filter_data = FilterData(warn_levels, operations=['MEDIANFILTERING'], sizes=[2], gradual_decr=False,
                                 change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.max(warn.warning), 0)

    def test_generate_warn_map_multiple_levels(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        wind_matrix[7, 7] = 70
        wind_matrix[1, 2] = 85
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([0.0, 20, 50, 80, 90])

        filter_data = FilterData(warn_levels, operations=['DILATION'], sizes=[1], gradual_decr=False, change_sm=False,
                                 size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.max(warn.warning), 3)  # max shouldn't be reduced
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

        filter_data = FilterData(warn_levels, operations=['EROSION'], sizes=[1], gradual_decr=False, change_sm=False,
                                 size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.max(warn.warning), 0)  # single points reduced to level 0
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_generate_warn_map_neutral_combination(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([0.0, 20, 50])

        filter_data = FilterData(warn_levels, operations=['DILATION', 'EROSION', 'MEDIANFILTERING'], sizes=[0, 0, 1],
                                 gradual_decr=False, change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 1)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_generate_warn_map_combination(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([0.0, 20, 50])

        filter_data = FilterData(warn_levels, operations=['DILATION', 'EROSION', 'MEDIANFILTERING'], sizes=[1, 1, 1],
                                 gradual_decr=False, change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 1)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_generate_warn_map_single_level(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([0.0, 50])

        filter_data = FilterData(warn_levels, operations=['DILATION', 'EROSION', 'MEDIANFILTERING'],
                                 sizes=[1, 1, 1],
                                 gradual_decr=False, change_sm=False, size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 1)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_generate_warn_map_gradual_decr(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[6, 6] = 40
        wind_matrix[7, 7] = 70
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([0.0, 20, 50, 80])

        filter_data = FilterData(warn_levels, operations=['DILATION'], sizes=[1], gradual_decr=False, change_sm=False,
                                 size_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)

        filter_data = FilterData(warn_levels, operations=['DILATION'], sizes=[1], gradual_decr=True, change_sm=False,
                                 size_sm=0)
        warn_exp = Warn.from_map(wind_matrix, coords, filter_data)

        # level 2 expand has no impact
        self.assertCountEqual(warn_exp.warning[warn_exp.warning == 2], warn.warning[warn.warning == 2])
        # expanding in level 1 is larger
        self.assertGreater(np.count_nonzero(warn_exp.warning[warn_exp.warning == 1]),
                           np.count_nonzero(warn.warning[warn.warning == 1]))
        # therefore, less in level 0 for expansion
        self.assertLess(np.count_nonzero(warn_exp.warning == 0), np.count_nonzero(warn.warning == 0))
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

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
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

        # check keeping regions
        wind_matrix[4, 4] = 40
        wind_matrix[4, 5] = 40
        wind_matrix[4, 6] = 40
        filter_data = FilterData(warn_levels, operations=[], sizes=[], gradual_decr=False, change_sm=True, size_sm=2)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 3)
        self.assertEqual(np.max(warn.warning), 1)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_increase_levels(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 1
        wind_matrix[4, 5] = 1
        wind_matrix[4, 6] = 2

        warn = Warn.increase_levels(wind_matrix, 3)
        self.assertEqual(np.sum(warn), 6)
        self.assertEqual(warn.shape, wind_matrix.shape)

    def test_reset_levels(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 3] = 1
        wind_matrix[4, 4] = 1
        wind_matrix[4, 5] = 1
        wind_matrix[4, 6] = 2

        warn = Warn.reset_levels(wind_matrix.astype(int), 2)
        self.assertEqual(np.sum(warn), 4)
        self.assertEqual(warn.shape, wind_matrix.shape)

    def test_reset_levels_float_error(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 3] = 1
        wind_matrix[4, 4] = 1
        wind_matrix[4, 5] = 1
        wind_matrix[4, 6] = 2

        self.assertRaises(TypeError, Warn.reset_levels, wind_matrix, 2)

    def test_rectangle_shaped_map(self):
        wind_matrix = np.zeros((12, 20))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.shape)

        warn_levels = np.array([0.0, 20, 50])

        filter_data = FilterData(warn_levels, operations=['DILATION'], sizes=[1], gradual_decr=False, change_sm=True,
                                 size_sm=1)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 5)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_geo_scatter_categorical(self):
        """Plots ones with geo_scatteR_categorical"""
        # test default with one plot
        values = np.array([[1, 20, 40, 45]])
        coord = np.array([[26, 0], [26, 1], [28, 0], [29, 1]])
        warn_levels = np.array([0.0, 20, 50])
        warn_params = Warn.WarnParameters(warn_levels, operations=[], gradual_decr=False, change_sm=False)
        warn = Warn.from_map(values, coord, warn_params)

        warn.plot_warning('value', 'test plot',
                        pop_name=True)
        plt.close()

        #test multiple plots with non default kwargs
        values = np.array([[1, 20, 40, 45]])
        coord = np.array([[26, 0], [26, 1], [28, 0], [29, 1]])
        warn_levels = np.array([0.0, 20, 50])
        warn_params = Warn.WarnParameters(warn_levels, operations=[], gradual_decr=False, change_sm=False)
        warn = Warn.from_map(values, coord, warn_params)
        warn.plot_warning('value', 'test plot',
                        cat_name={0: 'zero',
                                  1: 'int',
                                  2.0: 'float',
                                  'a': 'string'},
                        pop_name=False, cmap=plt.get_cmap('Set1'))
        plt.close()

        #test colormap warning
        values = np.array([[1, 20, 40, 45]])
        coord = np.array([[26, 0], [26, 1], [28, 0], [29, 1]])
        warn_levels = np.array([0.0, 20, 50])
        warn_params = Warn.WarnParameters(warn_levels, operations=[], gradual_decr=False, change_sm=False)
        warn = Warn.from_map(values, coord, warn_params)
        warn.plot_warning('value', 'test plot',
                        pop_name=False, cmap='viridis')

        plt.close()

        #test colormap warning with 256 colors
        values = np.array([[1, 20, 40, 45]])
        coord = np.array([[26, 0], [26, 1], [28, 0], [29, 1]])
        warn_levels = np.array([0.0, 20, 50])
        warn_params = Warn.WarnParameters(warn_levels, operations=[], gradual_decr=False, change_sm=False)
        warn = Warn.from_map(values, coord, warn_params)
        warn.plot_warning('value', 'test plot',
                        pop_name=False, cmap='tab20c')
        plt.close()

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestWarn)
    unittest.TextTestRunner(vaerbosity=2).run(TESTS)
