import unittest
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

from climada_petals.engine.warn import Warn, Operation
from climada.util.api_client import Client


class TestWarn(unittest.TestCase):
    def test_from_map(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.flatten().shape)

        # no operations = only binning of map
        warn_levels = np.array([0.0, 20, 50])
        filter_data = Warn.WarnParameters(warn_levels, operations=[],
                                          gradual_decr=False, change_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)

        self.assertEqual(warn.warning[4, 4], 1)
        self.assertEqual(np.sum(warn.warning), 1)

    # cosmo file is 2 GB, should that be on the data api?
    # def test_wind_from_cosmo(self):

    def test_from_hazard(self):
        client = Client()
        tc_dataset_infos = client.list_dataset_infos(data_type='tropical_cyclone')
        client.get_property_values(tc_dataset_infos,
                                   known_property_values={'country_name': 'Haiti'})

        # Read hazard
        tc_haiti = client.get_hazard('tropical_cyclone', properties={'country_name': 'Haiti',
                                                                     'climate_scenario': 'rcp45',
                                                                     'ref_year': '2040',
                                                                     'nb_synth_tracks': '10'})

        warn_levels = [0, 20, 30, 1000]
        filter_data = Warn.WarnParameters(warn_levels, operations=[],
                                          gradual_decr=False, change_sm=0)
        warn = Warn.from_hazard(tc_haiti, filter_data)

        self.assertGreater(warn.warning.shape[0], 1)
        self.assertGreater(warn.warning.shape[1], 1)
        self.assertEqual(np.max(warn.warning), 2)
        self.assertEqual(np.min(warn.warning), 0)

    def test_bin_map(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[2, 2] = -2
        wind_matrix[3, 3] = 20
        wind_matrix[4, 4] = 40
        wind_matrix[5, 5] = 50
        wind_matrix[6, 6] = 60
        coords = np.random.randint(0, 100, wind_matrix.flatten().shape)

        # no operations = only binning of map
        warn_levels = np.array([0.0, 20, 50])
        filter_data = Warn.WarnParameters(warn_levels, operations=[],
                                          gradual_decr=False, change_sm=0)
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
        coords = np.random.randint(0, 100, wind_matrix.flatten().shape)

        warn_levels = np.array([-5, 20, 50])

        filter_data = Warn.WarnParameters(warn_levels, operations=[],
                                          gradual_decr=False, change_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 1)

    def test_bin_map_non_increasing(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.flatten().shape)

        warn_levels = np.array([5, -20, 50])

        filter_data = Warn.WarnParameters(warn_levels, operations=[],
                                          gradual_decr=False, change_sm=0)
        self.assertRaises(ValueError, Warn.from_map, wind_matrix, coords, filter_data)

    def test_filtering(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.flatten().shape)

        warn_levels = np.array([0, 20, 50])

        filter_data = Warn.WarnParameters(warn_levels, operations=[(Operation.dilation, 1)],
                                          gradual_decr=False, change_sm=False)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 5)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

        filter_data = Warn.WarnParameters(warn_levels, operations=[(Operation.erosion, 1)],
                                          gradual_decr=False, change_sm=False)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 0)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

        filter_data = Warn.WarnParameters(warn_levels, operations=[(Operation.median_filtering, 1)],
                                          gradual_decr=False, change_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 1)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

        # test not allowed operation
        self.assertRaises(ValueError, Warn.WarnParameters, warn_levels, operations=[('BLA', 1)])

    def test_generate_warn_map_functionality(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.flatten().shape)

        warn_levels = np.array([0.0, 20, 50])

        filter_data = Warn.WarnParameters(warn_levels,  operations=[(Operation.dilation, 1)],
                                          gradual_decr=False, change_sm=False)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 5)  # 5, because of round filter shape
        self.assertEqual(np.max(warn.warning), 1)  # 5, because of round filter shape

        filter_data = Warn.WarnParameters(warn_levels, operations=[(Operation.erosion, 1)],
                                          gradual_decr=False, change_sm=False)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.max(warn.warning), 0)

        filter_data = Warn.WarnParameters(warn_levels, operations=[(Operation.median_filtering, 2)],
                                          gradual_decr=False, change_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.max(warn.warning), 0)

    def test_generate_warn_map_multiple_levels(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        wind_matrix[7, 7] = 70
        wind_matrix[1, 2] = 85
        coords = np.random.randint(0, 100, wind_matrix.flatten().shape)

        warn_levels = np.array([0.0, 20, 50, 80, 90])

        filter_data = Warn.WarnParameters(warn_levels, operations=[(Operation.dilation, 1)],
                                          gradual_decr=False, change_sm=False)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.max(warn.warning), 3)  # max shouldn't be reduced
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

        filter_data = Warn.WarnParameters(warn_levels, operations=[(Operation.erosion, 1)],
                                          gradual_decr=False, change_sm=False)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.max(warn.warning), 0)  # single points reduced to level 0
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_generate_warn_map_neutral_combination(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.flatten().shape)

        warn_levels = np.array([0.0, 20, 50])

        filter_data = Warn.WarnParameters(warn_levels, operations=[(Operation.erosion, 0),
                                                                   (Operation.dilation, 0),
                                                                   (Operation.median_filtering, 1)],
                                          gradual_decr=False, change_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 1)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_generate_warn_map_combination(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.flatten().shape)

        warn_levels = np.array([0.0, 20, 50])

        filter_data = Warn.WarnParameters(warn_levels, operations=[(Operation.dilation, 1),
                                                                   (Operation.erosion, 1),
                                                                   (Operation.median_filtering, 1)],
                                          gradual_decr=False, change_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 1)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_generate_warn_map_single_level(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.flatten().shape)

        warn_levels = np.array([0.0, 50])

        filter_data = Warn.WarnParameters(warn_levels, operations=[(Operation.dilation, 1),
                                                                   (Operation.erosion, 1),
                                                                   (Operation.median_filtering, 1)],
                                          gradual_decr=False, change_sm=0)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 0)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_generate_warn_map_gradual_decr(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[6, 6] = 40
        wind_matrix[7, 7] = 70
        coords = np.random.randint(0, 100, wind_matrix.flatten().shape)

        warn_levels = np.array([0.0, 20, 50, 80])

        filter_data = Warn.WarnParameters(warn_levels, operations=[(Operation.dilation, 1)],
                                          gradual_decr=False, change_sm=False)
        warn = Warn.from_map(wind_matrix, coords, filter_data)

        filter_data = Warn.WarnParameters(warn_levels, operations=[(Operation.dilation, 1)],
                                          gradual_decr=True, change_sm=False)
        warn_exp = Warn.from_map(wind_matrix, coords, filter_data)

        # level 2 expand has no impact
        self.assertCountEqual(warn_exp.warning[warn_exp.warning == 2],
                              warn.warning[warn.warning == 2])
        # expanding in level 1 is larger
        self.assertGreater(np.count_nonzero(warn_exp.warning[warn_exp.warning == 1]),
                           np.count_nonzero(warn.warning[warn.warning == 1]))
        # therefore, less in level 0 for expansion
        self.assertLess(np.count_nonzero(warn_exp.warning == 0),
                        np.count_nonzero(warn.warning == 0))
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_reset_levels(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 3] = 1
        wind_matrix[4, 4] = 1
        wind_matrix[4, 5] = 1
        wind_matrix[4, 6] = 2

        warn = Warn._reset_levels(wind_matrix.astype(int), 2)
        self.assertEqual(np.sum(warn), 4)
        self.assertEqual(warn.shape, wind_matrix.shape)

    def test_reset_levels_float_error(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 3] = 1
        wind_matrix[4, 4] = 1
        wind_matrix[4, 5] = 1
        wind_matrix[4, 6] = 2

        self.assertRaises(TypeError, Warn._reset_levels, wind_matrix, 2)

    def test_change_small_regions(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.flatten().shape)

        warn_levels = np.array([0.0, 20, 50])

        # check removal of to small region
        filter_data = Warn.WarnParameters(warn_levels, operations=[],
                                          gradual_decr=False, change_sm=True)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 0)
        self.assertEqual(np.max(warn.warning), 0)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

        # check keeping regions
        wind_matrix[4, 4] = 40
        wind_matrix[4, 5] = 40
        wind_matrix[4, 6] = 40
        filter_data = Warn.WarnParameters(warn_levels, operations=[],
                                          gradual_decr=False, change_sm=True)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 3)
        self.assertEqual(np.max(warn.warning), 1)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_increase_levels(self):
        wind_matrix = np.zeros((10, 10))
        wind_matrix[4, 4] = 30
        wind_matrix[4, 5] = 30
        wind_matrix[4, 6] = 40
        coords = np.random.randint(0, 100, wind_matrix.flatten().shape)

        warn_levels = np.array([0.0, 20, 50])

        filter_data = Warn.WarnParameters(warn_levels, operations=[],
                                          gradual_decr=False, change_sm=2)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 3)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_rectangle_shaped_map(self):
        wind_matrix = np.zeros((12, 20))
        wind_matrix[4, 4] = 40
        coords = np.random.randint(0, 100, wind_matrix.flatten().shape)

        warn_levels = np.array([0.0, 20, 50])

        filter_data = Warn.WarnParameters(warn_levels, operations=[(Operation.dilation, 1)],
                                          gradual_decr=False, change_sm=True)
        warn = Warn.from_map(wind_matrix, coords, filter_data)
        self.assertEqual(np.sum(warn.warning), 5)
        self.assertEqual(warn.warning.shape, wind_matrix.shape)

    def test_group_cosmo_ensembles(self):
        wind_matrix = np.zeros((2, 10, 10))
        wind_matrix[0, 4, 4] = 2
        wind_matrix[1, 4, 4] = 2

        reduced_matrix = Warn._group_cosmo_ensembles(wind_matrix, 0.7)

        self.assertEqual(reduced_matrix.shape, (10, 10))
        self.assertEqual(reduced_matrix[4, 4], 2)
        self.assertEqual(np.sum(reduced_matrix), 2)

    def test_zeropadding(self):
        row = np.array([0, 0, 1, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])

        reduced_matrix, coord = Warn.zeropadding(row, col, data)

        self.assertEqual(reduced_matrix.shape, (3, 3))
        self.assertEqual(reduced_matrix[0, 2], 2)
        self.assertEqual(reduced_matrix[2, 2], 6)
        self.assertEqual(np.sum(reduced_matrix), np.sum(data))
        self.assertEqual(reduced_matrix.size*2, coord.size)
        np.testing.assert_array_equal(np.unique(coord[:,1]), np.arange(0,3,1))
        np.testing.assert_array_equal(np.unique(coord[:,1]), np.arange(0,3,1))

    def test_zeropadding_island(self):
        row = np.array([1, 1, 8, 9])
        col = np.array([3, 4, 7, 8])
        data = np.array([20, 25, 28, 32])

        reduced_matrix, coord = Warn.zeropadding(row, col, data)

        self.assertEqual(reduced_matrix.shape, (9, 6))
        self.assertEqual(reduced_matrix[0, 0], 20)
        self.assertEqual(reduced_matrix[8, 5], 32)
        self.assertEqual(np.sum(reduced_matrix), np.sum(data))
        self.assertEqual(reduced_matrix.size*2, coord.size)
        np.testing.assert_array_equal(np.unique(coord[:,0]), np.arange(1,10,1))
        np.testing.assert_array_equal(np.unique(coord[:,1]), np.arange(3,9,1))

    def test_zeropadding_resolution_error(self):
        row = np.array([0, 0, 1.125, 2.3333, 2.3333, 2.3333])
        col = np.array([0, 2.4444, 2.4444, 0, 1.375, 2.4444])
        data = np.array([1, 2, 3, 4, 5, 6])

        with self.assertRaises(ValueError):
            reduced_matrix, coord = Warn.zeropadding(row, col, data)

    def test_plot_warning(self):
        """Plots ones with geo_scatteR_categorical"""
        # test default with one plot
        values = np.array([[1, 20, 40, 45]])
        coord = np.array([[26, 0], [26, 1], [28, 0], [29, 1]])
        warn_levels = np.array([0.0, 20, 50])
        warn_params = Warn.WarnParameters(warn_levels, operations=[],
                                          gradual_decr=False, change_sm=False)
        warn = Warn.from_map(values, coord, warn_params)

        warn.plot_warning('value', 'test plot',
                          pop_name=True)
        plt.close()

    def test_plot_warning_meteoswiss(self):
        """Plots ones with geo_scatteR_categorical"""
        # test default with one plot
        values = np.array([[1, 20, 40, 45]])
        coord = np.array([[26, 0], [26, 1], [28, 0], [29, 1]])
        warn_levels = np.array([0.0, 20, 50])
        warn_params = Warn.WarnParameters(warn_levels, operations=[],
                                          gradual_decr=False, change_sm=False)
        warn = Warn.from_map(values, coord, warn_params)

        warn.plot_warning_meteoswiss_style('value', 'test plot')
        plt.close()

        # test error message when too many levels for meteoswiss
        values = np.array([[1, 30, 55, 65, 75, 85]])
        coord = np.array([[26, 0], [26, 1], [28, 0], [29, 1], [29, 1], [29, 1]])
        warn_levels = np.array([0.0, 20, 50, 60, 70, 80, 90])
        warn_params = Warn.WarnParameters(warn_levels, operations=[],
                                          gradual_decr=False, change_sm=False)
        warn = Warn.from_map(values, coord, warn_params)

        self.assertRaises(ValueError, warn.plot_warning_meteoswiss_style, 'value', 'test plot')

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestWarn)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
