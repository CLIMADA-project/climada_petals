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

Test plot submodule
"""

import unittest

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from climada_petals.hazard.tc_surge_geoclaw.plot import (
    _colormap_coastal_dem,
    _plot_bounds,
    LinearSegmentedNormalize,
    plot_dems,
)


class TestPlotHelpers(unittest.TestCase):
    """Test plotting helper functions and classes"""

    def test_linear_segmented_normalize(self):
        """Test LinearSegmentedNormalize class"""
        norm = LinearSegmentedNormalize([0.4, 0.7, 1.5])

        # test return type
        result = norm(0.4)
        self.assertEqual(result.shape, ())
        self.assertEqual(result.dtype, float)

        # test thresh points
        self.assertEqual(norm(0.4)[()], 0.0)
        self.assertEqual(norm(0.7)[()], 0.5)
        self.assertEqual(norm(1.5)[()], 1.0)

        # test intermediate points
        self.assertAlmostEqual(norm(0.55)[()], 0.25)
        self.assertAlmostEqual(norm(1.1)[()], 0.75)
        self.assertAlmostEqual(norm(0.46)[()], 0.1)
        self.assertAlmostEqual(norm(1.02)[()], 0.7)

        # test out-of-bounds values
        self.assertEqual(norm(0.3)[()], 0.0)
        self.assertEqual(norm(-0.1)[()], 0.0)
        self.assertEqual(norm(1.6)[()], 1.0)
        self.assertEqual(norm(2.6)[()], 1.0)

        # test array-like input
        np.testing.assert_array_equal(norm([0.4, 0.7, 1.5]), [0.0, 0.5, 1.0])

    def test_colormap_coastal_dem(self):
        """Test _colormap_coastal_dem function"""
        cmap, cnorm = _colormap_coastal_dem()
        test_data = [
            # values are height above geoid in meters
            (-9000.0, (0.00, 0.00, 0.00, 1.0)),  # deepest ocean => black
            (-4000.0, (0.01, 0.16, 0.25, 1.0)),  # deep ocean => dark blue
            (-300.0, (0.15, 0.43, 0.83, 1.0)),  # shallow ocean => medium blue
            (-1.0, (0.91, 0.94 , 0.71, 1.0)),  # coastal => (blueish) yellow
            (1.0, (0.93, 0.97, 0.66, 1.0)),  # coastal => (greenish) yellow
            (30.0, (0.11, 0.55, 0.10, 1.0)),  # low terrain => medium green
            (200.0, (0.17, 0.43, 0.04, 1.0)),  # high terrain => dark green
            (1400.0, (0.46, 0.33, 0.0, 1.0)),  # highest terrain => dark brown
        ]
        for val, col in test_data:
            np.testing.assert_allclose(cmap(cnorm(val)), col, atol=0.01)

    def test_plot_bounds(self):
        """Test _plot_bounds function"""
        axes = plt.gca()
        bounds = [0.5, 4.0, 7.4, 23.0]
        _plot_bounds(axes, bounds, c="black")

        # exactly one line (the circumference of the rectangle) is drawn
        self.assertEqual(len(axes.lines), 1)

        # check line properties
        [line] = axes.lines
        self.assertEqual(line.get_color(), "black")

        # check that line string has correct length, is recurrent, and uses only input coordinates
        xdata, ydata = line.get_data()
        self.assertEqual(len(xdata), 5)
        self.assertEqual(len(ydata), 5)
        self.assertEqual(xdata[0], xdata[-1])
        self.assertEqual(ydata[0], ydata[-1])
        self.assertTrue(all(x in [bounds[0], bounds[2]] for x in xdata))
        self.assertTrue(all(y in [bounds[1], bounds[3]] for y in ydata))

        # lines are parallel to coordinate axes:
        # (this allows clockwise or counter-clockwise plotting, starting at any node)
        np.testing.assert_array_equal(np.diff(xdata) == 0, np.diff(ydata) != 0)


class TestMainPlotFunctions(unittest.TestCase):
    """Test the main plotting functions that are provided by the module"""

    def test_plot_dems(self):
        """Test plot_dems function"""
        plot_dems([
            ((0.0, 0.0, 1.0, 1.0), np.array([[0.4, 0.7], [0.2, 0.0]])),
            ((0.3, 0.2, 0.8, 0.5), np.array([[4.5, 7.0, 5.3], [13.0, -300.0, -132.2]])),
        ])
        fig = plt.gcf()

        # the first axes is the (main) GeoAxes, the second is for the color bar
        self.assertEqual(len(fig.axes), 2)
        ax_geo, ax_col = fig.axes

        # one line (for bounds) and one image for each DEM
        self.assertEqual(len(ax_geo.lines), 2)
        self.assertEqual(len(ax_geo.images), 2)

        # in standard-lat/lon the extent is the unit square
        np.testing.assert_allclose(
            ax_geo.get_extent(crs=ccrs.PlateCarree()),
            (0.0, 1.0, 0.0, 1.0),
            atol=1e-5,
        )


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestPlotHelpers)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMainPlotFunctions))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
