import unittest
from unittest.mock import patch, MagicMock, DEFAULT
from tempfile import TemporaryDirectory
from pathlib import Path

import xarray as xr
import xarray.testing as xrt
import geopandas as gpd
import pandas.testing as pdt

from climada.util.constants import DEF_CRS
from climada_petals.hazard.rf_glofas.river_flood_computation import (
    _maybe_open_dataarray,
    RiverFloodInundation,
)


class TestLocalRepositoriesAndFiles(unittest.TestCase):
    """Check the maybe_open_dataarray context manager"""

    def cdsapi_exists(self):
        """Create the input array"""
        self.tempdir = TemporaryDirectory()
        self.arr_1 = xr.DataArray(data=[0], coords={"dim": [0]})
        self.arr_2 = xr.DataArray(data=[1], coords={"dim": [1]})
        self.filename = Path(self.tempdir.name) / "file.nc"
        self.arr_2.to_netcdf(self.filename)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLocalRepositoriesAndFiles)
    # TESTS.addTests(
    #     unittest.TestLoader().loadTestsFromTestCase(TestRiverFloodInundation)
    # )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
