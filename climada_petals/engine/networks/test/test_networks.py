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

Test network modules
"""

from pathlib import Path
import unittest
import numpy as np

from climada import CONFIG
from climada.entity.exposures.base import Exposures
from climada.entity import ImpactFuncSet, ImpfTropCyclone
from climada.hazard.base import Hazard
from climada.util.constants import EXP_DEMO_H5
from climada.util.api_client import Client
from climada.util.files_handler import download_file


class TestNWBase(unittest.TestCase):
    
    def test_read_wiot(self):
        """Test reading of wiod table."""
        sup = SupplyChain()
        sup.read_wiod16(year='test',
                        range_rows=(5,117),
                        range_cols=(4,116),
                        col_iso3=2, col_sectors=1)

        self.assertAlmostEqual(sup.mriot_data[0, 0], 12924.1797, places=3)
        self.assertAlmostEqual(sup.mriot_data[0, -1], 0, places=3)
        self.assertAlmostEqual(sup.mriot_data[-1, 0], 0, places=3)
        self.assertAlmostEqual(sup.mriot_data[-1, -1], 22.222, places=3)

        self.assertAlmostEqual(sup.mriot_data[0, 0],
                               sup.mriot_data[sup.reg_pos[list(sup.reg_pos)[0]][0],
                                              sup.reg_pos[list(sup.reg_pos)[0]][0]],
                               places=3)
        self.assertAlmostEqual(sup.mriot_data[-1, -1],
                               sup.mriot_data[sup.reg_pos[list(sup.reg_pos)[-1]][-1],
                                              sup.reg_pos[list(sup.reg_pos)[-1]][-1]],
                               places=3)
        self.assertEqual(np.shape(sup.mriot_data), (112, 112))
        self.assertAlmostEqual(sup.total_prod.sum(), 3533367.89439, places=3)

 
## Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNWBase)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
