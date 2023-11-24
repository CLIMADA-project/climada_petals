        
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

Tests on Coastal Flood Hazard"""

import unittest
import requests
import numpy as np
from climada_petals.hazard.coastal_flood import AQUEDUCT_SOURCE_LINK

RCPS = ['historical', '45', '85']
TARGET_YEARS = ['hist', '2030', '2050', '2080']
RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 250, 500, 1000]
SUBSIDENCE = ['nosub', 'wtsub']
PERCENTILES = ['5', '50', '95']

def rcp_name(rcp):
    return f"rcp{rcp[0]}p{rcp[1]}" if rcp in ['45', '85'] else rcp

def rp_name(return_period):
    return f"{str(return_period).zfill(4)}"

def perc_name(percentile):
    return f"0_perc_{percentile.zfill(2)}" if percentile in ['05', '50'] else '0'

class TestReader(unittest.TestCase):
    """Test that Coastal flood data exist"""
    def test_files_exist(self):

        file_names = [
            f'inuncoast_{rcp_name(rcp)}_{sub}_{year}_rp{rp_name(rp)}_{perc_name(perc)}.tif'
                for rcp in RCPS
                for sub in SUBSIDENCE
                for year in TARGET_YEARS
                for rp in RETURN_PERIODS
                for perc in PERCENTILES
                # You can't have: 
                # - year historic with rcp different than historical
                # - rcp historical, no subsidence and year different than historic
                # - rcp historical and SLR scenarios' percentiles
                if not (((year == 'hist') & (rcp != 'historical')) |
                        ((rcp == 'historical') & (sub == 'nosub') & (year != 'hist')) |
                        ((rcp == 'historical') & (perc != '0')))
                ]

        test_files_pos = np.random.choice(range(len(file_names)),
                                          size=20,
                                          replace=False)
        for i in test_files_pos:
            file_path = "".join([AQUEDUCT_SOURCE_LINK, file_names[i]])
            request_code = requests.get(file_path).status_code
            self.assertTrue(request_code == 200)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
