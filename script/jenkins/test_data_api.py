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

Test files_handler module.
"""

from pathlib import Path
import unittest
import xmlrunner

from climada_petals.hazard.tc_tracks_forecast import TCForecast


class TestDataAvail(unittest.TestCase):
    """Test availability of data used through APIs"""

    def test_ecmwf_tc_bufr(self):
        """Test availability ECMWF essentials TC forecast."""
        fcast = TCForecast.fetch_bufr_ftp()
        [f.close() for f in fcast]


# Execute Tests
if __name__ == '__main__':
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDataAvail)
    from sys import argv
    outputdir = argv[1] if len(argv) > 1 else str(Path.cwd().joinpath('tests_xml'))
    xmlrunner.XMLTestRunner(output=outputdir).run(TESTS)
