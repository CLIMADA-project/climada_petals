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

Test openstreetmap modules.
"""


import unittest
import numpy as np
import shapely

class TestOSMRaw(unittest.TestCase):
    """Test OSMRaw class"""

    def test_osmraw_f1(self):
        """ test methods of osmraw"""
        pass

    def test_osmraw_f2(self):
        """test methods of osmraw" """
        pass
    
class TestOSMFileQuery(unittest.TestCase):
    """Test OSMFileQuery class"""

    def test_osmfileq_f1(self):
        """ test methods of osmfq"""
        pass

    def test_osmfileq_f2(self):
        """test methods of osmfq" """
        pass
    
class TestOSMApiQuery(unittest.TestCase):
    """Test OSMApiQuery class"""

    def test_osmapiq_f1(self):
        """ test methods of osmaq"""
        pass

    def test_osmapiq_f2(self):
        """test methods of osmaq" """
        pass
    
# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestOSMRaw)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOSMFileQuery))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOSMApiQuery))
    unittest.TextTestRunner(verbosity=2).run(TESTS)