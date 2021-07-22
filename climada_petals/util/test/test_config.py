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

Test config module.
"""
import unittest
from pathlib import Path

from climada.util.constants import DEMO_DIR
from climada.util.config import CONFIG

class TestConfig(unittest.TestCase):
    """Test Config methods"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_check_constants_depending_on_config(self):
        """Check whether the petals configuration correctly supersedes."""
        self.assertEqual(CONFIG.local_data.demo.str(), './data/demo')
        self.assertEqual(DEMO_DIR, Path('./data/demo').absolute())


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestConfig)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
