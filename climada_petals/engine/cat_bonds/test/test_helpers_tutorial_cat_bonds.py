"""Tests for utils_tutorial helper functions."""

import unittest
from datetime import date
from unittest.mock import MagicMock

from climada_petals.engine.cat_bonds.helpers_tutorial_cat_bonds import (
    create_earthquake_mmi_impact_function,
    remap_event_years_even,
)


class TestUtilsTutorial(unittest.TestCase):
    """Minimal unit tests for tutorial helpers."""

    def test_create_earthquake_mmi_impact_function(self):
        """The helper returns an EQ impact function that starts at zero intensity."""
        impf_set = create_earthquake_mmi_impact_function()
        func = impf_set.get_func(haz_type="EQ", fun_id=1)

        self.assertIsNotNone(func)
        self.assertEqual(func.intensity[0], 0.0)
        self.assertEqual(func.mdd[0], 0.0)
        self.assertEqual(func.paa[0], 0.0)

    def test_remap_event_years_even(self):
        """The helper changes years but keeps each event's month and day."""
        original_dates = [date(1980, 3, 10), date(1980, 7, 22), date(1980, 11, 5)]
        hazard = MagicMock()
        hazard.date = [value.toordinal() for value in original_dates]

        remap_event_years_even(hazard, start_year=2000, n_years=3)

        remapped_dates = [date.fromordinal(int(value)) for value in hazard.date]
        self.assertEqual([value.year for value in remapped_dates], [2000, 2001, 2002])
        self.assertEqual([(value.month, value.day) for value in remapped_dates], [(3, 10), (7, 22), (11, 5)])


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestUtilsTutorial)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
