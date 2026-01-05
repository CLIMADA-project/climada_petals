"""Tests for cat bond utility functions."""

import pandas as pd
import numpy as np
from climada_petals.engine.cat_bonds import utils_cat_bonds
import unittest


class TestUtils(unittest.TestCase):
    """Unit tests for utils_cat_bonds."""


    def test_multi_level_es_basic(self):
        """Validate VaR and ES for a basic loss series."""
        losses = pd.Series([0, 1, 2, 3, 4, 5])
        alphas = [0.5, 0.8]

        var_list, es_list = utils_cat_bonds.multi_level_es(losses, alphas)

        # VaR checks
        assert np.isclose(var_list[0], losses.quantile(0.5))
        assert np.isclose(var_list[1], losses.quantile(0.8))

        # ES checks (mean of tail losses > VaR)
        es_expected_50 = losses[losses > var_list[0]].mean() # type: ignore
        es_expected_80 = losses[losses > var_list[1]].mean() # type: ignore

        assert np.isclose(es_list[0], es_expected_50)
        assert np.isclose(es_list[1], es_expected_80)


    def test_multi_level_es_all_equal_losses(self):
        """Validate VaR/ES when all losses are equal."""
        losses = pd.Series([1, 1, 1, 1])
        alphas = [0.95]

        var_list, es_list = utils_cat_bonds.multi_level_es(losses, alphas)

        # VaR = 1
        assert var_list[0] == 1

        # ES = 1
        assert es_list[0] == 1


    def test_multi_level_es_no_tail_losses(self):
        """Validate ES when tail losses are empty."""
        losses = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        alphas = [0.9]  # VaR = 0

        var_list, es_list = utils_cat_bonds.multi_level_es(losses, alphas)

        # VaR must be 0
        assert var_list[0] == 0

        # ES must be 1
        assert es_list[0] == 1


    def test_allocate_single_payout_partial_first_tranche(self):
        """Validate payout within the first tranche."""
        nominals = np.array([100, 100, 100])
        payout = 30

        remaining, alloc = utils_cat_bonds.allocate_single_payout(payout, nominals)
        assert np.allclose(alloc, [30., 0., 0.])
        assert np.allclose(remaining, [70., 100., 100.])


    def test_allocate_single_payout_exact_first_tranche(self):
        """Validate payout exactly exhausting first tranche."""
        nominals = np.array([50, 100])
        payout = 50

        remaining, alloc = utils_cat_bonds.allocate_single_payout(payout, nominals)

        assert np.allclose(alloc, [50, 0])
        assert np.allclose(remaining, [0, 100])


    def test_allocate_single_payout_spans_multiple_tranches(self):
        """Validate payout spanning multiple tranches."""
        nominals = np.array([50, 100, 200])
        payout = 180  # eats all of tranche 1 and 2, 30 of tranche 3

        remaining, alloc = utils_cat_bonds.allocate_single_payout(payout, nominals)

        assert np.allclose(alloc, [50, 100, 30])
        assert np.allclose(remaining, [0, 0, 170])


    def test_allocate_single_payout_larger_than_all_nominals(self):
        """Validate payout larger than total nominals."""
        nominals = np.array([40, 40])
        payout = 200  # everything wiped

        remaining, alloc = utils_cat_bonds.allocate_single_payout(payout, nominals)

        assert np.allclose(alloc, [40, 40])
        assert np.allclose(remaining, [0, 0])


    def test_allocate_single_payout_zero_payout(self):
        """Validate zero payout leaves nominals unchanged."""
        nominals = np.array([50, 100])
        payout = 0

        remaining, alloc = utils_cat_bonds.allocate_single_payout(payout, nominals)

        assert np.allclose(alloc, [0, 0])
        assert np.allclose(remaining, nominals)


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestUtils)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
