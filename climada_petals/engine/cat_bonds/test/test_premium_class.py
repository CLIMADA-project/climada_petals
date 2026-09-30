"""Tests for premium calculation utilities."""

import numpy as np
import pandas as pd
from climada_petals.engine.cat_bonds.premium_class import PremiumCalculations  

import unittest

# regression coefficients for chatoro premium calculation (extracted from Chatoro et al., 2022)
b_0 = -0.5907
b_1 = 1.3986
b_2 = 2.2520
b_3 = 0.0377
b_4 = 0.4613
b_5 = -0.0239
b_6 = -2.6742
b_7 = 0.7057


class TestPremiumCalculations(unittest.TestCase):
    """Unit tests for PremiumCalculations."""

    def setUp(self):
        """Set up common fixtures for tests."""
        self.loss_metrics = {"Expected_annual_loss":  0.1}
        self.term = 1
        self.df_loss_month = pd.DataFrame({
        "losses": [[0], [0], [0], [0], [0], [0.1]],
        "months": [[], [], [], [], [] , [1]]
    })
        
    def test_find_sharpe(self):
        """
        Validate net cash flow logic. 
        1 year, 1 loss in June, simple numbers.
        """

        pc = PremiumCalculations(self)
        manual_premium = 0.1
        # manually compute NCF with premium=0.1:
        # Year 1-5: 
        #   - NCF: 0.1
        # Year 6: 
        #   - Pre-event NCF: (1.0 * 0.1)/12 * 1 = 0.008333333333333333
        #   - Post-event NCF: (0.8 * 0.1)/12 * (12 - 1) - 0.1 = -0.0175
        #   NCF = 0.008333333333333333 - 0.0175 = -0.009166666666666668
        NCF_manual = [0.1, 0.1, 0.1, 0.1, 0.1, -0.009166666666666668]
        manual_sharpe = (np.mean(NCF_manual) / np.std(NCF_manual))

        pc.calc_benchmark_premium(target_sharpe=float(manual_sharpe))

        assert np.isclose(np.array(pc.benchmark_prem_rate), np.array(manual_premium), rtol=1e-6) 

    def test_calc_chatoro_premium(self):
        """
        Test if the formula for chatoro premium is used as intended.
        """
        peak_multi = 0
        investment_graded = 0
        hybrid_trigger = 0
        GCIndex = 180
        BBSpread = 1.6
        pc = PremiumCalculations(self)
        pc.calc_chatoro_premium(peak_multi=peak_multi, investment_graded=investment_graded, hybrid_trigger=hybrid_trigger)

        manual_chat_prem = (b_0 + b_1 * self.loss_metrics['Expected_annual_loss'] * 100 + b_2 * peak_multi + b_3 * GCIndex + b_4 * BBSpread + b_5 * self.term * 12 + b_6 * investment_graded + b_7 * hybrid_trigger) / 100

        assert manual_chat_prem == pc.chatoro_prem_rate

    def test_monoexp(self):
        """Verify monoExp evaluates the exponential form correctly."""
        pc = PremiumCalculations(self)
        x = np.array([0.0, 1.0, 2.0])
        a = 2.0
        k = 0.5
        b = 0.1
        expected = a * np.exp(-k * x) + b

        np.testing.assert_allclose(pc.monoExp(x, a, k, b), expected, rtol=1e-12)

    def test_ibrd_premium(self):
        """
        Test if premium rates for the IBRD method are calculated as intended.
        """
        pc = PremiumCalculations(self)
        pc.calc_ibrd_premium()

        assert 0.1 < pc.ibrd_prem_rate < 0.3

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestPremiumCalculations)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
