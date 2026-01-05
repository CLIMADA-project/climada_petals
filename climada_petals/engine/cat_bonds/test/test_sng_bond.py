"""Tests for SingleCountryBondSimulation."""

import pandas as pd
import numpy as np
from climada_petals.engine.cat_bonds.sng_bond_simulation import SingleCountryBondSimulation
import unittest

class TestSingleCountryBond(unittest.TestCase):
    """Unit tests for SingleCountryBondSimulation."""
    def setUp(self):
        """Set up common fixtures for tests."""
        self.principal = 100
        self.pay_vs_dam = pd.DataFrame()  # DataFrame to be set in tests

    def test_init_bond_loss(self):
        """Validate per-term loss calculation and principal exhaustion."""
        sim = SingleCountryBondSimulation(subarea_calc=self, term=1, number_of_terms=1)

        # Year 0 events:
        df = pd.DataFrame({
            "month": [1, 2, 3],
            "pay":   [30, 50, 60],   # cumulative = 30, 80 → exhaust occurs at event 2
            "damage":[40, 60, 70]
        })

        rel_losses, df_month, tot_pay, tot_dam = sim.init_bond_loss([df])

        # principal exhausted at payout 2: payout[2] becomes (100 - 80) = 20
        assert np.allclose(rel_losses.values, [100/100])  
        assert np.allclose(tot_pay, 100)
        assert tot_dam == 40 + 60 + 70

        # Monthly losses divided by principal
        assert df_month["losses"].iloc[0] == [0.30, 0.50, 0.20]
        assert df_month["months"].iloc[0] == [1, 2, 3]

    def test_init_loss_simulation(self):
        """Validate aggregated loss metrics and VaR/ES outputs."""

        self.pay_vs_dam = pd.DataFrame({
            "year":  [2000,2000,2001,2001,2002,2002, 2003,2003],
            "month": [1,      2,    1,  2,   1,   2,    1,   2],
            "pay":   [0,     10,   50,  0, 100,   0,    0,   0],
            "damage":[0,     20,   60,  0, 120,   0,    0,   0],
        })

        sim = SingleCountryBondSimulation(subarea_calc=self, term=3, number_of_terms=2)
        sim.init_loss_simulation(confidence_levels=[0.95])

        # Expected annual losses for each year
        # 2000 → 10
        # 2001 → 50
        # 2002 → 40
        expected = np.array([10/100, 50/100, 40/100, 50/100, 50/100, 0/100])
        actual = sim.df_loss_month["losses"].apply(sum).to_numpy()

        assert np.allclose(actual, expected)

        metrics = sim.loss_metrics
        assert np.isclose(metrics["EL_ann"], expected.mean())
        assert np.isclose(metrics["AP_ann"], (expected > 0).mean())
        assert metrics["Tot_payout"] == 10 + 50 + 40 + 50 + 50 + 0
        assert metrics["Tot_damages"] == 20 + 60 + 120 + 60 + 120 + 0

        # VaR and ES present
        assert "VaR_95_ann" in metrics
        assert "ES_95_ann" in metrics

    def test_init_return_simulation(self):
        """Validate annual premium and return calculations over terms."""
        sim = SingleCountryBondSimulation(subarea_calc=self, term=3, number_of_terms=2)

        # Create simple df_loss_month:
        # Year 0: no losses → full premium
        # Year 1: one loss at month 3 of size 0.20
        sim.df_loss_month = pd.DataFrame({
            "losses": [[0.0], [0.25, 0.25], [0.5], [0.25, 0.25], [0.5], [0.2]],
            "months": [[1], [3, 12], [1], [3, 12], [1], [1]]
        })

        sim.init_return_simulation(premium=0.1)

        out = sim.return_metrics
        # First Term:
        # Year 0 premium: 100 * 0.1 = 10
        # Year 1 premium:
        #   - first premium: month 3: 100 * 0.1/12 * 3 = 2.5
        #   - loss at month 3: 0.25 * 100 = 25 → new nominal = 75
        #   - second premium: month 3 to 12: 75 * 0.1/12 * 9 = 5.625
        #   - loss at month 12: 0.25 * 100 = 25 → new nominal = 50
        # Total prem year 1 = 8.125
        # Year 2 premium: 
        #   - first premium month 1: 50 * 0.1/12 * 1 = 0.4167
        #   - loss at month 1: 0.5 * 100 = 50 → new nominal = 0
        #   - premium for rest of year = 0
        # Total premium year 2: 0.4167

        # Second term: 
        # Year 0 premium:
        #   - first premium: month 3: 100 * 0.1/12 * 3 = 2.5
        #   - loss at month 3: 0.25 * 100 = 25 → new nominal = 75
        #   - second premium: month 3 to 12: 75 * 0.1/12 * 9 = 5.625
        #   - loss at month 12: 0.25 * 100 = 25 → new nominal = 50
        # Total prem year 0 = 8.125
        # Year 1 premium: 
        #   - first premium month 1: 50 * 0.1/12 * 1 = 0.4167
        #   - loss at month 1: 0.5 * 100 = 50 → new nominal = 0
        #   - premium for rest of year = 0
        # Total premium year 2: 0.4167
        # Year 3 premium: 
        # Total premium year 3: 0 (as principal is depleted)
        assert np.allclose(out["annual_premiums"], [10/100, 8.125/100, 0.4167/100, 8.125/100, 0.4167/100, 0], atol=1e-3)

        # Returns:
        # Premiums - losses
        assert np.allclose(out["annual_returns"], [10/100, (8.125/100)-0.5, 0.4167/100-0.5, (8.125/100)-0.5, 0.4167/100-0.5, 0], atol=1e-3)

    def test_init_bond_loss_same_month_events(self):
        """Validate handling of multiple events in the same month."""
        sim = SingleCountryBondSimulation(subarea_calc=self, term=1, number_of_terms=1)

        df = pd.DataFrame({
            "month": [6, 6, 6],
            "pay": [30, 50, 40],
            "damage": [35, 55, 45],
        })

        rel_losses, df_month, tot_pay, tot_dam = sim.init_bond_loss([df])

        # principal exhausted after second event: third payout should be 20
        assert np.allclose(rel_losses.values, [100/100])
        assert np.allclose(tot_pay, 100)
        assert tot_dam == 35 + 55 + 45

        assert df_month["losses"].iloc[0] == [0.30, 0.50, 0.20]
        assert df_month["months"].iloc[0] == [6, 6, 6]


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSingleCountryBond)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
