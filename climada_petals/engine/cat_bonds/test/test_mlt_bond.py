import unittest
import pandas as pd
import numpy as np

from climada_petals.engine.cat_bonds.mlt_bond_simulation import MultiCountryBond

"""Tests for MultiCountryBond simulations."""

# ================================
# MOCK CLASS FOR COUNTRY BOND SIM
# ================================
class DummySubareaCalc:
    """Minimal subarea calculation stub."""
    def __init__(self, pay_vs_dam, principal):
        self.pay_vs_dam = pay_vs_dam
        self.principal = principal

class DummyBondSim:
    """Minimal bond simulation stub."""
    def __init__(self, pay_vs_dam, principal, df_loss_month):
        self.subarea_calc = DummySubareaCalc(pay_vs_dam, principal)
        self.df_loss_month = df_loss_month

# ================================
# UNIT TESTS
# ================================

class TestMultiCountryBond(unittest.TestCase):
    """Unit tests for MultiCountryBond."""

    def setUp(self):
        """Set up common fixtures for tests."""
        # simple mock event data
        self.pay_vs_dam_0 = pd.DataFrame({
            "year": [2000, 2001],
            "month": [6, 6],
            "pay": [0.1, 0.2],
            "damage": [0.1, 0.2]
        })
        self.pay_vs_dam_1 = pd.DataFrame({
            "year": [2000, 2001],
            "month": [6, 6],
            "pay": [0.05, 0.1],
            "damage": [0.05, 0.1]
        })

        self.df_loss_month_0 = pd.DataFrame({
            'losses': [[0.1], [0.2]],
            'months': [6, 6]
        })

        self.df_loss_month_1 = pd.DataFrame({
            'losses': [[0.05], [0.1]],
            'months': [6, 6]
        })

        self.country_dict = {
            0: DummyBondSim(self.pay_vs_dam_0, principal=1.0, df_loss_month = self.df_loss_month_0),
            1: DummyBondSim(self.pay_vs_dam_1, principal=1.0, df_loss_month = self.df_loss_month_1)
        }

    def test_prepare_data(self):
        """Check data preparation populates required fields."""
        bond = MultiCountryBond(self.country_dict, term=1, number_of_terms=2)
        # check min_year is correctly detected
        self.assertEqual(bond.min_year, 2000)
        # check pay_vs_dam_dic populated
        self.assertIn(0, bond.pay_vs_dam_dic)
        self.assertIn(1, bond.pay_vs_dam_dic)
        self.assertIn(0, bond.principal_dic_cty)
        self.assertIn(1, bond.principal_dic_cty)

    def test_init_bond_loss(self):
        """Validate per-term loss computation for a single event."""
        bond = MultiCountryBond(self.country_dict, term=1, number_of_terms=2)
        events_per_year = [
            pd.DataFrame({
                "month": [6],
                "country_code": [0],
                "pay": [0.1],
                "damage": [0.1]
            })
        ]
        rel_ann_bond_losses, rel_ann_cty_losses, rel_bond_monthly_losses, coverage_tot, coverage_cty = bond._init_bond_loss(events_per_year, principal=1.0)
        self.assertEqual(rel_ann_bond_losses[0], 0.1)
        self.assertEqual(rel_ann_cty_losses[0][0], 0.1)
        self.assertEqual(rel_ann_cty_losses[1][0], 0.0)
        self.assertEqual(coverage_tot['payout'], 0.1)
        self.assertEqual(coverage_cty[0]['payout'], 0.1)
        self.assertEqual(coverage_cty[0]['damage'], 0.1)
        self.assertEqual(coverage_cty[1]['payout'], 0.0)
        self.assertEqual(coverage_cty[1]['damage'], 0.0)
        pd.testing.assert_frame_equal(rel_bond_monthly_losses, pd.DataFrame({
            'losses': [0.1],
            'months': [6]},
            dtype='object'))

    def test_init_loss_simulation(self):
        """Validate aggregated loss metrics and monthly losses."""
        # -------------------------
        # Create bond
        # -------------------------
        bond = MultiCountryBond(
            country_dictionary=self.country_dict,
            term=1,
            number_of_terms=2
        )

        principal = 2.0
        bond.init_loss_simulation(principal)

        # -------------------------
        # Expected results
        # -------------------------
        expected_annual_losses = np.array([0.075, 0.15]) # toatal payout / 2 as principal = 2
        expected_EL = expected_annual_losses.mean()
        expected_AP = 1.0
        expected_total_payout = 0.45
        expected_total_damage = 0.45

        # -------------------------
        # Assertions (results only)
        # -------------------------
        self.assertAlmostEqual(bond.loss_metrics["EL_ann"], expected_EL, places=5)
        self.assertAlmostEqual(bond.loss_metrics["AP_ann"], expected_AP, places=5)
        self.assertAlmostEqual(bond.loss_metrics["Payout"], expected_total_payout, places=5)
        self.assertAlmostEqual(bond.loss_metrics["Damage"], expected_total_damage, places=5)
        # Monthly losses should match annual losses
        pd.testing.assert_frame_equal(bond.df_loss_month, pd.DataFrame({
            'losses': [[0.05 , 0.025], [0.1, 0.05]],
            'months': [[6, 6], [6, 6]]
        }))

    def test_init_return_simulation(self):
        """Validate return and premium allocation with a single loss."""
        bond = MultiCountryBond(self.country_dict, term=1, number_of_terms=2)
        # One year, one loss in June
        bond.df_loss_month = pd.DataFrame({
            "losses": [[0.2]],   # 20% principal loss
            "months": [[6]]      # loss occurs in June
        })

        # Two countries with simple shares
        bond.tot_coverage_cty = {
            0: {"share_EL": 0.5},
            1: {"share_EL": 0.5}
        }

        premium = 0.1   # 10% annual premium
        rf = 0.0

        # -------------------------
        # Run function
        # -------------------------
        bond.init_return_simulation(premium=premium, rf=rf)

        # -------------------------
        # Manual calculations
        # -------------------------
        # Pre-loss premium (Jan–Jun):
        #   1.0 * 0.1 / 12 * 6 = 0.05
        #
        # Nominal after loss:
        #   1.0 - 0.2 = 0.8
        #
        # Post-loss premium (Jun–Dec):
        #   0.8 * 0.1 / 12 * 6 = 0.04
        #
        # Total premium:
        #   0.05 + 0.04 = 0.09
        #
        # Net cash flow:
        #   0.05 + 0.04 - 0.2 = -0.11

        expected_total_premium = [0.09]
        expected_ncf = [-0.11]

        # Country allocations (50/50)
        expected_cty_0 = [0.045]
        expected_cty_1 = [0.045]

        # -------------------------
        # Assertions
        # -------------------------
        np.testing.assert_allclose(
            bond.ncf["Total"].to_numpy(),
            expected_ncf,
            rtol=1e-10
        )

        np.testing.assert_allclose(
            bond.prem_cty_df["Total"].to_numpy(),
            expected_total_premium,
            rtol=1e-10
        )

        np.testing.assert_allclose(
            bond.prem_cty_df[0].to_numpy(),
            expected_cty_0,
            rtol=1e-10
        )

        np.testing.assert_allclose(
            bond.prem_cty_df[1].to_numpy(),
            expected_cty_1,
            rtol=1e-10
        )

    def test_init_required_principal(self):
        """Validate required principal from simulated losses."""
        bond = MultiCountryBond(self.country_dict, term=3, number_of_terms=2)
        # Two countries
        bond.countries = [0, 1]

        # Uneven country nominals
        bond.principal_dic_cty = {
            0: 1.0,   # country 0
            1: 0.5    # country 1
        }

        # Dummy pay_vs_dam_dic (only structure matters here)
        bond.pay_vs_dam_dic = {
            0: pd.DataFrame({"year": [2000, 2001, 2002, 2003], "pay": [0.0, 0.0, 0.2, 0.0]}),
            1: pd.DataFrame({"year": [2000, 2001, 2002, 2003], "pay": [0.2, 0.2, 0.1, 0.0]}),
        }

        bond.init_required_principal()

        self.assertEqual(bond.requ_principal, 0.7)
        
    def test__init_equ_nom_sim(self):
        """Validate nominal-based loss aggregation."""
        bond = MultiCountryBond(self.country_dict, term=1, number_of_terms=2)
        events_per_year = [
            pd.DataFrame({
                "pay": [0.1, 0.2, 0.4, 0.9],
                "country_code": [0, 1, 0, 1]
            })
        ]
        nominal_dic_cty = {0: 1.0, 1: 1.0}
        tot_loss = bond._init_equ_nom_sim(events_per_year, nominal_dic_cty)
        self.assertEqual(tot_loss, 1.5)

    def test_init_return_simulation_tranches(self):
        """Validate tranche cashflows and premium allocation."""

        bond = MultiCountryBond.__new__(MultiCountryBond)
        bond.term = 1

        bond.df_loss_month = pd.DataFrame({
            "losses": [[0.2]],
            "months": [[6]]
        })

        bond.tot_coverage_cty = {
            0: {"share_EL": 0.5},
            1: {"share_EL": 0.5}
        }

        premiums = [0.1, 0.1]
        tranches = [0.6, 0.4]

        bond.init_return_simulation_tranches(premiums, tranches)

        # Tranche cashflows
        self.assertAlmostEqual(bond.ncf_tranches["0.6"][0], -0.15, places=5)
        self.assertAlmostEqual(bond.ncf_tranches["0.4"][0], 0.04, places=5)

        # Total cashflow
        self.assertAlmostEqual(bond.ncf_tranches["Total"][0], -0.11, places=5)

        # Premium allocation
        self.assertAlmostEqual(bond.prem_cty_df_tranches["Total"][0], 0.09, places=5)
        self.assertAlmostEqual(bond.prem_cty_df_tranches[0][0], 0.045, places=5)
        self.assertAlmostEqual(bond.prem_cty_df_tranches[1][0], 0.045, places=5)

    def test_simulate_bond_pool_n(self):
        """
        Verify:
        - all returned objects are MultiCountryBond
        - every country is assigned to exactly one bond
        - pool_allocation contains all countries
        - pool_allocation and bond contents are consistent
        """

        # ----------------------------------
        # Run function
        # ----------------------------------
        mlt_bond_simulation_dic, pool_alloc, algo_res = MultiCountryBond.simulate_bond_pool_n(
            country_dictionary=self.country_dict,
            term=1,
            number_of_terms=2,
            principal=2.0,
            number_pools=1,
            n_opt_rep=1
        )

        # ----------------------------------
        # Assertions: pool allocation
        # ----------------------------------
        self.assertEqual(len(pool_alloc), 1)


        # ----------------------------------
        # Assertions: MultiCountryBond objects
        # ----------------------------------
        for bond in mlt_bond_simulation_dic.values():
            self.assertIsInstance(bond, MultiCountryBond)

        # ----------------------------------
        # Assertions: country membership in bonds
        # ----------------------------------
        countries_in_bonds = []

        for bond in mlt_bond_simulation_dic.values():
            countries_in_bonds.extend(bond.country_dictionary.keys())

        # Every country must appear exactly once
        self.assertCountEqual(
            countries_in_bonds,
            list(self.country_dict.keys())
        )


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestMultiCountryBond)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
