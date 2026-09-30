"""Tests for subarea calculations."""

import numpy as np
import pandas as pd
import geopandas as gpd
from climada_petals.engine.cat_bonds import subarea_calculations
from climada.hazard import Hazard
from climada.hazard.centroids import Centroids
from scipy import sparse
from shapely.geometry import Polygon, Point
import unittest
from unittest.mock import MagicMock, patch
from scipy.optimize import OptimizeResult

class TestSubareaCalculations(unittest.TestCase):
    """Unit tests for SubareaCalculations."""

    def setUp(self):
        """Set up shared fixtures for subarea tests."""
        self.mock_subarea_calc_test = MagicMock()
        self.mock_subarea_calc_test.haz_int = pd.DataFrame({"intensity": [0, 5, 10, 15, 20]})
        self.mock_subarea_calc_test.haz_int_dict = {"TC": pd.DataFrame({
            "A": [1, 2],
            "B": [3, 4],
            "year": [2000, 2000],
            "month": [1, 2]
        })}
        self.mock_subarea_calc_test.principal = 100
        self.mock_subarea_calc_test.exposure = type("exp", (), {"gdf": {"value": pd.Series([100, 300])}})
        self.mock_subarea_calc_test.imp_subarea = pd.DataFrame({"A": [0, 60], "B": [5, 80]})

        
    def test_calc_payout_basic(self):
        """Test basic functionality of calc_payout function."""
        min_trig = 5
        max_trig = 15

        payouts = subarea_calculations.calc_payout(min_trig, max_trig, self.mock_subarea_calc_test.haz_int, self.mock_subarea_calc_test.principal )

        # Expected:
        # intensity < 5 → 0
        # 5 → 0
        # 10 → 100
        # ≥ 15 → 100
        assert np.allclose(payouts, [0, 0, 50, 100, 100])

    def test_calc_attachment_principal_expected(self):

        class DummyImpact:
            def calc_freq_curve(self, rp):
                return type("obj", (), {"impact": rp * 10})  # simple mapping


        s = subarea_calculations.SubareaCalculations(self.mock_subarea_calc_test, index_stat="mean", initial_guess=(1.0,2.0))

        imp = DummyImpact()

        # Exposure share: 0.1 → 40, 0.5 → 200
        principal, attachment = s._calc_attachment_principal(
            imp,
            attachment_point=0.1,
            exhaustion_point=0.5,
            attachment_point_method="Exposure_Share",
            exhaustion_point_method="Exposure_Share",
        )

        assert attachment == 40
        assert principal == 200

        # Return period method (rp→impact=rp*10)
        principal_rp, attachment_rp = s._calc_attachment_principal(
            imp,
            attachment_point=5,
            exhaustion_point=20,
            attachment_point_method="Return_Period",
            exhaustion_point_method="Return_Period",
        )
        assert attachment_rp == 50
        assert principal_rp == 200

    def test_calc_parametric_index(self):
        centroids = Centroids(lat=np.array([0, 1, 3, 4]), lon=np.array([0, 1, 3, 4]))
        hazard_dummy = Hazard(haz_type="TC",
                              event_id=np.array([0, 1]),
                              event_name=["evt1", "evt2"],
                              date=np.array([700101, 700102]),
                              intensity=sparse.csr_matrix(np.array([[10, 20, 20, 40], [30, 40, 0, 40]])),
                              centroids=centroids,
                              units="m/s")
        d = {'subarea_letter': ['A', 'B'], 'geometry': [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])]}
        self.mock_subarea_calc_test.hazard = hazard_dummy
        self.mock_subarea_calc_test.subareas_gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")

        subareas_calc_dummy = subarea_calculations.SubareaCalculations(
            subareas=self.mock_subarea_calc_test, index_stat="mean"
        )

        out = subareas_calc_dummy._calc_parametric_index()

        df = out["TC"]

        # For mean, event 0: mean(10,20)=15; event1: mean(30,40)=35
        assert df["A"].tolist() == [15, 35]
        assert df["B"].tolist() == [30, 20]
        assert df["year"].tolist() == [1917, 1917]
        assert df["month"].tolist() == [10, 10]

        subareas_calc_dummy_2 = subarea_calculations.SubareaCalculations(
            subareas=self.mock_subarea_calc_test, index_stat=50
        )

        out_2 = subareas_calc_dummy_2._calc_parametric_index()

        df_2 = out_2["TC"]

        # For median, event 0: median(10,20)=15; event1: median(30,40)=35
        assert df_2["A"].tolist() == [15, 35]
        assert df_2["B"].tolist() == [30, 20]

    def test_calc_pay_vs_dam_expected(self):

        class DummyImpact:
            at_event = np.array([5, 140])  # event damages

        imp = DummyImpact()

        s = subarea_calculations.SubareaCalculations(subareas=None, index_stat="mean", initial_guess=(1, 2))

        # thresholds
        min_t = np.array([1, 0])
        max_t = np.array([2, 4])

        df = s._calc_pay_vs_dam(
            imp, self.mock_subarea_calc_test.imp_subarea, attachment=0, principal=self.mock_subarea_calc_test.principal,
            opt_min_thresh=min_t, opt_max_thresh=max_t, haz_int=self.mock_subarea_calc_test.haz_int_dict
        )

        # event 0: Payout for Subarea B -> 60
        assert df.loc[0,"pay"] == 60
        # event 1: A pays 30 + B pays 60 → capped at principal = 100
        assert df.loc[1,"pay"] == 100
        # damage 
        assert df['damage'].to_list() == [5, 140]
        # year/month copied
        assert df["year"].tolist() == [2000, 2000]
        assert df["month"].tolist() == [1, 2]

    def test_objective_fct_expected(self):

        s = subarea_calculations.SubareaCalculations(subareas=self.mock_subarea_calc_test, index_stat="mean", initial_guess=(0, 1))

        damages = pd.Series([100, 200])

        out = s._objective_fct((0, 2), self.mock_subarea_calc_test.haz_int_dict['TC'], damages, self.mock_subarea_calc_test.principal)

        # expected = (100-50)^2 + (200-100)^2 = 50^2 + 100^2 = 12500
        assert out == 12500

    def test_calibration_converges_to_expected_thresholds(self):
        """
        Test that the optimization converges and reports a thresholds for all subareas.
        """
        principal = 100
        true_min = np.array([2.0, 3.0])
        true_max = np.array([8.0, 9.0])

        haz_int = {
            "TC": pd.DataFrame(
                {
                    "A": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10],
                    "B": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
                    "year": [2000] * 10,
                    "month": list(range(1, 11)),
                }
            )
        }
        imp_subarea_evt = pd.DataFrame(
            {
                "A": [0.0, 0.0, 0.0, 13.0, 27.0, 40.0, 53.0, 80.0, 80.0, 80.0],
                "B": [0.0, 0.0, 0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 90.0, 90.0],
            }
        )

        subareas_calc_dummy = subarea_calculations.SubareaCalculations(
            subareas=self.mock_subarea_calc_test, index_stat=50, initial_guess=(4, 7)
        )

        results, opt_min, opt_max = subareas_calc_dummy._calibrate_payout_fcts(
            haz_int=haz_int,
            principal=principal,
            attachment=0,
            imp_subarea_evt=imp_subarea_evt,
        )

        # optimizer should recover the synthetic thresholds closely
        assert np.allclose(opt_min, true_min, atol=1)
        assert np.allclose(opt_max, true_max, atol=1)

        # ensure both subareas were optimized
        assert len(results) == 2
        assert all(isinstance(r, OptimizeResult) for r in results.values())
        assert all(r.success for r in results.values())

    def test_calc_impact(self):
        """Validate impact calculation wiring and aggregation."""
        exposure_gdf = gpd.GeoDataFrame(
            {"value": [100], "geometry": [Point(0, 0)]},
            crs="EPSG:4326",
        )
        subareas_gdf = gpd.GeoDataFrame(
            {"subarea_letter": ["A"], "geometry": [Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])]},
            crs="EPSG:4326",
        )
        subareas = MagicMock()
        subareas.exposure = type("exp", (), {"gdf": exposure_gdf})
        subareas.vulnerability = MagicMock()
        subareas.hazard = MagicMock()
        subareas.subareas_gdf = subareas_gdf

        dummy_imp = MagicMock()
        dummy_imp.impact_at_reg.return_value = pd.DataFrame({"A": [1.0]})

        impact_calc = MagicMock()
        impact_calc.impact.return_value = dummy_imp

        with patch("climada_petals.engine.cat_bonds.subarea_calculations.ImpactCalc", return_value=impact_calc) as impact_cls:
            calc = subarea_calculations.SubareaCalculations(subareas, index_stat="mean", initial_guess=(1.0, 2.0))
            imp, imp_subareas_evt = calc._calc_impact()

        impact_cls.assert_called_once()
        impact_calc.impact.assert_called_once_with(save_mat=True)
        dummy_imp.impact_at_reg.assert_called_once()
        self.assertIs(imp, dummy_imp)
        pd.testing.assert_frame_equal(imp_subareas_evt, pd.DataFrame({"A": [1.0]}))

    def test_create_pay_vs_dam(self):
        """Validate that create_pay_vs_dam wires the helpers together."""
        calc = subarea_calculations.SubareaCalculations(self.mock_subarea_calc_test, index_stat="mean", initial_guess=(1.0, 2.0))

        with patch.object(calc, "_calc_impact", return_value=("imp", "imp_subareas")) as m_impact, \
             patch.object(calc, "_calc_parametric_index", return_value={"TC": "haz"}) as m_idx, \
             patch.object(calc, "_calc_attachment_principal", return_value=(123.0, 4.0)) as m_attach, \
             patch.object(calc, "_calibrate_payout_fcts", return_value=("results", "min_t", "max_t")) as m_cal, \
             patch.object(calc, "_calc_pay_vs_dam", return_value="pay_vs_dam") as m_pay:
            calc.create_pay_vs_dam(
                attachment_point=0.1,
                exhaustion_point=0.5,
                methods_attachment_point="Exposure_Share",
                methods_exhaustion_point="Return_Period",
            )

        assert all(m.called for m in (m_impact, m_idx, m_attach, m_cal, m_pay))

    def test_fallback_thresholds_basic(self):
        """Fallback returns median-of-nonzero and max of parametric index."""
        pi_df = pd.DataFrame({
            "A": [0, 2, 6],
            "B": [4, 0, 8],
            "year": [2000, 2001, 2002],
            "month": [1, 2, 3],
        })
        min_t, max_t = subarea_calculations.SubareaCalculations._fallback_thresholds(pi_df)
        # non-zero index values are [2, 6, 4, 8] → median = 5.0, max = 8
        assert max_t == 8.0
        assert min_t == 5.0

    def test_fallback_thresholds_degenerate(self):
        """When all non-zero values equal the max, min_trig = 0.5 * max_trig."""
        pi_df = pd.DataFrame({
            "A": [0, 10],
            "B": [10, 10],
            "year": [2000, 2001],
            "month": [1, 2],
        })
        min_t, max_t = subarea_calculations.SubareaCalculations._fallback_thresholds(pi_df)
        assert max_t == 10.0
        assert min_t == 5.0  # 0.5 * 10

    def test_calibration_uses_fallback_on_failure(self):
        """When optimization fails, fallback thresholds are used and arrays match subarea count."""
        mock_sub = MagicMock()

        calc = subarea_calculations.SubareaCalculations(
            subareas=mock_sub, index_stat="mean", initial_guess=(1.0, 2.0)
        )

        haz_int = {"TC": pd.DataFrame({"A": [1, 2], "B": [3, 4], "year": [2000, 2000], "month": [1, 2]})}
        imp_subarea_evt = pd.DataFrame({"A": [0, 60], "B": [5, 80]})

        # Force minimize to always fail
        failed_result = OptimizeResult(x=np.array([0.0, 0.0]), success=False, message="forced failure")

        with patch("climada_petals.engine.cat_bonds.subarea_calculations.minimize", return_value=failed_result):
            results, opt_min, opt_max = calc._calibrate_payout_fcts(
                haz_int=haz_int, principal=100, attachment=0, imp_subarea_evt=imp_subarea_evt
            )

        # Both subareas must be present even though optimization failed
        assert len(opt_min) == 2
        assert len(opt_max) == 2
        # Fallback from parametric index: non-zero values [1,2,3,4] → median=2.5, max=4
        np.testing.assert_allclose(opt_min, [2.5, 2.5])
        np.testing.assert_allclose(opt_max, [4.0, 4.0])

    

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSubareaCalculations)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
