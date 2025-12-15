import numpy as np
import pandas as pd
import geopandas as gpd
from climada_petals.engine.cat_bonds import subarea_calculations
from climada.hazard import Hazard
from climada.hazard.centroids import Centroids
from scipy import sparse
from shapely.geometry import Polygon
import unittest
from unittest.mock import MagicMock
from scipy.optimize import OptimizeResult

class TestSubareaCalculations(unittest.TestCase):

    def setUp(self):
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


        s = subarea_calculations.SubareaCalculations(self.mock_subarea_calc_test, index_stat="mean", intitial_guess=[1,2])

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

        # For mean, event 0: mean(10,20)=15; event1: mean(30,40)=35
        assert df_2["A"].tolist() == [15, 35]
        assert df_2["B"].tolist() == [30, 20]

    def test_calc_pay_vs_dam_expected(self):

        class DummyImpact:
            at_event = np.array([5, 130])  # event damages

        imp = DummyImpact()

        s = subarea_calculations.SubareaCalculations(subareas=None, index_stat="mean", intitial_guess=[1,2])

        # thresholds
        min_t = np.array([1, 0])
        max_t = np.array([2, 4])

        df = s._calc_pay_vs_dam(
            imp, self.mock_subarea_calc_test.imp_subarea, attachment=0, principal=self.mock_subarea_calc_test.principal,
            opt_min_thresh=min_t, opt_max_thresh=max_t, haz_int=self.mock_subarea_calc_test.haz_int_dict
        )

        # event 0: Payout for Subarea B -> 60
        assert df.loc[0,"pay"] == 60
        # event 1: A pays 30 + B pays 60 → capped at principal = 40
        assert df.loc[1,"pay"] == 100
        # damage 
        assert df['damage'].to_list() == [5, 130]
        # year/month copied
        assert df["year"].tolist() == [2000, 2000]
        assert df["month"].tolist() == [1, 2]

    def test_objective_fct_expected(self):

        s = subarea_calculations.SubareaCalculations(subareas=self.mock_subarea_calc_test, index_stat="mean", intitial_guess=(0, 1))

        damages = np.array([100, 200])

        out = s._objective_fct((0, 2), self.mock_subarea_calc_test.haz_int_dict['TC'], damages, self.mock_subarea_calc_test.principal)

        # expected = (100-50)^2 + (200-100)^2 = 50^2 + 100^2 = 12500
        assert out == 12500

    def test_calibration_converges_to_expected_thresholds(self):
        """
        Test that the optimization converges and reports a thresholds for all subareas.
        """
        subareas_calc_dummy = subarea_calculations.SubareaCalculations(
            subareas=self.mock_subarea_calc_test, index_stat=50, intitial_guess=[2,4]
        )

        attachment = 0 

        results, opt_min, opt_max = subareas_calc_dummy._calibrate_payout_fcts(
            haz_int=self.mock_subarea_calc_test.haz_int_dict,
            principal=self.mock_subarea_calc_test.principal,
            attachment=attachment,
            imp_subarea_evt=self.mock_subarea_calc_test.imp_subarea
        )

        # optimizer should converge near
        assert np.allclose(opt_min, [1, 2], atol=1)
        assert np.allclose(opt_max, [3, 4], atol=1)

        # ensure both subareas were optimized
        assert len(results) == 2
        assert all(isinstance(r, OptimizeResult) for r in results.values())
        assert all(r.success for r in results.values())

    

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSubareaCalculations)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
