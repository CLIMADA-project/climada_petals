import numpy as np
import pandas as pd
import geopandas as gpd
from climada_petals.engine.cat_bonds import subarea_calculations
from climada.hazard import Hazard
from climada.hazard.centroids import Centroids
from scipy import sparse
from shapely.geometry import Polygon
from climada_petals.util.config import LOGGER


def test_calc_payout_basic():
    """Test basic functionality of calc_payout function."""

    haz_int = pd.DataFrame({"intensity": [0, 5, 10, 15, 20]})
    min_trig = 5
    max_trig = 15
    max_pay = 100

    payouts = subarea_calculations.calc_payout(min_trig, max_trig, haz_int, max_pay)

    # Expected:
    # intensity < 5 → 0
    # 5 → 0
    # 10 → 50
    # ≥ 15 → 100
    assert np.allclose(payouts, [0, 0, 50, 100, 100])

def test_calc_attachment_principal_expected():

    class DummyImpact:
        def calc_freq_curve(self, rp):
            return type("obj", (), {"impact": rp * 10})  # simple mapping

    class Dummy:
        exposure = type("exp", (), {"gdf": {"value": pd.Series([100, 300])}})
        pass

    dummy = Dummy()

    s = subarea_calculations.SubareaCalculations(dummy, index_stat="mean", intitial_guess=[1,2])

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

def test_calc_parametric_index():
    centroids = Centroids(lat=np.array([0, 1, 3, 4]), lon=np.array([0, 1, 3, 4]))
    hazard_dummy = Hazard(haz_type="TC",
                          event_id=np.array([0, 1]),
                          event_name=["evt1", "evt2"],
                          date=np.array([700101, 700102]),
                          intensity=sparse.csr_matrix(np.array([[10, 20, 20, 40], [30, 40, 0, 40]])),
                          centroids=centroids,
                          units="m/s")

    class DummySubareas:
        hazard = hazard_dummy
        d = {'subarea_letter': ['A', 'B'], 'geometry': [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])]}
        subareas_gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    
    subareas_dummy = DummySubareas()
    subareas_calc_dummy = subarea_calculations.SubareaCalculations(
        subareas=subareas_dummy, index_stat="mean"
    )

    out = subareas_calc_dummy._calc_parametric_index()

    df = out["TC"]

    # For mean, event 0: mean(10,20)=15; event1: mean(30,40)=35
    assert df["A"].tolist() == [15, 35]
    assert df["B"].tolist() == [30, 20]
    assert df["year"].tolist() == [1917, 1917]
    assert df["month"].tolist() == [10, 10]

    subareas_calc_dummy_2 = subarea_calculations.SubareaCalculations(
        subareas=subareas_dummy, index_stat=50
    )

    out_2 = subareas_calc_dummy_2._calc_parametric_index()

    df_2 = out_2["TC"]

    # For mean, event 0: mean(10,20)=15; event1: mean(30,40)=35
    assert df_2["A"].tolist() == [15, 35]
    assert df_2["B"].tolist() == [30, 20]

def test_calc_pay_vs_dam_expected():

    class DummyImpact:
        at_event = np.array([10, 70])  # event damages

    imp = DummyImpact()

    imp_sub = pd.DataFrame({"A": [0, 30], "B": [5, 40]})

    haz_int = {"TC": pd.DataFrame({
        "A": [1, 2],
        "B": [3, 4],
        "year": [2000, 2000],
        "month": [1, 2]
    })}

    s = subarea_calculations.SubareaCalculations(subareas=None, index_stat="mean", intitial_guess=[1,2])

    # thresholds
    min_t = np.array([1, 0])
    max_t = np.array([2, 4])
    
    df = s._calc_pay_vs_dam(
        imp, imp_sub, attachment=0, principal=50,
        opt_min_thresh=min_t, opt_max_thresh=max_t, haz_int=haz_int
    )

    # event 0: no payouts → 0
    assert df.loc[0,"pay"] == 30
    # event 1: A pays 30 + B pays 40 → capped at principal = 50
    assert df.loc[1,"pay"] == 50
    # year/month copied
    assert df["year"].tolist() == [2000, 2000]
    assert df["month"].tolist() == [1, 2]

def test_objective_fct_expected():

    class Dummy:
        pass

    s = subarea_calculations.SubareaCalculations(subareas=Dummy(), index_stat="mean", intitial_guess=(0, 1))

    damages = np.array([0, 10, 20])
    haz_int = pd.DataFrame({
        "A": np.array([0, 1, 2]), 
        "year": [2000, 2000, 2000],
        "month": [1, 1, 1]
    })

    # principal less than max_dam → max_pay = principal
    principal = 20

    out = s._objective_fct((0, 2), haz_int, damages, principal)

    # expected = (0-0)^2 + (10-10)^2 + (20-20)^2 = 0 + 25 + 25 = 0
    assert out == 0

if __name__ == "__main__":
    test_calc_payout_basic()
    LOGGER.info("test_calc_payout_basic passed")
    test_calc_attachment_principal_expected()
    LOGGER.info("test_calc_attachment_principal_expected passed")
    test_calc_parametric_index()
    LOGGER.info("test_calc_parametric_index passed")
    test_calc_pay_vs_dam_expected()
    LOGGER.info("test_calc_pay_vs_dam_expected passed")
    test_objective_fct_expected()
    LOGGER.info("test_objective_fct_expected passed")
