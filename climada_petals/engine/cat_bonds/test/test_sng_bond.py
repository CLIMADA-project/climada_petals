import pandas as pd
import numpy as np
from scipy import sparse
import logging
from climada_petals.engine.cat_bonds.sng_bond_simulation import SingleCountryBondSimulation

class DummySubareaCalc:
    def __init__(self, principal=100):
        self.principal = principal
        self.pay_vs_dam = pd.DataFrame()  # DataFrame to be set in tests

logging.basicConfig(
     format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%Y-%m-%d %H:%M",
     level=logging.INFO,
 )
LOGGER = logging.getLogger(__name__)

def test_init_bond_loss():
    sub = DummySubareaCalc(principal=100)
    sim = SingleCountryBondSimulation(subarea_calc=sub, term=1, number_of_terms=1)

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

def test_init_loss_simulation():
    sub = DummySubareaCalc(principal=100)

    sub.pay_vs_dam = pd.DataFrame({
        "year":  [2000,2000,2001,2001,2002,2002, 2003,2003],
        "month": [1,      2,    1,  2,   1,   2,    1,   2],
        "pay":   [0,     10,   50,  0, 100,   0,    0,   0],
        "damage":[0,     20,   60,  0, 120,   0,    0,   0],
    })

    sim = SingleCountryBondSimulation(subarea_calc=sub, term=3, number_of_terms=2)
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


if __name__ == "__main__":
    test_init_bond_loss()
    LOGGER.info("test_init_bond_loss passed")
    test_init_loss_simulation()
    LOGGER.info("test_init_loss_simulation passed")