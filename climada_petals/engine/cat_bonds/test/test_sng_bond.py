import pandas as pd
import numpy as np
from scipy import sparse
import logging
from climada_petals.engine.cat_bonds.sng_bond_simulation import SingleCountryBondSimulation

class DummySubareaCalc:
    def __init__(self, principal=100):
        self.principal = principal
        self.pay_vs_dam = None  # filled separately

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


if __name__ == "__main__":
    test_init_bond_loss()
    LOGGER.info("test_init_bond_loss passed")