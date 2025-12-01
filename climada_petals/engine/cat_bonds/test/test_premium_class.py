import numpy as np
import pandas as pd
import logging
from climada_petals.engine.cat_bonds.premium_class import PremiumCalculations  

logging.basicConfig(
     format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%Y-%m-%d %H:%M",
     level=logging.INFO,
 )
LOGGER = logging.getLogger(__name__)

class DummyBondSim:
    """Minimum mock needed for PremiumCalculations."""
    def __init__(self, EL_ann, term, df_loss_month=None):
        self.loss_metrics = {"EL_ann": EL_ann}
        self.term = term
        self.df_loss_month = df_loss_month


def test_find_sharpe():
    """
    Validate net cash flow logic. 
    1 year, 1 loss in June, simple numbers.
    """
    df = pd.DataFrame({
        "losses": [[0], [0], [0], [0], [0], [0.1]],
        "months": [[], [], [], [], [] , [1]]
    })
    LOGGER.info(df)
    dummy_el = 0.1
    bond = DummyBondSim(EL_ann=dummy_el, term=1, df_loss_month=df)
    pc = PremiumCalculations(bond)
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

    pc.calc_benchmark_premium(target_sharpe=manual_sharpe)

    assert np.isclose(np.array(pc.benchmark_prem_rate), np.array(manual_premium), rtol=1e-6) 


if __name__ == "__main__":
    test_find_sharpe()
    LOGGER.info("test_find_sharpe passed")
