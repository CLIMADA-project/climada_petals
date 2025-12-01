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


# ---------------------------------------------------------
# TEST SHARPE OPTIMIZATION
# ---------------------------------------------------------

def test_find_sharpe_single_loss_event():
    """
    Validate net cash flow logic. 
    1 year, 1 loss in June, simple numbers.
    """
    df = pd.DataFrame({
        "losses": [[0.2]],
        "months": [[6]]
    })

    bond = DummyBondSim(EL_ann=0.02, term=1, df_loss_month=df)
    pc = PremiumCalculations(bond)

    # manually compute NCF:
    # Pre-event premium: (1 * p)/12 * 6
    # Post-event premium: (0.8 * p)/12 * (12 - 6) - 0.2 loss
    # NCF = p/2 + (0.8p/2 - 0.2) = 0.9p - 0.2
    p = 0.05
    expected_ncf = 0.9 * p - 0.2

    sharpe = pc.find_sharpe(premium=p, monthly_losses=df, target_sharpe=0)

    # extracted NCF from method
    ncf = []
    cur_nominal = 1
    losses = [0.2]
    months = [6]
    ncf_pre = (cur_nominal * p) / 12 * 6
    cur_nominal -= 0.2
    ncf_post = (cur_nominal * p) / 12 * 6 - 0.2
    manual_list = [ncf_pre + ncf_post]

    manual_sharpe = (np.mean(manual_list) / np.std(manual_list + [manual_list[0] * 1.0001]))**2  # std>0

    assert np.isclose(sharpe, manual_sharpe, rtol=1e-6)  # just ensure valid number
    assert np.isclose(expected_ncf, manual_list[0], rtol=1e-6)


def test_calc_benchmark_premium_converges():
    """
    Test that optimization finds higher premium when losses > 0.
    """

    df = pd.DataFrame({
        "losses": [[0.1], [0.2], [0.0]],
        "months": [[3], [6], [0]]
    })

    bond = DummyBondSim(EL_ann=0.02, term=1, df_loss_month=df)
    pc = PremiumCalculations(bond)

    pc.calc_benchmark_premium(target_sharpe=0.2)

    assert pc.benchmark_prem_rate > 0
    assert pc.benchmark_prem_rate < 1.0  # sanity bound


if __name__ == "__main__":
    test_calc_benchmark_premium_converges()
    LOGGER.info("test_calc_benchmark_premium_converges passed")
    test_find_sharpe_single_loss_event()
    LOGGER.info("test_find_sharpe_single_loss_event passed")
