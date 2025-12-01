import pandas as pd
import numpy as np
import logging
from climada_petals.engine.cat_bonds import utils_cat_bonds

logging.basicConfig(
     format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%Y-%m-%d %H:%M",
     level=logging.INFO,
 )
LOGGER = logging.getLogger(__name__)

def test_multi_level_es_basic():
    losses = pd.Series([0, 1, 2, 3, 4, 5])
    alphas = [0.5, 0.8]

    var_list, es_list = utils_cat_bonds.multi_level_es(losses, alphas)

    # VaR checks
    assert np.isclose(var_list[0], losses.quantile(0.5))
    assert np.isclose(var_list[1], losses.quantile(0.8))

    # ES checks (mean of tail losses > VaR)
    es_expected_50 = losses[losses > var_list[0]].mean() # type: ignore
    es_expected_80 = losses[losses > var_list[1]].mean() # type: ignore

    assert np.isclose(es_list[0], es_expected_50)
    assert np.isclose(es_list[1], es_expected_80)


def test_multi_level_es_all_equal_losses():
    losses = pd.Series([1, 1, 1, 1])
    alphas = [0.95]

    var_list, es_list = utils_cat_bonds.multi_level_es(losses, alphas)

    # VaR = 1
    assert var_list[0] == 1

    # ES = 1
    assert es_list[0] == 1


def test_multi_level_es_no_tail_losses():
    losses = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    alphas = [0.9]  # VaR = 0

    var_list, es_list = utils_cat_bonds.multi_level_es(losses, alphas)

    # VaR must be 0
    assert var_list[0] == 0

    # ES must be 1
    assert es_list[0] == 1

if __name__ == "__main__":
    test_multi_level_es_basic()
    LOGGER.info("test_multi_level_es_basic passed")
    test_multi_level_es_all_equal_losses()
    LOGGER.info("test_multi_level_es_all_equal_losses passed")
    test_multi_level_es_no_tail_losses()
    LOGGER.info("test_multi_level_es_no_tail_losses passed")