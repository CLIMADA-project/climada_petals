import numpy as np
import pandas as pd

"""Utility functions for catastrophe bond analytics."""
def multi_level_es(losses: pd.Series, confidence_levels: list[float]):
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES) for multiple confidence levels.

    Parameters
    ----------
    losses : pandas.Series
        Loss samples used to estimate tail risk metrics.
    confidence_levels : list[float]
        Confidence levels (e.g., [0.95, 0.99]).

    Returns
    -------
    var_list : list[float]
        VaR values in the order of the given confidence levels.
    es_list : list[float]
        ES values in the order of the given confidence levels.
    """

    # Compute VaR and ES
    var_list = [losses.quantile(confidence_level) for confidence_level in confidence_levels]

    # Avoid empty slices by using conditional logic
    es_list = [
        1 if var == 1 else losses[losses > var].mean()
        for var in var_list
    ]

    return var_list, es_list


def allocate_single_payout(payout: float, nominals: np.ndarray):
    """
    Allocate a single payout across tranche nominals.
    
    Parameters
    ----------
    payout : float
        Payout amount to allocate.
    nominals : array-like
        1D array of tranche nominal values.

    Returns
    -------
    remaining_nominals : numpy.ndarray
        Remaining nominal values after the payout.
    payout_per_tranche : numpy.ndarray
        Amount allocated to each tranche.
    """

    nominals = np.asarray(nominals, float)

    # cumulative nominal capacity per tranche
    cum_nom = np.cumsum(nominals)
    cum_nom_prev = cum_nom - nominals

    # intersection of [0, payout] with each tranche interval [cum_nom_prev, cum_nom]
    payout_per_tranche = np.minimum(cum_nom, payout) - np.maximum(cum_nom_prev, 0)

    # clip negative / unused intervals
    payout_per_tranche = np.clip(payout_per_tranche, 0, None)

    remaining_nominals = nominals - payout_per_tranche


    return remaining_nominals, payout_per_tranche
