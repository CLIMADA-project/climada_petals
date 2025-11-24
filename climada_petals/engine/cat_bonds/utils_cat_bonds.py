import numpy as np

'''Calculate value at risk and expected shorfall for various alphas'''
def multi_level_es(losses, confidence_levels):
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES) for multiple confidence levels.

    Parameters:
    - losses: array-like, list of losses
    - confidence_levels: list of floats, confidence levels (e.g., [0.95, 0.99])

    Returns:
    - risk_metrics: dict, VaR and ES values keyed by confidence level
    """

    # Compute VaR and ES
    var_list = [losses.quantile(confidence_level) for confidence_level in confidence_levels]

    # Avoid empty slices by using conditional logic
    es_list = [
        1 if var == 1 else losses[losses > var].mean()
        for var in var_list
    ]

    return var_list, es_list


def allocate_single_payout(payout, nominals):
    """
    Vectorised allocation of one payout across tranche nominals (FIFO).
    
    Parameters
    ----------
    payout : float
    nominals : 1D array of tranche nominal values

    Returns
    -------
    alloc : array of size (T,)  -- how much each tranche pays
    remaining_nominals : array -- leftover nominals after the payout
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