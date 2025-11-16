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