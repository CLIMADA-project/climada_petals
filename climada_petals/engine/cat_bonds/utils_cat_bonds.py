
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
    # Convert losses to a NumPy array
    losses = np.array(losses)
    # Sort losses once
    sorted_losses = np.sort(losses)
    n = len(sorted_losses)
    risk_metrics = {}
    for cl in confidence_levels:
        # Calculate index for VaR
        var_index = int(np.ceil(n * cl)) - 1
        var = sorted_losses[var_index]
        # Calculate ES
        tail_losses = sorted_losses[var_index + 1:]
        es = tail_losses.mean() if len(tail_losses) > 0 else var
        # Store metrics
        risk_metrics[cl] = {'VaR': var, 'ES': es}
    return risk_metrics