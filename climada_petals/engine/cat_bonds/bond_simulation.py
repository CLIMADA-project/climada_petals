import pandas as pd
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

class bond_simulation:

    def __init__(self, subarea_calc, term, number_terms):
        self.term = term
        self.simulated_years = number_terms * term
        self.subarea_calc = subarea_calc



    '''Simulate one term of bond to derive losses'''
def init_bond_exp_loss(term, events_per_year, principal):
    """
    Calculates the expected losses for a catastrophe bond over its term.
    This function simulates the bond's loss experience given a sequence of event data per year,
    tracking payouts, remaining nominal value, and the timing of losses. It returns the relative
    losses per year, the total relative loss, and a DataFrame detailing losses and their corresponding months.
    Parameters
    ----------
    term : int
        The term of the bond in years.
    events_per_year : list of pandas.DataFrame
        A list where each element is a DataFrame representing events in a year. Each DataFrame must
        contain at least 'month' and 'pay' columns, where 'pay' is the payout for each event.
    principal : float
        The initial principal value of the bond.
    Returns
    -------
    rel_annual_losses : numpy.ndarray
        Array of relative losses per year (losses divided by principal).
    rel_term_loss : float
        Total relative loss over the bond's term (sum of losses divided by principal).
    rel_monthly_loss : pandas.DataFrame
        DataFrame with columns 'losses' and 'months', detailing the losses and their corresponding
        months for each year.
    """

    losses = []
    rel_monthly_loss = pd.DataFrame(columns=['losses', 'months'])
    current_principal = principal.copy()

    for k in range(term):

        if events_per_year[k].empty:
            sum_payouts = [0]
            months = []
        else:
            events_per_year[k] = events_per_year[k].sort_values(by='month')
            months = events_per_year[k]['month'].tolist()

            sum_payouts = []
            for o in range(len(events_per_year[k])):
                payout = events_per_year[k].loc[events_per_year[k].index[o], 'pay']
                #If there are events in the year, sample that many payouts and the associated damages
                if payout == 0 or current_principal == 0:
                    sum_payouts.append(0)
                elif payout > 0:
                    event_payout = payout 
                    current_principal -= event_payout
                    if current_principal < 0:
                        event_payout += current_principal
                        current_principal = 0
                    else:
                        pass
                    sum_payouts.append(event_payout)

        losses.append(np.sum(sum_payouts))
        rel_monthly_loss.loc[k] = [sum_payouts, months]
    rel_term_loss = np.sum(losses) /principal
    rel_annual_losses = np.array(losses) / principal
    rel_monthly_loss['losses'] = rel_monthly_loss['losses'].apply(lambda x: [i / principal for i in x])
    return rel_annual_losses, rel_term_loss, rel_monthly_loss


'''Loop over all terms of bond to derive losses'''
def init_exp_loss_att_prob_simulation(self):
    """
    Simulates expected annual loss and attachment probability for a catastrophe bond over multiple years.
    This function processes a DataFrame of payout and damage events, simulates bond losses over a specified term,
    and computes risk metrics including Value-at-Risk (VaR) and Expected Shortfall (ES) at 95% and 99% confidence levels.
    It returns the expected annual loss, attachment probability, a DataFrame of monthly losses, and a dictionary of risk metrics.
    Parameters
    ----------
        pay_dam_df (pd.DataFrame): DataFrame containing payout and damage event data.
        nominal (float): The nominal value of the bond.
        print_prob (bool, optional): If True, prints the expected loss and attachment probability. Defaults to True.
    Returns
    -------
        exp_loss_ann (float): Expected annual loss.
        att_prob (float): Annual attachment probability (probability that the bond is triggered).
        df_loss_month (pd.DataFrame): DataFrame containing monthly loss data for all simulations.
        es_metrics (dict): Dictionary containing VaR and ES metrics at 95% and 99% confidence levels for annual and total losses.
    """

    annual_losses = []
    total_losses = []
    list_loss_month = []
    for i in range(self.simulated_years-self.term):
        events_per_year = []
        for j in range(self.term):
            if 'year' in self.subarea_calc.pay_dam_df.columns:
                events_per_year.append(self.subarea_calc.pay_dam_df[self.subarea_calc.pay_dam_df['year'] == (i+j)])
            else:
                events_per_year.append(pd.DataFrame({'pay': [0], 'damage': [0]}))
        annual_losses_per_term, term_loss, monthly_losses = init_bond_exp_loss(self.term, events_per_year, self.subarea_calc.principal)
        list_loss_month.append(monthly_losses)

        annual_losses.extend(annual_losses_per_term)
        total_losses.append(term_loss)
    
    df_loss_month = pd.concat(list_loss_month, ignore_index=True)

    att_prob = annual_losses.count(lambda x: x > 0) / len(annual_losses)
    exp_loss_ann = np.mean(annual_losses)

    annual_losses = pd.Series(annual_losses)
    total_losses = pd.Series(total_losses)

    VaR_99_ann = annual_losses.quantile(0.99)
    VaR_95_ann = annual_losses.quantile(0.95)
    if VaR_99_ann == 1:
        ES_99_ann = 1
    else:
        ES_99_ann = annual_losses[annual_losses > VaR_99_ann].mean()
    if VaR_95_ann == 1:
        ES_95_ann = 1
    else:
        ES_95_ann = annual_losses[annual_losses > VaR_95_ann].mean()

    metrics = {'EL_ann': exp_loss_ann, 'AP_ann': att_prob, 'VaR_99_ann': VaR_99_ann, 'VaR_95_ann': VaR_95_ann,
               'ES_99_ann': ES_99_ann, 'ES_95_ann': ES_95_ann}

    LOGGER.info(f'Expected Loss = {exp_loss_ann}')
    LOGGER.info(f'Attachment Probability = {att_prob}')

    return metrics, df_loss_month