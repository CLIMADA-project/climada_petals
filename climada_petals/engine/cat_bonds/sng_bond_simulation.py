import pandas as pd
import numpy as np
import logging
from utils_cat_bonds import multi_level_es

LOGGER = logging.getLogger(__name__)

class sng_bond_simulation:

    def __init__(self, subarea_calc, term, number_of_terms):
        self.term = term
        self.simulated_years = number_of_terms * term
        self.subarea_calc = subarea_calc



    '''Simulate one term of bond to derive losses'''
    def init_bond_loss(self, events_per_year):
        """
        Calculates the expected losses for a catastrophe bond over its term.
        This function simulates the bond's loss experience given a sequence of event data per year,
        tracking payouts, damages, remaining princpal value, and the timing of losses. It returns the relative
        losses per year, the total payouts and damages per term, and a DataFrame detailing losses and their corresponding months.

        Parameters
        ----------
        self : bond_simulation
            An instance of the bond_simulation class containing a payout vs damage table, bond term, and the principal.
        events_per_year : list of pandas.DataFrame
            A list where each element is a DataFrame representing events in a year. Each DataFrame must
            contain at least 'month' and 'pay' columns, where 'pay' is the payout for each event.
        Returns
        -------
        rel_annual_losses : numpy.ndarray
            Array of relative payouts/losses per year (losses divided by principal).
        rel_monthly_loss : pandas.DataFrame
            DataFrame with columns 'losses' and 'months', detailing the losses and their corresponding
            months for each year.
        summed_payouts : float
            The total summed payouts over the bond's term.
        summed_damages : float
            The total summed damages over the bond's term.
        """

        losses = []
        rel_monthly_loss = pd.DataFrame(columns=['losses', 'months'])
        current_principal = self.subarea_calc.principal

        summed_damages = 0
        for k in range(self.term):

            if events_per_year[k].empty:
                sum_payouts = [0]
                months = []
            else:
                events_per_year[k] = events_per_year[k].sort_values(by='month')
                months = events_per_year[k]['month'].tolist()

                sum_payouts = []
                for o in range(len(events_per_year[k])):
                    payout = events_per_year[k].loc[events_per_year[k].index[o], 'pay']
                    summed_damages += events_per_year[k].loc[events_per_year[k].index[o], 'damage']
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
        summed_payouts = np.sum(losses)
        rel_annual_losses = np.array(losses) / self.subarea_calc.principal
        rel_monthly_loss['losses'] = rel_monthly_loss['losses'].apply(lambda x: [i / self.subarea_calc.principal for i in x])
        return rel_annual_losses, rel_monthly_loss, summed_payouts, summed_damages


    def init_loss_simulation(self, confidence_levels=[0.95, 0.99]):
        """
        Simulate losses, payouts, damages, and risk metrics for a catastrophe bond.

        Returns
        -------
        df_loss_month : pd.DataFrame
            Monthly loss data for all simulations.
        loss_metrics : dict
            Expected loss, attachment probability, total payouts/damages,
            VaR and ES metrics for given confidence levels.
        """

        pay_vs_dam = self.subarea_calc.pay_vs_dam
        min_year = pay_vs_dam['year'].min()

        annual_losses = []
        list_loss_month = []
        total_payouts = 0
        total_damages = 0

        # Iterate directly over year-starts
        for start_year in range(min_year, min_year + self.simulated_years - self.term):

            # Collect events for the full term (vectorized selection)
            events_per_year = [
                pay_vs_dam[pay_vs_dam['year'] == (start_year + offset)]
                for offset in range(self.term)
            ]

            ann_losses_term, monthly_losses, summed_payouts, summed_damages = (
                self.init_bond_loss(events_per_year)
            )

            annual_losses.extend(ann_losses_term)
            list_loss_month.append(monthly_losses)
            total_payouts += summed_payouts
            total_damages += summed_damages

        # Combine monthly losses
        self.df_loss_month = pd.concat(list_loss_month, ignore_index=True)

        annual_losses = pd.Series(annual_losses)
        exp_loss_ann = annual_losses.mean()
        att_prob = (annual_losses > 0).mean()

        # Save metrics
        self.loss_metrics = {
            'EL_ann': exp_loss_ann,
            'AP_ann': att_prob,
            'Tot_payout': total_payouts,
            'Tot_damages': total_damages,
        }

        var_list, es_list = multi_level_es(annual_losses, confidence_levels)

        for cl, var, es in zip(confidence_levels, var_list, es_list):
            self.loss_metrics[f'VaR_{int(cl*100)}_ann'] = var
            self.loss_metrics[f'ES_{int(cl*100)}_ann'] = es

        LOGGER.info(f'Expected Loss = {exp_loss_ann}')
        LOGGER.info(f'Attachment Probability = {att_prob}')


    '''Simulate over all terms of bond to derive returns'''
    def init_return_simulation(self, premium):
        """
        Simulates the performance of a catastrophe bond over the simulation period, premiums and returns.
        This function models the bond's payouts, premiums, and returns over a series of simulated years.
        It aggregates annual and total returns and computes Sharpe ratios.

        Parameters
        ----------
            self: bond_simulation
                An instance of the bond_simulation class containing monthly loss data, premium rate, and term. 
        Returns
        -------
            return_metrics (pd.DataFrame): DataFrame containing annual premiums, annual returns, total returns, and total premiums for the bond.
        """

        premiums_tot = []
        ncf_tot = []
        cur_nominal = 1
        for i in range(len(self.df_loss_month)):
            losses = self.df_loss_month['losses'].iloc[i]
            months = self.df_loss_month['months'].iloc[i]
            if np.sum(losses) == 0:
                prem_tmp = cur_nominal * premium
                premiums_tot.append(prem_tmp)
                ncf_tot.append(prem_tmp)
            else:
                ncf_tot_tmp = []
                premiums_tot_tmp = []
                prem_tmp = cur_nominal * premium / 12 * months[0]
                premiums_tot_tmp.append(prem_tmp)
                ncf_tot_tmp.append(prem_tmp)
                for j in range(len(losses)):
                    loss = losses[j]
                    month = months[j]
                    cur_nominal -= loss
                    if cur_nominal < 0:
                        loss += cur_nominal
                        cur_nominal = 0
                    else:
                        pass
                    if j + 1 < len(losses):
                        next_month = months[j+1]
                        prem_tmp = ((cur_nominal * premium) / 12 * (next_month - month))
                        premiums_tot_tmp.append(prem_tmp)
                        ncf_tot_tmp.append(prem_tmp - loss)
                    else:
                        prem_tmp = ((cur_nominal * premium) / 12 * (12- month))
                        premiums_tot_tmp.append(prem_tmp)
                        ncf_tot_tmp.append(prem_tmp - loss)
                ncf_tot.append(np.sum(ncf_tot_tmp))
                premiums_tot.append(np.sum(premiums_tot_tmp))
            if (i + 1) % self.term == 0:
                cur_nominal = 1

        sharpe_ratio = (np.mean(ncf_tot) / np.std(ncf_tot)) if np.std(ncf_tot) != 0 else np.nan
    
        self.return_metrics = {'annual_premiums': np.array(premiums_tot), 'annual_returns': np.array(ncf_tot),
                               'total_returns': np.sum(np.array(ncf_tot)) * self.subarea_calc.principal , 'total_premiums': np.sum(np.array(premiums_tot)) * self.subarea_calc.principal,
                               'sharpe_ratio': sharpe_ratio}
