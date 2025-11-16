import pandas as pd
import numpy as np
import logging

from utils_cat_bonds import multi_level_es

LOGGER = logging.getLogger(__name__)

class mlt_bond_simulation:

    def __init__(self, subarea_calc_list, countries_list, term, number_of_terms, tranches):
        self.countries = countries_list
        self.term = term
        self.simulated_years = number_of_terms * term
        self.tranches = tranches
        self.subarea_calc = subarea_calc_list



    def _prepare_data(self):
        self.pay_vs_dam_dic = {}
        self.principal_dic_cty = {}
        min_year_list = []
        for idx, cty in enumerate(self.countries):
            self.pay_vs_dam_dic[cty] = self.subarea_calc[idx].pay_vs_dam
            self.principal_dic_cty[cty] = self.subarea_calc[idx].principal
            min_year_list.append(self.subarea_calc[idx].pay_vs_dam['year'].min())

        min_year = min(min_year_list)

        return min_year
        



    '''Simulate one term of bond to derive losses'''
    def init_bond_loss(self, events_per_year, principal):
        '''
        Simulates the expected losses and payouts for a multi-country catastrophe bond over its term.
        This function iterates over each year (term) and processes event data for each country, calculating
        payouts and damages based on the provided nominal values and per-country nominal allocations. It tracks
        losses, damages, and payouts for each country and for the bond as a whole, and computes several summary
        statistics.
        Parameters
        ----------
        self: mlt_bond_simulation
            A class instance Dictionary mapping country codes to their allocated nominal values.
        principal : float
            The total principal value of the catastrophe bond.
        events_per_year : list of pandas.DataFrame
            List of DataFrames, one per year of the bond's term, each containing event data with columns:
            'month', 'country_code', 'pay', and 'damage'.
        Returns
        -------
        rel_ann_bond_losses : list of floats
            List of relative annual losses (as a fraction of the total principal) for each year of the bond's term.
        rel_ann_cty_losses : dict
            Dictionary mapping country codes to arrays of relative annual losses for each year.
        rel_bond_monthly_losses : pandas.DataFrame
            DataFrame containing, for each year, the array of event payouts ('losses') and corresponding months ('months'),
            both normalized by the total principal.
        coverage_tot : dict
            Dictionary with total payout and total damage over the bond's term: {'payout': ..., 'damage': ...}.
        coverage_cty : dict
            Dictionary mapping country codes to their cumulative payout and damage over the bond's term:
            {country_code: {'payout': ..., 'damage': ...}, ...}.
        Notes
        -----
        - The function assumes that the term (number of years) is inferred from the length of `events_per_year`.
        - Payouts are capped by the remaining principal value for the bond and by the per-country princpal allocation.
        - All losses and payouts are normalized by the total principal value before being returned.
        '''
        ann_loss = np.zeros(self.term)  
        loss_month_data = []
        cur_nominal = principal
        cur_nom_cty = self.principal_dic_cty.copy() 
        tot_damage = []
        rel_ann_cty_losses = {country: np.zeros(self.term) for country in self.countries}  
        coverage_cty = {}
        for code in self.countries:
            coverage_cty[code] = {'payout': 0, 'damage': 0}

        for k in range(self.term):
            cty_losses_event = {country: [] for country in self.countries}
            cty_damages_event = {country: [] for country in self.countries}
            sum_payouts = np.zeros(len(events_per_year[k]))

            if not events_per_year[k].empty:
                events = events_per_year[k].sort_values(by='month')
                months = events['month'].to_numpy()
                cties = events['country_code'].to_numpy()
                pay = events['pay'].to_numpy()
                dam = events['damage'].to_numpy()

                sum_payouts = np.zeros(len(events))  
                sum_damages = np.zeros(len(events)) 
                for payout, country, damage in zip(pay, countries, damages):

                    if payout == 0 or cur_nominal == 0 or cur_nom_cty[int(cty)] == 0:
                        event_payout = 0
                    else:
                        event_payout = payout
                        cur_nom_cty[int(cty)] -= event_payout
                        if cur_nom_cty[int(cty)] < 0:
                            event_payout += cur_nom_cty[int(cty)]
                            cur_nom_cty[int(cty)] = 0
                        cur_nominal -= event_payout
                        if cur_nominal < 0:
                            event_payout += cur_nominal
                            cur_nominal = 0

                    sum_payouts[o] = event_payout
                    sum_damages[o] = damage
                    cty_losses_event[cty].append(event_payout)
                    cty_damages_event[cty].append(damage)
                losses = np.sum(sum_payouts)
                damages = np.sum(sum_damages)
                for cty, cty_loss in cty_losses_event.items():
                    rel_ann_cty_losses[cty][k] = np.sum(cty_loss)
                    coverage_cty[cty]['payout'] += sum(cty_losses_event[cty])
                    coverage_cty[cty]['damage'] += sum(cty_damages_event[cty])
            else:
                losses = 0
                damages = 0
                months = []

            ann_loss[k] = losses
            tot_damage.append(damages)
            loss_month_data.append((sum_payouts, months))

        rel_bond_monthly_losses = pd.DataFrame(loss_month_data, columns=['losses', 'months'])

        rel_ann_bond_losses = list(np.array(ann_loss) / principal)
        for key in rel_ann_cty_losses.keys():
            rel_ann_cty_losses[key] = rel_ann_cty_losses[key] / principal 
        rel_bond_monthly_losses['losses'] = rel_bond_monthly_losses['losses'].values / principal
        coverage_tot = {'payout': np.sum(ann_loss), 'damage': np.sum(tot_damage)}
        return rel_ann_bond_losses, rel_ann_cty_losses, rel_bond_monthly_losses, coverage_tot, coverage_cty


    '''Loop over all terms of bond to derive losses'''
    def init_loss_simulation(self, principal, confidence_levels=[0.95, 0.99]):
        """
        Simulates expected loss and attachment probability for a multi-country catastrophe bond over simulation period.
        This function aggregates event data for multiple countries over a specified simulation period, computes annual and total losses,
        calculates risk metrics (Value-at-Risk and Expected Shortfall) at given confidence levels, and evaluates coverage and expected loss
        shares for each country. It also computes the probability that the bond is triggered (attachment probability) and can print summary statistics.
        Parameters
        ----------
        self: mlt_bond_simulation
            A class instance containing a list of countrie codes and a list of subarea_calc classes with principal values, and pay_vs_dam tables.
        principal : float
            The total principal value of the catastrophe bond.
        confidence_levels : list, optional
            List of confidence levels (floats between 0 and 1) for risk metrics calculation (default is [0.95, 0.99]).
        Returns
        -------
        df_loss_month : pandas.DataFrame
            DataFrame containing monthly relative losses for the entire bond.
        loss_metrics : dict
            Dictionary containing expected annual loss, annual attachment probability, payout, damage, and risk metrics (VaR and ES) at specified confidence levels for annual losses.
        tot_coverage_cty : dict
            Dictionary mapping each country code to its total payout, damage, coverage ratio, annual expected loss, and share of annual expected loss.
        Notes
        -----
        - The function relies on the helper functions `init_bond_loss` and `multi_level_es` for loss simulation and risk metric calculation.
        - The function expects event data to be structured such that each country's DataFrame contains a 'year' and 'month' column for filtering events.

        """

        min_year = self._prepare_data()

        annual_losses = []
        total_losses = []
        list_loss_month = []
        ann_cty_losses = {cty: [] for cty in self.countries}
        coverage = {'payout': 0, 'damage': 0}
        self.tot_coverage_cty = {}
        for cty in self.countries:
            self.tot_coverage_cty[cty] = {'payout': [], 'damage': [], 'coverage': [], 'EL': 0, 'share_EL': 0}

        for i in range(self.simulated_years-self.term):
            events_per_year = []
            for j in range(self.term):
                events_per_cty = []  
                for cty in self.countries:
                    events = self.pay_vs_dam_dic[int(cty)][self.pay_vs_dam_dic[int(cty)]['year'] == (min_year+i)+j].copy()
                    events['country_code'] = cty
                    events_per_cty.append(events)  
                year_events_df = pd.concat(events_per_cty, ignore_index=True) if events_per_cty else pd.DataFrame()
                events_per_year.append(year_events_df)

            rel_ann_bond_losses, rel_ann_cty_losses, rel_bond_monthly_losses, coverage_tot, coverage_cty = self.init_bond_loss(events_per_year, principal)

            list_loss_month.append(rel_bond_monthly_losses)
            annual_losses.extend(rel_ann_bond_losses)
            coverage['payout'] += coverage_tot['payout']
            coverage['damage'] += coverage_tot['damage']

            for key in coverage_cty.keys():
                self.tot_coverage_cty[key]['payout'].append(coverage_cty[key]['payout'])
                self.tot_coverage_cty[key]['damage'].append(coverage_cty[key]['damage'])

            for key in rel_ann_cty_losses:
                ann_cty_losses[key].extend(rel_ann_cty_losses[key])

        self.df_loss_month = pd.concat(list_loss_month, ignore_index=True)

        att_prob_ann = sum(1 for x in annual_losses if x > 0) / len(annual_losses)
        exp_loss_ann = np.mean(annual_losses)

        annual_losses = pd.Series(annual_losses)
        total_losses = pd.Series(total_losses)

        risk_metrics_annual = multi_level_es(annual_losses, confidence_levels)

        for key in self.tot_coverage_cty.keys():
            self.tot_coverage_cty[key]['payout'] = sum(self.tot_coverage_cty[key]['payout'])
            self.tot_coverage_cty[key]['damage'] = sum(self.tot_coverage_cty[key]['damage'])
            self.tot_coverage_cty[key]['coverage'] = self.tot_coverage_cty[key]['payout'] / self.tot_coverage_cty[key]['damage']
            self.tot_coverage_cty[key]['EL'] = np.mean(ann_cty_losses[key])

        for key in self.tot_coverage_cty:
            self.tot_coverage_cty[key]['share_EL'] = self.tot_coverage_cty[key]['EL'] / exp_loss_ann
        


        self.loss_metrics = {'EL_ann': exp_loss_ann,
                             'AP_ann': att_prob_ann,
                             'Payout': coverage['payout'], 
                             'Damage': coverage['damage'], 
                             'VaR_99_ann': risk_metrics_annual[0.99]['VaR'], 
                             'VaR_95_ann': risk_metrics_annual[0.95]['VaR'], 
                             'ES_99_ann': risk_metrics_annual[0.99]['ES'], 
                             'ES_95_ann': risk_metrics_annual[0.95]['ES']}

        LOGGER.info(f'Expected Loss = {exp_loss_ann}')
        LOGGER.info(f'Attachment Probability = {att_prob_ann}')


  