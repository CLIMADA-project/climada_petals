import pandas as pd
import numpy as np
import logging

from utils_cat_bonds import multi_level_es, allocate_single_payout
import pooling_functions as pf

LOGGER = logging.getLogger(__name__)

class MultiCountryBondSimulation:

    def __init__(self, country_dictionary, term, number_of_terms):
        self.country_dictionary = country_dictionary
        self.term = term
        self.simulated_years = number_of_terms * term
        self._prepare_data()



    def _prepare_data(self):
        self.pay_vs_dam_dic = {}
        self.principal_dic_cty = {}
        self.countries = list(self.country_dictionary.keys())
        min_year_list = []
        for cty, bond_sim_class in self.country_dictionary.items():
            self.pay_vs_dam_dic[cty] = bond_sim_class.subarea_calc.pay_vs_dam
            self.principal_dic_cty[cty] = bond_sim_class.subarea_calc.principal
            min_year_list.append(bond_sim_class.subarea_calc.pay_vs_dam['year'].min())

        min_year = min(min_year_list)

        self.min_year = min_year
        


    @classmethod
    def simulate_bond_pool_n(cls, country_dictionary, term, number_of_terms, principal, number_pools, n_opt_rep=100):
        """
        Class method to create an instance of MultiCountryBondSimulation and run the loss simulation.

        Parameters
        ----------
        n : int
            Number of countries to include in the pool.
        subarea_calc_list : list
            List of subarea_calc instances for each country.
        countries_list : list
            List of country codes.
        term : int
            Term of the bond in years.
        number_of_terms : int
            Number of terms to simulate.
        tranches : list
            List of tranche nominal values.
        principal : float
            Total principal value of the bond.

        Returns
        -------
        bond_simulation : MultiCountryBondSimulation
            Instance of MultiCountryBondSimulation with simulated losses.
        """
        countries_list = list(country_dictionary.keys())
        cls_bond_simulation = [country_dictionary[cty] for cty in countries_list]
        pool_allocation, algorithm_result = pf.process_n_pools(number_pools, countries_list, cls_bond_simulation, n_opt_rep=n_opt_rep)
        pool_dict = {}
        for cty in countries_list:
            print(cty)
            if pool_dict.get(pool_allocation[cty][0]) is None:
                pool_dict[pool_allocation[cty][0]] = []
            pool_dict[pool_allocation[cty][0]].append(cty)
            print(pool_dict)

        mlt_bond_simulation_dic = {}
        for key, countries in pool_dict.items():
            print(key, countries)
            LOGGER.info(f"Pool {key}: Countries {pool_dict[key]}")
            pool_country_dictionary = {cty: country_dictionary[cty] for cty in countries}
            mlt_bond_simulation_dic[key] = cls(pool_country_dictionary, term, number_of_terms)
            mlt_bond_simulation_dic[key].init_loss_simulation(principal)

        return mlt_bond_simulation_dic[key], pool_allocation, algorithm_result



    '''Simulate one term of bond to derive losses'''
    def _init_bond_loss(self, events_per_year, principal):
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
        sum_damages = 0.0
        rel_ann_cty_losses = {country: np.zeros(self.term) for country in self.countries}  
        coverage_cty = {}
        for code in self.countries:
            coverage_cty[code] = {'payout': 0, 'damage': 0}
        
        for k in range(self.term):

            if not events_per_year[k].empty:
                events = events_per_year[k].sort_values(by='month')
                months = events['month'].to_numpy()
                countries = events['country_code'].to_numpy()
                pay = events['pay'].to_numpy()
                damages = events['damage'].to_numpy()
                sum_damages += np.sum(damages)

                sum_payouts = np.zeros(len(events))  
                for payout, country, damage, idx in zip(pay, countries, damages, range(len(events))):

                    if payout == 0 or cur_nominal == 0 or cur_nom_cty[int(country)] == 0:
                        event_payout = 0
                    else:
                        event_payout = payout
                        cur_nom_cty[int(country)] -= event_payout
                        if cur_nom_cty[int(country)] < 0:
                            event_payout += cur_nom_cty[int(country)]
                            cur_nom_cty[int(country)] = 0
                        cur_nominal -= event_payout
                        if cur_nominal < 0:
                            event_payout += cur_nominal
                            cur_nominal = 0

                    sum_payouts[idx] = event_payout
                    coverage_cty[country]['payout'] += event_payout
                    coverage_cty[country]['damage'] += damage
                    rel_ann_cty_losses[country][k] += event_payout / principal

            else:
                sum_payouts = 0
                months = []

            ann_loss[k] = np.sum(sum_payouts)
            loss_month_data.append((sum_payouts, months))

        rel_bond_monthly_losses = pd.DataFrame(loss_month_data, columns=['losses', 'months'])

        rel_ann_bond_losses = list(np.array(ann_loss) / principal)
        rel_bond_monthly_losses['losses'] = rel_bond_monthly_losses['losses'].values / principal
        coverage_tot = {'payout': np.sum(ann_loss), 'damage': sum_damages}

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

        annual_losses = []
        total_losses = []
        list_loss_month = []
        ann_cty_losses = {cty: [] for cty in self.countries}
        coverage = {'payout': 0, 'damage': 0}
        self.tot_coverage_cty = {}
        for cty in self.countries:
            self.tot_coverage_cty[cty] = {'payout': 0.0, 'damage': 0.0, 'coverage': 0.0, 'EL': 0, 'share_EL': 0}

        for i in range(self.simulated_years-self.term):
            events_per_year = []
            for j in range(self.term):
                events_per_cty = []  
                for cty in self.countries:
                    events = self.pay_vs_dam_dic[int(cty)][self.pay_vs_dam_dic[int(cty)]['year'] == (self.min_year+i)+j].copy()
                    events['country_code'] = cty
                    events_per_cty.append(events)  
                year_events_df = pd.concat(events_per_cty, ignore_index=True) if events_per_cty else pd.DataFrame()
                events_per_year.append(year_events_df)

            rel_ann_bond_losses, rel_ann_cty_losses, rel_bond_monthly_losses, coverage_tot, coverage_cty = self._init_bond_loss(events_per_year, principal)

            list_loss_month.append(rel_bond_monthly_losses)
            annual_losses.extend(rel_ann_bond_losses)
            coverage['payout'] += coverage_tot['payout']
            coverage['damage'] += coverage_tot['damage']

            for key in coverage_cty.keys():
                self.tot_coverage_cty[key]['payout'] += coverage_cty[key]['payout']
                self.tot_coverage_cty[key]['damage'] += coverage_cty[key]['damage']

            for key in rel_ann_cty_losses:
                ann_cty_losses[key].extend(rel_ann_cty_losses[key])

        self.df_loss_month = pd.concat(list_loss_month, ignore_index=True)

        att_prob_ann = sum(1 for x in annual_losses if x > 0) / len(annual_losses)
        exp_loss_ann = np.mean(annual_losses)

        annual_losses = pd.Series(annual_losses)
        total_losses = pd.Series(total_losses)

        var_list, es_list = multi_level_es(annual_losses, confidence_levels)

        for key in self.tot_coverage_cty.keys():
            self.tot_coverage_cty[key]['coverage'] = self.tot_coverage_cty[key]['payout'] / self.tot_coverage_cty[key]['damage']
            self.tot_coverage_cty[key]['EL'] = np.mean(ann_cty_losses[key])
            self.tot_coverage_cty[key]['share_EL'] = self.tot_coverage_cty[key]['EL'] / exp_loss_ann
        


        self.loss_metrics = {'EL_ann': exp_loss_ann,
                             'AP_ann': att_prob_ann,
                             'Payout': coverage['payout'], 
                             'Damage': coverage['damage']}
        
        for cl, var, es in zip(confidence_levels, var_list, es_list):
            self.loss_metrics[f'VaR_{int(cl*100)}_ann'] = var
            self.loss_metrics[f'ES_{int(cl*100)}_ann'] = es

        LOGGER.info(f'Expected Loss = {exp_loss_ann}')
        LOGGER.info(f'Attachment Probability = {att_prob_ann}')


    '''reduced function to derive returns of the bond -> was used to save time during calculation'''
    def init_return_simulation(self, premium, rf=0.0):
        """
        Simulates the net cash flows (NCF) and premium allocations for a multi-country catastrophe bond structure over the simiulation period.
        This function calculates the premium payments, net cash flows, and premium allocations for the whole bond and all countries, 
        considering monthly losses and country exposure shares. It accounts for loss events, premium payments, 
        and risk-free rates, and distributes premiums according to country exposure shares represented by the country's Expected Marginal Loss.

        Parameters
        ----------
        self: mlt_bond_simulation
            Class instance of mlt_bond_simulation containing monthly loss data, country exposure shares, and the term of the bond.
        premium : float
            The annual premium rate for the bond.
        rf : float, optional
            Risk-free rate to be added to the premium (default is 0.0).

        Returns
        -------
        ncf : pandas.DataFrame
            DataFrame containing net cash flows for the bond.
        prem_cty_df : pandas.DataFrame
            DataFrame containing premium allocations for each country (based on their exposure share) and total premiums.

        Notes
        -----
        - The function resets the nominal value at the end of each term.
        - Premiums and cash flows are calculated monthly, accounting for loss events and remaining nominal.
        """

        premiums_tot = []
        ncf_tot = []
        cur_nominal = 1
        for i in range(len(self.df_loss_month)):
            losses = self.df_loss_month['losses'].iloc[i]
            months = self.df_loss_month['months'].iloc[i]
            if np.sum(losses) == 0:
                premiums_tot.append(cur_nominal * premium)
                ncf_tmp = cur_nominal * (premium + rf)
                ncf_tot.append(ncf_tmp)
            else:
                ncf_tot_tmp = []
                premiums_tot_tmp = []
                premiums_tot_tmp.append(cur_nominal * premium / 12 * months[0])
                prem_pre_tmp = cur_nominal * (premium + rf) / 12 * months[0]
                ncf_tot_tmp.append(prem_pre_tmp)
                for j in range(len(losses)):
                    loss = losses[j]
                    month = months[j]
                    cur_nominal -= loss
                    if cur_nominal < 0:
                        loss += cur_nominal
                        cur_nominal = 0
                    if j + 1 < len(losses):
                        nex_month = months[j+1]
                        premiums_tot_tmp.append(cur_nominal * premium / 12 * (nex_month - month))
                        prem_tmp = ((cur_nominal * (premium + rf)) / 12 * (nex_month - month))
                        ncf_tot_tmp.append(prem_tmp - loss)
                    else:
                        premiums_tot_tmp.append(cur_nominal * premium / 12 * (12 - month))
                        prem_tmp = ((cur_nominal * (premium + rf)) / 12 * (12- month))
                        ncf_tot_tmp.append(prem_tmp - loss)
                ncf_tot.append(np.sum(ncf_tot_tmp))
                premiums_tot.append(np.sum(premiums_tot_tmp))
            if (i + 1) % self.term == 0:
                cur_nominal = 1
        prem_cty_dic = {country: [] for country in self.tot_coverage_cty}
        for country in prem_cty_dic:
            prem_cty_dic[country] = np.array(premiums_tot) * self.tot_coverage_cty[country]['share_EL']
        prem_cty_dic['Total'] = premiums_tot
        
        self.ncf = pd.DataFrame(ncf_tot, columns=['Total'])
        self.prem_cty_df = pd.DataFrame(prem_cty_dic)


  
    '''reduced function to derive returns of the bond -> was used to save time during calculation'''
    def init_return_simulation_tranches(self, premiums, tranches, rf=0.0):
        """
        Simulates the net cash flows (NCF) and premium allocations for a multi-country catastrophe bond structure over the simiulation period.
        This function calculates the premium payments, net cash flows, and premium allocations for each tranche and country, 
        considering monthly losses, tranche structures, and country exposure shares. It accounts for loss events, premium payments, 
        and risk-free rates, and distributes premiums according to country exposure shares represented by the country's Expected Marginal Loss.

        Parameters
        ----------
        self: mlt_bond_simulation
            Class instance of mlt_bond_simulation containing monthly loss data, country exposure shares, tranche structures, and the term of the bond.
        premiums : float
            List of annual premium rates for each tranche.
        rf : float, optional
            Risk-free rate to be added to the premium (default is 0.0).

        Returns
        -------
        ncf : pandas.DataFrame
            DataFrame containing net cash flows for each tranche and the total across all tranches for each period.
        prem_cty_df : pandas.DataFrame
            DataFrame containing premium allocations for each country (based on their exposure share), total premiums (if bond is priced as one), and alternative total premiums (if each tranche is priced seperately).

        Notes
        -----
        - The function resets the nominal value at the end of each term.
        - Premiums and cash flows are calculated monthly, accounting for loss events and remaining nominal.
        - Alternative premium calculation is provided for country-level allocation.
        - Losses are allocated to tranches in reverse order (from highest to lowest risk).
        """

        ncf = {str(tranche): [] for tranche in tranches}
        premiums_tot = []
        cur_nominal_tranches = tranches.copy()
        for i in range(len(self.df_loss_month)):
            losses = self.df_loss_month['losses'].iloc[i]
            months = self.df_loss_month['months'].iloc[i]
            if np.sum(losses) == 0:
                prem_it_alt = 0
                for k, tranche in enumerate(tranches):
                    ncf[str(tranche)].append(cur_nominal_tranches[k] * (premiums[k] + rf))
                    prem_it_alt +=  cur_nominal_tranches[k] * premiums[k]
                premiums_tot.append(prem_it_alt)
            else:
                ncf_tmp = {str(tranche): [] for tranche in tranches}
                prem_it_alt = 0
                premiums_tot_tmp = []
                for k, tranche in enumerate(tranches):
                    ncf_tmp[str(tranche)].append(cur_nominal_tranches[k] * (premiums[k] + rf) / 12 * months[0])
                    prem_it_alt += cur_nominal_tranches[k] * premiums[k] / 12 * months[0]
                premiums_tot_tmp.append(prem_it_alt)
                losses_per_tranche = np.zeros(len(tranches))  # accumulate over all events in this period
                for j in range(len(losses)):
                    loss = losses[j]
                    month = months[j]
                    cur_nominal_tranches, payout_per_tranche = allocate_single_payout(loss, cur_nominal_tranches)
                    losses_per_tranche += payout_per_tranche
                    if j + 1 < len(losses):
                        nex_month = months[j+1]
                        prem_it_alt = 0
                        for k, tranche in enumerate(tranches):
                            ncf_tmp[str(tranche)].append(((cur_nominal_tranches[k] * (premiums[k] + rf)) / 12 * (nex_month - month)))
                            prem_it_alt += cur_nominal_tranches[k] * premiums[k] / 12 * (nex_month - month)
                        premiums_tot_tmp.append(prem_it_alt)
                    else:
                        prem_it_alt = 0
                        for k, tranche in enumerate(tranches):
                            ncf_tmp[str(tranche)].append(((cur_nominal_tranches[k] * (premiums[k] + rf)) / 12 * (12- month)))
                            prem_it_alt += cur_nominal_tranches[k] * premiums[k] / 12 * (12- month)
                        premiums_tot_tmp.append(prem_it_alt)
                premiums_tot.append(np.sum(premiums_tot_tmp))
                for idx, tranche in enumerate(tranches):
                    ncf[str(tranche)].append(np.sum(ncf_tmp[str(tranche)]) - losses_per_tranche[idx])
            if (i + 1) % self.term == 0:
                cur_nominal_tranches = tranches.copy()

        prem_cty_dic = {country: [] for country in self.tot_coverage_cty}
        for country in prem_cty_dic:
            prem_cty_dic[country] = np.array(premiums_tot) * self.tot_coverage_cty[country]['share_EL']
        prem_cty_dic['Total'] = premiums_tot

        self.ncf_tranches = pd.DataFrame(ncf)
        self.ncf_tranches['Total'] = self.ncf_tranches.sum(axis=1)
        self.prem_cty_df_tranches = pd.DataFrame(prem_cty_dic)


    '''Calculates required nominal for multi-country bonds -> derives maximal loss over simulation period'''
    def init_required_principal(self):
        """
        Calculates the required nominal value for a multi-country catastrophe bond based on simulated event losses.
        This function simulates event losses over a specified term for multiple countries, aggregates the losses,
        and determines the maximum total loss across all simulation periods. The required nominal value is the
        maximum loss observed, which can be used to set the bond's principal.
        Args:
            countries (list): List of country codes to include in the simulation.
            pay_dam_df_dic (dict): Dictionary mapping country codes to pandas DataFrames containing event loss data.
                Each DataFrame must have a 'year' column and relevant loss information.
            nominal_dic_cty (dict): Dictionary mapping country codes to their respective nominal values.
        Returns:
            float: The required nominal value for the catastrophe bond, equal to the maximum simulated total loss.
        """

        total_losses = []

        for i in range(self.simulated_years-self.term):
            events_per_year = []
            for j in range(self.term):
                events_per_cty = [self.pay_vs_dam_dic[int(cty)].loc[self.pay_vs_dam_dic[int(cty)]['year'] == (self.min_year + i) + j].assign(country_code=cty) for cty in self.countries]

                year_events_df = pd.concat(events_per_cty, ignore_index=True) if events_per_cty else pd.DataFrame()
                events_per_year.append(year_events_df)

            tot_loss = self._init_equ_nom_sim(events_per_year, self.principal_dic_cty)

            total_losses.append(tot_loss)

        self.requ_principal = np.max(total_losses)

    
    '''derives losses for one term of bond'''
    def _init_equ_nom_sim(self, events_per_year, nominal_dic_cty):
        """
        Simulates total losses for a multi-country catastrophe bond over a specified term.
        For each year in the bond's term, the function processes a list of event dataframes, each containing
        payout amounts and country codes. It calculates the payouts for each event, ensuring that payouts do not
        exceed the remaining nominal value for each country. The annual losses are accumulated, and the total loss
        over the term is returned.

        Parameters
        ----------
        events_per_year : list of pandas.DataFrame
            A list where each element corresponds to a year and contains a DataFrame of events with columns
            'pay' (payout amount) and 'country_code' (identifier for the country).
        nominal_dic_cty : dict
            Dictionary mapping country codes to their initial nominal values for the bond.

        Returns
        -------
        tot_loss : float
            The total loss over the bond's term, accounting for country-specific nominal limits.
        """

        ann_loss = np.zeros(self.term)
        cur_nom_cty = nominal_dic_cty.copy()

        for k in range(self.term):
            if not events_per_year[k].empty:
                events = events_per_year[k]
                payouts = events['pay'].to_numpy()
                cties = events['country_code'].to_numpy()

                sum_payouts = np.zeros(len(events))

                for idx, (payout, cty) in enumerate(zip(payouts, cties)):
                    if payout == 0 or cur_nom_cty[cty] == 0:
                        event_payout = 0
                    else:
                        event_payout = payout
                        cur_nom_cty[cty] -= event_payout
                        if cur_nom_cty[cty] < 0:
                            event_payout += cur_nom_cty[cty]
                            cur_nom_cty[cty] = 0
                    sum_payouts[idx] = event_payout

                ann_loss[k] = np.sum(sum_payouts)
            else:
                ann_loss[k] = 0

        tot_loss = np.sum(ann_loss)
        return tot_loss
    