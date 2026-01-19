import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit, minimize

from climada_petals.util.config import LOGGER

# path to data folder of ibrd cat bond data
DATA_DIR = (Path(__file__).parent.parent.parent).joinpath('data/cat_bonds')


# regression coefficients for chatoro premium calculation (extracted from Chatoro et al., 2022)
b_0 = -0.5907
b_1 = 1.3986
b_2 = 2.2520
b_3 = 0.0377
b_4 = 0.4613
b_5 = -0.0239
b_6 = -2.6742
b_7 = 0.7057

class PremiumCalculations:
    """
    Premium calculation utilities for catastrophe bond simulations.

    This class provides premium estimation methods based on Chatoro regression,
    IBRD bond curve fitting, and Sharpe ratio benchmarks.

    Parameters
    ----------
    bond_simulation_class : object
        Bond simulation instance exposing `loss_metrics`, `df_loss_month`,
        and `term` attributes.
    """

    def __init__(self, bond_simulation_class):
        """
        Initialize with a bond simulation object.

        Parameters
        ----------
        bond_simulation_class : object
            Bond simulation instance exposing `loss_metrics`, `df_loss_month`,
            and `term` attributes.
        """
        self.bond_simulation_class = bond_simulation_class

    ### CHATORO-PRICING ###
    def calc_chatoro_premium(self, peak_multi: int, investment_graded: int, hybrid_trigger: int, GCIndex = None, BBSpread = None):
        """
        Calculate premium using the Chatoro et al. (2022) regression model.

        Parameters
        ----------
        peak_multi : float
            Indicator wether the cat bond is considered peak peril or if it is covering multiple perils. If either is ture = 1, else 0.
        investment_graded : int 
            Indicator for investment-grade status (e.g., 1 for yes, 0 for no).
        hybrid_trigger : int 
            Indicator for hybrid trigger structure (e.g., 1 for yes, 0 for no).
        GCIndex : float, optional
            Guy Carpenter Global Property Catastrophe Rate on Line Index value.
        BBSpread : float, optional
            ICE BofA BB US High Yield Index OAS value.

        Sets
        -------
        chatoro_prem_rate: float
            Calculated Chatoro premium rate.
        """

        if GCIndex is None:
            GCIndex = 180 #Guy Carpenter Global Property Catastrophe Rate on Line Index (January, 2025)
            LOGGER.info(f'Using default GCIndex value of {GCIndex}')
        else:
            pass
        if BBSpread is None:
            BBSpread = 1.6 #ICE BofA BB US High Yield Index Option-Adjusted Spread (January, 2025)
            LOGGER.info(f'Using default BBSpread value of {BBSpread}')
        else:
            pass

        self.chatoro_prem_rate = (b_0 + b_1 * self.bond_simulation_class.loss_metrics['EL_ann'] * 100 + b_2 * peak_multi + 
                                  b_3 * GCIndex + b_4 * BBSpread + b_5 * self.bond_simulation_class.term * 12 + 
                                  b_6 * investment_graded + b_7 * hybrid_trigger) / 100
        LOGGER.info(f'Calculated Chatoro premium rate: {self.chatoro_prem_rate}')
        


    ### IBRD-PRICING ###
    def monoExp(self, x, a, k, b):
        """
        Exponential function for risk multiple curve fitting.

        Parameters
        ----------
        x : float or np.ndarray
            Input value(s) for the expected loss.
        a : float
            Amplitude parameter.
        k : float
            Decay rate parameter.
        b : float
            Offset parameter.

        Returns
        -------
        float or np.ndarray
            Evaluated exponential function values.
        """
        return a * np.exp(-k * x) + b

    def calc_ibrd_premium(self, peril=None, year=None):
        """
        Fits a monotonic exponential curve to catastrophe bond data for bonds issued by the World Bank to estimate premium parameters.
        This function loads IBRD bond data from an Excel file, optionally filters the data by peril type or issuing year,
        and fits a monotonic exponential function to the relationship between expected loss and risk multiple.
        The fitted parameters are returned. 

        Parameters
        ----------
        peril : str, optional
            Peril type to filter the bonds. If None, no filtering by peril is applied.
        year : list[int] or int, optional
            Issuing year(s) to filter the bonds. If None, no filtering by year is applied.

        Sets
        -------
        ibrd_prem_rate : float
            The estimated premium rate based on the fitted curve and the bond's expected annual loss.
        """
        ibrd_bonds = pd.read_excel(DATA_DIR.joinpath('IBRD_bonds.xlsx'))
        if peril is not None: 
            flt_ibrd_bonds = ibrd_bonds[ibrd_bonds['Peril'] == peril]
            flt_ibrd_bonds = flt_ibrd_bonds.reset_index(drop=True)

        elif year is not None:
            flt_ibrd_bonds = ibrd_bonds[ibrd_bonds['Issuing date'].isin(year)]
            flt_ibrd_bonds = flt_ibrd_bonds.reset_index(drop=True)

        else:
            flt_ibrd_bonds = ibrd_bonds.copy()
        #perform the fit
        params_prem_ibrd, cv = curve_fit(self.monoExp, flt_ibrd_bonds['Expected Loss'], flt_ibrd_bonds['Risk Multiple'])

        a, k, b = params_prem_ibrd
        LOGGER.info(f'Fitted IBRD premium parameters: a={a}, k={k}, b={b}')
        self.ibrd_prem_rate = self.monoExp(self.bond_simulation_class.loss_metrics['EL_ann']*100, a, k, b) * self.bond_simulation_class.loss_metrics['EL_ann']
        LOGGER.info(f"Calculated IBRD premium rate: {self.ibrd_prem_rate}")
        


    ### BENCHMARK SHARPE RATIO PREMIUMS ###
    '''Benchmark pricing function -> goes through all losses and determines required premium to achieve a certain target Sharpe ratio'''
    def find_sharpe(self, premium: float, monthly_losses: pd.DataFrame, target_sharpe: float):
        """
        Calculates the squared difference between the Sharpe ratio of a cat bond cash flow and a target Sharpe ratio.
        The function simulates the annual cash flows of a catastrophe bond investment, adjusting for losses and premium payments.
        It computes the average return and standard deviation of the net cash flows, then calculates the Sharpe ratio and returns
        the squared difference from the target Sharpe ratio.

        Parameters
        ----------
        premium : float
            Annual premium rate paid to the investor.
        monthly_losses : pandas.DataFrame
            DataFrame with columns 'losses' (list of loss amounts per event)
            and 'months' (list of event months) for each year.
        target_sharpe : float
            Target Sharpe ratio to compare against.

        Returns
        -------
        float
            Squared difference between the calculated Sharpe ratio and the target Sharpe ratio.
        """

        ncf = []
        cur_nominal = 1
        for i in range(len(monthly_losses)):
            losses = monthly_losses['losses'].iloc[i]
            months = monthly_losses['months'].iloc[i]
            if np.sum(losses) == 0:
                ncf.append(cur_nominal * premium)
            else:
                ncf_pre_event = (cur_nominal * premium) / 12 * (months[0])
                ncf_post_event = []
                for j in range(len(losses)):
                    loss = losses[j]
                    month = months[j]
                    cur_nominal -= loss
                    if cur_nominal < 0:
                        loss += cur_nominal
                        cur_nominal = 0
                    if j + 1 < len(losses):
                        nex_month = months[j+1]
                        ncf_post_event.append(((cur_nominal * premium) / 12 * (nex_month - month)) - loss)
                    else:
                        ncf_post_event.append(((cur_nominal * premium) / 12 * (12- month)) - loss)
                ncf.append(ncf_pre_event + np.sum(ncf_post_event))
            if (i + 1) % self.bond_simulation_class.term == 0:
                cur_nominal = 1

        avg_ret = np.mean(ncf)
        sigma = np.std(ncf)
        return (avg_ret / sigma - target_sharpe)**2


    '''Benchmark pricing function -> wrapper function to call the optimization'''
    def calc_benchmark_premium(self, target_sharpe: float):        
        """
        Calculates the initial premium required to achieve a target Sharpe ratio for a given set of annual losses.
        This function uses numerical optimization to find the premium value that results in the desired Sharpe ratio.

        Parameters
        ----------
        target_sharpe : float
            Desired Sharpe ratio to be achieved.

        Returns
        -------
        float
            Optimal premium value that achieves the target Sharpe ratio.
        """

        result = minimize(lambda p: self.find_sharpe(p, self.bond_simulation_class.df_loss_month, target_sharpe), 
                          x0=[0.05])
        self.benchmark_prem_rate = result.x[0]
