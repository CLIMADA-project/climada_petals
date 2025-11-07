import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

import logging

# path to data folder
DATA_DIR = (Path(__file__).parent.parent).joinpath('data/cat_bonds')

# setup logger 
LOGGER = logging.getLogger(__name__)

# regression coefficients for chatoro premium calculation (extracted from Chatoro et al., 2022)
b_0 = -0.5907
b_1 = 1.3986
b_2 = 2.2520
b_3 = 0.0377
b_4 = 0.4613
b_5 = -0.0239
b_6 = -2.6742
b_7 = 0.7057

class premium_calculations:

    def __init__(self, bond_simulation_class):
        self.bond_simulation_class = bond_simulation_class

    ### CHATORO-PRICING ###
    def chatoro_premium(self, peak_multi, investment_graded, hybrid_trigger, GCIndex=None, BBSpread=None):
        '''Linear regression formula to calculate the premium based on the regression model presented in Chatoro et al., 2022'''

        if GCIndex is None:
            GCIndex = 180 #Guy Carpenter Global Property Catastrophe Rate on Line Index (January, 2025)
        else:
            pass
        if BBSpread is None:
            BBSpread = 1.6 #ICE BofA BB US High Yield Index Option-Adjusted Spread (January, 2025)
        else:
            pass

        self.chatoro_prem_rate = b_0 + b_1 * self.bond_simulation_class.loss_metrics['EL_ann'] + b_2 * peak_multi + b_3 * GCIndex + b_4 * BBSpread + b_5 * self.bond_simulation_class.term * 12 + b_6 * investment_graded + b_7 * hybrid_trigger


    ### IBRD-PRICING ###
    def monoExp(self, x, a, k, b):
        '''Exponential function to fit the risk multiple curve'''
        return a * np.exp(-k * x) + b

    def init_prem_ibrd(self, peril=None, year=None):
        """
        Fits a monotonic exponential curve to catastrophe bond data for bonds issued by the World Bank to estimate premium parameters.
        This function loads IBRD bond data from an Excel file, optionally filters the data by peril type or issuing year,
        and fits a monotonic exponential function to the relationship between expected loss and risk multiple.
        The fitted parameters are returned. Optionally, the function can generate and save a plot of the fit.
        Parameters
        ----------
        peril : str, optional
            Peril type to filter the bonds (e.g., 'Earthquake', 'Flood'). If None, no filtering by peril is applied.
        year : list or int, optional
            Issuing year(s) to filter the bonds. If None, no filtering by year is applied.
        Returns
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
