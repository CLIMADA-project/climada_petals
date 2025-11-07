import pandas as pd
import numpy as np
import logging

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

