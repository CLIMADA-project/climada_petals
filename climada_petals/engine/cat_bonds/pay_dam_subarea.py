import pandas as pd
import matplotlib.pyplot as plt

import subarea_calculations

class Pay_dam_subarea:
    def __init__(self, subareas, index_stat, exhaustion_point, attachment_point):
        self.subareas_class = subareas
        self.index_stat = index_stat
        self.exhaustion_point = exhaustion_point
        self.attachment = attachment_point
        self._get_pay_vs_dam()

    def _get_pay_vs_dam(self):
        calculation_class = subarea_calculations.Subarea_Calculations(self.subareas_class, self.index_stat, self.exhaustion_point, self.attachment)
        self.pay_vs_dam, self.principal = calculation_class.create_pay_vs_dam()

    def plot_pay_vs_dam(self, calculation_class):
        tot_exp = calculation_class.exposure.gdf['value'].sum()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

        ax1.scatter(self.pay_vs_dam/tot_exp, payout_flt/tot_exp, marker='o', color='blue', label='Events')
        ax1.plot([0, nominal/tot_exp], [0, nominal/tot_exp], color='black', linestyle='--', label='Trendline')
        ax1.axhline(y = nominal/tot_exp, color = 'r', linestyle = '-', label='Principal') 
        ax1.axhline(y = 0.05, color = 'r', linestyle = '-', label='Attachment Point') 
        ax1.axvline(x = 0.05, color = 'r', linestyle = '--', label='Min. Damage') 
        ax1.set_xlabel("Damage [share of GDP]", fontsize=12)
        ax1.set_ylabel("Payout [share of GDP]", fontsize=12)
        ax1.legend(loc='lower right', borderpad=2.0)

        ax2.scatter(damages/tot_exp, pay_dam_df['pay']/tot_exp, marker='o', color='blue', label='Events')
        ax2.axhline(y = nominal/tot_exp, color = 'r', linestyle = '-', label='Principal') 
        ax2.axhline(y = 0.05, color = 'r', linestyle = '-', label='Attachment Point') 
        ax2.axvline(x = 0.05, color = 'black', linestyle = '--', label='Min. Damage') 
        ax2.set_xscale('log')
        ax2.set_xlabel("Damage [share of GDP]", fontsize=12)
        ax2.set_ylabel("Payout [share of GDP]", fontsize=12)

        panel_labels = ["a)", "b)"]
        for i, ax in enumerate([ax1, ax2]):
            ax.annotate(panel_labels[i], 
                xy=(-0.1, 1),  
                xycoords="axes fraction", 
                fontsize=14, 
                fontweight="bold")
            
        plt.tight_layout()
        plt.show()
