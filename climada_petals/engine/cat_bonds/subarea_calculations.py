import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

# import climada modules
from climada.engine import ImpactCalc

# set logging basics
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)


class Subarea_Calculations:
    def __init__(self, subareas, index_stat, exhaustion_point, attachment_point):

        '''
        Attributes
        ----------
        self.exposure : climada.Exposure
            Exposure object containing geospatial exposure data (with a GeoDataFrame attribute `gdf`).
        self.hazard : climada.Hazard
            Hazard object containing hazard events.
        self.vulnerability : climada.Vulnerability
            Vulnerability object containing vulnerability functions.
        self.subareas : geopandas.GeoDataFrame
            GeoDataFrame of CAT bond subareas for spatial aggregation of impacts.
        self.exhaustion_point: float
            The exhaustion_point value (maximum possible payout) of the CAT bond. If it is a string with 'Exp', it is treated as a
            share of total exposure. If it is a string with 'RP', it is treated as a return period. If it is a float,
            it is treated as a monetary value.
        self.attachment: float
            The attachment value (minimum possible payout) of the CAT bond. If it is a string with 'Exp', it is treated as a
            share of total exposure. If it is a string with 'RP', it is treated as a return period. If it is a float,
            it is treated as a monetary value.
        self.index_stat: str or float
            The statistic to calculate. Can either be a number to calculate percentile or the string 'mean' to calculate the average.
        '''

        self.subareas = subareas.subareas_gdf
        self.exposure = subareas.exposure
        self.hazard = subareas.hazard
        self.vulnerability = subareas.vulnerability
        self.index_stat = index_stat
        self.exhaustion_point = exhaustion_point

        self.initial_guess_dict = {
            "TC": (30, 40)
        }  # initial guess for wind speed in m/s

    def _calc_impact(self):
        """
        Initializes and calculates impact based on exposure, hazard, vulnerability and optional subareas.
        This function performs the following steps:
        1. Calculates impact using the provided exposure, vulnerability and hazard datasets.
        2. If subareas are provided, aggregates impact per subarea using spatial joins.

        Parameters
        ----------
        self: class instance
            Instance of the Subarea_Calculations class.

        Returns
        -------
        imp : climada.ImpactCalc
            Impact calculation object containing results and methods.
        imp_per_event : numpy.ndarray
            Array of impact values per event for each exposure point.
        imp_subareas_evt : pandas.DataFrame
            DataFrame of aggregated impacts per subarea and event.
        """

        # perform impact calcualtion
        imp = ImpactCalc(self.exposure, self.vulnerability, self.hazard).impact(
            save_mat=True
        )

        # save impact per exposure point
        imp_per_event = imp.at_event

        # save impact per exposure point
        imp_per_exp = imp.imp_mat
        exp_crs = self.exposure.gdf
        exp_crs = exp_crs.to_crs(self.subareas.crs)

        # Perform a spatial join to associate each exposure point with calculated impact with a subarea
        exp_to_admin = exp_crs.sjoin(self.subareas, how="left", predicate="within")

        # group each exposure point according to subarea letter
        agg_exp = exp_to_admin.groupby("subarea_letter").apply(
            lambda x: x.index.tolist()
        )

        # Dictionary to store the impacts for each subarea
        imp_subarea_csr = {}
        # Loop through each subarea and its corresponding line numbers
        for letter, line_numbers in agg_exp.items():
            selected_values = imp_per_exp[
                :, line_numbers
            ]  # Select all impact values per subarea
            imp_subarea_csr[letter] = (
                selected_values  # Store them in dictionary per subarea
            )
        imp_subareas_evt = {}  # total damage for each event per subarea

        # sum all impacts per subarea
        for i in imp_subarea_csr:
            imp_subareas_evt[i] = imp_subarea_csr[i].sum(
                axis=1
            )  # calculate sum of impacts per subarea
            imp_subareas_evt[i] = [
                matrix.item() for matrix in imp_subareas_evt[i]
            ]  # single values per event are stored in 1:1 matrix -> only save value

        # transform matrix to data frame
        imp_subareas_evt = pd.DataFrame.from_dict(imp_subareas_evt)

        return imp, imp_per_event, imp_subareas_evt

    def _calc_attachment_principal(self, impact):
        """
        Initializes and calculates the attachment point and principal value for a CAT bond.
        The function determines the attachment point/principal amount based on either a protection return period
        using the provided climada ImpactCalc object, or as a share of the total exposure value using the toal exposure.
        If the principal is already a monetary values, it is used directly.

        Parameters
        ----------
            self: class instance
                Instance of the Subarea_Calculations class.
            impact : climada.ImpactCalc
                Impact calculation object containing results and methods.

        Returns
        ----------
            exhaustion_point: float
                The calculated exhaustion_point value for the CAT bond.
        """
        tot_exp = self.exposure.gdf["value"].sum()

        if isinstance(self.attachment, float):
            attachment = self.attachment

        elif isinstance(self.attachment, str):
            if "Exp" in self.attachment:
                self.attachment = float(self.attachment.split(" ")[0])
                attachment = tot_exp * self.attachment
            elif "RP" in self.attachment:
                self.attachment = float(self.attachment.split(" ")[0])
                attachment = impact.calc_freq_curve(self.attachment).impact
            else:
                raise ValueError(
                    "Invalid attachment format. Use 'Exp' for exposure share or 'RP' for return period."
                )

        else:
            raise ValueError(
                "Attachment must be a float or a string containing 'Exp' or 'RP'."
            )

        if isinstance(self.exhaustion_point, float):
            principal = self.exhaustion_point

        elif isinstance(self.exhaustion_point, str):
            if "Exp" in self.exhaustion_point:
                self.exhaustion_point = float(self.exhaustion_point.split(" ")[0])
                principal = tot_exp * self.exhaustion_point
            elif "RP" in self.exhaustion_point:
                self.exhaustion_point = float(self.exhaustion_point.split(" ")[0])
                principal = impact.calc_freq_curve(self.exhaustion_point).impact
            else:
                raise ValueError(
                    "Invalid exhaustion point format. Use 'Exp' for exposure share or 'RP' for return period."
                )

        else:
            raise ValueError(
                "Exhaustion point must be a float or a string containing 'Exp' or 'RP'."
            )

        logger.info(
            f"The attachment point and the principal of the CAT bond is: {round(attachment, 3)} and {round(principal, 3)} [USD], respectively."
        )
        logger.info(
            f"Attachment point and principal as share of exposure: {round(attachment/tot_exp, 3)} and {round(principal/tot_exp, 3)}, respectively."
        )

        return principal, attachment

    def _calc_index(self):
        """
        Calculates a specified statistic (mean, percentiles) for each events parametrix incex for each subarea.

        Parameters
        ----------
            self: class instance
                Instance of the Subarea_Calculations class.

        Returns
        -------
            int_sub_dict: dict
                A dictionary containing a pandas.DataFrame with the calculated statistics per subarea with labels as columns and year and month for each event.
                The key to the dataframe is the hazard type (e.g., 'TC' for tropical cyclones).
        """

        hazard = self.hazard.centroids.gdf
        hazard = hazard.to_crs(self.subareas.crs)
        centrs_to_sub = hazard.sjoin(self.subareas, how="left", predicate="intersects")
        agg_exp = centrs_to_sub.groupby("subarea_letter").apply(lambda x: x.index.tolist())
        
        int_sub = {
            letter: [np.nan] * len(self.hazard.event_id) for letter in agg_exp.keys()
        }

        int_sub["year"] = [0 for _ in range(len(self.hazard.event_id))]
        int_sub["month"] = [0 for _ in range(len(self.hazard.event_id))]

        # Iterate over each event
        for i in range(len(self.hazard.event_id)):
            date = pd.to_datetime(self.hazard.get_event_date()[i])
            int_sub["year"][i] = date.year
            int_sub["month"][i] = date.month
            # For each subarea, calculate the desired statistic
            for letter, line_numbers in agg_exp.items():
                selected_values = self.hazard.intensity[i, line_numbers]
                if self.index_stat == "mean":
                    int_sub[letter][i] = selected_values.mean()
                elif isinstance(self.index_stat, (int, float)):
                    dense_array = selected_values.toarray()
                    flattened_array = dense_array.flatten()
                    int_sub[letter][i] = np.percentile(flattened_array, self.index_stat)
                else:
                    raise ValueError(
                        "Invalid statistic choice. Choose number for percentile or 'mean'"
                    )
        int_sub = pd.DataFrame.from_dict(int_sub)

        int_sub_dict = {}
        int_sub_dict[self.hazard.haz_type] = int_sub

        return int_sub_dict

    def _objective_fct(self, params, haz_int, damages, principal):
        """
        Defines the objective function used to minimize basis risk by adjusting minimum and maximum trigger thresholds in the payout function.
        This function computes the squared difference between actual damages and payouts,
        given a set of parameters and hazard intensity data. It determines the maximum payout
        based on the principal value and observed damages, then calculates payouts using
        a payout initialization function.

        Parameters
        ----------
            self: class instance
                Instance of the Subarea_Calculations class.
            params: tuple
                A tuple containing the minimum and maximum trigger values.
            haz_int: dict
                Hazard intensity data.
            damages: array-like
                Observed damages for each subarea and hazard event.
            principal: float
                The principal value (maximum possible payout).

        Returns
        -------
            basis_risk: float
                The calculated basis risk as the sum of squared differences between damages and payouts.
        """

        min_trig, max_trig = params
        max_dam = np.max(damages)
        if max_dam < principal:
            max_pay = max_dam
        else:
            max_pay = principal
        payouts = calc_payout(min_trig, max_trig, haz_int, max_pay)
        arr_damages = np.array(damages)
        basis_risk = np.sum((arr_damages - payouts) ** 2)
        return basis_risk

    # funtion to minimze basis risk by adjusting minimum and maximum parametric index thresholds used in the payout funciton
    def _calibrate_payout_fcts(self, haz_int, principal, attachment, imp_subarea_evt):
        """
        Initializes and performs the optimization of the payout function which is based on a paramteric index for each subarea.
        This function iterates over subareas, selects appropriate initial guesses and damage data depending on the parametric index,
        and applies the COBYLA optimization algorithm to minimize the objective function (basis risk) for each subarea.
        The results for each subarea are collected and returned, along with arrays of optimized parameters.

        Parameters
        ----------
        self: class instance
            Instance of the Subarea_Calculations class.
        attachment : float
            The attachment point (minimum payout) for payouts.
        haz_int : dict
            DataFrame containing paramteric index values for each subarea and additional columns.
        principal : float
            Principal of the CAT bond used in the optimization objective function.
        imp_subarea_evt : pandas.DataFrame
            Damages per event and subarea.

        Returns
        -------
        results : dict
            Dictionary mapping subarea indices to optimization result objects.
        opt_min_thresh : numpy.ndarray
            Array of minimum paramteric index threshold for each subarea
        opt_max_thresh : numpy.ndarray
            Array of maximum parametric index threshold for each subarea.
        """
        imp_subarea_evt_flt = imp_subarea_evt.copy()
        imp_subarea_evt_flt.loc[imp_subarea_evt_flt.sum(axis=1) < attachment, :] = 0

        hazard_type = list(haz_int.keys())[0]

        subareas = range(len(haz_int[hazard_type].columns) - 2)
        subarea_specific_results = {}

        results = {}
        for subarea in subareas:

            damages = imp_subarea_evt_flt.iloc[:, subarea]

            # Perform optimization for each subarea
            result = minimize(
                self._objective_fct,
                self.initial_guess_dict[hazard_type],
                args=(haz_int[hazard_type].iloc[:, [subarea, -1]], damages, principal),
                method="COBYLA",
                options={"maxiter": 100000},
            )

            results[subarea] = result

            if result.success:
                opt_min, opt_max = result.x
                subarea_specific_results[subarea] = (opt_min, opt_max)
            else:
                print(f"Optimization failed for subarea {subarea}: {result.message}")

        opt_min_thresh = np.array(
            [values[0] for values in subarea_specific_results.values()]
        )
        opt_max_thresh = np.array(
            [values[1] for values in subarea_specific_results.values()]
        )

        return results, opt_min_thresh, opt_max_thresh

    def _calc_pay_vs_dam(
        self,
        imp_per_event,
        imp_subareas_evt,
        attachment,
        principal,
        opt_min_thresh,
        opt_max_thresh,
        haz_int,
    ):
        """
        Calculates payouts versus damages for hazard events.
        This function computes the payout for each event based on optimized threshold parameters and parametric index data.
        It compares the payouts to the corresponding damages, applying constraints such as minimum payout and principal cap.

        Parameters
        ----------
        imp_per_event : numpy.ndarray
            Array of impact values per event for each exposure point.
        imp_subareas_evt : pandas.DataFrame
            Damages per subarea and event used for payout calculations.
        attachment : float
            The attachment point (minimum payout) for payouts.
        principal : float
            The principal value of the CAT bond.
        opt_min_thresh : array-like
            Thresholds for mimimum payouts for each payout function.
        opt_max_thresh : array-like
            Thresholds for maximum payouts for each payout function.
        haz_int : dict
            Dictionary containing parametric index data for each event, including year and month columns.

        Returns
        -------
        pay_dam_df : pandas.DataFrame
            DataFrame containing calculated payouts, damages, year, and month for each event.

        Notes
        -----
        - Payouts are capped at the nominal value and set to zero if below the minimum payout threshold.
        - The function relies on an external `_calc_payout` function for payout calculation.
        """

        imp_per_event_df = pd.DataFrame({"Damage": imp_per_event})
        imp_per_event_arr = np.array(imp_per_event_df)
        imp_per_event_arr[imp_per_event_arr < attachment] = 0

        hazard_type = list(haz_int.keys())[0]

        b = len(imp_per_event_arr)
        max_damage = imp_per_event_arr.max()
        if max_damage < 1:
            minimum_payout = 0
        else:
            minimum_payout = imp_per_event_arr[imp_per_event_arr > 0].min()

        payout_evt_grd = pd.DataFrame(
            {letter: [None] * b for letter in haz_int[hazard_type].columns[:-2]}
        )
        pay_dam_df = pd.DataFrame(
            {"pay": [0.0] * b, "damage": [0.0] * b, "year": [0] * b, "month": [0] * b}
        )

        for i in range(len(imp_per_event_arr)):
            tot_dam = imp_per_event_arr[i]
            pay_dam_df.loc[i, "damage"] = tot_dam
            pay_dam_df.loc[i, "year"] = int(haz_int[hazard_type]["year"][i])
            pay_dam_df.loc[i, "month"] = int(haz_int[hazard_type]["month"][i])
            for j in range(len(haz_int[hazard_type].columns) - 3):
                sub_hazint = haz_int[hazard_type].iloc[:, [j, -1]]
                max_dam = np.max(imp_subareas_evt.iloc[:, j])
                if max_dam < principal:
                    max_pay = max_dam
                else:
                    max_pay = principal
                payouts = calc_payout(
                    opt_min_thresh[j], opt_max_thresh[j], sub_hazint, max_pay
                )
                payout_evt_grd.iloc[:, j] = payouts
            tot_pay = np.sum(payout_evt_grd.iloc[i, :])
            if tot_pay > principal:
                tot_pay = principal
            elif tot_pay < minimum_payout:
                tot_pay = 0
            else:
                pass
            pay_dam_df.loc[i, "pay"] = tot_pay

        return pay_dam_df

    def create_pay_vs_dam(self):

        imp, imp_per_event, imp_subareas_evt = self._calc_impact() 
        par_idx = self._calc_index()
        self.principal, self.attachment = self._calc_attachment_principal(imp, )
        self.results, self.opt_min_thresh, self.opt_max_thresh = self._calibrate_payout_fcts(par_idx, self.principal, self.attachment, imp_subareas_evt)
        pay_vs_dam = self._calc_pay_vs_dam(imp_per_event=imp_per_event, imp_subareas_evt=imp_subareas_evt, attachment=self.attachment, principal=self.principal, opt_min_thresh=self.opt_min_thresh, opt_max_thresh=self.opt_max_thresh, haz_int=par_idx)
        
        return pay_vs_dam, self.principal
    
# this function calculates the payout for an event in a subarea -> defines the payout function
def calc_payout(min_trig, max_trig, haz_int, max_pay):

    """
    Calculates payout values based on a linear payout function using hazard intensities and trigger thresholds.
    
    Parameters
    ----------
    min_trig : float
        The minimum trigger threshold for hazard intensity.
    max_trig : float
        The maximum trigger threshold for hazard intensity.
    haz_int : pandas.DataFrame
        DataFrame containing hazard intensity values in the first column.
    max_pay : float
        The maximum payout value.
    
    Returns
    -------
    payouts : numpy.ndarray
        Array of calculated payout values corresponding to each hazard intensity.
    """

    intensities = np.array(haz_int.iloc[:, 0])
    payouts = np.zeros_like(intensities)
    payouts[intensities >= max_trig] = max_pay
    mask = (intensities >= min_trig) & (intensities < max_trig)
    payouts[mask] = (intensities[mask] - min_trig) / (max_trig - min_trig) * max_pay

    return payouts
