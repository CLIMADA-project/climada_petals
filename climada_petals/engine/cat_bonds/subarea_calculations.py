import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

# import climada modules
from climada.engine import ImpactCalc

# set logging basics
LOGGER = logging.getLogger(__name__)



class SubareaCalculations:
    def __init__(self, subareas, index_stat):
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

        self.subareas = subareas
        self.index_stat = index_stat

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
        imp_subareas_evt : pandas.DataFrame
            DataFrame of aggregated impacts per subarea and event.
        """

        # perform impact calcualtion
        imp = ImpactCalc(self.subareas.exposure, self.subareas.vulnerability, self.subareas.hazard).impact(
            save_mat=True
        )

        # get exp gdf
        exp_gdf = self.subareas.exposure.gdf

        # Perform a spatial join to associate each exposure point with calculated impact with a subarea
        exp_to_admin = exp_gdf.sjoin(self.subareas.subareas_gdf, how="left", predicate="within")
        if exp_to_admin['subarea_letter'].isnull().any():
            LOGGER.warning("Some exposure points were not assigned to any subarea. Subareas may be to small.")
        # group each exposure point according to subarea letter
        agg_exp = exp_to_admin.subarea_letter.to_list()
        imp_subareas_evt = imp.impact_at_reg(agg_exp)

        return imp, imp_subareas_evt

    def _calc_attachment_principal(self, impact, attachment_point, exhaustion_point, attachment_point_method=None, exhaustion_point_method=None):
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
            attachment_point : float
                The attachment point value for the CAT bond. Can be expressed as a monetary value, a share of total exposure, or a return period.
            exhaustion_point : float
                The exhaustion point value for the CAT bond. Can be expressed as a monetary value, a share of total exposure, or a return period.
            attachment_point_method : str, optional
                Method to interpret the attachment point. Options are 'Exposure_Share' or 'Return_Period'. If None, the attachment_point is treated as a monetary value.
            exhaustion_point_method : str, optional
                Method to interpret the exhaustion point. Options are 'Exposure_Share' or 'Return_Period'. If None, the exhaustion_point is treated as a monetary value.

        Returns
        ----------
            attachment: float
                The calculated attachment point value for the CAT bond.
            principal: float
                The calculated principal value for the CAT bond.
        """

        tot_exp = self.subareas.exposure.gdf["value"].sum()

        if attachment_point_method is None:
            attachment = attachment_point
        elif attachment_point_method == "Exposure_Share":
            attachment = tot_exp * attachment_point
        elif attachment_point_method == "Return_Period":
            attachment = impact.calc_freq_curve(attachment_point).impact
        else:
            raise ValueError(
                "Invalid attachment point method. Choose 'Exposure_Share' or 'Return_Period'."
            )
        if exhaustion_point_method is None:
            principal = exhaustion_point
        elif exhaustion_point_method == "Exposure_Share":
            principal = tot_exp * exhaustion_point
        elif exhaustion_point_method == "Return_Period":
            principal = impact.calc_freq_curve(exhaustion_point).impact
        else:
            raise ValueError(
                "Invalid exhaustion point method. Choose 'Exposure_Share' or 'Return_Period'."
            )

        LOGGER.info(
            f"The attachment point and the principal of the CAT bond is: {round(attachment, 3)} and {round(principal, 3)} [USD], respectively."
        )

        return principal, attachment

    def _calc_parametric_index(self):
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

        hazard = self.subareas.hazard.centroids.gdf
        hazard = hazard.to_crs(self.subareas.subareas_gdf.crs)
        centrs_to_sub = hazard.sjoin(self.subareas.subareas_gdf, how="left", predicate="intersects")
        agg_exp = centrs_to_sub.groupby("subarea_letter").apply(lambda x: x.index.tolist())
        
        int_sub = {
            letter: [np.nan] * len(self.subareas.hazard.event_id) for letter in agg_exp.keys()
        }

        int_sub["year"] = [0 for _ in range(len(self.subareas.hazard.event_id))]
        int_sub["month"] = [0 for _ in range(len(self.subareas.hazard.event_id))]

        # Iterate over each event
        for i in range(len(self.subareas.hazard.event_id)):
            date = pd.to_datetime(self.subareas.hazard.get_event_date()[i])
            int_sub["year"][i] = date.year
            int_sub["month"][i] = date.month
            # For each subarea, calculate the desired statistic
            for letter, line_numbers in agg_exp.items():
                selected_values = self.subareas.hazard.intensity[i, line_numbers]
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
        int_sub_dict[self.subareas.hazard.haz_type] = int_sub

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
        impact,
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
        impact : climada.ImpactCalc
            Impact calculation object containing results and methods.
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

        imp_per_event = impact.at_event
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

    def create_pay_vs_dam(self, attachment_point, exhaustion_point, methods_attachment_point=None, methods_exhaustion_point=None):

        imp, imp_subareas_evt = self._calc_impact() 
        parametric_index = self._calc_parametric_index()
        if methods_attachment_point is not None and methods_exhaustion_point is not None:
            self.principal, self.attachment = self._calc_attachment_principal(imp, attachment_point, exhaustion_point, methods_attachment_point, methods_exhaustion_point)
        else:
            self.principal, self.attachment = exhaustion_point, attachment_point
        self.results, self.opt_min_thresh, self.opt_max_thresh = self._calibrate_payout_fcts(parametric_index, self.principal, self.attachment, imp_subareas_evt)
        self.pay_vs_dam = self._calc_pay_vs_dam(impact=imp, imp_subareas_evt=imp_subareas_evt, attachment=self.attachment, principal=self.principal, opt_min_thresh=self.opt_min_thresh, opt_max_thresh=self.opt_max_thresh, haz_int=parametric_index)

    
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
