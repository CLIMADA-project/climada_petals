import pandas as pd
import numpy as np
from datetime import date as _date
from scipy.optimize import minimize

# import climada modules
from climada.engine import ImpactCalc

from climada_petals.util.config import LOGGER


class SubareaCalculations:
    """
    Subarea-level calculations for parametric catastrophe bond payout setup.

    This class computes impacts, parametric indices, payout thresholds, and
    pay-versus-damage tables for a set of subareas.

    Parameters
    ----------
    subareas : object
        Container with `exposure`, `hazard`, `vulnerability`, and `subareas_gdf`.
    index_stat : str or float
        Statistic for the parametric index (e.g., "mean" or a percentile).
    intitial_guess : tuple[float, float], optional
        Initial guess for minimum and maximum payout trigger thresholds.

    Attributes
    ----------
    subareas : object
        Input container holding exposure, hazard, vulnerability, and subareas.
    index_stat : str or float
        Statistic used to summarize hazard intensity per subarea which is used as parametric index.
    initial_guess : tuple[float, float]
        Initial guess for minimum/maximum thresholds for the payout function.
    """
    def __init__(self, subareas, index_stat: float | str, initial_guess: tuple[float, float] | None =None):
        """
        Initialize the subarea calculation helper.

        Parameters
        ----------
        subareas : object
            Container with `exposure`, `hazard`, `vulnerability`, and `subareas_gdf`.
        index_stat : str or float
            Statistic used for the parametric index ("mean" or a percentile).
        initial_guess : tuple[float, float], optional
            Initial guess for min/max payout trigger thresholds. When omitted, the
            30th and 60th percentiles of hazard intensity are used.
        """

        self.subareas = subareas
        self.index_stat = index_stat
        if initial_guess is not None:
            self.initial_guess = initial_guess
        else:
            self.initial_guess = (np.percentile(subareas.hazard.intensity.data, 30), np.percentile(subareas.hazard.intensity.data, 60))

    def _calc_impact(self):
        """
        Calculate impact based on exposure, hazard, vulnerability, and subareas.
        This function performs the following steps:
        1. Calculates impact using the provided exposure, vulnerability and hazard datasets.
        2. If subareas are provided, aggregates impact per subarea using spatial joins.

        Parameters
        ----------
        None

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

        # Perform a spatial join to associate each exposure point with calculated impact with a subarea
        exp_to_admin = self.subareas.exposure.gdf.sjoin(self.subareas.subareas_gdf, how="left", predicate="within")
        if exp_to_admin['subarea_letter'].isnull().any():
            LOGGER.warning("Some exposure points were not assigned to any subarea. Subareas may be to small.")
        # group each exposure point according to subarea letter
        agg_exp = exp_to_admin.subarea_letter.to_list()
        imp_subareas_evt = imp.impact_at_reg(agg_exp)

        return imp, imp_subareas_evt

    def _calc_attachment_principal(self, impact, attachment_point: float, exhaustion_point: float, attachment_point_method: str | None = None, exhaustion_point_method: str | None = None):
        """
        Initializes and calculates the attachment point and principal value for a CAT bond.
        The function determines the attachment point/principal amount based on either a protection return period
        using the provided climada ImpactCalc object, or as a share of total exposure.
        If the values are already monetary amounts, they are used directly.

        Parameters
        ----------
        impact : climada.ImpactCalc
            Impact calculation object containing results and methods.
        attachment_point : float
            Attachment point value for the CAT bond. Is either a monetary amount, a fraction of total exposure, or a return period in years.
        exhaustion_point : float
            Exhaustion point value for the CAT bond. Is either a monetary amount, a fraction of total exposure, or a return period in years.
        attachment_point_method : str, optional
            Interpretation method: "Exposure_Share", "Return_Period", or None.
        exhaustion_point_method : str, optional
            Interpretation method: "Exposure_Share", "Return_Period", or None.

        Returns
        ----------
        attachment : float
            Calculated attachment point value for the CAT bond.
        principal : float
            Calculated principal value for the CAT bond.
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
        Calculate a specified statistic as the parametric index (mean or percentile) for each event's index in each subarea.

        Parameters
        ----------
        None

        Returns
        -------
        int_sub_dict : dict
            Dictionary mapping hazard type to a DataFrame with per-subarea
            statistics and event year/month columns.
        """
        hazard = self.subareas.hazard.centroids.gdf
        hazard = hazard.to_crs(self.subareas.subareas_gdf.crs)
        centrs_to_sub = hazard.sjoin(self.subareas.subareas_gdf, how="left", predicate="intersects")
        agg_exp = centrs_to_sub.groupby("subarea_letter").apply(lambda x: x.index.tolist())
        
        num_events = len(self.subareas.hazard.event_id)
        
        # Validate index_stat choice early
        if not (self.index_stat == "mean" or isinstance(self.index_stat, (int, float))):
            raise ValueError(
                "Invalid statistic choice. Choose number for percentile or 'mean'"
            )
        
        # Extract all dates once using vectorized conversion
        dates = [_date.fromordinal(d) for d in self.subareas.hazard.date]
        years = np.fromiter((d.year for d in dates), dtype=int)
        months = np.fromiter((d.month for d in dates), dtype=int)
        
        # Pre-allocate result dictionary with NumPy arrays
        int_sub = {letter: np.full(num_events, np.nan, dtype=float) for letter in agg_exp.keys()}
        
        # Iterate over subareas, fully vectorized per subarea
        for letter, line_numbers in agg_exp.items():
            line_numbers = np.array(line_numbers)
            
            if self.index_stat == "mean":
                # True vectorized mean: extract full subarea block and compute mean across centroids
                sub_matrix = self.subareas.hazard.intensity[:, line_numbers]
                int_sub[letter] = np.asarray(sub_matrix.mean(axis=1)).ravel()
            else:
                # True batch percentile: extract and densify full subarea block, then compute percentile across centroids
                sub_matrix = self.subareas.hazard.intensity[:, line_numbers].toarray()
                int_sub[letter] = np.percentile(sub_matrix, self.index_stat, axis=1)
        
        # Build result - convert arrays directly to lists for DataFrame construction
        int_sub_dict_result = {letter: int_sub[letter].tolist() for letter in agg_exp.keys()}
        int_sub_dict_result["year"] = years.tolist()
        int_sub_dict_result["month"] = months.tolist()
        int_sub_df = pd.DataFrame.from_dict(int_sub_dict_result)

        int_sub_dict = {}
        int_sub_dict[self.subareas.hazard.haz_type] = int_sub_df

        return int_sub_dict

    @staticmethod
    def _fallback_thresholds(parametric_index_df):
        """Compute fallback trigger thresholds from the parametric index.

        Used when payout function optimization fails for a subarea. Derives
        thresholds from all subarea columns of the parametric index DataFrame
        (excluding the ``year`` and ``month`` columns) so the fallback works
        even when a single subarea has no local damage data.

        Parameters
        ----------
        parametric_index_df : pandas.DataFrame
            Parametric index values per event.  Subarea columns followed by
            ``year`` and ``month`` columns (last two).

        Returns
        -------
        min_trig : float
            Median of non-zero parametric index values across all subareas
            (payout onset).
        max_trig : float
            Maximum parametric index value across all subareas (full payout).
        """
        index_values = parametric_index_df.iloc[:, :-2].values.ravel()
        max_trig = float(np.nanmax(index_values))
        nonzero_vals = index_values[index_values > 0]
        if len(nonzero_vals) > 0:
            min_trig = float(np.median(nonzero_vals))
        else:
            min_trig = 0.0
        if min_trig >= max_trig:
            min_trig = 0.5 * max_trig
        return min_trig, max_trig

    def _objective_fct(self, params: tuple[float, float], haz_int: pd.DataFrame, damages: pd.Series, principal: float):
        """
        Defines the objective function used to minimize basis risk by adjusting minimum and maximum trigger thresholds in the payout function.
        This function computes the squared difference between actual damages and payouts,
        given a set of parameters and hazard intensity data. It determines the maximum payout
        based on the principal value and observed damages, then calculates payouts using
        a payout initialization function.

        Parameters
        ----------
        params : tuple[float, float]
            Minimum and maximum trigger values.
        haz_int : pandas.DataFrame
            Hazard intensity data for a subarea.
        damages : array-like
            Observed damages for each subarea and hazard event.
        principal : float
            Principal value (maximum possible payout).

        Returns
        -------
        basis_risk : float
            Sum of squared differences between damages and payouts.
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
        Initialize and optimize payout functions based on a parametric index per subarea.
        This function iterates over subareas, selects appropriate initial guesses and damage data depending on the parametric index,
        and applies the COBYLA optimization algorithm to minimize the objective function (basis risk) for each subarea.
        The results for each subarea are collected and returned, along with arrays of optimized parameters.

        Parameters
        ----------
        haz_int : dict
            Parametric index values for each subarea.
        principal : float
            Principal of the CAT bond used in the optimization objective function.
        attachment : float
            Attachment point (minimum payout) for payouts.
        imp_subarea_evt : pandas.DataFrame
            Damages per event and subarea.

        Returns
        -------
        results : dict
            Dictionary mapping subarea indices to optimization result objects.
        opt_min_thresh : numpy.ndarray
            Array of minimum parametric index thresholds for each subarea.
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
                self.initial_guess,
                args=(haz_int[hazard_type].iloc[:, [subarea, -1]], damages, principal),
                method="COBYLA",
                options={"maxiter": 100000},
            )

            results[subarea] = result

            if result.success:
                opt_min, opt_max = result.x
                subarea_specific_results[subarea] = (opt_min, opt_max)
            else:
                LOGGER.warning(
                    "Optimization failed for subarea %s: %s. Using fallback thresholds which sets the minimum and maximum parametric index thresholds to the median and maximum parametric index values using all available hazard data. Basis risk may be higher for this subarea. The available hazard data may be insufficient or inadequate to optimize the payout function. Consider adjusting the initial guess or reviewing the hazard intensity data for this subarea.",
                    subarea, result.message,
                )
                subarea_specific_results[subarea] = self._fallback_thresholds(
                    haz_int[hazard_type]
                )

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
        imp_subareas_evt: pd.DataFrame,
        attachment: float,
        principal: float,
        opt_min_thresh: np.ndarray,
        opt_max_thresh: np.ndarray,
        haz_int: dict,
    ):
        """
        Calculates payouts versus damages for hazard events.
        This function computes the payout for each event based on optimized payout threshold parameters and parametric index data.
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
            Thresholds for minimum payouts for each payout function.
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
        - The function relies on the module-level `calc_payout` function for payout calculation.
        """

        imp_per_event = impact.at_event
        imp_per_event_arr = np.array(imp_per_event).flatten()
        imp_per_event_arr[imp_per_event_arr < attachment] = 0

        hazard_type = list(haz_int.keys())[0]
        haz_int_data = haz_int[hazard_type]
        num_events = len(imp_per_event_arr)
        num_subareas = len(haz_int_data.columns) - 2

        # Extract year and month once
        years = haz_int_data["year"].values.astype(int)
        months = haz_int_data["month"].values.astype(int)

        # Calculate minimum payout threshold
        max_damage = imp_per_event_arr.max()
        if max_damage < 1:
            minimum_payout = 0
        else:
            minimum_payout = imp_per_event_arr[imp_per_event_arr > 0].min()

        # Pre-compute max_pay per subarea 
        max_pay_per_subarea = np.zeros(num_subareas)
        for j in range(num_subareas):
            max_dam_subarea = np.max(imp_subareas_evt.iloc[:, j])
            max_pay_per_subarea[j] = min(max_dam_subarea, principal)

        # Iterate over subareas
        payouts_grid = np.zeros((num_events, num_subareas))
        for j in range(num_subareas):
            intensities = haz_int_data.iloc[:, j].values
            min_trig = opt_min_thresh[j]
            max_trig = opt_max_thresh[j]
            max_pay = max_pay_per_subarea[j]

            # Vectorized payout calculation for all events at once
            payouts = np.zeros_like(intensities, dtype=float)
            payouts[intensities >= max_trig] = max_pay
            mask = (intensities >= min_trig) & (intensities < max_trig)
            payouts[mask] = (intensities[mask] - min_trig) / (max_trig - min_trig) * max_pay

            payouts_grid[:, j] = payouts

        # Sum payouts across subareas and apply caps
        tot_pays = payouts_grid.sum(axis=1)
        tot_pays = np.minimum(tot_pays, principal)
        tot_pays[tot_pays < minimum_payout] = 0

        # Build result DataFrame
        pay_dam_df = pd.DataFrame({
            "pay": tot_pays,
            "damage": imp_per_event_arr,
            "year": years,
            "month": months
        })

        return pay_dam_df

    def create_pay_vs_dam(self, attachment_point: float, exhaustion_point: float, methods_attachment_point: str | None = None, methods_exhaustion_point: str | None = None):
        """
        Build the pay-versus-damage table for the bond structure.

        This runs impact calculation, parametric index creation, payout
        calibration, and computes pay/damage per event.

        Parameters
        ----------
        attachment_point : float
            Attachment point value or parameter for the selected method. Is either a monetary amount, a fraction of total exposure, or a return period in years.
        exhaustion_point : float
            Exhaustion point value or parameter for the selected method. Is either a monetary amount, a fraction of total exposure, or a return period in years.
        methods_attachment_point : str, optional
            Interpretation method for attachment (e.g., "Exposure_Share" or "Return_Period").
        methods_exhaustion_point : str, optional
            Interpretation method for exhaustion (e.g., "Exposure_Share" or "Return_Period").

        Returns
        -------
        None
            Sets `principal`, `attachment`, `results`, `opt_min_thresh`,
            `opt_max_thresh`, and `pay_vs_dam` on the instance.
        """
        LOGGER.info("Calculating impact and subarea-level impacts per event.")
        imp, imp_subareas_evt = self._calc_impact() 
        LOGGER.info("Calculating parametric index per subarea.")
        parametric_index = self._calc_parametric_index()
        if methods_attachment_point is not None and methods_exhaustion_point is not None:
            self.principal, self.attachment = self._calc_attachment_principal(imp, attachment_point, exhaustion_point, methods_attachment_point, methods_exhaustion_point)
        else:
            self.principal, self.attachment = exhaustion_point, attachment_point
        LOGGER.info("Calibrating payout functions by optimizing trigger thresholds to minimize basis risk.")
        self.results, self.opt_min_thresh, self.opt_max_thresh = self._calibrate_payout_fcts(parametric_index, self.principal, self.attachment, imp_subareas_evt)
        LOGGER.info("Calculating pay-versus-damage table for all events using optimized payout functions.")
        self.pay_vs_dam = self._calc_pay_vs_dam(impact=imp, imp_subareas_evt=imp_subareas_evt, attachment=self.attachment, principal=self.principal, opt_min_thresh=self.opt_min_thresh, opt_max_thresh=self.opt_max_thresh, haz_int=parametric_index)

    
# this function calculates the payout for an event in a subarea -> defines the payout function
def calc_payout(min_trig: float, max_trig: float, haz_int: pd.DataFrame, max_pay: float):

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
