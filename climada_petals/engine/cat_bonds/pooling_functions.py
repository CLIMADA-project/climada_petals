import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer
import pandas as pd
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.repair.rounding import RoundingRepair


def process_n_pools(number_pools, countries, cls_bond_simulations, n_opt_rep=100):
    """
    Runs risk concentration minimization for a given number of pools using a genetic algorithm,
    processes the optimization results, and generates convergence plots.

    Parameters
    ----------
    n : int
        Number of pools to optimize.
    countries : list
        List of country names corresponding to the bond simulations.
    cls_bond_simulations : list
        List of SingleCountryBondSimulation instances for each country, containing the principals and 
        monthly losses.
    n_opt_rep : int, optional
        Number of optimization repetitions for seed analysis (default is 100).

    Returns
    -------
    country_allocation : pandas.DataFrame
        DataFrame containing a optimal country allocation for the minimum concentration solution.
    algorithm_result: pymoo.core.result.Result
        Result object from the optimization containing details of the optimization process.
    """

    annual_losses_dic_cty = {}
    principal_sng = []
    for idx, cty in enumerate(countries):
        annual_losses_dic_cty[cty] = cls_bond_simulations[idx].df_loss_month['losses'].apply(lambda x: sum(x) if len(x) > 0 else 0)
        principal_sng.append(cls_bond_simulations[idx].subarea_calc.principal)
    df_losses = pd.DataFrame(annual_losses_dic_cty)

    # Short-circuit: with only 1 pool every country is assigned to it,
    # so the GA optimisation (which would have xl == xu == 0) is skipped.
    if number_pools == 1:
        country_allocation = pd.DataFrame([[1] * len(countries)], columns=countries)
        country_allocation['min_conc'] = np.nan
        return country_allocation, None

    opt_rep = range(0,n_opt_rep,1)

    ### CALCULATE ALPHA FOR RISK CONCENTRATION OPTIMIZATION ###
    RT = len(df_losses[countries[0]])
    alpha = 1-1/RT 
    
    bools = df_losses >= np.quantile(df_losses, alpha, axis=0)

    risk_concentration = np.inf # ensures first valid result always wins
    algorithm_result = None
    country_allocation = None
    # Loop through repetitions for seed analysis
    for index in opt_rep:
        # Define Problem and Algorithm (same as inside the loop)
        problem = PoolOptimizationFixedNumber(principal_sng, df_losses, bools, alpha, number_pools, calc_pool_conc)
        algorithm = GA(
            pop_size=2000,
            sampling=IntegerRandomSampling(),
            crossover=HalfUniformCrossover(),
            mutation=PolynomialMutation(repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        # Solve the problem
        res_reg = minimize(problem.fixed_number_pools_problem, algorithm, verbose=False, save_history=True)

        # Process results (same code as inside the loop)
        x = res_reg.X
        risk_concentration_new = res_reg.F
        if risk_concentration_new is not None and risk_concentration is not None and risk_concentration_new <= risk_concentration:
            algorithm_result = res_reg
            risk_concentration = risk_concentration_new
            sorted_unique = sorted(set(x))
            rank_dict = {value: rank + 1 for rank, value in enumerate(sorted_unique)}
            x = [rank_dict[value] for value in x]

            # Add to dump dataframe
            country_allocation = pd.DataFrame(columns=[countries, 'min_conc'])
            country_allocation = pd.DataFrame([x], columns=countries)
            country_allocation['min_conc'] = pd.DataFrame([res_reg.F], columns=['min_conc'])

    return country_allocation, algorithm_result


def process_maximum_principal_pools(maximum_principal, countries, cls_bond_simulations, n_opt_rep=100):
    """
    Runs risk concentration minimization for pools with a maximum principal constraint using a genetic algorithm,
    processes the optimization results, and generates convergence plots.

    Parameters
    ----------
    maximum_principal : float
        Maximum principal allowed per pool.
    countries : list
        List of country names corresponding to the bond simulations.
    cls_bond_simulations : list
        List of SingleCountryBondSimulation instances for each country, containing the principals and 
        monthly losses.
    n_opt_rep : int, optional
        Number of optimization repetitions for seed analysis (default is 100).
    
    Returns
    -------
    country_allocation : pandas.DataFrame
        DataFrame containing a optimal country allocation for the minimum concentration solution.
    algorithm_result: pymoo.core.result.Result
        Result object from the optimization containing details of the optimization process.
    """
    
    annual_losses_dic_cty = {}
    principal_sng = []
    for idx, cty in enumerate(countries):
        annual_losses_dic_cty[cty] = cls_bond_simulations[idx].df_loss_month['losses'].apply(lambda x: sum(x) if len(x) > 0 else 0)
        principal_sng.append(cls_bond_simulations[idx].subarea_calc.principal)
    df_losses = pd.DataFrame(annual_losses_dic_cty)

    opt_rep = range(0,n_opt_rep,1)

    ### CALCULATE ALPHA FOR RISK CONCENTRATION OPTIMIZATION ###
    RT = len(df_losses[countries[0]])
    alpha = 1-1/RT 
    
    bools = df_losses >= np.quantile(df_losses, alpha, axis=0)

    risk_concentration = np.inf # ensures first valid result always wins
    algorithm_result = None
    country_allocation = None
    # Loop through repetitions for seed analysis
    for index in opt_rep:
        # Define Problem and Algorithm (same as inside the loop)
        
        problem = PoolOptimizationMaximumPrincipal(principal_sng, maximum_principal, df_losses, bools, alpha, calc_pool_conc)
        algorithm = GA(
            pop_size=2000,
            sampling=IntegerRandomSampling(),
            crossover=HalfUniformCrossover(),
            mutation=PolynomialMutation(repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        # Solve the problem
        res_reg = minimize(problem.max_principal_problem, algorithm, verbose=False, save_history=True)

        # Process results (same code as inside the loop)
        x = res_reg.X
        risk_concentration_new = res_reg.F
        if risk_concentration_new is not None and risk_concentration is not None and risk_concentration_new <= risk_concentration:

            algorithm_result = res_reg
            risk_concentration = risk_concentration_new
            sorted_unique = sorted(set(x))
            rank_dict = {value: rank + 1 for rank, value in enumerate(sorted_unique)}
            x = [rank_dict[value] for value in x]

            # Add to dump dataframe
            country_allocation = pd.DataFrame(columns=[countries, 'min_conc'])
            country_allocation = pd.DataFrame([x], columns=countries)
            country_allocation['min_conc'] = pd.DataFrame([res_reg.F], columns=['min_conc'])

    return country_allocation, algorithm_result


def calc_pool_conc(x, data_arr, bools, alpha):
    """Calculate diversification of a given pool. Used to 
    find the best pool.

    x : bool
        Countries to consider in the pool
    data_arr : np.array
        Numpy array with annual damages for all countries
    bools : np.array
        Numpy array with the same shape as data, indicating when 
        annual damages are higher/lower than the country VaR
    alpha : float
        Point at which to calculate VaR and ES
    """

    dam = data_arr[:,x]
    cntry_bools = bools[:,x]
    tot_damage = dam.sum(1)
    
    VAR_tot = np.quantile(tot_damage[~np.isnan(tot_damage)], alpha)
    bool_tot = tot_damage >= VAR_tot

    ES_cntry = []
    MES = []

    for cntry_pos in range(dam.shape[1]):
        dummy_dam = dam[:,cntry_pos][cntry_bools[:,cntry_pos]]

        ES_cntry.append(np.nanmean(dummy_dam))
        MES.append(np.nanmean(dam[:,cntry_pos][bool_tot]))

    ES_cntry = np.array(ES_cntry)
    MES = np.array(MES)

    # if no countries are picked
    if x.sum() == 0:
        POOL_CONC = 1.
    else:
        ES_tot = np.nansum(MES)
        POOL_CONC = ES_tot / np.nansum(ES_cntry)

    return np.round(POOL_CONC, 2)


class PoolOptimizationFixedNumber():
    """Genetic algorithm optimization problem for assigning countries to a fixed number of pools.

    Minimizes average risk concentration across all pools subject to the constraint that
    exactly N pools are formed.

    Parameters
    ----------
    nominals : list of float
        Principal amounts for each country.
    data : pandas.DataFrame
        Annual losses per country (rows = simulations, columns = countries).
    bools : pandas.DataFrame
        Boolean mask of the same shape as data; True where losses exceed the country VaR.
    alpha : float
        Quantile level used to compute VaR and ES.
    N : int
        Exact number of pools to form.
    fun : callable
        Risk concentration function with signature fun(indices, data, bools, alpha) -> float.
    """

    def __init__(self, nominals, data, bools, alpha, N, fun, **kwargs):
        self.data_arr = data
        self.bools = bools
        self.alpha = alpha
        self.N = N
        self.fun = fun
        self.nominals = np.array(nominals)
        self.n_countries = len(nominals)
        self.fixed_number_pools_problem = self._init_optimisation_problem(len(nominals), N, **kwargs)
        self.fixed_number_pools_problem._evaluate = self._evaluate

    @classmethod
    def _init_optimisation_problem(cls, n_countries, n_pools, **kwargs):
        """Build the pymoo ElementwiseProblem with integer variables bounded to valid pool IDs.

        Parameters
        ----------
        n_countries : int
            Number of decision variables (one pool ID per country).
        n_pools : int
            Number of pools; pool IDs range from 0 to n_pools - 1.

        Returns
        -------
        ElementwiseProblem
        """
        return ElementwiseProblem(
            n_var=n_countries,
            n_obj=1,
            n_constr=1,
            vars={f"x{i}": Integer(lb=0, ub=n_pools - 1) for i in range(n_countries)},
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objective and constraint for a candidate solution.

        Parameters
        ----------
        x : np.ndarray of int
            Pool ID assignment for each country; shape (n_countries,).
        out : dict
            pymoo output dict; sets out["F"] (objective) and out["G"] (constraint).
            F is the mean risk concentration across pools.
            G is 1 if the number of distinct pools != N, else 0.
        """
        pool_ids = np.unique(x)
        pools = {pid: np.where(x == pid)[0] for pid in pool_ids}

        total_concentration = 0
        for pool_key, pool_countries in pools.items():
            pool1_col = self.data_arr.columns[pool_countries]
            pool1_data = self.data_arr[pool1_col].values
            pool1_bools = self.bools[pool1_col].values
            conc = self.fun(np.arange(0, len(pool_countries)), pool1_data, pool1_bools, self.alpha)
            total_concentration += conc
        constraints = 0
        if len(pools) != self.N:
            constraints += 1

        out["F"] = total_concentration/len(pools)
        out["G"] = constraints


class PoolOptimizationMaximumPrincipal():
    """Genetic algorithm optimization problem for assigning countries to pools under a principal cap.

    Minimizes the principal-weighted average risk concentration across all pools subject to the
    constraint that no pool's total principal exceeds max_nominal. The number of pools is not
    fixed in advance; the GA discovers it subject to the principal constraint.

    Parameters
    ----------
    nominals : list of float
        Principal amounts for each country.
    max_nominal : float
        Maximum total principal allowed in any single pool.
    data : pandas.DataFrame
        Annual losses per country (rows = simulations, columns = countries).
    bools : pandas.DataFrame
        Boolean mask of the same shape as data; True where losses exceed the country VaR.
    alpha : float
        Quantile level used to compute VaR and ES.
    fun : callable
        Risk concentration function with signature fun(indices, data, bools, alpha) -> float.
    """

    def __init__(self, nominals, max_nominal, data, bools, alpha, fun, **kwargs):
        self.data_arr = data
        self.bools = bools
        self.alpha = alpha
        self.fun = fun
        self.nominals = np.array(nominals)
        self.n_countries = len(nominals)
        self.max_nominal = max_nominal
        self.max_principal_problem = self._init_optimisation_problem(len(nominals), **kwargs)
        self.max_principal_problem._evaluate = self._evaluate

    @classmethod
    def _init_optimisation_problem(cls, n_countries, **kwargs):
        """Build the pymoo ElementwiseProblem with integer variables bounded to valid pool IDs.

        Parameters
        ----------
        n_countries : int
            Number of decision variables (one pool ID per country).
            Pool IDs range from 0 to n_countries - 1 since in the worst case every country
            gets its own pool.

        Returns
        -------
        ElementwiseProblem
        """
        return ElementwiseProblem(
            n_var=n_countries,
            n_obj=1,
            n_constr=1,
            vars={f"x{i}": Integer(lb=0, ub=n_countries - 1) for i in range(n_countries)},
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objective and constraint for a candidate solution.

        Parameters
        ----------
        x : np.ndarray of int
            Pool ID assignment for each country; shape (n_countries,).
        out : dict
            pymoo output dict; sets out["F"] (objective) and out["G"] (constraint).
            F is the principal-weighted mean risk concentration, normalised by total principal.
            G is the total excess principal summed over all pools that breach max_nominal
            (0 means fully feasible).
        """
        pool_ids = np.unique(x)
        pools = {pid: np.where(x == pid)[0] for pid in pool_ids}

        total_concentration = 0
        for pool_key, pool_countries in pools.items():
            pool1_col = self.data_arr.columns[pool_countries]
            pool1_data = self.data_arr[pool1_col].values
            pool1_bools = self.bools[pool1_col].values
            conc = self.fun(np.arange(0, len(pool_countries)), pool1_data, pool1_bools, self.alpha) * np.sum(self.nominals[pool_countries])
            total_concentration += conc
        constraints = 0
        for members in pools.values():
            pool_nominal_diff = np.sum(self.nominals[members]) - self.max_nominal
            if pool_nominal_diff > 0:
                constraints += pool_nominal_diff

        out["F"] = total_concentration / np.sum(self.nominals)
        out["G"] = constraints
