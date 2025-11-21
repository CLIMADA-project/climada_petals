'''Risk pooling optimization functions adapted from Ciullo et al., 2022'''
import numpy as np
from math import comb
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer
import pandas as pd
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.repair.rounding import RoundingRepair

def process_n(n, cntry_names, sng_ann_losses, principal_sng_dic, n_opt_rep=100):
    """
    Runs risk concentration minimization for a given number of pools using a genetic algorithm,
    processes the optimization results, and generates convergence plots.
    Parameters
    ----------
    n : int
        Number of pools to optimize.
    cntry_names : list of str
        List of country names corresponding to the columns in the loss data.
    sng_ann_losses : pandas.DataFrame
        Dataframe of annual loss data for each country.
    principal_sng_dic : dict
        Principal values for each country.
    n_opt_rep : int, optional
        Number of optimization repetitions for seed analysis (default is 100).
    Returns
    -------
    country_allocation : pandas.DataFrame
        DataFrame containing a optimal country allocation for the minimum concentration solution.
    algorithm_result: pymoo.core.result.Result
        Result object from the optimization containing details of the optimization process.
    """
    opt_rep = range(0,n_opt_rep,1)
    df_losses = pd.DataFrame(sng_ann_losses)

    ### TRANSFROM PRINCIPAL VALUES TO LIST ###
    principal_sng = principal_sng_dic.set_index('Key').loc[cntry_names, 'Value'].tolist()

    ### CALCULATE ALPHA FOR RISK CONCENTRATION OPTIMIZATION ###
    RT = len(df_losses[cntry_names[0]])
    alpha = 1-1/RT 
    
    bools = df_losses >= np.quantile(df_losses, alpha, axis=0)

    risk_concentration = 1.0
    # Loop through repetitions for seed analysis
    for index in opt_rep:
        # Define Problem and Algorithm (same as inside the loop)
        problem = PoolOptimizationFixedNumber(principal_sng, df_losses, bools, alpha, n, calc_pool_conc)
        algorithm = GA(
            pop_size=2000,
            sampling=IntegerRandomSampling(),
            crossover=HalfUniformCrossover(),
            mutation=PolynomialMutation(repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        # Solve the problem
        res_reg = minimize(problem, algorithm, verbose=False, save_history=True)

        # Process results (same code as inside the loop)
        x = res_reg.X
        risk_concentration_new = res_reg.F
        if risk_concentration_new is not None and risk_concentration is not None and risk_concentration_new < risk_concentration:
            algorithm_result = res_reg
            risk_concentration = risk_concentration_new
            sorted_unique = sorted(set(x))
            rank_dict = {value: rank + 1 for rank, value in enumerate(sorted_unique)}
            x = [rank_dict[value] for value in x]

            # Add to dump dataframe
            country_allocation = pd.DataFrame(columns=[cntry_names, 'min_conc'])
            country_allocation = pd.DataFrame([x], columns=cntry_names)
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

'''Pool optimization problem using fixed number of pools'''
class PoolOptimizationFixedNumber(ElementwiseProblem):
    def __init__(self, nominals, data, bools, alpha, N, fun, **kwargs):
        self.data_arr = data
        self.bools = bools
        self.alpha = alpha
        self.N = N
        self.fun = fun
        self.nominals = np.array(nominals)
        self.n_countries = len(nominals)
        super().__init__(
            n_var=self.data_arr.shape[1],
            n_obj=1,  
            n_constr = 1,
            xl=0,                  
            xu=self.N - 1,
            type_var=int,
            vars=[Integer((0, self.n_countries - 1)) for _ in range(self.n_countries)],
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pools = {i: [] for i in np.unique(x)}
        for i, pool_id in enumerate(x):
            if len(np.where(x == i)[0]) > 0:
                pool_mask = np.where(x == i)[0]
                pools[i].append(pool_mask)
        
        total_concentration = 0
        for pool_key, pool_countries in pools.items():
            pool1_col = self.data_arr.columns[pool_countries[0]]
            pool1_data = self.data_arr[pool1_col].values
            pool1_bools = self.bools[pool1_col].values
            conc = self.fun(np.arange(0, len(pool_countries[0])), pool1_data, pool1_bools, self.alpha)
            total_concentration += conc
        constraints = 0
        if len(pools) != self.N:
            constraints += 1

        out["F"] = total_concentration/len(pools)
        out["G"] = constraints

def pop_num(n, k):
    combinations = comb(n + k - 1, k)
    return combinations

'''Pool optimization problem using maximum nominal constraint'''
class PoolOptimizationMaximumPrincipal(ElementwiseProblem):
    def __init__(self, nominals, max_nominal, data, bools, alpha, N, fun, **kwargs):
        self.data_arr = data
        self.bools = bools
        self.alpha = alpha
        self.N = N
        self.fun = fun
        self.nominals = np.array(nominals)
        self.n_countries = len(nominals)
        self.max_nominal = max_nominal
        super().__init__(
            n_var=self.data_arr.shape[1],
            n_obj=1,  
            n_constr = 1,
            xl=0,                  
            xu=self.N - 1,
            type_var=int,
            vars=[Integer((0, self.n_countries - 1)) for _ in range(self.n_countries)],
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pools = {i: [] for i in np.unique(x)}
        for i, pool_id in enumerate(x):
            if len(np.where(x == i)[0]) > 0:
                pool_mask = np.where(x == i)[0]
                pools[i].append(pool_mask)
        
        total_concentration = 0
        for pool_key, pool_countries in pools.items():
            pool1_col = self.data_arr.columns[pool_countries[0]]
            pool1_data = self.data_arr[pool1_col].values
            pool1_bools = self.bools[pool1_col].values
            conc = self.fun(np.arange(0, len(pool_countries[0])), pool1_data, pool1_bools, self.alpha) * np.sum(self.nominals[pool_countries[0]])
            total_concentration += conc
        constraints = 0
        for members in pools.values():
            pool_nominal_diff = np.sum(self.nominals[members[0]]) - self.max_nominal
            if pool_nominal_diff > 0:
                constraints += pool_nominal_diff

        out["F"] = total_concentration / np.sum(self.nominals)
        out["G"] = constraints
