'''Risk pooling optimization functions adapted from Ciullo et al., 2022'''
import numpy as np
from math import comb
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer

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
