"""Tests for pooling function utilities."""

import unittest
from unittest import mock
import numpy as np
import pandas as pd

from climada_petals.engine.cat_bonds import pooling_functions


def _dummy_conc(x, data_arr, bools, alpha):
    """Return the sum of pooled data for deterministic tests."""
    return float(np.sum(data_arr))


class TestPoolingFunctions(unittest.TestCase):
    """Unit tests for fixed-number pooling logic."""

    def setUp(self):
        """Set up a small deterministic dataset."""
        self.data = pd.DataFrame(
            {
                "A": [1.0, 4.0],
                "B": [2.0, 5.0],
                "C": [3.0, 6.0],
            }
        )
        self.bools = pd.DataFrame(False, index=self.data.index, columns=self.data.columns)

    def test_pool_optimization_fixed_number_class(self):
        """Validate objective and constraint in ElementwiseProblem implementation."""
        x = np.array([0, 0, 1])
        out = {}
        problem = pooling_functions.PoolOptimizationFixedNumber(
            nominals=[1.0, 1.0, 1.0],
            data=self.data,
            bools=self.bools,
            alpha=0.95,
            N=2,
            fun=_dummy_conc,
        )

        problem._evaluate(x, out)

        expected_obj = (12.0 + 9.0) / 2.0
        assert np.isclose(out["F"], expected_obj)
        assert out["G"] == 0

    def test_fixed_number_prefers_uncorrelated_pooling(self):
        """Validate uncorrelated losses yield lower concentration when pooled."""
        np.random.seed(0)
        data = pd.DataFrame(
            {
                "A": [10.0, 10.0, 10.0, 0.0],
                "B": [0.0, 0.0, 0.0, 10.0],
                "C": [10.0, 10.0, 10.0, 10.0],
            }
        )
        alpha = 0.75
        bools = data >= np.quantile(data, alpha, axis=0)

        x_ab = np.array([0, 0, 1])  # A+B pooled, C separate
        x_ac = np.array([0, 1, 0])  # A+C pooled, B separate

        problem = pooling_functions.PoolOptimizationFixedNumber(
            nominals=[1.0, 1.0, 1.0],
            data=data,
            bools=bools,
            alpha=alpha,
            N=2,
            fun=pooling_functions.calc_pool_conc,
        )

        out_ab = {}
        out_ac = {}
        problem._evaluate(x_ab, out_ab)
        problem._evaluate(x_ac, out_ac)
        assert out_ab["F"] < out_ac["F"]

    def test_pool_optimization_maximum_principal_class(self):
        """Validate objective and constraints for maximum principal pooling."""
        data = pd.DataFrame({"A": [1.0], "B": [1.0], "C": [1.0]})
        bools = pd.DataFrame(False, index=data.index, columns=data.columns)
        x = np.array([0, 0, 1])
        out = {}
        problem = pooling_functions.PoolOptimizationMaximumPrincipal(
            nominals=[1.0, 2.0, 3.0],
            max_nominal=2.0,
            data=data,
            bools=bools,
            alpha=0.95,
            fun=_dummy_conc,
        )

        problem._evaluate(x, out)

        expected_total = (2.0 * (1.0 + 2.0)) + (1.0 * 3.0)
        expected_obj = expected_total / sum([1.0, 2.0, 3.0])
        assert np.isclose(out["F"], expected_obj)
        assert out["G"] == 2.0 # constraint: total principal - max_nominal -> should be zero -> constrained violated

    def test_maximum_principal_prefers_uncorrelated_pooling(self):
        """Validate uncorrelated pooling under max nominal constraint."""
        data = pd.DataFrame(
            {
                "A": [10.0, 0.0, 10.0, 0.0],
                "B": [0.0, 10.0, 0.0, 10.0],
                "C": [10.0, 10.0, 10.0, 10.0],
            }
        )
        alpha = 0.75
        bools = data >= np.quantile(data, alpha, axis=0)
        nominals = [1.0, 1.0, 1.0]
        max_nominal = 2.0

        x_ab = np.array([0, 0, 1])  # A+B pooled, C separate
        x_ac = np.array([0, 1, 0])  # A+C pooled, B separate

        problem = pooling_functions.PoolOptimizationMaximumPrincipal(
            nominals=nominals,
            max_nominal=max_nominal,
            data=data,
            bools=bools,
            alpha=alpha,
            fun=pooling_functions.calc_pool_conc,
        )

        out_ab = {}
        out_ac = {}
        problem._evaluate(x_ab, out_ab)
        problem._evaluate(x_ac, out_ac)

        assert out_ab["G"] == 0
        assert out_ac["G"] == 0
        assert out_ab["F"] < out_ac["F"]

    def test_process_n_pools_returns_allocation(self):
        """Validate process_n_pools returns ranked allocation and result."""
        np.random.seed(0)
        countries = ["A", "B", "C"]

        class DummySubarea:
            def __init__(self, principal):
                self.principal = principal

        class DummyBond:
            def __init__(self):
                self.df_loss_month = pd.DataFrame({"losses": [[1.0], [0.0], [1.0]]})
                self.subarea_calc = DummySubarea(principal=1.0)

        cls_bonds = [DummyBond(), DummyBond(), DummyBond()]

        class DummyRes:
            X = np.array([0, 0, 1])
            F = 0.5

        with mock.patch.object(pooling_functions, "minimize", return_value=DummyRes()), \
             mock.patch.object(pooling_functions, "GA", return_value=object()):
            alloc, res = pooling_functions.process_n_pools(
                number_pools=2,
                countries=countries,
                cls_bond_simulations=cls_bonds,
                n_opt_rep=1,
            )

        assert res is not None
        assert alloc.columns.tolist() == ["A", "B", "C", "min_conc"]
        assert alloc.iloc[0].tolist() == [1, 1, 2, 0.5]

    def test_process_maximum_principal_pools_returns_allocation(self):
        """Validate process_maximum_principal_pools returns ranked allocation."""
        np.random.seed(0)
        countries = ["A", "B", "C"]

        class DummySubarea:
            def __init__(self, principal):
                self.principal = principal

        class DummyBond:
            def __init__(self):
                self.df_loss_month = pd.DataFrame({"losses": [[0.0], [1.0], [0.0]]})
                self.subarea_calc = DummySubarea(principal=1.0)

        cls_bonds = [DummyBond(), DummyBond(), DummyBond()]

        class DummyRes:
            X = np.array([1, 0, 1])
            F = 0.4

        with mock.patch.object(pooling_functions, "minimize", return_value=DummyRes()), \
             mock.patch.object(pooling_functions, "GA", return_value=object()):
            alloc, res = pooling_functions.process_maximum_principal_pools(
                maximum_principal=2.0,
                countries=countries,
                cls_bond_simulations=cls_bonds,
                n_opt_rep=1,
            )

        assert res is not None
        assert alloc.columns.tolist() == ["A", "B", "C", "min_conc"]
        assert alloc.iloc[0].tolist() == [2, 1, 2, 0.4]

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestPoolingFunctions)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
