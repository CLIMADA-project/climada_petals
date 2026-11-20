"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Tests for the network cascade uncertainty calculation (CalcCascade)

The toy model is a chain network

    people(0) - road(1) - road(2) - road(3) - healthcare(4)

where people depend on the hospital via the road. Two parameters are sampled:

- ``x_pop`` scales the population of the people node (``POP * x_pop``),
- ``x_haz`` scales a uniform hazard intensity. The hospital fails when its
  impact exceeds 0.5, i.e. when ``x_haz > HAZ_THRESH``.

Hence, for each sample, the population losing healthcare access is
``POP * x_pop`` if ``x_haz > HAZ_THRESH`` and 0 otherwise.
"""

import copy as cp
import functools

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import scipy as sp
from matplotlib import pyplot as plt
from scipy import sparse
from shapely.geometry import LineString, Point

from climada.engine.unsequa import InputVar, UncOutput
from climada.entity import ImpactFunc, ImpactFuncSet
from climada.hazard import Centroids, Hazard

from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_calcs import NetworkCalcs
from climada_petals.engine.networks.nw_impacts import exposure_from_nodes
from climada_petals.engine.networks.nw_utils import POP_COL
from climada_petals.engine.unsequa import CalcCascade, UncCascadeOutput
from climada_petals.engine.unsequa import calc_cascade

POP = 100.0
HAZ_THRESH = 1.5
HAZ_THRESH_B = 1.25
CI_TYPES = ["healthcare"]

DEP_TABLE = pd.DataFrame(
    [
        {
            "source": "healthcare",
            "target": "people",
            "type_I": "enduser",
            "type_II": "logical",
            "via_link": "road",
            "thresh_func": 1,
            "link_condition": "distance",
            "thresh_dist": 1e7,
            "thresh_dur": np.inf,
            "bidir_link": False,
            "access_cnstr": True,
            "n_links": 1,
        }
    ]
)


# ========================================================================
# Toy model and input variables
# ========================================================================


@functools.lru_cache(maxsize=1)
def _network_with_dependencies():
    """Chain network with base state and dependencies, before the base-state
    cascade (cached: built once per process, copied before any change)."""
    points = [Point(i, i) for i in range(5)]
    nodes = gpd.GeoDataFrame(
        {
            "ci_type": ["people", "road", "road", "road", "healthcare"],
            POP_COL: [POP, 0, 0, 0, 0],
            "geometry": points,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    edges = gpd.GeoDataFrame(
        {
            "from_id": [0, 1, 2, 3],
            "to_id": [1, 2, 3, 4],
            "ci_type": "road",
            "distance": [157200.0] * 4,
            "geometry": [LineString([points[i], points[i + 1]]) for i in range(4)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    nw_calc = NetworkCalcs(Network(edges=edges, nodes=nodes), dep_table=DEP_TABLE)
    nw_calc.initialize_base_state()
    nw_calc.setup_dependencies()
    return nw_calc.network


def _nw_func(x_pop=1.0):
    """Network input: (network, dep_table, exp_list), with x_pop times the
    population and the base-state cascade run with this population."""
    network = cp.deepcopy(_network_with_dependencies())
    is_people = network.nodes["ci_type"] == "people"
    network.nodes.loc[is_people, POP_COL] = POP * x_pop
    nw_calc = NetworkCalcs(network, dep_table=DEP_TABLE)
    nw_calc.cascade()
    network = nw_calc.network
    exp_list = [exposure_from_nodes(network, "healthcare", value=1)]
    return network, DEP_TABLE, exp_list


def _impf_tuple():
    """Impact function input: (impf_set, failure_thresh).

    The impact of the hospital grows linearly from 0 at intensity 1 to 1 at
    intensity 2; the hospital fails when the impact exceeds 0.5.
    """
    impf = ImpactFunc(
        haz_type="TC",
        id="healthcare",
        intensity=np.array([0.0, 1.0, 2.0, 100.0]),
        mdd=np.array([0.0, 0.0, 1.0, 1.0]),
        paa=np.ones(4),
        intensity_unit="m/s",
        name="healthcare",
    )
    return ImpactFuncSet([impf]), {"healthcare": 0.5}


def _nw_func_two_exposures(x_pop=1.0):
    """Network input with two exposures of the hospital (tags ``healthcare``
    and ``healthcare_b``), e.g. two damage mechanisms."""
    network, dep_table, exp_list = _nw_func(x_pop)
    exp_list.append(
        exposure_from_nodes(network, "healthcare", value=1, tag="healthcare_b")
    )
    return network, dep_table, exp_list


def _impf_tuple_two_exposures():
    """Impact functions of the two hospital exposures: the same curve, but the
    exposure ``healthcare_b`` fails when the impact exceeds 0.25, i.e. when
    the intensity exceeds ``HAZ_THRESH_B``."""
    impf_set, failure_thresh = _impf_tuple()
    impf_b = ImpactFunc(
        haz_type="TC",
        id="healthcare_b",
        intensity=np.array([0.0, 1.0, 2.0, 100.0]),
        mdd=np.array([0.0, 0.0, 1.0, 1.0]),
        paa=np.ones(4),
        intensity_unit="m/s",
        name="healthcare_b",
    )
    impf_set.append(impf_b)
    return impf_set, {**failure_thresh, "healthcare_b": 0.25}


def _haz_func(x_haz=1.0):
    """Single event hazard with intensity x_haz at all nodes."""
    coords = np.arange(5.0)
    return Hazard(
        haz_type="TC",
        centroids=Centroids(lat=coords, lon=coords),
        event_id=np.array([1]),
        event_name=["ev1"],
        date=np.array([1]),
        frequency=np.array([1.0]),
        intensity=sparse.csr_matrix(np.full((1, 5), x_haz)),
        fraction=sparse.csr_matrix(np.ones((1, 5))),
    )


def _expected_access_loss(samples_df):
    """Expected population losing healthcare access for each sample."""
    return np.where(samples_df["x_haz"] > HAZ_THRESH, POP * samples_df["x_pop"], 0.0)


@pytest.fixture
def calc():
    """CalcCascade with uncertain population and hazard intensity."""
    nw_iv = InputVar(_nw_func, {"x_pop": sp.stats.uniform(0.5, 0.5)})
    haz_iv = InputVar(_haz_func, {"x_haz": sp.stats.uniform(0.5, 4)})
    return CalcCascade(nw_iv, _impf_tuple(), haz_iv, ci_types=CI_TYPES)


@pytest.fixture
def unc_output(calc):
    """Uncertainty output for 6 latin hypercube samples."""
    unc_sample = calc.make_sample(
        N=6, sampling_method="latin", sampling_kwargs={"seed": 42}
    )
    return calc.uncertainty(unc_sample)


# ========================================================================
# Initialisation and sampling
# ========================================================================


def test_init(calc):
    """Input variables, metric names and unit are set."""
    assert calc._input_var_names == ("nw_input_var", "impf_input_var", "haz_input_var")
    assert tuple(calc._metric_names) == ("imp_met",)
    assert calc.ci_types == CI_TYPES
    assert calc.impact_kwargs == {}
    assert calc.cascade_kwargs == {}
    assert set(calc.distr_dict) == {"x_pop", "x_haz"}

    network, dep_table, exp_list = calc.nw_input_var.evaluate(x_pop=0.5)
    assert network.nodes.loc[0, POP_COL] == 0.5 * POP
    assert dep_table is DEP_TABLE
    assert exp_list[0].description == "healthcare"
    impf_set, failure_thresh = calc.impf_input_var.evaluate()
    assert failure_thresh == {"healthcare": 0.5}
    assert calc.haz_input_var.evaluate(x_haz=2).intensity.max() == 2


def test_init_fixed_inputs():
    """Fixed tuples and a Hazard can be given instead of InputVars."""
    calc = CalcCascade(_nw_func(), _impf_tuple(), _haz_func(), ci_types=CI_TYPES)

    assert calc.distr_dict == {}
    network, _, _ = calc.nw_input_var.evaluate()
    impf_set, _ = calc.impf_input_var.evaluate()
    hazard = calc.haz_input_var.evaluate()
    assert isinstance(network, Network)
    assert isinstance(impf_set, ImpactFuncSet)
    assert isinstance(hazard, Hazard)


def test_init_ci_types_string_raises():
    """ci_types must be a list: a string would be read letter by letter."""
    with pytest.raises(TypeError, match=r"\['healthcare'\]"):
        CalcCascade(_nw_func(), _impf_tuple(), _haz_func(), ci_types="healthcare")


def test_make_sample(calc):
    """Samples are drawn for all parameters of all input variables."""
    unc_sample = calc.make_sample(N=2, sampling_kwargs={"calc_second_order": True})

    assert unc_sample.n_samples == 2 * (2 * 2 + 2)  # N * (2 * D + 2)
    assert list(unc_sample.samples_df.columns) == ["x_pop", "x_haz"]

    unc_sample = calc.make_sample(
        N=3, sampling_method="latin", sampling_kwargs={"seed": 1}
    )
    assert unc_sample.n_samples == 3


# ========================================================================
# Uncertainty
# ========================================================================


def test_uncertainty_without_sample_raises(calc):
    """uncertainty needs samples."""
    with pytest.raises(ValueError, match="No sample"):
        calc.uncertainty(UncOutput(pd.DataFrame()))


def test_uncertainty(calc, unc_output):
    """One row of network statistics per sample, with the expected values."""
    samples_df = unc_output.samples_df
    unc_df = unc_output.imp_met_unc_df

    assert isinstance(unc_output, UncCascadeOutput)
    assert unc_output.unit == "people"
    assert unc_output.uncertainty_metrics == ["imp_met"]
    assert len(unc_df) == len(samples_df) == 6
    assert {
        "total_population",
        "directly_affected",
        "indirectly_affected",
        "total_affected",
        "healthcare_access_loss",
        "healthcare_infra_impact",
    } <= set(unc_df.columns)
    # only the requested ci_types are evaluated (not e.g. road)
    assert not any(col.startswith("road_") for col in unc_df.columns)
    # one unit per statistic: hospitals are counted, the rest is population
    assert list(unc_output.imp_met_unit_df.columns) == list(unc_df.columns)
    units = unc_output.imp_met_units
    assert units["healthcare_infra_impact"] == "nodes"
    assert units["healthcare_access_loss"] == "people"
    assert {
        unit for col, unit in units.items() if "_infra_" not in col
    } == {"people"}

    np.testing.assert_allclose(unc_df["total_population"], POP * samples_df["x_pop"])
    np.testing.assert_allclose(
        unc_df["healthcare_access_loss"], _expected_access_loss(samples_df)
    )
    np.testing.assert_array_equal(
        unc_df["healthcare_infra_impact"],
        (samples_df["x_haz"] > HAZ_THRESH).astype(int),
    )
    # both outcomes are present in the sample
    assert 0 < (samples_df["x_haz"] > HAZ_THRESH).sum() < len(samples_df)


def test_uncertainty_fixed_network_not_modified():
    """A fixed (network, dep_table, exp_list) tuple is reused for all samples:
    the calculation must not change it."""
    network, dep_table, exp_list = _nw_func()
    nodes_before = network.nodes.copy()
    edges_before = network.edges.copy()
    haz_iv = InputVar(_haz_func, {"x_haz": sp.stats.uniform(0.5, 4)})
    calc = CalcCascade(
        (network, dep_table, exp_list), _impf_tuple(), haz_iv, ci_types=CI_TYPES
    )
    unc_sample = calc.make_sample(
        N=4, sampling_method="latin", sampling_kwargs={"seed": 3}
    )

    unc_output = calc.uncertainty(unc_sample)

    pd.testing.assert_frame_equal(network.nodes, nodes_before)
    pd.testing.assert_frame_equal(network.edges, edges_before)
    np.testing.assert_allclose(
        unc_output.imp_met_unc_df["healthcare_access_loss"],
        np.where(unc_output.samples_df["x_haz"] > HAZ_THRESH, POP, 0.0),
    )


def test_uncertainty_hazard_with_several_events_raises():
    """The network impact calculation needs a hazard with one event."""
    haz = _haz_func()
    haz_two = Hazard(
        haz_type="TC",
        centroids=haz.centroids,
        event_id=np.array([1, 2]),
        event_name=["ev1", "ev2"],
        date=np.array([1, 2]),
        frequency=np.array([1.0, 1.0]),
        intensity=sparse.csr_matrix(np.full((2, 5), 2.0)),
        fraction=sparse.csr_matrix(np.ones((2, 5))),
    )
    nw_iv = InputVar(_nw_func, {"x_pop": sp.stats.uniform(0.5, 0.5)})
    calc = CalcCascade(nw_iv, _impf_tuple(), haz_two, ci_types=CI_TYPES)
    unc_sample = calc.make_sample(
        N=2, sampling_method="latin", sampling_kwargs={"seed": 1}
    )

    with pytest.raises(ValueError, match="one event"):
        calc.uncertainty(unc_sample)


def test_uncertainty_stores_cascade_options(calc):
    """Cascade options are stored and used (no rerouting: same result, as the
    hospital is the only source)."""
    unc_sample = calc.make_sample(
        N=2, sampling_method="latin", sampling_kwargs={"seed": 1}
    )

    unc_output = calc.uncertainty(unc_sample, cascade_kwargs={"rerouting": False})

    assert calc.cascade_kwargs == {"rerouting": False}
    assert calc.friction_surf is None
    assert calc.impact_kwargs == {}
    np.testing.assert_allclose(
        unc_output.imp_met_unc_df["healthcare_access_loss"],
        _expected_access_loss(unc_output.samples_df),
    )


def test_uncertainty_impact_kwargs_combine_exposures():
    """impact_kwargs reach NetworkImpactCalc: two exposures on the hospital
    raise without combine_exposures; with it, the hospital fails if one of
    them makes it fail (here the second one, above HAZ_THRESH_B)."""
    nw_iv = InputVar(_nw_func_two_exposures, {"x_pop": sp.stats.uniform(0.5, 0.5)})
    haz_iv = InputVar(_haz_func, {"x_haz": sp.stats.uniform(0.5, 4)})
    calc = CalcCascade(nw_iv, _impf_tuple_two_exposures(), haz_iv, ci_types=CI_TYPES)
    # x_haz = 1.4: only the exposure healthcare_b fails
    unc_sample = UncOutput(
        pd.DataFrame({"x_pop": [1.0, 1.0, 1.0], "x_haz": [1.0, 1.4, 2.0]})
    )

    with pytest.raises(ValueError, match="combine_exposures"):
        calc.uncertainty(unc_sample)

    unc_output = calc.uncertainty(
        unc_sample, impact_kwargs={"combine_exposures": True}
    )

    assert calc.impact_kwargs == {"combine_exposures": True}
    np.testing.assert_allclose(
        unc_output.imp_met_unc_df["healthcare_access_loss"], [0.0, POP, POP]
    )


def test_uncertainty_cascade_kwargs_reach_cascade(monkeypatch):
    """cascade_kwargs are passed to NetworkCalcs.cascade, with friction_surf."""
    # fixed inputs, built before the spy: only the cascades run by CalcCascade
    # are recorded (not the base-state cascade of _nw_func)
    calc = CalcCascade(_nw_func(), _impf_tuple(), _haz_func(2.0), ci_types=CI_TYPES)
    calls = []
    original = NetworkCalcs.cascade

    @functools.wraps(original)  # keeps the signature checked by uncertainty
    def recording_cascade(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(NetworkCalcs, "cascade", recording_cascade)
    unc_sample = UncOutput(pd.DataFrame({"unused": [0.0]}))  # fixed inputs

    cascade_kwargs = {"rerouting": False, "source_var": "generation"}

    calc.uncertainty(unc_sample, cascade_kwargs=cascade_kwargs)

    assert calc.cascade_kwargs == cascade_kwargs
    # one call for the computation time estimate, one for the sample
    assert calls == 2 * [
        {
            "friction_surf": None,
            "rerouting": False,
            "source_var": "generation",
        }
    ]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"impact_kwargs": {"haz": None}}, "set by CalcCascade"),
        ({"impact_kwargs": {"edge_rul": "max_ratio"}}, "not arguments of NetworkImpactCalc"),
        ({"cascade_kwargs": {"friction_surf": None}}, "set by CalcCascade"),
        ({"cascade_kwargs": {"p_sourc": "plant"}}, r"not arguments of NetworkCalcs\.cascade"),
    ],
)
def test_uncertainty_kwargs_checked(calc, kwargs, match):
    """impact_kwargs and cascade_kwargs may only contain arguments of the
    called function that CalcCascade does not set itself."""
    unc_sample = UncOutput(pd.DataFrame({"x_pop": [1.0], "x_haz": [2.0]}))

    with pytest.raises(ValueError, match=match):
        calc.uncertainty(unc_sample, **kwargs)


def test_uncertainty_unknown_ci_type_raises():
    """ci_types that are not in the network raise instead of giving zero
    impacts."""
    calc = CalcCascade(
        _nw_func(), _impf_tuple(), _haz_func(), ci_types=["healthcare", "hospital"]
    )
    unc_sample = UncOutput(pd.DataFrame({"unused": [0.0]}))  # fixed inputs

    with pytest.raises(ValueError, match=r"\['hospital'\] are not in the network"):
        calc.uncertainty(unc_sample)


def test_uncertainty_line_ci_type():
    """A ci_type of the edges (road) is accepted."""
    calc = CalcCascade(
        _nw_func(), _impf_tuple(), _haz_func(), ci_types=["healthcare", "road"]
    )
    unc_sample = UncOutput(pd.DataFrame({"unused": [0.0]}))  # fixed inputs

    unc_output = calc.uncertainty(unc_sample)

    unc_df = unc_output.imp_met_unc_df
    assert unc_df.loc[0, "road_infra_impact"] == 0  # roads are not exposed
    assert unc_df.loc[0, "road_infra_total"] > 0
    units = unc_output.imp_met_units
    assert units["road_infra_total"] == "m"
    assert units["healthcare_infra_total"] == "nodes"


@pytest.mark.parametrize(
    "nw_input, impf_input, match",
    [
        (
            lambda: _nw_func()[0],
            _impf_tuple,
            r"nw_input_var must evaluate to a tuple \(network, dep_table, "
            r"exp_list\), got Network",
        ),
        (
            lambda: _nw_func()[:2],
            _impf_tuple,
            r"nw_input_var .* got tuple of length 2",
        ),
        (
            _nw_func,
            lambda: _impf_tuple()[0],
            r"impf_input_var must evaluate to a tuple \(impf_set, "
            r"failure_thresh\), got ImpactFuncSet",
        ),
    ],
)
def test_uncertainty_wrong_input_shape_raises(nw_input, impf_input, match):
    """Input variables that evaluate to the wrong shape raise a TypeError
    naming the expected tuple."""
    calc = CalcCascade(nw_input(), impf_input(), _haz_func(), ci_types=CI_TYPES)
    unc_sample = UncOutput(pd.DataFrame({"unused": [0.0]}))  # fixed inputs

    with pytest.raises(TypeError, match=match):
        calc.uncertainty(unc_sample)


def test_uncertainty_parallel(calc, unc_output):
    """Parallel computation gives the same results."""
    unc_sample = calc.make_sample(
        N=6, sampling_method="latin", sampling_kwargs={"seed": 42}
    )

    unc_output_pool = calc.uncertainty(unc_sample, processes=2)

    pd.testing.assert_frame_equal(
        unc_output_pool.imp_met_unc_df, unc_output.imp_met_unc_df
    )


def test_compute_metrics_is_evaluated_immediately(calc, monkeypatch):
    """_compute_metrics runs the computation when called (not lazily), so
    that the timing used for the computation time estimate is meaningful."""
    calls = []

    def fake_map(sample_chunks, *args):
        calls.append(len(sample_chunks))
        stats = pd.DataFrame({"total_population": [0.0] * len(sample_chunks)})
        return stats, {"total_population": "people"}

    monkeypatch.setattr(calc_cascade, "_map_cascade_calc", fake_map)
    calc.friction_surf = None
    samples_df = pd.DataFrame({"x_pop": [1.0, 1.0, 1.0], "x_haz": [1.0, 1.0, 1.0]})

    result = calc._compute_metrics(samples_df, chunksize=2, processes=1)

    assert calls == [2, 1]
    assert isinstance(result, list)
    assert len(result) == 2


# ========================================================================
# Sensitivity and output
# ========================================================================


def test_sensitivity(calc):
    """Sobol indices are computed for every network statistic."""
    unc_sample = calc.make_sample(N=4)
    unc_output = calc.uncertainty(unc_sample)

    sens_output = calc.sensitivity(unc_output, sensitivity_method="sobol")

    assert sens_output.sensitivity_method == "sobol"
    assert sens_output.sensitivity_metrics == ["imp_met"]
    sens_df = sens_output.imp_met_sens_df
    assert "healthcare_access_loss" in sens_df.columns
    assert set(sens_df["param"].dropna()) >= {"x_pop", "x_haz"}
    sens_output.plot_sensitivity()
    plt.close("all")


def test_unc_cascade_output_init():
    """The output stores samples, unit and the metric dataframe."""
    samples_df = pd.DataFrame({"x_pop": [1.0, 0.5]})
    imp_met_unc_df = pd.DataFrame({"healthcare_access_loss": [100.0, 0.0]})

    output = UncCascadeOutput(samples_df, "people", imp_met_unc_df)

    assert output.unit == "people"
    assert output.imp_met_unc_df is imp_met_unc_df
    assert output.imp_met_unit_df is None
    assert output.imp_met_sens_df is None
    # without a units table, every statistic has the unit `unit`
    assert output.imp_met_units == {"healthcare_access_loss": "people"}
    pd.testing.assert_frame_equal(output.get_uncertainty(), imp_met_unc_df)


def test_save_load(unc_output, tmp_path):
    """The uncertainty output is saved to and loaded from hdf5 with its
    units, as an UncCascadeOutput."""
    filename = unc_output.to_hdf5(tmp_path / "unc_cascade")

    loaded = UncCascadeOutput.from_hdf5(filename)

    assert isinstance(loaded, UncCascadeOutput)
    assert loaded.unit == "people"
    assert loaded.imp_met_units == unc_output.imp_met_units
    assert loaded.imp_met_sens_df is None
    pd.testing.assert_frame_equal(loaded.imp_met_unc_df, unc_output.imp_met_unc_df)
    pd.testing.assert_frame_equal(loaded.samples_df, unc_output.samples_df)
    # the base class loads the same data (without the unit labels in plots)
    loaded_base = UncOutput.from_hdf5(filename)
    pd.testing.assert_frame_equal(
        loaded_base.imp_met_unit_df, unc_output.imp_met_unit_df
    )


def test_from_hdf5_without_units_table(tmp_path):
    """Outputs saved without a units table load with the unit `unit` for
    every statistic."""
    output = UncCascadeOutput(
        pd.DataFrame({"x_pop": [1.0, 0.5]}),
        "people",
        pd.DataFrame({"healthcare_access_loss": [100.0, 0.0]}),
    )
    filename = output.to_hdf5(tmp_path / "unc_cascade_no_units")

    loaded = UncCascadeOutput.from_hdf5(filename)

    assert isinstance(loaded, UncCascadeOutput)
    assert loaded.imp_met_unit_df is None
    assert loaded.imp_met_units == {"healthcare_access_loss": "people"}


def test_from_hdf5_not_cascade_output_raises(tmp_path):
    """A file without imp_met_unc_df is not an output of CalcCascade."""
    filename = UncOutput(pd.DataFrame({"x_pop": [1.0, 0.5]})).to_hdf5(
        tmp_path / "unc_other"
    )

    with pytest.raises(ValueError, match="not an output of CalcCascade"):
        UncCascadeOutput.from_hdf5(filename)


def test_plot_uncertainty_labels_with_units():
    """Each statistic is labelled with its own unit."""
    rng = np.random.default_rng(0)
    output = UncCascadeOutput(
        pd.DataFrame({"x_pop": rng.uniform(size=20)}),
        "people",
        pd.DataFrame(
            {
                "healthcare_access_loss": rng.uniform(0, 100, size=20),
                "road_infra_impact": rng.uniform(0, 5000, size=20),
            }
        ),
        pd.DataFrame([{"healthcare_access_loss": "people", "road_infra_impact": "m"}]),
    )

    axes = output.plot_uncertainty()

    labels = {
        ax.get_xlabel().split(" [")[0]: ax.get_xlabel()
        for ax in np.ravel(axes)
        if ax.get_xlabel()
    }
    plt.close("all")
    assert labels["healthcare_access_loss"].endswith(" people] ")
    assert labels["road_infra_impact"].endswith(" m] ")
