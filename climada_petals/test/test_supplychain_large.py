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

Computation intensive and downloading integration tests for supplychain module.
"""

import datetime as dt

import os

import numpy as np
import pandas as pd

from pymrio import IOSystem
import pytest

from climada.engine.impact_calc import ImpactCalc
from climada.entity import ImpactFuncSet, ImpfTropCyclone
from climada.util.api_client import Client
from climada_petals.engine.supplychain import BoARIOModel, DirectShocksSet, StaticIOModel
from climada_petals.engine.supplychain.mriot_handling import MRIOT_BASENAME, MRIOT_MONETARY_FACTOR, get_mriot
from climada_petals.engine.supplychain.utils import *

client = Client()


@pytest.fixture(params=[{"country":"JPN"}])
def f_exposure(request):
    return client.get_litpop(request.param["country"])

@pytest.fixture(params=[{"country":"JPN", "year":2019}])
def f_hazard(request):
    tc_jpn = client.get_hazard(
     "tropical_cyclone",
        properties={"country_iso3alpha": request.param["country"], "event_type": "observed"},
    )
    target_year = request.param["year"]
    events_in_target_year = np.array(
        [
            tc_jpn.event_name[i]
            for i in range(len(tc_jpn.event_name))
            if dt.datetime.fromordinal(tc_jpn.date[i]).year == target_year
        ]
    )

    tc_jpn_target_year = tc_jpn.select(event_names=events_in_target_year)

    return tc_jpn_target_year


@pytest.fixture
def f_impact(f_exposure, f_hazard):
    # Define impact function
    impf_tc = ImpfTropCyclone.from_emanuel_usa()
    impf_set = ImpactFuncSet()
    impf_set.append(impf_tc)
    impf_set.check()

    # Calculate direct impacts to JPN due to TC
    imp_calc = ImpactCalc(f_exposure, impf_set, f_hazard)
    return imp_calc.impact()

@pytest.fixture(params=["WIOD16",
                        "EXIOBASE3",
                        #"OECD23"
                        ])
def f_mriot(request):
    return get_mriot(request.param, 2010, save=True)


@pytest.mark.skipif(not (os.getenv("LARGE_TESTS") == "1"), reason="Skipping large test if not asked for")
class TestSupplyChainLargeTests:

    @pytest.mark.parametrize("mriot_name, expected", [
        ("EXIOBASE3", (MRIOT_MONETARY_FACTOR["EXIOBASE3"],MRIOT_BASENAME["EXIOBASE3"])),
        #("OECD23", (MRIOT_MONETARY_FACTOR["OECD23"],MRIOT_BASENAME["OECD23"])),
        ("WIOD16", (MRIOT_MONETARY_FACTOR["WIOD16"],MRIOT_BASENAME["WIOD16"])),
    ])
    def test_get_mriot(self, mriot_name, expected):
        mriot = get_mriot(mriot_name, 2010, redownload=True, save=True)
        assert isinstance(mriot,IOSystem)
        assert mriot.monetary_factor == expected[0]
        assert mriot.basename == expected[1]
        assert mriot.year == "2010"

    @pytest.fixture
    def f_direct_shock(self, f_mriot, f_exposure, f_impact):
        return DirectShocksSet.from_exp_and_imp(
            mriot=f_mriot,
            exposure=f_exposure,
            impact=f_impact,
            shock_name="Test impact",
            affected_sectors=f_mriot.get_sectors()[:6],
            impact_distribution=None,  # None distribute the impact by the production
)


    def test_direct_shock_all_sector(self, f_mriot, f_exposure, f_impact):
        direct_shock = DirectShocksSet.from_exp_and_imp(
            mriot=f_mriot,
            exposure=f_exposure,
            impact=f_impact,
            shock_name="Test impact",
            affected_sectors="all",
            impact_distribution=None,  # None distribute the impact by the production
)

        pd.testing.assert_index_equal(f_mriot.get_sectors(), direct_shock.mriot_sectors)
        pd.testing.assert_index_equal(f_mriot.get_regions(), direct_shock.mriot_regions)
        assert direct_shock.mriot_name == f_mriot.name

        np.testing.assert_array_equal(f_impact.event_id, direct_shock.event_ids)
        np.testing.assert_array_equal(f_impact.date, direct_shock.event_dates)
        np.testing.assert_almost_equal(direct_shock.exposure_assets.sum() * direct_shock.monetary_factor, f_exposure.gdf["value"].sum(), decimal=1)
        pd.testing.assert_series_equal((direct_shock.impacted_assets.sum(axis=1) * direct_shock.monetary_factor), pd.Series(f_impact.at_event, index=direct_shock.event_ids))

    def test_direct_shock_select_sector(self, f_mriot, f_exposure, f_impact):
        direct_shock = DirectShocksSet.from_exp_and_imp(
            mriot=f_mriot,
            exposure=f_exposure,
            impact=f_impact,
            shock_name="Test impact",
            affected_sectors=f_mriot.get_sectors()[:6],
            impact_distribution=None,  # None distribute the impact by the production
)

        pd.testing.assert_index_equal(f_mriot.get_sectors(), direct_shock.mriot_sectors)
        pd.testing.assert_index_equal(f_mriot.get_regions(), direct_shock.mriot_regions)
        assert direct_shock.mriot_name == f_mriot.name

        np.testing.assert_array_equal(f_impact.event_id, direct_shock.event_ids)
        np.testing.assert_array_equal(f_impact.date, direct_shock.event_dates)
        np.testing.assert_almost_equal(direct_shock.exposure_assets.sum() * direct_shock.monetary_factor, f_exposure.gdf["value"].sum(), decimal=1)
        pd.testing.assert_series_equal((direct_shock.impacted_assets.sum(axis=1) * direct_shock.monetary_factor), pd.Series(f_impact.at_event, index=direct_shock.event_ids))

    def test_static_io_model(self, f_mriot, f_direct_shock):
        model = StaticIOModel(f_mriot, f_direct_shock)
        assert model.direct_shocks is f_direct_shock
        res = model.calc_indirect_impacts(event_ids=None)
        assert isinstance(res,pd.DataFrame)
        pd.testing.assert_index_equal(res.columns, pd.Index(["event_id","region","sector","method","metric","value"]))
        assert f_direct_shock.event_ids in res.event_id.unique()
        assert f_mriot.get_regions() in res.region.unique()
        assert f_mriot.get_sectors() in res.sector.unique()

    def test_boario_model(self, f_mriot, f_direct_shock):
        dyn_model = BoARIOModel(
            f_mriot,
            f_direct_shock,
        )
        dyn_model.run_sim()
