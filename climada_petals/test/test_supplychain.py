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

Integration tests for supplychain module.
"""

import numpy as np
import pandas as pd
import pymrio
from climada.entity import Exposures
from climada.engine import Impact
from climada_petals.engine.supplychain.core import DirectShocksSet
from climada.util.constants import DEF_CRS
from scipy import sparse
import pytest

from climada_petals.engine.supplychain.mriot_handling import lexico_reindex


# Mock MRIOT
@pytest.fixture
def mock_mriot_miller():
    """
    This is an hypothetical Multi-Regional Input-Output Table adapted from the one in:
    Miller, R. E., & Blair, P. D. (2009). Input-output analysis: foundations and
    extensions. : Cambridge University Press.
    """

    idx_names = ["region", "sector"]
    fd_names = ["region", "final demand cat"]
    _sectors = ["Nat. Res.", "Manuf. & Const.", "Service"]
    _final_demand = ["final demand"]
    _regions = ["USA", "PHL"]
    _Z_multiindex = pd.MultiIndex.from_product([_regions, _sectors], names=idx_names)

    _Y_multiindex = pd.MultiIndex.from_product(
        [_regions, _final_demand], names=fd_names
    )

    _Z_data = np.array(
        [
            [150, 500, 50, 25, 75, 0],
            [200, 100, 400, 200, 100, 0],
            [300, 500, 50, 60, 40, 0],
            [75, 100, 60, 200, 250, 0],
            [50, 25, 25, 150, 100, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    Z = pd.DataFrame(data=_Z_data, index=_Z_multiindex, columns=_Z_multiindex)

    _X_data = np.array([1000, 2000, 1000, 1200, 800, 0]).reshape(6, 1)
    X = pd.DataFrame(data=_X_data, index=_Z_multiindex, columns=["gross production"])

    _Y_data = np.array(
        [[180, 800, 40, 65, 150, 0], [20, 200, 10, 450, 300, 0]]
    ).reshape(6, 2)

    Y = pd.DataFrame(
        data=_Y_data,
        index=_Z_multiindex,
        columns=_Y_multiindex,
    )

    io = pymrio.IOSystem(Z=Z, Y=Y, x=X, name="Mock_Miller")
    io.monetary_factor = 1
    return io


@pytest.fixture
def mock_exp():
    lat = np.array([121, -103.771556])
    lon = np.array([16, 44.967243])
    exp = Exposures(crs=DEF_CRS, lat=lat, lon=lon, value=np.array([100.0, 100.0]))
    exp.gdf["region_id"] = [608, 840]  # USA, PHL
    return exp


@pytest.fixture
def mock_impact(mock_exp):
    impact = Impact(
        event_id=np.arange(3) + 10,
        event_name=np.arange(3),
        date=np.arange(3),
        coord_exp=np.vstack([mock_exp.longitude, mock_exp.latitude]).T,
        crs=DEF_CRS,
        unit="USD",
        eai_exp=np.array([6, 4.33]),
        at_event=np.array([55, 35, 0]),
        frequency=np.array([1 / 6, 1 / 30, 1 / 2]),
        frequency_unit="1/month",
        aai_agg=10.34,
        imp_mat=sparse.csr_matrix(np.array([[30, 25], [30, 5], [0.0, 0.0]])),
    )
    return impact


def test_direct_shock_no_impact_distrib(mock_mriot_miller, mock_exp, mock_impact):
    direct_shocks = DirectShocksSet.from_exp_and_imp(
        mock_mriot_miller,
        mock_exp,
        mock_impact,
        "Test Shock",
        affected_sectors="all",
        impact_distribution=None,
        custom_mriot=True,
    )
    mock_mriot_miller = lexico_reindex(mock_mriot_miller)
    assert direct_shocks.mriot_name == "Mock_Miller"
    np.testing.assert_array_equal(
        direct_shocks.mriot_sectors, mock_mriot_miller.get_sectors()
    )
    np.testing.assert_array_equal(
        direct_shocks.mriot_regions, mock_mriot_miller.get_regions()
    )
    pd.testing.assert_index_equal(
        direct_shocks.mriot_industries,
        pd.MultiIndex.from_product(
            [mock_mriot_miller.get_regions(), mock_mriot_miller.get_sectors()],
            names=["region", "sector"],
        ),
    )
    pd.testing.assert_index_equal(
        direct_shocks.event_ids, pd.Index(mock_impact.event_id, name="event_id")
    )
    pd.testing.assert_index_equal(
        direct_shocks.event_ids_with_impact, pd.Index([10, 11], name="event_id")
    )
    assert direct_shocks.monetary_factor == mock_mriot_miller.monetary_factor
    assert direct_shocks.name == "Test Shock"
    pd.testing.assert_series_equal(
        direct_shocks.event_dates,
        pd.Series(
            mock_impact.date, index=pd.Index(mock_impact.event_id, name="event_id")
        ),
    )

    expected_exposure_assets = pd.Series(
        [40.0, 60.0, 0.0, 50.0, 25.0, 25.0],
        index=pd.MultiIndex.from_product(
            [
                mock_mriot_miller.get_regions(),
                mock_mriot_miller.get_sectors(),
            ],
            names=["region", "sector"],
        ),
    )
    pd.testing.assert_series_equal(
        direct_shocks.exposure_assets, expected_exposure_assets
    )
    pd.testing.assert_series_equal(
        direct_shocks.exposure_assets_not_null,
        expected_exposure_assets[expected_exposure_assets > 0],
    )

    expected_impacted_assets = pd.DataFrame(
        [
            [12.0, 18.0, 0.0, 12.5, 6.25, 6.25],
            [12.0, 18.0, 0.0, 2.5, 1.25, 1.25],
            [0, 0, 0, 0, 0, 0],
        ],
        columns=pd.MultiIndex.from_product(
            [mock_mriot_miller.get_regions(), mock_mriot_miller.get_sectors()],
            names=["region", "sector"],
        ),
        index=pd.Index(mock_impact.event_id, name="event_id"),
    )
    pd.testing.assert_frame_equal(
        direct_shocks.impacted_assets, expected_impacted_assets
    )
    pd.testing.assert_frame_equal(
        direct_shocks.impacted_assets_not_null,
        expected_impacted_assets.loc[
            expected_impacted_assets.ne(0).any(axis=1),
            expected_impacted_assets.ne(0).any(axis=0),
        ],
    )

    expected_relative_impact = pd.DataFrame(
        [
            [0.3, 0.3, 0.0, 0.25, 0.25, 0.25],
            [0.3, 0.3, 0.0, 0.05, 0.05, 0.05],
            [0, 0, 0, 0, 0, 0],
        ],
        columns=pd.MultiIndex.from_product(
            [mock_mriot_miller.get_regions(), mock_mriot_miller.get_sectors()],
            names=["region", "sector"],
        ),
        index=pd.Index(mock_impact.event_id, name="event_id"),
    )
    pd.testing.assert_frame_equal(
        direct_shocks.relative_impact, expected_relative_impact
    )


def test_direct_shock_with_impact_distrib(mock_mriot_miller, mock_exp, mock_impact):
    direct_shocks = DirectShocksSet.from_exp_and_imp(
        mock_mriot_miller,
        mock_exp,
        mock_impact,
        "Test Shock",
        affected_sectors="all",
        impact_distribution=mock_mriot_miller.x["gross production"],
        custom_mriot=True,
    )
    mock_mriot_miller = lexico_reindex(mock_mriot_miller)
    expected_impacted_assets = pd.DataFrame(
        [
            [12.0, 18.0, 0.0, 12.5, 6.25, 6.25],
            [12.0, 18.0, 0.0, 2.5, 1.25, 1.25],
            [0, 0, 0, 0, 0, 0],
        ],
        columns=pd.MultiIndex.from_product(
            [mock_mriot_miller.get_regions(), mock_mriot_miller.get_sectors()],
            names=["region", "sector"],
        ),
        index=pd.Index(mock_impact.event_id, name="event_id"),
    )
    print(expected_impacted_assets)
    print(direct_shocks.impacted_assets)
    pd.testing.assert_frame_equal(
        direct_shocks.impacted_assets, expected_impacted_assets
    )
    pd.testing.assert_frame_equal(
        direct_shocks.impacted_assets_not_null,
        expected_impacted_assets.loc[
            expected_impacted_assets.ne(0).any(axis=1),
            expected_impacted_assets.ne(0).any(axis=0),
        ],
    )

    expected_relative_impact = pd.DataFrame(
        [
            [0.3, 0.3, 0.0, 0.25, 0.25, 0.25],
            [0.3, 0.3, 0.0, 0.05, 0.05, 0.05],
            [0, 0, 0, 0, 0, 0],
        ],
        columns=pd.MultiIndex.from_product(
            [mock_mriot_miller.get_regions(), mock_mriot_miller.get_sectors()],
            names=["region", "sector"],
        ),
        index=pd.Index(mock_impact.event_id, name="event_id"),
    )
    pd.testing.assert_frame_equal(
        direct_shocks.relative_impact, expected_relative_impact
    )
