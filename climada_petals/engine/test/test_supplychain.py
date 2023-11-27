"""This file is part of CLIMADA.

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

Test Supplychain class.
"""

import unittest
import numpy as np
import pandas as pd
import pymrio

from climada.entity.exposures.base import Exposures
from climada.engine.impact_calc import Impact
from climada_petals.engine.supplychain import (
    SupplyChain,
    MRIOT_DIRECTORY,
    parse_mriot_from_df,
)
from climada.util.constants import DEF_CRS
from climada.util.api_client import Client
from climada.util.files_handler import download_file
from scipy import sparse

def build_mriot(iso='iso3'):
    """
    This is an hypothetical Multi-Regional Input-Output Table adapted from the one in:
    M. P. Timmer et al. “An Illustrated User Guide to the World Input–Output
    Database: the Case of Global Automotive Production”. In: Review of International
    Economics 23.3 (2015), pp. 575–605.
    """

    idx_names = ['regions', 'sectors']
    _sectors = ['Services', 'Nat. Res.', 'Manuf. & Const.']

    if iso == 'iso3':
        _regions = ['USA', 'ROW']

    elif iso == 'iso2':
        _regions = ['US', 'ROW']

    _Z_multiindex = pd.MultiIndex.from_product(
	                [_regions, _sectors],
                    names = idx_names
                    )

    _Z_multiindex = pd.MultiIndex.from_product(
        [_regions, _sectors],
        names = idx_names
        )

    _Z_data = np.array([[634, 41, 1, 2241, 271, 3],
                        [29, 62, 2, 311, 128, 1],
                        [9, 1, 735, 3574, 311, 947],
                        [813, 123, 724, 20813, 11113, 594],
                        [490, 83, 796, 8196, 26678, 905],
                        [85, 9, 334, 1411, 1315, 1769]])

    Z = pd.DataFrame(
                data = _Z_data,
                index = _Z_multiindex,
                columns = _Z_multiindex
                )

    _X_data = np.array([4911, 797, 5963, 49398, 
                        93669, 6259]).reshape(6,1)
    X = pd.DataFrame(
                data = _X_data,
                index = _Z_multiindex,
                columns = ['total production']
                )

    _Y_data = np.array([1721, 264, 385, 15218, 
                        56522, 1337]).reshape(6,1)

    Y = pd.DataFrame(
	            data=_Y_data,
	            index = _Z_multiindex,
	            columns = ['final demand']
	            )

    io = pymrio.IOSystem()
    io.Z = Z
    io.Y = Y
    io.x = X

    return io

def dummy_exp_imp():
    " Generate dummy exposure and impacts "
    lat = np.array([1, 3])
    lon = np.array([1.5, 3.5])
    exp = Exposures(crs=DEF_CRS)
    exp.gdf['longitude'] = lon
    exp.gdf['latitude'] = lat
    exp.gdf['value'] = np.array([150., 80.])
    exp.gdf["region_id"] = [840, 608] # USA, PHL (ROW)

    imp = Impact(
        event_id=np.arange(2) + 10,
        event_name=np.arange(2),
        date=np.arange(2),
        coord_exp=np.vstack([lon, lat]).T,
        crs=DEF_CRS,
        unit="USD",
        eai_exp=np.array([6, 4.33]),
        at_event=np.array([55, 35]),
        frequency=np.array([1 / 6, 1 / 30]),
        frequency_unit="1/month",
        aai_agg=10.34,

        imp_mat=sparse.csr_matrix(
            np.array([[30, 25],
                      [30, 5]]))
        )

    return exp, imp

class TestSupplyChain(unittest.TestCase):
    def setUp(self) -> None:
        client = Client()

        tf = "WIOTtest_Nov16_ROW"
        if not MRIOT_DIRECTORY.joinpath(tf).is_file():
            dsf = client.get_dataset_info(name=tf, status="test_dataset").files[0]
            download_file(dsf.url, MRIOT_DIRECTORY)

        atl_prob_ds = client.get_dataset_info(
            name="atl_prob_no_name", status="test_dataset"
        )
        _, [self.HAZ_TEST_MAT] = client.download_dataset(atl_prob_ds)

    """Testing the SupplyChain class."""

    def test_read_wiot(self):
        """Test reading of wiod table."""

        file_loc = MRIOT_DIRECTORY / "WIOTtest_Nov16_ROW.xlsb"
        mriot_df = pd.read_excel(file_loc, engine="pyxlsb")
        Z, _, x = parse_mriot_from_df(
            mriot_df, col_iso3=2, col_sectors=1, rows_data=(5, 117), cols_data=(4, 116)
        )
        mriot = pymrio.IOSystem(Z=Z, x=x)

        sup = SupplyChain(mriot)

        self.assertAlmostEqual(sup.mriot.Z.iloc[0, 0], 12924.1797, places=3)
        self.assertAlmostEqual(sup.mriot.Z.iloc[0, -1], 0, places=3)
        self.assertAlmostEqual(sup.mriot.Z.iloc[-1, 0], 0, places=3)
        self.assertAlmostEqual(sup.mriot.Z.iloc[-1, -1], 22.222, places=3)

        self.assertAlmostEqual(
            sup.mriot.Z.iloc[0, 0],
            sup.mriot.Z.loc[
                (sup.mriot.get_regions()[0], sup.mriot.get_sectors()[0]),
                (sup.mriot.get_regions()[0], sup.mriot.get_sectors()[0]),
            ],
            places=3,
        )
        self.assertAlmostEqual(
            sup.mriot.Z.iloc[-1, -1],
            sup.mriot.Z.loc[
                (sup.mriot.get_regions()[-1], sup.mriot.get_sectors()[-1]),
                (sup.mriot.get_regions()[-1], sup.mriot.get_sectors()[-1]),
            ],
            places=3,
        )
        self.assertEqual(np.shape(sup.mriot.Z), (112, 112))
        self.assertAlmostEqual(sup.mriot.x.sum().values[0], 3533367.89439, places=3)

    def test_map_exp_to_mriot(self):
        mriot_iso3 = build_mriot()
        sup_iso3 = SupplyChain(mriot_iso3)

        mriot_iso2 = build_mriot(iso='iso2')
        sup_iso2 = SupplyChain(mriot_iso2)
    
        # Test a country listed in IOT, e.g. USA
        usa_regid = 840
        ## WIOD16
        self.assertEqual(
            sup_iso3.map_exp_to_mriot(usa_regid, 'WIOD16'), 'USA'
        )
        ## EXIOBASE3
        self.assertEqual(
            sup_iso2.map_exp_to_mriot(usa_regid, 'EXIOBASE3'), 'US'
        )
        ## OECD21
        self.assertEqual(
            sup_iso3.map_exp_to_mriot(usa_regid, 'OECD21'), 'USA'
        )
        ## Unspecified type
        self.assertEqual(
            sup_iso3.map_exp_to_mriot(usa_regid, ''), usa_regid
        )

        # Test a non-listed country in IOT, e.g. PHL
        phl_regid = 608
        ## WIOD16
        self.assertEqual(
            sup_iso3.map_exp_to_mriot(phl_regid, 'WIOD16'), 'ROW'
        )
        ## EXIOBASE3
        self.assertEqual(
            sup_iso2.map_exp_to_mriot(phl_regid, 'EXIOBASE3'), 'ROW'
        )
        ## OECD21
        self.assertEqual(
            sup_iso3.map_exp_to_mriot(phl_regid, 'OECD21'), 'ROW'
        )
        ## Unspecified type
        self.assertEqual(
            sup_iso3.map_exp_to_mriot(phl_regid, ''), phl_regid
        )

    def test_calc_shock_to_sectors(self):
        """Test sectorial exposure, impact and shock calculations."""

        mriot = build_mriot()
        sup = SupplyChain(mriot)

        # take one mriot type that supports iso-3 
        sup.mriot.meta.change_meta("name", "WIOD16-2011")

        exp, imp = dummy_exp_imp()
        sup.calc_shock_to_sectors(exp, imp)

        # Test sec exposure, impact and shock for one country (e.g. USA) and all sectors
        reg_id = 840
        reg_iso3 = 'USA'
        frac_per_sec = sup.mriot.x.loc[reg_iso3].values.T / sup.mriot.x.loc[reg_iso3].sum().values

        # Test sectorial exposure
        exp_cnt = exp.gdf[exp.gdf.region_id == reg_id].value.sum()
        expected_secs_exp = exp_cnt * frac_per_sec

        np.testing.assert_array_equal(
            sup.secs_exp[reg_iso3].values, expected_secs_exp
        )

        # Test sectorial impact
        imp_cnt = imp.imp_mat.todense()[:,exp.gdf.region_id == reg_id]
        expected_secs_imp = imp_cnt * frac_per_sec

        np.testing.assert_array_equal(
            sup.secs_imp[reg_iso3].values, expected_secs_imp
        )

        # Test sectorial shock with default shock factor
        expected_secs_shock = expected_secs_imp / expected_secs_exp

        np.testing.assert_array_equal(
            sup.secs_shock[reg_iso3].values, expected_secs_shock
        )

        # Test sectorial shock with user-defined shock factor
        shock_factor = pd.DataFrame(np.array([1,2,3,4,5,6])*0.1,
                                    index=sup.mriot.x.index)
        sup.calc_shock_to_sectors(exp, imp, shock_factor = shock_factor.values.flatten())

        expected_secs_shock = np.array(expected_secs_imp / 
                                       expected_secs_exp) * shock_factor.loc[reg_iso3].values.T

        np.testing.assert_array_equal(
            sup.secs_shock[reg_iso3].values, expected_secs_shock
        )
    
        # Test sec exposure, impact and shock for one country (e.g. USA) 
        # assuming only one sector is impacted (e.g. Services)
        aff_sec = 'Services'
        sup.calc_shock_to_sectors(exp, imp, impacted_secs=aff_sec)

        # Test sectorial exposure - since it's only one sector, frac is 1
        frac_exp_per_sec = np.array([1])
        expected_secs_exp = exp_cnt * frac_exp_per_sec

        np.testing.assert_array_equal(
            sup.secs_exp[reg_iso3, aff_sec].values, expected_secs_exp
        )

        # Test sectorial impact
        frac_imp_per_sec =  np.array([1])
        expected_secs_imp = np.array(imp_cnt).flatten() * frac_imp_per_sec

        np.testing.assert_array_equal(
            sup.secs_imp[reg_iso3, aff_sec].values, expected_secs_imp
        )

    def test_calc_impacts(self):
        """Test running indirect impact calculations."""

        mriot = build_mriot()
        sup = SupplyChain(mriot)

        # take one mriot type that supports iso-3
        sup.mriot.meta.change_meta("name", "WIOD16-2011")

        # apply 20 % shock to Service sector in the USA
        shock = pd.DataFrame(
                            np.array([[0.2, 0, 0, 0, 0, 0]]),
                            columns=sup.mriot.x.index
                            )
        sup.secs_shock = shock

        # calc prod losses according to ghosh
        sup.calc_impacts(io_approach="ghosh")

        # manually build a 20% loss in value added
        # to the USA service sector
        delta_v = np.array([570., 0, 0, 0, 0, 0])

        # the expected shock is then the dot product
        # of the value added loss and the ghosh inverse
        expected_prod_loss = delta_v.dot(sup.inverse['ghosh'])

        np.testing.assert_array_equal(
            sup.supchain_imp['ghosh'].round(0).values.flatten(), 
            expected_prod_loss.round(0)
            )

        # calc prod losses according to leontief
        sup.calc_impacts(io_approach="leontief")

        # manually build a 20% loss in demand
        # to the USA service sector
        delta_y = np.array([344., 0, 0, 0, 0, 0])

        # the expected shock is then the dot product
        # of the demand loss and the leontief inverse
        expected_prod_loss = sup.inverse['leontief'].dot(delta_y)

        np.testing.assert_array_equal(
            sup.supchain_imp['leontief'].round(0).values.flatten(),
            expected_prod_loss.round(0)
            )

        # total intensity vector: elements are the sector-specific
        # indirect risks per unit sector output
        tot_int_vec = shock.dot(sup.inverse['leontief'])
        expected_prod_loss = sup.mriot.x.values.flatten() * tot_int_vec.values[0]

        sup.calc_impacts(io_approach="eeioa")

        np.testing.assert_array_equal(
            sup.supchain_imp['eeioa'].round(0).values.flatten(),
            expected_prod_loss.round(0)
            )

## Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSupplyChain)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
