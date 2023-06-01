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
from climada.entity import ImpactFuncSet, ImpfTropCyclone
from climada.engine.impact_calc import Impact, ImpactCalc
from climada.hazard.base import Hazard
from climada_petals.engine.supplychain import (
    SupplyChain,
    MRIOT_DIRECTORY,
    parse_mriot_from_df,
)
from climada.util.constants import DEF_CRS, EXP_DEMO_H5
from climada.util.api_client import Client
from climada.util.files_handler import download_file
from scipy import sparse

def build_mriot(iso='iso3'):
    """
    This example of Multi-Regional Input-Output Table is inspired by the book:
    Miller, R. E., & Blair, P. D. (2009). Input-Output Analysis: Foundations and Extensions, 
    Second Edition. 784. pgg. 97-101.
    """

    idx_names = ['regions', 'sectors']
    _sectors = ['Nat. Res.', 'Manuf. & Const.', 'Services']

    if iso == 'iso3':
        _regions = ['USA', 'CHN', 'ROW']

    elif iso == 'iso2':
        _regions = ['US', 'CH', 'ROW']

    _Z_multiindex = pd.MultiIndex.from_product(
	                [_regions, _sectors],
                    names = idx_names
                    )

    _Z_multiindex = pd.MultiIndex.from_product(
        [_regions, _sectors],
        names = idx_names
        )

    _Z_data = np.array([[1724, 6312, 406, 188, 1206, 86, 14, 49, 4],
                        [2381, 18458, 2987, 301, 3331, 460, 39, 234, 57],
                        [709, 3883, 1811, 64, 432, 138, 5, 23, 5],
                        [149, 656, 42, 3564, 8828, 806, 103, 178, 15],
                        [463, 3834, 571, 3757, 34931, 5186, 202, 140, 268],
                        [49, 297, 99, 1099, 6613, 2969, 31, 163, 62],
                        [9, 51, 3, 33, 254, 18, 1581, 3154, 293],
                        [32, 272, 41, 123, 1062, 170, 1225, 6704, 1733],
                        [4, 25, 7, 25, 168, 47, 425, 2145, 1000]])

    Z = pd.DataFrame(
                data = _Z_data,
                index = _Z_multiindex,
                columns = _Z_multiindex
                )

    _X_data = np.array([16651, 49563, 15011, 27866, 
                        81253, 23667, 11661, 21107, 8910]).reshape(9,1)
    X = pd.DataFrame(
                data = _X_data,
                index = _Z_multiindex,
                columns = ['total production']
                )

    _Y_data = X - Z.sum(1).values.reshape(9,1)

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
    lat = np.array([1, 2, 3])
    lon = np.array([1.5, 2.5, 3.5])
    exp = Exposures(crs=DEF_CRS)
    exp.gdf['longitude'] = lon
    exp.gdf['latitude'] = lat
    exp.gdf['value'] = np.array([50., 100., 80])
    exp.gdf["region_id"] = [840, 156, 608] # USA, CHN, PHL (ROW)

    imp = Impact(
        event_id=np.arange(2) + 10,
        event_name=np.arange(2),
        date=np.arange(2),
        coord_exp=np.vstack([lon, lat]).T,
        crs=DEF_CRS,
        unit="USD",
        eai_exp=np.array([1.16, 1.16, 1.16]),
        at_event=np.array([3, 90]),
        frequency=np.array([1 / 6, 1 / 30]),
        frequency_unit="1/month",
        aai_agg=3.47,

        imp_mat=sparse.csr_matrix(
            np.array([[1, 1, 1],
                      [30, 30, 30]]))
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

        # Test sectorial exposure
        exp_cnt = exp.gdf[exp.gdf.region_id == reg_id].value.sum()
        frac_exp_per_sec = sup.mriot.x.loc[reg_iso3].values.T / sup.mriot.x.loc[reg_iso3].sum().values
        expected_secs_exp = exp_cnt * frac_exp_per_sec

        np.testing.assert_array_equal(
            sup.secs_exp[reg_iso3].values, expected_secs_exp
        )

        # Test sectorial impact
        imp_cnt = imp.imp_mat.todense()[:, exp.gdf.region_id == reg_id]
        frac_imp_per_sec = sup.mriot.x.loc[reg_iso3].values.T / sup.mriot.x.loc[reg_iso3].sum().values
        expected_secs_imp = imp_cnt * frac_imp_per_sec

        np.testing.assert_array_equal(
            sup.secs_imp[reg_iso3].values, expected_secs_imp
        )

        # Test sectorial shock with default shock factor
        expected_secs_shock = expected_secs_imp / expected_secs_exp

        np.testing.assert_array_equal(
            sup.secs_shock[reg_iso3].values, expected_secs_shock
        )

        # Test sectorial shock with user-defined shock factor
        shock_factor = pd.DataFrame(np.array([1,2,3,4,5,6,7,8,9])*0.1, 
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

        file_loc = MRIOT_DIRECTORY / "WIOTtest_Nov16_ROW.xlsb"
        mriot_df = pd.read_excel(file_loc, engine="pyxlsb")
        Z, _, x = parse_mriot_from_df(
            mriot_df, col_iso3=2, col_sectors=1, rows_data=(5, 117), cols_data=(4, 116)
        )
        Y = x.subtract(Z.sum(1), 0)
        mriot = pymrio.IOSystem(Z=Z, Y=Y, x=x)
        mriot.meta.change_meta("name", "WIOD16-test")
        mriot.unit = "M.EUR"

        sup = SupplyChain(mriot)

        # Tropical cyclone over Florida and Caribbean
        hazard = Hazard.from_mat(self.HAZ_TEST_MAT)

        # Read demo entity values
        # Set the entity default file to the demo one
        exp = Exposures.from_hdf5(EXP_DEMO_H5)
        exp.check()
        exp.gdf.region_id = 840 # USA
        exp.assign_centroids(hazard)

        impf_tc = ImpfTropCyclone.from_emanuel_usa()
        impf_set = ImpactFuncSet()
        impf_set.append(impf_tc)
        impf_set.check()

        impacted_secs = range(15, 25)
        impacted_secs = sup.mriot.get_sectors()[impacted_secs].tolist()
        imp = ImpactCalc(exp, impf_set, hazard)
        impact = imp.impact()

        io_approach = "ghosh"
        sup.calc_impacts(io_approach=io_approach,
                         exposure=exp,
                         impact=impact,
                         impacted_secs=impacted_secs)

        self.assertAlmostEqual(sup.mriot.Z.shape, sup.inverse[io_approach].shape)
        self.assertAlmostEqual(sup.mriot.Z.columns[43], sup.inverse[io_approach].columns[43])
        self.assertAlmostEqual(sup.mriot.Z.index[98], sup.inverse[io_approach].index[98])

        self.assertAlmostEqual(sup.inverse[io_approach].iloc[10, 0], 0.0735, places=3)
        self.assertAlmostEqual(sup.inverse[io_approach].iloc[0, 8], 0.00064, places=3)

        self.assertAlmostEqual(
            sup.supchain_imp[io_approach].values.sum(), 7093283.110973164, places=2
        )

        io_approach = "leontief"
        sup.calc_impacts(io_approach=io_approach)
 
        self.assertAlmostEqual(sup.mriot.Z.shape, sup.inverse[io_approach].shape)
        self.assertAlmostEqual(sup.mriot.Z.columns[56], sup.inverse[io_approach].columns[56])
        self.assertAlmostEqual(sup.mriot.Z.index[33], sup.inverse[io_approach].index[33])

        self.assertAlmostEqual(sup.inverse[io_approach].iloc[10, 0], 0.01690, places=3)
        self.assertAlmostEqual(sup.inverse[io_approach].iloc[0, 8], 0.0057, places=3)

        self.assertAlmostEqual(
            sup.supchain_imp[io_approach].values.sum(), 4460353.601586872, places=2
        )

        io_approach = "eeioa"
        sup.calc_impacts(io_approach=io_approach)

        self.assertAlmostEqual(sup.mriot.Z.shape, sup.inverse[io_approach].shape)

        self.assertAlmostEqual(sup.mriot.Z.columns[20], sup.inverse[io_approach].columns[20])
        self.assertAlmostEqual(sup.mriot.Z.index[15], sup.inverse[io_approach].index[15])

        self.assertAlmostEqual(sup.inverse[io_approach].iloc[10, 0], 0.016903, places=3)
        self.assertAlmostEqual(sup.inverse[io_approach].iloc[0, 8], 0.0057, places=3)

        self.assertAlmostEqual(
            sup.supchain_imp[io_approach].values.sum(), 13786581.420801481, places=2
        )

## Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSupplyChain)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
