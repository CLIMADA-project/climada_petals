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
import warnings
import numpy as np
import pandas as pd
import pymrio

from climada.entity.exposures.base import Exposures
from climada.engine.impact_calc import Impact
from climada_petals.engine.supplychain import (
    SupplyChain,
    MRIOT_DIRECTORY,
    VA_NAME,
    mriot_file_name,
    parse_mriot_from_df,
    calc_B,
    calc_va,
    calc_G,
    calc_x_from_G,
)
from climada.util.constants import DEF_CRS
from climada.util.api_client import Client
from climada.util.files_handler import download_file
from scipy import sparse

def build_mock_mriot_miller(iso='iso3'):
    """
    This is an hypothetical Multi-Regional Input-Output Table adapted from the one in:
    Miller, R. E., & Blair, P. D. (2009). Input-output analysis: foundations and
    extensions. : Cambridge University Press.
    """

    idx_names = ['region', 'sector']
    fd_names = ['region', 'final demand cat']
    _sectors = ['Nat. Res.', 'Manuf. & Const.', 'Service']
    _final_demand = ["final demand"]

    if iso == 'iso3':
        _regions = ['USA', 'ROW']

    elif iso == 'iso2':
        _regions = ['US', 'ROW']

    _Z_multiindex = pd.MultiIndex.from_product(
	                [_regions, _sectors],
                    names = idx_names
                    )

    _Y_multiindex = pd.MultiIndex.from_product(
	                [_regions, _final_demand],
                    names = fd_names
                    )

    _Z_data = np.array([[150,500,50,25,75,0],
                        [200,100,400,200,100,0],
                        [300,500,50,60,40,0],
                        [75,100,60,200,250,0],
                        [50,25,25,150,100,0],
                        [0,0,0,0,0,0]])

    Z = pd.DataFrame(
                data = _Z_data,
                index = _Z_multiindex,
                columns = _Z_multiindex
                )

    _X_data = np.array([1000,2000,1000,1200,800,0]).reshape(6,1)
    X = pd.DataFrame(
                data = _X_data,
                index = _Z_multiindex,
                columns = ['total production']
                )

    _Y_data = np.array([[180,800,40,65,150,0],
                      [20,200,10,450,300,0]]).reshape(6,2)

    Y = pd.DataFrame(
	            data=_Y_data,
	            index = _Z_multiindex,
	            columns = _Y_multiindex,
	            )

    io = pymrio.IOSystem()
    io.Z = Z
    io.Y = Y
    io.x = X

    return io

def build_mock_mriot_timmer(iso='iso3'):
    """
    This is an hypothetical Multi-Regional Input-Output Table adapted from the one in:
    M. P. Timmer et al. “An Illustrated User Guide to the World Input–Output
    Database: the Case of Global Automotive Production”. In: Review of International
    Economics 23.3 (2015), pp. 575–605.
    """

    idx_names = ['region', 'sector']
    fd_names = ['region', 'final demand cat']
    _sectors = ['Services', 'Nat. Res.', 'Manuf. & Const.']
    _final_demand = ["final demand"]

    if iso == 'iso3':
        _regions = ['USA', 'ROW']

    elif iso == 'iso2':
        _regions = ['US', 'ROW']

    _Z_multiindex = pd.MultiIndex.from_product(
	                [_regions, _sectors],
                    names = idx_names
                    )

    _Y_multiindex = pd.MultiIndex.from_product(
	                [_regions, _final_demand],
                    names = fd_names
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
                columns = ['indout']
                )

    _Y_data = np.array([[1721*0.2, 264*0.2, 385*0.2, 15218*0.2,
                        56522*0.2, 1337*0.2],
                      [1721*0.8, 264*0.8, 385*0.8, 15218*0.8,
                        56522*0.8, 1337*0.8]]).T

    Y = pd.DataFrame(
	            data=_Y_data,
	            index = _Z_multiindex,
	            columns = _Y_multiindex,
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

class TestCalcFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.mriot = build_mock_mriot_miller()
        self.expected_va = pd.DataFrame.from_dict({'index': [VA_NAME],
                                                  'columns': [('USA', 'Nat. Res.'),
                                                              ('USA', 'Manuf. & Const.'),
                                                              ('USA', 'Service'),
                                                              ('ROW', 'Nat. Res.'),
                                                              ('ROW', 'Manuf. & Const.'),
                                                              ('ROW', 'Service')],
                                                  'data': [[225, 775, 415, 565, 235, 0]],
                                                  'index_names': [None],
                                                  'column_names': ["region","sector"]}, orient="tight")

        self.expected_B = pd.DataFrame.from_dict({'index': [('USA', 'Nat. Res.'),
                                                            ('USA', 'Manuf. & Const.'),
                                                            ('USA', 'Service'),
                                                            ('ROW', 'Nat. Res.'),
                                                            ('ROW', 'Manuf. & Const.'),
                                                            ('ROW', 'Service')],
                                                  'columns': [('USA', 'Nat. Res.'),
                                                              ('USA', 'Manuf. & Const.'),
                                                              ('USA', 'Service'),
                                                              ('ROW', 'Nat. Res.'),
                                                              ('ROW', 'Manuf. & Const.'),
                                                              ('ROW', 'Service')],
                                                  'data': [[0.15, 0.1, 0.3, 0.0625, 0.0625, 0.0],
                                                           [0.5, 0.05, 0.5, 0.08333333333333334, 0.03125, 0.0],
                                                           [0.05, 0.2, 0.05, 0.05, 0.03125, 0.0],
                                                           [0.025, 0.1, 0.06, 0.16666666666666669, 0.1875, 0.0],
                                                           [0.075, 0.05, 0.04, 0.20833333333333334, 0.125, 0.0],
                                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                                                  'index_names': ['region', 'sector'],
                                                  'column_names': ['region', 'sector']}, orient="tight")

        self.expected_G = pd.DataFrame.from_dict({'index': [('USA', 'Nat. Res.'),
                                                       ('USA', 'Manuf. & Const.'),
                                                       ('USA', 'Service'),
                                                       ('ROW', 'Nat. Res.'),
                                                       ('ROW', 'Manuf. & Const.'),
                                                       ('ROW', 'Service')],
                                             'columns': [('USA', 'Nat. Res.'),
                                                         ('USA', 'Manuf. & Const.'),
                                                         ('USA', 'Service'),
                                                         ('ROW', 'Nat. Res.'),
                                                         ('ROW', 'Manuf. & Const.'),
                                                         ('ROW', 'Service')],
                                             'data': [[1.423409149449475,
                                                       0.31730628908222186,
                                                       0.6382907140159585,
                                                       0.22266222823849854,
                                                       0.18351388112243291,
                                                       0.0],
                                                      [0.9304525740622405,
                                                       1.4236653207934866,
                                                       1.0737395920925703,
                                                       0.3333461635366272,
                                                       0.227085251508225,
                                                       0.0],
                                                      [0.2909093898805612,
                                                       0.33533997280126726,
                                                       1.3362516095901116,
                                                       0.1644572429148015,
                                                       0.11571977927290393,
                                                       0.0],
                                                      [0.23003182777496692,
                                                       0.24552984791016436,
                                                       0.30014627080837747,
                                                       1.3406124940000408,
                                                       0.32319338350959714,
                                                       0.0],
                                                      [0.24324341481161507,
                                                       0.18233930089239148,
                                                       0.24861636642800972,
                                                       0.36484518239388875,
                                                       1.2538040568323914,
                                                       0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                                             'index_names': ['region', 'sector'],
                                             'column_names': ['region', 'sector']}, orient="tight")

        self.va_changed = pd.DataFrame.from_dict({'index': [VA_NAME],
                                                  'columns': [('USA', 'Nat. Res.'),
                                                              ('USA', 'Manuf. & Const.'),
                                                              ('USA', 'Service'),
                                                              ('ROW', 'Nat. Res.'),
                                                              ('ROW', 'Manuf. & Const.'),
                                                              ('ROW', 'Service')],
                                                  'data': [[225, 400, 415, 565, 235, 0]],
                                                  'index_names': [None],
                                                  'column_names': ["region","sector"]}, orient="tight")

        self.expected_x_changed = pd.DataFrame.from_dict({'index': [('USA', 'Nat. Res.'),
                                                                    ('USA', 'Manuf. & Const.'),
                                                                    ('USA', 'Service'),
                                                                    ('ROW', 'Nat. Res.'),
                                                                    ('ROW', 'Manuf. & Const.'),
                                                                    ('ROW', 'Service')],
                                                          'columns': ['indout'],
                                                          'data': [[881.0101415941668],
                                                                   [1466.1255047024426],
                                                                   [874.2475101995246],
                                                                   [1107.9263070336883],
                                                                   [731.6227621653532],
                                                                   [0.0]],
                                                          'index_names': ['region', 'sector'],
                                                          'column_names': [None]}, orient="tight")

    def test_calc_v(self):
        # Test calc_va with DataFrame
        va = calc_va(self.mriot.Z, self.mriot.x)
        pd.testing.assert_frame_equal(va, self.expected_va)

        # Test calc_va with NumPy array
        va = calc_va(self.mriot.Z.values, self.mriot.x.values)
        np.testing.assert_array_equal(va.values, self.expected_va.values)

    def test_calc_B(self):
        # Test calc_B with DataFrame
        B = calc_B(self.mriot.Z, self.mriot.x)
        pd.testing.assert_frame_equal(B, self.expected_B)

        # Test calc_B with NumPy array
        B = calc_B(self.mriot.Z.values, self.mriot.x.values)
        np.testing.assert_array_equal(B, self.expected_B.values)

    def test_calc_G(self):
        # Test calc_G with DataFrame
        G = calc_G(self.expected_B)
        pd.testing.assert_frame_equal(G, self.expected_G)

        # Test calc_G with NumPy array
        G = calc_G(self.expected_B.values)
        np.testing.assert_array_equal(G, self.expected_G.values)

    def test_calc_x_from_G(self):
        # Test calc_x_from_G with DataFrame
        x = calc_x_from_G(self.expected_G,self.va_changed)
        pd.testing.assert_frame_equal(x, self.expected_x_changed)

        # Test calc_x_from_G with NumPy array
        x = calc_x_from_G(self.expected_G.values, self.va_changed.values)
        np.testing.assert_array_equal(x, self.expected_x_changed.values)


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

    def test_mriot_file_name(self):
        self.assertEqual(mriot_file_name("EXIOBASE3", 2015), "IOT_2015_ixi.zip")
        self.assertEqual(mriot_file_name("WIOD16", 2016), "WIOT2016_Nov16_ROW.xlsb")
        self.assertEqual(mriot_file_name("OECD21", 2021), "ICIO2021_2021.csv")
        with self.assertRaises(ValueError):
            mriot_file_name("Unknown", 2020)

    def test_read_wiod(self):
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
        """Test mapping exposure to MRIOT."""

        mriot_iso3 = build_mock_mriot_timmer()
        sup_iso3 = SupplyChain(mriot_iso3)

        mriot_iso2 = build_mock_mriot_timmer(iso='iso2')
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

        mriot = build_mock_mriot_timmer()
        sup = SupplyChain(mriot)

        # take one mriot type that supports iso-3
        sup.mriot.meta.change_meta("name", "WIOD16-2011")

        exp, imp = dummy_exp_imp()
        sup.calc_shock_to_sectors(exp, imp)

        # Check that events_date are correctly set.
        np.testing.assert_array_equal(
            sup.events_date, imp.date
        )

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

        # Test sec exposure, impact and shock for one country (e.g. USA)
        # assuming a range of sector is impacted
        aff_sec = np.array([0,1])
        sup.calc_shock_to_sectors(exp, imp, impacted_secs=aff_sec)

        # Test sectorial exposure
        indus_aff = pd.IndexSlice[reg_iso3, mriot.get_sectors()[aff_sec]]
        frac_exp_per_sec = sup.mriot.x.loc[indus_aff,:].values.T / sup.mriot.x.loc[indus_aff,:].sum().values
        frac_imp_per_sec = frac_exp_per_sec
        expected_secs_exp = exp_cnt * frac_exp_per_sec

        np.testing.assert_array_equal(
            sup.secs_exp.loc[:,indus_aff].values,
            expected_secs_exp
        )

        # Test sectorial impact
        expected_secs_imp = (np.array(imp_cnt) * frac_imp_per_sec)

        np.testing.assert_array_equal(
            sup.secs_imp.loc[:,indus_aff].values,
            expected_secs_imp
        )
        with self.assertWarns(Warning):
            sup.calc_shock_to_sectors(exp, imp, impacted_secs=aff_sec, shock_factor=5)

    def test_calc_impacts_unknown(self):
        """Test running indirect impact calculations with unknown approach."""
        mriot = build_mock_mriot_timmer()
        sup = SupplyChain(mriot)
        with self.assertRaises(KeyError):
            sup.calc_impacts(io_approach="xx")

    def test_calc_impacts(self):
        """Test running indirect impact calculations with ghosh, leontief and eeioa."""

        mriot = build_mock_mriot_timmer()
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
        delta_v = np.array([[570., 0, 0, 0, 0, 0]])

        # the expected shock is then the dot product
        # of the value added loss and the ghosh inverse
        expected_prod_loss = sup.inverse['ghosh'].dot(delta_v.T)

        np.testing.assert_array_equal(
            sup.supchain_imp['ghosh'].round(0).values,
            expected_prod_loss.round(0).T
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
