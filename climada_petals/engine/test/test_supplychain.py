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
from climada.engine.impact_calc import ImpactCalc
from climada.hazard.base import Hazard
from climada_petals.engine.supplychain import (
    SupplyChain,
    MRIOT_DIRECTORY,
    parse_mriot_from_df,
)
from climada.util.constants import EXP_DEMO_H5
from climada.util.api_client import Client
from climada.util.files_handler import download_file


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

    def test_direct_production_impacts(self):
        """Test running direct impact calculations."""

        file_loc = MRIOT_DIRECTORY / "WIOTtest_Nov16_ROW.xlsb"
        mriot_df = pd.read_excel(file_loc, engine="pyxlsb")
        Z, _, x = parse_mriot_from_df(
            mriot_df, col_iso3=2, col_sectors=1, rows_data=(5, 117), cols_data=(4, 116)
        )
        mriot = pymrio.IOSystem(Z=Z, x=x)
        mriot.meta.change_meta("name", "WIOD16-test")
        mriot.unit = "M.EUR"

        sup = SupplyChain(mriot)

        # Tropical cyclone over Florida and Caribbean
        hazard = Hazard.from_mat(self.HAZ_TEST_MAT)

        # Read demo entity values
        # Set the entity default file to the demo one
        exp = Exposures.from_hdf5(EXP_DEMO_H5)
        exp.check()
        exp.gdf.region_id = 840  # assign right id for USA
        exp.assign_centroids(hazard)

        impf_tc = ImpfTropCyclone.from_emanuel_usa()
        impf_set = ImpactFuncSet()
        impf_set.append(impf_tc)
        impf_set.check()

        impacted_secs = list(range(10)) + list(range(15, 25))
        impacted_secs = sup.mriot.get_sectors()[impacted_secs].tolist()
        imp = ImpactCalc(exp, impf_set, hazard)
        impact = imp.impact()

        sup.calc_secs_exp_imp_shock(exp, impact, impacted_secs)

        sup.calc_direct_production_impacts()

        self.assertAlmostEqual(
            sup.dir_prod_impt_mat.values.sum(), 21595173505075.07, places=3
        )

        self.assertAlmostEqual(
            sup.dir_prod_impt_mat.values.sum(),
            sup.dir_prod_impt_mat.loc[:, "USA"].values.sum(),
            places=3,
        )

        self.assertAlmostEqual(
            sup.dir_prod_impt_mat.values.sum(),
            sup.dir_prod_impt_mat.loc[:, (slice(None), impacted_secs)].values.sum(),
            places=0,
        )

    def test_indirect_production_impact(self):
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
        exp.gdf.region_id = 840  # assign right id for USA
        exp.assign_centroids(hazard)

        impf_tc = ImpfTropCyclone.from_emanuel_usa()
        impf_set = ImpactFuncSet()
        impf_set.append(impf_tc)
        impf_set.check()

        impacted_secs = range(15, 25)
        impacted_secs = sup.mriot.get_sectors()[impacted_secs].tolist()
        imp = ImpactCalc(exp, impf_set, hazard)
        impact = imp.impact()

        sup.calc_secs_exp_imp_shock(exp, impact, impacted_secs)

        sup.calc_indirect_production_impacts(impact.event_id, io_approach="ghosh")

        self.assertAlmostEqual(sup.mriot.Z.shape, sup.inverse.shape)
        self.assertAlmostEqual(sup.mriot.Z.columns[43], sup.inverse.columns[43])
        self.assertAlmostEqual(sup.mriot.Z.index[98], sup.inverse.index[98])

        self.assertAlmostEqual(sup.inverse.iloc[10, 0], 0.0735, places=3)
        self.assertAlmostEqual(sup.inverse.iloc[0, 8], 0.00064, places=3)

        self.assertAlmostEqual(
            sup.indir_prod_impt_mat.values.sum(), 7093283110973.164, places=2
        )

        sup.calc_indirect_production_impacts(impact.event_id, io_approach="leontief")
 
        self.assertAlmostEqual(sup.mriot.Z.shape, sup.inverse.shape)
        self.assertAlmostEqual(sup.mriot.Z.columns[56], sup.inverse.columns[56])
        self.assertAlmostEqual(sup.mriot.Z.index[33], sup.inverse.index[33])

        self.assertAlmostEqual(sup.inverse.iloc[10, 0], 0.01690, places=3)
        self.assertAlmostEqual(sup.inverse.iloc[0, 8], 0.0057, places=3)

        self.assertAlmostEqual(
            sup.indir_prod_impt_mat.values.sum(), 4460353601586.872, places=2
        )

        sup.calc_indirect_production_impacts(impact.event_id, io_approach="eeioa")

        self.assertAlmostEqual(sup.mriot.Z.shape, sup.inverse.shape)

        self.assertAlmostEqual(sup.mriot.Z.columns[20], sup.inverse.columns[20])
        self.assertAlmostEqual(sup.mriot.Z.index[15], sup.inverse.index[15])

        self.assertAlmostEqual(sup.inverse.iloc[10, 0], 0.016903, places=3)
        self.assertAlmostEqual(sup.inverse.iloc[0, 8], 0.0057, places=3)

        self.assertAlmostEqual(
            sup.indir_prod_impt_mat.values.sum(), 13786581420801.48, places=2
        )

    def test_calc_total_impacts(self):
        """Test running total impact calculations."""

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
        exp.gdf.region_id = 840  # assign right id for USA
        exp.assign_centroids(hazard)

        impf_tc = ImpfTropCyclone.from_emanuel_usa()
        impf_set = ImpactFuncSet()
        impf_set.append(impf_tc)
        impf_set.check()

        impacted_secs = range(15, 25)
        impacted_secs = sup.mriot.get_sectors()[impacted_secs].tolist()
        imp = ImpactCalc(exp, impf_set, hazard)
        impact = imp.impact()

        sup.calc_production_impacts(
            impact, exp, impacted_secs=impacted_secs, io_approach="ghosh"
        )

        self.assertAlmostEqual(
            sup.tot_prod_impt_mat.values.sum(),
            sup.indir_prod_impt_mat.values.sum() + sup.dir_prod_impt_mat.values.sum(),
            places=0,
        )

    def calc_production_eai(self):
        """Test running production expected annual impact calculations."""

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
        exp.gdf.region_id = 840  # assign right id for USA
        exp.assign_centroids(hazard)

        impf_tc = ImpfTropCyclone.from_emanuel_usa()
        impf_set = ImpactFuncSet()
        impf_set.append(impf_tc)
        impf_set.check()

        impacted_secs = range(15, 25)
        impacted_secs = sup.mriot.get_sectors()[impacted_secs].tolist()
        imp = ImpactCalc(exp, impf_set, hazard)
        impact = imp.impact()

        sup.calc_production_impacts(
            impact, exp, impacted_secs=impacted_secs, io_approach="ghosh"
        )

        sup.calc_production_eai(impact.frequency)

        self.assertAlmostEqual((sup.mriot.Z.shape[0],), sup.dir_prod_impt_eai.shape)
        self.assertAlmostEqual((sup.mriot.Z.shape[0],), sup.indir_prod_impt_eai.shape)
        self.assertAlmostEqual(
            sup.dir_prod_impt_eai.sum(),
            sup.dir_prod_impt_eai.loc["USA"].sum(),
            places=3,
        )
        self.assertAlmostEqual(
            sup.dir_prod_impt_eai.values.sum(),
            sup.dir_prod_impt_eai.loc[(slice(None), impacted_secs)].values.sum(),
            places=3,
        )
        self.assertAlmostEqual(
            sup.dir_prod_impt_eai.sum(), 13413151245.388247, places=3
        )
        self.assertAlmostEqual(
            sup.indir_prod_impt_eai.sum(), 4405765907.4367485, places=3
        )
        self.assertAlmostEqual(
            sup.tot_prod_impt_eai.values.sum(),
            sup.indir_prod_impt_eai.values.sum() + sup.dir_prod_impt_eai.values.sum(),
            places=0,
        )

## Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSupplyChain)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
