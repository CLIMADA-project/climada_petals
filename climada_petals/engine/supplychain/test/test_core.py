from unittest import TestCase
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import numpy as np
import pandas as pd
from climada.engine import Impact
from climada.entity.exposures import Exposures
from climada.util.constants import DEF_CRS
from climada_petals.engine.supplychain.core import SIMULATION_LENGTH_BUFFER, BoARIOModel, DirectShocksSet, StaticIOModel
from climada_petals.engine.supplychain.utils import _thin_to_wide
from pymrio import IOSystem
from scipy import sparse


class TestDirectShock_inits(TestCase):
    def setUp(self):
        self.mriot_data = {
            "name": "Test_MRIOT",
            "sectors": ["A", "B", "C"],
            "regions": ["X", "Y"],
            "monetary_factor": 1_000,
        }

        lat = np.array([1, 3])
        lon = np.array([1.5, 3.5])
        exp = Exposures(crs=DEF_CRS, lat=lat, lon=lon, value=np.array([150.0, 80.0]))
        #exp.gdf = MagicMock() # Mock the gdf attribute
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
            imp_mat=sparse.csr_matrix(np.array([[30, 25], [30, 5]])),
        )

        mriot_mock_1 = MagicMock()
        mriot_mock_1.name = "Test_MRIOT"
        mriot_mock_1.get_sectors.return_value = ["Agriculture", "Industry", "Service"]
        mriot_mock_1.get_regions.return_value = ["USA", "ROW", "FRA"]
        mriot_mock_1.monetary_factor = 1_000

        self.exp = exp
        self.imp = imp
        self.mriot_dummy_exp = mriot_mock_1

        self.mock_mriot = MagicMock()
        self.mock_mriot.name = "Test_MRIOT"
        self.mock_mriot.get_sectors.return_value = ["A", "B", "C"]
        self.mock_mriot.get_regions.return_value = ["X", "Y", "Z"]
        self.mock_mriot.monetary_factor = 1_000

        self.mock_empty_mriot = MagicMock()
        self.mock_empty_mriot.name = "Empty_MRIOT"
        self.mock_empty_mriot.get_sectors.return_value = []
        self.mock_empty_mriot.get_regions.return_value = []
        self.mock_empty_mriot.monetary_factor = 1_000

        self.mock_mriot_missing_attrs = MagicMock()
        self.mock_mriot_missing_attrs.name = "Incomplete_MRIOT"
        self.mock_mriot_missing_attrs.get_sectors.return_value = ["A", "B"]
        # In this case, we simulate the missing attributes
        del self.mock_mriot_missing_attrs.get_regions
        del self.mock_mriot_missing_attrs.monetary_factor

        index = pd.MultiIndex.from_tuples(
            [("X", "A"), ("X", "B"), ("Y", "C")], names=["region", "sector"]
        )
        self.valid_exposure_assets = pd.Series([100, 200, 300], index=index, name="value")

        index_imp = ["event_1", "event_2"]
        columns = pd.MultiIndex.from_tuples(
            [("X", "A"), ("X", "B"), ("Y", "C")], names=["region", "sector"]
        )
        data = [[10, 20, 30], [40, 50, 60]]
        self.valid_impacted_assets = pd.DataFrame(data, index=index_imp, columns=columns)

        self.event_dates = np.array([737175, 737220])

        sample_exposure_assets = pd.Series(
            [100, 200],
            index=pd.MultiIndex.from_tuples(
                [("X", "A"), ("Y", "B")], names=["region", "sector"]
            ),
        )
        sample_impacted_assets = pd.DataFrame(
            [[10, 20]],
            index=["event_1"],
            columns=pd.MultiIndex.from_tuples(
                [("X", "A"), ("Y", "B")], names=["region", "sector"]
            ),
        )
        sample_event_dates = np.array([3])
        self.sample_exposure_assets = sample_exposure_assets
        self.sample_impacted_assets = sample_impacted_assets
        self.sample_event_dates = sample_event_dates

        self.mismatch_exposure_assets = pd.Series(
            [100, 200],
            index=pd.MultiIndex.from_tuples(
                [("X", "A"), ("Y", "B")], names=["region", "sector"]
            ),
        )
        self.mismatch_impacted_assets = pd.DataFrame(
            [[10, 20]],
            index=["event_1", "event_2"],
            columns=pd.MultiIndex.from_tuples(
                [("X", "A"), ("Y", "B")], names=["region", "sector"]
            ),
        )
        self.mismatch_event_dates = np.array([3])

        self.edge_case_exposure_assets = pd.Series(
            [100, 0, np.nan],
            index=pd.MultiIndex.from_tuples(
                [("X", "A"), ("Y", "B"), ("Z", "C")], names=["region", "sector"]
            ),
        )
        self.edge_case_impacted_assets = pd.DataFrame(
            [[10, 20, np.nan]],
            index=["event_1"],
            columns=pd.MultiIndex.from_tuples(
                [("X", "A"), ("Y", "B"), ("Z", "C")], names=["region", "sector"]
            ),
        )
        self.edge_case_event_dates = np.array([3])

        self.mismatched_index_exposure_assets = pd.Series(
            [100, 200],
            index=pd.MultiIndex.from_tuples(
                [("X", "A"), ("Y", "B")], names=["region", "sector"]
            ),
        )
        self.mismatched_index_impacted_assets = pd.DataFrame(
            [[10, 20, 30]],
            index=["event_1"],
            columns=pd.MultiIndex.from_tuples(
                [("X", "A"), ("Y", "B"), ("Z", "C")], names=["region", "sector"]
            ),
        )
        self.mismatched_index_event_dates = np.array([3])

    def test_directshock__init__(self):
        # Initialize the DirectShocksSetinstance
        shock = DirectShocksSet(
            mriot_name=self.mriot_data["name"],
            mriot_sectors=self.mriot_data["sectors"],
            mriot_regions=self.mriot_data["regions"],
            exposure_assets=self.valid_exposure_assets,
            impacted_assets=self.valid_impacted_assets,
            event_dates=self.event_dates,
            monetary_factor=self.mriot_data["monetary_factor"],
            shock_name="Test_Shock",
        )

        # Assertions
        assert shock.mriot_name == self.mriot_data["name"]
        assert shock.mriot_sectors == self.mriot_data["sectors"]
        assert shock.mriot_regions == self.mriot_data["regions"]
        assert shock.name == "Test_Shock"
        assert shock.monetary_factor == self.mriot_data["monetary_factor"]

        # Ensure _thin_to_wide transformation is applied correctly

        pd.testing.assert_series_equal(
            shock.exposure_assets,
            _thin_to_wide(self.valid_exposure_assets, shock.mriot_industries),
        )
        pd.testing.assert_frame_equal(
            shock.impacted_assets,
            _thin_to_wide(
                self.valid_impacted_assets, self.valid_impacted_assets.index, shock.mriot_industries
            ),
        )
        # Check event dates
        assert np.array_equal(shock.event_dates, self.event_dates)


    def test_directshock__init__missmatched(self):
        exposure_assets, impacted_assets, event_dates = self.mismatched_index_exposure_assets, self.mismatched_index_impacted_assets, self.mismatched_index_event_dates
        with pytest.warns(
            UserWarning,
            match="Some impacted assets do not have a corresponding exposure value",
        ):
            DirectShocksSet._init_with_mriot(
                mriot=self.mock_mriot,
                exposure_assets=exposure_assets,
                impacted_assets=impacted_assets,
                event_dates=event_dates,
                shock_name="Test_Shock",
            )


    def test_directshock__init___invalid_inputs(self):
        # Invalid exposure_assets (not a Series)
        with pytest.raises(ValueError, match="Exposure assets must be a pandas Series"):
            DirectShocksSet(
                mriot_name=self.mriot_data["name"],
                mriot_sectors=self.mriot_data["sectors"],
                mriot_regions=self.mriot_data["regions"],
                exposure_assets={"X": 100},  # Invalid type
                impacted_assets=pd.DataFrame(),
                event_dates=self.event_dates,
                monetary_factor=self.mriot_data["monetary_factor"],
            )

        # Invalid impacted_assets (not a DataFrame)
        with pytest.raises(ValueError, match="Impacted assets must be a pandas DataFrame"):
            DirectShocksSet(
                mriot_name=self.mriot_data["name"],
                mriot_sectors=self.mriot_data["sectors"],
                mriot_regions=self.mriot_data["regions"],
                exposure_assets=pd.Series(),
                impacted_assets={"event_1": 10},  # Invalid type
                event_dates=self.event_dates,
                monetary_factor=self.mriot_data["monetary_factor"],
            )

        # Mismatched indices
        invalid_exposure_assets = pd.Series([100], index=[("Z", "A")], name="value")
        with pytest.raises(
            ValueError, match="Exposure assets indices do not match MRIOT industries"
        ):
            DirectShocksSet(
                mriot_name=self.mriot_data["name"],
                mriot_sectors=self.mriot_data["sectors"],
                mriot_regions=self.mriot_data["regions"],
                exposure_assets=invalid_exposure_assets,
                impacted_assets=pd.DataFrame(),
                event_dates=self.event_dates,
                monetary_factor=self.mriot_data["monetary_factor"],
            )

        # Create a DataFrame with rows as event IDs and columns as (region, sector)
        index = ["event_1", "event_2"]
        columns = pd.MultiIndex.from_tuples(
            [("X", "A"), ("X", "B"), ("Z", "C")], names=["region", "sector"]
        )
        data = [[10, 20, 30], [40, 50, 60]]
        invalid_impacted_assets = pd.DataFrame(data, index=index, columns=columns)
        with pytest.raises(
            ValueError, match="Impacted assets columns do not match MRIOT industries"
        ):
            DirectShocksSet(
                mriot_name=self.mriot_data["name"],
                mriot_sectors=self.mriot_data["sectors"],
                mriot_regions=self.mriot_data["regions"],
                exposure_assets=pd.Series(),
                impacted_assets=invalid_impacted_assets,
                event_dates=self.event_dates,
                monetary_factor=self.mriot_data["monetary_factor"],
            )

        exposure_assets, impacted_assets, event_dates = self.mismatch_exposure_assets, self.mismatch_impacted_assets, self.mismatch_event_dates
        with pytest.raises(ValueError, match="Number of events mismatch"):
            DirectShocksSet(
                mriot_name=self.mriot_data["name"],
                mriot_sectors=self.mriot_data["sectors"],
                mriot_regions=self.mriot_data["regions"],
                exposure_assets=exposure_assets,
                impacted_assets=impacted_assets,
                event_dates=event_dates,
                monetary_factor=self.mriot_data["monetary_factor"],
            )


    def test_init_with_mriot_valid(self):
        exposure_assets, impacted_assets, event_dates = self.sample_exposure_assets, self.sample_impacted_assets, self.sample_event_dates

        # Call the _init_with_mriot method
        shock = DirectShocksSet._init_with_mriot(
            mriot=self.mock_mriot,
            exposure_assets=exposure_assets,
            impacted_assets=impacted_assets,
            event_dates=event_dates,
            shock_name="Test_Shock",
        )

        # Assertions
        assert shock.mriot_name == self.mock_mriot.name
        assert shock.mriot_sectors == self.mock_mriot.get_sectors()
        assert shock.mriot_regions == self.mock_mriot.get_regions()
        assert shock.monetary_factor == self.mock_mriot.monetary_factor
        assert shock.name == "Test_Shock"
        pd.testing.assert_series_equal(
            shock.exposure_assets, _thin_to_wide(exposure_assets, shock.mriot_industries)
        )
        pd.testing.assert_frame_equal(
            shock.impacted_assets,
            _thin_to_wide(
                impacted_assets, shock.impacted_assets.index, shock.mriot_industries
            ),
        )
        np.testing.assert_array_equal(shock.event_dates, event_dates)


    def test_init_with_mriot_empty(self):
        exposure_assets, impacted_assets, event_dates = self.sample_exposure_assets, self.sample_impacted_assets, self.sample_event_dates

        with pytest.raises(ValueError):
            shock = DirectShocksSet._init_with_mriot(
                mriot=self.mock_empty_mriot,
                exposure_assets=exposure_assets,
                impacted_assets=impacted_assets,
                event_dates=event_dates,
                shock_name="Empty_Shock",
            )


    def test_init_with_mriot_missing_attributes(self):
        exposure_assets, impacted_assets, event_dates = self.sample_exposure_assets, self.sample_impacted_assets, self.sample_event_dates


        # Call the _init_with_mriot method and expect AttributeError
        with pytest.raises(AttributeError):
            DirectShocksSet._init_with_mriot(
                mriot=self.mock_mriot_missing_attrs,
                exposure_assets=exposure_assets,
                impacted_assets=impacted_assets,
                event_dates=event_dates,
                shock_name="Incomplete_Shock",
            )


    def test_relative_impact_correct_calculation(self):
        exposure_assets, impacted_assets, event_dates = self.sample_exposure_assets, self.sample_impacted_assets, self.sample_event_dates

        shock = DirectShocksSet._init_with_mriot(
            mriot=self.mock_mriot,
            exposure_assets=exposure_assets,
            impacted_assets=impacted_assets,
            event_dates=event_dates,
            shock_name="Test_Shock",
        )
        exposure_assets = _thin_to_wide(exposure_assets, shock.mriot_industries)
        impacted_assets = _thin_to_wide(
            impacted_assets, shock.impacted_assets.index, shock.mriot_industries
        )
        # Compute relative impact
        expected = impacted_assets.div(exposure_assets, axis=1).fillna(0.0) * (
            exposure_assets > 0
        )
        result = shock.relative_impact
        pd.testing.assert_frame_equal(result, expected, check_dtype=True)


    def test_relative_impact_edge_cases(self):
        exposure_assets, impacted_assets, event_dates = self.edge_case_exposure_assets, self.edge_case_impacted_assets, self.edge_case_event_dates
        shock = DirectShocksSet._init_with_mriot(
            mriot=self.mock_mriot,
            exposure_assets=exposure_assets,
            impacted_assets=impacted_assets,
            event_dates=event_dates,
            shock_name="Test_Shock",
        )
        exposure_assets = _thin_to_wide(exposure_assets, shock.mriot_industries)
        impacted_assets = _thin_to_wide(
            impacted_assets, shock.impacted_assets.index, shock.mriot_industries
        )

        # Expected: division handles 0 and NaN
        expected = impacted_assets.div(exposure_assets, axis=1).fillna(0.0).replace(
            [np.inf, -np.inf], 0
        ) * (exposure_assets > 0)
        result = shock.relative_impact
        pd.testing.assert_frame_equal(result, expected)


    def test_relative_impact_mismatched_indices(self):
        exposure_assets, impacted_assets, event_dates = self.mismatched_index_exposure_assets, self.mismatched_index_impacted_assets, self.mismatched_index_event_dates
        with pytest.warns(
            UserWarning,
            match="Some impacted assets do not have a corresponding exposure value",
        ):
            shock = DirectShocksSet._init_with_mriot(
                mriot=self.mock_mriot,
                exposure_assets=exposure_assets,
                impacted_assets=impacted_assets,
                event_dates=event_dates,
                shock_name="Test_Shock",
            )

        exposure_assets = _thin_to_wide(exposure_assets, shock.mriot_industries)
        impacted_assets = _thin_to_wide(
            impacted_assets, shock.impacted_assets.index, shock.mriot_industries
        )
        expected = (
            impacted_assets.div(exposure_assets, axis=1)
            .replace([np.inf, -np.inf], 0)
            .reindex_like(impacted_assets)
        )
        expected = expected.fillna(0.0)
        result = shock.relative_impact
        pd.testing.assert_frame_equal(result, expected)


    # We just need to test that the methods and functions are correctly called, as this is a unittest
    @patch("climada_petals.engine.supplychain.core.translate_exp_to_regions")
    @patch("climada_petals.engine.supplychain.core.translate_exp_to_sectors")
    @patch("climada_petals.engine.supplychain.core.DirectShocksSet.from_assets_and_imp")
    def test_from_exp_and_imp(
            self,
        mock_from_assets_and_imp,
        mock_translate_exp_to_sectors,
        mock_translate_exp_to_regions,
    ):
        # Mock inputs
        mock_mriot = MagicMock(spec=IOSystem)
        mock_mriot.name = "TestMRIOT"
        mock_exposure = MagicMock(spec=Exposures)
        mock_impact = MagicMock(spec=Impact)
        mock_impact.date = np.array([723000])
        shock_name = "TestShock"
        affected_sectors = ["sector1", "sector2"]
        impact_distribution = None
        exp_value_col = "value"

        # Mock function outputs
        mock_exp_translated = MagicMock(spec=pd.DataFrame)
        mock_translate_exp_to_regions.return_value = mock_exp_translated

        mock_exposure_assets = MagicMock(spec=pd.Series)
        mock_translate_exp_to_sectors.return_value = mock_exposure_assets

        # Call the method
        DirectShocksSet.from_exp_and_imp(
            mriot=mock_mriot,
            exposure=mock_exposure,
            impact=mock_impact,
            affected_sectors=affected_sectors,
            impact_distribution=impact_distribution,
            shock_name=shock_name,
            exp_value_col=exp_value_col,
        )

        # Assertions
        mock_translate_exp_to_regions.assert_called_once_with(
            mock_exposure, mriot_name="TestMRIOT", custom_mriot=False
        )
        mock_translate_exp_to_sectors.assert_called_once_with(
            mock_exp_translated,
            affected_sectors=affected_sectors,
            mriot=mock_mriot,
            value_col=exp_value_col,
        )
        mock_from_assets_and_imp.assert_called_once_with(
            mriot=mock_mriot,
            exposure_assets=mock_exposure_assets,
            impact=mock_impact,
            shock_name=shock_name,
            affected_sectors=affected_sectors,
            impact_distribution=impact_distribution,
            custom_mriot=False
        )


    @patch("climada_petals.engine.supplychain.core.DirectShocksSet._init_with_mriot")
    @patch("climada_petals.engine.supplychain.core.translate_imp_to_regions")
    @patch("climada_petals.engine.supplychain.core.distribute_reg_impact_to_sectors")
    def test_from_assets_and_imp(
        self,
        mock_distribute_reg_impact_to_sectors,
        mock_translate_imp_to_regions,
        mock_init_with_mriot,
    ):
        # This methods actually does some things that need to be checked
        # We don't want to check the functions called so we define what
        # they should return with mock objects and check the rest of the code

        # data for mock object returns
        sectors = ["sector1", "sector2"]
        regions = ["FRA", "JPN", "USA"]
        wrong_regions = ["FRX", "JPX", "USX"]
        impacted_regions = ["JPN", "USA"]
        production_vector = pd.DataFrame(
            [[4.0], [12.0], [20.0], [60.0], [10.0], [90.0]],
            index=pd.MultiIndex.from_product([regions, sectors]),
            columns=["indout"],
        )

        # Only for impacted
        production_distribution = pd.Series(
            {
                ("JPN", "sector1"): 0.25,
                ("JPN", "sector2"): 0.75,
                ("USA", "sector1"): 0.1,
                ("USA", "sector2"): 0.9,
            }
        )
        production_distribution.name = "indout"
        event_dates = np.array([723000, 50408])
        event_index = [249, 3869]
        shock_name = "TestShock"
        regional_impact = pd.DataFrame.from_dict(
            {
                "index": event_index,
                "columns": regions,
                "data": [[0.0, 100.0, 50.0], [0.0, 0.0, 10.0]],
                "index_names": [None],
                "column_names": [None],
            },
            orient="tight",
        )
        regional_impact_no_zeros = pd.DataFrame.from_dict(
            {
                "index": event_index,
                "columns": impacted_regions,
                "data": [[100.0, 50.0], [0.0, 10.0]],
                "index_names": [None],
                "column_names": [None],
            },
            orient="tight",
        )
        region_translated_imp = pd.DataFrame.from_dict(
            {
                "index": event_index,
                "columns": impacted_regions,
                "data": [[100, 50.0], [0.0, 10.0]],
                "index_names": ["event_id"],
                "column_names": ["region_mriot"],
            },
            orient="tight",
        )
        sector_distributed_impact_case1 = pd.DataFrame.from_dict(
            {
                "index": event_index,
                "columns": [
                    ("JPN", "sector1"),
                    ("JPN", "sector2"),
                    ("USA", "sector1"),
                    ("USA", "sector2"),
                ],
                "data": [[250.0, 750.0, 50.0, 450.00000000000006], [0.0, 0.0, 10.0, 90.0]],
                "index_names": ["event_id"],
                "column_names": ["region", "sector"],
            },
            orient="tight",
        )
        impact_distribution_case2 = pd.Series(
            {
                ("JPN", "sector1"): 20.0,
                ("USA", "sector1"): 10.0,
            }
        )
        impact_distribution_case2.name = "indout"

        impact_distribution_case3_ext = {
            "sector1": 0.2,
            "sector2": 0.8,
        }
        impact_distribution_case3_intern = pd.Series(
            {
                "sector1": 0.2,
                "sector2": 0.8,
            }
        )

        impact_distribution_case4 = pd.Series(
            {
                ("JPN", "sector1"): 0.2,
                ("JPN", "sector2"): 0.8,
                ("USA", "sector1"): 0.1,
                ("USA", "sector2"): 0.9,
            }
        )
        impact_distribution_case5 = "wrong"

        # Mock objects and setting their return when needed
        mock_mriot = MagicMock(spec=IOSystem)
        mock_mriot.name = "TestMRIOT"
        mock_mriot.x = production_vector
        mock_mriot.get_sectors.return_value = sectors
        mock_impact = MagicMock(spec=Impact)
        mock_impact.date = event_dates
        mock_exposure_assets = MagicMock(spec=pd.Series)
        mock_impact.impact_at_reg.return_value = regional_impact
        mock_translate_imp_to_regions.return_value = region_translated_imp
        mock_distribute_reg_impact_to_sectors.return_value = sector_distributed_impact_case1

        # Calling the method
        DirectShocksSet.from_assets_and_imp(
            mock_mriot,
            mock_exposure_assets,
            mock_impact,
            shock_name,
            affected_sectors="all",
            impact_distribution=None,
        )

        # Check that impact_at_reg was called
        mock_impact.impact_at_reg.assert_called()

        # We can't used assert_called_with() with dataframe due to
        # dataframe == dataframe raising an error, thus we compare the argument
        # directly

        # we check that translate_imp_to_regions was called with
        # the impact_at_reg() filtered to non zero values and the mriot
        pd.testing.assert_frame_equal(
            mock_translate_imp_to_regions.call_args[0][0], regional_impact_no_zeros
        )
        assert mock_translate_imp_to_regions.call_args[0][1] == mock_mriot

        # We then check that distribute_reg_impact_to_sectors was called
        # with the return from translate_imp_to_regions and
        # with the correct impact distribution (based on production due to None).
        pd.testing.assert_frame_equal(
            mock_distribute_reg_impact_to_sectors.call_args[0][0], region_translated_imp
        )
        pd.testing.assert_series_equal(
            mock_distribute_reg_impact_to_sectors.call_args[0][1], production_vector.loc[impacted_regions,"indout"]
        )

        # Finally we check that _init_with_mriot was called
        # with the return from distribute_reg_impact_to_sectors
        # and the correct arguments.
        mock_init_with_mriot.assert_called_once_with(
            mriot=mock_mriot,
            exposure_assets=mock_exposure_assets,
            impacted_assets=sector_distributed_impact_case1,
            event_dates=mock_impact.date,
            shock_name=shock_name,
        )

        # Case 2
        # Now test selection of sectors
        mock_distribute_reg_impact_to_sectors.reset_mock()
        DirectShocksSet.from_assets_and_imp(
            mock_mriot,
            mock_exposure_assets,
            mock_impact,
            shock_name,
            affected_sectors="sector1",
            impact_distribution=None,
        )
        pd.testing.assert_series_equal(
            mock_distribute_reg_impact_to_sectors.call_args[0][1], impact_distribution_case2
        )

        # Case 3
        # Now test with dict
        mock_distribute_reg_impact_to_sectors.reset_mock()
        DirectShocksSet.from_assets_and_imp(
            mock_mriot,
            mock_exposure_assets,
            mock_impact,
            shock_name,
            affected_sectors="all",
            impact_distribution=impact_distribution_case3_ext,
        )
        pd.testing.assert_series_equal(
            mock_distribute_reg_impact_to_sectors.call_args[0][1],
            impact_distribution_case3_intern,
        )

        # Case 4
        # Now test with series
        mock_distribute_reg_impact_to_sectors.reset_mock()
        DirectShocksSet.from_assets_and_imp(
            mock_mriot,
            mock_exposure_assets,
            mock_impact,
            shock_name,
            affected_sectors="all",
            impact_distribution=impact_distribution_case4,
        )
        pd.testing.assert_series_equal(
            mock_distribute_reg_impact_to_sectors.call_args[0][1], impact_distribution_case4
        )

        # Case 5
        # Now test with wrong type
        with pytest.raises(
            ValueError, match="Impact_distribution could not be converted to a Series"
        ):
            DirectShocksSet.from_assets_and_imp(
                mock_mriot,
                mock_exposure_assets,
                mock_impact,
                shock_name,
                affected_sectors="all",
                impact_distribution=impact_distribution_case5,
            )

        # Case 6
        # Now test with incompatible mriot / affected
        mock_mriot = MagicMock(spec=IOSystem)
        mock_mriot.x = pd.DataFrame(
            [[4.0], [6.0], [20.0], [60.0], [10.0], [90.0]],
            index=pd.MultiIndex.from_product([wrong_regions, sectors]),
            columns=["indout"],
        )
        with pytest.raises(KeyError):
            DirectShocksSet.from_assets_and_imp(
                mock_mriot,
                mock_exposure_assets,
                mock_impact,
                shock_name,
                affected_sectors="all",
                impact_distribution=None,
            )


    def test_combine_event_dates_valid(self):
        event_dates = [
            pd.Series({"event1": 10, "event2": 20}),
            pd.Series({"event2": 20, "event3": 30}),
        ]
        expected = pd.Series({"event1": 10, "event2": 20, "event3": 30})
        result = DirectShocksSet._combine_event_dates(event_dates)
        pd.testing.assert_series_equal(result, expected)


    def test_combine_event_dates_conflicting(self):
        event_dates = [
            pd.Series({"event1": 10, "event2": 20}),
            pd.Series({"event2": 30, "event3": 40}),
        ]
        with pytest.raises(ValueError, match="Conflicting values"):
            DirectShocksSet._combine_event_dates(event_dates)


    def test_merge_imp_assets_empty_list(self):
        """Test merging an empty list of DataFrames."""
        with pytest.raises(ValueError):
            DirectShocksSet._merge_imp_assets([])


    def test_merge_imp_assets_single_dataframe(self):
        """Test merging a single DataFrame should return the same DataFrame."""
        df = pd.DataFrame({"A": [1, 2]}, index=["Industry1", "Industry2"], dtype=float)
        result = DirectShocksSet._merge_imp_assets([df])
        pd.testing.assert_frame_equal(result, df)


    def test_merge_imp_assets_identical_dataframes(self):
        """Test merging two identical DataFrames should result in doubled values."""
        df1 = pd.DataFrame({"A": [1, 2]}, index=["Industry1", "Industry2"], dtype=float)
        df2 = df1.copy()
        expected = pd.DataFrame(
            {"A": [2, 4]}, index=["Industry1", "Industry2"], dtype=float
        )
        result = DirectShocksSet._merge_imp_assets([df1, df2])
        pd.testing.assert_frame_equal(result, expected)


    def test_merge_imp_assets_different_indexes(self):
        """Test merging DataFrames with different industry indexes should result in NaN filled with 0s."""
        df1 = pd.DataFrame({"A": [1, 2]}, index=["Industry1", "Industry2"], dtype=float)
        df2 = pd.DataFrame({"A": [3, 4]}, index=["Industry3", "Industry4"], dtype=float)
        expected = pd.DataFrame(
            {"A": [1, 2, 3, 4]},
            index=["Industry1", "Industry2", "Industry3", "Industry4"],
            dtype=float,
        )
        result = DirectShocksSet._merge_imp_assets([df1, df2])
        pd.testing.assert_frame_equal(result, expected)


    def test_merge_imp_assets_multiple_columns(self):
        """Test merging DataFrames with multiple columns should sum matching columns only."""
        df1 = pd.DataFrame(
            {"A": [1, 2], "B": [3, 4]}, index=["Industry1", "Industry2"], dtype=float
        )
        df2 = pd.DataFrame(
            {"A": [5, 6], "C": [7, 8]}, index=["Industry1", "Industry2"], dtype=float
        )
        expected = pd.DataFrame(
            {
                "A": [6, 8],  # A columns sum up
                "B": [3, 4],  # B only appears in df1
                "C": [7, 8],  # C only appears in df2
            },
            index=["Industry1", "Industry2"],
            dtype=float,
        )
        result = DirectShocksSet._merge_imp_assets([df1, df2])
        pd.testing.assert_frame_equal(result, expected)


    def test_merge_imp_assets_with_missing_values(self):
        """Test merging DataFrames with missing values should treat NaNs as 0."""
        df1 = pd.DataFrame(
            {"A": [1, np.nan]}, index=["Industry1", "Industry2"], dtype=float
        )
        df2 = pd.DataFrame(
            {"A": [np.nan, 4]}, index=["Industry1", "Industry2"], dtype=float
        )
        expected = pd.DataFrame(
            {"A": [1, 4]}, index=["Industry1", "Industry2"], dtype=float
        )
        result = DirectShocksSet._merge_imp_assets([df1, df2])
        pd.testing.assert_frame_equal(result, expected)


class TestDirectShockCombine(TestCase):
    def setUp(self):
        # Create mock DirectShocksSetinstances
        self.sectors = ["sector1", "sector2", "sector3"]
        self.regions = ["region1", "region2", "region3"]
        self.industries = pd.MultiIndex.from_product(
            [self.regions, self.sectors],
            names=["region", "sector"],
        )
        self.industries_mismatch = pd.MultiIndex.from_product(
            [["regionA", "regionB", "regionC"], self.sectors],
            names=["region", "sector"],
        )

        self.exp_assets_1 = pd.Series(
            [100.0, 200.0, 400.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=self.industries
        )
        self.exp_assets_2 = pd.Series(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0], index=self.industries
        )
        self.exp_assets_1_and_2 = pd.Series(
            [100.0, 200.0, 400.0, 300.0, 0.0, 0.0, 0.0, 200.0, 0.0],
            index=self.industries,
        )

        self.exp_assets_1_diff_value = pd.Series(
            [1_500.0, 200.0, 400.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            index=self.industries,
        )
        self.exp_assets_1_ind_mismatch = pd.Series(
            [1_500.0, 200.0, 400.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            index=self.industries_mismatch,
        )

        self.event_dates = pd.Series({"event1": 10, "event2": 20})
        self.event_ids = self.event_dates.index.values
        self.imp_assets_1 = pd.DataFrame(
            [
                [10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [30.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            index=self.event_ids,
            columns=self.industries,
        )
        self.imp_assets_2 = pd.DataFrame(
            [
                [50.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [70.0, 0.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            index=self.event_ids,
            columns=self.industries,
        )

        # Mock attributes
        self.shock1 = MagicMock(spec=DirectShocksSet)
        self.shock2 = MagicMock(spec=DirectShocksSet)
        self.shock1.mriot_name = "TestMRIOT"
        self.shock2.mriot_name = "TestMRIOT"
        self.shock1.mriot_sectors = self.sectors
        self.shock2.mriot_sectors = self.sectors
        self.shock1.mriot_regions = self.regions
        self.shock2.mriot_regions = self.regions
        self.shock1.exposure_assets = self.exp_assets_1
        self.shock2.exposure_assets = self.exp_assets_1
        self.shock1.impacted_assets = self.imp_assets_1
        self.shock2.impacted_assets = self.imp_assets_2
        self.shock1.event_dates = self.event_dates
        self.shock2.event_dates = self.event_dates
        self.shock1.monetary_factor = 1.0
        self.shock2.monetary_factor = 1.0

    def test_combine_exp_assets_valid(self):
        expected = self.exp_assets_1_and_2
        result = DirectShocksSet._combine_exp_assets(
            [self.exp_assets_1, self.exp_assets_2]
        )
        pd.testing.assert_series_equal(result, expected)

    def test_combine_exp_assets_conflicting_values(self):
        with pytest.raises(ValueError, match="Conflicting values"):
            DirectShocksSet._combine_exp_assets(
                [self.exp_assets_1, self.exp_assets_1_diff_value]
            )

    def test_combine_exp_assets_mismatched_index(self):
        with pytest.raises(ValueError, match="Mismatching indexes"):
            DirectShocksSet._combine_exp_assets(
                [self.exp_assets_1, self.exp_assets_1_ind_mismatch]
            )

    @patch("climada_petals.engine.supplychain.core.DirectShocksSet._merge_imp_assets")
    @patch(
        "climada_petals.engine.supplychain.core.DirectShocksSet._combine_event_dates"
    )
    @patch("climada_petals.engine.supplychain.core.DirectShocksSet._combine_exp_assets")
    def test_combine_merge(
        self, mock_combine_exp_assets, mock_combine_event_dates, mock_merge_imp_assets
    ):
        # Mock _combine_exp_assets return value
        mock_combine_exp_assets.return_value = pd.Series(
            [100.0, 200.0, 400.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=self.industries
        )
        mock_combine_event_dates.return_value = pd.Series({"event1": 10, "event2": 20})
        mock_merge_imp_assets.return_value = pd.DataFrame(
            [
                [10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [30.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            index=["event1", "event2"],
            columns=self.industries,
        )
        # Call combine with kind="merge"
        combined_shock = DirectShocksSet.combine(
            [self.shock1, self.shock2], kind="merge"
        )
        # Assertions
        mock_combine_exp_assets.assert_called_once_with(
            [self.exp_assets_1, self.exp_assets_1]
        )
        self.assertEqual(combined_shock.mriot_name, "TestMRIOT")
        pd.testing.assert_series_equal(
            combined_shock.exposure_assets, mock_combine_exp_assets.return_value
        )
        pd.testing.assert_frame_equal(
            combined_shock.impacted_assets, mock_merge_imp_assets.return_value
        )

    @patch("climada_petals.engine.supplychain.core.DirectShocksSet._combine_exp_assets")
    def test_combine_concat(self, mock_combine_exp_assets):
        # Mock _combine_exp_assets return value
        mock_combine_exp_assets.return_value = pd.Series(
            [100.0, 200.0, 400.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            index=self.industries,
        )
        self.shock1.event_dates = pd.Series({"event1": 10, "event2": 20})
        self.shock2.event_dates = pd.Series({"event3": 10, "event4": 20})
        self.shock1.impacted_assets = pd.DataFrame(
            [
                [10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [30.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            index=["event1", "event2"],
            columns=self.industries,
        )
        self.shock2.impacted_assets = pd.DataFrame(
            [
                [50.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [70.0, 0.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            index=["event3", "event4"],
            columns=self.industries,
        )
        # Call combine with kind="concat"
        combined_shock = DirectShocksSet.combine(
            [self.shock1, self.shock2], kind="concat"
        )

        # Assertions
        mock_combine_exp_assets.assert_called_once_with(
            [self.exp_assets_1, self.exp_assets_1]
        )

        self.assertEqual(combined_shock.mriot_name, "TestMRIOT")
        expected_concat_assets = pd.DataFrame(
            [
                [10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [30.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [50.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [70.0, 0.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            index=["event1", "event2", "event3", "event4"],
            columns=self.industries,
        ).fillna(0)
        pd.testing.assert_frame_equal(
            combined_shock.impacted_assets, expected_concat_assets
        )

    def test_combine_different_mriot_names(self):
        # Test edge case with mismatched mriot names
        self.shock2.mriot_name = "AnotherMRIOT"

        with self.assertRaises(ValueError):
            DirectShocksSet.combine([self.shock1, self.shock2])


class TestStaticIOModel(TestCase):
    def test_init_bare(self):
        mriot = MagicMock()
        model = StaticIOModel(mriot)
        self.assertIs(model.mriot, mriot)
        self.assertEqual(model.direct_shocks, None)

    @patch("climada_petals.engine.supplychain.core.IndirectCostModel.shock_model_with")
    def test_init_shock(self, mock_shock_model_with):
        mriot = MagicMock()
        direct_shocks = MagicMock()
        model = StaticIOModel(mriot, direct_shocks)
        mock_shock_model_with.assert_called_once()

        self.assertIs(model.mriot, mriot)

    def test_shock_model_with_single_wrong_name(self):
        mriot = IOSystem()
        mriot.meta.change_meta("name", "test")
        direct_shocks = MagicMock(mriot_name="diffname")
        model = StaticIOModel(mriot)
        with pytest.raises(ValueError):
            model.shock_model_with(direct_shocks)

    def test_shock_model_with_single(self):
        mriot = IOSystem()
        mriot.meta.change_meta("name", "test")
        direct_shocks = MagicMock(spec=DirectShocksSet, mriot_name="test")
        model = StaticIOModel(mriot)
        model.shock_model_with(direct_shocks)
        self.assertEqual(model.direct_shocks, direct_shocks)

    @patch("climada_petals.engine.supplychain.core.DirectShocksSet.combine")
    def test_shock_model_with_list(self, mock_combine):
        mriot = IOSystem()
        mriot.meta.change_meta("name", "test")
        direct_shocks = MagicMock(spec=DirectShocksSet, mriot_name="test")
        direct_shocks2 = MagicMock(spec=DirectShocksSet, mriot_name="test")
        model = StaticIOModel(mriot)

        mock_combine.return_value = MagicMock(
            name="return", spec=DirectShocksSet, mriot_name="test"
        )

        model.shock_model_with([direct_shocks, direct_shocks2])
        mock_combine.assert_called_once_with([direct_shocks, direct_shocks2], "concat")
        self.assertEqual(model.direct_shocks, mock_combine.return_value)

    @patch("climada_petals.engine.supplychain.core.DirectShocksSet.combine")
    def test_shock_model_with_list_merge(self, mock_combine):
        mriot = IOSystem()
        mriot.meta.change_meta("name", "test")
        direct_shocks = MagicMock(spec=DirectShocksSet, mriot_name="test")
        direct_shocks2 = MagicMock(spec=DirectShocksSet, mriot_name="test")
        model = StaticIOModel(mriot)

        mock_combine.return_value = MagicMock(
            name="return", spec=DirectShocksSet, mriot_name="test"
        )

        model.shock_model_with([direct_shocks, direct_shocks2], combine_mode="merge")
        mock_combine.assert_called_once_with([direct_shocks, direct_shocks2], "merge")
        self.assertEqual(model.direct_shocks, mock_combine.return_value)

    def test_calc_indirect_impacts_none(self):
        mriot = MagicMock()
        model = StaticIOModel(mriot)
        ret = model.calc_indirect_impacts()
        assert ret is None

    @patch("climada_petals.engine.supplychain.core.StaticIOModel.calc_ghosh")
    @patch("climada_petals.engine.supplychain.core.StaticIOModel.calc_leontief")
    def test_calc_indirect_impacts(self, mock_leontief, mock_ghosh):
        mriot = MagicMock(
            x=pd.DataFrame(
                [[100], [100]],
                columns=["indout"],
                index=pd.MultiIndex.from_tuples(
                    [("regA", "sec1"), ("regB", "sec1")], names=["region", "sector"]
                ),
            )
        )
        mriot.name = "test"
        direct_shocks = MagicMock(
            spec=DirectShocksSet,
            mriot_name="test",
            impacted_assets=pd.DataFrame(
                [[1000, 1000], [2000, 2000]],
                index=pd.Index([1, 2], name="event_id"),
                columns=pd.MultiIndex.from_tuples(
                    [("regA", "sec1"), ("regB", "sec1")], names=["region", "sector"]
                ),
            ),
        )

        model = StaticIOModel(mriot, direct_shocks=direct_shocks)

        mock_leontief.return_value = pd.DataFrame(
            [[10, 20], [30, 40]],
            index=pd.Index([1, 2], name="event_id"),
            columns=pd.MultiIndex.from_tuples(
                [("regA", "sec1"), ("regB", "sec1")], names=["region", "sector"]
            ),
        )
        mock_ghosh.return_value = pd.DataFrame(
            [[50, 50], [50, 50]],
            index=pd.Index([1, 2], name="event_id"),
            columns=pd.MultiIndex.from_tuples(
                [("regA", "sec1"), ("regB", "sec1")], names=["region", "sector"]
            ),
        )

        expected = pd.DataFrame(
            data=[
                [5.00e01],
                [5.00e-02],
                [2.50e-02],
                [5.00e-01],
                [1.00e01],
                [1.00e-02],
                [5.00e-03],
                [1.00e-01],
                [5.00e01],
                [5.00e-02],
                [2.50e-02],
                [5.00e-01],
                [2.00e01],
                [2.00e-02],
                [1.00e-02],
                [2.00e-01],
                [5.00e01],
                [2.50e-02],
                [1.25e-02],
                [5.00e-01],
                [3.00e01],
                [1.50e-02],
                [7.50e-03],
                [3.00e-01],
                [5.00e01],
                [2.50e-02],
                [1.25e-02],
                [5.00e-01],
                [4.00e01],
                [2.00e-02],
                [1.00e-02],
                [4.00e-01],
            ],
            columns=["value"],
            index=pd.MultiIndex.from_product(
                [
                    [1, 2],
                    ["regA", "regB"],
                    ["sec1"],
                    ["ghosh", "leontief"],
                    [
                        "absolute production change",
                        "production lost to sector shock ratio",
                        "production lost to total shock ratio",
                        "relative production change",
                    ],
                ],
                names=["event_id", "region", "sector", "method", "metric"],
            ),
        ).reset_index()
        ret = model.calc_indirect_impacts(event_ids=None)
        pd.testing.assert_frame_equal(ret, expected)


@pytest.fixture()
def mock_mriot2():
    """Fixture for a mock MRIOT object."""
    mriot = MagicMock()
    mriot.name = "Test_MRIOT"
    mriot.x = pd.DataFrame([1,2,3,4],
        index =pd.MultiIndex.from_tuples(
            [("regA", "sec1"), ("regA", "sec2"), ("regB", "sec1"), ("regB", "sec2")]
        ),
                           columns= ["indout"]
 )
    mriot.L = pd.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        index=pd.MultiIndex.from_tuples(
            [("regA", "sec1"), ("regA", "sec2"), ("regB", "sec1"), ("regB", "sec2")]
        ),
        columns=pd.MultiIndex.from_tuples(
            [("regA", "sec1"), ("regA", "sec2"), ("regB", "sec1"), ("regB", "sec2")]
        ),
    )  # Mock Leontief matrix
    mriot.G = pd.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        index=pd.MultiIndex.from_tuples(
            [("regA", "sec1"), ("regA", "sec2"), ("regB", "sec1"), ("regB", "sec2")]
        ),
        columns=pd.MultiIndex.from_tuples(
            [("regA", "sec1"), ("regA", "sec2"), ("regB", "sec1"), ("regB", "sec2")]
        ),
    )  # Mock Ghosh matrix
    mriot.get_sectors.return_value = ["sec1", "sec2"]
    mriot.get_regions.return_value = ["regA", "regB"]
    return mriot


@pytest.fixture()
def mock_shock():
    """Fixture for a mock DirectShocksSet with impacts."""
    shock = MagicMock()
    shock.mriot_name = "Test_MRIOT"
    shock.event_ids = pd.Index([1, 2])
    shock.degraded_final_demand = pd.DataFrame(
        [[100, 200, 300, 400], [100, 200, 300, 400]],
        index=[1, 2],
        columns=pd.MultiIndex.from_tuples(
            [("regA", "sec1"), ("regA", "sec2"), ("regB", "sec1"), ("regB", "sec2")]
        ),
    )
    return shock


@pytest.fixture()
def model(mock_mriot2, mock_shock):
    """Fixture for a StaticIOModel instance with a mock MRIOT and shocks."""
    return StaticIOModel(mock_mriot2, direct_shocks=mock_shock)


def test_calc_leontief_no_shocks(mock_mriot2):
    """Test calc_leontief when no shocks are defined."""
    model = StaticIOModel(mock_mriot2)
    result = model.calc_leontief()
    pd.testing.assert_frame_equal(result, pd.DataFrame())  # Ensure DataFrame is empty


def test_calc_leontief_with_shocks(model):
    """Test calc_leontief with defined direct shocks."""
    # We don't care about the real computation calc_x_from_L, we just want to test the rest of the method.
    expected_output = pd.DataFrame([1, 1], index=[1, 2])
    # Mocking pymrio.calc_x_from_L to return a DataFrame with expected outputs
    with patch("pymrio.calc_x_from_L") as mock_calc:
        mock_calc.return_value = pd.DataFrame([1], columns=["indout"])
        result = model.calc_leontief()
        # Verify that the return matches the expected output
        pd.testing.assert_frame_equal(result, expected_output)


def test_calc_leontief_with_shocks_event_ids(model):
    """Test calc_leontief with defined direct shocks."""
    # We don't care about the real computation calc_x_from_L, we just want to test the rest of the method.
    expected_output = pd.DataFrame([1], index=[2])
    # Mocking pymrio.calc_x_from_L to return a DataFrame with expected outputs
    with patch("pymrio.calc_x_from_L") as mock_calc:
        mock_calc.return_value = pd.DataFrame([1], columns=["indout"])
        result = model.calc_leontief(event_ids=[2])
        # Verify that the return matches the expected output
        pd.testing.assert_frame_equal(result, expected_output)


def test_calc_ghosh_no_shocks(mock_mriot2):
    """Test calc_ghosh when no shocks are defined."""
    model = StaticIOModel(mock_mriot2)
    result = model.calc_ghosh()
    pd.testing.assert_frame_equal(result, pd.DataFrame())  # Ensure DataFrame is empty


def test_calc_ghosh_with_shocks(model):
    """Test calc_ghosh with defined direct shocks."""
    # We don't care about the real computation calc_x_from_L, we just want to test the rest of the method.
    expected_output = pd.DataFrame.from_dict(
        {
            "index": [1, 2],
            "columns": [
                ("regA", "sec1"),
                ("regA", "sec2"),
                ("regB", "sec1"),
                ("regB", "sec2"),
            ],
            "data": [[1000, 2000, 3000, 4000], [1000, 2000, 3000, 4000]],
            "index_names": [None],
            "column_names": [None, None],
        },
        orient="tight",
    )
    # Mocking pymrio.calc_x_from_L to return a DataFrame with expected outputs
    with patch(
        "climada_petals.engine.supplychain.core.StaticIOModel.degraded_value_added",
        new_callable=PropertyMock,
    ) as mock_value_added:
        mock_value_added.return_value = pd.DataFrame(
            [[100, 200, 300, 400], [100, 200, 300, 400]],
            index=[1, 2],
            columns=pd.MultiIndex.from_tuples(
                [("regA", "sec1"), ("regA", "sec2"), ("regB", "sec1"), ("regB", "sec2")]
            ),
        )
        result = model.calc_ghosh()
    # Verify that the return matches the expected output
    pd.testing.assert_frame_equal(result, expected_output)


def test_calc_ghosh_with_shocks_event_ids(model):
    """Test calc_ghosh with defined direct shocks."""
    # We don't care about the real computation calc_x_from_L, we just want to test the rest of the method.

    expected_output = pd.DataFrame.from_dict(
        {
            "index": [2],
            "columns": [
                ("regA", "sec1"),
                ("regA", "sec2"),
                ("regB", "sec1"),
                ("regB", "sec2"),
            ],
            "data": [[1000, 2000, 3000, 4000]],
            "index_names": [None],
            "column_names": [None, None],
        },
        orient="tight",
    )

    # Mocking pymrio.calc_x_from_L to return a DataFrame with expected outputs
    with patch(
        "climada_petals.engine.supplychain.core.StaticIOModel.degraded_value_added",
        new_callable=PropertyMock,
    ) as mock_value_added:
        mock_value_added.return_value = pd.DataFrame(
            [[100, 200, 300, 400], [100, 200, 300, 400]],
            index=[1, 2],
            columns=pd.MultiIndex.from_tuples(
                [("regA", "sec1"), ("regA", "sec2"), ("regB", "sec1"), ("regB", "sec2")]
            ),
        )
        result = model.calc_ghosh(event_ids=[2])
        pd.testing.assert_frame_equal(result, expected_output)


class TestBoARIOModel(TestCase):

    def setUp(self):
        self.mriot = MagicMock(spec=IOSystem,
                               Z=MagicMock(spec=pd.DataFrame),
                               Y=MagicMock(spec=pd.DataFrame),
                               x=MagicMock(spec=pd.DataFrame),
                               A=MagicMock(spec=pd.DataFrame),
                               )
        self.mriot.name = "test"
        self.direct_shocks = MagicMock(spec=DirectShocksSet,
                                       mriot_name="test",
                                       event_ids=[1,2]
                                       )
        self.direct_shocks.exposure_assets = MagicMock(spec=pd.Series)
        self.direct_shocks.event_dates = MagicMock(spec=pd.Series)
        self.direct_shocks.event_dates.min.return_value = 1
        self.direct_shocks.event_dates.max.return_value = 10


    @patch("climada_petals.engine.supplychain.core.ARIOPsiModel")
    @patch("climada_petals.engine.supplychain.core.Simulation")
    def test_init_bare(self, mock_simulation, mock_ariopsi):
        model = BoARIOModel(self.mriot, self.direct_shocks)

        self.assertEqual(model.mriot, self.mriot)
        self.assertEqual(model.direct_shocks, self.direct_shocks)
        mock_ariopsi.assert_called_once_with(self.mriot, productive_capital_vector=self.direct_shocks.exposure_assets)
        mock_simulation.assert_called_once_with(mock_ariopsi(), n_temporal_units_to_sim=10-1+SIMULATION_LENGTH_BUFFER)
