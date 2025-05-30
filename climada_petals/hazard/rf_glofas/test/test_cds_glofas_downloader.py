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

Tests for cds_glofas_downloader.py
"""

import tempfile
import unittest
import unittest.mock as mock
from copy import deepcopy
from pathlib import Path
from datetime import date

import cdsapi
import pandas as pd
from ruamel.yaml import YAML

from climada_petals.hazard.rf_glofas.cds_glofas_downloader import (
    glofas_request,
    glofas_request_single,
    request_to_md5,
    cleanup_download_dir,
    datetime_index_to_request,
    DEFAULT_REQUESTS,
    CLIENT_KW_DEFAULT,
)


class TestDateTimeIndexToRequest(unittest.TestCase):
    """Test the function turning a Pandas Datetime Index into a request sequence"""

    def setUp(self):
        """Create the default range"""
        self.index_default = pd.date_range("2000-01-01", "2001-02-02")
        self.target_default = {
            "year": [f"{year:04}" for year in self.index_default.year],
            "month": [f"{month:02}" for month in self.index_default.month],
            "day": [f"{day:02}" for day in self.index_default.day],
        }

    def test_range_default(self):
        """Test transfering default range"""
        for product in ("historical", "forecast"):
            with self.subTest(product=product):
                request = datetime_index_to_request(self.index_default, product=product)
                request_test = self.target_default
                if product == "historical":
                    request_test = {"h" + key: val for key, val in request_test.items()}
                self.assertDictEqual(request, request_test)

    def test_year_range(self):
        """Test range of two year starts"""
        request = datetime_index_to_request(
            pd.date_range("2000-01-01", "2001-01-01", freq="YS"), product="forecast"
        )
        self.assertDictEqual(
            request,
            {"year": ["2000", "2001"], "month": ["01", "01"], "day": ["01", "01"]},
        )


class TestGloFASRequest(unittest.TestCase):
    """Test requests to the CDS API"""

    def setUp(self):
        """Create temporary directory in case we download data"""
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Clean up the temporary directory"""
        self.tempdir.cleanup()

    def test_cleanup_download_dir(self):
        """Check if deleting download directory contents works"""
        _, filename = tempfile.mkstemp(dir=self.tempdir.name)
        cleanup_download_dir(self.tempdir.name, dry_run=True)
        self.assertTrue(Path(filename).is_file())
        cleanup_download_dir(self.tempdir.name, dry_run=False)
        self.assertFalse(Path(filename).is_file())

    @mock.patch(
        "climada_petals.hazard.rf_glofas.cds_glofas_downloader.Client", autospec=True
    )
    def test_request_single(self, client_mock):
        """Test execution of a single request without actually downloading stuff"""
        product = "product"
        request = deepcopy(DEFAULT_REQUESTS["forecast"])
        outdir = Path(self.tempdir.name)
        client_obj_mock = mock.create_autospec(cdsapi.Client)
        client_mock.return_value = client_obj_mock

        # Call once
        glofas_request_single(product, request, outdir, use_cache=True)
        client_mock.assert_called_once_with(**CLIENT_KW_DEFAULT)
        call_args = client_obj_mock.retrieve.call_args.args
        self.assertEqual(call_args[0], product)
        self.assertEqual(call_args[1], request)
        request_hash = request_to_md5(request)
        self.assertIn(request_hash, call_args[2].stem)
        self.assertIn(date.today().strftime("%y%m%d"), call_args[2].stem)

        # Check if request was correctly dumped
        outfile_yml = next(outdir.glob(f"*-{request_hash}.yml"))
        yaml = YAML()
        self.assertEqual(yaml.load(outfile_yml), request)

        # Call again to check caching, client should not have been called again
        with tempfile.NamedTemporaryFile(dir=outdir, suffix=f"-{request_hash}.grib"):
            client_mock.reset_mock()
            client_obj_mock.reset_mock()
            glofas_request_single(product, request, outdir, use_cache=True)
            client_mock.assert_not_called()
            client_obj_mock.retrieve.assert_not_called()

            # ...but it should when cache is not used
            # Also check client_kw here!
            glofas_request_single(
                product,
                request,
                outdir,
                use_cache=False,
                client_kw=dict(verify=True, debug=True),
            )
            client_mock.assert_called_once_with(
                **(CLIENT_KW_DEFAULT | {"verify": True, "debug": True})
            )
            call_args = client_obj_mock.retrieve.call_args.args
            self.assertEqual(call_args[0], product)
            self.assertEqual(call_args[1], request)
            self.assertIn(request_hash, call_args[2].stem)

            # Different request should also induce new download
            client_mock.reset_mock()
            client_obj_mock.reset_mock()
            new_request = deepcopy(request)
            new_request["leadtime_hour"][2] = "xx"
            glofas_request_single(product, new_request, outdir, use_cache=True)
            client_mock.assert_called_once_with(**CLIENT_KW_DEFAULT)
            call_args = client_obj_mock.retrieve.call_args.args
            self.assertEqual(call_args[0], product)
            self.assertEqual(call_args[1], new_request)
            self.assertIn(request_to_md5(new_request), call_args[2].stem)

    @mock.patch.multiple(
        "climada_petals.hazard.rf_glofas.cds_glofas_downloader",
        glofas_request_single=mock.DEFAULT,
        glofas_request_multiple=mock.DEFAULT,
        autospec=True,
    )
    def test_forecast_single(self, glofas_request_single, glofas_request_multiple):
        """Test request for a single request (no splitting)"""
        glofas_request(
            "forecast", self.tempdir.name, request_kw={"year": ["2000", "2099"]}
        )

        request = deepcopy(DEFAULT_REQUESTS["forecast"])
        request["year"] = ["2000", "2099"]
        glofas_request_single.assert_called_once_with(
            "cems-glofas-forecast",
            request,
            self.tempdir.name,
            True,
            None,
        )
        glofas_request_multiple.assert_not_called()

    @mock.patch(
        "climada_petals.hazard.rf_glofas.cds_glofas_downloader.Client", autospec=True
    )
    def test_forecast_filetype(self, client_mock):
        """Test correct filetype suffix"""
        client_obj_mock = mock.create_autospec(cdsapi.Client)
        client_mock.return_value = client_obj_mock

        # Use default grib
        request = deepcopy(DEFAULT_REQUESTS["forecast"])
        request["data_format"] = "grib2"
        glofas_request_single(
            "forecast",
            request,
            self.tempdir.name,
            use_cache=False,
        )
        call_args = client_obj_mock.retrieve.call_args.args
        self.assertEqual(call_args[2].suffix, ".grib")

        # Use nonsense (should be .nc then)
        request["data_format"] = "foo"
        glofas_request_single(
            "forecast",
            request,
            self.tempdir.name,
            use_cache=False,
        )
        call_args = client_obj_mock.retrieve.call_args.args
        self.assertEqual(call_args[2].suffix, ".nc")

    @mock.patch.multiple(
        "climada_petals.hazard.rf_glofas.cds_glofas_downloader",
        glofas_request_single=mock.DEFAULT,
        glofas_request_multiple=mock.DEFAULT,
        autospec=True,
    )
    def test_dispatch_to_request_multiple(
        self, glofas_request_single, glofas_request_multiple
    ):
        """Test requesting sequentially and in parallel"""
        default_request = deepcopy(DEFAULT_REQUESTS["forecast"])
        request_kw = {"year": ["2000", "2001"], "day": ["01", "02"]}

        # Single request
        glofas_request(
            "forecast",
            self.tempdir.name,
            request_kw=request_kw,
        )
        glofas_request_multiple.assert_not_called()
        request = glofas_request_single.call_args.args[1]
        self.assertEqual(request["year"], request_kw["year"])
        self.assertEqual(request["month"], default_request["month"])
        self.assertEqual(request["day"], request_kw["day"])
        self.assertEqual(glofas_request_single.call_args.args[2], self.tempdir.name)
        glofas_request_single.reset_mock()
        glofas_request_multiple.reset_mock()

        # Another single request
        glofas_request(
            "forecast",
            self.tempdir.name,
            request_kw=request_kw,
            requests=[{"month": "03"}]  # Also test if request is sanitized
        )
        glofas_request_multiple.assert_not_called()
        request = glofas_request_single.call_args.args[1]
        self.assertEqual(request["year"], request_kw["year"])
        self.assertEqual(request["month"], ["03"])  # Sanitized here
        self.assertEqual(request["day"], request_kw["day"])
        glofas_request_single.reset_mock()
        glofas_request_multiple.reset_mock()

        # Multiple requests
        glofas_request(
            "forecast",
            self.tempdir.name,
            request_kw=request_kw,
            requests=[{"month": "03"}, {"month": "04"}]
        )
        glofas_request_single.assert_not_called()
        requests = glofas_request_multiple.call_args.args[1]
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["year"], request_kw["year"])
        self.assertEqual(requests[1]["year"], request_kw["year"])
        self.assertEqual(requests[0]["month"], ["03"])
        self.assertEqual(requests[1]["month"], ["04"])
        self.assertEqual(glofas_request_multiple.call_args.args[2], self.tempdir.name)

    @mock.patch.multiple(
        "climada_petals.hazard.rf_glofas.cds_glofas_downloader",
        glofas_request_single=mock.DEFAULT,
        glofas_request_multiple=mock.DEFAULT,
        autospec=True,
    )
    def test_historical_single(self, glofas_request_single, glofas_request_multiple):
        """Test request for single historical year"""
        glofas_request("historical", self.tempdir.name, request_kw={"hyear": ["2019"]})
        request = deepcopy(DEFAULT_REQUESTS["historical"])
        request["hyear"] = ["2019"]
        glofas_request_multiple.assert_not_called()
        glofas_request_single.assert_called_once_with(
            "cems-glofas-historical",
            request,
            self.tempdir.name,
            True,
            None,
        )

    def test_wrong_product(self):
        """Test handling of unknown product"""
        with self.assertRaises(NotImplementedError):
            glofas_request("abc", self.tempdir.name)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDateTimeIndexToRequest)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGloFASRequest))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
