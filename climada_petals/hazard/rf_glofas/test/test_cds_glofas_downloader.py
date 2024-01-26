import tempfile
import unittest
import unittest.mock as mock
from copy import deepcopy
from pathlib import Path
from datetime import date

import cdsapi
from ruamel.yaml import YAML

from climada_petals.hazard.rf_glofas.cds_glofas_downloader import (
    glofas_request,
    glofas_request_single,
    request_to_md5,
    cleanup_download_dir,
    DEFAULT_REQUESTS,
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
        client_mock.assert_called_once_with(quiet=False, debug=False)
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
            client_mock.assert_called_once_with(quiet=False, debug=True, verify=True)
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
            client_mock.assert_called_once_with(quiet=False, debug=False)
            call_args = client_obj_mock.retrieve.call_args.args
            self.assertEqual(call_args[0], product)
            self.assertEqual(call_args[1], new_request)
            self.assertIn(request_to_md5(new_request), call_args[2].stem)

    @mock.patch(
        "climada_petals.hazard.rf_glofas.cds_glofas_downloader.glofas_request_multiple",
        autospec=True,
    )
    def test_forecast_single(self, mock_req):
        """Test request for a single forecast day"""
        glofas_request("forecast", "2022-01-01", None, self.tempdir.name)
        request = deepcopy(DEFAULT_REQUESTS["forecast"])
        request["month"] = "01"
        request["day"] = "01"
        mock_req.assert_called_once_with(
            "cems-glofas-forecast",
            [request],
            self.tempdir.name,
            1,
            True,
            None,
        )

    @mock.patch(
        "climada_petals.hazard.rf_glofas.cds_glofas_downloader.Client", autospec=True
    )
    def test_forecast_filetype(self, client_mock):
        """Test correct filetype suffix"""
        client_obj_mock = mock.create_autospec(cdsapi.Client)
        client_mock.return_value = client_obj_mock

        # Use default grib
        request = deepcopy(DEFAULT_REQUESTS["forecast"])
        request["format"] = "grib"
        glofas_request_single(
            "forecast",
            request,
            self.tempdir.name,
            use_cache=False,
        )
        call_args = client_obj_mock.retrieve.call_args.args
        self.assertEqual(call_args[2].suffix, ".grib")

        # Use nonsense (should be .nc then)
        request["format"] = "foo"
        glofas_request_single(
            "forecast",
            request,
            self.tempdir.name,
            use_cache=False,
        )
        call_args = client_obj_mock.retrieve.call_args.args
        self.assertEqual(call_args[2].suffix, ".nc")

    def test_forecast_wrong_date(self):
        """Test correct error for wrong date specification"""
        with self.assertRaises(ValueError):
            glofas_request("forecast", "2022-01-01", "2022-01111", self.tempdir.name)

    @mock.patch(
        "climada_petals.hazard.rf_glofas.cds_glofas_downloader.glofas_request_multiple",
        autospec=True,
    )
    def test_forecast_iter(self, mock_req):
        """Test request for multiple forecast days"""
        glofas_request("forecast", "2022-12-31", "2023-01-01", self.tempdir.name)
        requests = mock_req.call_args.args[1]
        self.assertEqual(requests[0]["year"], "2022")
        self.assertEqual(requests[1]["year"], "2023")
        self.assertEqual(requests[0]["month"], "12")
        self.assertEqual(requests[1]["month"], "01")
        self.assertEqual(requests[0]["day"], "31")
        self.assertEqual(requests[1]["day"], "01")
        self.assertEqual(mock_req.call_args.args[2], self.tempdir.name)

    @mock.patch(
        "climada_petals.hazard.rf_glofas.cds_glofas_downloader.glofas_request_multiple",
        autospec=True,
    )
    def test_historical_single(self, mock_req):
        """Test request for single historical year"""
        glofas_request("historical", "2019", None, self.tempdir.name)
        request = deepcopy(DEFAULT_REQUESTS["historical"])
        request["hyear"] = "2019"
        mock_req.assert_called_once_with(
            "cems-glofas-historical",
            [request],
            self.tempdir.name,
            1,
            True,
            None,
        )

    @mock.patch(
        "climada_petals.hazard.rf_glofas.cds_glofas_downloader.glofas_request_multiple",
        autospec=True,
    )
    def test_historical_iter(self, mock_req):
        """Test request for multiple historical years"""
        glofas_request("historical", "2019", "2021", self.tempdir.name)
        requests = mock_req.call_args.args[1]
        self.assertEqual(requests[0]["hyear"], "2019")
        self.assertEqual(requests[1]["hyear"], "2020")
        self.assertEqual(requests[2]["hyear"], "2021")
        self.assertEqual(
            mock_req.call_args.args[2],
            self.tempdir.name
        )

    def test_historical_wrong_date(self):
        """Test correct error for wrong date specification"""
        with self.assertRaises(ValueError):
            glofas_request("historical", "2022", "2022-01-01", self.tempdir.name)

    def test_wrong_product(self):
        """Test handling of unknown product"""
        with self.assertRaises(NotImplementedError):
            glofas_request("abc", "2022", "2022", self.tempdir.name)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestGloFASRequest)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
