import tempfile
import unittest
import unittest.mock as mock
from copy import deepcopy
from pathlib import Path

import cdsapi
from ruamel.yaml import YAML

from climada_petals.hazard.rf_glofas.cds_glofas_downloader import (
    glofas_request,
    glofas_request_single,
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

    @mock.patch(
        "climada_petals.hazard.rf_glofas.cds_glofas_downloader.Client", autospec=True
    )
    def test_request_single(self, client_mock):
        """Test execution of a single request without actually downloading stuff"""
        product = "product"
        request = deepcopy(DEFAULT_REQUESTS["forecast"])
        outfile = Path(self.tempdir.name, "request.nc")
        client_obj_mock = mock.create_autospec(cdsapi.Client)
        client_mock.return_value = client_obj_mock

        # Call once
        glofas_request_single(product, request, outfile, use_cache=True)
        client_mock.assert_called_once_with(quiet=False, debug=False)
        client_obj_mock.retrieve.assert_called_once_with(product, request, outfile)

        # Check if request was correctly dumped
        outfile_yml = outfile.with_suffix(".yml")
        self.assertTrue(outfile_yml.exists())
        yaml = YAML()
        self.assertEqual(yaml.load(outfile_yml), request)

        # Call again to check caching
        with tempfile.NamedTemporaryFile(dir=self.tempdir.name) as tmp_file:
            # Dump the request next to the (fake) outfile
            yaml.dump(request, Path(tmp_file.name).with_suffix(".yml"))

            # Client should not have been called again
            client_mock.reset_mock()
            client_obj_mock.reset_mock()
            glofas_request_single(product, request, tmp_file.name, use_cache=True)
            client_mock.assert_not_called()
            client_obj_mock.retrieve.assert_not_called()

            # ...but it should when cache is not used
            # Also check client_kw here!
            glofas_request_single(
                product,
                request,
                tmp_file.name,
                use_cache=False,
                client_kw=dict(verify=True, debug=True),
            )
            client_mock.assert_called_once_with(quiet=False, debug=True, verify=True)
            client_obj_mock.retrieve.assert_called_once_with(
                product, request, tmp_file.name
            )

            # Wrong request should also induce new download
            client_mock.reset_mock()
            client_obj_mock.reset_mock()
            wrong_request = deepcopy(request)
            wrong_request["leadtime_hour"][2] = "xx"
            yaml.dump(wrong_request, Path(tmp_file.name).with_suffix(".yml"))
            glofas_request_single(product, request, tmp_file.name, use_cache=True)
            client_mock.assert_called_once_with(quiet=False, debug=False)
            client_obj_mock.retrieve.assert_called_once_with(
                product, request, tmp_file.name
            )

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
            [Path(self.tempdir.name, "glofas-forecast-ensemble-2022-01-01.grib")],
            1,
            True,
            None,
        )

    @mock.patch(
        "climada_petals.hazard.rf_glofas.cds_glofas_downloader.glofas_request_multiple",
        autospec=True,
    )
    def test_forecast_filetype(self, mock_req):
        """Test correct filetype suffix"""
        glofas_request(
            "forecast",
            "2022-01-01",
            "2022-01-01",
            self.tempdir.name,
            request_kw=dict(format="grib"),
        )
        self.assertEqual(mock_req.call_args.args[2][0].suffix, ".grib")

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
        self.assertEqual(
            mock_req.call_args.args[2],
            [
                Path(self.tempdir.name, "glofas-forecast-ensemble-2022-12-31.grib"),
                Path(self.tempdir.name, "glofas-forecast-ensemble-2023-01-01.grib"),
            ],
        )

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
            [Path(self.tempdir.name, "glofas-historical-2019.grib")],
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
            [
                Path(self.tempdir.name, "glofas-historical-2019.grib"),
                Path(self.tempdir.name, "glofas-historical-2020.grib"),
                Path(self.tempdir.name, "glofas-historical-2021.grib"),
            ],
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
