import tempfile
import unittest
import unittest.mock as mock
from copy import deepcopy
from pathlib import Path

from climada_petals.util import glofas_request
from climada_petals.util.cds_glofas_downloader import DEFAULT_REQUESTS


class TestGloFASRequest(unittest.TestCase):
    """Test requests to the CDS API"""

    def setUp(self):
        """Create temporary directory in case we download data"""
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Clean up the temporary directory"""
        self.tempdir.cleanup()

    @mock.patch(
        "climada_petals.util.cds_glofas_downloader.glofas_request_multiple",
        autospec=True,
    )
    def test_forecast_single(self, mock_req):
        """Test request for a single forecast day"""
        glofas_request("forecast", "2022-01-01", "2022-01-01", self.tempdir.name)
        request = deepcopy(DEFAULT_REQUESTS["forecast"])
        request["month"] = "01"
        request["day"] = "01"
        mock_req.assert_called_once_with(
            "cems-glofas-forecast",
            [request],
            [Path(self.tempdir.name, "glofas-forecast-ensemble-2022-01-01.nc")],
            1,
        )

    @mock.patch(
        "climada_petals.util.cds_glofas_downloader.glofas_request_multiple",
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
        "climada_petals.util.cds_glofas_downloader.glofas_request_multiple",
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
                Path(self.tempdir.name, "glofas-forecast-ensemble-2022-12-31.nc"),
                Path(self.tempdir.name, "glofas-forecast-ensemble-2023-01-01.nc"),
            ],
        )

    @mock.patch(
        "climada_petals.util.cds_glofas_downloader.glofas_request_multiple",
        autospec=True,
    )
    def test_historical_single(self, mock_req):
        """Test request for single historical year"""
        glofas_request("historical", "2019", "2019", self.tempdir.name)
        request = deepcopy(DEFAULT_REQUESTS["historical"])
        request["hyear"] = "2019"
        mock_req.assert_called_once_with(
            "cems-glofas-historical",
            [request],
            [Path(self.tempdir.name, "glofas-historical-2019.grib")],
            1,
        )

    @mock.patch(
        "climada_petals.util.cds_glofas_downloader.glofas_request_multiple",
        autospec=True,
    )
    def test_historical_iter(self, mock_req):
        """Test request for multiple historical years"""
        glofas_request("historical", "2019", "2020", self.tempdir.name)
        requests = mock_req.call_args.args[1]
        self.assertEqual(requests[0]["hyear"], "2019")
        self.assertEqual(requests[1]["hyear"], "2020")
        self.assertEqual(
            mock_req.call_args.args[2],
            [
                Path(self.tempdir.name, "glofas-historical-2019.grib"),
                Path(self.tempdir.name, "glofas-historical-2020.grib"),
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
