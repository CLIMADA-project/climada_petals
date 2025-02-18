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

Tests on Coastal Flood Hazard"""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import tempfile
from climada.hazard import Hazard
from climada_petals.hazard.coastal_flood import AQUEDUCT_SOURCE_LINK, CoastalFlood


class TestReader(unittest.TestCase):
    """
    Test that CoastalFlood.from_aqueduct_tif:
        1) Constructs the correct file URLs and triggers downloads as needed.
        2) Calls from_raster with the expected file paths.
    """

    @patch("climada_petals.hazard.coastal_flood.u_fh.download_file")
    @patch.object(Hazard, "from_raster")
    def test_from_aqueduct_tif(self, mock_from_raster, mock_download_file):

        rcp = "45"
        target_year = "2030"
        return_periods = [50, 100]

        with tempfile.TemporaryDirectory() as temp_dir:

            temp_path = Path(temp_dir)

            mock_hazard_instance = MagicMock()
            mock_from_raster.return_value = mock_hazard_instance

            def download_side_effect(url, download_dir):

                filename = url.split("/")[-1]
                path = download_dir / filename
                path.touch()  # create an empty file

                return path

            mock_download_file.side_effect = download_side_effect

            result_haz = CoastalFlood.from_aqueduct_tif(
                rcp=rcp,
                target_year=target_year,
                return_periods=return_periods,
                subsidence="wtsub",
                percentile="95",
                dwd_dir=temp_path,
            )

            # Check the calls to u_fh.download_file
            expected_files = [
                "inuncoast_rcp4p5_wtsub_2030_rp0100_0.tif",
                "inuncoast_rcp4p5_wtsub_2030_rp0050_0.tif",
            ]

            expected_urls = [
                AQUEDUCT_SOURCE_LINK + filename for filename in expected_files
            ]

            expected_call_args_list = [
                call(url, download_dir=temp_path) for url in expected_urls
            ]

            # Order of calls does not matter.
            call_args_list = mock_download_file.call_args_list
            self.assertCountEqual(call_args_list, expected_call_args_list)

            # Check the calls to from_raster.
            mock_from_raster.assert_called_once()

            _, call_kwargs = mock_from_raster.call_args
            files_intensity = [f.name for f in call_kwargs["files_intensity"]]
            self.assertCountEqual(expected_files, files_intensity)

            self.assertIs(result_haz, mock_hazard_instance)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
