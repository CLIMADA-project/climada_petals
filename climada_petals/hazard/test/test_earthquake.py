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

Test earthquake module.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climada.hazard import Centroids
from climada_petals.hazard.earthquake import (
    MIN_MMI,
    Earthquake,
    attenuation_MMI,
)


DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def _read_isc_gem_catalogue(path):
    """Read the ISC-GEM catalogue using its commented header."""
    with path.open() as file_handle:
        for line in file_handle:
            if line.startswith("#"):
                header = line
            elif line.strip():
                break

    catalogue = pd.read_csv(
        path,
        delimiter=" , ",
        comment="#",
        header=None,
        engine="python",
    )
    catalogue.columns = [name.strip() for name in header[1:].split(",")]
    return catalogue


class TestEarthquake:
    """Test earthquake hazard construction and event sampling."""

    def setup_method(self):
        """Create a small catalogue and centroid set shared by the tests."""
        self.centroids = Centroids.from_lat_lon(
            lat=np.array([0.0, 10.0]),
            lon=np.array([0.0, 10.0]),
        )
        self.catalogue = pd.DataFrame(
            {
                "lat": [0.0, 10.0, 5.0, 50.0],
                "lon": [0.0, 10.0, 5.0, 50.0],
                "mw": [7.0, 2.0, 6.0, 6.0],
                "depth": [10.0, 10.0, 10.0, 10.0],
                "eventid": [1, 2, 3, 4],
                "date": [
                    "2000-01-01 00:00:00.000000",
                    "2005-01-01 00:00:00.000000",
                    "2010-01-01 00:00:00.000000",
                    "1990-01-01 00:00:00.000000",
                ],
            }
        )

    def test_from_mw_depth(self):
        """Build a hazard, filter distant events, and preserve frequency."""
        hazard = Earthquake.from_Mw_depth(self.catalogue, self.centroids)

        assert hazard.event_id.tolist() == [1, 2, 3]
        assert hazard.intensity.shape == (3, 2)
        assert hazard.intensity.nnz == 1
        assert hazard.intensity[0, 0] == pytest.approx(9.0)
        np.testing.assert_allclose(hazard.frequency, 1.0 / 21.0)
        np.testing.assert_array_equal(hazard.orig, np.ones(3, dtype=bool))
        assert hazard.units == "MMI"

    def test_from_mw_depth_with_isc_gem_catalogue(self):
        """Build a hazard from the packaged ISC-GEM catalogue and centroids."""
        data_dir = Path(__file__).parent / "data"
        catalogue_file = data_dir / "isc-gem-cat.csv"
        centroids_file = data_dir / "NZL_NewZealand_centroids.hdf5"

        if not catalogue_file.exists() or not centroids_file.exists():
            pytest.skip("ISC-GEM catalogue or New Zealand centroids are unavailable")

        catalogue = _read_isc_gem_catalogue(catalogue_file)
        centroids = Centroids.from_hdf5(centroids_file)

        hazard = Earthquake.from_Mw_depth(catalogue, centroids)

        assert hazard.event_id.size > 0
        assert hazard.intensity.shape == (hazard.event_id.size, centroids.size)
        assert hazard.intensity.nnz > 0
        assert hazard.units == "MMI"
        assert np.all(hazard.frequency > 0)

    def test_from_mw_depth_empty_intensity(self):
        """Retain an event even when all calculated intensities are zero."""
        weak_catalogue = self.catalogue.iloc[[1]].copy()

        hazard = Earthquake.from_Mw_depth(
            weak_catalogue,
            self.centroids,
            orig=False,
        )

        assert hazard.intensity.shape == (1, 2)
        assert hazard.intensity.nnz == 0
        np.testing.assert_array_equal(hazard.orig, [False])

    def test_uniform_random_events(self, monkeypatch):
        """Apply deterministic location, magnitude, depth, and date changes."""
        centroids = Centroids.from_lat_lon([89.8], [179.8])
        catalogue = pd.DataFrame(
            {
                "lat": [89.8, 89.7, 0.0],
                "lon": [179.8, 179.7, 0.0],
                "mw": [6.0, 7.0, 8.0],
                "depth": [10.0, 20.0, 30.0],
                "eventid": [10, 20, 30],
                "date": [
                    "2000-02-29 00:00:00.000000",
                    "2010-01-01 00:00:00.000000",
                    "1990-01-01 00:00:00.000000",
                ],
            }
        )

        random_values = iter(
            [
                np.full(4, 0.5),
                np.full(4, 0.5),
                np.full(4, 0.25),
                np.full(4, 0.1),
            ]
        )
        monkeypatch.setattr(
            np.random,
            "uniform",
            lambda low, high, size: next(random_values),
        )

        captured = {}

        def capture_hazard(cls, df, centroids, orig=True):
            captured["df"] = df
            captured["orig"] = orig
            return "hazard"

        monkeypatch.setattr(
            Earthquake,
            "from_Mw_depth",
            classmethod(capture_hazard),
        )

        result = Earthquake.uniform_random_events(
            catalogue,
            centroids,
            n=2,
        )

        assert result == "hazard"
        assert captured["orig"] is False

        generated = captured["df"]
        np.testing.assert_allclose(generated["lat"], 90.0)
        np.testing.assert_allclose(generated["lon"], -179.7)
        np.testing.assert_allclose(generated["mw"], [6.25, 7.25, 6.25, 7.25])
        np.testing.assert_allclose(generated["depth"], [11.0, 22.0, 11.0, 22.0])

        dates = [
            datetime.strptime(value, DATE_FORMAT)
            for value in generated["date"]
        ]
        assert dates == [
            datetime(2021, 2, 28),
            datetime(2031, 1, 1),
            datetime(2042, 2, 28),
            datetime(2052, 1, 1),
        ]

    @pytest.mark.parametrize(
        ("catalogue", "centroids", "message"),
        [
            ("empty", "normal", "catalogue is empty"),
            ("normal", "empty", "centroid set is empty"),
            ("far", "normal", "No earthquake catalogue events"),
        ],
    )
    def test_from_mw_depth_validation(self, catalogue, centroids, message):
        """Reject invalid inputs before constructing a hazard."""
        catalogues = {
            "normal": self.catalogue.iloc[[0]],
            "empty": self.catalogue.iloc[0:0],
            "far": self.catalogue.iloc[[3]],
        }
        centroid_sets = {
            "normal": self.centroids,
            "empty": Centroids.from_lat_lon([], []),
        }

        with pytest.raises(ValueError, match=message):
            Earthquake.from_Mw_depth(
                catalogues[catalogue],
                centroid_sets[centroids],
            )

    @pytest.mark.parametrize(
        ("catalogue", "centroids", "n", "message"),
        [
            ("normal", "normal", 0, "positive integer"),
            ("empty", "normal", 1, "catalogue is empty"),
            ("normal", "empty", 1, "centroid set is empty"),
            ("far", "normal", 1, "buffered centroid extent"),
        ],
    )
    def test_uniform_random_events_validation(
        self,
        catalogue,
        centroids,
        n,
        message,
    ):
        """Reject invalid sampling inputs."""
        catalogues = {
            "normal": self.catalogue.iloc[[0]],
            "empty": self.catalogue.iloc[0:0],
            "far": self.catalogue.iloc[[3]],
        }
        centroid_sets = {
            "normal": self.centroids,
            "empty": Centroids.from_lat_lon([], []),
        }

        with pytest.raises(ValueError, match=message):
            Earthquake.uniform_random_events(
                catalogues[catalogue],
                centroid_sets[centroids],
                n=n,
            )


def test_attenuation_mmi():
    """Apply the lower threshold, upper cap, and distance validation."""
    mmi = attenuation_MMI(
        np.array([0.0, 100.0, 1.0e6]),
        mag=7.0,
        depth=10.0,
    )

    assert mmi[0] == pytest.approx(9.0)
    assert MIN_MMI <= mmi[1] <= 9.0
    assert mmi[2] == 0.0

    with pytest.raises(ValueError, match="must not be negative"):
        attenuation_MMI([-1.0], mag=7.0, depth=10.0)
