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

init Copernicus forecast handler
"""

from .create_seasonal_forecast_hazard import *  # This will import all functions from create_seasonal_forecast_hazard.py
from .downloader import (
    download_data,
)  # This will import all functions from downloader.py
from .heat_index import *  # This will import all functions from heat_index.py
from .index_definitions import *  # This will import all functions from index_definitions.py
from .seasonal_statistics import *  # This will import all functions from seasonal_statistics.py
from .time_utils import *  # Time-related helpers (e.g. month conversion, leadtime calculation)