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

Export functions of the GloFAS River Flood Module
"""
# ruff: noqa

try:
    from .setup import setup_all
    from .river_flood_computation import RiverFloodInundation
    from .rf_glofas import hazard_series_from_dataset
    from .transform_ops import save_file
except ImportError as e:
    if 'xesmf' in str(e) or 'esmpy' in str(e):
        raise ImportError(
            "climada_petals.hazard.rf_glofas requires ESMF support. "
            "Install it with:\n"
            "  conda install -c conda-forge esmpy xesmf\n"
            "Note: On Windows ESMF support may not be available."
        ) from e
    raise

