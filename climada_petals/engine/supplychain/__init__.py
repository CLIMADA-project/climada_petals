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
"""

import logging
#logging.getLogger("boario").setLevel(logging.DEBUG)

LOGGER = logging.getLogger(__name__)

from .core import DirectShocksSet, IndirectCostModel, StaticIOModel, BoARIOModel # noqa: E402 (ignore import order PEP8 rule because we have to define the logger before importing any other sub-modules)
from .mriot_handling import get_mriot # noqa: E402

__all__ = ["DirectShocksSet","IndirectCostModel", "StaticIOModel", "BoARIOModel", "get_mriot"]
