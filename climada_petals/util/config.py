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

Define configuration parameters.
"""

__all__ = [
    'CONFIG',
]


import logging
from pathlib import Path

import climada
import climada.util.config
from climada.util.config import (
    Config, _fetch_conf, CONFIG_NAME, CONFIG, CONSOLE)

LOGGER = logging.getLogger('climada_petals')
LOGGER.propagate = False
LOGGER.addHandler(CONSOLE)

CORE_DIR = Path(climada.__file__).parent
SOURCE_DIR = Path(__file__).absolute().parent.parent

CONFIG.__dict__ = Config.from_dict(_fetch_conf([
    CORE_DIR / 'conf',  # default config from the climada repository
    SOURCE_DIR / 'conf',  # default config from the climada_petals repository
    Path.home() / 'climada' / 'conf',  # ~/climada/conf directory
    Path.home() / '.config',  # ~/.config directory
    Path.cwd(),  # current working directory
], CONFIG_NAME)).__dict__
LOGGER.setLevel(getattr(logging, CONFIG.log_level.str()))
