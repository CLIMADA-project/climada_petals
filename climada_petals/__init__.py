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

climada init
"""
from shutil import copyfile
from pathlib import Path

import climada
from climada.util.config import setup_logging
from .util.config import CONFIG
from .util.constants import *


__all__ = ['init']

REPO_DATA = {
    'data/system': [
    ],
    'data/demo': [
        HAZ_DEMO_FLDDPH,
        HAZ_DEMO_FLDFRC,
        DEMO_GDP2ASSET,
    ]
}


def setup_climada_data(reload=False):

    for dirpath in [DEMO_DIR, SYSTEM_DIR]:
        dirpath.mkdir(parents=True, exist_ok=True)

    for src_dir, path_list in REPO_DATA.items():
        for path in path_list:
            if not path.exists() or reload:
                src = Path(__file__).parent.parent.joinpath(src_dir, path.name)
                copyfile(src, path)


def init():
    climada.init()
    setup_climada_data()
    setup_logging(CONFIG.log_level.str())


init()
