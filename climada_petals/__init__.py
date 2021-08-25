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
import climada_petals.util.config
from .util.constants import *

REPO_DATA = {
    'data/system': [
    ],
    'data/demo': [
        HAZ_DEMO_FLDDPH,
        HAZ_DEMO_FLDFRC,
        DEMO_GDP2ASSET,
    ]
}


def copy_repo_data(reload=False):
    for src_dir, path_list in REPO_DATA.items():
        for path in path_list:
            if not path.exists() or reload:
                src = Path(__file__).parent.parent.joinpath(src_dir, path.name)
                copyfile(src, path)

copy_repo_data()
