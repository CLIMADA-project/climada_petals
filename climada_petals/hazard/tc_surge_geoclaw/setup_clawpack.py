"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Download and install a version of the flow solver GeoClaw (part of the Clawpack library)
"""

import importlib
import logging
import pathlib
import site
import subprocess
import sys
from typing import Optional, Tuple
import warnings

from climada import CONFIG


LOGGER = logging.getLogger(__name__)

CLAWPACK_VERSION = CONFIG.hazard.tc_surge_geoclaw.resources.clawpack_version.str()
"""Version or git decorator (tag, branch) of Clawpack to use."""

CLAWPACK_GIT_URL = CONFIG.hazard.tc_surge_geoclaw.resources.clawpack_git.str()
"""URL of the official Clawpack git repository."""

CLAWPACK_SRC_DIR = CONFIG.hazard.tc_surge_geoclaw.clawpack_src_dir.dir()
"""Directory for Clawpack source checkouts (if it doesn't exist)"""


def clawpack_info() -> Tuple[Optional[pathlib.Path], Tuple[str]]:
    """Information about the available clawpack version

    Returns
    -------
    path : Path or None
        If the python package clawpack is not available, None is returned.
        Otherwise, the CLAW source path is returned.
    decorators : tuple of str
        Strings describing the available version of clawpack. If it's a git
        checkout, the first string will be the full commit hash and the
        following strings will be git decorators such as tags or branch names
        that point to this checkout.
    """
    git_cmd = ["git", "log", "--pretty=format:%H%D", "-1"]
    try:
        # pylint: disable=import-outside-toplevel
        import clawpack
    except ImportError:
        return None, ()

    ver = clawpack.__version__
    path = pathlib.Path(clawpack.__file__).parent.parent
    LOGGER.info("Found Clawpack version %s in %s", ver, path)

    proc = subprocess.Popen(git_cmd, stdout=subprocess.PIPE, cwd=path)
    out = proc.communicate()[0].decode()
    if proc.returncode != 0:
        return path, (ver,)
    decorators = [out[:40]] + out[40:].split(", ")
    decorators = [d.replace("tag: ", "") for d in decorators]
    decorators = [d.replace("HEAD -> ", "") for d in decorators]
    return path, tuple(decorators)


def setup_clawpack(version : str = CLAWPACK_VERSION, overwrite: bool = False) -> None:
    """Install the specified version of clawpack if not already present

    Parameters
    ----------
    version : str, optional
        A git (short or long) hash, branch name or tag.
    overwrite : bool, optional
        If ``True``, perform a fresh install even if an existing installation is found.
        Defaults to ``False``.
    """
    if sys.platform.startswith("win"):
        raise RuntimeError(
            "The TCSurgeGeoClaw feature only works on Mac and Linux since Windows is not"
            "supported by the GeoClaw package."
        )

    path, git_ver = clawpack_info()
    if overwrite or (
        path is None or version not in git_ver and version not in git_ver[0]
    ):
        LOGGER.info("Installing Clawpack version %s", version)
        pkg = f"git+{CLAWPACK_GIT_URL}@{version}#egg=clawpack"
        cmd = [
            sys.executable, "-m", "pip", "install", "--src", CLAWPACK_SRC_DIR,
            "--no-build-isolation", "-e", pkg,
        ]
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            LOGGER.warning("pip install failed with return code %d and stdout:", exc.returncode)
            print(exc.output.decode("utf-8"))
            raise RuntimeError(
                "pip install failed with return code %d (see output above)."
                " Make sure that a Fortran compiler (e.g. gfortran) is available on "
                "your machine before using tc_surge_geoclaw!"
            ) from exc
        importlib.reload(site)
        importlib.invalidate_caches()

    with warnings.catch_warnings():
        # pylint: disable=unused-import,import-outside-toplevel
        warnings.filterwarnings(
            "ignore",
            message="unclosed <socket.socket",
            module="clawpack",
            category=ResourceWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="the imp module is deprecated",
            module="clawpack",
            category=DeprecationWarning,
        )
        import clawpack.pyclaw
