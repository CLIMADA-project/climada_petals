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

Wrapper class for work directory setup and running of GeoClaw simulations
"""

import contextlib
import datetime as dt
import inspect
import logging
import os
import pathlib
import re
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import psutil
import rasterio
import xarray as xr

from climada.hazard.trop_cyclone import (
    KN_TO_MS,
    NM_TO_KM,
    MBAR_TO_PA,
)
import climada.util.coordinates as u_coord
from .plot import plot_dems
from .setup_clawpack import clawpack_info


LOGGER = logging.getLogger(__name__)


class GeoClawRunner():
    """"Wrapper for work directory setup and running of GeoClaw simulations.

    Attributes
    ----------
    surge_h : ndarray
        Maximum height of inundation recorded at given centroids.
    gauge_data : list of dicts
        For each gauge, a dict containing `location`, `base_sea_level`, `topo_height`, `time`,
        `height_above_geoid`, `height_above_ground`, and `amr_level` information.
    """
    def __init__(
        self,
        base_dir : str,
        track : xr.Dataset,
        time_offset : np.datetime64,
        areas : Dict,
        centroids : np.ndarray,
        topo_path : Union[pathlib.Path, str],
        topo_res_as : float = 30.0,
        gauges : Optional[List] = None,
        sea_level : Union[Callable, float] = 0.0,
        outer_pad_deg : float = 5,
        boundary_conditions : str = "extrap",
        output_freq_s : float = 0.0,
        recompile : bool = False,
    ):
        """Initialize GeoClaw working directory with ClawPack rundata

        Parameters
        ----------
        base_dir : str
            Location where to create the working directory.
        track : xr.Dataset
            Single tropical cyclone track.
        time_offset : np.datetime64
            Usually, time of landfall
        areas : dict
            Landfall event (single iterator output from TCSurgeEvents).
        centroids : ndarray
            Points for which to record the maximum height of inundation.
            Each row is a lat-lon point.
        topo_path : Path or str
            Path to raster file containing gridded elevation data.
        topo_res_as : float, optional
            The resolution at which to extract topography data in arc-seconds. Needs to be at
            least 3 since lower values have been found to be unstable numerically. Default: 30
        gauges : list of pairs (lat, lon), optional
            The locations of tide gauges where to measure temporal changes in sea level height.
            This is used mostly for validation purposes.
        sea_level : float or function, optional
            The sea level (above geoid) of the ocean at rest, used as a starting level for the
            surge simulation. Instead of a constant scalar value, a function can be specified that
            gets a `bounds` and a `period` argument and returns a scalar value. In this case, the
            first argument is a tuple of floats (lon_min, lat_min, lon_max, lat_max) and the
            second argument is a pair of np.datetime64 (start, end). For example, see the helper
            function `sea_level_from_nc` that reads the value from a NetCDF file. Default: 0
        outer_pad_deg : float, optional
            An additional padding (in degrees) around the model domain where the automatic mesh
            refinement is disabled to stabilize boundary interactions. If you find that your run of
            GeoClaw is numerically unstable, takes exceedingly long, or produces unrealistic
            results, it might help to modify this parameter by a few degrees. Default: 5
        boundary_conditions : str, optional
            One of "extrap" (extrapolation, non-reflecting outflow), "periodic", or "wall"
            (reflecting, solid wall boundary conditions). For more information about the possible
            settings, see the chapter "Boundary conditions" in the Clawpack documentation.
            Default: "extrap"
        output_freq_s : float, optional
            Frequency of writing GeoClaw output files (for debug use) in 1/seconds. No output
            files are written if the value is 0.0. Default: 0.0
        recompile : bool, optional
            If True, force the GeoClaw Fortran code to be recompiled. Note that, without
            recompilation, changes to environment variables like FC, FFLAGS or OMP_NUM_THREADS are
            ignored! Default: False
        """
        self.recompile = recompile

        gauges = [] if gauges is None else gauges

        if topo_res_as < 3:
            raise ValueError("Specify a topo resolution of at least 3 arc-seconds!")
        self.topo_resolution_as = [max(360, topo_res_as), max(120, topo_res_as), topo_res_as]

        LOGGER.info("Prepare GeoClaw to determine surge on %d centroids", centroids.shape[0])
        self.track = track
        self.areas = areas
        self.centroids = centroids
        self.time_offset = time_offset
        self.time_offset_str = _dt64_to_pydt(self.time_offset).strftime("%Y-%m-%d-%H")
        self.output_freq_s = output_freq_s
        self.outer_pad_deg = outer_pad_deg
        self.boundary_conditions = boundary_conditions
        self.topo_path = topo_path
        self.gauge_data = [
            {
                'location': g,
                'base_sea_level': 0,
                'topo_height': -32768.0,
                'time': [],
                'height_above_ground': [],
                'height_above_geoid': [],
                'amr_level': [],
                'in_domain': True,
            } for g in gauges
        ]
        self.sea_level_fun = sea_level
        if np.isscalar(sea_level):
            self.sea_level_fun = lambda bounds, period: sea_level
        self.surge_h = np.zeros(centroids.shape[0])

        # compute time horizon
        self.time_horizon = tuple([
            int((t - self.time_offset)  / np.timedelta64(1, 's'))
            for t in self.track["time"][[0, -1]]
        ])

        # create work directory
        self.work_dir = base_dir.joinpath(self.time_offset_str)
        if self.work_dir.exists():
            LOGGER.info("Resuming in GeoClaw working directory: %s", self.work_dir)
        else:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Init GeoClaw working directory: %s", self.work_dir)

        # write Makefile
        path = self.work_dir.joinpath("Makefile")
        if not path.exists():
            with path.open("w") as file_p:
                file_p.write(f"""\
CLAW = {clawpack_info()[0]}
CLAW_PKG = geoclaw
EXE = xgeoclaw
include $(CLAW)/geoclaw/src/2d/shallow/Makefile.geoclaw
SOURCES = $(CLAW)/riemann/src/rpn2_geoclaw.f \\
          $(CLAW)/riemann/src/rpt2_geoclaw.f \\
          $(CLAW)/riemann/src/geoclaw_riemann_utils.f
include $(CLAW)/clawutil/src/Makefile.common
""")
        path = self.work_dir.joinpath("setrun.py")
        if not path.exists():
            with path.open("w") as file_p:
                file_p.write("")

        self.write_rundata()


    def run(self) -> None:
        """Run GeoClaw script and set `surge_h` attribute."""
        self.stdout = ""
        self.stdout_printed = False
        if self.work_dir.joinpath("gc_terminated").exists():
            LOGGER.info("Skip running GeoClaw since it terminated previously ...")
            self.stdout = self.work_dir.joinpath("stdout.log").read_text()
        else:
            self._run_subprocess()
        LOGGER.info("Reading GeoClaw output ...")
        try:
            self.read_fgmax_data()
            self.read_gauge_data()
        except FileNotFoundError:
            self.print_stdout()
            LOGGER.warning("Reading GeoClaw output failed (see output above).")


    def _run_subprocess(self) -> None:
        LOGGER.info("Running GeoClaw in %s ...", self.work_dir)
        time_span = self.time_horizon[1] - self.time_horizon[0]
        perc = -100
        last_perc = -100
        stopped = False
        with subprocess.Popen(
            ["make"] + (["new"] if self.recompile else []) + [".output"],
            cwd=self.work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={
                "FFLAGS": "-O2 -fopenmp",
                "OMP_NUM_THREADS": str(psutil.cpu_count(logical=False)),
                **os.environ,
            },
        ) as proc:
            for line in proc.stdout:
                line = line.decode()
                self.stdout += line
                with self.work_dir.joinpath("stdout.log").open("a") as fp:
                    fp.write(line)
                line = line.rstrip()
                error_strings = [
                    "ABORTING CALCULATION",
                    "Stopping calculation",
                    "  free list full with ",
                ]
                if any(err in line for err in error_strings):
                    stopped = True
                re_m = re.match(r".*t = ([-ED0-9\.\+]+)$", line)
                if re_m is not None:
                    time = float(re_m.group(1).replace("D", "E"))
                    perc = 100 * (time - self.time_horizon[0]) / time_span
                    if perc - last_perc >= 10:
                        # for parallelized output, print the time offset each time
                        LOGGER.info("%s: %d%%", self.time_offset_str, perc)
                        last_perc = perc
        self.work_dir.joinpath("gc_terminated").write_text("True")
        if perc < 99.9:
            # sometimes, GeoClaw fails without a specific error output
            stopped = True
        elif int(last_perc) != 100:
            LOGGER.info("%s: 100%%", self.time_offset_str)
        if proc.returncode != 0 or stopped:
            self.print_stdout()
            raise RuntimeError("GeoClaw run failed (see output above).")


    def print_stdout(self) -> None:
        """"Print standard (and error) output of GeoClaw run."""
        if not self.stdout_printed:
            LOGGER.info("Output of 'make .output' in GeoClaw work directory:")
            print(self.stdout)
            # make sure to print at most once
            self.stdout_printed = True


    def read_fgmax_data(self) -> None:
        """Read fgmax output data from GeoClaw working directory."""
        # pylint: disable=import-outside-toplevel
        from clawpack.geoclaw import fgmax_tools
        outdir = self.work_dir.joinpath("_output")
        fg_path = outdir.joinpath("fgmax0001.txt")

        if not fg_path.exists():
            raise FileNotFoundError("GeoClaw quit without creating fgmax data!")

        fgmax_grid = fgmax_tools.FGmaxGrid()
        fg_fname = self.work_dir.joinpath("fgmax_grids.data")
        with contextlib.redirect_stdout(None):
            fgmax_grid.read_fgmax_grids_data(1, fg_fname)
            fgmax_grid.read_output(outdir=outdir)
        assert fgmax_grid.point_style == 0
        self.surge_h[:] = fgmax_grid.h
        self.surge_h[fgmax_grid.arrival_time.mask] = 0


    def read_gauge_data(self) -> None:
        """Read gauge output data from GeoClaw working directory."""
        # pylint: disable=import-outside-toplevel
        from clawpack.pyclaw.gauges import GaugeSolution
        outdir = self.work_dir.joinpath("_output")
        for i_gauge, gauge in enumerate(self.gauge_data):
            if not gauge['in_domain']:
                continue
            gauge['base_sea_level'] = self.rundata.geo_data.sea_level
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                # suppress warnings about empty gauge files (which is not a problem for us)
                g = GaugeSolution(gauge_id=i_gauge + 1, path=outdir)
            if g.t is None:
                continue
            gauge['time'] = self.time_offset + g.t * np.timedelta64(1, 's')
            mask_amr_max = (g.level == g.level.max())
            gauge['topo_height'] = (g.q[1, mask_amr_max] - g.q[0, mask_amr_max]).mean()
            gauge['height_above_ground'] = g.q[0, :]
            gauge['height_above_geoid'] = g.q[1, :]
            gauge["amr_level"] = g.level


    def write_rundata(self) -> None:
        """Create rundata config files in work directory or read if already existent."""
        if not self._read_rundata():
            self._set_rundata_claw()
            self._set_rundata_amr()
            self._set_rundata_geo()
            self._set_rundata_fgmax()
            self._set_rundata_storm()
            self._set_rundata_gauges()
            with contextlib.redirect_stdout(None):
                self.rundata.write(out_dir=self.work_dir)


    def _read_rundata(self) -> bool:
        """Read rundata object from files, return whether it was succesful

        Returns
        -------
        bool
        """
        # pylint: disable=import-outside-toplevel
        import clawpack.amrclaw.data
        import clawpack.geoclaw.data
        self._clear_rundata()
        for dataobject in self.rundata.data_list:
            if isinstance(dataobject, clawpack.geoclaw.data.FixedGridData):
                # ignore since it's deprecated, hence unused
                continue
            fname = inspect.signature(dataobject.write).parameters["out_file"].default
            path = self.work_dir / fname
            if not path.exists():
                self._clear_rundata()
                return False
            is_gauge_data = isinstance(dataobject, clawpack.amrclaw.data.GaugeData)
            read_args = [] if is_gauge_data else [path]
            read_kwargs = dict(data_path=self.work_dir) if is_gauge_data else {}
            with contextlib.redirect_stdout(None):
                dataobject.read(*read_args, **read_kwargs)

        # the "in_domain" gauge attribute is determined from the gaugedata settings
        gauge_nos = [g_no for g_no, _, _, _, _ in self.rundata.gaugedata.gauges]
        for i_gauge, gauge in enumerate(self.gauge_data):
            gauge["in_domain"] = i_gauge + 1 in gauge_nos

        # resume from checkpoint if it exists and the previous run didn't finish
        chk_files = list(self.work_dir.glob("_output/fort.chk*"))
        if len(chk_files) > 1 and not self.work_dir.joinpath("gc_terminated").exists():
            idx_by_mtimes = np.argsort([p.stat().st_mtime for p in chk_files])
            # the latest might be corrupt after kill during I/O; use the previous
            self.rundata.clawdata.restart_file = chk_files[idx_by_mtimes[-2]].name
            self.rundata.clawdata.restart = True
            self.rundata.clawdata.write(out_file=self.work_dir / "claw.data")
        return True


    def _clear_rundata(self) -> None:
        """Reset the rundata object to its initial, empty state"""
        import clawpack.clawutil.data
        self.rundata = clawpack.clawutil.data.ClawRunData(pkg="geoclaw", num_dim=2)


    def _set_rundata_claw(self) -> None:
        """Set the rundata parameters in the `clawdata` category."""
        clawdata = self.rundata.clawdata
        clawdata.verbosity = 1
        clawdata.checkpt_style = -3
        clawdata.checkpt_interval = 25
        clawdata.lower = [lim - self.outer_pad_deg for lim in self.areas['wind_area'][:2]]
        clawdata.upper = [lim + self.outer_pad_deg for lim in self.areas['wind_area'][2:]]
        clawdata.num_cells = [
            # coarsest resolution: appx. 0.25 degrees
            int(np.ceil((clawdata.upper[0] - clawdata.lower[0]) * 4)),
            int(np.ceil((clawdata.upper[1] - clawdata.lower[1]) * 4)),
        ]
        clawdata.num_eqn = 3
        clawdata.num_aux = 3 + 1 + 3
        clawdata.capa_index = 2
        clawdata.t0, clawdata.tfinal = self.time_horizon
        if self.output_freq_s > 0.0:
            clawdata.output_style = 1
            clawdata.num_output_times = int(
                (clawdata.tfinal - clawdata.t0) * self.output_freq_s
            )
            clawdata.output_t0 = True
            clawdata.output_format = 'binary64'
            clawdata.output_q_components = 'all'
            clawdata.output_aux_components = 'all'
            clawdata.output_aux_onlyonce = False
        else:
            clawdata.num_output_times = 0
            clawdata.output_t0 = False
        clawdata.dt_initial = 0.8 / max(clawdata.num_cells)
        clawdata.cfl_desired = 0.75
        clawdata.num_waves = 3
        clawdata.limiter = ['mc', 'mc', 'mc']
        clawdata.use_fwaves = True
        clawdata.source_split = 'godunov'
        clawdata.bc_lower = [self.boundary_conditions, self.boundary_conditions]
        clawdata.bc_upper = [self.boundary_conditions, self.boundary_conditions]


    def _set_rundata_amr(self) -> None:
        """Set AMR-related rundata attributes."""
        clawdata = self.rundata.clawdata
        amrdata = self.rundata.amrdata
        refinedata = self.rundata.refinement_data
        amrdata.refinement_ratios_x = self.compute_refinement_ratios()
        amrdata.refinement_ratios_y = amrdata.refinement_ratios_x
        amrdata.refinement_ratios_t = amrdata.refinement_ratios_x
        amrdata.amr_levels_max = len(amrdata.refinement_ratios_x) + 1
        amrdata.aux_type = ['center', 'capacity', 'yleft', 'center', 'center', 'center', 'center']
        amrdata.regrid_interval = 3
        amrdata.regrid_buffer_width = 2
        amrdata.verbosity_regrid = 0
        regions = self.rundata.regiondata.regions
        t_1, t_2 = clawdata.t0, clawdata.tfinal
        maxlevel = amrdata.amr_levels_max
        (x_1, y_1), (x_2, y_2) = clawdata.lower, clawdata.upper
        regions.append([1, 1, t_1, t_2, x_1, x_2, y_1, y_2])
        x_1, y_1, x_2, y_2 = self.areas['wind_area']
        regions.append([1, 4, t_1, t_2, x_1, x_2, y_1, y_2])
        x_1, y_1, x_2, y_2 = self.areas['landfall_area']
        regions.append([max(1, maxlevel - 3), maxlevel, t_1, t_2, x_1, x_2, y_1, y_2])
        for area in self.areas['surge_areas']:
            x_1, y_1, x_2, y_2 = area
            regions.append([maxlevel - 1, maxlevel, t_1, t_2, x_1, x_2, y_1, y_2])
        refinedata.speed_tolerance = list(np.arange(1.0, maxlevel - 2))
        refinedata.variable_dt_refinement_ratios = True
        refinedata.wave_tolerance = 1.0


    def compute_refinement_ratios(self) -> None:
        # select the refinement ratios so that:
        # * the last but one resolution is less than self.topo_resolution_as[-1]
        # * the list of ratios is non-decreasing
        # * not more than 6 ratios
        # * no single ratio larger than 8
        clawdata = self.rundata.clawdata
        base_res = (clawdata.upper[0] - clawdata.lower[0]) / clawdata.num_cells[0]
        total_fact = base_res / (self.topo_resolution_as[-1] / 3600)
        n_ratios = min(5, int(np.round(np.log2(total_fact))))
        if n_ratios < 2:
            ratios = [2]
        else:
            ratios = [2] * (n_ratios - 2)
            target = total_fact / np.prod(ratios)
            ratio1 = np.arange(
                max(2, np.ceil(target / 8)),
                max(2, min(np.sqrt(target), 8)) + 1,
            )
            ratio2 = np.fmax(ratio1, np.ceil(target / ratio1))
            i_ratio = np.argmin(ratio1 * ratio2)
            ratios += [int(ratio1[i_ratio]), int(ratio2[i_ratio])]
        ratios += [min(8, ratios[-1] + 1)]
        LOGGER.info("GeoClaw resolution in arc-seconds: %s",
                    str([f"{3600 * base_res / r:.2f}" for r in np.cumprod([1] + ratios)]))
        return ratios


    def _set_rundata_geo(self) -> None:
        """Set geo-related rundata attributes."""
        clawdata = self.rundata.clawdata
        frictiondata = self.rundata.friction_data
        geodata = self.rundata.geo_data
        topodata = self.rundata.topo_data

        # lat-lon coordinate system
        geodata.coordinate_system = 2

        # different friction on land and at sea
        geodata.friction_forcing = True
        frictiondata.variable_friction = True
        frictiondata.friction_regions.append([
            clawdata.lower, clawdata.upper, [np.infty, 0.0, -np.infty], [0.050, 0.025],
        ])
        geodata.dry_tolerance = 1.e-2

        # get sea level information for affected areas and time period
        tr_period = (self.track["time"].values[0], self.track["time"].values[-1])
        geodata.sea_level = np.mean([
            self.sea_level_fun(area, tr_period)
            for area in self.areas['surge_areas']
        ])

        # load elevation data, resolution depending on area of refinement
        topodata.topofiles = []
        areas = [
            tuple(clawdata.lower) + tuple(clawdata.upper),
            self.areas['landfall_area']
        ] + self.areas['surge_areas']
        resolutions = self.topo_resolution_as[:2]
        resolutions += [self.topo_resolution_as[2]] * len(self.areas['surge_areas'])
        dems_for_plot = []
        for res_as, bounds in zip(resolutions, areas):
            bounds, topo = _load_topography(self.topo_path, bounds, res_as)
            if 0 in topo.Z.shape:
                LOGGER.warning("Area is ignored because it is too small.")
                continue
            tt3_fname = 'topo_{}s_{}.tt3'.format(res_as, _bounds_to_str(bounds))
            tt3_fname = self.work_dir.joinpath(tt3_fname)
            topo.write(tt3_fname)
            topodata.topofiles.append([3, tt3_fname])
            dems_for_plot.append((bounds, topo.Z))
        plot_dems(
            dems_for_plot,
            track=self.track,
            # for debugging purposes, you might want to plot the centroids as scatter:
            # centroids=self.centroids,
            path=self.work_dir.joinpath("dems.pdf"),
        )


    def _set_rundata_fgmax(self) -> None:
        """Set monitoring-related rundata attributes."""
        # pylint: disable=import-outside-toplevel
        from clawpack.geoclaw import fgmax_tools

        # monitor max height values on centroids
        self.rundata.fgmax_data.num_fgmax_val = 1
        fgmax_grid = fgmax_tools.FGmaxGrid()
        fgmax_grid.point_style = 0
        fgmax_grid.tstart_max = self.rundata.clawdata.t0
        fgmax_grid.tend_max = self.rundata.clawdata.tfinal
        fgmax_grid.dt_check = 0
        fgmax_grid.min_level_check = self.rundata.amrdata.amr_levels_max - 1
        fgmax_grid.arrival_tol = 1.e-2
        fgmax_grid.npts = self.centroids.shape[0]
        fgmax_grid.X = self.centroids[:, 1]
        fgmax_grid.Y = self.centroids[:, 0]
        self.rundata.fgmax_data.fgmax_grids.append(fgmax_grid)


    def _set_rundata_storm(self) -> None:
        """Set storm-related rundata attributes."""
        surge_data = self.rundata.surge_data
        surge_data.wind_forcing = True
        surge_data.drag_law = 1
        surge_data.pressure_forcing = True
        surge_data.storm_specification_type = 'holland80'
        surge_data.storm_file = str(self.work_dir.joinpath("track.storm"))
        gc_storm = _climada_xarray_to_geoclaw_storm(
            self.track, offset=_dt64_to_pydt(self.time_offset),
        )
        gc_storm.write(surge_data.storm_file, file_format='geoclaw')


    def _set_rundata_gauges(self) -> None:
        """Set gauge-related rundata attributes."""
        clawdata = self.rundata.clawdata
        for i_gauge, gauge in enumerate(self.gauge_data):
            lat, lon = gauge['location']
            if (clawdata.lower[0] > lon or clawdata.lower[1] > lat
                or clawdata.upper[0] < lon or clawdata.upper[1] < lat):
                # skip gauges outside of model domain
                gauge['in_domain'] = False
                continue
            self.rundata.gaugedata.gauges.append(
                [i_gauge + 1, lon, lat, clawdata.t0, clawdata.tfinal]
            )
        # q[0]: height above topography (above ground, where ground might be sea floor)
        self.rundata.gaugedata.q_out_fields = [0]


def _climada_xarray_to_geoclaw_storm(
    track : xr.Dataset,
    offset : Optional[dt.datetime] = None,
) -> Any:
    """Convert CLIMADA's xarray TC track to GeoClaw storm object

    Parameters
    ----------
    track : xr.Dataset
        Single tropical cyclone track.
    offset : datetime
        Time zero for internal use in GeoClaw.

    Returns
    -------
    gc_storm : clawpack.geoclaw.surge.storm.Storm
    """
    # pylint: disable=import-outside-toplevel
    from clawpack.geoclaw.surge.storm import Storm
    gc_storm = Storm()
    gc_storm.t = _dt64_to_pydt(track["time"].values)
    if offset is not None:
        gc_storm.time_offset = offset
    gc_storm.eye_location = np.stack([track["lon"].values, track["lat"].values], axis=-1)
    gc_storm.max_wind_speed = track["max_sustained_wind"].values * KN_TO_MS
    gc_storm.max_wind_radius = track["radius_max_wind"].values * NM_TO_KM * 1000
    gc_storm.central_pressure = track["central_pressure"].values * MBAR_TO_PA
    gc_storm.storm_radius = track["radius_oci"].values * NM_TO_KM * 1000
    return gc_storm


def _load_topography(
    path : Union[pathlib.Path, str],
    bounds : Tuple[float, float, float, float],
    res_as : float,
) -> Tuple[Tuple[float, float, float, float], Any]:
    """Load topographical elevation data in specified bounds and resolution

    The bounds of the returned topodata are always larger than the requested bounds to make sure
    that the pixel centers still cover the requested region.

    Parameters
    ----------
    path : Path or str
        Path to raster file containing elevation data above reference geoid.
    bounds : tuple
        Bounds (lon_min, lat_min, lon_max, lat_max) of region of interest.
    res_as : float
        Resolution in arc-seconds.

    Returns
    -------
    bounds : tuple
        Bounds (lon_min, lat_min, lon_max, lat_max) actually covered by the returned topodata.
    topo : clawpack.geoclaw.topotools.Topography object
        The object's x, y and Z attributes contain the loaded topodata.
    """
    # pylint: disable=import-outside-toplevel
    from clawpack.geoclaw import topotools

    LOGGER.info("Load elevation data [%s, %s] from %s", res_as, bounds, path)
    res = res_as / (60 * 60)
    with rasterio.Env(VRT_SHARED_SOURCE=0):
        # without this env-setting, reading might crash in a multi-threaded environment:
        # https://gdal.org/drivers/raster/vrt.html#multi-threading-issues
        zvalues, transform = u_coord.read_raster_bounds(
            path, bounds, res=res, bands=[1], resampling="average", global_origin=(-180, 90),
        )
    zvalues = zvalues[0]
    xres, _, xmin, _, yres, ymin = transform[:6]
    xmax, ymax = xmin + zvalues.shape[1] * xres, ymin + zvalues.shape[0] * yres
    if xres < 0:
        zvalues = np.flip(zvalues, axis=1)
        xres, xmin, xmax = -xres, xmax, xmin
    if yres < 0:
        zvalues = np.flip(zvalues, axis=0)
        yres, ymin, ymax = -yres, ymax, ymin
    xmin, xmax = u_coord.lon_normalize(
        np.array([xmin, xmax]), center=0.5 * (bounds[0] + bounds[2]),
    )
    bounds = (xmin, ymin, xmax, ymax)
    xcoords = np.arange(xmin + xres / 2, xmax, xres)
    ycoords = np.arange(ymin + yres / 2, ymax, yres)

    nan_msk = np.isnan(zvalues)
    nan_count = nan_msk.sum()
    if nan_count > 0:
        LOGGER.warning(
            "Elevation data contains %d NaN values that are replaced with -1000!", nan_count,
        )
        zvalues[nan_msk] = -1000

    topo = topotools.Topography()
    topo.set_xyZ(xcoords, ycoords, zvalues.astype(np.float64))
    return bounds, topo


def _bounds_to_str(bounds : Tuple[float, float, float, float]) -> str:
    """Convert longitude/latitude bounds to a human-readable string

    Example
    -------
    >>> _bounds_to_str((-4.2, 1.0, -3.05, 2.125))
    '1N-2.125N_4.2W-3.05W'

    Parameters
    ----------
    bounds : tuple
        (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    string : str
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    return '{:.4g}{}-{:.4g}{}_{:.4g}{}-{:.4g}{}'.format(
        abs(lat_min), 'N' if lat_min >= 0 else 'S',
        abs(lat_max), 'N' if lat_max >= 0 else 'S',
        abs(lon_min), 'E' if lon_min >= 0 else 'W',
        abs(lon_max), 'E' if lon_max >= 0 else 'W',
    )


def _dt64_to_pydt(
    date : Union[np.datetime64, np.ndarray],
) -> Union[dt.datetime, List[dt.datetime]]:
    """Convert datetime64 value or array to python datetime object or list

    Parameters
    ----------
    date : np.datetime64 or array

    Returns
    -------
    dt : datetime or list of datetime objects
    """
    result = pd.DatetimeIndex(np.atleast_1d(date).ravel()).to_pydatetime()
    if isinstance(date, np.datetime64):
        return result[0]
    return list(result)
