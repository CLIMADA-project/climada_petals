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
Forecast skill utility functions for seasonal forecast analysis.

This module provides functionality to:
- Download precomputed seasonal forecast skill metrics (e.g., MSESS) from Zenodo
- Organize the data in CLIMADA Copernicus Seasonal Forecast Module's structured 
seasonal forecast directories
- Plot spatial skill maps for selected initiation months and climate indices

Currently, plotting is supported only for the 'Tmax' index (tasmax-based skill files).

"""

import os
import logging
from pathlib import Path
import requests
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from climada import CONFIG

# % pip install zenodo-get

# Initialize logger
LOGGER = logging.getLogger(__name__)

##########  Forescast Sills (experimental)  ##########


def download_forecast_skills(index_metric="Tmax", initiation_months=None):
    """
    Download seasonal forecast skill NetCDF files from Zenodo for a given climate index
    and selected initiation months.

    Files are downloaded based on standardized Zenodo naming patterns and saved into the
    CLIMADA seasonal forecast data directory under a subfolder `skills/<index_metric>`.

    Parameters
    ----------
    index_metric : str
        Climate index name (e.g., 'Tmax', 'Tmean', 'HW'). This determines the variable prefix
        used in the Zenodo filenames (e.g., 'tasmax', 'tmean', etc.).

    initiation_months : list of str, optional
        List of 2-digit month strings (e.g., ['03', '04', '05']) indicating which SHC months
        to download. Defaults to all months ['01'–'12'] if not provided.

    Returns
    -------
    list of Path or str
        A list containing the local file paths to downloaded files, or a message string if
        the download failed or file already exists.
    """

    # Map index metric to file prefix used on Zenodo
    zenodo_prefix_map = {
        "Tmax": "tasmax",
        "Tmean": "tmean",
        "Tmin": "tasmin",
        "RH": "rh",
        "TX30": "tx30",
        "TR": "tr",
        "HW": "hw",
        "WSD": "wsd",
        "WSDI": "wsdi",
        "WSS": "wss",
        "LPR": "lpr",
    }
    file_prefix = zenodo_prefix_map.get(index_metric, index_metric.lower())

    if initiation_months is None:
        initiation_months = [f"{i:02d}" for i in range(1, 13)]  # Default to all months

    zenodo_base_url = "https://zenodo.org/records/14103378/files"
    file_prefix_full = f"{file_prefix}MSESS_subyr_gcfs21_shc"
    file_suffix = "-climatology_r1i1p1_1990-2019.nc"

    data_out = Path(CONFIG.hazard.copernicus.seasonal_forecasts.dir())
    skill_path = (
        data_out / "skills" / index_metric
    )  # <-- save under index name (e.g. 'Tmax')
    skill_path.mkdir(parents=True, exist_ok=True)

    results = []

    for month in initiation_months:
        file_name = f"{file_prefix_full}{month}{file_suffix}"
        url = f"{zenodo_base_url}/{file_name}?download=1"
        output_path = skill_path / file_name

        if output_path.exists():
            results.append(output_path)
            continue

        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
            results.append(output_path)
        except Exception as err:
            LOGGER.warning("Failed to download %s: %s", file_name, err)
            results.append(f"Failed: {file_name}")

    return results


def plot_forecast_skills(bounds, bounds_str, index_metric, init_months):
    """
    Plot forecast skill metrics for the given climate index and initiation months
    using pre-downloaded Zenodo NetCDF files.

    The plots show spatial patterns of forecast skill metrics (MSE, MSESS) restricted
    to the specified bounding box. Skill data must exist in the corresponding
    `skills/<index_metric>` directory.

    Parameters
    ----------
    bounds : list of float
        Bounding box [west, south, east, north] defining the spatial extent to plot.

    bounds_str : str
        Label or identifier for the region used in plot titles.

    index_metric : str
        Climate index name (currently only 'Tmax' supported for plotting).

    init_months : list of str
        List of 2-digit month strings (e.g., ['03', '04']) corresponding to the SHC
        initiation months whose skill maps should be plotted.

    Returns
    -------
    None
        Plots are rendered interactively using matplotlib and cartopy.
    """
    if index_metric.lower() != "tmax":
        raise ValueError("Forecast skills are only available for the 'Tmax' index.")

    base_dir = (
        Path(CONFIG.hazard.copernicus.seasonal_forecasts.dir()) / "skills" / "tmax"
    )
    file_name_pattern = (
        "tasmaxMSESS_subyr_gcfs21_shc{month}-climatology_r1i1p1_1990-2019.nc"
    )

    for month_str in init_months:
        file_path = base_dir / file_name_pattern.format(month=month_str)

        if not file_path.exists():
            LOGGER.warning(
                "Skill data file for month %s not found: %s", month_str, file_path
            )
            continue

        try:
            with xr.open_dataset(file_path) as input_dataset:
                west, south, east, north = bounds
                subset_ds = input_dataset.sel(
                    lon=slice(west, east), lat=slice(north, south)
                )

                variables = [
                    "tasmax_fc_mse",
                    "tasmax_ref_mse",
                    "tasmax_msess",
                    "tasmax_msessSig",
                ]

                for var in variables:
                    if var in subset_ds:
                        plt.figure(figsize=(10, 8))
                        plot_axis = plt.axes(projection=ccrs.PlateCarree())

                        vmin = subset_ds[var].quantile(0.05).item()
                        vmax = subset_ds[var].quantile(0.95).item()

                        handle = (
                            subset_ds[var]
                            .isel(time=0)
                            .plot(
                                ax=plot_axis,
                                cmap="coolwarm",
                                vmin=vmin,
                                vmax=vmax,
                                add_colorbar=False,
                            )
                        )

                        cbar = plt.colorbar(
                            handle, ax=plot_axis, orientation="vertical", pad=0.1, shrink=0.7
                        )
                        cbar.set_label(var, fontsize=10)

                        plot_axis.set_extent(
                            [west, east, south, north], crs=ccrs.PlateCarree()
                        )
                        plot_axis.add_feature(cfeature.BORDERS, linestyle=":")
                        plot_axis.add_feature(cfeature.COASTLINE)
                        plot_axis.add_feature(cfeature.LAND, edgecolor="black", alpha=0.3)
                        plot_axis.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

                        plt.title(f"{var} for month {month_str}, {bounds_str}")
                        plt.show()
                    else:
                        LOGGER.warning(
                            "Variable %s not found in dataset for month %s.",
                            var,
                            month_str,
                        )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load or process data for month {month_str}: {error}"
            ) from error
