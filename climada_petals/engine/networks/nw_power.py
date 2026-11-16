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

Input data for power networks

The power cascade (``GraphCalcs.powercap_from_clusters``) needs, on the nodes:

- ``el_generation`` on the power plants (``ci_type="power_plant"``);
- ``el_consumption`` on the people (``ci_type="people"``), 0 where people
  have no electricity access;
- optionally ``electrified`` on the people: ``True`` for the people cells
  with access to electricity, ``False`` for the others (see
  ``assign_electrification``). To link only the cells with access to the
  power network, set ``target_filter = "electrified"`` in the row
  ``power_line -> people`` of the dependency table; the cells without access
  then get no power link and no supply.

Typical workflow for a country::

    plants = load_power_plants_wri(path_wri_csv, iso3)
    rate = get_electricity_access_rate(iso3)  # or a value you know
    people["electrified"] = assign_electrification(people, rate, path_targets)
    total = load_final_consumption(path_consumption_csv, iso3)
    people, plants = assign_el_consumption_generation(people, plants, total)

Data sources:

- power plants: WRI Global Power Plant Database (CSV, one row per plant,
  columns ``country`` (ISO3), ``name``, ``latitude``, ``longitude``,
  ``estimated_generation_gwh_2017``);
- electricity access rate: World Bank indicator ``EG.ELC.ACCS.ZS``
  (share of the population with access to electricity, in %);
- electrified areas: an electrification-targets raster (e.g. the gridfinder
  "targets" layer, Arderne et al. 2020), non-zero cells are electrified;
- total electricity consumption: a CSV with the columns ``ISO3`` and
  ``el_consumption`` (e.g. IEA "Total final consumption, electricity").
"""

import logging
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import requests
from shapely.geometry import box

from climada_petals.engine.networks.nw_utils import POP_COL

LOGGER = logging.getLogger(__name__)

WB_ACCESS_INDICATOR = "EG.ELC.ACCS.ZS"
"""World Bank indicator: access to electricity (% of population)."""

WB_API_URL = (
    "https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}"
    "?mrnev=1&format=json"
)


def load_power_plants_wri(path, iso3, generation_col="estimated_generation_gwh_2017"):
    """Power plants of a country from the WRI Global Power Plant Database

    Parameters
    ----------
    path : str or Path
        CSV file of the database.
    iso3 : str
        ISO3 code of the country (column ``country``).
    generation_col : str, optional
        Column with the generation of each plant, kept in the result.
        Default is ``"estimated_generation_gwh_2017"``.

    Returns
    -------
    gpd.GeoDataFrame
        One point per plant (EPSG:4326) with the columns ``name`` and
        ``generation_col`` (numeric, NaN where missing). Empty if the country
        has no plant in the database.
    """
    plants = pd.read_csv(path, low_memory=False)
    plants = plants.loc[plants["country"] == iso3].reset_index(drop=True)
    return gpd.GeoDataFrame(
        {
            "name": plants["name"],
            generation_col: pd.to_numeric(plants[generation_col], errors="coerce"),
        },
        geometry=gpd.points_from_xy(plants["longitude"], plants["latitude"]),
        crs="EPSG:4326",
    )


def get_electricity_access_rate(iso3, timeout=30, retries=3, wait=5):
    """Share of the population with access to electricity (World Bank)

    Queries the most recent value of the World Bank indicator
    ``EG.ELC.ACCS.ZS``. Needs an internet connection; to work offline, pass
    a known rate to ``assign_electrification`` instead.

    Parameters
    ----------
    iso3 : str
        ISO3 code of the country.
    timeout : float, optional
        Timeout of each request in seconds. Default is 30.
    retries : int, optional
        Number of attempts. Default is 3.
    wait : float, optional
        Seconds between attempts. Default is 5.

    Returns
    -------
    float
        Access rate in % (0 to 100).

    Raises
    ------
    ValueError
        If the World Bank has no value for the country.
    RuntimeError
        If the request fails after all attempts.
    """
    url = WB_API_URL.format(iso3=iso3.lower(), indicator=WB_ACCESS_INDICATOR)
    error = None
    for attempt in range(retries):
        if attempt > 0:
            time.sleep(wait)
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            content = response.json()
        except (requests.RequestException, ValueError) as err:
            error = err
            LOGGER.info("World Bank query failed (attempt %i): %s", attempt + 1, err)
            continue
        records = content[1] if len(content) > 1 and content[1] else []
        if not records or records[0].get("value") is None:
            raise ValueError(
                f"The World Bank has no electricity access rate for '{iso3}'. "
                "Pass a known rate to assign_electrification instead."
            )
        return float(records[0]["value"])
    raise RuntimeError(
        f"Could not get the electricity access rate of '{iso3}' from the World "
        f"Bank ({error}). Pass a known rate to assign_electrification instead."
    )


def _load_electrification_targets(path, geometry):
    """Electrified areas from a targets raster, within ``geometry``

    Returns
    -------
    gpd.GeoDataFrame
        One disc per non-zero cell, centred on the cell, with a radius of
        half a cell, in the CRS of the raster.
    float
        Cell size in the units of the raster CRS.
    """
    with rasterio.open(path) as src:
        crs, res, nodata = src.crs, abs(src.res[0]), src.nodata
        try:
            data, transform = rasterio.mask.mask(src, [geometry], crop=True)
        except ValueError:  # geometry does not overlap the raster
            return gpd.GeoDataFrame(geometry=[], crs=crs), res
    values = data[0].astype(float)
    targets = (values != 0) & ~np.isnan(values)
    if nodata is not None:
        targets &= values != nodata
    rows, cols = np.nonzero(targets)
    x, y = rasterio.transform.xy(transform, rows, cols)
    discs = gpd.points_from_xy(np.atleast_1d(x), np.atleast_1d(y)).buffer(res / 2)
    return gpd.GeoDataFrame(geometry=discs, crs=crs), res


def assign_electrification(
    gdf_people,
    el_rate,
    targets_path=None,
    geometry=None,
    full_access_rate=90,
    buffer_step=0.01,
    max_iter=10,
    pop_col=POP_COL,
):
    """Which people cells have access to electricity

    - If ``el_rate`` is at least ``full_access_rate``, all cells are
      electrified.
    - Otherwise, the cells inside the electrified areas of the targets raster
      are electrified. While the electrified share of the population is below
      ``el_rate``, the electrified areas are widened by ``buffer_step`` and
      the cells are assigned again, at most ``max_iter`` times. The share
      can then end up above ``el_rate``; if it is still below after
      ``max_iter`` widenings, a warning is logged.

    Parameters
    ----------
    gdf_people : gpd.GeoDataFrame
        People cells (points) with the population in ``pop_col``.
    el_rate : float
        Share of the population with access to electricity, in % (e.g. from
        ``get_electricity_access_rate``).
    targets_path : str or Path, optional
        Electrification-targets raster (non-zero cells are electrified).
        Required if ``el_rate < full_access_rate``.
    geometry : shapely geometry, optional
        Area of the raster to read, in the raster CRS (e.g. the country
        shape). Default: the extent of ``gdf_people``, enlarged by the largest
        widening and one cell.
    full_access_rate : float, optional
        Rate (%) from which all cells are electrified. Default is 90.
    buffer_step : float, optional
        Widening per iteration, in the units of the raster CRS (degrees for
        EPSG:4326: 0.01 is about 1.1 km at the equator). Default is 0.01.
    max_iter : int, optional
        Maximum number of widenings. Default is 10.
    pop_col : str, optional
        Population column. Default is ``POP_COL`` (``"counts"``).

    Returns
    -------
    pd.Series
        Boolean, indexed like ``gdf_people``: ``True`` for electrified cells.

    Raises
    ------
    ValueError
        If ``el_rate`` is not between 0 and 100, or if ``targets_path`` is
        missing although it is needed.
    """
    if not 0 <= el_rate <= 100:
        raise ValueError(f"el_rate must be a percentage (0 to 100), got {el_rate}.")
    if el_rate >= full_access_rate:
        return pd.Series(True, index=gdf_people.index, name="electrified")
    if targets_path is None:
        raise ValueError(
            f"An electrification-targets raster (targets_path) is needed for an "
            f"access rate below {full_access_rate} % (got {el_rate} %)."
        )

    with rasterio.open(targets_path) as src:
        raster_crs, res = src.crs, abs(src.res[0])
    points = gdf_people[["geometry"]].to_crs(raster_crs)
    if geometry is None:
        geometry = box(*points.total_bounds).buffer(buffer_step * max_iter + res)
    targets, _ = _load_electrification_targets(targets_path, geometry)

    population = gdf_people[pop_col].to_numpy(dtype=float)
    total = population.sum()

    def electrified_cells(areas):
        joined = gpd.sjoin(points, areas, how="inner", predicate="within")
        return points.index.isin(joined.index)

    electrified = electrified_cells(targets)
    share = 100 * population[electrified].sum() / total if total else 100.0
    n_iter = 0
    while share < el_rate and n_iter < max_iter:
        targets = gpd.GeoDataFrame(geometry=targets.buffer(buffer_step), crs=raster_crs)
        electrified = electrified_cells(targets)
        share = 100 * population[electrified].sum() / total
        n_iter += 1
    if share < el_rate:
        LOGGER.warning(
            "Electrified share of the population %.1f %% below the access rate "
            "%.1f %% after %i widenings of the electrified areas.",
            share,
            el_rate,
            max_iter,
        )
    return pd.Series(electrified, index=gdf_people.index, name="electrified")


def load_final_consumption(path, iso3):
    """Total electricity consumption of a country from a CSV file

    Parameters
    ----------
    path : str or Path
        CSV file with the columns ``ISO3`` and ``el_consumption`` (e.g. IEA
        "Total final consumption, electricity").
    iso3 : str
        ISO3 code of the country.

    Returns
    -------
    float
        Consumption in the unit of the file, NaN if the country is missing.
    """
    table = pd.read_csv(path)
    values = table.loc[table["ISO3"] == iso3, "el_consumption"]
    return float(values.iloc[0]) if len(values) else np.nan


def assign_el_consumption_generation(
    gdf_people,
    gdf_plants,
    total_consumption=None,
    electrified_col="electrified",
    generation_col="estimated_generation_gwh_2017",
    pop_col=POP_COL,
):
    """Electricity consumption of people cells and generation of power plants

    Total generation equals total consumption, so that a power network with
    all plants and people in one cluster is fully supplied, and the supply /
    demand ratio of a cluster is its share of generation over its share of
    consumption.

    - Consumption (``el_consumption``): ``total_consumption`` distributed over
      the electrified cells in proportion to their population; 0 for the
      other cells. Without ``total_consumption`` (``None`` or NaN), each
      electrified cell consumes its population.
    - Generation (``el_generation``): the same total distributed over the
      plants in proportion to ``generation_col`` (plants without a value get
      the mean of the others). Without ``total_consumption``, or if no plant
      has a positive value in ``generation_col``, it is distributed equally.

    Parameters
    ----------
    gdf_people : gpd.GeoDataFrame
        People cells with the columns ``pop_col`` and ``electrified_col``.
    gdf_plants : gpd.GeoDataFrame
        Power plants (e.g. from ``load_power_plants_wri``).
    total_consumption : float, optional
        Total electricity consumption of the area (e.g. from
        ``load_final_consumption``), in any unit.
    electrified_col : str, optional
        Boolean column of the people cells (e.g. from
        ``assign_electrification``). Default is ``"electrified"``.
    generation_col : str, optional
        Column with the generation of each plant, used as weights. Default is
        ``"estimated_generation_gwh_2017"``.
    pop_col : str, optional
        Population column. Default is ``POP_COL`` (``"counts"``).

    Returns
    -------
    gdf_people, gdf_plants : gpd.GeoDataFrame
        Copies with the new columns ``el_consumption`` and ``el_generation``.

    Raises
    ------
    ValueError
        If there is no power plant or no electrified people cell.
    """
    if gdf_plants.empty:
        raise ValueError("No power plant: cannot assign electricity generation.")
    electrified = gdf_people[electrified_col].fillna(False).astype(bool).to_numpy()
    if not electrified.any():
        raise ValueError("No electrified people cell: no electricity consumption.")
    people = gdf_people.copy()
    plants = gdf_plants.copy()

    population = people[pop_col].to_numpy(dtype=float)
    if total_consumption is None or np.isnan(total_consumption):
        consumption = np.where(electrified, population, 0.0)
        total = consumption.sum()
        weights = np.ones(len(plants))
    else:
        total = float(total_consumption)
        consumption = np.where(
            electrified, population / population[electrified].sum() * total, 0.0
        )
        weights = np.ones(len(plants))
        if generation_col in plants.columns:
            generation = pd.to_numeric(plants[generation_col], errors="coerce")
            if generation.notna().any() and generation.sum() > 0:
                weights = generation.fillna(generation.mean()).to_numpy(dtype=float)
    people["el_consumption"] = consumption
    plants["el_generation"] = weights / weights.sum() * total
    return people, plants
