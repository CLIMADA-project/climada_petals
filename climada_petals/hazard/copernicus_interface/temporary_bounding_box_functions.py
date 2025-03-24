# intermediate util file until below functions are part of climada.util.coordinates.py

import numpy as np
from climada.util.coordinates import get_country_geometries, latlon_bounds
from shapely import Polygon


def bounding_box_global():
    """
    Return global bounds in EPSG 4326

    Returns
    -------
    tuple:
        The global bounding box as (min_lon, min_lat, max_lon, max_lat)
    """
    return (-180, -90, 180, 90)


def bounding_box_from_countries(country_names, buffer=1.0):
    """
    Return bounding box in EPSG 4326 containing given countries.

    Parameters
    ----------
    country_names : list or str
        list with ISO 3166 alpha-3 codes of countries, e.g ['ZWE', 'GBR', 'VNM', 'UZB']
    buffer : float, optional
        Buffer to add to both sides of the bounding box. Default: 1.0.

    Returns
    -------
    tuple
        The bounding box containing all given coutries as (min_lon, min_lat, max_lon, max_lat)
    """

    country_geometry = get_country_geometries(country_names).geometry
    longitudes, latitudes = [], []

    for multipolygon in country_geometry:
        if isinstance(multipolygon, Polygon):  # Single polygon case
            for (
                coord
            ) in multipolygon.exterior.coords:  # From 'polygon' to 'multipolygon'
                longitudes.append(coord[0])
                latitudes.append(coord[1])
        else:  # MultiPolygon case
            for polygon in multipolygon.geoms:
                for coord in polygon.exterior.coords:
                    longitudes.append(coord[0])
                    latitudes.append(coord[1])

    return latlon_bounds(np.array(latitudes), np.array(longitudes), buffer=buffer)


def bounding_box_from_cardinal_bounds(*, northern, eastern, western, southern):
    """
    Return and normalize bounding box in EPSG 4326 from given cardinal bounds.

    Parameters
    ----------
    northern : (int, float)
        Northern boundary of bounding box
    eastern : (int, float)
        Eastern boundary of bounding box
    western : (int, float)
        Western boundary of bounding box
    southern : (int, float)
        Southern boundary of bounding box

    Returns
    -------
    tuple
        The resulting normalized bounding box (min_lon, min_lat, max_lon, max_lat) with -180 <= min_lon < max_lon < 540

    """

    # latitude bounds check
    if not ((90 >= northern > southern >= -90)):
        raise ValueError(
            "Given northern bound is below given southern bound or out of bounds"
        )

    eastern = (eastern + 180) % 360 - 180
    western = (western + 180) % 360 - 180

    # Ensure eastern > western
    if western > eastern:
        eastern += 360

    return (western, southern, eastern, northern)
