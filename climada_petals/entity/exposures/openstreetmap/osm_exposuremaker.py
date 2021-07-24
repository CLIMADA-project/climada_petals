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

Obtain data from OpenStreetMap.
"""

import logging
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import fiona
from fiona.crs import from_epsg
import geopandas
from geopandas import GeoDataFrame
from shapely.geometry import  mapping, shape
from shapely import geometry
from shapely.ops import transform, nearest_points
import pyproj

from climada.entity import Exposures
from climada.entity import LitPop

LOGGER = logging.getLogger(__name__)


def get_highValueArea(bbox, save_path=None, Low_Value_gdf=None, check_plot=1):
    """
    In case low-value features were queried with get_features_OSM(),
    calculate the "counter-shape" representig high value area for a given bounding box.

    Parameters:
        bbox (array): List of coordinates in format [South, West, North, East]
        save_path (str or pathlib.Path): path for results
        Low_Value_gdf (str): absolute path of gdf of low value items which is to be inverted.
          If left empty, searches for OSM_features_gdf_combined_lat_lon.shp in save_path.
        checkplot

    Returns:
        High_Value_Area (gdf): GeoDataFrame of High Value Area as High_Value_Area_lat_lon

    Example:
        High_Value_gdf_47_8 = get_highValueArea([47.16, 8.0, 47.3, 8.0712], save_path = save_path,\
                                    Low_Value_gdf = save_path+'/Low_Value_gdf_combined_47_8.shp')
    important: Use same bbox and save_path as for get_features_OSM().
    """
    if save_path is None:
        save_path = Path.cwd()
    elif isinstance(save_path, str):
        save_path = Path(save_path)

    Outer_Poly = geometry.Polygon([(bbox[1], bbox[2]), (bbox[1], bbox[0]),
                                   (bbox[3], bbox[0]), (bbox[3], bbox[2])])

    if Low_Value_gdf is None:
        filepath = save_path.joinpath(
            'OSM_features_gdf_combined_' + str(int(bbox[0])) + '_' + str(int(bbox[1])) + '.shp')
        try:
            Low_Value_gdf = geopandas.read_file(filepath)
        except Exception as err:
            raise type(err)(f'No Low-Value-Union found with name {filepath}: ' + str(err)) from err
    else:
        Low_Value_gdf = geopandas.read_file(Low_Value_gdf)

    # Making one Union of individual shapes in gdfs
    Low_Value_Union = _makeUnion(Low_Value_gdf)

    # subtract low-value areas from high-value polygon
    High_Value_Area = Outer_Poly.difference(Low_Value_Union)

    # save high value multipolygon as shapefile and re-read as gdf:
    schema = {'geometry': 'MultiPolygon', 'properties': {'Name': 'str:80'}}
    shapeout = save_path.joinpath('High_Value_Area_' + str(int(bbox[0]))
                + '_' + str(int(bbox[1])) + ".shp")
    with fiona.open(shapeout, 'w', crs=from_epsg(4326), driver='ESRI Shapefile',
                    schema=schema) as output:
        prop1 = {'Name': 'High Value Area'}
        geom = mapping(High_Value_Area)
        output.write({'geometry': geom, 'properties': prop1})

    High_Value_Area = geopandas.read_file(shapeout)

    # plot
    if check_plot == 1:
        f, ax = plt.subplots(1)
        ax = High_Value_Area.plot(ax=ax)
        f.suptitle('High Value Area ' + str(int(bbox[0])) + ' ' + str(int(bbox[1])))
        plt.show()
        f.savefig('High Value Area ' + str(int(bbox[0])) + '_' + str(int(bbox[1])) +
                  '.pdf', bbox_inches='tight')

    return High_Value_Area

def _get_litpop_bbox(country, highValueArea, **kwargs):
    """get litpop exposure for the bbox area of the queried OSM features
    Parameters:
        country (str)
        highValueArea (GeoDataFrame)
        bbox (array)
        kwargs (dict): arguments for LitPop set_country method
    Returns:
        exp_sub (LitPop)
    """
    # Load LitPop Exposure for whole country, and High Value Area
    exp = LitPop()
    exp.set_country(country, **kwargs)
    exp.set_geometry_points()

    # Crop bbox of High Value Area from Country Exposure
    exp.gdf = exp.gdf.cx[min(highValueArea.bounds.minx):max(highValueArea.bounds.maxx),
                         min(highValueArea.bounds.miny):max(highValueArea.bounds.maxy)]

    return exp

def _split_exposure_highlow(exp_sub, mode, High_Value_Area_gdf):
    """divide litpop exposure into high-value exposure and low-value exposure
    according to area queried in OSM, re-assign all low values to high-value centroids
    Parameters:
        exp_sub (Exposures)
        mode (str)
        High_Value_Area_gdf (GeoDataFrame)
    Returns:
        exp_sub_high (Exposures)
    """

    exp_sub_high = pd.DataFrame(columns=exp_sub.gdf.columns)
    exp_sub_low = pd.DataFrame(columns=exp_sub.gdf.columns)
    for i, pt in enumerate(exp_sub.gdf.geometry):
        if pt.within(High_Value_Area_gdf.loc[0]['geometry']):
            exp_sub_high = exp_sub_high.append(exp_sub.gdf.iloc[i])
        else:
            exp_sub_low = exp_sub_low.append(exp_sub.gdf.iloc[i])

    exp_sub_high = GeoDataFrame(exp_sub_high, crs=exp_sub.crs, geometry=exp_sub_high.geometry)
    exp_sub_low = GeoDataFrame(exp_sub_low, crs=exp_sub.crs, geometry=exp_sub_low.geometry)

    if mode == "nearest":
        # assign asset values of low-value points to nearest point in high-value df.
        points_to_assign = exp_sub_high.geometry.unary_union
        exp_sub_high["addedValNN"] = 0
        for i in range(0, len(exp_sub_low)):
            nearest = exp_sub_high.geometry == nearest_points(exp_sub_low.iloc[i].geometry,
                                                              points_to_assign)[1]  # point
            exp_sub_high.addedValNN.loc[nearest] += exp_sub_low.iloc[i].value
        exp_sub_high["combinedValNN"] = exp_sub_high[['addedValNN', 'value']].sum(axis=1)
        exp_sub_high.rename(columns={'value': 'value_old', 'combinedValNN': 'value'},
                            inplace=True)

    elif mode == "even":
        # assign asset values of low-value points evenly to points in high-value df.
        exp_sub_high['addedValeven'] = sum(exp_sub_low.value) / len(exp_sub_high)
        exp_sub_high["combinedValeven"] = exp_sub_high[['addedValeven', 'value']].sum(axis=1)
        exp_sub_high.rename(columns={'value': 'value_old', 'combinedValeven': 'value'},
                            inplace=True)

    elif mode == "proportional":
        # assign asset values of low-value points proportionally
        # to value of points in high-value df.
        exp_sub_high['addedValprop'] = 0
        for i in range(0, len(exp_sub_high)):
            asset_factor = exp_sub_high.iloc[i].value / sum(exp_sub_high.value)
            exp_sub_high.addedValprop.iloc[i] = asset_factor * sum(exp_sub_low.value)
        exp_sub_high["combinedValprop"] = exp_sub_high[['addedValprop', 'value']].sum(axis=1)
        exp_sub_high.rename(columns={'value': 'value_old', 'combinedValprop': 'value'},
                            inplace=True)

    else:
        print("No proper re-assignment mode set. "
              "Please choose either 'nearest', 'even' or 'proportional'.")

    exp = exp_sub.copy(deep=False)
    exp.set_gdf(exp_sub_high)
    return exp

def get_osmstencil_litpop(bbox, country, mode, highValueArea=None,
                          save_path=None, check_plot=1, **kwargs):
    """
    Generate climada-compatible exposure by downloading LitPop exposure for a bounding box,
    corrected for centroids which lie inside a certain high-value multipolygon area
    from previous OSM query.

    Parameters:
        bbox (array): List of coordinates in format [South, West, North, East]
        Country (str): ISO3 code or name of country in which bbox is located
        highValueArea (str): path of gdf of high-value area from previous step.
          If empty, searches for cwd/High_Value_Area_lat_lon.shp
        mode (str): mode of re-assigning low-value points to high-value points.
          "nearest", "even", or "proportional"
        kwargs (dict): arguments for LitPop set_country method

    Returns:
        exp_sub_high_exp (Exposure): (CLIMADA-compatible) with re-allocated asset
          values with name exposure_high_lat_lon

    Example:
        exposure_high_47_8 = get_osmstencil_litpop([47.16, 8.0, 47.3, 8.0712],\
                          'CHE',"proportional", highValueArea = \
                          save_path + '/High_Value_Area_47_8.shp' ,\
                          save_path = save_path)
    """
    if save_path is None:
        save_path = Path.cwd()
    elif isinstance(save_path, str):
        save_path = Path(save_path)

    if highValueArea is None:
        try:
            filepath = str(Path.cwd().joinpath(
                'High_Value_Area_' + str(int(bbox[0])) + '_' + str(int(bbox[1])) + ".shp"))
            High_Value_Area_gdf = geopandas.read_file(filepath)
        except Exception as err:
            raise type(err)(f'No file found of form {filepath}. '
                            'Please add or specify path: ' + str(err)) from err
    else:
        High_Value_Area_gdf = geopandas.read_file(highValueArea)

    exp_sub = _get_litpop_bbox(country, High_Value_Area_gdf, **kwargs)

    exp_sub_high_exp = _split_exposure_highlow(exp_sub, mode, High_Value_Area_gdf)
    exp_sub_high_exp.check()

    # save as hdf5 file:
    exp_sub_high_exp.write_hdf5(save_path.joinpath('exposure_high_' + str(int(bbox[0])) +
                                '_' + str(int(bbox[1])) + '.h5'))
    # plotting
    if check_plot == 1:
        # normal hexagons
        exp_sub_high_exp.plot_hexbin(pop_name=True)
        # select the OSM background image from the available ctx.sources - doesnt work atm
        #fig, ax = exp_sub_high_exp.plot_basemap(buffer=30000, url=ctx.sources.OSM_C, cmap='brg')

    return exp_sub_high_exp

def _get_midpoints(highValueArea):
    """get midpoints from polygon and multipolygon shapes for current CLIMADA-
    exposure compatibility (centroids / points)

    Parameters:
        highValueArea (gdf)

    Returns:
        High_Value_Area_gdf
    """
    High_Value_Area_gdf = geopandas.read_file(highValueArea)

    # For current exposure structure, simply get centroid
    # and area (in m2) for each building polygon
    High_Value_Area_gdf['projected_area'] = 0
    High_Value_Area_gdf['Midpoint'] = 0

    for index in High_Value_Area_gdf.index:
        High_Value_Area_gdf.loc[index, "Midpoint"] = \
        High_Value_Area_gdf.loc[index, "geometry"].centroid.wkt
        s = shape(High_Value_Area_gdf.loc[index, "geometry"])
        # turn warnings off, otherwise Future and Deprecation warnings are flooding the logs
        logging.captureWarnings(True)
        proj = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
                       pyproj.Proj(init='epsg:3857'))
        High_Value_Area_gdf.loc[index, "projected_area"] = transform(proj, s).area
        # turn warnings on again
        logging.captureWarnings(False)

    # change active geometry from polygons to midpoints
    from shapely.wkt import loads
    High_Value_Area_gdf = High_Value_Area_gdf.rename(columns={'geometry': 'geo_polys',
                                                              'Midpoint': 'geometry'})
    High_Value_Area_gdf['geometry'] = High_Value_Area_gdf['geometry'].apply(lambda x: loads(x))
    High_Value_Area_gdf = High_Value_Area_gdf.set_geometry('geometry')

    return High_Value_Area_gdf

def _assign_values_exposure(High_Value_Area_gdf, mode, country, **kwargs):
    """add value-columns to high-resolution exposure gdf
    according to m2 area of underlying features.

    Parameters
    ----------
    High_Value_Area_gdf : GeoDataFrame
         high-resolution exposure gdf
    mode : str
        'LitPop' or 'default'
    country : str
        country alpha-3 code
    kwargs : dict
        arguments for LitPop set_country method

    Returns
    -------
    GeoDataFrame
        High_Value_Area_gdf, the transformed input dataframe
    """

    if mode == "LitPop":
        # assign LitPop values of this area to houses.
        exp_sub = _get_litpop_bbox(country, High_Value_Area_gdf, **kwargs)
        totalValue = sum(exp_sub.gdf.value)
        totalArea = sum(High_Value_Area_gdf['projected_area'])
        High_Value_Area_gdf['value'] = 0
        for index in High_Value_Area_gdf.index:
            High_Value_Area_gdf.loc[index, 'value'] = \
            High_Value_Area_gdf.loc[index, 'projected_area'] / totalArea * totalValue

    elif mode == "default":  # 5400 Chf / m2 base area
        High_Value_Area_gdf['value'] = 0
        for index in High_Value_Area_gdf.index:
            High_Value_Area_gdf.loc[index, 'value'] = \
            High_Value_Area_gdf.loc[index, 'projected_area'] * 5400

    return High_Value_Area_gdf

def make_osmexposure(highValueArea, mode="default", country=None,
                     save_path=None, check_plot=1, **kwargs):
    """
    Generate climada-compatiple entity by assigning values to midpoints of
    individual house shapes from OSM query, according to surface area and country.

    Parameters:
        highValueArea (str): absolute path for gdf of building features queried
          from get_features_OSM()
        mode (str): "LitPop" or "default": Default assigns a value of 5400 Chf to
          each m2 of building, LitPop assigns total LitPop value for the region
          proportionally to houses (by base area of house)
        Country (str): ISO3 code or name of country in which entity is located.
          Only if mode = LitPop
        kwargs (dict): arguments for LitPop set_country method

    Returns:
        exp_building (Exposure): (CLIMADA-compatible) with allocated asset values.
          Saved as exposure_buildings_mode_lat_lon.h5

    Example:
        buildings_47_8 = \
        make_osmexposure(save_path + '/OSM_features_47_8.shp',
                         mode="default", save_path = save_path, check_plot=1)
    """
    if save_path is None:
        save_path = Path.cwd()
    elif isinstance(save_path, str):
        save_path = Path(save_path)

    High_Value_Area_gdf = _get_midpoints(highValueArea)

    High_Value_Area_gdf = _assign_values_exposure(High_Value_Area_gdf, mode, country, **kwargs)

    # put back into CLIMADA-compatible entity format and save as hdf5 file:
    exp_buildings = Exposures(High_Value_Area_gdf)
    exp_buildings.set_lat_lon()
    exp_buildings.check()
    exp_buildings.write_hdf5(save_path.joinpath('exposure_buildings_' + mode + '_' +
                             str(int(min(High_Value_Area_gdf.bounds.miny))) +
                             '_' + str(int(min(High_Value_Area_gdf.bounds.minx))) + '.h5'))

    # plotting
    if check_plot == 1:
        # normal hexagons
        exp_buildings.plot_hexbin(pop_name=True)
        # select the OSM background image from the available ctx.sources
        # - returns connection error, left out for now:
        #fig, ax = exp_buildings.plot_basemap(buffer=30000, url=ctx.sources.OSM_C, cmap='brg')

    return exp_buildings
