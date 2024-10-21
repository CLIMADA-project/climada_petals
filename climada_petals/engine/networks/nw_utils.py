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

"""

import geopandas as gpd
import pandas as pd
import shapely
import numpy as np
from scipy.spatial import cKDTree
import rasterio
from rasterio.enums import Resampling
import logging
from pathlib import Path
import urllib.request
import requests
import time

from climada.util import coordinates as u_coords
from climada_petals.util.constants import DICT_SPEEDS

# Energy conversion factors
TJ_TO_GWH = 0.277778
HRS_PER_YEAR = 8760

LOGGER = logging.getLogger(__name__)

# =============================================================================
# Spatial analysis util functions
# =============================================================================


def make_edge_geometries(vs_geoms_from, vs_geoms_to):
    """
    create straight shapely LineString geometries between lists of
    from and to nodes, to be added to newly created edges as attributes
    """
    return [shapely.geometry.LineString([geom_from, geom_to]) for
            geom_from, geom_to in
            zip(vs_geoms_from, vs_geoms_to)]


def _preselect_destinations(vs_assign, vs_base, dist_thresh):
    points_base = np.array([(x.x, x.y) for x in vs_base['geometry']])
    point_tree = cKDTree(points_base)

    points_assign = np.array([(x.x, x.y) for x in vs_assign['geometry']])
    ix_matches = []
    for assign_loc in points_assign:
        ix_matches.append(point_tree.query_ball_point(assign_loc, dist_thresh))
    return ix_matches


def _ckdnearest(vs_assign, gdf_base, k=1, dist_thresh=np.inf):
    """
    see https://gis.stackexchange.com/a/301935

    Parameters
    ----------
    vs_assign : gpd.GeoDataFrame or Point
    gdf_base : gpd.GeoDataFrame

    Returns
    ----------

    """
    # TODO: this mixed input options (1 vertex vs gdf) is not nicely solved
    if isinstance(vs_assign, (gpd.GeoDataFrame, pd.DataFrame)):
        n_assign = np.array(
            list(vs_assign.geometry.apply(lambda x: (x.x, x.y))))
    else:
        n_assign = np.array([(vs_assign.geometry.x, vs_assign.geometry.y)])
    n_base = np.array(list(gdf_base.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(n_base)
    dist, idx = btree.query(n_assign, k=k, distance_upper_bound=dist_thresh)

    vld_ind = gdf_base.iloc[idx[dist < np.inf]].index.values
    vld_ind_formatted = np.empty(dist.shape)
    vld_ind_formatted[:] = np.nan
    vld_ind_formatted[np.where(dist < np.inf)] = vld_ind
    return dist, vld_ind_formatted


def _resample_res(filepath, upscale_factor, nodata):

    with rasterio.open(filepath) as dataset:
        # resample data to target shape
        arr = dataset.read(
            out_shape=(dataset.count, int(dataset.height * upscale_factor),
                       int(dataset.width * upscale_factor)),
            resampling=Resampling.average)
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / arr.shape[-1]),
            (dataset.height / arr.shape[-2]))

    arr = np.where(arr == nodata, 0, arr)
    arr = arr*(1/upscale_factor)**2

    return arr, transform


def load_resampled_raster(filepath, upscale_factor, nodata=-99999.):

    arr, transform = _resample_res(filepath, upscale_factor, nodata)

    grid = u_coords.raster_to_meshgrid(transform, arr.shape[-1],
                                       arr.shape[-2])
    gdf = gpd.GeoDataFrame({'counts': arr.squeeze().flatten(),
                            'geometry': gpd.points_from_xy(
                                grid[0].flatten(), grid[1].flatten())})
    gdf = gdf[gdf.counts != 0].reset_index(drop=True)

    # manual correction for over-estimate after aggregation:
    arr_orig, __ = _resample_res(filepath, 1, nodata)
    corr_factor = arr_orig.squeeze().flatten().sum() / \
        arr.squeeze().flatten().sum()
    gdf['counts'] = gdf.counts * corr_factor

    return gdf

# =============================================================================
# General results analysis util functions
# =============================================================================


def service_dict():
    return {'power': 'actual_supply_power_line_people',
            'healthcare': 'actual_supply_health_people',
            'education': 'actual_supply_education_people',
            'telecom': 'actual_supply_celltower_people',
            'mobility': 'actual_supply_road_people',
            'water': 'actual_supply_wastewater_people'}


def number_noservice(service, graph):

    no_service = (1-np.array(graph.graph.vs.select(
        ci_type='people')[service_dict()[service]]))
    pop = np.array(graph.graph.vs.select(
        ci_type='people')['counts'])
    return (no_service*pop).sum()


def number_noservices(graph, services=['power', 'healthcare', 'education',
                                       'telecom', 'mobility', 'water']):

    servstats_dict = {}
    for service in services:
        servstats_dict[service] = number_noservice(service, graph)
    return servstats_dict


def number_noservice_df(service, df, service_dict=service_dict()):
    """
    Number of population having '0' or '-1' in service state for a respective service

    Parameters
    -----------
    service : str
        the service to check (in-)availability for
    df : dataframe
        a (geo-)dataframe containing information on population clusters
        and the respective state of certain sevices, e.g. from a df_res extracted
        for saving from disrupted graph, or from network.nodes instance

    Note
    -----
    same as number_noservice, just that it's performed on a df, not on the graph
    """
    return df[
        (df.ci_type == 'people') &
        (df[service_dict[service]] <= 0)
    ].counts.sum()


def number_noservices_df(df, services=['power', 'healthcare', 'education',
                                       'telecom', 'mobility', 'water']):
    """
    Note
    -----
    same as number_noservices, just that it's performed on a df, not on the graph
    """
    servstats_dict = {}
    for service in services:
        servstats_dict[service] = number_noservice_df(service, df)
    return servstats_dict


def disaster_impact_service_geoseries(service, pre_graph, post_graph):

    no_service_post = (1-np.array(post_graph.graph.vs.select(
        ci_type='people')[service_dict()[service]]))
    no_service_pre = (1-np.array(pre_graph.graph.vs.select(
        ci_type='people')[service_dict()[service]]))
    geom = np.array(post_graph.graph.vs.select(
        ci_type='people')['geom_wkt'])
    return gpd.GeoSeries.from_wkt(
        geom[np.where((no_service_post-no_service_pre) > 0)])


def disaster_impact_service(service, pre_graph, post_graph):

    no_service_post = (1-np.array(post_graph.graph.vs.select(
        ci_type='people')[service_dict()[service]]))
    no_service_pre = (1-np.array(pre_graph.graph.vs.select(
        ci_type='people')[service_dict()[service]]))
    pop = np.array(pre_graph.graph.vs.select(
        ci_type='people')['counts'])
    return ((no_service_post-no_service_pre)*pop).sum()


def disaster_impact_allservices(pre_graph, post_graph,
                                services=['power', 'healthcare', 'education',
                                          'telecom', 'mobility', 'water']):

    dict_pre = number_noservices(pre_graph, services)
    dict_post = number_noservices(post_graph, services)
    dict_delta = {}
    for key, value in dict_post.items():
        dict_delta[key] = value-dict_pre[key]
    return dict_delta


def disaster_impact_allservices_df(df_pre, df_post,
                                   services=['power', 'healthcare', 'education',
                                             'telecom', 'mobility', 'water']):
    """
    Note
    -----
    same as disaster_impact_allservices, just that it's performed on a df,
    not on the graph
    """
    dict_pre = number_noservices_df(df_pre, services)
    dict_post = number_noservices_df(df_post, services)
    dict_delta = {}
    for key, value in dict_post.items():
        dict_delta[key] = value-dict_pre[key]
    return dict_delta


def get_graphstats(graph):
    from collections import Counter
    stats_dict = {}
    stats_dict['no_edges'] = len(graph.graph.es)
    stats_dict['no_nodes'] = len(graph.graph.vs)
    stats_dict['edge_types'] = Counter(graph.graph.es['ci_type'])
    stats_dict['node_types'] = Counter(graph.graph.vs['ci_type'])
    return stats_dict


# =============================================================================
# Worldpop Data
# =============================================================================
def get_worldpop_data(iso3, save_path, res=1000):

    if res == 1000:
        download_url = 'https://data.worldpop.org/GIS/Population/' + \
            f'Global_2000_2020_1km_UNadj/2020/{iso3}/' + \
            f'{iso3.lower()}_ppp_2020_1km_Aggregated_UNadj.tif'
    elif res == 100:
        download_url = 'https://data.worldpop.org/GIS/Population/' + \
            f'Global_2000_2020/2020/{iso3}/{iso3.lower()}_ppp_2020_UNadj.tif'

    local_filepath = Path(save_path, download_url.split('/')[-1])

    if not Path(local_filepath).is_file():
        LOGGER.info(f'Downloading file as {local_filepath}')
        urllib.request.urlretrieve(download_url, local_filepath)
    else:
        LOGGER.info(f'file already exists as {local_filepath}')


def get_pop_cutoff(gdf_people, cutoff):
    """
    find the maximum population value per grid point which accounts cumulatively across
    the entire gdf for less than a cutoff fraction of the entire population number
    to decrease the
    """
    bins = [0, 10, 20, 35, 50, 75]
    bins.extend(
        list(np.linspace(start=100, stop=gdf_people["counts"].max(), num=30)))
    df_cum = gdf_people.groupby(
        pd.cut(gdf_people["counts"], bins)).sum(numeric_only=True)/gdf_people["counts"].sum()
    cutoff_bool = (df_cum.cumsum() >= cutoff).counts.values
    cutoff_interval = df_cum.index.categories[cutoff_bool][0]
    return cutoff_interval.left

# =============================================================================
# Distance Threshold setting
# =============================================================================


def set_travel_distance_threshs(df_dependencies, iso3, hrs_max=1):
    """
    set road travel distance threshold according to the average distance
    covered within a specified amount of hours in the respective country.

    Data taken from Road Quality and Mean Speed Score
    Author/Editor:Marian Moszoro ; Mauricio Soto
    ISBN: 9798400210440/1018-5941
    """
    try:
        thresh_dist = int(DICT_SPEEDS[iso3]*1000*hrs_max)
    except KeyError:
        thresh_dist = int(DICT_SPEEDS['other']*1000*hrs_max)

    df_dependencies.loc[(df_dependencies.source == 'health') |
                        (df_dependencies.source == 'education') &
                        (df_dependencies.target == 'people'),
                        'thresh_dist'] = thresh_dist

    return df_dependencies

# =============================================================================
# Power Supply & Demand Data
# =============================================================================


class PowerFunctionalData():

    def load_pplants_wri(self, path_pplants_wri, iso3):
        """
        load power plant gdf from an excel file containing the WRI global
        power plant database.
        """
        gdf_pp_world = gpd.read_file(path_pplants_wri, crs='EPSG:4326')
        gdf_pp = gdf_pp_world[gdf_pp_world.country == f'{iso3}'][
            ['estimated_generation_gwh_2017', 'latitude', 'longitude',
             'name']]
        del gdf_pp_world
        if not gdf_pp.empty:
            gdf_pp = gpd.GeoDataFrame(
                gdf_pp, geometry=gpd.points_from_xy(gdf_pp.longitude, gdf_pp.latitude))
            gdf_pp = gdf_pp.drop(['latitude', 'longitude'], axis=1)

        return gdf_pp

    def query_elaccess_wb(self, iso3):
        ind = 'EG.ELC.ACCS.ZS'
        query = f'http://api.worldbank.org/v2/country/{iso3.lower()}/indicator/{ind}?mrnev=1&format=json'
        trials = 0
        while trials < 5:
            trials += 1
            try:
                response = requests.get(query)
                return response.json()[1][0]['value']
            except:
                time.sleep(5)
                LOGGER.info('Sleeping for 5 secs until next WB data query try')

    def load_eltargets(self, cntry_shape, path_et):
        meta_et, arr_et = u_coords.read_raster(path_et, src_crs={'epsg': '4326'},
                                               geometry=[cntry_shape])
        grid = u_coords.raster_to_meshgrid(meta_et['transform'], meta_et['width'],
                                           meta_et['height'])
        gdf_et = gpd.GeoDataFrame({'counts': arr_et.squeeze().flatten(),
                                   'geometry': gpd.points_from_xy(
            grid[0].flatten(), grid[1].flatten())})
        gdf_et = gdf_et[gdf_et.counts != 0].reset_index(drop=True)
        gdf_et['geometry'] = gdf_et.geometry.buffer(meta_et['transform'][0]/2)
        return gdf_et

    def assign_el_targets(self, gdf_people, iso3, cntry_shape, path_et):

        el_rate = self.query_elaccess_wb(iso3)

        if el_rate < 90:
            gdf_et = self.load_eltargets(cntry_shape, path_et)
            gdf_people['electrified'] = \
                gdf_people.sjoin(gdf_et, how='left', predicate="within"
                                 )['counts_right'].groupby(level=0).sum() > 0
            counter = 0
            while (gdf_people[gdf_people['electrified']].counts.sum() < (gdf_people.counts.sum()*el_rate/100)
                   and counter < 10):
                gdf_et['geometry'] = gdf_et.geometry.buffer(0.01)
                gdf_people['electrified'] = (gdf_people.sjoin(gdf_et, how='left', predicate="within")[
                                             'counts_right'].groupby(level=0).sum() > 0)
                counter += 1
        else:
            gdf_people['electrified'] = True

        return gdf_people.electrified

    def assign_el_prod_consump(self, gdf_people, gdf_pplants, iso3, path_final_cons):
        """
        Takes a countries' annual electricity consumption value (as
        gathered by the IEA in https://www.iea.org/data-and-statistics/data-tables?,
        "Total Final Consumption, Electricity") and assigns this
        i) to gdf_people, proportional to population count
        ii) to gdf_pplants, either proportional to plant capacity reported in
        the global power plant database, or equally distributed.

        If no consumption value is available for the country, dummy demands
        proportional to population are assigned to gdf_people, and equally
        distributed onto gdf_pplants.

        Returns
        ------
        gdf_people with column el_consumption
        gdf_pplants with column el_generation
        """

        # easiest default for population and power plants - assign count dummies
        gdf_people['el_consumption'] = 0
        gdf_people.loc[gdf_people.electrified, 'el_consumption'] = \
            gdf_people.loc[gdf_people.electrified, 'counts']
        gdf_pplants['el_generation'] = \
            gdf_people['el_consumption'].sum()/len(gdf_pplants)

        # check if better data available from IEA & WRI
        df_final_cons = pd.read_csv(path_final_cons)
        final_cons = np.nan
        if iso3 in df_final_cons.ISO3.values:
            final_cons = df_final_cons[df_final_cons.ISO3 == iso3
                                       ].el_consumption.values[0]
        if not np.isnan(final_cons):
            # assign electricity consumption to population
            gdf_people.loc[gdf_people.electrified, 'el_consumption'] = \
                gdf_people.loc[gdf_people.electrified, 'counts'] /\
                gdf_people.loc[gdf_people.electrified, 'counts'].sum(
            )*final_cons
            if 'estimated_generation_gwh_2017' in gdf_pplants.columns:
                # assign electricity generation to power plants
                gdf_pplants['estimated_generation_gwh_2017'] = pd.to_numeric(
                    gdf_pplants.estimated_generation_gwh_2017, errors='coerce')
                gdf_pplants['estimated_generation_gwh_2017'].fillna(
                    np.nanmean(gdf_pplants['estimated_generation_gwh_2017']),
                    inplace=True)
                gdf_pplants['el_generation'] = (
                    gdf_pplants.estimated_generation_gwh_2017 /
                    gdf_pplants.estimated_generation_gwh_2017.sum()*final_cons)
                gdf_pplants = gdf_pplants.drop('estimated_generation_gwh_2017',
                                               axis=1)
            else:
                gdf_pplants['el_generation'] = final_cons/len(gdf_pplants)

        return gdf_people, gdf_pplants

    def assign_edemand_iea(self, gdf_people, path_elcons_iea):
        """
        Assigns annual electricity consumptions to power
        clusters based on per capita consuption statistics
        retrieved from IEA.org --> Heat & Electricity --> Electricity Consumption
        per capita.
        Returns a pd.Series with el_consumption in GWh for the last reported year
        (currently 2019)
        """

        df_el_cons = pd.read_csv(path_elcons_iea, skiprows=4)

        if df_el_cons.Units.iloc[0] != 'MWh/capita':
            LOGGER.warning('Units of per capita electricity consumption are' +
                           f'different than expected. ({df_el_cons.Units.iloc[0]})')

        per_cap_cons = df_el_cons['Electricity consumption/population'].iloc[-1]
        LOGGER.info("Taking per capita electricity consumption value for year" +
                    f" {df_el_cons['Unnamed: 0'].iloc[-1]}")
        return gdf_people.counts * per_cap_cons/1000

    def assign_esupply_iea(self, gdf_pplants,  path_elgen_iea):
        """
        Assigns annual electricity generation (in GWh) to each power plant
        reported in the global power plant database from the WRI.
        Electricity generation is taken from IEA.org (expects a column
        'estimated_generation_gwh_2017') and re-distributed upon
        power plants proportionally to the generation values given in the
        WRI database, and the rest distributed equally on missing values.
        """

        df_el_gen = pd.read_csv(path_elgen_iea, skiprows=4)

        # Latest Electricity Generation data from the IEA (2019)
        if df_el_gen.iloc[-1]['Units'] != 'GWh':
            LOGGER.warning('Expected different units for generation.')
        gen = np.nansum(df_el_gen[
            list(set(df_el_gen.columns.values).difference(
                {'Units', 'Unnamed: 0'}))
        ].iloc[-1].values)

        # generation from WRI power plants database (usually incomplete)
        gdf_pplants.estimated_generation_gwh_2017 = pd.to_numeric(
            gdf_pplants.estimated_generation_gwh_2017, errors='coerce')

        plant_gen = gdf_pplants.estimated_generation_gwh_2017.fillna(
            np.nanmean(gdf_pplants.estimated_generation_gwh_2017))

        return np.array(plant_gen/sum(plant_gen)*gen)

    def assign_impexp_iea(self, gdf_pplants, path_elimpexp_iea, var_name):
        """
        Places an import / export node outside the country,
        to account for this delta as reported in IEA.org
        """
        # Latest annual Import/Export data from the IEA (2019)
        df_el_impexp = pd.read_csv(path_elimpexp_iea, skiprows=4)

        if df_el_impexp.iloc[-1]['Units'] != 'TJ':
            LOGGER.warning('Expected different units for import/export.')
        if 'Exports' not in df_el_impexp.columns:
            df_el_impexp['Exports'] = 0
            LOGGER.warning('No export column. Setting exports to 0.')
        if 'Imports' not in df_el_impexp.columns:
            df_el_impexp['Imports'] = 0
            LOGGER.warning('No import column. Setting imports to 0.')

        el_imp = df_el_impexp.iloc[-1]['Imports']*TJ_TO_GWH
        el_exp = df_el_impexp.iloc[-1]['Exports']*TJ_TO_GWH
        imp_exp = el_imp + el_exp

        return gpd.GeoDataFrame({'geometry': [shapely.geometry.Point(max(gdf_pplants.geometry.x)+1,
                                                                     max(gdf_pplants.geometry.y)+1)],
                                 'name': ['imp_exp_balance'],
                                 f'{var_name}': [imp_exp]})

    def balance_el_generation(self, gdf_pplants, per_cap_cons, pop_no):

        gdf_pplants.estimated_generation_gwh_2017 = pd.to_numeric(
            gdf_pplants.estimated_generation_gwh_2017, errors='coerce')

        tot_cons_mw = per_cap_cons*pop_no/HRS_PER_YEAR
        tot_prod_mw = gdf_pplants.estimated_generation_gwh_2017.sum()*1000/HRS_PER_YEAR

        unassigned = tot_cons_mw - tot_prod_mw
        nans = np.isnan(gdf_pplants.estimated_generation_gwh_2017).sum()

        gdf_pplants['el_gen_mw'] = gdf_pplants.estimated_generation_gwh_2017 * \
            1000/HRS_PER_YEAR
        gdf_pplants['el_gen_mw'][np.isnan(
            gdf_pplants.el_gen_mw)] = unassigned/nans

        return gdf_pplants
