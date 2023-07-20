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

Define functions to download openstreetmap data
"""
import itertools
import logging
from pathlib import Path
import subprocess
import time
import urllib.request

import geopandas as gpd
import numpy as np
from osgeo import ogr, gdal
import overpy
import shapely
from tqdm import tqdm

from climada import CONFIG
from climada_petals.util.constants import DICT_GEOFABRIK, DICT_CIS_OSM


LOGGER = logging.getLogger(__name__)
DATA_DIR = CONFIG.exposures.openstreetmap.local_data.dir()
OSM_CONFIG_FILE = Path(__file__).parent.joinpath('osmconf.ini')
gdal.SetConfigOption("OSM_CONFIG_FILE", str(OSM_CONFIG_FILE))


class OSMRaw:
    """
    functions to obtain entire raw datasets from OSM,
    from different sources"""

    def __init__(self):
        self.geofabrik_url = 'https://download.geofabrik.de/'
        self.planet_url = 'https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf'

    def _create_gf_download_url(self, iso3, file_format):
        """
        create string with download-api from geofabrik

        Parameters
        ----------
        iso3 : str
            ISO3 code of country to download
        file_format : str
            Format in which file should be downloaded; ESRI Shapefiles ('shp')
            or osm-Protocolbuffer Binary Format ('pbf')

        Returns
        -------
        str : Geofabrik ownload-api for the requested country.

        See also
        --------
        DICT_GEOFABRIK for exceptions / special regions.
        """
        try:
            if file_format == 'shp':
                return f'{self.geofabrik_url}{DICT_GEOFABRIK[iso3][0]}/{DICT_GEOFABRIK[iso3][1]}-latest-free.shp.zip'
            if file_format == 'pbf':
                return f'{self.geofabrik_url}{DICT_GEOFABRIK[iso3][0]}/{DICT_GEOFABRIK[iso3][1]}-latest.osm.pbf'
        except KeyError:
            if iso3=='RUS':
                raise KeyError("""Russia comes in two files. Please specify either
                             'RUS-A for the Asian or RUS-E for the European part.""")
            else:
                raise KeyError("""The provided iso3 is not a recognised
                             code. Please have a look on Geofabrik.de if it
                             exists, or check your iso3 code.""")

        return LOGGER.error('invalid file format. Please choose one of [shp, pbf]')

    def get_data_geofabrik(self, iso3, file_format='pbf', save_path=DATA_DIR):
        """
        Download country files with all OSM map info from the provider
        Geofabrik.de, if doesn't exist, yet.

        Parameters
        ----------
        iso3 : str
            ISO3 code of country to download
            Exceptions: Russia is divided into European and Asian part
            ('RUS-E', 'RUS-A'), Canary Islands are 'IC'.
        file_format : str
            Format in which file should be downloaded; options are
            ESRI Shapefiles (shp), which can easily be loaded into gdfs,
            or osm-Protocolbuffer Binary Format (pbf), which is smaller in
            size, but has a more complicated query syntax to load (functions
            are provided in the OSMFileQuery class).
        save_path : str or pathlib.Path
            Folder in which to save the file

        Returns
        -------
        None
            File is downloaded and stored under save_path + the Geofabrik filename

        See also
        --------
        DICT_GEOFABRIK for exceptions / special regions.
        """

        download_url = self._create_gf_download_url(iso3, file_format)
        local_filepath = Path(save_path , download_url.split('/')[-1])
        if not Path(local_filepath).is_file():
            LOGGER.info(f'Downloading file as {local_filepath}')
            urllib.request.urlretrieve(download_url, local_filepath)
        else:
            LOGGER.info(f'file already exists as {local_filepath}')

    def get_data_planet(self,
                        save_path=Path(DATA_DIR,'planet-latest.osm.pbf')):
        """
        Download the entire planet file from the OSM server (ca. 60 GB).

        Parameters
        ----------
        save_path : str or pathlib.Path
        """

        if not Path(save_path).is_file():
            LOGGER.info(f'Downloading file as {save_path}')
            urllib.request.urlretrieve(self.planet_url, save_path)
        else:
            LOGGER.info(f'file already exists as {save_path}')

    def _osmosis_extract(self, shape, path_planet, path_extract,
                         overwrite=False):
        """
        Runs the command line tool osmosis to cut out all map info within
        shape, from the osm planet file, unless file already exists.

        If your device doesn't have osmosis yet, see installation instructions:
        https://wiki.openstreetmap.org/wiki/Osmosis/Installation

        Parameters
        -----------
        shape : list or str
            bounding box [xmin, ymin, xmax, ymax] or file path to a .poly file
        path_planet : str or pathlib.Path
            file path to planet.osm.pbf
        path_extract : str or pathlib.Path
            file path (incl. name & ending) under which extract will be stored
        overwrite : bool
            default is False. Whether to overwrite files if they already exist.

        Returns
        -------
        None or subprocess
        """

        if ((not Path(path_extract).is_file()) or
            (Path(path_extract).is_file() and overwrite)):

            LOGGER.info("""File doesn`t yet exist or overwriting old one.
                        Assembling osmosis command.""")
            if isinstance(shape, (list, tuple)):
                cmd = ['osmosis', '--read-pbf', 'file='+str(path_planet),
                       '--bounding-box', f'top={shape[3]}', f'left={shape[0]}',
                       f'bottom={shape[1]}', f'right={shape[2]}',
                       '--write-pbf', 'file='+str(path_extract)]
            elif isinstance(shape, str):
                cmd = ['osmosis', '--read-pbf', 'file='+str(path_planet),
                       '--bounding-polygon', 'file='+shape, '--write-pbf',
                       'file='+str(path_extract)]

            LOGGER.info('''Extracting from the osm planet file...
                        This will take a while''')

            return subprocess.run(cmd, stdout=subprocess.PIPE,
                                  universal_newlines=True)

        if (Path(path_extract).is_file() and (overwrite is False)):
            LOGGER.info("Extracted file already exists!")
        else:
            LOGGER.info("""Something went wrong with Path specifications.
                        'Please enter either a valid string or pathlib.Path""")
        return None

    def get_data_planetextract(self, shape, path_extract,
                               path_planet=Path(DATA_DIR, 'planet-latest.osm.pbf'),
                               overwrite=False):
        """
        get OSM raw data from a custom shape / bounding-box, which is extracted
        from the entire OSM planet file. Accepts bbox lists or .poly files for
        non-rectangular shapes.

        Parameters
        ----------
        shape : list or str
            bounding box [xmin, ymin, xmax, ymax] or file path to a .poly file
        path_extract : str or pathlib.Path
            file path (incl. name & ending) under which extract will be stored
        path_planet : str or pathlib.Path
            file path to planet-latest.osm.pbf. Will download & store it as
            indicated, if doesn`t yet exist.
            Default is DATA_DIR/planet-latest.osm.pbf

        Note
        ----
        For more info on what .poly files are (incl. several tools for
        creating them), see
        https://wiki.openstreetmap.org/wiki/Osmosis/Polygon_Filter_File_Format

        For creating .poly files on admin0 to admin3 levels of any place on the
        globe, see the GitHub repo https://github.com/ElcoK/osm_clipper
        (especially the function make_poly_file())

        Note
        ----
        This function uses the command line tool osmosis to cut out new
        osm.pbf files from the original ones.
        Installation instructions (windows, linux, apple) - see
        https://wiki.openstreetmap.org/wiki/Osmosis/Installation
        """

        if not Path(path_planet).is_file():
            LOGGER.info("planet-latest.osm.pbf wasn't found. Downloading it.")
            self.get_data_planet(path_planet)

        self._osmosis_extract(shape, path_planet, path_extract, overwrite)

    def get_data_fileextract(self, shape, path_extract, path_parentfile,
                             overwrite=False):
        """
        Extract a geographic sub-set from a raw osm-pbf file.

        Note
        ----
        The shape must be entirely contained within the file to extract from,
        else it will yield weird results.

        Parameters
        ----------
        shape : list or str
            bounding box [xmin, ymin, xmax, ymax] or file path to a .poly file
        path_extract : str or pathlib.Path
            file path (incl. name & ending) under which extract will be stored
        path_parentfile : str or pathlib.Path
            file path to parentfile.osm.pbf from which the shape will be cut out
        overwrite : bool
            default is False. Whether to overwrite files if they already exist.

        Note
        ----
        This function uses the command line tool osmosis to cut out new
        osm.pbf files from the original ones.
        Installation instructions (windows, linux, apple) - see
        https://wiki.openstreetmap.org/wiki/Osmosis/Installation
        """

        self._osmosis_extract(shape, path_parentfile, path_extract, overwrite)


class OSMFileQuery:
    """
    Load features from raw osm.pbf files.
    """
    def __init__(self, osm_path):
        """
        Parameters
        ----------
        osm_path : str or pathlib.Path
            file path to the .osm.pbf file to extract info from.

        Raises
        ------
        ValueError
            if the given path is not a file
        """
        if not Path(osm_path).is_file():
            raise ValueError(f"the given path is not a file: {osm_path}")
        self.osm_path = str(osm_path)

    def _query_builder(self, geo_type, constraint_dict):
        """
        from BenDickens/trails repo (https://github.com/BenDickens/trails.git, see
                                     extract.py)
        This function builds an SQL query from the values passed to the retrieve()
        function.

        Parameters
        ---------
        geo_type : str
            Type of geometry to retrieve. One of [points, lines, multipolygons]
        constraint_dict :  dict

        Returns
        -------
        query : str
            an SQL query string.
        """
        # columns which to report in output
        query =  "SELECT osm_id"
        for key in constraint_dict['osm_keys']:
            query+= ","+ key
        # filter condition(s)
        query+= " FROM " + geo_type + " WHERE " + constraint_dict['osm_query']

        return query

    def retrieve(self, geo_type, osm_keys, osm_query):
        """
        Function to extract geometries and tag info for entires in the OSM file
        matching certain OSM key-value constraints.
        from an OpenStreetMap osm.pbf file.
        adapted from BenDickens/trails repo
        (https://github.com/BenDickens/trails.git, see extract.py)

        Parameters
        ----------
        geo_type : str
            Type of geometry to retrieve. One of [points, lines, multipolygons]
        osm_keys : list
            a list with all the osm keys that should be reported as columns in
            the output gdf.
        osm_query : str
            query string of the syntax
            "key(='value') (and/or further queries)".
            See examples in DICT_CIS_OSM in case of doubt.

        Returns
        -------
        gpd.GeoDataFrame
            A gdf with all results from the osm.pbf file matching the
            specified constraints.

        Note
        ----
        1) The keys that are searchable are specified in the osmconf.ini file.
        Make sure that they exist in the attributes=... paragraph under the
        respective geometry section.
        For example, to retrieve multipolygons with building='yes',
        building must be in the attributes under
        the [multipolygons] section of the file. You can find it in the same
        folder as the osm_dataloader.py module is located.
        2) OSM keys that have : in their name must be changed to _ in the
        search dict, but not in the osmconf.ini
        E.g. tower:type is called tower_type, since it would interfere with the
        SQL syntax otherwise, but still tower:type in the osmconf.ini

        See also
        --------
        https://taginfo.openstreetmap.org/ to check what keys and key/value
        pairs are valid.
        https://overpass-turbo.eu/ for a direct visual output of the query,
        and to quickly check the validity. The wizard can help you find the
        correct keys / values you are looking for.
        """
        constraint_dict = {
            'osm_keys' : osm_keys,
            'osm_query' : osm_query}

        driver = ogr.GetDriverByName('OSM')
        data = driver.Open(self.osm_path)
        query = self._query_builder(geo_type, constraint_dict)
        LOGGER.debug("query: %s", query)
        sql_lyr = data.ExecuteSQL(query)
        features = []
        if data is not None:
            LOGGER.info('query is finished, lets start the loop')
            for feature in tqdm(sql_lyr, desc=f'extract {geo_type}'):
                try:
                    fields = [feature.GetField(key) for key in
                              ['osm_id', *constraint_dict['osm_keys']]]
                    wkb = feature.geometry().ExportToWkb()
                    geom = shapely.wkb.loads(bytes(wkb))
                    if geom is None:
                        continue
                    fields.append(geom)
                    features.append(fields)
                except Exception as exc:
                    LOGGER.info('%s - %s', exc.__class__, exc)
                    LOGGER.warning("skipped OSM feature")
        else:
            LOGGER.error("""Nonetype error when requesting SQL. Check the
                         query and the OSM config file under the respective
                         geometry - perhaps key is unknown.""")

        return gpd.GeoDataFrame(
            features, columns=['osm_id', *constraint_dict['osm_keys'], 'geometry'])

    def retrieve_cis(self, ci_type):
        """
        A wrapper around retrieve() to conveniently retrieve map info for a
        selection of  critical infrastructure types from the given osm.pbf file.
        No need to search for osm key/value tags and relevant geometry types.

        Parameters
        ----------
        ci_type : str
            one of DICT_CIS_OSM.keys(), i.e. 'education', 'healthcare',
            'water', 'telecom', 'road', 'rail', 'air', 'gas', 'oil', 'power',
            'wastewater', 'food'

        See also
        -------
        DICT_CIS_OSM for the keys and key/value tags queried for the respective
        CIs. Modify if desired.
        """
        # features consisting in points and multipolygon results:
        if ci_type in ['healthcare','education','food']:
            gdf = self.retrieve('points', DICT_CIS_OSM[ci_type]['osm_keys'],
                                 DICT_CIS_OSM[ci_type]['osm_query'])
            gdf = gdf.append(
                self.retrieve('multipolygons', DICT_CIS_OSM[ci_type]['osm_keys'],
                              DICT_CIS_OSM[ci_type]['osm_query']))

        # features consisting in multipolygon results:
        elif ci_type in ['air']:
            gdf = self.retrieve('multipolygons', DICT_CIS_OSM[ci_type]['osm_keys'],
                                 DICT_CIS_OSM[ci_type]['osm_query'])

        # features consisting in points, multipolygons and lines:
        elif ci_type in ['gas','oil','telecom','water','wastewater','power',
                         'rail','road']:
            gdf = self.retrieve('points', DICT_CIS_OSM[ci_type]['osm_keys'],
                                 DICT_CIS_OSM[ci_type]['osm_query'])
            gdf = gdf.append(
                self.retrieve('multipolygons', DICT_CIS_OSM[ci_type]['osm_keys'],
                                 DICT_CIS_OSM[ci_type]['osm_query']))
            gdf = gdf.append(
                self.retrieve('lines', DICT_CIS_OSM[ci_type]['osm_keys'],
                                 DICT_CIS_OSM[ci_type]['osm_query']))
        else:
            LOGGER.warning('feature not in DICT_CIS_OSM. Returning empty gdf')
            gdf = gpd.GeoDataFrame()
        return gdf

class OSMApiQuery:
    """
    Queries features directly via the overpass turbo API.

    area: tuple (xmin, ymin, xmax, ymax), list [xmin, ymin, xmax, ymax]
        or shapely.geometry.Polygon
    query: str
        must be of format '["key"]' or '["key"="value"]', etc.
    """

    def __init__(self, area, condition):
        self.area = self._area_to_queryformat(area)
        self.condition = condition

    def _area_to_queryformat(self, area):
        """
        reformat lat/lon info as in OSM convention, meaning
        bbox: (S,W,N,E) instead of (xmin, ymin, xmax, ymax)
        Points: lat / lon instead of (x,y)
        """
        if isinstance(area,(tuple, list)):
            xmin, ymin, xmax, ymax = area
            return (ymin, xmin, ymax, xmax)

        if isinstance(area,shapely.geometry.Polygon):
            lon, lat = area.exterior.coords.xy
            lat_lon_str = " ".join([str(y)+" "+str(x) for y, x in zip(lat, lon)])
            return f'(poly:"{lat_lon_str}")'

    def _insistent_osm_api_query(self, query_clause, read_chunk_size=100000, end_of_patience=127):
        """Runs a single Overpass API query through overpy.Overpass.query.
        In case of failure it tries again after an ever increasing waiting period.
        If the waiting period surpasses a given limit an exception is raised.

        Parameters:
            query_clause (str): the query
            read_chunk_size (int): paramter passed over to overpy.Overpass.query
            end_of_patience (int): upper limit for the next waiting period to proceed.

        Returns:
            result as returned by overpy.Overpass.query
        """
        api = overpy.Overpass(read_chunk_size=read_chunk_size)
        waiting_period = 1
        while True:
            try:
                return api.query(query_clause)
            except overpy.exception.OverpassTooManyRequests:
                if waiting_period < end_of_patience:
                    LOGGER.warning("""Too many Overpass API requests -
                                   trying again in {waiting_period} seconds """)
                else:
                    raise Exception("Overpass API is consistently unavailable")
            except Exception as exc:
                if waiting_period < end_of_patience:
                    LOGGER.warning(f"""{exc}
                                   Trying again in {waiting_period} seconds""")
                else:
                    raise Exception("The Overpass API is consistently unavailable")
            time.sleep(waiting_period)
            waiting_period *= 2

    def _overpass_query_string(self):
        return f'[out:json][timeout:180];(nwr{self.condition}{self.area};(._;>;););out;'

    def _assemble_from_relations(self, result):
        """
        pick out those nodes and ways from resul instance that belong to
        relations. Assemble relations into gdfs. Keep track of which nodes
        and ways have been "used up" already

        Parameters
        ---------
        result : overpy.Overpass result object

        Returns
        -------
        nodes_taken : list
            node-ids that have been used to construct relations.
            Not "available" anymore for further constructions
        ways_taken : list
            way-ids that have been used to construct relations.
            Not "available" anymore for further constructions
        gdf_rels : gpd.GeoDataFrame
            gdf with relations that were assembled from result object
        """

        nodes_taken = []
        ways_taken = []
        data_geom = []
        data_id = []
        data_tags = []

        for relation in result.relations:
            data_tags.append(relation.tags)
            data_id.append(relation.id)
            roles = []
            relationways = []

            for way in relation.members:
                relationways.append(way.ref)
                roles.append(way.role)

            ways_taken.append(relationways)

            nodes_taken_mp, gdf_polys = self._assemble_from_ways(
                result, relationways, closed_lines_are_polys=True)

            nodes_taken.append(nodes_taken_mp)

            gdf_polys['role'] = roles

            # separate relationways into inner, outer polygons and linestrings,
            # combine them.

            # step 1: polygons to multipolygons
            inner_mp = shapely.geometry.MultiPolygon(
                gdf_polys.geometry[(gdf_polys.geometry.type=='Polygon') &
                                   (gdf_polys.role=='inner')].values)
            outer_mp = shapely.geometry.MultiPolygon(
                gdf_polys.geometry[(gdf_polys.geometry.type=='Polygon') &
                                   (gdf_polys.role=='outer')].values)

            # step 2: poly from lines --> multiline --> line --> polygon
            lines = gdf_polys.geometry[
                (gdf_polys.geometry.type=='LineString')].values
            if len(lines) > 0:
                poly = shapely.geometry.Polygon(
                    shapely.ops.linemerge(shapely.geometry.MultiLineString(lines)))
            else:
                poly = shapely.geometry.Polygon([])

            # step 3: combine to one multipoly
            multipoly = shapely.ops.unary_union([outer_mp - inner_mp, poly])
            data_geom.append(multipoly)

            if multipoly.area == 0:
                LOGGER.info('Empty geometry encountered.')

        gdf_rels =  gpd.GeoDataFrame(
            data={'osm_id': data_id,'geometry': data_geom, 'tags':data_tags})

        # list of lists into list:
        nodes_taken = list(itertools.chain.from_iterable(nodes_taken))
        ways_taken = list(itertools.chain.from_iterable(ways_taken))

        return nodes_taken, ways_taken, gdf_rels

    def _assemble_from_ways(self, result, ways_avail, closed_lines_are_polys):

        """
        pick out those nodes and ways from result instance that belong to
        ways. Assemble ways into gdfs. Keep track of which nodes
        and ways have been "used up" already

        Parameters
        ---------
        result : overpy.Overpass result object
        ways_avail : list
            way-ids that have not yet been used for relation construction
            and are hence available for way constructions
        closed_lines_are_polys : bool
            whether closed lines are polygons

        Returns
        -------
        nodes_taken : list
            node-ids that have been used to construct relations.
            Not "available" anymore for further constructions
        ways_taken : list
            way-ids that have been used to construct relations.
            Not "available" anymore for further constructions
        gdf_ways : gpd.GeoDataFrame
            gdf with ways that were assembled from result object
        """

        nodes_taken = []
        data_geom = []
        data_id = []
        data_tags = []

        for way in result.ways:
            if way.id in ways_avail:
                node_lat_lons = []
                for node in way.nodes:
                    nodes_taken.append(node.id)
                    node_lat_lons.append((float(node.lat), float(node.lon)))
                data_geom.append(shapely.geometry.LineString(node_lat_lons))
                data_id.append(way.id)
                data_tags.append(way.tags)

            if closed_lines_are_polys:
                data_geom = [shapely.geometry.Polygon(way) if way.is_closed
                             else way for way in data_geom]

        gdf_ways = gpd.GeoDataFrame(
            data={'osm_id': data_id,'geometry': data_geom, 'tags':data_tags})

        return nodes_taken, gdf_ways

    def _assemble_from_nodes(self, result, nodes_avail):
        """
        pick out those nodes and ways from result instance that belong to
        ways. Assemble ways into gdfs. Keep track of which nodes
        and ways have been "used up" already

        Parameters
        ---------
        result : overpy.Overpass result object
        nodes_avail : list
            node-ids that have not yet been used for relation and way
            construction and are hence available for node constructions

        Returns
        -------
        gdf_nodes : gpd.GeoDataFrame
            gdf with nodes that were assembled from result object
        """
        data_geom = []
        data_id = []
        data_tags = []

        for node in result.nodes:
            if node.id in nodes_avail:
                data_geom.append(shapely.geometry.Point(node.lat, node.lon))
                data_id.append(node.id)
                data_tags.append(node.tags)

        gdf_nodes = gpd.GeoDataFrame(
            data={'osm_id': data_id,'geometry': data_geom, 'tags':data_tags})

        return gdf_nodes

    def _update_availability(self, full_set, to_remove):
        """
        update id availabilities from whole result set after using up some for
        geometry construction
        """
        return [item for item in full_set if item not in to_remove]

    def _assemble_results(self, result, closed_lines_are_polys=True):
        """
        assemble an overpass result object with results, ways, nodes etc. from the
        format-specific structure into one geodataframe

        Parameters
        ---------
        result : overpy.Overpass result object
        closed_lines_are_polys : bool
                whether closed lines are polygons. Default is True

        Returns
        ------
        gdf_results : gpd.GeoDataFrame
            Result-gdf from the overpass query.
        """
        gdf_results = gdf_ways = gdf_nodes = gpd.GeoDataFrame(
            columns=['osm_id','geometry','tags'])
        nodes_avail = result.node_ids
        ways_avail = result.way_ids

        if len(result.relations) > 0:
            nodes_taken, ways_taken, gdf_rels = self._assemble_from_relations(result)
            gdf_results = gdf_results.append(gdf_rels)
            nodes_avail = self._update_availability(nodes_avail, nodes_taken)
            ways_avail = self._update_availability(ways_avail, ways_taken)

        if len(ways_avail) > 0:
            nodes_taken, gdf_ways = self._assemble_from_ways(
                result,  ways_avail, closed_lines_are_polys)
            gdf_results = gdf_results.append(gdf_ways)
            nodes_avail = self._update_availability(nodes_avail, nodes_taken)

        if len(nodes_avail) > 0:
            gdf_nodes = self._assemble_from_nodes(result, nodes_avail)
            gdf_results =  gdf_results.append(gdf_nodes)

        if len(result.nodes) == 0:
            LOGGER.warning('empty result gdf returned.')

        return gdf_results.reset_index(drop=True)

    def _osm_geoms_to_gis(self, gdf):
        """
        convert lat / lon ordering of OSM convention back to conventional
        GIS ordering (x,y) instead of (lat / lon)

        Parameters
        ----------
        gdf : gpd.GeoDataFrame

        Returns
        -------
        gpd.GeoSeries
            Geometry series with swapped coordinates.
        """

        return gdf.geometry.map(lambda geometry:
                                shapely.ops.transform(
                                    lambda x, y: (y, x), geometry))

    def get_data_overpass(self, closed_lines_are_polys=True):
        """
        wrapper for all helper funcs to get & assemble data
        """

        query_clause = self._overpass_query_string()
        result = self._insistent_osm_api_query(query_clause)
        gdf_result = self._assemble_results(result, closed_lines_are_polys)
        gdf_result['geometry'] = self._osm_geoms_to_gis(gdf_result)

        return gdf_result
