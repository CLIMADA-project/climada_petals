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
import geopandas as gpd
import logging
from osgeo import ogr, gdal
from pathlib import Path
import shapely
import subprocess
from tqdm import tqdm
import urllib.request
import numpy as np
import itertools
import time
import overpy

from climada import CONFIG

LOGGER = logging.getLogger(__name__)
gdal.SetConfigOption("OSM_CONFIG_FILE",
                     str(Path(__file__).resolve().parent.joinpath('osmconf.ini'))) #"/Users/evelynm/climada_python/climada/entity/exposures/openstreetmap/osmconf.ini"

# =============================================================================
# Define constants
# =============================================================================
DATA_DIR = CONFIG.exposures.openstreetmap.local_data.dir()

# from osm_clipper (GitHub: https://github.com/ElcoK/osm_clipper)
DICT_GEOFABRIK = {
   'AFG' : ('asia','afghanistan'),
   'ALB' : ('europe','albania'),
   'DZA' : ('africa','algeria'),
   'AND' : ('europe','andorra'),
   'AGO' : ('africa','angola'),
   'BEN' : ('africa', 'benin'),
   'BWA' : ('africa', 'botswana'),
   'BFA' : ('africa', 'burkina-faso'),
   'BDI' : ('africa', 'burundi'),
   'CMR' : ('africa', 'cameroon'),
   'IC' : ('africa', 'canary-islands'),
   'CPV' : ('africa', 'cape-verde'),
   'CAF' : ('africa', 'central-african-republic'),
   'TCD' : ('africa', 'chad'),
   'COM' : ('africa', 'comores'),
   'COG' : ('africa', 'congo-brazzaville'),
   'COD' : ('africa', 'congo-democratic-republic'),
   'DJI' : ('africa', 'djibouti'),
   'EGY' : ('africa', 'egypt'),
   'GNQ' : ('africa', 'equatorial-guinea'),
   'ERI' : ('africa', 'eritrea'),
   'ETH' : ('africa', 'ethiopia'),
   'GAB' : ('africa', 'gabon'),
   'GMB' : ('africa', 'senegal-and-gambia'), #TOGETHER WITH SENEGAL
   'GHA' : ('africa', 'ghana'),
   'GIN' : ('africa', 'guinea'),
   'GNB' : ('africa', 'guinea-bissau'),
   'CIV' : ('africa', 'ivory-coast'),
   'KEN' : ('africa', 'kenya'),
   'LSO' : ('africa', 'lesotho'),
   'LBR' : ('africa', 'liberia'),
   'LBY' : ('africa', 'libya'),
   'MDG' : ('africa', 'madagascar'),
   'MWI' : ('africa', 'malawi'),
   'MLI' : ('africa', 'mali'),
   'MRT' : ('africa', 'mauritania'),
   'MAR' : ('africa', 'morocco'),
   'MOZ' : ('africa', 'mozambique'),
   'NAM' : ('africa', 'namibia'),
   'NER' : ('africa', 'niger'),
   'NGA' : ('africa', 'nigeria'),
   'RWA' : ('africa', 'rwanda'),
   'SHN' : ('africa', 'saint-helena-ascension-and-tristan-da-cunha'),
   'STP' : ('africa', 'sao-tome-and-principe'),
   'SEN' : ('africa', 'senegal-and-gambia'), #TOGETHER WITH THE GAMBIA
   'SYC' : ('africa', 'seychelles'),
   'SLE' : ('africa', 'sierra-leone'),
   'SOM' : ('africa', 'somalia'),
   'ZAF' : ('africa', 'south-africa'),
   'SDN' : ('africa', 'sudan'),
   'SSD' : ('africa', 'south-sudan'),
   'SWZ' : ('africa', 'swaziland'),
   'TZA' : ('africa', 'tanzania'),
   'TGO' : ('africa', 'togo'),
   'TUN' : ('africa', 'tunisia'),
   'UGA' : ('africa', 'uganda'),
   'ZMB' : ('africa', 'zambia'),
   'ZWE' : ('africa', 'zimbabwe'),
   'ARM' : ('asia', 'armenia'),
   'AZE' : ('asia', 'azerbaijan'),
   'BGD' : ('asia', 'bangladesh'),
   'BTN' : ('asia', 'bhutan'),
   'KHM' : ('asia', 'cambodia'),
   'CHN' : ('asia', 'china'),
   'SAU' : ('asia', 'gcc-states'), #Together with Kuwait, the United Arab Emirates, Qatar, Bahrain, and Oman
   'KWT' : ('asia', 'gcc-states'), #Together with Saudi Arabia, the United Arab Emirates, Qatar, Bahrain, and Oman
   'ARE' : ('asia', 'gcc-states'), #Together with Saudi Arabia, Kuwait, Qatar, Bahrain, and Oman
   'QAT' : ('asia', 'gcc-states'), #Together with Saudi Arabia, Kuwait, the United Arab Emirates, Bahrain, and Oman
   'OMN' : ('asia', 'gcc-states'), #Together with Saudi Arabia, Kuwait, the United Arab Emirates, Qatar and Oman
   'BHR' : ('asia', 'gcc-states'), #Together with Saudi Arabia, Kuwait, the United Arab Emirates, Qatar and Bahrain
   'IND' : ('asia', 'india'),
   'IDN' : ('asia', 'indonesia'),
   'IRN' : ('asia', 'iran'),
   'IRQ' : ('asia', 'iraq'),
   'ISR' : ('asia', 'israel-and-palestine'),       # TOGETHER WITH PALESTINE
   'PSE' : ('asia', 'israel-and-palestine'),       # TOGETHER WITH ISRAEL
   'JPN' : ('asia', 'japan'),
   'JOR' : ('asia', 'jordan'),
   'KAZ' : ('asia', 'kazakhstan'),
   'KGZ' : ('asia', 'kyrgyzstan'),
   'LAO' : ('asia', 'laos'),
   'LBN' : ('asia', 'lebanon'),
   'MYS' : ('asia', 'malaysia-singapore-brunei'), # TOGETHER WITH SINGAPORE AND BRUNEI
   'SGP' : ('asia', 'malaysia-singapore-brunei'), # TOGETHER WITH MALAYSIA AND BRUNEI
   'BRN' : ('asia', 'malaysia-singapore-brunei'), # TOGETHER WITH MALAYSIA AND SINGAPORE
   'MDV' : ('asia', 'maldives'),
   'MNG' : ('asia', 'mongolia'),
   'MMR' : ('asia', 'myanmar'),
   'NPL' : ('asia', 'nepal'),
   'PRK' : ('asia', 'north-korea'),
   'PAK' : ('asia', 'pakistan'),
   'PHL' : ('asia', 'philippines'),
   'RUS-A' : ('asia', 'russia'), # Asian part of Russia
   'KOR' : ('asia', 'south-korea'),
   'LKA' : ('asia', 'sri-lanka'),
   'SYR' : ('asia', 'syria'),
   'TWN' : ('asia', 'taiwan'),
   'TJK' : ('asia', 'tajikistan'),
   'THA' : ('asia', 'thailand'),
   'TKM' : ('asia', 'turkmenistan'),
   'UZB' : ('asia', 'uzbekistan'),
   'VNM' : ('asia', 'vietnam'),
   'YEM' : ('asia', 'yemen'),
   'BHS' : ('central-america', 'bahamas'),
   'BLZ' : ('central-america', 'belize'),
   'CUB' : ('central-america', 'cuba'),
   'GTM' : ('central-america', 'guatemala'),
   'HTI' : ('central-america', 'haiti-and-domrep'),  # TOGETHER WITH DOMINICAN REPUBLIC
   'DOM' : ('central-america', 'haiti-and-domrep'),  # TOGETHER WITH HAITI
   'JAM' : ('central-america', 'jamaica'),
   'HND' : ('central-america', 'honduras'),
   'NIC' : ('central-america', 'nicaragua'),
   'SLV' : ('central-america', 'el-salvador'),
   'CRI' : ('central-america', 'costa-rica'),
   'AUT' : ('europe', 'austria'),
   'BLR' : ('europe', 'belarus'),
   'BEL' : ('europe', 'belgium'),
   'BIH' : ('europe', 'bosnia-herzegovina'),
   'BGR' : ('europe', 'bulgaria'),
   'HRV' : ('europe', 'croatia'),
   'CYP' : ('europe', 'cyprus'),
   'CZE' : ('europe', 'czech-republic'),
   'DNK' : ('europe', 'denmark'),
   'EST' : ('europe', 'estonia'),
   'FRO' : ('europe', 'faroe-islands'),
   'FIN' : ('europe', 'finland'),
   'FRA' : ('europe', 'france'),
   'GEO' : ('europe', 'georgia'),
   'DEU' : ('europe', 'germany'),
   'GBR' : ('europe', 'great-britain'),        # DOES NOT INCLUDE NORTHERN ISLAND
   'GRC' : ('europe', 'greece'),
   'HUN' : ('europe', 'hungary'),
   'ISL' : ('europe', 'iceland'),
   'IRL' : ('europe', 'ireland-and-northern-ireland'),
   'IMN' : ('europe', 'isle-of-man'),
   'ITA' : ('europe', 'italy'),
   'LVA' : ('europe', 'latvia'),
   'LIE' : ('europe', 'liechtenstein'),
   'LTU' : ('europe', 'lithuania'),
   'LUX' : ('europe', 'luxembourg'),
   'MKD' : ('europe', 'macedonia'),
   'MLT' : ('europe', 'malta'),
   'MDA' : ('europe', 'moldova'),
   'MCO' : ('europe', 'monaco'),
   'MNE' : ('europe', 'montenegro'),
   'NLD' : ('europe', 'netherlands'),
   'NOR' : ('europe', 'norway'),
   'POL' : ('europe', 'poland'),
   'PRT' : ('europe', 'portugal'),
   'ROU' : ('europe', 'romania'),
   'RUS-E' : ('europe', 'russia'), # European part of Russia
   'SRB' : ('europe', 'serbia'),
   'SVK' : ('europe', 'slovakia'),
   'SVN' : ('europe', 'slovenia'),
   'ESP' : ('europe', 'spain'),
   'SWE' : ('europe', 'sweden'),
   'CHE' : ('europe', 'switzerland'),
   'TUR' : ('europe', 'turkey'),
   'UKR' : ('europe', 'ukraine'),
   'CAN' : ('north-america', 'canada'),
   'GRL' : ('north-america', 'greenland'),
   'MEX' : ('north-america', 'mexico'),
   'USA' : ('north-america', 'us'),
   'AUS' : ('australia-oceania', 'australia'),
   'COK' : ('australia-oceania', 'cook-islands'),
   'FJI' : ('australia-oceania', 'fiji'),
   'KIR' : ('australia-oceania', 'kiribati'),
   'MHL' : ('australia-oceania', 'marshall-islands'),
   'FSM' : ('australia-oceania', 'micronesia'),
   'NRU' : ('australia-oceania', 'nauru'),
   'NCL' : ('australia-oceania', 'new-caledonia'),
   'NZL' : ('australia-oceania', 'new-zealand'),
   'NIU' : ('australia-oceania', 'niue'),
   'PLW' : ('australia-oceania', 'palau'),
   'PNG' : ('australia-oceania', 'papua-new-guinea'),
   'WSM' : ('australia-oceania', 'samoa'),
   'SLB' : ('australia-oceania', 'solomon-islands'),
   'TON' : ('australia-oceania', 'tonga'),
   'TUV' : ('australia-oceania', 'tuvalu'),
   'VUT' : ('australia-oceania', 'vanuatu'),
   'ARG' : ('south-america', 'argentina'),
   'BOL' : ('south-america', 'bolivia'),
   'BRA' : ('south-america', 'brazil'),
   'CHL' : ('south-america', 'chile'),
   'COL' : ('south-america', 'colombia'),
   'ECU' : ('south-america', 'ecuador'),
   'PRY' : ('south-america', 'paraguay'),
   'PER' : ('south-america', 'peru'),
   'SUR' : ('south-america', 'suriname'),
   'URY' : ('south-america', 'uruguay'),
   'VEN' : ('south-america', 'venezuela'),
}

OSM_CONSTRAINT_DICT = {
        'education' : {
            'osm_keys' : ['amenity','building','name'],
            'osm_query' : """building='school' or amenity='school' or
                             building='kindergarten' or amenity='kindergarten' or
                             building='college' or amenity='college' or
                             building='university' or amenity='university' or
                             building='college' or amenity='college' or
                             building='childcare' or amenity='childcare'"""},
        'healthcare' : {
            'osm_keys' : ['amenity','building','healthcare','name'],
            'osm_query' : """amenity='hospital' or healthcare='hospital' or building='hospital' or
                             amenity='clinic' or healthcare='clinic' or building='clinic' or
                             amenity='doctors' or healthcare='doctors' or
                             amenity='dentist' or healthcare='dentist' or
                             amenity='pharmacy' or
                             amenity='nursing_home' or
                             healthcare='*'"""},
        'water' : {
            'osm_keys' : ['man_made','pump','pipeline','emergency','name'],
            'osm_query' : """man_made='water_well' or man_made='water_works' or
                             man_made='water_tower' or
                             man_made='reservoir_covered' or landuse='reservoir' or
                             (man_made='pipeline' and substance='water') or
                             (pipeline='substation' and substance='water') or
                             pump='powered' or pump='manual' or pump='yes' or
                             emergency='fire_hydrant' or
                             (man_made='storage_tank' and content='water')"""},
        'telecom' : {
            'osm_keys' : ['man_made','tower_type','telecom','communication_mobile_phone','name'],
            'osm_query' : """tower_type='communication' or man_made='mast' or
                             communication_mobile_phone='*' or telecom='antenna' or
                             telecom='poles' or communication='pole' or
                             telecom='central_office' or telecom='street_cabinet' or
                             telecom='exchange' or telecom='data_center' or
                             telecom='distribution_point' or telecom='connection_point' or
                             telecom='line' or communication='line' or
                             utility='telecom'"""},# original in osm query lang.: tower:type -> Only works with tower_type
        'road' :  {
            'osm_keys' : ['highway','man_made','public_transport','bus','name'],
            'osm_query' : """highway='motorway' or highway='motorway_link' or
                             highway='trunk' or highway='trunk_link' or
                             highway='primary' or highway='primary_link' or
                             highway='secondary' or highway='secondary_link' or
                             highway='tertiary' or highway='tertiary_link' or
                             highway='residential' or highway='road' or
                             highway='service' or highway='unclassified' or
                             highway='traffic_signals' or
                             (public_transport='*' and bus='yes') or
                             man_made='bridge' or man_made='tunnel'"""},
        'rail' : {
            'osm_keys' : ['railway','name'],
            'osm_query' : """railway='rail' or railway='tram' or
                             railway='subway' or railway='narrow_gauge' or
                             railway='light_rail' or
                             railway='station' or railway='platform' or
                             railway='stop' or railway='tram_stop' or
                             railway='signal' or railway='switch'"""},
         'air' : {
             'osm_keys' : ['aeroway','name'],
             'osm_query' : """aeroway='aerodrome'"""},
         'gas' : {
             'osm_keys' : ['man_made','pipeline', 'utility','name'],
             'osm_query' : """(man_made='pipeline' and substance='gas') or
                              (pipeline='substation' and substance='gas') or
                              (man_made='storage_tank' and content='gas') or
                              utility='gas'"""},
        'oil' : {
             'osm_keys' : ['pipeline','man_made','amenity','name'],
             'osm_query' : """(pipeline='substation' and substance='oil') or
                              (man_made='pipeline' and substance='oil') or
                              man_made='petroleum_well' or man_made='oil_refinery' or
                              amenity='fuel'"""},

        'power' : {
              'osm_keys' : ['power','voltage','utility','name'],
              'osm_query' : """power='line' or power='cable' or power='minor_line' or
                               power='plant' or
                               power='generator' or power='substation' or
                               power='transformer' or
                               power='pole' or power='portal' or power='tower' or
                               power='terminal' or power='switch' or
                               power='catenary_mast' or
                               utility='power'"""},
        'wastewater' : {
              'osm_keys' : ['reservoir_type','man_made','utility','natural','name'],
              'osm_query' : """reservoir_type='sewage' or
                               (man_made='storage_tank' and content='sewage') or
                               (man_made='pipeline' and substance='sewage') or
                               substance='waterwaste' or substance='wastewater' or
                               (natural='water' and water='wastewater') or
                               man_made='wastewater_plant' or
                               man_made='wastewater_tank' or
                               utility='sewerage'"""},
         'food' : {
             'osm_keys' : ['shop','name'],
             'osm_query' : """shop='supermarket' or shop='greengrocer' or
                              shop='grocery' or shop='general' or shop='bakery'"""},
         }


class OSMRaw:
    """functions to obtain entire raw datasets from OSM, from differnt sources"""

    def __init__(self):
        self.geofabrik_url = 'https://download.geofabrik.de/'
        self.planet_url = 'https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf'

    def _create_gf_download_url(self, iso3, file_format):
        """create string with download-api from geofabrik
        iso3 : str
            ISO3 code of country to download
        file_format : str
            shp or pbf. Default is 'pbf'
        """

        if file_format == 'shp':
            return f'{self.geofabrik_url}{DICT_GEOFABRIK[iso3][0]}/{DICT_GEOFABRIK[iso3][1]}-latest-free.shp.zip'
        elif file_format == 'pbf':
            return f'{self.geofabrik_url}{DICT_GEOFABRIK[iso3][0]}/{DICT_GEOFABRIK[iso3][1]}-latest.osm.pbf'
        else:
            LOGGER.error('invalid file format. Please choose one of [shp, pbf]')

    def get_data_geofabrik(self, iso3, file_format='pbf', save_path=DATA_DIR):
        """
        iso3 : str
            ISO3 code of country to download
            Exceptions: Russia is divided into European and Asian part ('RUS-E', 'RUS-A'),
            Canary Islands are 'IC'
        file_format : str
            'shp' or 'pbf'. Default is 'pbf'.
        """
        download_url = self._create_gf_download_url(iso3, file_format)
        local_filepath = save_path + '/' + download_url.split('/')[-1]
        if not Path(local_filepath).is_file():
            LOGGER.info(f'Downloading file as {local_filepath}')
            urllib.request.urlretrieve(download_url, local_filepath)
        else:
            LOGGER.info(f'file already exists as {local_filepath}')

    def _get_osm_planet(self, save_path=DATA_DIR):
        """
        This function will download the planet file from the OSM servers.
        """
        local_filepath = save_path + '/planet-latest.osm.pbf'

        if not Path(local_filepath).is_file():
            LOGGER.info(f'Downloading file as {local_filepath}')
            urllib.request.urlretrieve(self.planet_url, local_filepath)
        else:
            LOGGER.info(f'file already exists as {local_filepath}')

    def _osmosis_extract(self, shape, planet_fp, dest_fp=DATA_DIR):
        """
        planet_fp : str
            file path to planet.osm.pbf
        dest_fp : str
            file path to extracted_place.osm.pbf
        shape : list or str
            bounding box [xmin, ymin, xmax, ymax] or file path to a .poly file
        """

        if not Path(dest_fp).is_file():
            print('file doesnt exist yet')
            if (isinstance(shape, list) or isinstance(shape, tuple)):
                print('Cutting from bounding box')
                cmd = ['osmosis', '--read-pbf', 'file='+planet_fp, '--bounding-box',
                       f'top={shape[3]}', f'left={shape[0]}', f'bottom={shape[1]}',
                       f'right={shape[2]}', '--write-pbf', 'file='+dest_fp]
            elif isinstance(shape, str):
                print('Cutting from poly file')
                cmd = ['osmosis', '--read-pbf', 'file='+planet_fp, '--bounding-polygon',
                       'file='+shape, '--write-pbf', 'file='+dest_fp]
            print('Generating extract from planet file.')
            LOGGER.info('Generating extract from planet file.')
            return subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        else:
            LOGGER.info("Extracted file already exists!")
            return None

    def get_data_planetextract(self, shape,
                               path_planet=DATA_DIR, path_extract=DATA_DIR):
        """
        get OSM raw data from a custom shape / bounding-box, which is extracted
        from the entire OSM planet file. Accepts bbox lists or .poly files for
        non-rectangular shapes.

        Note
        ----
        For more info on what .poly files are (incl. several tools for
        creating them), see https://wiki.openstreetmap.org/wiki/Osmosis/Polygon_Filter_File_Format

        For creating .poly files on admin0 to admin3 levels of any place on the
        globe, see the GitHub repo https://github.com/ElcoK/osm_clipper
        (especially the function make_poly_file())
        """

        if not Path(path_planet+'/planet-latest.osm.pbf').is_file():
            LOGGER.info("planet-latest.osm.pbf wasn't found. Downloading it.")
            self._get_osm_planet(path_planet)

        self._osmosis_extract(shape, path_planet, path_extract)


class OSMFileQuery:
    """
    Query features from raw OSM files.
    """
    def __init__(self, osm_path):
        """
        Parameters
        ----------
        osm_path : str
            file path to the .osm.pbf file to extract info from.
        """
        self.osm_path = osm_path

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

    def retrieve(self, geo_type, constraint_dict):
        """
        adapted from BenDickens/trails repo
        (https://github.com/BenDickens/trails.git, see extract.py)
        Function to extract specified geometry and keys/values from an
        OpenStreetMap osm.pbf file.

        Parameters
        ----------
        geo_type : str
            Type of geometry to retrieve. One of [points, lines, multipolygons]
        constraint_dict :  dict
            A dict with the keys "osm_keys" and "osm_query". osm_keys contains
            a list with all the osm keys that should be reported in the output
            gdf. osm_query contains an osm query string of the syntax
            "key(='value') (and/or further queries)".
            See examples in OSM_CONSTRAINT_DICT in case of doubt.

        Returns
        -------
        GeoDataFrame : a gpd GeoDataFrame
            with all reqiested osm_keys as columns, incl. geometries of the
            queried features.

        See also
        --------
        https://taginfo.openstreetmap.org/ to check what keys and key/value
        pairs are valid.
        https://overpass-turbo.eu/ for a direct visual output of the query,
        and to quickly check the validity. The wizard can help you find the
        correct keys / values you are looking for.
        """
        driver = ogr.GetDriverByName('OSM')
        data = driver.Open(self.osm_path)
        query = self._query_builder(geo_type, constraint_dict)
        sql_lyr = data.ExecuteSQL(query)
        features = []
        if data is not None:
            LOGGER.info('query is finished, lets start the loop')
            for feature in tqdm(sql_lyr, desc=f'extract {geo_type}'):
                try:
                    fields = []
                    for key in ['osm_id', *constraint_dict['osm_keys']]:
                        fields.append(feature.GetField(key))
                    geom = shapely.wkb.loads(feature.geometry().ExportToWkb())
                    if geom is None:
                        continue
                    fields.append(geom)
                    features.append(fields)
                except:
                    LOGGER.warning("skipped OSM feature")
        else:
            LOGGER.error("Nonetype error when requesting SQL. Check required.")

        return gpd.GeoDataFrame(
            features, columns=['osm_id', *constraint_dict['osm_keys'], 'geometry'])

    def retrieve_cis(self, ci_type):
        """

        Parameters
        ----------
        ci_type : str
            healthcare or education or telecom or water or food or oil or road or
            rail or power or wastewater or gas
        """
        # features consisting in points and multipolygon results:
        if ci_type in ['healthcare','education','food']:
            gdf = self.retrieve('points', OSM_CONSTRAINT_DICT[ci_type])
            gdf = gdf.append(
                self.retrieve('multipolygons', OSM_CONSTRAINT_DICT[ci_type]))
        # features consisting in multipolygon results:
        elif ci_type in ['air']:
            gdf = self.retrieve('multipolygons', OSM_CONSTRAINT_DICT[ci_type])

        # features consisting in points, multipolygons and lines:
        elif ci_type in ['gas','oil','telecom','water','wastewater','power',
                         'rail','road']:
            gdf = self.retrieve('points', OSM_CONSTRAINT_DICT[ci_type])
            gdf = gdf.append(
                self.retrieve('multipolygons', OSM_CONSTRAINT_DICT[ci_type]))
            gdf = gdf.append(
                self.retrieve('lines', OSM_CONSTRAINT_DICT[ci_type]))
        else:
            LOGGER.warning('feature not in OSM_CONSTRAINT_DICT. Returning empty gdf')
            gdf = gpd.GeoDataFrame()
        return gdf

class OSMApiQuery:
    """
    Query features directly via the overpass turbo API.
    """

    def __init__(self, area, condition):
        self.area = area
        self.condition = condition

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
                    print(' WARNING: too many Overpass API requests - try again in {} seconds'.format(
                        waiting_period))
                else:
                    raise Exception("Overpass API is consistently unavailable")
            except Exception as exc:
                if waiting_period < end_of_patience:
                    print(' WARNING: !!!!\n {}\n try again in {} seconds'.format(exc, waiting_period))
                else:
                    raise Exception("The Overpass API is consistently unavailable")
            time.sleep(waiting_period)
            waiting_period *= 2

    def _overpass_query_string(self):
        return f'[out:json][timeout:180];(nwr{self.condition}{self.area};(._;>;););out;'

    def _assemble_from_relations(self, result):

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

            # separate relation into polygons and linestrings, then combine into 1 multipolygon
            inner_polys = gdf_polys.geometry[
                (gdf_polys.geometry.type=='Polygon') &
                (gdf_polys.role=='inner')].values
            outer_polys = gdf_polys.geometry[
                (gdf_polys.geometry.type=='Polygon') &
                (gdf_polys.role=='outer')].values
            lines = gdf_polys.geometry[
                (gdf_polys.geometry.type=='LineString')].values
            # mp from (outer polygons --> outer mp) - (inner polygons --> mp inner)
            mp1 = shapely.geometry.MultiPolygon(outer_polys) - \
                  shapely.geometry.MultiPolygon(inner_polys)
            # poly from linestrings (lines --> multilines --> line --> polygon)
            p2 = shapely.geometry.Polygon(
                shapely.ops.linemerge(shapely.geometry.MultiLineString(lines)))

            data_geom.append(shapely.ops.unary_union([mp1, p2]))

        gdf_rels = gpd.GeoDataFrame(
            data=np.array([data_id,data_geom,data_tags]).T,
            columns=['osm_id','geometry','tags'])

        # list of lists into list:
        nodes_taken = list(itertools.chain.from_iterable(nodes_taken))
        ways_taken = list(itertools.chain.from_iterable(ways_taken))

        return nodes_taken, ways_taken, gdf_rels

    def _assemble_from_ways(self, result, ways_avail, closed_lines_are_polys):

        """assemble gdfs from ways, """

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
            data=np.array([data_id,data_geom,data_tags]).T,
            columns=['osm_id','geometry','tags'])

        return nodes_taken, gdf_ways

    def _assemble_from_nodes(self, result, nodes_avail):

        data_geom = []
        data_id = []
        data_tags = []

        for node in result.nodes:
            if node.id in nodes_avail:
                data_geom.append(shapely.geometry.Point(node.lat, node.lon))
                data_id.append(node.id)
                data_tags.append(node.tags)

        gdf_nodes = gpd.GeoDataFrame(
            data=np.array([data_id,data_geom,data_tags]).T,
            columns=['osm_id','geometry','tags'])

        return gdf_nodes

    def _update_availability(self, full_set, to_remove):
        return [item for item in full_set if item not in to_remove]

    def _assemble_results(self, result, closed_lines_are_polys=True):

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


    def get_data_overpass(self, closed_lines_are_polys=True):
       """wrapper for all helper funcs to get & assemble data"""
       query_clause = self._overpass_query_string()
       result = self._insistent_osm_api_query(query_clause)
       return self._assemble_results(result, closed_lines_are_polys)