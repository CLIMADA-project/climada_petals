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
import pandas as pd
import numpy as np
from pathlib import Path
import shapely
import subprocess
from tqdm import tqdm
import urllib.request

from climada import CONFIG
from climada.util import coordinates as u_coords

LOGGER = logging.getLogger(__name__)
gdal.SetConfigOption("OSM_CONFIG_FILE", str(Path(__file__).resolve().parent.joinpath('osmconf.ini')))
#gdal.SetConfigOption("OSM_CONFIG_FILE", '/Users/evelynm/climada_python/climada/entity/exposures/openstreetmap/osmconf.ini')

# =============================================================================
# Define constants
# =============================================================================
DATA_DIR = CONFIG.exposures.openstreetmap.local_data.dir()

LOGGER = logging.getLogger(__name__)

URL_GEOFABRIK = 'https://download.geofabrik.de/'

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
                               power='pole' or power='portal' or 
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
        
        
# =============================================================================
# Download entire regional map data from extracts (geofabrik only)
# =============================================================================

# TODO: make class osm_rawdata()

def _create_download_url(iso3, file_format):
    """create string with download-api from geofabrik
    iso3 : str
        ISO3 code of country to download
    file_format : str
        shp or pbf. Default is 'pbf'
    """
    
    if file_format == 'shp':
        return f'{URL_GEOFABRIK}{DICT_GEOFABRIK[iso3][0]}/{DICT_GEOFABRIK[iso3][1]}-latest-free.shp.zip'
    elif file_format == 'pbf':
        return f'{URL_GEOFABRIK}{DICT_GEOFABRIK[iso3][0]}/{DICT_GEOFABRIK[iso3][1]}-latest.osm.pbf'
    else:
        LOGGER.error('invalid file format. Please choose one of [shp, pbf]')

def get_data_geofab(iso3, file_format='pbf', save_path=DATA_DIR):
    """
    iso3 : str
        ISO3 code of country to download
        Exceptions: Russia is divided into European and Asian part ('RUS-E', 'RUS-A'),
        Canary Islands are 'IC'
    file_format : str
        'shp' or 'pbf'. Default is 'pbf'.
    """
    download_url = _create_download_url(iso3, file_format)
    local_filepath = save_path + '/' + download_url.split('/')[-1]
    if not Path(local_filepath).is_file():
        LOGGER.info(f'Downloading file as {local_filepath}')
        urllib.request.urlretrieve(download_url, local_filepath)
    else:
        LOGGER.info(f'file already exists as {local_filepath}')


# =============================================================================
# Download entire Planet from OSM and extract customized areas
# =============================================================================


def get_osm_planet(save_path=DATA_DIR):
    """
    This function will download the planet file from the OSM servers. 
    """
    download_url = 'https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf'
    local_filepath = save_path + '/planet-latest.osm.pbf'

    if not Path(local_filepath).is_file():
        LOGGER.info(f'Downloading file as {local_filepath}')
        urllib.request.urlretrieve(download_url, local_filepath)
    else:
        LOGGER.info(f'file already exists as {local_filepath}')

def osmosis_extract(planet_fp, dest_fp, shape):
    """
    planet_fp : str
        file path to planet.osm.pbf
    dest_fp : str
        file path to extracted_place.osm.pbf
    shape : list or str
        bounding box [xmin, ymin, xmax, ymax] or file path to a .poly file
        
    Note
    ----
    The function make_poly_file creates .poly files for a desired country / region.
    """
    if not Path(planet_fp).is_file():
        LOGGER.info("planet.osm file wasn't found. Downloading it.")
        get_osm_planet()

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

def make_poly_file(data_path, global_shape, regionalized=False):
    """
    Taken from https://github.com/ElcoK/osm_clipper:
    
    Create .poly files for all countries from a world shapefile. These
    .poly files are used to extract data from the openstreetmap osm.pbf files.
    This function is adapted from the OSMPoly function in QGIS.
    
    Parameters
    ----------
    data_path : 
        folder for poly files
        
    global_shape*: 
        exact path to the global shapefile used to create the poly files.
        Can be downloaded in .gpkg format from https://biogeo.ucdavis.edu/data/gadm3.6/gadm36_levels_gpkg.zip
        Contains also admin-x (sub-country) shapes.
        
    regionalized  : Default is **False**. Set to **True** will perform the analysis 
        on a regional level.
    
    Returns
    -------
    .poly file
        for each country in a new dir in the working directory.
    """     

    # Load country shapes and country list and only keep the required countries
    wb_poly = gpd.read_file(global_shape, crs={'init' :'epsg:4326'})
    
    # filter polygon file
    if regionalized == True:
        wb_poly = wb_poly.loc[wb_poly['GID_0'] != '-']
    else:
        wb_poly = wb_poly.loc[wb_poly['GID_0'] != '-']

    num = 0
    # iterate over the counties (rows) in the world shapefile
    for f in wb_poly.iterrows():
        f = f[1]
        num = num + 1
        geom=f.geometry

        # this will create a list of the different subpolygons
        if geom.geom_type == 'MultiPolygon':
            polygons = geom
        
        # the list will be lenght 1 if it is just one polygon
        elif geom.geom_type == 'Polygon':
            polygons = [geom]

        # define the name of the output file, based on the ISO3 code
        ctry = f['GID_0']
        if regionalized == True:
            attr=f['GID_1']
        else:
            attr=f['GID_0']
        
        # start writing the .poly file
        f = open(data_path + "/" + attr +'.poly', 'w')
        f.write(attr + "\n")

        i = 0
        
        # loop over the different polygons, get their exterior and write the 
        # coordinates of the ring to the .poly file
        for polygon in polygons:

            if ctry == 'CAN':
                dist =  u_coords.dist_approx(*reversed(polygon.centroid.coords[:1][0]), 83.24,-79.80)
                if dist < 2000:
                    continue
            if ctry == 'RUS':
                dist =  u_coords.dist_approx(*reversed(polygon.centroid.coords[:1][0]), 82.26,58.89)
                if dist < 500:
                    continue
                
            polygon = np.array(polygon.exterior)

            j = 0
            f.write(str(i) + "\n")

            for ring in polygon:
                j = j + 1
                f.write("    " + str(ring[0]) + "     " + str(ring[1]) +"\n")

            i = i + 1
            # close the ring of one subpolygon if done
            f.write("END" +"\n")

        # close the file when done
        f.write("END" +"\n")
        f.close()

# =============================================================================
# Querying and conversion to gdf
# =============================================================================


def query_builder(geo_type, constraint_dict):
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

def _retrieve(osm_path, geo_type, constraint_dict):
    """
    adapted from BenDickens/trails repo 
    (https://github.com/BenDickens/trails.git, see extract.py)
    Function to extract specified geometry and keys/values from an 
    OpenStreetMap osm.pbf file.
    
    Parameters
    ----------
    osm_path : str
        file path to the .osm.pbf file to extract info from.
    geo_type : str
        Type of geometry to retrieve. One of [points, lines, multipolygons]
    constraint_dict :  dict 
        
    Returns
    -------
    GeoDataFrame : a gpd GeoDataFrame 
        with all columns, geometries, and constraints specified.
    """
    driver = ogr.GetDriverByName('OSM')
    data = driver.Open(osm_path)
    query = query_builder(geo_type, constraint_dict)
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

def retrieve_cis(osm_path, feature):
    """
    
    Parameters
    ----------
    feature : str
        healthcare or education or telecom or water or food or oil or road or
        rail or power or wastewater or gas
    """
    # features consisting in points and multipolygon results:
    if feature in ['healthcare','education','food']:
        gdf = _retrieve(osm_path, 'points', OSM_CONSTRAINT_DICT[feature])
        gdf = gdf.append(
            _retrieve(osm_path, 'multipolygons', OSM_CONSTRAINT_DICT[feature]))
    # features consisting in multipolygon results:
    if feature in ['air']:
        gdf = _retrieve(osm_path, 'multipolygons', OSM_CONSTRAINT_DICT[feature])
        
    # features consisting in points, multipolygons and lines:
    elif feature in ['gas','oil','telecom','water,','wastewater','power',
                     'rail','road']:
        gdf = _retrieve(osm_path, 'points', OSM_CONSTRAINT_DICT[feature])
        gdf = gdf.append(
            _retrieve(osm_path, 'multipolygons', OSM_CONSTRAINT_DICT[feature]))
        gdf = gdf.append(
            _retrieve(osm_path, 'lines', OSM_CONSTRAINT_DICT[feature]))
    else:
        LOGGER.warning('feature not in OSM_CONSTRAINT_DICT')
    
    return gdf

# =============================================================================
# Download customized data from API (overpass-api)
# =============================================================================

def _query_overpass():
    # TODO: Implement
    pass

def get_data_overpass():
    # TODO: Implement
    pass