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

gdal.SetConfigOption("OSM_CONFIG_FILE", str(Path(__file__).resolve().parent.joinpath('osmconf.ini')))
gdal.SetConfigOption("OSM_CONFIG_FILE", '/Users/evelynm/climada_python/climada/entity/exposures/openstreetmap/osmconf.ini')

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
       'RUS-A' : ('asia', 'russia'),
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
       'RUS-E' : ('europe', 'russia'),           
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
            'amenity' : ["='school' or ",
                         "='kindergarten' or ",
                         "='college' or ",
                         "='university' or ",
                         "='childcare'"],
            'building' : ["='school' or ",
                          "='kindergarten' or ",
                          "='college' or ",
                          "='university' or ",
                          "='childcare'"]},
        'healthcare' : {
            'amenity' : ["='hospital' or ",
                         "='doctors'"],
            'building' : ["='hospital' or ",
                          "='doctors'"]},
        'water' : {
            'man_made' : ["='water_works' or ",
                          "='water_well' or ",
                          "='water_tower' or ",
                          "='wastewater_plant' or ",
                          "='reservoir_covered'"],
            'landuse' : ["='reservoir'"]},
        'telecom' : {
            'tower_type' : ["='communication'"]},
        'road' :  {
            'highway' : ["='primary' or ",
                         "='trunk' or ",
                         "='motorway' or ",
                         "='motorway_link' or ",
                         "='trunk_link' or ",
                         "='primary_link' or ",
                         "='secondary' or ",
                         "='secondary_link' or ",
                         "='tertiary' or ",
                         "='tertiary_link'"]},
        'rail' : {
            'service' : [" IS NOT NULL"]},
         'air' : {
             'aeroway' : ["='aerodrome'"]},
         'fuel' : {
             'amenity' : ["='fuel'"]},
         'food' : {
             'shop' : ["='supermarket' or ",
                       "='greengrocer' or ",
                       "='bakery'"]},
         'power' : {},
         }
        # TODO: issue with colon
        # original in osm query lang -> tower:type. Only works with tower_type but no results.
        #     'man_made' : ["='tower'"]}
        # ['tower_type','man_made'],**{"tower_type":[" IS NOT NULL"]
        # also tried:
        # 'tower\\:type'
        
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

# 
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

# =============================================================================
# Querying and conversion to gdf
# =============================================================================

def query_b(geoType,keyCol,**valConstraint):
    """
    from BenDickens/trails repo (https://github.com/BenDickens/trails.git, see
                                 extract.py)
    This function builds an SQL query from the values passed to the retrieve()
    function.
    
    Parameters
    ---------
    *geoType* : Type of geometry (osm layer) to search for.
    *keyCol* : A list of keys/columns that should be selected from the layer.
    ***valConstraint* : A dictionary of constraints for the values. e.g.
         WHERE 'value'>20 or 'value'='constraint'
    
    Returns
    -------
        *string: : a SQL query string.
    """
    query = "SELECT " + "osm_id"
    for a in keyCol: query+= ","+ a
    query += " FROM " + geoType + " WHERE "
    # If there are values in the dictionary, add constraint clauses
    if valConstraint:
        for a in [*valConstraint]:
            # For each value of the key, add the constraint
            for b in valConstraint[a]: query += a + b
        query+= " AND "
    # Always ensures the first key/col provided is not Null.
    query+= ""+str(keyCol[0]) +" IS NOT NULL"
    return query


def _retrieve(osm_path,geoType,keyCol,**valConstraint):
    """
    from BenDickens/trails repo (https://github.com/BenDickens/trails.git, see
                                 extract.py)
    Function to extract specified geometry and keys/values from OpenStreetMap
    Parameters
    ----------
        *osm_path* : file path to the .osm.pbf file of the region
        for which we want to do the analysis.
        *geoType* : Type of Geometry to retrieve. e.g. lines, multipolygons, etc.
        *keyCol* : These keys will be returned as columns in the dataframe.
        ***valConstraint: A dictionary specifiying the value constraints.
        A key can have multiple values (as a list) for more than one constraint for key/value.
    Returns
    -------
        *GeoDataFrame* : a gpd GeoDataFrame with all columns, geometries, and constraints specified.
    """
    driver=ogr.GetDriverByName('OSM')
    data = driver.Open(osm_path)
    query = query_b(geoType,keyCol,**valConstraint)
    sql_lyr = data.ExecuteSQL(query)
    features =[]
    # cl = columns
    cl = ['osm_id']
    for a in keyCol: cl.append(a)
    if data is not None:
        print('query is finished, lets start the loop')
        for feature in tqdm(sql_lyr,desc='extract'):
            try:
                if feature.GetField(keyCol[0]) is not None:
                    geom = shapely.wkb.loads(feature.geometry().ExportToWkb())
                    if geom is None:
                        continue
                    # field will become a row in the dataframe.
                    field = []
                    for i in cl: field.append(feature.GetField(i))
                    field.append(geom)
                    features.append(field)
            except:
                print("WARNING: skipped OSM feature")
    else:
        print("ERROR: Nonetype error when requesting SQL. Check required.")
    cl.append('geometry')
    if len(features) > 0:
        return gpd.GeoDataFrame(features,columns=cl)
    else:
        print("WARNING: No features or No Memory. returning empty GeoDataFrame")
        return gpd.GeoDataFrame(columns=['osm_id','geometry'])

def retrieve_cis(osm_path, feature):
    """
    
    Parameters
    ----------
    feature : str
        healthcare or education or telecom or water or food or fuel or road or rail or power
    """
    # TODO: implement sth smarter than pd.concat (sjoin from gpd) for duplicate points / multipolys
    if ((feature == 'healthcare') or (feature == 'education') or 
        (feature == 'telecom') or (feature == 'fuel') or (feature == 'food') or
        (feature == 'water')):
        # allow for several keys, and point or polygon
        gdf = gpd.GeoDataFrame()
        for key in OSM_CONSTRAINT_DICT[feature].keys():
            gdf = pd.concat([gdf, _retrieve(osm_path, 'points',[key],
                                              **{key : OSM_CONSTRAINT_DICT[feature][key]})]) 
            gdf = pd.concat([gdf, _retrieve(osm_path, 'multipolygons',[key],
                                              **{key : OSM_CONSTRAINT_DICT[feature][key]})]) #[key]
    elif feature == 'air':
        gdf = _retrieve(osm_path, 'multipolygons',['aeroway'],
                        **OSM_CONSTRAINT_DICT[feature]) 

    elif feature == 'rail':
        gdf = gpd.GeoDataFrame(_retrieve(osm_path, 'lines',['railway', 'service'],
                                          **OSM_CONSTRAINT_DICT[feature]))

    elif feature == 'road':
        gdf = gpd.GeoDataFrame(_retrieve(osm_path,'lines',
                                      ['highway','oneway','lanes','maxspeed'],
                                      **OSM_CONSTRAINT_DICT[feature]))

    elif feature == 'power':
        gdf = gpd.GeoDataFrame(_retrieve(osm_path,'lines',['power','voltage'],))
        gdf = pd.concat([gdf, _retrieve(osm_path,'points',['power','voltage'])
                         ]) #**{'voltage':[" IS NULL"]} # no further constraints
    
    return gpd.GeoDataFrame(gdf).reset_index(drop=True)
# =============================================================================
# Download customized data from API (overpass-api)
# =============================================================================

def _query_overpass():
    # TODO: Implement
    pass

def get_data_overpass():
    # TODO: Implement
    pass