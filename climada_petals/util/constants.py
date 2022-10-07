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

Define constants.
"""

__all__ = ['SYSTEM_DIR',
           'DEMO_DIR',
           'ENT_DEMO_TODAY',
           'ENT_DEMO_FUTURE',
           'HAZ_DEMO_MAT',
           'HAZ_DEMO_FL',
           'HAZ_DEMO_FLDDPH',
           'HAZ_DEMO_FLDFRC',
           'ENT_TEMPLATE_XLS',
           'HAZ_TEMPLATE_XLS',
           'ONE_LAT_KM',
           'EARTH_RADIUS_KM',
           'GLB_CENTROIDS_MAT',
           'GLB_CENTROIDS_NC',
           'ISIMIP_GPWV3_NATID_150AS',
           'NATEARTH_CENTROIDS',
           'DEMO_GDP2ASSET',
           'RIVER_FLOOD_REGIONS_CSV',
           'TC_ANDREW_FL',
           'HAZ_DEMO_H5',
           'EXP_DEMO_H5',
           'WS_DEMO_NC']

from climada.util.constants import *


HAZ_DEMO_FLDDPH = DEMO_DIR.joinpath('flddph_2000_DEMO.nc')
"""NetCDF4 Flood depth from isimip simulations"""

HAZ_DEMO_FLDFRC = DEMO_DIR.joinpath('fldfrc_2000_DEMO.nc')
"""NetCDF4 Flood fraction from isimip simulations"""

DEMO_GDP2ASSET = DEMO_DIR.joinpath('gdp2asset_CHE_exposure.nc')
"""Exposure demo file for GDP2Asset"""


"""
dictionary for the generation of the correct download api-address at
geofabrik.de, relating ISO3-country codes to the region & written-out name.
Adapted from the GitHub repo osm_clipper (https://github.com/ElcoK/osm_clipper)
Used by OSMRaw().get_data_geofabrik().

Note: A few small countries will be downloaded as a multi-country file, as
indicated in the comments.

Note: "special" ISO-3 codes - Canary Islands (IC), Asian part of Russia (RUS-A),
European part of Russia (RUS-E)
"""
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
   'GMB' : ('africa', 'senegal-and-gambia'),  #TOGETHER WITH SENEGAL
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
   'SEN' : ('africa', 'senegal-and-gambia'),  #TOGETHER WITH THE GAMBIA
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
   'SAU' : ('asia', 'gcc-states'),  #Together with Kuwait, the United Arab Emirates, Qatar, Bahrain, and Oman
   'KWT' : ('asia', 'gcc-states'),  #Together with Saudi Arabia, the United Arab Emirates, Qatar, Bahrain, and Oman
   'ARE' : ('asia', 'gcc-states'),  #Together with Saudi Arabia, Kuwait, Qatar, Bahrain, and Oman
   'QAT' : ('asia', 'gcc-states'),  #Together with Saudi Arabia, Kuwait, the United Arab Emirates, Bahrain, and Oman
   'OMN' : ('asia', 'gcc-states'),  #Together with Saudi Arabia, Kuwait, the United Arab Emirates, Qatar and Oman
   'BHR' : ('asia', 'gcc-states'),  #Together with Saudi Arabia, Kuwait, the United Arab Emirates, Qatar and Bahrain
   'IND' : ('asia', 'india'),
   'IDN' : ('asia', 'indonesia'),
   'IRN' : ('asia', 'iran'),
   'IRQ' : ('asia', 'iraq'),
   'ISR' : ('asia', 'israel-and-palestine'),  # TOGETHER WITH PALESTINE
   'PSE' : ('asia', 'israel-and-palestine'),  # TOGETHER WITH ISRAEL
   'JPN' : ('asia', 'japan'),
   'JOR' : ('asia', 'jordan'),
   'KAZ' : ('asia', 'kazakhstan'),
   'KGZ' : ('asia', 'kyrgyzstan'),
   'LAO' : ('asia', 'laos'),
   'LBN' : ('asia', 'lebanon'),
   'MYS' : ('asia', 'malaysia-singapore-brunei'),  # TOGETHER WITH SINGAPORE AND BRUNEI
   'SGP' : ('asia', 'malaysia-singapore-brunei'),  # TOGETHER WITH MALAYSIA AND BRUNEI
   'BRN' : ('asia', 'malaysia-singapore-brunei'),  # TOGETHER WITH MALAYSIA AND SINGAPORE
   'MDV' : ('asia', 'maldives'),
   'MNG' : ('asia', 'mongolia'),
   'MMR' : ('asia', 'myanmar'),
   'NPL' : ('asia', 'nepal'),
   'PRK' : ('asia', 'north-korea'),
   'PAK' : ('asia', 'pakistan'),
   'PHL' : ('asia', 'philippines'),
   'RUS-A' : ('asia', 'russia'),  # Asian part of Russia
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
   'GBR' : ('europe', 'great-britain'),  # DOES NOT INCLUDE NORTHERN ISLAND
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
   'RUS-E' : ('europe', 'russia'),  # European part of Russia
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
   'PCN' : ('australia-oceania', 'pitcairn-islands'),
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

"""
nested dictionary that contains collections of relevant columns (osm_keys) and
key - value pairs (osm_query) to extract critical infrastructure data from an
osm.pbf file, via the function OSM_FileQuery().retrieve_cis()

Currently implemented for:
    * educational facilities,
    * electric power,
    * food supply,
    * healthcare facilities,
    * natural gas infrastructure,
    * oil infrastructure,
    * road,
    * rail,
    * telecommunications,
    * water supply,
    * wastewater.

Note: If modified, make sure that key exists in osm.config file, under the
respective geometry/-ies.
"""
DICT_CIS_OSM =  {
        'education' : {
            'osm_keys' : ['amenity','building','name'],
            'osm_query' : """building='school' or amenity='school' or
                             building='kindergarten' or 
                             amenity='kindergarten' or
                             building='college' or amenity='college' or
                             building='university' or amenity='university' or
                             building='college' or amenity='college' or
                             building='childcare' or amenity='childcare'"""},
        'healthcare' : {
            'osm_keys' : ['amenity','building','healthcare','name'],
            'osm_query' : """amenity='hospital' or healthcare='hospital' or
                             building='hospital' or building='clinic' or
                             amenity='clinic' or healthcare='clinic' or 
                             amenity='doctors' or healthcare='doctors'
                             """},
        'water' : {
            'osm_keys' : ['man_made','pump','pipeline','emergency','name'],
            'osm_query' : """man_made='water_well' or man_made='water_works' or
                             man_made='water_tower' or
                             man_made='reservoir_covered' or 
                             landuse='reservoir' or
                             (man_made='pipeline' and substance='water') or
                             (pipeline='substation' and substance='water') or
                             pump='powered' or pump='manual' or pump='yes' or
                             emergency='fire_hydrant' or
                             (man_made='storage_tank' and content='water')"""},
        'telecom' : {
            'osm_keys' : ['man_made','tower_type','telecom',
                          'communication_mobile_phone','name'],
            'osm_query' : """tower_type='communication' or man_made='mast' or
                             communication_mobile_phone='*' or
                             telecom='antenna' or
                             telecom='poles' or communication='pole' or
                             telecom='central_office' or 
                             telecom='street_cabinet' or
                             telecom='exchange' or telecom='data_center' or
                             telecom='distribution_point' or 
                             telecom='connection_point' or
                             telecom='line' or communication='line' or
                             utility='telecom'"""},
        'road' :  {
            'osm_keys' : ['highway','man_made','name'],
            'osm_query' : """highway in ('motorway', 'motorway_link', 'trunk', 'trunk_link',
                            'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary',
                            'tertiary_link', 'residential', 'road', 'unclassified')
                             """},
        'main_road' :  {
            'osm_keys' : ['highway','man_made','name'],
            'osm_query' : """highway in ('primary', 'primary_link', 'secondary',
                             'secondary_link', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link', 
                             'motorway', 'motorway_link')
                            """},
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
                              man_made='petroleum_well' or 
                              man_made='oil_refinery' or
                              amenity='fuel'"""},
        'power' : {
              'osm_keys' : ['power','voltage','utility','name'],
              'osm_query' : """power='line' or power='cable' or
                               power='minor_line' or power='plant' or
                               power='generator' or power='substation' or
                               power='transformer' or
                               power='pole' or power='portal' or 
                               power='tower' or power='terminal' or 
                               power='switch' or power='catenary_mast' or
                               utility='power'"""},
        'wastewater' : {
              'osm_keys' : ['reservoir_type','man_made','utility','natural',
                            'name'],
              'osm_query' : """reservoir_type='sewage' or
                               (man_made='storage_tank' and content='sewage') or
                               (man_made='pipeline' and substance='sewage') or
                               substance='waterwaste' or 
                               substance='wastewater' or
                               (natural='water' and water='wastewater') or
                               man_made='wastewater_plant' or
                               man_made='wastewater_tank' or
                               utility='sewerage'"""},
         'food' : {
             'osm_keys' : ['shop','name'],
             'osm_query' : """shop='supermarket' or shop='greengrocer' or
                              shop='grocery' or shop='general' or 
                              shop='bakery'"""},
                              
        'buildings' : {
            'osm_keys' : ['building','amenity','name'],
            'osm_query' : """building='yes' or building='house' or 
                            building='residential' or building='detached' or 
                            building='hut' or building='industrial' or 
                            building='shed' or building='apartments'"""}
                              }
        
DICT_SPEEDS =  {'BTN': 38.0,
 'NPL': 40.0,
 'TLS': 40.0,
 'BGD': 41.0,
 'HTI': 41.0,
 'NIC': 46.0,
 'RWA': 47.0,
 'BOL': 50.0,
 'LKA': 50.0,
 'GIN': 50.0,
 'BDI': 51.0,
 'VNM': 51.0,
 'MDG': 51.0,
 'TTO': 51.0,
 'TJK': 52.0,
 'PHL': 52.0,
 'GMB': 53.0,
 'GTM': 53.0,
 'CRI': 55.0,
 'IDN': 55.0,
 'NGA': 55.0,
 'YEM': 55.0,
 'KHM': 55.0,
 'GHA': 56.0,
 'SLV': 56.0,
 'MNG': 56.0,
 'HND': 56.0,
 'CMR': 56.0,
 'TZA': 57.0,
 'AFG': 57.0,
 'BIH': 57.0,
 'ARM': 57.0,
 'COL': 57.0,
 'KEN': 57.0,
 'IND': 58.0,
 'SOM': 58.0,
 'SSD': 59.0,
 'PNG': 59.0,
 'GUY': 59.0,
 'MNE': 59.0,
 'LSO': 60.0,
 'ECU': 60.0,
 'LAO': 60.0,
 'LBN': 60.0,
 'GNB': 60.0,
 'GAB': 60.0,
 'ETH': 61.0,
 'KGZ': 61.0,
 'CAF': 73.0,
 'JAM': 61.0,
 'COG': 62.0,
 'ERI': 62.0,
 'PER': 62.0,
 'TGO': 63.0,
 'DRC': 63.0,
 'BFA': 63.0,
 'BEN': 63.0,
 'TCD': 63.0,
 'UGA': 64.0,
 'GEO': 64.0,
 'SLE': 64.0,
 'ALB': 65.0,
 'XKO': 65.0,
 'SUR': 65.0,
 'LBR': 66.0,
 'MDA': 67.0,
 'BLZ': 67.0,
 'PRY': 67.0,
 'NER': 69.0,
 'DJI': 69.0,
 'SWZ': 69.0,
 'SEN': 71.0,
 'UZB': 71.0,
 'MMR': 71.0,
 'SYR': 72.0,
 'KAZ': 72.0,
 'BRA': 72.0,
 'MLI': 72.0,
 'PAN': 72.0,
 'SDN': 72.0,
 'ROU': 73.0,
 'ZMB': 73.0,
 'NOR': 73.0,
 'GNQ': 74.0,
 'DOM': 74.0,
 'MKD': 74.0,
 'MWI': 75.0,
 'UKR': 75.0,
 'RUS': 76.0,
 'CYP': 76.0,
 'JOR': 77.0,
 'THA': 77.0,
 'LVA': 77.0,
 'ISL': 77.0,
 'MRT': 77.0,
 'AGO': 78.0,
 'CUB': 78.0,
 'MOZ': 78.0,
 'TUN': 78.0,
 'CIV': 78.0,
 'PRI': 78.0,
 'DNK': 78.0,
 'IRQ': 79.0,
 'TKM': 79.0,
 'AZE': 80.0,
 'ARE': 80.0,
 'EST': 81.0,
 'JPN': 81.0,
 'QAT': 82.0,
 'URY': 82.0,
 'NZL': 83.0,
 'EGY': 83.0,
 'ZWE': 83.0,
 'FIN': 83.0,
 'VEN': 83.0,
 'ISR': 84.0,
 'KWT': 85.0,
 'BLR': 85.0,
 'PAK': 86.0,
 'CHE': 87.0,
 'NLD': 87.0,
 'GBR': 87.0,
 'BGR': 88.0,
 'BRN': 88.0,
 'DZA': 88.0,
 'IRL': 88.0,
 'LTU': 89.0,
 'LBY': 90.0,
 'CHN': 90.0,
 'SVN': 90.0,
 'MEX': 90.0,
 'ARG': 91.0,
 'BWA': 91.0,
 'TWN': 91.0,
 'MYS': 92.0,
 'POL': 92.0,
 'CHL': 92.0,
 'BEL': 92.0,
 'GRC': 93.0,
 'KOR': 93.0,
 'TUR': 93.0,
 'SVK': 93.0,
 'SWE': 94.0,
 'IRN': 94.0,
 'SRB': 94.0,
 'ITA': 95.0,
 'MAR': 95.0,
 'AUS': 96.0,
 'AUT': 96.0,
 'HUN': 96.0,
 'DEU': 97.0,
 'CZE': 98.0,
 'HRV': 98.0,
 'NAM': 99.0,
 'ZAF': 100.0,
 'OMN': 102.0,
 'ESP': 103.0,
 'FRA': 105.0,
 'CAN': 106.0,
 'SAU': 106.0,
 'PRT': 106.0,
 'USA': 107.0,
 'other': 73.0}
