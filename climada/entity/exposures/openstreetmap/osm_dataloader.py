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




from climada import CONFIG
DATA_DIR = CONFIG.entity.openstreetmap.local_data.user_data.dir()

#DATA_DIR = '/Users/evelynm/Documents/WCR/3_PhD/1_coding_stuff/x_data/osm_countries'


URL_GEOFABRIK = 'https://download.geofabrik.de/'


# =============================================================================
# Download entire regional map data from extracts (geofabrik only)
# =============================================================================

def create_download_url(country, continent, file_format):
    """create string with download-api from geofabrik
    country : str
        country name
    continent : str
        continent (foldername on geofabrik.de in which country stored)
    file_format : str
        shp or pbf
    """
    if file_format == 'shp':
        return API_GEOFABRIK + continent + '/' + country + '-latest-free.shp.zip'
    elif file_format == 'pbf':
        return API_GEOFABRIK + continent + '/' + country + '-latest.osm.pbf'

def download_data_api(download_url):
    local_filename = download_url.split('/')[-1]
    local_filepath = DATA_DIR + '/' + local_filename
    if not Path(local_filepath).is_file():
        print(f'Downloading file as {local_filepath}')
        with requests.get(download_url, stream=True) as r:
            with open(local_filepath, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    else:
        print(f'file already exists as {local_filepath}')
    return local_filepath

# =============================================================================
# Download customized data from API (overpass-api)
# =============================================================================