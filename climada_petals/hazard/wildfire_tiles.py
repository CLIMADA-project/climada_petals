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

Define functions to preprocess MODIS burnt area data
"""

import re
import os
import json
import numpy as np
import glob
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix, find
from osgeo import gdal
import pyproj
import h5py

from climada.hazard import Hazard
from climada.hazard import Centroids

from climada_petals.hazard.wildfire import create_wf_haz

DAYS_LEAP = 366
DAYS_NORMAL = 365
SIZE_MODIS_TILE = 2400**2
modis_date_pattern = re.compile(r'\.A(\d{4})\d{3}\.')  # Matches .A2020123.
tile_pattern = re.compile(r'\.(h\d{2}v\d{2})\.')


def create_config_MODIS(dir_code, dir_data):
    
    dir_MODIS = os.path.join(dir_data, "0_original_MODIS")
    dir_ba = os.path.join(dir_MODIS, "BurntArea")
    dir_ba_tile = os.path.join(dir_ba, "{tile}")
    
    dir_lc = os.path.join(dir_MODIS, "LandCover")
    dir_lc_year = os.path.join(dir_lc, "{year}.01.01")
    modis_data_structure = "product.date.tile.processing_date.hdf"
    
    dir_modis_preprocessing = os.path.join(dir_data, "1_MODIS_preprocessing")
    list_tiles = os.path.join(dir_modis_preprocessing, "list_tiles.txt")
    dir_tiles = os.path.join(dir_modis_preprocessing, "tiles")
    tile_hdf = os.path.join(dir_tiles, "{variable}_{tile}.hdf")
    tile_coord = os.path.join(dir_tiles, "coords_{tile}.csv")
    tile_years = os.path.join(dir_tiles, "years_{tile}.csv")
    tile_hazard = os.path.join(dir_modis_preprocessing, "tiles_hazard", "hazard_{variable}_{tile}.hdf")
    dir_tiles_merged = os.path.join(dir_modis_preprocessing, "tiles_hazard_merged", "hazard_chunk_{chunk}.hdf")
    landcover_annual = os.path.join(dir_modis_preprocessing, "landcover_annual", "landcover_{year}.hdf")
    
    wf_global_monthly = os.path.join(dir_modis_preprocessing, "global", "hist_global_wf_monthly.hdf")
    wf_global_annual = os.path.join(dir_modis_preprocessing, "global", "hist_global_wf_annual.hdf")
    wf_global_monthly05 = os.path.join(dir_modis_preprocessing, "global", "hist_global_wf_monthly_05degree.hdf")
    wf_global_annual05 = os.path.join(dir_modis_preprocessing, "global", "hist_global_wf_annual_05degree.hdf")
    cent_05 = os.path.join(dir_modis_preprocessing, "centroids_05deg.hdf5")
    
    dir_global_modis = os.path.join(dir_data, "2_global_model_input")
    wf_global= os.path.join(dir_global_modis, "hist_global_wf.hdf")


    config_data = {
        "dir_code": dir_code,
        "dir_MODIS": dir_MODIS,
        "dir_ba":dir_ba,
        "dir_ba_tile":dir_ba_tile,
        "dir_lc":dir_lc,
        "dir_lc_year":dir_lc_year,
        "modis_data_structure":modis_data_structure,
        
        "dir_modis_preprocessing": dir_modis_preprocessing,
        "list_tiles": list_tiles,
        
        "tile_hdf": tile_hdf,
        "tile_coord": tile_coord,
        "tile_years":tile_years,
        "tile_hazard": tile_hazard,
        "dir_tiles_merged":dir_tiles_merged,
        "landcover_annual":landcover_annual,
        
        "wf_global_monthly": wf_global_monthly,
        "wf_global_annual": wf_global_annual,
        "wf_global_monthly05":wf_global_monthly05,
        "wf_global_annual05": wf_global_annual05,
        "cent_05":cent_05,
        
        "wf_global": wf_global,
        
        }
    
    with open('config_MODIS.json', 'w') as file:
        json.dump(config_data, file, indent=4)




def MODIS_grid_tile(MODIS_file):
    """
    Construct centroid coordinates by reprojecting MODIS own coordinate system 
    (does this only have to be done once per tile?)

    Parameters
    ----------
    MODIS_file : string 
        MODIS tile file.

    Returns
    -------
    xv : np.array
        MODIS x coordinate (original coordinate systems).
    yv : np.array
        MODIS y coordinate (original coordinate systems).
    lon : np.array
        Longitude in WGS84.
    lat : np.array
        Latitude in WGS84.

    """
    
    hdf_handle = gdal.Open(MODIS_file)
    sds_list = hdf_handle.GetSubDatasets()
    burndate_handle = gdal.Open(sds_list[0][0])
    x0, xinc, _, y0, _, yinc = burndate_handle.GetGeoTransform()
    nx, ny = (burndate_handle.RasterXSize, burndate_handle.RasterYSize)
    x = np.linspace(x0, x0 + xinc*nx, nx)
    y = np.linspace(y0, y0 + yinc*ny, ny)
    xv, yv = np.meshgrid(x, y)
    
    #transform grid
    sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    wgs84 = pyproj.Proj("+init=EPSG:4326") 
    lon, lat= pyproj.transform(sinu, wgs84, xv, yv)

    return xv, yv, lon, lat



def read_burndate(file, variable='Burn Date'): 
    """
    Read burn date from MODIS monthly tile and reshape to centroid array

    Parameters
    ----------
    file : string
        MODIS tile file.
        
    variable : string
        Variable name to be read from MODIS tile. 
        Options: 'Burn Date', 'First Day', 'Last Day ' (of burn)
        Default: 'Burn Date'

    Returns
    -------
    burndate : csr.matrix
        DESCRIPTION.

    """

    # Open the HDF file using GDAL
    dataset = gdal.Open(file)

    # Get the list of subdatasets
    subdatasets = dataset.GetSubDatasets()

    # Loop through subdatasets and find the one containing "Burn Date"
    burn_date_subdataset = None
    for subdataset in subdatasets:
        if variable in subdataset[0]:
            burn_date_subdataset = subdataset[0]
            break

    # Check if we found the "Burn Date" subdataset
    if burn_date_subdataset is None:
        raise ValueError(f"Could not find {variable} subdataset.")

    # Open the "Burn Date" subdataset
    subdataset = gdal.Open(burn_date_subdataset)

    # Read the data into a numpy array
    burndate_data = subdataset.ReadAsArray()
    
    # get rid of unvalid centroids
    burndate_data[burndate_data == -2] = 0
    burndate_data[burndate_data == -1] = 0 #unvalid cells
    burndate_data[np.isnan(burndate_data)] = 0

    # Optional: Convert to sparse matrix
    from scipy import sparse
    burndate = sparse.csr_matrix(burndate_data.reshape(-1))
    
    return burndate



"""CREATE HAZARD PER TILE"""
def get_time_array(year_range): 
    start_year = year_range[0]
    end_year = year_range[1]
    
    years = np.arange(start_year, end_year+1)
    time_array = pd.date_range(start=str(start_year)+"-01-01", end=str(end_year)+"-12-31")
    
    # Create a dictionary to store year: index pairs
    first_of_january_indices_dict = {}
    # Iterate through the time_array to find the indices of the first of January dates
    for date in time_array:
        if date.month == 1 and date.day == 1:  # Check if the date is the first of January
            year = date.year
            index = time_array.get_loc(date)
            first_of_january_indices_dict[year] = index
    
    return years, time_array, first_of_january_indices_dict


def extract_year_from_filename(filename):
    match = modis_date_pattern.search(os.path.basename(filename))
    if match:
        return int(match.group(1))  # Extracted year
    else:
        raise ValueError(f"Year not found in the filename: {filename}")


def get_tile_year_array(file_list):
    # Initialize an empty list for the years corresponding to each file
    years_list = []
    
    # Loop over the files in file_list and extract the year for each
    for file in file_list:
        year = extract_year_from_filename(file)
        years_list.append(year)
    
    # Convert the list to a numpy array if needed
    years_array = np.array(years_list)
    
    return years_array

def extract_files_by_tile_and_year(all_files, list_tiles, year_range):
    # Initialize the dictionary to store files by tile
    files_by_tile = {tile: [] for tile in list_tiles}

    # Define the start and end year from the range
    start_year = year_range[0]
    end_year = year_range[1]
    years = np.arange(start_year, end_year + 1)

    # Iterate over all files in the directory
    for f in all_files:
        if not os.path.isfile(f):
            continue
        
        # Search for the date and tile pattern in the filename
        date_match = modis_date_pattern.search(f)
        tile_match = tile_pattern.search(f)

        if not date_match or not tile_match:
            continue  # Skip files that don't match the expected pattern

        # Extract the year and tile from the match groups
        file_year = int(date_match.group(1))
        file_tile = tile_match.group(1).strip()  # Remove any accidental whitespace

        # Ensure file_tile is exactly what we expect
        if file_tile not in list_tiles:
            continue

        # Check if the file year is within the desired range
        if file_year in years:
            files_by_tile[file_tile].append(f)

    # Now, sort the files for each tile by the year (and then by filename if needed)
    for tile in files_by_tile:
        # Sort by the year extracted from the filename (and secondarily by filename if needed)
        files_by_tile[tile].sort(key=lambda x: (int(modis_date_pattern.search(x).group(1)), x))


    return files_by_tile


def save_ba_tiles(config_MODIS, list_tiles, variable, year_range=(2001, 2025)):
    
    dir_ba = config_MODIS['dir_ba']
    
    all_files = list(glob.iglob(os.path.join(dir_ba, "*"), recursive=True))
    files_by_tile = extract_files_by_tile_and_year(all_files, list_tiles, year_range)
    

    years, _, _ = get_time_array(year_range)
    
    
    for tile in files_by_tile:
        print("Processing tile ", tile)
        
        file_list = files_by_tile[tile]
        
        ba_tile = lil_matrix((len(file_list), SIZE_MODIS_TILE))
        # ba_year_tile = dok_matrix((len(file_list), SIZE_MODIS_TILE))
        
        # loop over all MODIS files in the time array
        for idx_file, file in enumerate(file_list):             
            burndate = read_burndate(file, variable) 
            ba_tile[idx_file, :] = burndate
        
        
        final_ba = sparse.csr_matrix(ba_tile)
        # Save burn date data
        output_file = config_MODIS['tile_hdf'].format(variable=variable, tile=tile)
        
        # Save coordinates
        xv, yv, lon, lat = MODIS_grid_tile(file_list[0])
        
        # Save time array
        year_list = get_tile_year_array(file_list)

        save_hdf5(final_ba, lon, lat, year_list, output_file)


def save_hdf5(final_ba, lon, lat, year_list, output_file):

    with h5py.File(output_file, "w") as f:
        f.create_dataset("data", data=final_ba.data)
        f.create_dataset("indices", data=final_ba.indices)
        f.create_dataset("indptr", data=final_ba.indptr)
        f.create_dataset("shape", data=final_ba.shape)
        
        # Spatial metadata
        f.create_dataset("latitude", data=lat.reshape(-1))
        f.create_dataset("longitude", data=lon.reshape(-1))
        f.create_dataset("year", data=np.array(year_list))


def create_tile_haz(config_MODIS, list_tiles, variable):
    

    for tile in list_tiles:
        
        # Path to the HDF5 file
        burn_file = config_MODIS['tile_hdf'].format(variable=variable, tile=tile)
        
        # Open the HDF5 file in read mode
        with h5py.File(burn_file, 'r') as file:
            # Access datasets
            data = file['data'][:]
            indices = file['indices'][:]
            indptr = file['indptr'][:]
            shape = file['shape'][:]
            
            lat = file['latitude'][:]
            lon = file['longitude'][:]
            tile_years = file['year'][:]
            
            
        centr = Centroids(lat=lat, lon=lon)
        
        #positive values
        final_c = csr_matrix((data, indices, indptr), shape=shape)
        row_indices, col_indices, _ = find(final_c > 0)
        values = final_c[row_indices, col_indices].astype(int)
        
        year_range = (np.min(tile_years), np.max(tile_years))
        _, time_array, first_of_january_indices_dict = get_time_array(year_range)
        year_data = pd.DataFrame(tile_years[row_indices], columns=['year'])
        start_idx_year_data = year_data['year'].map(first_of_january_indices_dict).values
        
        matrix_hazard = lil_matrix((len(time_array), SIZE_MODIS_TILE))
        matrix_hazard[(start_idx_year_data+values)-1, col_indices] = 1
        intensity = csr_matrix(matrix_hazard)
        
        haz = create_wf_haz(intensity, centr, time_array)
        haz.write_hdf5(config_MODIS['tile_hazard'].format(variable=variable, tile=tile))        


"""CREATE GLOBAL HAZARD"""
def create_global_haz(config_MODIS, variable, year_range, chunk_size=10):

    search_pattern = config_MODIS['tile_hazard'].replace('{variable}', variable).replace('{tile}', '*')
    tile_haz = sorted(glob.glob(search_pattern))
    
    _, time_array, _ = get_time_array(year_range)
    
    merged_tiles = []
    for i in range(0, len(tile_haz), chunk_size):
        print("Chunk: ", i)
        chunk_files = tile_haz[i:i + chunk_size]
        chunk_haz = merge_tiles(chunk_files, time_array)
        chunk_file = config_MODIS['dir_tiles_merged'].format(chunk=str(i))
        chunk_haz.write_hdf5(chunk_file)
        merged_tiles.append(chunk_file)
    
    # Final merge
    global_haz = merge_tiles(merged_tiles, time_array)
    
    return global_haz


def merge_tiles(tile_haz, time_array):
    
    for idx_tile, file in enumerate(tile_haz):
        haz = Hazard.from_hdf5(file)
        
        tile_intensity = (haz.intensity.tocsr()).tolil()
        tile_lat = haz.centroids.lat
        tile_lon = haz.centroids.lon
        
        if idx_tile == 0:
            global_intensity = tile_intensity
            global_lat = tile_lat
            global_lon = tile_lon
        else:
            global_intensity = sparse.hstack((global_intensity, tile_intensity))  
            global_lat = np.hstack((global_lat, tile_lat))
            global_lon = np.hstack((global_lon, tile_lon))
        
        print('Appended hazard for tile ', idx_tile, file)
    
    global_centr = Centroids(lat=global_lat, lon=global_lon)
    
    global_intensity_csr = sparse.csr_matrix(global_intensity)
    global_haz = create_wf_haz(global_intensity_csr, global_centr, time_array)
    
    return global_haz


# """MODIS LAND COVER"""

# def read_landcover(file):
#     ds_modis = rioxarray.open_rasterio(filename=file, masked=True).squeeze(drop=True)
#     LC_data = ds_modis['LC_Type1'].values
    
#     # get rid of unvalid centroids
#     LC_data[np.isnan(LC_data)] = 0
#     # permanent snow and ice
#     LC_data[LC_data == 15] = 0
#     # barren
#     LC_data[LC_data == 16] = 0
#     # water bodies
#     LC_data[LC_data == 17] = 0
#     # unclassified
#     LC_data[LC_data == 255] = 0
    

#     landcover = sparse.csr_matrix(LC_data.astype(int).reshape(-1))
    
#     ds_modis.close()
    
#     return landcover


# def global_landcover_annual(config_MODIS, years=(2001, 2022)):
#     with open(config_MODIS['list_tiles'], 'r') as file:
#         # Read the contents of the file as a string
#         tiles_string = file.read()
#     tiles_list = eval(tiles_string)
     
#     start_year = years[0]
#     end_year = years[1]
#     for year in range(start_year, end_year+1):
#         dir_lc_year = config_MODIS['dir_lc_year'].format(year=year)
        
#         # List to store the file paths
#         file_paths = []
        
#         # Iterate through the files in the directory
#         for filename in os.listdir(dir_lc_year):
#             # Check if the filename contains any of the tiles from your list and ends with '.hdf'
#             if any(tile in filename for tile in tiles_list) and filename.endswith('.hdf'):
#                 # If yes, append the file path to the list
#                 file_paths.append(os.path.join(dir_lc_year, filename))
        
#         files_year = sorted(file_paths)
        
#         for idx_tile, file in enumerate(files_year): 
#             landcover = read_landcover(file)
            
#             print("Processing file ", file)
            
#             if idx_tile == 0:
#                 global_landcover = landcover
#             else:
#                 global_landcover = sparse.hstack((global_landcover, landcover))
                    
#         final_landcover = sparse.csr_matrix(global_landcover)
        
#         output_file = config_MODIS['landcover_annual'].format(year=year)
#         with h5py.File(output_file, "w") as f:
#             f.create_dataset("data", data=final_landcover.data)
#             f.create_dataset("indices", data=final_landcover.indices)
#             f.create_dataset("indptr", data=final_landcover.indptr)
#             f.create_dataset("shape", data=final_landcover.shape)





