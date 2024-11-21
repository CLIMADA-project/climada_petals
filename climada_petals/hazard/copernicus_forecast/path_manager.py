import logging
from pathlib import Path
from climada.util.constants import SYSTEM_DIR


DATA_OUT = SYSTEM_DIR / "copernicus_forecast"
LOGGER = logging.getLogger(__name__)

class PathManager:
    """Centralized path management for input, processed, and hazard data."""
    def __init__(self, base_dir=DATA_OUT):
        self.base_dir = Path(base_dir)

    def construct_path(self, sub_dir, file_name):
        """Constructs a full path from a sub-directory and file name."""
        return self.base_dir / sub_dir / file_name

    def get_download_path(self, originating_centre, year, month, index_metric, area_str, format):
        """
        Path for downloaded grib data.
        Example: input_data/dwd/grib/2002/07/HW_dwd_area4_49_33_20_200207.grib
        """
        sub_dir = f"input_data/{originating_centre}/{format}/{year}/{month:02d}"
        file_name = f"{index_metric}_{originating_centre}_{area_str}_{year}{month:02d}.{format}"
        return self.construct_path(sub_dir, file_name)

    def get_daily_processed_path(self, originating_centre, year, month, index_metric, area_str, format="nc"):
        """
        Path for processed daily netcdf data.
        Example: input_data/netcdf/daily/dwd/2002/07/dwd_t2m_area4_49_33_20_200207.nc
        """
        sub_dir = f"input_data/netcdf/daily/{originating_centre}/{year}/{month:02d}"
        file_name = f"{originating_centre}_{index_metric.lower()}_{area_str}_{year}{month:02d}.{format}"
        return self.construct_path(sub_dir, file_name)
    # iNCIDES paths
    def get_daily_index_path(self, originating_centre, year, month, index_metric, area_str, format="nc"):
        """
        Path for daily NetCDF index data.
        Example: indices/dwd/WBGT/2024/06/daily_WBGT_dwd_area4_56_45_16_202406.nc
        """
        sub_dir = f"indices/{originating_centre}/{index_metric}/{year}/{month:02d}"
        file_name = f"daily_{index_metric}_{originating_centre}_{area_str}_{year}{month:02d}.{format}"
        return self.construct_path(sub_dir, file_name)

    def get_monthly_index_path(self, originating_centre, year, month, index_metric, area_str, format="nc"):
        """
        Path for monthly NetCDF index data.
        Example: indices/dwd/WBGT/2024/06/WBGT_dwd_area4_56_45_16_202406.nc
        """
        sub_dir = f"indices/{originating_centre}/{index_metric}/{year}/{month:02d}"
        file_name = f"{index_metric}_{originating_centre}_{area_str}_{year}{month:02d}.{format}"
        return self.construct_path(sub_dir, file_name)

    def get_stats_index_path(self, originating_centre, year, month, index_metric, area_str, format="nc"):
        """
        Path for statistics NetCDF index data.
        Example: indices/dwd/WBGT/2024/06/stats/stats_WBGT_dwd_area4_56_45_16_202406.nc
        """
        sub_dir = f"indices/{originating_centre}/{index_metric}/{year}/{month:02d}/stats"
        file_name = f"stats_{index_metric}_{originating_centre}_{area_str}_{year}{month:02d}.{format}"
        return self.construct_path(sub_dir, file_name)
        
    # hazard path

    def get_hazard_path(self, originating_centre, year, month, index_metric, area_str, format="hdf5"):
        """
        Path for hazard HDF5 output data.
        Example: hazard/dwd/HW/2002/07/hazard_HW_dwd_area4_49_33_20_200207.hdf5
        """
        sub_dir = f"hazard/{originating_centre}/{index_metric}/{year}/{month:02d}"
        file_name = f"hazard_{index_metric}_{originating_centre}_{area_str}_{year}{month:02d}.{format}"
        return self.construct_path(sub_dir, file_name)

