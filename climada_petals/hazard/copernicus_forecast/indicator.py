# indicator.py

import xarray as xr
import numpy as np
import pandas as pd
import logging
import os
from climada.util.coordinates import country_to_iso, get_country_geometries
import calendar


def get_index_params(index):
    """
    Retrieves parameters associated with a specific climate index.

    Parameters:
    index (str): The climate index identifier.

    Returns:
    dict: A dictionary containing the parameters for the specified index.
    """
    var_specs = {
        "2m_temperature": {
            "unit": "K",
            "standard_name": "air_temperature",
            "short_name": "t2m",
            "full_name": "2m_temperature",
        },
        "2m_dewpoint_temperature": {
            "unit": "K",
            "standard_name": "dew_point_temperature",
            "short_name": "d2m",
            "full_name": "2m_dewpoint_temperature",
        },
    }

    index_params = {
        "HIA": {
            "variables": [
                var_specs["2m_temperature"]["full_name"],
                var_specs["2m_dewpoint_temperature"]["full_name"],
            ],
            "filename_lead": "2m_temps",
            "index_long_name": "Heat_Index_Adjusted",
        },
        "HIS": {
            "variables": [
                var_specs["2m_temperature"]["full_name"],
                var_specs["2m_dewpoint_temperature"]["full_name"],
            ],
            "filename_lead": "2m_temps",
            "index_long_name": "Heat_Index_Simplified",
        },
        "Tmean": {
            "variables": [var_specs["2m_temperature"]["full_name"]],
            "filename_lead": "2m_temps",
            "index_long_name": "Mean_Temperature",
        },
        "Tmin": {
            "variables": [var_specs["2m_temperature"]["full_name"]],
            "filename_lead": "2m_temps",
            "index_long_name": "Minimum_Temperature",
        },
        "Tmax": {
            "variables": [var_specs["2m_temperature"]["full_name"]],
            "filename_lead": "2m_temps",
            "index_long_name": "Maximum_Temperature",
        },
        "HW": {
            "variables": [var_specs["2m_temperature"]["full_name"]],
            "filename_lead": "2m_temps",
            "index_long_name": "Heat_Wave",
        },
        "TR": {
            "variables": [var_specs["2m_temperature"]["full_name"]],
            "filename_lead": "2m_temps",
            "index_long_name": "Tropical_Nights",
        },
        "HotDays": {
            "variables": [var_specs["2m_temperature"]["full_name"]],
            "filename_lead": "2m_temps",
            "index_long_name": "Hot Days (Tmax > 30°C)",
        },
    }
    return index_params.get(index)


def get_bounds_for_area_selection(area_selection, margin=0.2):
    """
    Determines the geographic bounds based on an area selection string.

    Parameters:
    area_selection (str): Specifies the area for data selection.
    margin (float): Additional margin to be added to the bounds in degrees.

    Returns:
    list: A list of four floats representing the bounds [north, east, south, west].
    """
    if area_selection.lower() == "global":
        return [90, -180, -90, 180]  # north, east, south, west
    else:
        try:
            user_bounds = list(map(float, area_selection.split(",")))
            if len(user_bounds) == 4:
                north, east, south, west = user_bounds
                north += margin
                east -= margin
                south -= margin
                west += margin
                return [north, east, south, west]
        except ValueError:
            pass

        countries = area_selection.split(",")
        combined_bounds = [180, 90, -180, -90]
        for country in countries:
            iso = country_to_iso(country.strip())
            geo = get_country_geometries(iso).to_crs(epsg=4326)
            bounds = geo.total_bounds
            min_lon, min_lat, max_lon, max_lat = bounds

            lat_margin = margin * (max_lat - min_lat)
            lon_margin = margin * (max_lon - min_lon)

            combined_bounds[0] = min(combined_bounds[0], min_lon - lon_margin)
            combined_bounds[1] = min(combined_bounds[1], min_lat - lat_margin)
            combined_bounds[2] = max(combined_bounds[2], max_lon + lon_margin)
            combined_bounds[3] = max(combined_bounds[3], max_lat + lat_margin)
        return [combined_bounds[3], combined_bounds[0], combined_bounds[1], combined_bounds[2]]


def calculate_relative_humidity_percent(t2k, tdk):
    """
    Calculates the relative humidity percentage from temperature and dewpoint temperature.

    Parameters:
    t2k (float or array-like): 2-meter temperature in Kelvin.
    tdk (float or array-like): 2-meter dewpoint temperature in Kelvin.

    Returns:
    float or array-like: Relative humidity as a percentage.
    """
    t2c = t2k - 273.15
    tdc = tdk - 273.15

    es = 6.11 * 10.0 ** (7.5 * t2c / (237.3 + t2c))
    e = 6.11 * 10.0 ** (7.5 * tdc / (237.3 + tdc))

    rh = (e / es) * 100
    rh = np.clip(rh, 0, 100)
    return rh


def calculate_heat_index_simplified(t2k, tdk):
    """
    Calculates the simplified heat index based on temperature and dewpoint temperature.

    Parameters:
    t2k (float or array-like): 2-meter temperature in Kelvin.
    tdk (float or array-like): 2-meter dewpoint temperature in Kelvin.

    Returns:
    float or array-like: Simplified heat index in degrees Celsius.
    """
    t2c = t2k - 273.15
    rh = calculate_relative_humidity_percent(t2k, tdk)
    hi = (
        -8.784695
        + 1.61139411 * t2c
        + 2.338549 * rh
        - 0.14611605 * t2c * rh
        + -1.2308094e-2 * t2c**2
        + -1.6424828e-2 * rh**2
        + 2.211732e-3 * t2c**2 * rh
        + 7.2546e-4 * t2c * rh**2
        - 3.582e-6 * t2c**2 * rh**2
    )
    return hi


def calculate_heat_index_adjusted(t2k, tdk):
    """
    Calculates the adjusted heat index based on temperature and dewpoint temperature.

    Parameters:
    t2k (float or array-like): 2-meter temperature in Kelvin.
    tdk (float or array-like): 2-meter dewpoint temperature in Kelvin.

    Returns:
    float or array-like: Adjusted heat index in degrees Celsius.
    """
    rh = calculate_relative_humidity_percent(t2k, tdk)
    t2f = (t2k - 273.15) * 9 / 5 + 32
    hi = (
        -42.379
        + 2.04901523 * t2f
        + 10.14333127 * rh
        - 0.22475541 * t2f * rh
        + -0.00683783 * t2f**2
        + -0.05481717 * rh**2
        + 0.00122874 * t2f**2 * rh
        + 0.00085282 * t2f * rh**2
        - 0.00000199 * t2f**2 * rh**2
    )
    return (hi - 32) * 5 / 9  # convert back to Celsius


def calculate_heat_index(da_t2k, da_tdk, index):
    if index == "HIS":
        index_long_name = "heat_index_simplified"
        unit = "degC"
        tf_index_data = calculate_heat_index_simplified(da_t2k.data, da_tdk.data)
    elif index == "HIA":
        index_long_name = "heat_index_adjusted"
        unit = "degC"
        tf_index_data = calculate_heat_index_adjusted(da_t2k.data, da_tdk.data)
    else:
        logging.error(f'Index {index} is not implemented, use either "HIS" or "HIA".')
        return None
    da_index = xr.DataArray(
        tf_index_data,
        coords=da_tdk.coords,
        dims=da_tdk.dims,
        attrs={"description": index_long_name, "units": unit},
    )
    return da_index


def calculate_heat_indices(data_out, year_list, month_list, area_selection, overwrite, tf_index):
    """
    Calculates and saves heat indices or temperature metrics.

    Parameters:
    data_out (str): Base directory path for output data.
    year_list (list of int): Years for which to calculate indices.
    month_list (list of int): Months to calculate indices (1-12).
    area_selection (str): Area specification.
    overwrite (bool): If True, overwrites existing files.
    tf_index (str): The climate index being processed.
    """
    index_params = get_index_params(tf_index)
    area = get_bounds_for_area_selection(area_selection)
    index_out = f"{data_out}/{tf_index}"
    area_str = f"{int(area[1])}_{int(area[0])}_{int(area[2])}_{int(area[3])}"

    for year in year_list:
        if not os.path.exists(f"{index_out}/{year}"):
            os.makedirs(f"{index_out}/{year}")

        for month in month_list:
            month_str = f"{month:02d}"
            index_file = f"{index_out}/{year}/{tf_index}_{year}{month_str}.nc"

            if os.path.isfile(index_file) and not overwrite:
                logging.info(f"Corresponding index file {index_file} already exists!")
                continue  # Skip if the file already exists and overwrite is False

            daily_out = f"{data_out}/netcdf/daily/{year}/{month_str}"

            try:
                with xr.open_dataset(
                    f'{daily_out}/{index_params["filename_lead"]}_{area_str}_{year}{month_str}.nc'
                ) as daily_ds:
                    # Handling various indices
                    if tf_index == "Tmean":
                        # Calculate mean temperature
                        da_index = daily_ds["t2m"] - 273.15  # Convert from Kelvin to Celsius
                        da_index.attrs["units"] = "degC"
                    elif tf_index == "Tmax":
                        # Calculate max daily temperature
                        da_index = (
                            daily_ds["t2m"].resample(step="1D").max() - 273.15
                        )
                        da_index.attrs["units"] = "degC"
                    elif tf_index == "Tmin":
                        # Calculate min daily temperature
                        da_index = (
                            daily_ds["t2m"].resample(step="1D").min() - 273.15
                        )
                        da_index.attrs["units"] = "degC"
                    elif tf_index == "HIS":
                        # Calculate simplified heat index
                        da_index = calculate_heat_index(
                            daily_ds["t2m"], daily_ds["d2m"], "HIS"
                        )
                    elif tf_index == "HIA":
                        # Calculate adjusted heat index
                        da_index = calculate_heat_index(
                            daily_ds["t2m"], daily_ds["d2m"], "HIA"
                        )
                    else:
                        raise ValueError(f"Unsupported index: {tf_index}")

                    # Save the index to a NetCDF file
                    ds_combined = xr.Dataset({tf_index: da_index})
                    ds_combined.to_netcdf(f"{daily_out}/{tf_index}_{year}{month_str}.nc")
                    logging.info(f"{tf_index} saved to {daily_out}/{tf_index}_{year}{month_str}.nc")

            except FileNotFoundError:
                logging.warning(
                    f'Data file {daily_out}/{index_params["filename_lead"]}_{area_str}_{year}{month_str}.nc does not exist!'
                )
                continue

            # Now handle statistics
            valid_times = pd.to_datetime(daily_ds.valid_time.values)
            forecast_months = valid_times.to_period("M")
            forecast_months_str = forecast_months.astype(str)
            step_to_month = dict(zip(daily_ds.step.values, forecast_months_str))
            forecast_month_da = xr.DataArray(
                list(step_to_month.values()), coords=[daily_ds.step], dims=["step"]
            )

            da_index.coords["forecast_month"] = forecast_month_da

            # Calculate monthly means
            monthly_means = da_index.groupby("forecast_month").mean(dim="step")
            monthly_means = monthly_means.rename(tf_index)

            # Rename forecast_month to step
            monthly_means = monthly_means.rename({"forecast_month": "step"})

            # Save the monthly means for all members in one file
            ds_member_means = xr.Dataset({f"{tf_index}": monthly_means})

            # Ensure 'number' dimension starts from 1 instead of 0
            ds_member_means = ds_member_means.assign_coords(number=ds_member_means.number)

            # Save the dataset
            ds_member_means.to_netcdf(index_file)
            logging.info(f"Monthly means saved to {index_file}")

            # Now calculate ensemble statistics across members
            da_index_ens_mean = monthly_means.mean("number")
            da_index_ens_median = monthly_means.median("number")
            da_index_ens_max = monthly_means.max("number")
            da_index_ens_min = monthly_means.min("number")
            da_index_ens_std = monthly_means.std("number")
            percentile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
            ensemble_percentiles = monthly_means.quantile(percentile_levels, dim="number")

            ds_stats = xr.Dataset(
                {
                    "ensemble_mean": da_index_ens_mean,
                    "ensemble_median": da_index_ens_median,
                    "ensemble_max": da_index_ens_max,
                    "ensemble_min": da_index_ens_min,
                    "ensemble_std": da_index_ens_std,
                }
            )
            for i, level in enumerate(percentile_levels):
                label = f"ensemble_p{int(level * 100)}"
                ds_stats[label] = ensemble_percentiles.sel(quantile=level)

            stats_file = f"{index_out}/{year}/{tf_index}_{year}{month_str}_statistics.nc"
            ds_stats.to_netcdf(stats_file)
            logging.info(f"Ensemble statistics saved to {stats_file}")


def calculate_and_save_tropical_nights_per_lag(base_path, year_list, month_list, tf_index, area_selection):
    """
    Calculates and saves the tropical nights index.

    Parameters:
    base_path (str): Base directory path for input and output data.
    year_list (list of int): Years for which to calculate the index.
    month_list (list of int): Months for which to calculate the index (1-12).
    tf_index (str): The climate index being processed ('TR').
    area_selection (str): Area specification.
    """
    index_params = get_index_params(tf_index)
    area = get_bounds_for_area_selection(area_selection)
    area_str = f'{int(area[1])}_{int(area[0])}_{int(area[2])}_{int(area[3])}'
    
    for year in year_list:
        for month in month_list:
            month_str = f"{month:02d}"
            grib_file_path = f"{base_path}/grib/{year}/{month_str}/{index_params['filename_lead']}_{area_str}_{year}{month_str}.grib"
            output_dir = f"{base_path}/{tf_index}/{year}/"
            os.makedirs(output_dir, exist_ok=True)
            output_file_path = os.path.join(output_dir, f"{tf_index}_{year}{month_str}.nc")
            
            try:
                ds = xr.open_dataset(grib_file_path, engine="cfgrib")
                t2m_celsius = ds["t2m"] - 273.15
                daily_min_temp = t2m_celsius.resample(step="1D").min()
                valid_times = pd.to_datetime(ds.valid_time.values)
                forecast_months = valid_times.to_period("M")
                forecast_months_str = forecast_months.astype(str)
                step_to_month = dict(zip(ds.step.values, forecast_months_str))
                forecast_month_da = xr.DataArray(list(step_to_month.values()), coords=[ds.step], dims=["step"])
                daily_min_temp.coords["forecast_month"] = forecast_month_da
                tropical_nights = daily_min_temp >= 20
                tropical_nights_count = tropical_nights.groupby("forecast_month").sum(dim="step")
                tropical_nights_count = tropical_nights_count.rename(tf_index)

                tropical_nights_count = tropical_nights_count.rename({"forecast_month": "step"})
                tropical_nights_count.to_netcdf(output_file_path)
                print(f"Tropical nights saved to {output_file_path}")

                ensemble_mean = tropical_nights_count.mean(dim="number")
                ensemble_median = tropical_nights_count.median(dim="number")
                ensemble_max = tropical_nights_count.max(dim="number")
                ensemble_min = tropical_nights_count.min(dim="number")
                ensemble_std = tropical_nights_count.std(dim="number")
                percentile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
                ensemble_percentiles = tropical_nights_count.quantile(percentile_levels, dim="number")
                
                ds_stats = xr.Dataset({
                    "ensemble_mean": ensemble_mean,
                    "ensemble_median": ensemble_median,
                    "ensemble_max": ensemble_max,
                    "ensemble_min": ensemble_min,
                    "ensemble_std": ensemble_std,
                })
                
                for i, level in enumerate(percentile_levels):
                    label = f"ensemble_p{int(level * 100)}"
                    ds_stats[label] = ensemble_percentiles.sel(quantile=level)
                
                stats_output_dir = os.path.join(output_dir, "stats")
                os.makedirs(stats_output_dir, exist_ok=True)
                stats_file = os.path.join(stats_output_dir, f"{tf_index}_{year}{month_str}_statistics.nc")
                ds_stats.to_netcdf(stats_file)
                print(f"Tropical nights statistics saved to {stats_file}")
            
            except FileNotFoundError as e:
                print(f"File not found: {e.filename}")
            except Exception as e:
                print(f"An error occurred: {e}")


def calculate_and_save_hot_days_per_lag(base_path, year_list, month_list, tf_index, area_selection):
    """
    Calculates and saves the Hot Days index (Tmax > 30°C).

    Parameters:
    base_path (str): Base directory path for input and output data.
    year_list (list of int): Years for which to calculate the index.
    month_list (list of int): Months for which to calculate the index (1-12).
    tf_index (str): The climate index being processed ('HotDays').
    area_selection (str): Area specification.
    """
    index_params = get_index_params(tf_index)
    area = get_bounds_for_area_selection(area_selection)
    area_str = f'{int(area[1])}_{int(area[0])}_{int(area[2])}_{int(area[3])}'
    
    for year in year_list:
        for month in month_list:
            month_str = f"{month:02d}"
            grib_file_path = f"{base_path}/grib/{year}/{month_str}/{index_params['filename_lead']}_{area_str}_{year}{month_str}.grib"
            output_dir = f"{base_path}/{tf_index}/{year}/"
            os.makedirs(output_dir, exist_ok=True)
            output_file_path = os.path.join(output_dir, f"{tf_index}_{year}{month_str}.nc")
            
            try:
                # Load the dataset
                ds = xr.open_dataset(grib_file_path, engine="cfgrib")
                t2m_celsius = ds["t2m"] - 273.15  # Convert temperature from Kelvin to Celsius
                
                # Resample to get daily max temperatures
                daily_max_temp = t2m_celsius.resample(step="1D").max()
                
                # Align forecast steps with calendar months using "valid_time"
                valid_times = pd.to_datetime(ds.valid_time.values)
                forecast_months = valid_times.to_period("M")
                forecast_months_str = forecast_months.astype(str)
                step_to_month = dict(zip(ds.step.values, forecast_months_str))
                forecast_month_da = xr.DataArray(list(step_to_month.values()), coords=[ds.step], dims=["step"])
                
                # Assign the "forecast_month" to the daily_max_temp DataArray
                daily_max_temp.coords["forecast_month"] = forecast_month_da

                # Calculate Hot Days: Days where Tmax > 30°C
                hot_days = daily_max_temp > 30  # Boolean array where True means a Hot Day
                
                # Count the number of Hot Days per forecast month
                hot_days_count = hot_days.groupby("forecast_month").sum(dim="step")
                hot_days_count = hot_days_count.rename(tf_index)
                
                # Save the result to NetCDF
                hot_days_count = hot_days_count.rename({"forecast_month": "step"})
                hot_days_count.to_netcdf(output_file_path)
                print(f"Hot Days saved to {output_file_path}")

                # Calculate ensemble statistics (mean, median, etc.)
                ensemble_mean = hot_days_count.mean(dim="number")
                ensemble_median = hot_days_count.median(dim="number")
                ensemble_max = hot_days_count.max(dim="number")
                ensemble_min = hot_days_count.min(dim="number")
                ensemble_std = hot_days_count.std(dim="number")
                
                # Percentiles
                percentile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
                ensemble_percentiles = hot_days_count.quantile(percentile_levels, dim="number")
                
                # Store statistics in a dataset
                ds_stats = xr.Dataset({
                    "ensemble_mean": ensemble_mean,
                    "ensemble_median": ensemble_median,
                    "ensemble_max": ensemble_max,
                    "ensemble_min": ensemble_min,
                    "ensemble_std": ensemble_std,
                })
                
                for i, level in enumerate(percentile_levels):
                    label = f"ensemble_p{int(level * 100)}"
                    ds_stats[label] = ensemble_percentiles.sel(quantile=level)
                
                # Save the statistics
                stats_output_dir = os.path.join(output_dir, "stats")
                os.makedirs(stats_output_dir, exist_ok=True)
                stats_file = os.path.join(stats_output_dir, f"{tf_index}_{year}{month_str}_statistics.nc")
                ds_stats.to_netcdf(stats_file)
                print(f"Hot Days statistics saved to {stats_file}")
            
            except FileNotFoundError as e:
                print(f"File not found: {e.filename}")
            except Exception as e:
                print(f"An error occurred: {e}")


def calculate_index(data_out, year_list, month_list, area_selection, overwrite, tf_index):
    """
    Calculates the specified climate index for given years and months.

    Parameters:
    data_out (str): Base directory path for output data.
    year_list (list of int): Years for which to calculate the index.
    month_list (list of int): Months for which to calculate the index (1-12).
    area_selection (str): Area specification.
    overwrite (bool): If True, overwrites existing files.
    tf_index (str): The climate index to be calculated.
    """
    area = get_bounds_for_area_selection(area_selection)

    if tf_index in ["HIS", "HIA", "Tmean", "Tmax", "Tmin"]:
        # Calculate heat indices like HIS, HIA, Tmean, Tmax, Tmin
        calculate_heat_indices(data_out, year_list, month_list, area_selection, overwrite, tf_index)

    elif tf_index == "TR":
        # Handle Tropical Nights (TR)
        calculate_and_save_tropical_nights_per_lag(data_out, year_list, month_list, tf_index, area_selection)

    elif tf_index == "HotDays":
        # Handle Hot Days (Tmax > 30°C)
        calculate_and_save_hot_days_per_lag(data_out, year_list, month_list, tf_index, area_selection)

    elif tf_index == "HW":
        # Handle Heat Wave Days (3 consecutive days Tmax > threshold)
        calculate_and_save_heat_wave_days_per_lag(data_out, year_list, month_list, tf_index, area_selection)

    else:
        logging.error(f"Index {tf_index} is not implemented. Supported indices are 'HIS', 'HIA', 'Tmean', 'Tmax', 'Tmin', 'HotDays', 'TR', and 'HW'.")