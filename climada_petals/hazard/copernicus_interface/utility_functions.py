import warnings
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from matplotlib.ticker import FuncFormatter
from climada.engine import Impact
from climada.util.config import CONFIG
import matplotlib.gridspec as gridspec
from climada_petals.hazard.copernicus_interface.create_seasonal_forecast_hazard import SeasonalForecast

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")

#plot forescast
def plot_forecast(forecast_year, initiation_month_str, target_month, handler, index_metric="TX30"):
    """
    Plots ensemble members for a given seasonal forecast.

    Parameters:
    - forecast_year (int): The year of the forecast.
    - initiation_month_str (str): The initiation month as a string (e.g., "03" for March).
    - target_month (str): The target month in "YYYY-MM" format (e.g., "2022-07").
    - handler (SeasonalForecast): An instance of SeasonalForecast.
    - index_metric (str, optional): The climate index variable to plot. Default is "TX30".

    Raises:
    - ValueError if the dataset cannot be loaded or required keys are missing.
    """

    # Retrieve the dataset path
    indices_paths = handler.get_pipeline_path(forecast_year, initiation_month_str, "indices")

    # Ensure "monthly" exists in the returned dictionary
    if not isinstance(indices_paths, dict) or "monthly" not in indices_paths:
        raise ValueError("'monthly' key not found in the indices path dictionary.")
    
    path_to_hazard = indices_paths["monthly"]
    print(f"Using monthly index file: {path_to_hazard}")

    # Load dataset
    try:
        ds = xr.open_dataset(path_to_hazard, engine="netcdf4")
        print(f"Successfully loaded dataset: {path_to_hazard}")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")


    # Select data for the specific month
    if "step" in ds.coords:
        data_for_month = ds.sel(step=target_month)
    else:
        raise ValueError(f"'step' coordinate not found. Available coordinates: {list(ds.coords.keys())}")

    # Verify variable name
    if index_metric not in ds.variables:
        raise ValueError(f"Variable '{index_metric}' not found in dataset. Available variables: {list(ds.data_vars.keys())}")

    # Create subplots
    fig, axs = plt.subplots(nrows=5, ncols=10, figsize=(25, 15), subplot_kw={"projection": ccrs.PlateCarree()})
    axs = axs.flatten()  # Flatten array for easy iteration

    # Plot each ensemble member
    for i in range(50):
        ax = axs[i]
        member_data = data_for_month.isel(number=i)[index_metric]  # Select dynamically
        p = member_data.plot(ax=ax, transform=ccrs.PlateCarree(), x="longitude", y="latitude", 
                             add_colorbar=False, cmap="viridis")

        ax.coastlines(color="white", linewidth=1)  # Set coastline color to white
        ax.add_feature(cfeature.BORDERS, edgecolor="white", linewidth=1)  # Add country borders in white
        ax.set_title(f"Member {i+1}")

    # Adjust layout for colorbar
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95, wspace=0.1, hspace=0.1)

    # Add a color bar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.015])
    fig.colorbar(p, cax=cbar_ax, orientation="horizontal")

    plt.show()


#########extract statitcs

def extract_statistics(file_path, forecast_months):
    """Extracts tropical nights statistics from a dataset for the given months."""
    try:
        ds_stats = xr.open_dataset(file_path, engine="netcdf4")
        stats = {
            "Mean": ds_stats["ensemble_mean"].sel(step=forecast_months).mean(dim=["latitude", "longitude"]),
            "Median": ds_stats["ensemble_median"].sel(step=forecast_months).mean(dim=["latitude", "longitude"]),
            "Max": ds_stats["ensemble_max"].sel(step=forecast_months).mean(dim=["latitude", "longitude"]),
            "Min": ds_stats["ensemble_min"].sel(step=forecast_months).mean(dim=["latitude", "longitude"]),
            "Std": ds_stats["ensemble_std"].sel(step=forecast_months).mean(dim=["latitude", "longitude"]),
        }
        return pd.DataFrame({key: val.values for key, val in stats.items()}, index=forecast_months)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def plot_tropical_nights_statistics(forecast_year, initiation_months, valid_periods, handler, index_metric="TX30"):
    """Plots tropical nights forecast statistics for multiple datasets."""

    forecast_months = [f"{forecast_year}-{month}" for month in valid_periods]  # Convert to "YYYY-MM" format
    valid_period_str = "_".join(valid_periods)  # Format valid periods for path lookup

    # Retrieve file paths dynamically using handler
    file_paths = []
    for init_month in initiation_months:
        indices_paths = handler.get_pipeline_path(forecast_year, init_month, "indices")

        if isinstance(indices_paths, dict) and "stats" in indices_paths:
            file_paths.append(str(indices_paths["stats"]))
        else:
            print(f"Skipping initiation month {init_month}: 'stats' file not found.")

    if not file_paths:
        print("No valid data available for plotting.")
        return

    # Extract data for each dataset
    dataframes = [extract_statistics(fp, forecast_months) for fp in file_paths]
    dataframes = [df for df in dataframes if df is not None]  # Remove None values

    if not dataframes:
        print("No valid data extracted for plotting.")
        return

    # Set common y-limits for comparison
    y_min = min(df["Min"].min() for df in dataframes)
    y_max = max(df["Max"].max() for df in dataframes)

    # Create plots
    fig, axs = plt.subplots(1, len(dataframes), figsize=(7 * len(dataframes), 6))

    if len(dataframes) == 1:
        axs = [axs]  # Ensure axs is iterable

    for ax, df, file_path in zip(axs, dataframes, file_paths):
        init_month = file_path.split("/init")[1][:2]  # Extract initiation month from path
        ax.plot(df.index, df["Mean"], label="Mean", color="darkblue", linewidth=2)
        ax.plot(df.index, df["Median"], label="Median", color="orange", linewidth=1)
        ax.fill_between(df.index, df["Min"], df["Max"], color="skyblue", alpha=0.3, label="Min-Max Range")
        ax.fill_between(df.index, df["Mean"] - df["Std"], df["Mean"] + df["Std"], color="salmon", alpha=0.4, label="Mean ± Std")
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"{index_metric} Forecast (Init: {init_month})", fontsize=14)
        ax.set_xlabel("Forecast Month", fontsize=12)
        ax.set_ylabel(f"Number of {index_metric}", fontsize=12)
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

###########


def get_dynamic_dataset_paths(forecast_year, initiation_months, valid_periods, handler):
    """
    Dynamically retrieve dataset paths for given forecast parameters.

    Parameters:
    - forecast_year (int): Year of the forecast.
    - initiation_months (list of str): List of initiation months (e.g., ["04", "05"]).
    - valid_periods (list of str): List of valid forecast months (e.g., ["06", "07", "08"]).
    - handler (SeasonalForecast): Instance of SeasonalForecast class.

    Returns:
    - dict: Dictionary mapping labels (e.g., "2022_init04_valid06_08") to file paths.
    """
    dataset_paths = {}

    valid_period_str = "_".join(valid_periods)

    for init_month in initiation_months:
        indices_paths = handler.get_pipeline_path(forecast_year, init_month, "indices")

        if isinstance(indices_paths, dict) and "monthly" in indices_paths:
            dataset_label = f"{forecast_year}_init{init_month}_valid{valid_period_str}"
            dataset_paths[dataset_label] = indices_paths["monthly"]
        else:
            print(f"Skipping initiation month {init_month}: 'monthly' file not found.")

    return dataset_paths

# Load dataset function
def load_dataset(file_path):
    """Opens an xarray dataset safely, handling errors."""
    try:
        return xr.open_dataset(file_path, engine="netcdf4")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def plot_seasonal_metrics_by_lat(forecast_year, initiation_months, valid_periods, handler, threshold=5):
    index_metric = handler.index_metric
    dataset_paths = get_dynamic_dataset_paths(forecast_year, initiation_months, valid_periods, handler)

    if not dataset_paths:
        print("No valid datasets found. Exiting.")
        return

    metrics_functions = [
        ('Ensemble Spread', calculate_ensemble_spread),
        ('SMR', calculate_smr),
        ('Interquartile Range', calculate_interquartile_range),
        ('Ensemble Agreement Index', calculate_ensemble_agreement_index)
    ]

    fig, axs = plt.subplots(len(metrics_functions), len(dataset_paths), figsize=(20, 15), sharex='col', sharey='row')

    for col, (init_label, file_path) in enumerate(dataset_paths.items()):
        ds = load_dataset(file_path)
        if ds is None or index_metric not in ds:
            print(f"Skipping dataset {init_label}: Variable '{index_metric}' not found.")
            continue

        data_var = ds[index_metric]
        forecast_steps = ds["step"].values
        latitudes = ds["latitude"].values

        for row, (name, func) in enumerate(metrics_functions):
            metric_data = func(data_var, threshold)
            c = axs[row, col].pcolormesh(forecast_steps, latitudes, metric_data.T, shading='auto', cmap='coolwarm')
            axs[row, col].set_title(f"{name} ({init_label})")
            axs[row, col].set_xlabel("Forecast Month")
            axs[row, col].set_ylabel("Latitude")
            axs[row, col].set_xticks(forecast_steps)
            fig.colorbar(c, ax=axs[row, col], orientation='horizontal', pad=0.2, fraction=0.08).set_label('Metric Scale')

    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    description = """
    Each plot represents a different metric for assessing forecast skills in climate models:
    - Ensemble Spread: Standard deviation of forecasts across ensemble members.
    - SMR (Spread to Mean Ratio): Ratio of forecast spread to mean value.
    - Interquartile Range: Spread between 25th and 75th percentiles of the forecasts.
    - Ensemble Agreement Index: Fraction of ensemble members exceeding the threshold.
    X-axis (Forecast Month) represents forecast steps, while Y-axis (Latitude) represents spatial variation.
    """
    fig.text(0.1, -0.13, description, wrap=True, fontsize=17)
    plt.show()

def calculate_ensemble_spread(data, threshold):
    return data.std(dim='number').mean(dim='longitude')

def calculate_smr(data, threshold):
    return (data.std(dim='number') / data.mean(dim='number')).mean(dim='longitude')

def calculate_interquartile_range(data, threshold):
    return (data.quantile(0.75, dim='number') - data.quantile(0.25, dim='number')).mean(dim='longitude')

def calculate_ensemble_agreement_index(data, threshold):
    event_occurrence = data > threshold
    return (event_occurrence.sum(dim='number') / event_occurrence.sizes['number']).mean(dim='longitude')



############

climada_base_path = CONFIG.hazard.copernicus.local_data.dir()

def thousands_comma_formatter(x, pos):
    return "{:,.0f}K".format(x * 1e-3)

def load_impact_data(init_month, year_list, index_metric):
    year_str = str(year_list[0])

    impact_dir = os.path.join(climada_base_path, "seasonal_forecasts", "dwd", "sys21", year_str,
                              f"init{init_month}", "valid06_08", "impact", index_metric)

    if not os.path.isdir(impact_dir):
        raise FileNotFoundError(f"Impact directory does not exist: {impact_dir}")

    impact_files = [f for f in os.listdir(impact_dir) if f.startswith(f"{index_metric}_") and f.endswith(".hdf5")]
    if not impact_files:
        raise FileNotFoundError(f"No impact file found in {impact_dir}")

    impact_path = os.path.join(impact_dir, impact_files[0])
    return Impact.from_hdf5(impact_path)

def plot_individual_and_aggregated_impacts(year_list, index_metric, *init_months):
    month_names = ["Jun", "Jul", "Aug"]
    cmap = plt.cm.get_cmap("tab20", 50)

    fig, axes = plt.subplots(nrows=len(init_months), figsize=(5, 3 * len(init_months)), sharex=True)

    if len(init_months) == 1:
        axes = [axes]

    for ax, init_month in zip(axes, init_months):
        impact_data = load_impact_data(init_month, year_list, index_metric)

        unique_dates = np.unique(impact_data.date)[:3]
        month_labels = [datetime.fromordinal(date).strftime("%b") for date in unique_dates]

        all_member_impacts = [[] for _ in range(50)]

        for i, date in enumerate(unique_dates):
            event_indices = np.where(impact_data.date == date)[0][:50]

            for j, idx in enumerate(event_indices):
                all_member_impacts[j].append(impact_data.at_event[idx])

        for i in range(50):
            ax.plot(month_labels, all_member_impacts[i], marker="o", linestyle="-", linewidth=1, alpha=0.3, color=cmap(i), label=f"Member {i}" if i == 0 else "_nolegend_")

        mean_impact = np.mean(all_member_impacts, axis=0)
        median_impact = np.median(all_member_impacts, axis=0)
        std_impact = np.std(all_member_impacts, axis=0)

        ax.plot(month_labels, mean_impact, "k--", linewidth=1.5, label="Mean")
        ax.plot(month_labels, median_impact, "k-", linewidth=1.5, label="Median")
        ax.fill_between(month_labels, mean_impact - std_impact, mean_impact + std_impact, color="grey", alpha=0.15, label="Mean ± Std")

        ax.set_title(f"Impact Values - Init {init_month}", fontsize=10)
        ax.set_ylabel("Population Affected (Thousands)", fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_comma_formatter))
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True)

    axes[-1].set_xlabel("Forecast Month", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

###########

climada_base_path = CONFIG.hazard.copernicus.local_data.dir()

def month_num_to_name(month_num):
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    try:
        return month_names[int(month_num) - 1]
    except (ValueError, IndexError):
        return f"Invalid month number: {month_num}"

def log_formatter(x, pos):
    return f"$10^{{{int(x)}}}$"

def load_impact_data(init_month, year_list, index_metric):
    year_str = str(year_list[0])
    impact_dir = os.path.join(climada_base_path, "seasonal_forecasts", "dwd", "sys21", year_str,
                              f"init{init_month}", "valid06_08", "impact", index_metric)

    if not os.path.isdir(impact_dir):
        raise FileNotFoundError(f"Impact directory does not exist: {impact_dir}")

    impact_files = [f for f in os.listdir(impact_dir) if f.startswith(f"{index_metric}_") and f.endswith(".hdf5")]
    if not impact_files:
        raise FileNotFoundError(f"No impact file found in {impact_dir}")

    impact_path = os.path.join(impact_dir, impact_files[0])
    return Impact.from_hdf5(impact_path)

def plot_impact_distributions(year_list, index_metric, init_months):
    max_members = 50

    for year in year_list:
        for month in init_months:
            impact_data = load_impact_data(month, [year], index_metric)

            unique_dates = np.unique(impact_data.date)[:3]
            month_labels = [datetime.fromordinal(date).strftime("%b") for date in unique_dates]

            all_impacts = {}
            for i, date in enumerate(unique_dates):
                event_indices = np.where(impact_data.date == date)[0][:max_members]

                for idx in event_indices:
                    month_num = datetime.fromordinal(date).month
                    impact_value = impact_data.at_event[idx]
                    log_impact = np.log10(impact_value + 1)  # Log transform

                    if month_num not in all_impacts:
                        all_impacts[month_num] = []
                    all_impacts[month_num].append(log_impact)

            fig, axes = plt.subplots(nrows=(len(all_impacts) + 1) // 2, ncols=2, figsize=(8, 5))
            axes = axes.flatten()

            for i, (month_num, impacts) in enumerate(sorted(all_impacts.items())):
                impacts = np.array(impacts)
                mean_log = np.mean(impacts)
                median_log = np.median(impacts)
                mean_real = int(10**mean_log - 1)
                median_real = int(10**median_log - 1)

                bins = np.linspace(impacts.min(), impacts.max(), 10)
                hist, edges = np.histogram(impacts, bins=bins)
                hist_percentage = (hist / max_members) * 100
                bin_midpoints = edges[:-1] + np.diff(edges) / 2

                axes[i].bar(bin_midpoints, hist_percentage, width=np.diff(edges), color="black", alpha=0.5, edgecolor="black", align="edge")

                if len(bin_midpoints) > 3:  # Ensure at least 4 points for cubic fit
                    try:
                        coefficients = np.polyfit(bin_midpoints, hist_percentage, min(3, len(bin_midpoints)-1))
                        polynomial = np.poly1d(coefficients)
                        xs = np.linspace(bin_midpoints[0], bin_midpoints[-1], 100)
                        ys = polynomial(xs)
                        axes[i].plot(xs, ys, color="black", linewidth=1.2, linestyle="--")
                    except np.linalg.LinAlgError:
                        print(f"Skipping polynomial fit for {month_num_to_name(month_num)} {year} due to instability.")

                axes[i].set_xlabel("Population Affected (Log Scale)")
                axes[i].set_ylabel("Members % of Agreement")
                axes[i].axvline(mean_log, color="blue", linestyle="-", label=f"Mean: {mean_real}")
                axes[i].axvline(median_log, color="green", linestyle="-", label=f"Median: {median_real}")
                axes[i].set_title(f"Impacts for {month_num_to_name(month_num)} {year}")
                axes[i].xaxis.set_major_formatter(FuncFormatter(log_formatter))

                axes[i].annotate(f"Mean: {mean_log:.2f} ({mean_real:,})\nMedian: {median_log:.2f} ({median_real:,})",
                                 xy=(0.05, 0.95), xycoords="axes fraction", xytext=(10, -10),
                                 textcoords="offset points", va="top", ha="left",
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5))

            plt.tight_layout()
            plt.figtext(0.1, -0.1, 
                        "The first month is the initiation month. Subsequent months are forecast months, "
                        "corresponding to increasing leads based on the ordinal dates of the impact data. "
                        "The blue line represents the mean, and the green line represents the median. "
                        "To address the skewed nature of population distribution, the plots use a logarithmic scale.", 
                        wrap=True)

            plt.show()



############

# Suppress specific runtime warnings
warnings.filterwarnings("ignore", message="All-NaN axis encountered")
warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
warnings.filterwarnings("ignore", message="invalid value encountered in intersects")

# Base path configuration
climada_base_path = CONFIG.hazard.copernicus.local_data.dir()

def month_num_to_name_stats(month_num):
    """Convert a month number to the corresponding month name."""
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    return month_names[month_num - 1]

def load_impact_data(init_month, year_list, index_metric):
    """Load impact data dynamically based on initiation month and year."""
    year_str = str(year_list[0])
    impact_dir = os.path.join(climada_base_path, "seasonal_forecasts", "dwd", "sys21", year_str,
                              f"init{init_month}", "valid06_08", "impact", index_metric)

    if not os.path.isdir(impact_dir):
        raise FileNotFoundError(f"Impact directory does not exist: {impact_dir}")

    impact_files = [f for f in os.listdir(impact_dir) if f.startswith(f"{index_metric}_") and f.endswith(".hdf5")]
    if not impact_files:
        raise FileNotFoundError(f"No impact file found in {impact_dir}")

    impact_path = os.path.join(impact_dir, impact_files[0])
    return Impact.from_hdf5(impact_path)

def plot_statistics_per_location(year_list, index_metric, *init_months, scale="normal"):
    """
    Plots Mean, Median, Standard Deviation, Maximum, and Minimum impact across members for each location.

    Parameters:
    - year_list (list): List of years to process (e.g., [2023]).
    - index_metric (str): Climate index metric (e.g., "TX30").
    - *init_months (str): List of initiation months (e.g., "04", "05").
    - scale (str): 'normal' or 'log' scale for plotting.

    Returns:
    - None. Displays statistical maps with histograms and colorbars.
    """
    
    proj = ccrs.PlateCarree()
    valid_months = ["06", "07", "08"]  # Corresponds to Jun, Jul, Aug
    
    for init_month in init_months:
        try:
            impact_data = load_impact_data(init_month, year_list, index_metric)
        except FileNotFoundError as e:
            print(f"Skipping initiation month {init_month}: {e}")
            continue

        init_month_name = month_num_to_name_stats(int(init_month))

        # Extract unique forecast months
        unique_dates = np.unique(impact_data.date)[:3]  # Limit to 3 months (June, July, August)
        month_labels = [month_num_to_name_stats(int(valid_months[i])) for i in range(len(unique_dates))]

        # Reshape data per location and across members
        impact_array = impact_data.imp_mat.toarray().reshape(len(impact_data.event_id), -1)  # (members, locations)
        
        if impact_array.shape[0] == 0 or impact_array.shape[1] == 0:
            print(f"Skipping {init_month}: No data available.")
            continue

        # Compute statistics across members for each location
        statistics_data = {
            "Mean": np.nanmean(impact_array, axis=0),
            "Median": np.nanmedian(impact_array, axis=0),
            "Standard Deviation": np.nanstd(impact_array, axis=0),
            "Max": np.nanmax(impact_array, axis=0),
            "Min": np.nanmin(impact_array, axis=0),
        }

        # Define number of plots
        statistics_to_plot = list(statistics_data.keys())
        number_of_plots = len(statistics_to_plot)

        # Plot maps for each forecast month
        for i, forecast_month in enumerate(month_labels):
            fig = plt.figure(figsize=(5 * number_of_plots, 5))
            gs = gridspec.GridSpec(4, number_of_plots, height_ratios=[20, 1, 5, 1], hspace=0.3)
            fig.suptitle(f"Statistics for {forecast_month} {year_list[0]} (Initiation {init_month_name})", 
                         y=0.90, fontsize=12, weight="bold")

            # Iterate over each statistic
            for j, stat_name in enumerate(statistics_to_plot):
                stat_data = statistics_data[stat_name]
                vmin, vmax = np.nanmin(stat_data), np.nanmax(stat_data)
                norm = LogNorm(vmin=max(vmin, 1e-6), vmax=vmax) if scale == "log" and vmin > 0 and vmax > 0 else None

                # Map Plot
                ax_map = fig.add_subplot(gs[0, j], projection=proj)
                img = ax_map.scatter(impact_data.coord_exp[:, 1], impact_data.coord_exp[:, 0], 
                                     c=stat_data, cmap="RdYlBu_r", norm=norm, transform=ccrs.PlateCarree())
                ax_map.coastlines()
                ax_map.add_feature(cfeature.BORDERS, linestyle=":")
                ax_map.set_title(stat_name, fontsize=10)

                # Colorbar below each plot
                cbar_ax = fig.add_subplot(gs[1, j])
                cbar = plt.colorbar(img, cax=cbar_ax, orientation="horizontal")
                cbar.ax.tick_params(labelsize=8)

                # Histogram
                ax_hist = fig.add_subplot(gs[2, j])
                hist_data = stat_data[~np.isnan(stat_data)]
                if hist_data.size > 0:
                    bins = np.logspace(np.log10(max(vmin, 1e-6)), np.log10(vmax), 30) if scale == "log" else 30
                    ax_hist.hist(hist_data, bins=bins, color="gray", alpha=0.7)
                    if scale == "log":
                        ax_hist.set_xscale("log")
                    ax_hist.set_xlabel(f"{stat_name} Impacted Population", fontsize=10)
                    ax_hist.set_ylabel("Frequency", fontsize=10)
                    ax_hist.axvline(np.mean(hist_data), color="red", linestyle="--", linewidth=2, label="Mean")
                    ax_hist.axvline(np.median(hist_data), color="blue", linestyle="--", linewidth=2, label="Median")
                    ax_hist.legend(fontsize=8)

                # Display statistics under the histogram
                sum_val = np.nansum(hist_data)
                mean_val = np.nanmean(hist_data)
                std_val = np.nanstd(hist_data)
                min_val = np.nanmin(hist_data)
                max_val = np.nanmax(hist_data)

                stats_text = f"Sum: {sum_val:,.0f}\nMean: {mean_val:,.0f}\nStd: {std_val:.2f}\nMin: {min_val:,.0f}\nMax: {max_val:,.0f}"

                # Position statistics text directly below the histogram
                ax_hist.text(0, -1, stats_text, transform=ax_hist.transAxes, fontsize=10, 
                             verticalalignment="top", horizontalalignment="left")

            plt.show()




####### plot statistics per member

def month_num_to_name_stats(month_num):
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    return month_names[month_num - 1]

def event_num_to_month_name(base_month, event_num):
    new_month_index = (base_month - 1 + event_num - 1) % 12
    return month_num_to_name_stats(new_month_index + 1)

def load_impact_data(init_month, year_list, index_metric):
    climada_base_path = CONFIG.hazard.copernicus.local_data.dir()
    year_str = str(year_list[0])
    impact_dir = os.path.join(climada_base_path, "seasonal_forecasts", "dwd", "sys21", year_str,
                              f"init{init_month}", "valid06_08", "impact", index_metric)
    
    if not os.path.isdir(impact_dir):
        raise FileNotFoundError(f"Impact directory does not exist: {impact_dir}")
    
    impact_files = [f for f in os.listdir(impact_dir) if f.startswith(f"{index_metric}_") and f.endswith(".hdf5")]
    if not impact_files:
        raise FileNotFoundError(f"No impact file found in {impact_dir}")
    
    impact_path = os.path.join(impact_dir, impact_files[0])
    return Impact.from_hdf5(impact_path)

def plot_statistics_and_member_agreement(year_list, index_metric, agreement_threshold, *init_months):
    proj = ccrs.PlateCarree()
    valid_months = ["06", "07", "08"]
    
    for init_month in init_months:
        try:
            impact_data = load_impact_data(init_month, year_list, index_metric)
        except FileNotFoundError as e:
            print(f"Skipping initiation month {init_month}: {e}")
            continue
        
        init_month_name = month_num_to_name_stats(int(init_month))
        unique_dates = np.unique(impact_data.date)[:3]  # Only valid for 3 months
        month_labels = [month_num_to_name_stats(int(valid_months[i])) for i in range(len(unique_dates))]
        impact_array = impact_data.imp_mat.toarray().reshape(len(impact_data.event_id), -1)  
        
        if impact_array.shape[0] == 0 or impact_array.shape[1] == 0:
            print(f"Skipping {init_month}: No data available.")
            continue
        
        # Compute statistics
        mean_impact = np.nanmean(impact_array, axis=0)
        std_impact = np.nanstd(impact_array, axis=0)  # Standard deviation
        max_impact = np.nanmax(impact_array, axis=0)
        p95_impact = np.nanpercentile(impact_array, 95, axis=0)

        for i, forecast_month in enumerate(month_labels):
            fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), subplot_kw={'projection': proj})
            fig.suptitle(f"Statistics for {forecast_month} {year_list[0]} (Initiation {init_month_name})", fontsize=12, weight="bold", y=0.87)
            
            data_list = [mean_impact, max_impact, p95_impact]
            titles = ["Log Mean Exposed Population", "Log Maximum Exposed Population", "Log Percentile 95 Exposed Population"]
            
            cmap = plt.get_cmap("RdYlBu_r")
            norm_c = mcolors.BoundaryNorm(np.arange(0, 110, 10), 11, clip=True)
            cmap_c = mcolors.ListedColormap([
                "#4FA4A3", "#5CB8B2", "#9BD5CF", "#CDE7E6",
                "#F3F3F2", "#F7F3C4", "#F8E891", "#F2D973",
                "#DBB342", "#C99E32", "#AE8232"
            ])
            
            # Determine zoomed region
            min_lon, max_lon = np.min(impact_data.coord_exp[:, 1]), np.max(impact_data.coord_exp[:, 1])
            min_lat, max_lat = np.min(impact_data.coord_exp[:, 0]), np.max(impact_data.coord_exp[:, 0])
            zoom_extent = [min_lon - 1, max_lon + 1, min_lat - 1, max_lat + 1]  # Adding buffer

            for j in range(3):
                norm = mcolors.LogNorm(vmin=max(np.nanmin(data_list[j]), 1e-6), vmax=np.nanmax(data_list[j]))
                img = axs[j, 0].scatter(impact_data.coord_exp[:, 1], impact_data.coord_exp[:, 0], c=data_list[j], cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
                axs[j, 0].coastlines()
                axs[j, 0].add_feature(cfeature.BORDERS, linestyle=":")
                axs[j, 0].set_title(titles[j], fontsize=12)
                axs[j, 0].set_extent(zoom_extent, crs=proj)
                cbar = fig.colorbar(img, ax=axs[j, 0], orientation="horizontal", pad=0.08, aspect=20, shrink=0.6)
                cbar.set_label(label=titles[j], size=12)
                cbar.ax.tick_params(labelsize=12)
                
                # Compute agreement based on one standard deviation around the statistic
                lower_bound = data_list[j] - std_impact
                upper_bound = data_list[j] + std_impact
                agreement_data = ((impact_array >= lower_bound) & (impact_array <= upper_bound)).astype(int).mean(axis=0)
                
                # Show high agreement (>70% of members)
                high_agreement_data = (agreement_data > agreement_threshold).astype(int)

                agreement_img = axs[j, 1].scatter(impact_data.coord_exp[:, 1], impact_data.coord_exp[:, 0], c=high_agreement_data, cmap="gray", transform=ccrs.PlateCarree())
                axs[j, 1].coastlines()
                axs[j, 1].add_feature(cfeature.BORDERS, linestyle=":")
                axs[j, 1].set_title(f"High Agreement (>{agreement_threshold*100}%)", fontsize=12)
                axs[j, 1].set_extent(zoom_extent, crs=proj)
                fig.colorbar(agreement_img, ax=axs[j, 1], orientation="horizontal", pad=0.08, aspect=20, shrink=0.6)
                
                perc_agreement_img = axs[j, 2].scatter(impact_data.coord_exp[:, 1], impact_data.coord_exp[:, 0], c=agreement_data*100, cmap=cmap_c, norm=norm_c, transform=ccrs.PlateCarree())
                axs[j, 2].coastlines()
                axs[j, 2].add_feature(cfeature.BORDERS, linestyle=":")
                axs[j, 2].set_title("Percentage Members Agreement", fontsize=12)
                axs[j, 2].set_extent(zoom_extent, crs=proj)
                fig.colorbar(perc_agreement_img, ax=axs[j, 2], orientation="horizontal", pad=0.08, aspect=20, shrink=0.6)
            
            plt.figtext(
                0.0, 0.0, 
                "This map displays the spatial distribution of the exposed population and the agreement among members for seasonal forecasts. "
                "Each row represents different aspects: the mean, maximum, and 95th percentile of exposed population (log-transformed). "
                "From left to right: "
                "1) The first column shows the impact data, representing the log-transformed mean, max, and 95th percentile. "
                "2) The second column highlights areas where at least 70% of members agree that the index value is within one standard deviation of the respective metric (Mean, Max, P95). "
                "3) The third column presents the full range of agreement, displaying the percentage of members that fall within this range across all locations.",  
                wrap=True, horizontalalignment='left', fontsize=12
            )
            plt.subplots_adjust(hspace=0.05, wspace=0.1)
            plt.show()

