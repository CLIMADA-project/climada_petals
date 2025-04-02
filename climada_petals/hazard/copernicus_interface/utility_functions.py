import os
import warnings
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from climada.hazard import Hazard
from climada.engine import Impact
from climada.util.config import CONFIG
from climada_petals.hazard.copernicus_interface.create_seasonal_forecast_hazard import SeasonalForecast

warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
warnings.filterwarnings("ignore", message="All-NaN axis encountered")
warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
warnings.filterwarnings("ignore", message="invalid value encountered in intersects")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="This figure includes Axes that are not compatible with tight_layout",
)

climada_base_path = CONFIG.hazard.copernicus.local_data.dir()



#### plot forescast ####
def plot_forecast(
    forecast_year, initiation_month_str, target_month, handler, index_metric="TX30"
):
    """
    Plot all 50 ensemble members for a given seasonal forecast month.

    Parameters
    ----------
    forecast_year : int
        The year of the seasonal forecast (e.g., 2023).
    initiation_month_str : str
        Forecast initiation month in string format (e.g., "03" for March).
    target_month : str
        Forecast valid month in "YYYY-MM" format (must match the 'step' coordinate in the dataset).
    handler : SeasonalForecast
        Object that provides access to the file structure via get_pipeline_path(...).
    index_metric : str, optional
        Name of the climate index variable to visualize. Default is "TX30".

    Raises
    ------
    ValueError
        If dataset or required variables/coordinates are not found, or loading fails.

    Returns
    -------
    None
        Displays a grid of maps (5 rows × 10 columns) with one plot per ensemble member.
    """
    indices_paths = handler.get_pipeline_path(
        forecast_year, initiation_month_str, "indices"
    )

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
        raise ValueError(
            f"'step' coordinate not found. Available coordinates: {list(ds.coords.keys())}"
        )

    # Verify variable name
    if index_metric not in ds.variables:
        raise ValueError(
            f"Variable '{index_metric}' not found in dataset. Available variables: {list(ds.data_vars.keys())}"
        )

    # Create subplots
    fig, axs = plt.subplots(
        nrows=5,
        ncols=10,
        figsize=(25, 15),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axs = axs.flatten()  # Flatten array for easy iteration

    # Plot each ensemble member
    for i in range(50):
        ax = axs[i]
        member_data = data_for_month.isel(number=i)[index_metric]  # Select dynamically
        p = member_data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            x="longitude",
            y="latitude",
            add_colorbar=False,
            cmap="viridis",
        )

        ax.coastlines(color="white", linewidth=1)  # Set coastline color to white
        ax.add_feature(
            cfeature.BORDERS, edgecolor="white", linewidth=1
        )  # Add country borders in white
        ax.set_title(f"Member {i+1}")

    # Adjust layout for colorbar
    plt.subplots_adjust(
        bottom=0.1, top=0.9, left=0.05, right=0.95, wspace=0.1, hspace=0.1
    )

    # Add a color bar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.015])
    fig.colorbar(p, cax=cbar_ax, orientation="horizontal")

    plt.show()



#### plot_ensemble_index_summary ####

def extract_statistics(file_path, forecast_months):
    """
    Extract ensemble summary statistics from a NetCDF dataset for specified forecast months.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the NetCDF file containing ensemble statistics.
    forecast_months : list of str
        List of forecast months in "YYYY-MM" format to extract (matches 'step' coordinate in NetCDF).

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with rows indexed by forecast_months and columns:
        'Mean', 'Median', 'Max', 'Min', 'Std'.
        Returns None if file not found or extraction fails.
    """
    try:
        ds_stats = xr.open_dataset(file_path, engine="netcdf4")
        stats = {
            "Mean": ds_stats["ensemble_mean"]
            .sel(step=forecast_months)
            .mean(dim=["latitude", "longitude"]),
            "Median": ds_stats["ensemble_median"]
            .sel(step=forecast_months)
            .mean(dim=["latitude", "longitude"]),
            "Max": ds_stats["ensemble_max"]
            .sel(step=forecast_months)
            .mean(dim=["latitude", "longitude"]),
            "Min": ds_stats["ensemble_min"]
            .sel(step=forecast_months)
            .mean(dim=["latitude", "longitude"]),
            "Std": ds_stats["ensemble_std"]
            .sel(step=forecast_months)
            .mean(dim=["latitude", "longitude"]),
        }
        return pd.DataFrame(
            {key: val.values for key, val in stats.items()}, index=forecast_months
        )
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def plot_ensemble_index_summary(
    forecast_year, initiation_months, valid_periods, handler, index_metric="TX30"
):
    """
    Plot ensemble forecast statistics (Mean, Median, Min, Max, Std) per initiation month.

    Parameters
    ----------
    forecast_year : int
        Year of the forecast initialization.
    initiation_months : list of str
        Initialization months (e.g., ['03', '04']).
    valid_periods : list of str
        Forecast valid months (e.g., ['06', '07', '08']).
    handler : object
        Object with method get_pipeline_path(...) that returns file paths for the given config.
    index_metric : str, optional
        Name of the index to be plotted (default is 'TX30').

    Returns
    -------
    None
        Displays matplotlib plots showing forecast statistics over time for each init month.
    """

    forecast_months = [
        f"{forecast_year}-{month}" for month in valid_periods
    ]  # Convert to "YYYY-MM" format
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
        init_month = file_path.split("/init")[1][
            :2
        ]  # Extract initiation month from path
        ax.plot(df.index, df["Mean"], label="Mean", color="darkblue", linewidth=2)
        ax.plot(df.index, df["Median"], label="Median", color="orange", linewidth=1)
        ax.fill_between(
            df.index,
            df["Min"],
            df["Max"],
            color="skyblue",
            alpha=0.3,
            label="Min-Max Range",
        )
        ax.fill_between(
            df.index,
            df["Mean"] - df["Std"],
            df["Mean"] + df["Std"],
            color="salmon",
            alpha=0.4,
            label="Mean ± Std",
        )
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"{index_metric} Forecast (Init: {init_month})", fontsize=14)
        ax.set_xlabel("Forecast Month", fontsize=12)
        ax.set_ylabel(f"Number of {index_metric}", fontsize=12)
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True)

    plt.tight_layout()
    plt.show()



#### plot_individual_and_aggregated_impacts ####

def thousands_comma_formatter(x, pos):
    """
    Format tick values as thousands with commas.

    Parameters
    ----------
    x : float
        Tick value (usually raw population value).
    pos : int
        Tick position (required by matplotlib, not used here).

    Returns
    -------
    str
        Tick label in 'X,XXXK' format.
    """
    return "{:,.0f}K".format(x * 1e-3)


def load_impact_data(init_month, year_list, index_metric):
    """
    Load CLIMADA Impact object for a given year, initialization month, and index metric.

    Parameters
    ----------
    init_month : str
        Initialization month as a two-digit string (e.g., "04").
    year_list : list of int
        List of forecast years (only the first year is used).
    index_metric : str
        Climate index used to select the impact file (e.g., "TX30").

    Returns
    -------
    Impact
        CLIMADA Impact object for the selected configuration.

    Raises
    ------
    FileNotFoundError
        If the impact directory or relevant HDF5 file cannot be found.
    """
    year_str = str(year_list[0])

    impact_dir = os.path.join(
        climada_base_path,
        "seasonal_forecasts",
        "dwd",
        "sys21",
        year_str,
        f"init{init_month}",
        "valid06_08",
        "impact",
        index_metric,
    )

    if not os.path.isdir(impact_dir):
        raise FileNotFoundError(f"Impact directory does not exist: {impact_dir}")

    impact_files = [
        f
        for f in os.listdir(impact_dir)
        if f.startswith(f"{index_metric}_") and f.endswith(".hdf5")
    ]
    if not impact_files:
        raise FileNotFoundError(f"No impact file found in {impact_dir}")

    impact_path = os.path.join(impact_dir, impact_files[0])
    return Impact.from_hdf5(impact_path)


def plot_individual_and_aggregated_impacts(year_list, index_metric, *init_months):
    """
    Plot individual ensemble member impact time series and aggregated statistics (mean, median, std).

    Parameters
    ----------
    year_list : list of int
        List of forecast years to load (only the first element is used).
    index_metric : str
        Name of the climate index metric (e.g., "TX30").
    *init_months : str
        One or more initialization months as strings (e.g., "04", "05").

    Returns
    -------
    None
        Displays line plots showing ensemble impacts over time for each init month.

    Notes
    -----
    - Each subplot corresponds to an initialization month.
    - Up to 50 ensemble members are visualized.
    - Overlaid lines show the ensemble mean, median, and ±1 std range.
    - The y-axis uses a thousands-based formatter for better readability.
    """
    month_names = ["Jun", "Jul", "Aug"]
    cmap = plt.cm.get_cmap("tab20", 50)

    fig, axes = plt.subplots(
        nrows=len(init_months), figsize=(5, 3 * len(init_months)), sharex=True
    )

    if len(init_months) == 1:
        axes = [axes]

    for ax, init_month in zip(axes, init_months):
        impact_data = load_impact_data(init_month, year_list, index_metric)

        unique_dates = np.unique(impact_data.date)[:3]
        month_labels = [
            datetime.fromordinal(date).strftime("%b") for date in unique_dates
        ]

        all_member_impacts = [[] for _ in range(50)]

        for i, date in enumerate(unique_dates):
            event_indices = np.where(impact_data.date == date)[0][:50]

            for j, idx in enumerate(event_indices):
                all_member_impacts[j].append(impact_data.at_event[idx])

        for i in range(50):
            ax.plot(
                month_labels,
                all_member_impacts[i],
                marker="o",
                linestyle="-",
                linewidth=1,
                alpha=0.3,
                color=cmap(i),
                label=f"Member {i}" if i == 0 else "_nolegend_",
            )

        mean_impact = np.mean(all_member_impacts, axis=0)
        median_impact = np.median(all_member_impacts, axis=0)
        std_impact = np.std(all_member_impacts, axis=0)

        ax.plot(month_labels, mean_impact, "k--", linewidth=1.5, label="Mean")
        ax.plot(month_labels, median_impact, "k-", linewidth=1.5, label="Median")
        ax.fill_between(
            month_labels,
            mean_impact - std_impact,
            mean_impact + std_impact,
            color="grey",
            alpha=0.15,
            label="Mean ± Std",
        )

        ax.set_title(f"Impact Values - Init {init_month}", fontsize=10)
        ax.set_ylabel("Population Affected (Thousands)", fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_comma_formatter))
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True)

    axes[-1].set_xlabel("Forecast Month", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()



#### plot_impact_distributions ####

def month_num_to_name(month_num):
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    try:
        return month_names[int(month_num) - 1]
    except (ValueError, IndexError):
        return f"Invalid month number: {month_num}"


def log_formatter(x, pos):
    """
    Format tick labels for log-scaled x-axis using base-10 notation.

    Parameters
    ----------
    x : float
        Tick value.
    pos : int
        Tick position (unused but required by matplotlib formatter).

    Returns
    -------
    str
        Formatted tick label as a LaTeX-style power of 10.
    """
    return f"$10^{{{int(x)}}}$"


def load_impact_data(init_month, year_list, index_metric):
    """
    Load the Impact object for a given year, initialization month, and index metric.

    Parameters
    ----------
    init_month : str
        Initialization month as a two-digit string (e.g., "04").
    year_list : list of int
        List of forecast years (only the first year is used).
    index_metric : str
        Name of the index metric (e.g., "TX30").

    Returns
    -------
    Impact
        A CLIMADA Impact object loaded from the corresponding HDF5 file.

    Raises
    ------
    FileNotFoundError
        If the impact directory or matching file is not found.
    """
    year_str = str(year_list[0])
    impact_dir = os.path.join(
        climada_base_path,
        "seasonal_forecasts",
        "dwd",
        "sys21",
        year_str,
        f"init{init_month}",
        "valid06_08",
        "impact",
        index_metric,
    )

    if not os.path.isdir(impact_dir):
        raise FileNotFoundError(f"Impact directory does not exist: {impact_dir}")

    impact_files = [
        f
        for f in os.listdir(impact_dir)
        if f.startswith(f"{index_metric}_") and f.endswith(".hdf5")
    ]
    if not impact_files:
        raise FileNotFoundError(f"No impact file found in {impact_dir}")

    impact_path = os.path.join(impact_dir, impact_files[0])
    return Impact.from_hdf5(impact_path)


def plot_impact_distributions(year_list, index_metric, init_months):
    """
    Plot log-scaled histograms of forecasted impact values across ensemble members.

    Parameters
    ----------
    year_list : list of int
        List of forecast years to analyze.
    index_metric : str
        Name of the climate index (e.g., "TX30").
    init_months : list of str
        List of initialization months (e.g., ["04", "05", "06"]).

    Returns
    -------
    None

    Notes
    -----
    - Impacts are log-transformed to better represent skewed distributions.
    - Each subplot shows the distribution for one valid forecast month.
    - Mean and median lines are overlaid for interpretability.
    - A fitted curve (up to cubic polynomial) is added to illustrate distribution shape.
    - Uses up to 50 ensemble members per valid forecast month.
    """
    max_members = 50

    for year in year_list:
        for month in init_months:
            impact_data = load_impact_data(month, [year], index_metric)

            unique_dates = np.unique(impact_data.date)[:3]
            month_labels = [
                datetime.fromordinal(date).strftime("%b") for date in unique_dates
            ]

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

            fig, axes = plt.subplots(
                nrows=(len(all_impacts) + 1) // 2, ncols=2, figsize=(8, 5)
            )
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

                axes[i].bar(
                    bin_midpoints,
                    hist_percentage,
                    width=np.diff(edges),
                    color="black",
                    alpha=0.5,
                    edgecolor="black",
                    align="edge",
                )

                if len(bin_midpoints) > 3:  # Ensure at least 4 points for cubic fit
                    try:
                        coefficients = np.polyfit(
                            bin_midpoints,
                            hist_percentage,
                            min(3, len(bin_midpoints) - 1),
                        )
                        polynomial = np.poly1d(coefficients)
                        xs = np.linspace(bin_midpoints[0], bin_midpoints[-1], 100)
                        ys = polynomial(xs)
                        axes[i].plot(
                            xs, ys, color="black", linewidth=1.2, linestyle="--"
                        )
                    except np.linalg.LinAlgError:
                        print(
                            f"Skipping polynomial fit for {month_num_to_name(month_num)} {year} due to instability."
                        )

                axes[i].set_xlabel("Population Affected (Log Scale)")
                axes[i].set_ylabel("Members % of Agreement")
                axes[i].axvline(
                    mean_log, color="blue", linestyle="-", label=f"Mean: {mean_real}"
                )
                axes[i].axvline(
                    median_log,
                    color="green",
                    linestyle="-",
                    label=f"Median: {median_real}",
                )
                axes[i].set_title(f"Impacts for {month_num_to_name(month_num)} {year}")
                axes[i].xaxis.set_major_formatter(FuncFormatter(log_formatter))

                axes[i].annotate(
                    f"Mean: {mean_log:.2f} ({mean_real:,})\nMedian: {median_log:.2f} ({median_real:,})",
                    xy=(0.05, 0.95),
                    xycoords="axes fraction",
                    xytext=(10, -10),
                    textcoords="offset points",
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5),
                )

            plt.tight_layout()
            plt.figtext(
                0.1,
                -0.1,
                "The first month is the initiation month. Subsequent months are forecast months, "
                "corresponding to increasing leads based on the ordinal dates of the impact data. "
                "The blue line represents the mean, and the green line represents the median. "
                "To address the skewed nature of population distribution, the plots use a logarithmic scale.",
                wrap=True,
            )

            plt.show()



#### plot_statistics_per_location ####

# Suppress specific runtime warnings
warnings.filterwarnings("ignore", message="All-NaN axis encountered")
warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
warnings.filterwarnings("ignore", message="invalid value encountered in intersects")

def month_num_to_name_stats(month_num):
    """
    Convert a numeric month to its English name.

    Parameters
    ----------
    month_num : int
        Numeric representation of the month (1 = January, ..., 12 = December).

    Returns
    -------
    str
        English name of the corresponding month.
    """
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    return month_names[month_num - 1]


def load_impact_data(init_month, year_list, index_metric):
    """
    Load CLIMADA Impact object for a given initialization month and year.

    Parameters
    ----------
    init_month : str
        Initialization month as a two-digit string (e.g., '04').
    year_list : list of int
        List containing the target forecast year. Only the first entry is used.
    index_metric : str
        Climate index metric (e.g., 'TX30').

    Returns
    -------
    climada.hazard.Impact
        Loaded Impact object containing impact data.

    Raises
    ------
    FileNotFoundError
        If the impact directory or the impact file does not exist.
    """
    year_str = str(year_list[0])
    impact_dir = os.path.join(
        climada_base_path,
        "seasonal_forecasts",
        "dwd",
        "sys21",
        year_str,
        f"init{init_month}",
        "valid06_08",
        "impact",
        index_metric,
    )

    if not os.path.isdir(impact_dir):
        raise FileNotFoundError(f"Impact directory does not exist: {impact_dir}")

    impact_files = [
        f
        for f in os.listdir(impact_dir)
        if f.startswith(f"{index_metric}_") and f.endswith(".hdf5")
    ]
    if not impact_files:
        raise FileNotFoundError(f"No impact file found in {impact_dir}")

    impact_path = os.path.join(impact_dir, impact_files[0])
    return Impact.from_hdf5(impact_path)


def plot_statistics_per_location(year_list, index_metric, *init_months, scale="normal"):
    """
    Plot spatial impact statistics across members for each location.

    Parameters
    ----------
    year_list : list of int
        List of years to process (e.g., [2023]).
    index_metric : str
        Climate index metric (e.g., 'TX30').
    *init_months : str
        Variable number of initialization months (e.g., '04', '05').
    scale : str, optional
        Scale of the plots: 'normal' for linear or 'log' for logarithmic (default is 'normal').

    Returns
    -------
    None

    Notes
    -----
    For each forecast month (June–August), this function:
    - Loads the corresponding impact matrix.
    - Computes statistics (mean, median, std, min, max) across members per grid cell.
    - Generates:
        1) A spatial scatter map of each statistic.
        2) A histogram of grid-level values.
        3) Summary statistics printed below each histogram.
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
        unique_dates = np.unique(impact_data.date)[
            :3
        ]  # Limit to 3 months (June, July, August)
        month_labels = [
            month_num_to_name_stats(int(valid_months[i]))
            for i in range(len(unique_dates))
        ]

        # Reshape data per location and across members
        impact_array = impact_data.imp_mat.toarray().reshape(
            len(impact_data.event_id), -1
        )  # (members, locations)

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
            gs = gridspec.GridSpec(
                4, number_of_plots, height_ratios=[20, 1, 5, 1], hspace=0.3
            )
            fig.suptitle(
                f"Statistics for {forecast_month} {year_list[0]} (Initiation {init_month_name})",
                y=0.90,
                fontsize=12,
                weight="bold",
            )

            # Iterate over each statistic
            for j, stat_name in enumerate(statistics_to_plot):
                stat_data = statistics_data[stat_name]
                vmin, vmax = np.nanmin(stat_data), np.nanmax(stat_data)
                norm = (
                    LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
                    if scale == "log" and vmin > 0 and vmax > 0
                    else None
                )

                # Map Plot
                ax_map = fig.add_subplot(gs[0, j], projection=proj)
                img = ax_map.scatter(
                    impact_data.coord_exp[:, 1],
                    impact_data.coord_exp[:, 0],
                    c=stat_data,
                    cmap="RdYlBu_r",
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                )
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
                    bins = (
                        np.logspace(np.log10(max(vmin, 1e-6)), np.log10(vmax), 30)
                        if scale == "log"
                        else 30
                    )
                    ax_hist.hist(hist_data, bins=bins, color="gray", alpha=0.7)
                    if scale == "log":
                        ax_hist.set_xscale("log")
                    ax_hist.set_xlabel(f"{stat_name} Impacted Population", fontsize=10)
                    ax_hist.set_ylabel("Frequency", fontsize=10)
                    ax_hist.axvline(
                        np.mean(hist_data),
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label="Mean",
                    )
                    ax_hist.axvline(
                        np.median(hist_data),
                        color="blue",
                        linestyle="--",
                        linewidth=2,
                        label="Median",
                    )
                    ax_hist.legend(fontsize=8)

                # Display statistics under the histogram
                sum_val = np.nansum(hist_data)
                mean_val = np.nanmean(hist_data)
                std_val = np.nanstd(hist_data)
                min_val = np.nanmin(hist_data)
                max_val = np.nanmax(hist_data)

                stats_text = f"Sum: {sum_val:,.0f}\nMean: {mean_val:,.0f}\nStd: {std_val:.2f}\nMin: {min_val:,.0f}\nMax: {max_val:,.0f}"

                # Position statistics text directly below the histogram
                ax_hist.text(
                    0,
                    -1,
                    stats_text,
                    transform=ax_hist.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    horizontalalignment="left",
                )

            plt.show()



#### plot statistics per member ####

def month_num_to_name_stats(month_num):
    """
    Convert a numeric month (1–12) to its English name.

    Parameters
    ----------
    month_num : int
        Month number (1 = January, ..., 12 = December).

    Returns
    -------
    str
        Month name corresponding to the input number.
    """
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    return month_names[month_num - 1]


def event_num_to_month_name(base_month, event_num):
    """
    Convert an event number into a calendar month name, based on the base month.

    Parameters
    ----------
    base_month : int
        Initialization month number (1 = January, ..., 12 = December).
    event_num : int
        Relative event offset (e.g., 1 = first month, 2 = second month...).

    Returns
    -------
    str
        Forecast month name accounting for offset from initialization month.
    """
    new_month_index = (base_month - 1 + event_num - 1) % 12
    return month_num_to_name_stats(new_month_index + 1)


def load_impact_data(init_month, year_list, index_metric):
    """
    Load a CLIMADA Impact object from the impact directory for a given init month and year.

    Parameters
    ----------
    init_month : str
        Initialization month (e.g., '03').
    year_list : list of int
        List containing a single forecast year (used to resolve folder structure).
    index_metric : str
        Name of the climate index (e.g., 'TX30', 'HW').

    Returns
    -------
    climada.hazard.Impact
        Loaded Impact object containing impact data for the given forecast.

    Raises
    ------
    FileNotFoundError
        If the impact directory or expected file does not exist.
    """
    year_str = str(year_list[0])
    impact_dir = os.path.join(
        climada_base_path,
        "seasonal_forecasts",
        "dwd",
        "sys21",
        year_str,
        f"init{init_month}",
        "valid06_08",
        "impact",
        index_metric,
    )

    if not os.path.isdir(impact_dir):
        raise FileNotFoundError(f"Impact directory does not exist: {impact_dir}")

    impact_files = [
        f
        for f in os.listdir(impact_dir)
        if f.startswith(f"{index_metric}_") and f.endswith(".hdf5")
    ]
    if not impact_files:
        raise FileNotFoundError(f"No impact file found in {impact_dir}")

    impact_path = os.path.join(impact_dir, impact_files[0])
    return Impact.from_hdf5(impact_path)


def plot_statistics_and_member_agreement(
    year_list, index_metric, agreement_threshold, *init_months
):
    """
    Plot spatial impact statistics and ensemble member agreement for forecast data.

    Parameters
    ----------
    year_list : list of int
        List of forecast years (currently only the first entry is used).
    index_metric : str
        Climate index metric (e.g., 'TX30', 'HW').
    agreement_threshold : float
        Threshold for agreement (e.g., 0.7 = 70%) used to highlight consistent regions.
    *init_months : str
        Variable number of initialization months (e.g., '03', '04', '05').

    Returns
    -------
    None

    Notes
    -----
    For each initialization month, this function will:
    - Load the corresponding Impact object.
    - Calculate mean, max, std, and 95th percentile values.
    - Create a 3x3 plot grid:
        Column 1: spatial distribution of log-transformed statistics.
        Column 2: binary high agreement map where member values fall within ±1 std.
        Column 3: percentage of members that agree with the central estimate.
    """
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
        month_labels = [
            month_num_to_name_stats(int(valid_months[i]))
            for i in range(len(unique_dates))
        ]
        impact_array = impact_data.imp_mat.toarray().reshape(
            len(impact_data.event_id), -1
        )

        if impact_array.shape[0] == 0 or impact_array.shape[1] == 0:
            print(f"Skipping {init_month}: No data available.")
            continue

        # Compute statistics
        mean_impact = np.nanmean(impact_array, axis=0)
        std_impact = np.nanstd(impact_array, axis=0)  # Standard deviation
        max_impact = np.nanmax(impact_array, axis=0)
        p95_impact = np.nanpercentile(impact_array, 95, axis=0)

        for i, forecast_month in enumerate(month_labels):
            fig, axs = plt.subplots(
                nrows=3, ncols=3, figsize=(15, 15), subplot_kw={"projection": proj}
            )
            fig.suptitle(
                f"Statistics for {forecast_month} {year_list[0]} (Initiation {init_month_name})",
                fontsize=12,
                weight="bold",
                y=0.87,
            )

            data_list = [mean_impact, max_impact, p95_impact]
            titles = [
                "Log Mean Exposed Population",
                "Log Maximum Exposed Population",
                "Log Percentile 95 Exposed Population",
            ]

            cmap = plt.get_cmap("RdYlBu_r")
            norm_c = mcolors.BoundaryNorm(np.arange(0, 110, 10), 11, clip=True)
            cmap_c = mcolors.ListedColormap(
                [
                    "#4FA4A3",
                    "#5CB8B2",
                    "#9BD5CF",
                    "#CDE7E6",
                    "#F3F3F2",
                    "#F7F3C4",
                    "#F8E891",
                    "#F2D973",
                    "#DBB342",
                    "#C99E32",
                    "#AE8232",
                ]
            )

            # Determine zoomed region
            min_lon, max_lon = np.min(impact_data.coord_exp[:, 1]), np.max(
                impact_data.coord_exp[:, 1]
            )
            min_lat, max_lat = np.min(impact_data.coord_exp[:, 0]), np.max(
                impact_data.coord_exp[:, 0]
            )
            zoom_extent = [
                min_lon - 1,
                max_lon + 1,
                min_lat - 1,
                max_lat + 1,
            ]  # Adding buffer

            for j in range(3):
                norm = mcolors.LogNorm(
                    vmin=max(np.nanmin(data_list[j]), 1e-6),
                    vmax=np.nanmax(data_list[j]),
                )
                img = axs[j, 0].scatter(
                    impact_data.coord_exp[:, 1],
                    impact_data.coord_exp[:, 0],
                    c=data_list[j],
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                )
                axs[j, 0].coastlines()
                axs[j, 0].add_feature(cfeature.BORDERS, linestyle=":")
                axs[j, 0].set_title(titles[j], fontsize=12)
                axs[j, 0].set_extent(zoom_extent, crs=proj)
                cbar = fig.colorbar(
                    img,
                    ax=axs[j, 0],
                    orientation="horizontal",
                    pad=0.08,
                    aspect=20,
                    shrink=0.6,
                )
                cbar.set_label(label=titles[j], size=12)
                cbar.ax.tick_params(labelsize=12)

                # Compute agreement based on one standard deviation around the statistic
                lower_bound = data_list[j] - std_impact
                upper_bound = data_list[j] + std_impact
                agreement_data = (
                    ((impact_array >= lower_bound) & (impact_array <= upper_bound))
                    .astype(int)
                    .mean(axis=0)
                )

                # Show high agreement (>70% of members)
                high_agreement_data = (agreement_data > agreement_threshold).astype(int)

                agreement_img = axs[j, 1].scatter(
                    impact_data.coord_exp[:, 1],
                    impact_data.coord_exp[:, 0],
                    c=high_agreement_data,
                    cmap="gray",
                    transform=ccrs.PlateCarree(),
                )
                axs[j, 1].coastlines()
                axs[j, 1].add_feature(cfeature.BORDERS, linestyle=":")
                axs[j, 1].set_title(
                    f"High Agreement (>{agreement_threshold*100}%)", fontsize=12
                )
                axs[j, 1].set_extent(zoom_extent, crs=proj)
                fig.colorbar(
                    agreement_img,
                    ax=axs[j, 1],
                    orientation="horizontal",
                    pad=0.08,
                    aspect=20,
                    shrink=0.6,
                )

                perc_agreement_img = axs[j, 2].scatter(
                    impact_data.coord_exp[:, 1],
                    impact_data.coord_exp[:, 0],
                    c=agreement_data * 100,
                    cmap=cmap_c,
                    norm=norm_c,
                    transform=ccrs.PlateCarree(),
                )
                axs[j, 2].coastlines()
                axs[j, 2].add_feature(cfeature.BORDERS, linestyle=":")
                axs[j, 2].set_title("Percentage Members Agreement", fontsize=12)
                axs[j, 2].set_extent(zoom_extent, crs=proj)
                fig.colorbar(
                    perc_agreement_img,
                    ax=axs[j, 2],
                    orientation="horizontal",
                    pad=0.08,
                    aspect=20,
                    shrink=0.6,
                )

            plt.figtext(
                0.0,
                0.0,
                "This map displays the spatial distribution of the exposed population and the agreement among members for seasonal forecasts. "
                "Each row represents different aspects: the mean, maximum, and 95th percentile of exposed population (log-transformed). "
                "From left to right: "
                "1) The first column shows the impact data, representing the log-transformed mean, max, and 95th percentile. "
                "2) The second column highlights areas where at least 70% of members agree that the index value is within one standard deviation of the respective metric (Mean, Max, P95). "
                "3) The third column presents the full range of agreement, displaying the percentage of members that fall within this range across all locations.",
                wrap=True,
                horizontalalignment="left",
                fontsize=12,
            )
            plt.subplots_adjust(hspace=0.05, wspace=0.1)
            plt.show()



#### plot_intensity_distributions ####

def analyze_hazard_intensities(year, months, handler):
    """
    Load CLIMADA Hazard objects and extract intensity values for each event.

    Parameters
    ----------
    year : int
        Forecast year of interest.
    months : list of str
        List of initiation months (e.g., ['03', '04', '05']).
    handler : object
        Object providing the method `get_pipeline_path(year, month, "hazard")`
        to resolve paths to hazard HDF5 files.

    Returns
    -------
    dict
        Dictionary mapping each month to a sub-dictionary with event names as keys
        and flattened intensity arrays as values. Example:
        {
            '03': {'member0': array([...]), 'member1': array([...]), ...},
            '04': {...},
            ...
        }
    """
    def load_hazard_data(year, month):
        hazard_path = handler.get_pipeline_path(year, month, "hazard")
        try:
            return Hazard.from_hdf5(hazard_path)
        except FileNotFoundError:
            print(f"No hazard data for month {month}")
            return None

    hazards = {month: load_hazard_data(year, month) for month in months}

    intensity_data = {}
    for month, hazard in hazards.items():
        if hazard:
            intensity_data[month] = {
                event: hazard.intensity[i, :].toarray().flatten()
                for i, event in enumerate(hazard.event_name)
            }
        else:
            intensity_data[month] = {}
    return intensity_data

def print_summary_statistics(year, months, handler):
    """
    Print summary statistics (min, max, mean, median, std) of hazard intensities.

    Parameters
    ----------
    year : int
        Forecast year to analyze.
    months : list of str
        List of initiation months (e.g., ['03', '04', '05']).
    handler : object
        Object used to resolve file paths to hazard HDF5 files.

    Returns
    -------
    None
    """
    intensity_data = analyze_hazard_intensities(year, months, handler)
    for month, intensity_dict in intensity_data.items():
        all_intensities = np.concatenate(list(intensity_dict.values())) if intensity_dict else np.array([])
        print(f"\nInitiation Month {month}")
        if all_intensities.size > 0:
            print(f"  Max intensity: {np.max(all_intensities):.2f}")
            print(f"  Min intensity: {np.min(all_intensities):.2f}")
            print(f"  Mean: {np.mean(all_intensities):.2f}")
            print(f"  Median: {np.median(all_intensities):.2f}")
            print(f"  Std Dev: {np.std(all_intensities):.2f}")
        else:
            print("No intensity data found.")

def plot_intensity_distributions(year, months, handler):
    """
    Plot histograms of hazard intensities for each forecast initiation month.

    Parameters
    ----------
    year : int
        Forecast year to visualize.
    months : list of str
        List of initiation months (e.g., ['03', '04', '05']).
    handler : object
        Object used to resolve file paths to hazard HDF5 files.

    Returns
    -------
    None
    """
    intensity_data = analyze_hazard_intensities(year, months, handler)

    fig, axes = plt.subplots(1, len(intensity_data), figsize=(15, 5), sharey=True)
    if len(intensity_data) == 1:
        axes = [axes]
    colors = sns.color_palette("tab10", n_colors=50)

    for ax, (month, intensity_dict) in zip(axes, intensity_data.items()):
        if intensity_dict:
            all_intensities = np.concatenate(list(intensity_dict.values()))
            for idx, (_, data) in enumerate(intensity_dict.items()):
                ax.hist(data, bins=50, alpha=0.3, color=colors[idx % len(colors)], density=False)
            ax.axvline(np.mean(all_intensities), color="red", linestyle="--", linewidth=2)
            ax.axvline(np.median(all_intensities), color="blue", linestyle="-.", linewidth=2)
            ax.set_title(f"Forecast Intensity Distribution, Init Month {month}")
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Frequency")
        else:
            ax.set_title(f"No Data: Month {month}")
    plt.tight_layout()
    plt.show()



#### forecast_skills_metrics ####

def plot_smr_line(ax, forecast_years, init_months, handler, index_metric):
    """
    Plot the Spread to Mean Ratio (SMR) of a climate index across forecast months.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to draw the plot on.
    forecast_years : list of int
        List of forecast years to include.
    init_months : list of str
        List of initialization months (e.g., ['03', '04']).
    handler : object
        Handler object that provides the `get_pipeline_path(...)` method.
    index_metric : str
        Name of the climate index (e.g., 'TX30').

    Returns
    -------
    None
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_idx = 0

    for year in forecast_years:
        for init_month in init_months:
            try:
                paths = handler.get_pipeline_path(year, init_month, "indices")
                monthly_path = paths.get("monthly")
                if not monthly_path or not monthly_path.exists():
                    print(f"Monthly file missing for {year}-init{init_month}.")
                    continue

                ds = xr.open_dataset(monthly_path)
                if index_metric not in ds:
                    print(f"Variable {index_metric} not found in {monthly_path}.")
                    continue

                da = ds[index_metric]
                steps = ds["step"].values

                smrs = []
                for step in steps:
                    group = da.sel(step=step)
                    std_map = group.std(dim="number")
                    mean_map = group.mean(dim="number")
                    smr_map = std_map / mean_map
                    smr_value = smr_map.mean(dim=["latitude", "longitude"])
                    smrs.append(smr_value.values)

                ax.plot(steps, smrs, marker='o', linestyle='-',
                        color=colors[color_idx % len(colors)], label=f'{year}-init{init_month}')
                color_idx += 1

            except Exception as e:
                print(f"Error for {year}-init{init_month}: {e}")

    ax.axhline(y=1.0, color='gray', linestyle='--', label='SMR Threshold')
    ax.set_xlabel('Forecast Month', fontsize=14)
    ax.set_ylabel('Spread to Mean Ratio (SMR)', fontsize=14)
    ax.set_title('Spread to Mean Ratio (SMR)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True)


def plot_iqr_line(ax, forecast_years, init_months, handler, index_metric):
    """
    Plot the Interquartile Range (IQR) of a climate index across forecast months.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to draw the plot on.
    forecast_years : list of int
        List of forecast years to include.
    init_months : list of str
        List of initialization months (e.g., ['03', '04']).
    handler : object
        Handler object that provides the `get_pipeline_path(...)` method.
    index_metric : str
        Name of the climate index (e.g., 'TX30').

    Returns
    -------
    None
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_idx = 0

    for year in forecast_years:
        for init_month in init_months:
            try:
                paths = handler.get_pipeline_path(year, init_month, "indices")
                monthly_path = paths.get("monthly")
                if not monthly_path or not monthly_path.exists():
                    print(f"Monthly file missing for {year}-init{init_month}.")
                    continue

                ds = xr.open_dataset(monthly_path)
                if index_metric not in ds:
                    print(f"Variable {index_metric} not found in {monthly_path}.")
                    continue

                da = ds[index_metric]
                steps = ds["step"].values

                iqrs = []
                for step in steps:
                    group = da.sel(step=step)
                    q75 = group.quantile(0.75, dim="number")
                    q25 = group.quantile(0.25, dim="number")
                    iqr = (q75 - q25).mean(dim=["latitude", "longitude"])
                    iqrs.append(iqr.values)

                ax.plot(steps, iqrs, marker='o', linestyle='-',
                        color=colors[color_idx % len(colors)], label=f'{year}-init{init_month}')
                color_idx += 1

            except Exception as e:
                print(f"Error for {year}-init{init_month}: {e}")

    ax.axhline(y=50, color='gray', linestyle='--', label='IQR Threshold')
    ax.set_xlabel('Forecast Month', fontsize=14)
    ax.set_ylabel('Interquartile Range (IQR)', fontsize=14)
    ax.set_title('Interquartile Range (IQR)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True)


def plot_ensemble_spread_line(ax, forecast_years, init_months, handler, index_metric):
    """
    Plot the ensemble standard deviation (spread) of a climate index across forecast months.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to draw the plot on.
    forecast_years : list of int
        List of forecast years to include.
    init_months : list of str
        List of initialization months (e.g., ['03', '04']).
    handler : object
        Handler object that provides the `get_pipeline_path(...)` method.
    index_metric : str
        Name of the climate index (e.g., 'TX30').

    Returns
    -------
    None
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_idx = 0

    for year in forecast_years:
        for init_month in init_months:
            try:
                paths = handler.get_pipeline_path(year, init_month, "indices")
                monthly_path = paths.get("monthly")
                if not monthly_path or not monthly_path.exists():
                    print(f"Monthly file missing for {year}-init{init_month}.")
                    continue

                ds = xr.open_dataset(monthly_path)
                if index_metric not in ds:
                    print(f"Variable {index_metric} not found in {monthly_path}.")
                    continue

                da = ds[index_metric]
                steps = ds["step"].values

                spreads = []
                for step in steps:
                    group = da.sel(step=step)
                    spread = group.std(dim="number").mean(dim=["latitude", "longitude"])
                    spreads.append(spread.values)

                ax.plot(steps, spreads, marker='o', linestyle='-',
                        color=colors[color_idx % len(colors)], label=f'{year}-init{init_month}')
                color_idx += 1

            except Exception as e:
                print(f"Error for {year}-init{init_month}: {e}")

    ax.axhline(y=100, color='gray', linestyle='--', label='Ensemble Spread Threshold')
    ax.set_xlabel('Forecast Month', fontsize=14)
    ax.set_ylabel('Ensemble Spread', fontsize=14)
    ax.set_title('Ensemble Spread', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True)


def plot_eai_line(ax, forecast_years, init_months, forecast_months, handler, index_metric, threshold):
    """
    Plot the Ensemble Agreement Index (EAI) for a climate index across forecast months.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to draw the plot on.
    forecast_years : list of int
        List of forecast years to include.
    init_months : list of str
        List of initialization months (e.g., ['03', '04']).
    forecast_months : list of str or None
        List of valid forecast months (not used internally, can be None).
    handler : object
        Handler object that provides the `get_pipeline_path(...)` method.
    index_metric : str
        Name of the climate index (e.g., 'TX30').
    threshold : float
        Threshold above which a grid cell is considered to have an "event".

    Returns
    -------
    None
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_idx = 0

    for year in forecast_years:
        for init_month in init_months:
            try:
                paths = handler.get_pipeline_path(year, init_month, "indices")

                monthly_path = paths.get("monthly")
                if not monthly_path or not monthly_path.exists():
                    print(f"Monthly file missing for {year}-init{init_month}.")
                    continue

                ds = xr.open_dataset(monthly_path)

                if index_metric not in ds:
                    print(f"Variable {index_metric} not found in {monthly_path}.")
                    continue

                above_thresh = ds[index_metric] > threshold
                eai = above_thresh.sum(dim="number") / above_thresh.sizes["number"]
                eai_avg = eai.mean(dim=["latitude", "longitude"])
                forecast_steps = ds["step"].values

                ax.plot(
                    forecast_steps,
                    eai_avg.values,
                    marker='o',
                    linestyle='-',
                    color=colors[color_idx % len(colors)],
                    label=f"{year}-init{init_month}"
                )
                color_idx += 1

            except Exception as e:
                print(f"Error for {year}-init{init_month}: {e}")

    ax.axhline(y=0.1, color='gray', linestyle='--', label='EAI Threshold')
    ax.set_xlabel("Forecast Month", fontsize=14)
    ax.set_ylabel("Ensemble Agreement Index (EAI)", fontsize=14)
    ax.set_title("Ensemble Agreement Index (EAI)", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True)
    
def forecast_skills_metrics(forecast_years, init_months, handler, index_metric, threshold):
    """
    Plot all ensemble forecast skill metrics (SMR, IQR, Spread, EAI) in a 2x2 grid.

    Parameters
    ----------
    forecast_years : list of int
        List of forecast years to include in the evaluation.
    init_months : list of str
        List of initialization months (e.g., ['03', '04']).
    handler : object
        Handler object that provides the `get_pipeline_path(...)` method.
    index_metric : str
        Name of the climate index (e.g., 'TX30').
    threshold : float
        Threshold used for computing the Ensemble Agreement Index (EAI).

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle("Ensemble Skill Metrics", fontsize=16)

    plot_smr_line(axes[0, 0], forecast_years, init_months, handler, index_metric)
    plot_iqr_line(axes[0, 1], forecast_years, init_months, handler, index_metric)
    plot_ensemble_spread_line(axes[1, 0], forecast_years, init_months, handler, index_metric)
    plot_eai_line(axes[1, 1], forecast_years, init_months, forecast_months=None, handler=handler, index_metric=index_metric, threshold=threshold)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.figtext(0.02, -0.05, "The first month in each plot represents the initialization. "
                             "Subsequent months are forecast-valid months based on the index 'step' coordinate.",
                wrap=True, fontsize=14)

    plt.figtext(0.02, -0.10, "SMR (Spread to Mean Ratio): Measures variability relative to the mean. "
                             "Higher SMR implies greater uncertainty. Threshold line: SMR = 1.",
                wrap=True, fontsize=14)

    plt.figtext(0.02, -0.15, "IQR (Interquartile Range): Difference between Q3 and Q1 across ensemble members. "
                             "Larger IQR signals higher spread. Threshold line: IQR = 50.",
                wrap=True, fontsize=14)

    plt.figtext(0.02, -0.20, "Ensemble Spread: Standard deviation among ensemble members. "
                             "High values indicate more disagreement. Threshold line: 100.",
                wrap=True, fontsize=14)

    plt.figtext(0.02, -0.25, "EAI (Ensemble Agreement Index): Proportion of members above threshold. "
                             "High EAI implies strong forecast agreement. Threshold line: 0.1.",
                wrap=True, fontsize=14)

    plt.show()