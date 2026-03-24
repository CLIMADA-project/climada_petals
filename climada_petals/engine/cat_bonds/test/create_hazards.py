import numpy as np
# import CLIMADA core modules
from climada.hazard import TCTracks, Centroids, TropCyclone
from climada.entity import LitPop

from climada.hazard import Centroids
from climada.util.constants import ONE_LAT_KM
# Fires in Portugal 2017
import os
import pandas as pd

from climada.util.constants import DEMO_DIR
from climada_petals.hazard import WildFire

def generate_spread_dates(nb_dates, seed=None, min_date=736694, max_date=809744):
    """
    Generate dates spread equally across the timeframe.
    
    Parameters:
    - nb_dates (int): Number of dates to generate.
    - seed (int): Random seed for reproducibility.
    - min_date (int): Minimum date (default: 736694).
    - max_date (int): Maximum date (default: 809744, representing 200 years).
    
    Returns:
    - np.array: Array of dates spread across the timeframe.
    """
    date_range = max_date - min_date
    interval = date_range / nb_dates
    dates = []
    np.random.seed(seed)
    for i in range(nb_dates):
        start = int(min_date + i * interval)
        end = int(min_date + (i + 1) * interval)
        dates.append(np.random.randint(start, end + 1))
    return np.array(sorted(dates))

# Define country-specific parameters
country_params = {
    659: {  # St. Kitts and Nevis
        'dates': generate_spread_dates(51, seed=12)  # Generate 51 dates spread across the timeframe with a fixed seed for reproducibility
    },
    388: {  # Jamaica
        'dates': generate_spread_dates(51, seed=24)  # Use a different seed for Jamaica to ensure different dates
    },
    84: {  # Belize
        'dates': generate_spread_dates(51, seed=42)  # Use a different seed for Belize to ensure different dates
    }
}

def create_tc_hazards_for_country(country_code, storm_id=None, bbox=None, file_name='IBTrACS.ALL.v04r01.nc', nb_synth_tracks=50, res=0.02):
    """
    Create hazards dataset for a given country.

    Parameters:
    - country_code (int): ISO3N country code.
    - storm_id (str): Storm ID for IBTrACS (required if country not in country_params).
    - bbox (tuple): Bounding box (min_lat, max_lat, min_lon, max_lon) (required if country not in country_params).
    - file_name (str): IBTrACS file name.
    - nb_synth_tracks (int): Number of synthetic tracks.
    - res (float): Resolution for centroids.

    Returns:
    - tc (TropCyclone): The tropical cyclone hazard object.
    - exp (LitPop): The exposure object.
    """
    # Check if country code is in country_params; if not, require storm_id and bbox
    if country_code in country_params:
        params = country_params[country_code]
        dates = params.get('dates')
    else:
        dates = generate_spread_dates(nb_synth_tracks+1)
    
    min_lat, max_lat, min_lon, max_lon = bbox

    # Import tropical cyclone tracks
    tr = TCTracks.from_ibtracs_netcdf(
        provider="usa", storm_id=storm_id, file_name=file_name
    )
    tr.equal_timestep()  # ensure equal timesteps
    tr.calc_perturbed_trajectories(nb_synth_tracks=nb_synth_tracks)  # create synthetic tracks

    # Create centroids
    cent = Centroids.from_pnt_bounds((min_lon, min_lat, max_lon, max_lat), res=res)

    # Construct tropical cyclones
    tc = TropCyclone.from_tracks(tr, centroids=cent)

    # Create exposure object
    exp = LitPop.from_countries(
        [country_code], fin_mode='gdp', reference_year=2020
    )

    exp.gdf.loc[exp.gdf.region_id == country_code, 'impf_TC'] = 1  # assign region id for vulnerability

    # Change dates of tc events to allow simulation of multiple years
    tc.date = dates

    return tc, exp


def create_wildfire_hazards_exposure_portugal():

    # load FIRMS data for 2016-2018 an select main land Portugal only
    firms_seasons = pd.read_csv(os.path.join(DEMO_DIR, "Portugal_firms_2016_17_18_MODIS.csv"))
    firms_seasons = firms_seasons[firms_seasons['latitude']>35.]
    firms_seasons = firms_seasons[firms_seasons['longitude']>-12.]


    wf_years = WildFire()
    wf_years.set_hist_fire_seasons_FIRMS(firms_seasons, centr_res_factor=1/1.) # we use MODIS data resolution (1 km)
    # generate 1 probabilistic fire season for Portugal
    wf_years.set_proba_fire_seasons(n_fire_seasons=3)

    exp = LitPop.from_countries([351], fin_mode='gdp', reference_year=2020) # Portugal country code
    exp.gdf.loc[exp.gdf.region_id == 351, 'impf_WF'] = 1  # assign region id for vulnerability

    return wf_years, exp