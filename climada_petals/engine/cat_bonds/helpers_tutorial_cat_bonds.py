import numpy as np
from datetime import date
# import CLIMADA core modules
from climada.entity import ImpactFuncSet, ImpactFunc


def create_earthquake_mmi_impact_function(
    impf_id=1,
    haz_type="EQ",
):
    """
    Create a stylized earthquake impact function for MMI-based hazards.

    The curve uses separate smooth transitions for the affected asset share
    (PAA) and the mean damage degree conditional on being affected (MDD).

    Parameters
    ----------
    impf_id : int, optional
        Impact function identifier.
    haz_type : str, optional
        Hazard type associated with the impact function.

    Returns
    -------
    ImpactFuncSet
        Impact function set containing a single MMI-based impact function.
    """
    mmi_values = np.linspace(0.0, 12.0, 121)

    # Share of assets affected rises meaningfully only from moderate shaking on.
    paa_mid = 6.5
    paa_steepness = 1.1
    paa_values = 1.0 / (1.0 + np.exp(-paa_steepness * (mmi_values - paa_mid)))

    # Conditional damage increases later and is capped below full destruction
    # for most of the MMI range seen in the Jamaica hazard.
    mdd_mid = 8.0
    mdd_steepness = 0.9
    mdd_cap = 0.85
    mdd_values = mdd_cap / (1.0 + np.exp(-mdd_steepness * (mmi_values - mdd_mid)))

    # Ensure the impact curve passes exactly through the origin so CLIMADA does
    # not need to evaluate a non-zero mean damage ratio at zero intensity.
    paa_values[0] = 0.0
    mdd_values[0] = 0.0

    impf_eq = ImpactFunc(
        id=impf_id,
        haz_type=haz_type,
        name="Earthquake MMI sigmoidal",
        intensity_unit="MMI",
        intensity=mmi_values,
        mdd=mdd_values,
        paa=paa_values,
    )
    return ImpactFuncSet([impf_eq])

def remap_event_years_even(hazard, start_year=1980, n_years=1000):
    """
    This function is intended for tropical cyclone data sets based on STORM which can be downloaded from the CLIMADA data API. 
    In these data sets, all events are assigned the same year which is not suitable for CAT bond simulations that rely on the distribution of events over time.
    This function remaps the event years to be evenly distributed over a specified number of years while preserving the original month/day seasonality of the events.
    """
    ords = np.asarray(hazard.date, dtype=int)

    # Extract month/day and day-of-year from original ordinal dates
    md = []
    doy = []
    for o in ords:
        d = date.fromordinal(int(o))
        md.append((d.month, d.day))
        doy.append(d.timetuple().tm_yday)

    doy = np.asarray(doy)
    idx_sorted = np.argsort(doy, kind="stable")
    ranks = np.empty_like(idx_sorted)
    ranks[idx_sorted] = np.arange(len(ords))

    # Evenly assign years by rank
    years = start_year + (ranks * n_years) // len(ords)

    # Rebuild ordinal dates with new years, same month/day
    new_ords = np.empty_like(ords)
    for i, (m, d) in enumerate(md):
        y = int(years[i])
        # Handle Feb-29 if target year is non-leap
        if m == 2 and d == 29:
            try:
                new_ords[i] = date(y, 2, 29).toordinal()
            except ValueError:
                new_ords[i] = date(y, 2, 28).toordinal()
        else:
            new_ords[i] = date(y, m, d).toordinal()

    hazard.date = new_ords
    return hazard