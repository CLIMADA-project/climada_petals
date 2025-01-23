import numpy as np
def fahrenheit_to_kelvin(temp_f):
    """Convert temperature from Fahrenheit to Kelvin."""
    return (temp_f - 32) * 5/9 + 273.15

def celsius_to_kelvin(temp_c):
    """Convert temperature from Celsius to Kelvin."""
    return temp_c + 273.15


def kelvin_to_celsius(temp_k):
    """Convert temperature from Kelvin to Celsius."""
    return temp_k - 273.15

def kelvin_to_fahrenheit(temp_k):
    """Convert temperature from Kelvin to Fahrenheit."""
    return (temp_k - 273.15) * 9/5 + 32

def calculate_relative_humidity_percent(t2_k, td_k):
    """
    Relative Humidity in percent
        :param t2_k: (float array) 2m temperature [K]
        :param td_k: (float array) dew point temperature [K]
        returns relative humidity [%]
    """

    t2_c = kelvin_to_celsius(t2_k)
    td_c = kelvin_to_celsius(td_k)
    # saturated vapour pressure
    es = 6.11 * 10.0 ** (7.5 * t2_c / (237.3 + t2_c))
    # vapour pressure
    e = 6.11 * 10.0 ** (7.5 * td_c / (237.3 + td_c))
    rh = (e / es) * 100
    return rh

def calculate_heat_index_adjusted(t2_k, td_k):
    """
    Heat Index adjusted
       :param t2_k: (float array) 2m temperature [K]
       :param td_k: (float array) 2m dewpoint temperature  [K]
       returns heat index [K]
    Reference: https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
    """

    rh = calculate_relative_humidity_percent(t2_k, td_k)
    t2_f = kelvin_to_fahrenheit(t2_k)

    hiarray = [
        42.379,
        2.04901523,
        10.1433312,
        0.22475541,
        0.00683783,
        0.05481717,
        0.00122874,
        0.00085282,
        0.00000199,
    ]

    hi_initial = 0.5 * (t2_f + 61 + ((t2_f - 68) * 1.2) + (rh * 0.094))

    hi = (
        -hiarray[0]
        + hiarray[1] * t2_f
        + hiarray[2] * rh
        - hiarray[3] * t2_f * rh
        - hiarray[4] * t2_f**2
        - hiarray[5] * rh**2
        + hiarray[6] * t2_f**2 * rh
        + hiarray[7] * t2_f * rh**2
        - hiarray[8] * t2_f**2 * rh**2
    )

    hi_filter1 = np.where(t2_f > 80)
    hi_filter2 = np.where(t2_f < 112)
    hi_filter3 = np.where(rh <= 13)
    hi_filter4 = np.where(t2_f < 87)
    hi_filter5 = np.where(rh > 85)
    hi_filter6 = np.where(t2_f < 80)
    hi_filter7 = np.where((hi_initial + t2_f) / 2 < 80)

    f_adjust1 = hi_filter1 and hi_filter2 and hi_filter3
    f_adjust2 = hi_filter1 and hi_filter4 and hi_filter5

    adjustment1 = (
        (13 - rh[f_adjust1]) / 4 * np.sqrt(17 - np.abs(t2_f[f_adjust1] - 95) / 17)
    )

    adjustment2 = (rh[f_adjust2] - 85) / 10 * ((87 - t2_f[f_adjust2]) / 5)

    adjustment3 = 0.5 * (
        t2_f[hi_filter6]
        + 61.0
        + ((t2_f[hi_filter6] - 68.0) * 1.2)
        + (rh[hi_filter6] * 0.094)
    )

    hi[f_adjust1] = hi[f_adjust1] - adjustment1

    hi[f_adjust2] = hi[f_adjust2] + adjustment2

    hi[hi_filter6] = adjustment3

    hi[hi_filter7] = hi_initial[hi_filter7]

    hi_k = fahrenheit_to_kelvin(hi)

    return hi_k

# Define input data
t2m = np.array([310, 300, 277.2389906])  # Temperature (K)
td = np.array([280, 290, 273.1714606])   # Dew Point Temperature (K)

# Compute HIA directly
hia_result = calculate_heat_index_adjusted(t2m, td)

# Print results
print("Input Temperature (T2M)  :", t2m)
print("Input Dew Point (TD)     :", td)
print("Computed Heat Index Adj. :", hia_result)

# --------------------------------------
# Expected Output:
# --------------------------------------
# Input Temperature (T2M)  : [310.        300.        277.2389906]
# Input Dew Point (TD)     : [280.        290.        273.1714606]
# Computed Heat Index Adj. : [307.73725497 300.6820209  275.65534664]
# --------------------------------------

def calculate_heat_index_simplified(t2_k, rh):
    """
    Heat Index
        :param t2m: (float array) 2m temperature [K]
        :param rh: (float array) relative humidity [%]
        returns heat index [K]
    Reference: Blazejczyk et al. (2012)
    https://doi.org/10.1007/s00484-011-0453-2
    """

    t2_c = kelvin_to_celsius(t2_k)

    hiarray = [
        8.784695,
        1.61139411,
        2.338549,
        0.14611605,
        1.2308094e-2,
        1.6424828e-2,
        2.211732e-3,
        7.2546e-4,
        3.582e-6,
    ]
    hi = np.copy(t2_c)

    hi_filter1 = np.where(t2_c > 20)

    hi[hi_filter1] = (
        -hiarray[0]
        + hiarray[1] * t2_c[hi_filter1]
        + hiarray[2] * rh[hi_filter1]
        - hiarray[3] * t2_c[hi_filter1] * rh[hi_filter1]
        - hiarray[4] * t2_c[hi_filter1] ** 2
        - hiarray[5] * rh[hi_filter1] ** 2
        + hiarray[6] * t2_c[hi_filter1] ** 2 * rh[hi_filter1]
        + hiarray[7] * t2_c[hi_filter1] * rh[hi_filter1] ** 2
        - hiarray[8] * t2_c[hi_filter1] ** 2 * rh[hi_filter1] ** 2
    )

    hi_k = celsius_to_kelvin(hi)

    return hi_k

# Compute HIS directly
rh_calc = calculate_relative_humidity_percent(t2m, td)
his_result = calculate_heat_index_simplified(t2m, rh_calc)

# Print results
print("Input Temperature (T2M)  :", t2m)
print("Input Dew Point (TD)     :", td)
print("Computed Heat Index Sim. :", his_result)

# --------------------------------------
# Expected Output:
# --------------------------------------
# Input Temperature (T2M)  : [310.        300.        277.2389906]
# Input Dew Point (TD)     : [280.        290.        273.1714606]
# Computed Heat Index Sim. : [307.73725785 300.68203086 277.2389906 ]
# --------------------------------------
