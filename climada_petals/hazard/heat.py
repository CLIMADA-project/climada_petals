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

Define Heat class.
"""

__all__ = ['Heat']

import logging
import datetime as dt
import copy
from pathlib import Path
import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd

from climada.hazard.base import Hazard
from climada.hazard.centroids import Centroids
from climada.hazard.tag import Tag as TagHazard



LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'HT'
"""Hazard type acronym Heat"""


class Heat(Hazard):
    """Contains heat events
    
    Heat events comprise the challange that there is no clear definition of
    the exact variable in the literature. In line with several studies on 
    heat-related mortality, we rely on daily mean temperature in this module.
    This yields the advantage of a simple univariate variable which can later
    on still be summarized to weekly, seasonal, or annual impacts.
    
    In this module heat events are calculated using single model initial
    condition large ensemble (SMILE) climate data which yields the advantage
    that no extrapolation to extremes are equired. Several large ensemble runs
    are available at
    https://www.cesm.ucar.edu/projects/community-projects/MMLEA/
    The model is also compatible with normal CMIP/CORDEX/...
    climate data.
    
    Especially for heat-related mortality, the calculated impacts can be
    extremely sensitive to small biases. Hence, we strongly recommend to
    perform a bias correction. A small bias of 0.5 degree C which is very
    common in climate models might hide a whole signal from climate change.
    

    Attributes
    ----------
    
    """

    def __init__(self):
        """Empty constructor"""

        Hazard.__init__(self, HAZ_TYPE)
    
    @classmethod
    def from_SMILE(cls, filelist, lat, lon, temp_obs=None,
                t_start=None, t_end=None, temp_var='tas'):

        """ Wrapper to fill heat hazard with single model initial concition large
        ensemble data
        
        Parameters
        ----------
        filelist : list
            List of file paths to climate data (in .nc files)
        lat : np.array()
            Array of latitudes
        lon : np.array()
            Array of longitudes
        temp_obs : list of pd.dataframe()
            Timeseries of observational temperature data for bias correction.
            List needs to be in line with lat/lon arrays. Each pd.dataframe()
            must contain a column 'date' and 'temp'
        t_start : string (yyyy-mm-dd)
            Start date
        t_end : string (yyyy-mm-dd)
            End date
        temp_var : string
            Variable name of temperature varibale to be read from SMILE data
        
        
        Returns
        -------
        haz : Heat instance
        
        Raises
        ------
        NameError
    
            """

        if filelist is None:
            raise NameError('No filelist for climate data set')
        if lat is None:
            raise NameError('No lat set')
        if lon is None:
            raise NameError('No lon set')
        
        loc_lon = xr.DataArray(lon, dims="points")
        loc_lat = xr.DataArray(lat, dims="points")
        
        # read all ensemble members given in filelist_model
        print('Start with loading and slicing large ensemble members')
        if t_start or t_end is None:
            print('Full time range is returned')
        for i, file in enumerate(filelist):
            print(i)
            # load and slice ensemble member
            data_ESM = xr.open_dataset(file) # read file
            dat = data_ESM[temp_var].sel(lon=loc_lon, lat=loc_lat, method="nearest")
            # slice over time
            if t_start and t_end is not None:
                dat = dat.sel(time=slice(t_start, t_end))
            # concat nearest point to xarray
            if i==0:
                d = dat
            else:
                d =  xr.concat([d, dat], dim='ens')
        
        # do bias correction if temp_obs is given
        intensity = np.zeros([d.time.size*d.ens.size, loc_lon.size])
        if temp_obs is not None:
            print('Start with bias correction')
            for i, obs_data in enumerate(temp_obs):
                # get date range from observational data
                sdate = obs_data.date.min().strftime('%Y-%m-%d')
                edate = obs_data.date.max().strftime('%Y-%m-%d')
                obs_data.dropna(subset = ['temp'], inplace=True)
                
                # get model location data
                data_city = d.sel(points=i)
                data_ref = data_city.sel(time=slice(sdate, edate)).to_pandas()
                data_per = data_city.to_pandas()
                
                # do bias correction
                data_bc = bias_correct_ensemble(data_ref.T, obs_data.temp,
                                                data_per.T)
                intensity[:,i] = data_bc.to_numpy().ravel()
                
        else:
            print('No bias correction is performed.')
            for i in range(loc_lon.size):
                data_city = d.sel(points=i)
                data_per = data_city.to_pandas()
                intensity[:,i] = data_per.to_numpy().ravel()
                
        #!! continue coding here !!#
        
        haz = cls()
        haz.tag = TagHazard('Heat')
        
        cent = Centroids()
        haz.centroids = cent.from_lat_lon(lat, lon)
        haz.units = 'C'
        haz.event_id = np.arange(1, intensity.shape[0]+1).astype(int)
        haz.event_name = list(map(str, haz.event_id))
        haz.date = np.tile(data_per.columns.to_datetimeindex(), len(filelist))
        if len(filelist)>1:
            haz.ens_member = np.repeat(np.arange(0, len(filelist)),
                                       len(data_per.columns.to_datetimeindex()))
        haz.orig = np.zeros(len(haz.date), bool)

        haz.intensity = sp.sparse.csr_matrix(intensity)
        freq = 1/(data_per.columns.to_datetimeindex()[-1].year -
                  data_per.columns.to_datetimeindex()[0].year+1)
        haz.frequency = np.ones(haz.event_id.size)*freq
        haz.fraction = haz.intensity.copy()
        haz.fraction.data.fill(1.0)
        
        return haz

    
def bias_correct_time_series(dat_mod, dat_obs, dat_mod_all,
                                 minq=0.001, maxq=1.000, incq=0.001):
    """ Bias correction is performed using a quantile mapping approach.
    This code is a slightly simplified version of the method published by
    Rajczak et al. (2016). doi:10.1002/joc.4417 and is available in R under
    https://github.com/SvenKotlarski/qmCH2018 
        
    Wrapper function for bias correction. Calculates quantiles for mapping,
    estimates cumulative distribution function (CDF) of observational
    (dat_obs) and modelled (dat_mod) data to estimate quantile specific
    correction function. CDF of these two timeseries need to correspond to
    the same time period. The estimated correction function is then applied
    to dat_mod_all which is model output data the same model than dat_mod
    but can cover any time range.

    Parameters
    ----------
    dat_mod : pd.Series
        Model data series as reference for bias adjustment
    dat_obs : pd.Series
        Observational data series as ground truth
    dat_mod_all : pd.Series
        Data series to be bias adjusted
    minq : float
        Minimum quantile for correction function
    maxq: float
        Maximum quantile for correction function
    incq : float
        Quantile increment for correction function (bin size)

    Returns
    -------
    dat_mod_all_corrected : pd.Series
        bias corrected dat_mod_all

        """
    # define quantiles used for mapping
    q = np.arange(minq, maxq, incq)
    # (1) calc cdf of observational and modeled data
    cdf_obs = _calc_cdf(dat_obs, q)
    cdf_mod = _calc_cdf(dat_mod, q)

    # (2) estimate correction function
    cdf_dif = cdf_mod - cdf_obs

    # (3) perform quantile mapping to data
    dat_mod_all_corrected = _map_quantile(dat_mod_all, cdf_mod, cdf_dif, q)

    return dat_mod_all_corrected
    
def bias_correct_ensemble(dat_mod, dat_obs, dat_mod_all,
                          minq=0.001, maxq=1.000, incq=0.001):
    """ Wrapper function for bias correction of a large ensemble.
     Calculates quantiles for mapping, estimates cumulative distribution
     function (CDF) of observational (dat_obs) and modelled (dat_mod) data
     to estimate one single quantile specific correction function for the
     whole ensemble. CDF of the observational data and the ensemble
     DataFrame need to correspond to the same time period. The estimated
     correction function is then applied to dat_mod_all which is ensemble
     model output data of the same model than dat_mod but can cover
     any time range.
    
    Parameters
    ----------
    dat_mod : pd.DataFrame
        DataFrame with climate data from large ensemble. Needs to cover
        same range as dat_obs
    dat_obs : pd.Series
        Observational data series as ground truth. Needs to cover same
        range as dat_mod
    dat_mod_all : pd.DataFrame
        DataFrame to be bias adjusted
    minq : float
        Minimum quantile for correction function
    maxq: float
        Maximum quantile for correction function
    incq : float
        Quantile increment for correction function (bin size)
    
    Returns
    -------
    dat_mod_all_corrected : pd.DataFrame
        bias corrected dat_mod_all
    
        """
    # define quantiles used for mapping
    q = np.arange(minq, maxq, incq)
    # (1) calc cdf of observational and modeled data
    cdf_obs = _calc_cdf(dat_obs, q)
    cdf_mod = _calc_cdf_ens(dat_mod, q)
    
    # (2) estimate correction function
    cdf_dif = cdf_mod - cdf_obs
    
    # (3) perform quantile mapping to data
    dat_mod_corrected = _map_quantile_ens(dat_mod_all, cdf_mod, cdf_dif, q)
    
    return dat_mod_corrected

def _calc_cdf(data_series, q):
    """ Calculates cumulative distribution function (CDF) of any time series.
    Takes no assumption on distribution.

    Parameters
    ----------
    data_series : pd.Series
        Data series
    q : np.array
        quantiles of cdf to be calculated

    Returns
    -------
    cdf : np.array
        cdf of data_series on quantiles q

    """

    # sort data
    dat_sorted = np.sort(data_series.dropna().values)
    # calculate the proportional values of samples
    p = 1. * np.arange(len(dat_sorted)) / (len(dat_sorted) - 1)
    # map to percentiles
    cdf = np.interp(q, p, dat_sorted)

    return cdf

def _map_quantile(dat_mod_all, cdf_mod_orig, cdf_dif, q):
    """ Performs bias correction using quantile mapping
    
    Parameters
    ----------
    dat_mod_all : pd.Series
        Data series to be bias adjusted
    cdf_mod_orig : np.array
        original cdf of model used for bias correction
    cdf_dif : np.array
        cdf correction function
    q : np.array
        quantiles of cdf to be calculated
    
    Returns
    -------
    dat_mod_adj : pd.Series
        bias corrected data series
    
    """
    # calc percentile value of each temperature value in modelled time series
    perc_mod = np.interp(dat_mod_all, cdf_mod_orig, q)
    # correction term for each temperature value in modelled time series
    cor_term = np.interp(perc_mod, q, cdf_dif)
    # adjust for bias
    dat_mod_adj = dat_mod_all-cor_term
    
    return pd.Series(dat_mod_adj)

def _calc_cdf_ens(dat_mod, q):
    """ Calculates cumulative distribution function (CDF) an ensemble.
    Ensemble CDF is calculated as the mean over all CDF of the ensemble
    members.
    Takes no assumption on distribution.
    
    Parameters
    ----------
    dat_mod : pd.DataFrame
        DataFrame with climate data from large ensemble
    q : np.array
        quantiles of cdf to be calculated
    
    Returns
    -------
    cdf : np.array
        mean cdf over all ensemble members on quantiles q
    
    """
    
    # array to store cdfs
    cdf_array = np.zeros((dat_mod.shape[1], len(q)))
    for i in range(dat_mod.shape[1]):
        # calc cdf per member
        cdf_array[i,:] = _calc_cdf(dat_mod.iloc[:,i], q)
    
    # average cdf
    cdf_ens = np.nanmean(cdf_array, axis=0)
    
    return cdf_ens

def _map_quantile_ens(dat_mod_all, cdf_mod, cdf_dif, q):
    """ Performs bias correction for each ensemble member.
    
    Parameters
    ----------
    dat_mod_all : pd.DataFrame
        DataFrame to be bias adjusted
    cdf_mod_orig : np.array
        original cdf of model used for bias correction
    cdf_dif : np.array
        cdf correction function
    q : np.array
        quantiles of cdf to be calculated
    
    Returns
    -------
    dat_mod_adj : pd.DataFrame
        bias corrected data series
    
    """
    ens_array = np.zeros(dat_mod_all.shape)
    for i in range(dat_mod_all.shape[1]):
        ens_array[:,i] = _map_quantile(dat_mod_all.iloc[:,i], cdf_mod, cdf_dif, q)
    dat_mod_corrected = pd.DataFrame(ens_array)
    dat_mod_corrected.columns = dat_mod_all.columns
    
    return dat_mod_corrected
            