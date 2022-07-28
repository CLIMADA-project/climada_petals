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
    that no extrapolation to extremes are required. Several large ensemble runs
    are available at
    https://www.cesm.ucar.edu/projects/community-projects/MMLEA/
    The model is also compatible with normal CMIP/CORDEX/... climate data.
    Alternatively, data can also be loaded from a pandas.DataFrame.

    Especially for heat-related mortality, the calculated impacts can be
    extremely sensitive to small biases. Hence, we strongly recommend to
    perform a bias correction. A small bias of 0.5 degree C is very
    common in climate models but might hide a whole climate change signal.


    Attributes
    ----------
    ens_member : np.array
        indicates SMILE ensemble member of each event

    """

    def __init__(self):
        """Empty constructor"""

        Hazard.__init__(self, HAZ_TYPE)

    @classmethod
    def from_SMILE(cls, filelist, lat, lon, temp_obs=None, t_start=None,
                   t_end=None, mmt=None, kind='heat', temp_var='tas'):

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
        MMT : np.array()
            Array of location specific temperature of minimum mortality
        kind : string
            'heat' or 'cold'
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

        haz = cls()

        loc_lon = xr.DataArray(lon, dims="points")
        loc_lat = xr.DataArray(lat, dims="points")

        # read all ensemble members given in filelist_model
        LOGGER.info('Start with loading large ensemble members')
        for i, file in enumerate(filelist):
            if np.mod(i, 10)==0:
                LOGGER.info('('+str(i)+'/'+str(len(filelist))+')')
            # load and slice ensemble member
            data_esm = xr.open_dataset(file) # read file
            dat = data_esm[temp_var].sel(lon=loc_lon, lat=loc_lat, method="nearest")
            # concat nearest point to xarray
            if i==0:
                d = dat
            else:
                d =  xr.concat([d, dat], dim='ens')

        # do bias correction if temp_obs is given
        if temp_obs is not None:
            LOGGER.info('Start with bias correction')
            for i, obs_data in enumerate(temp_obs):
                # get date range from observational data
                sdate = obs_data.date.min().strftime('%Y-%m-%d')
                edate = obs_data.date.max().strftime('%Y-%m-%d')
                obs_data.dropna(subset = ['temp'], inplace=True)

                # get model location data
                data_city = d.sel(points=i)
                data_ref = data_city.sel(time=slice(sdate, edate)).to_pandas()
                if t_start and t_end is not None:
                    data_per = data_city.sel(
                        time=slice(t_start, t_end)).to_pandas()
                else:
                    data_per = data_city.to_pandas()

                # do bias correction
                data_bc = haz.bias_correct_ensemble(data_ref.T, obs_data.temp,
                                                data_per.T)
                if i==0:
                    intensity = data_bc.to_numpy().ravel('F')
                else:
                    intensity = np.stack((intensity,
                                          data_bc.to_numpy().ravel('F')), axis=1)

        else:
            LOGGER.info('No bias correction is performed.')
            for i in range(loc_lon.size):
                data_city = d.sel(points=i)
                if t_start and t_end is not None:
                    data_per = data_city.sel(time=slice(t_start, t_end)).to_pandas()
                else:
                    data_per = data_city.to_pandas()
                if i==0:
                    intensity = data_per.to_numpy().ravel('C')
                else:
                    intensity = np.stack((intensity,
                                          data_per.to_numpy().ravel('C')), axis=1)

        # create hazard class
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
        freq = 1/((data_per.columns.to_datetimeindex()[-1].year -
                  data_per.columns.to_datetimeindex()[0].year+1)*len(filelist))
        haz.frequency = np.ones(haz.event_id.size)*freq
        haz.fraction = haz.intensity.copy()
        haz.fraction.data.fill(1.0)

        if mmt is not None:
            haz.set_mmt(mmt, kind)
        if np.mod(haz.intensity.shape[0],365) != 0:
            LOGGER.warning("No 365 day calender for hazard intensity."
                           "This might lead to wrong results in impacts.")

        return haz

    @classmethod
    def from_pandas(cls, pd_df, lat, lon, temp_obs=None, t_start=None,
                   t_end=None, mmt=None, kind='heat', temp_var='tas'):

        """ Wrapper to fill heat hazard with temperature data from pd.DataFrame

        Parameters
        ----------
        pd_df : pd.DataFrame()
            DataFrame with data. Must contain a first pd.Timestamp column
            'date'. Order of addtional columns must be in line with
            lat/lon arrays.
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
        MMT : np.array()
            Array of location specific temperature of minimum mortality
        kind : string
            'heat' or 'cold'
        temp_var : string
            Variable name of temperature varibale to be read from SMILE data


        Returns
        -------
        haz : Heat instance

        Raises
        ------
        NameError

            """

        if pd_df is None:
            raise NameError('No pd_df for temperature data')
        if lat is None:
            raise NameError('No lat set')
        if lon is None:
            raise NameError('No lon set')

        haz = cls()

        # do bias correction if temp_obs is given
        if temp_obs is not None:
            LOGGER.info('Start with bias correction')
            for i, obs_data in enumerate(temp_obs):
                # get date range from observational data
                sdate = obs_data.date.min().strftime('%Y-%m-%d')
                edate = obs_data.date.max().strftime('%Y-%m-%d')
                obs_data.dropna(subset = ['temp'], inplace=True)

                # get location data
                data_ref = pd_df.loc[(pd_df.date >= pd.Timestamp(sdate)) &
                                        (pd_df.date <= pd.Timestamp(edate))]
                data_ref = data_ref.iloc[:,i+1]
                if t_start and t_end is not None:
                    data_per = pd_df.loc[(pd_df.date >= pd.Timestamp(t_start)) &
                                            (pd_df.date <= pd.Timestamp(t_end))]
                    data_per = data_per.iloc[:,i+1]

                else:
                    data_per = pd_df.iloc[:,i+1]

                # do bias correction
                data_bc = haz.bias_correct_time_series(data_ref, obs_data.temp,
                                                data_per)
                if i==0:
                    intensity = data_bc.to_numpy(dtype=float)
                else:
                    intensity = np.stack((intensity,
                                          data_bc.to_numpy(dtype=float)),
                                         axis=1)

        else:
            LOGGER.info('No bias correction is performed.')
            if t_start and t_end is not None:
                data_per = pd_df.loc[(pd_df.date >= pd.Timestamp(t_start)) &
                                  (pd_df.date <= pd.Timestamp(t_end))]
            else:
                data_per = pd_df
            intensity = data_per.to_numpy(dtype=float)[:,1:]

        # create hazard class
        haz.tag = TagHazard('Heat')

        cent = Centroids()
        haz.centroids = cent.from_lat_lon(lat, lon)
        haz.units = 'C'
        haz.event_id = np.arange(1, intensity.shape[0]+1).astype(int)
        haz.event_name = list(map(str, haz.event_id))
        haz.date = data_per.date.values
        haz.orig = np.zeros(len(haz.date), bool)

        haz.intensity = sp.sparse.csr_matrix(intensity)
        freq = 1/((data_per.date.max().year - data_per.date.min().year+1))
        haz.frequency = np.ones(haz.event_id.size)*freq
        haz.fraction = haz.intensity.copy()
        haz.fraction.data.fill(1.0)

        if mmt is not None:
            haz.set_mmt(mmt, kind)
        if np.mod(haz.intensity.shape[0],365) != 0:
            LOGGER.warning("No 365 day calender for hazard intensity."
                           "This might lead to wrong results in impacts.")

        return haz

    def set_mmt(self, mmt, kind='heat'):
        """ Set temperature of minimum mortality. Temperatures below (above)
        the location specific MMT are set to zero for heat and cold related
        mortality respectively.

        Parameters
        ----------
        MMT : np.array()
            Array of location specific temperature of minimum mortality.
            Length needs to be in line with  numer of locations in the hazard.
        kind : string
            'heat' or 'cold' Variable name of temperature varibale to be read from SMILE data

        Returns
        -------
        sel : Heat instance

        Raises
        ------
        ValueError
        """

        if self.intensity.shape[1] != len(mmt):
            raise ValueError('MMT does not match number of locations in hazard')

        intensity = self.intensity.toarray()
        if kind=='heat':
            for i, temp in enumerate(mmt):
                intensity[:,i][intensity[:,i]<temp] = 0
        elif kind=='cold':
            for i, temp in enumerate(mmt):
                intensity[:,i][intensity[:,i]>temp] = 0
        else:
            raise ValueError('"kind" does not equal "heat" or "cold"')

        self.intensity = sp.sparse.csr_matrix(intensity)

    @classmethod
    def bias_correct_time_series(cls, dat_mod, dat_obs, dat_mod_all,
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
        qt = np.arange(minq, maxq, incq)
        # (1) calc cdf of observational and modeled data
        cdf_obs = cls._calc_cdf(dat_obs, qt)
        cdf_mod = cls._calc_cdf(dat_mod, qt)

        # (2) estimate correction function
        cdf_dif = cdf_mod - cdf_obs

        # (3) perform quantile mapping to data
        dat_mod_all_corrected = cls._map_quantile(dat_mod_all, cdf_mod,
                                                   cdf_dif, qt)

        return dat_mod_all_corrected

    @classmethod
    def bias_correct_ensemble(cls, dat_mod, dat_obs, dat_mod_all,
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
        qt = np.arange(minq, maxq, incq)
        # (1) calc cdf of observational and modeled data
        cdf_obs = cls._calc_cdf(dat_obs, qt)
        cdf_mod = cls._calc_cdf_ens(dat_mod, qt)

        # (2) estimate correction function
        cdf_dif = cdf_mod - cdf_obs

        # (3) perform quantile mapping to data
        dat_mod_corrected = cls._map_quantile_ens(dat_mod_all, cdf_mod, cdf_dif, qt)

        return dat_mod_corrected

    @staticmethod
    def _calc_cdf(data_series, qt):
        """ Calculates cumulative distribution function (CDF) of any time series.
        Takes no assumption on distribution.

        Parameters
        ----------
        data_series : pd.Series
            Data series
        qt : np.array
            quantiles of cdf to be calculated

        Returns
        -------
        cdf : np.array
            cdf of data_series on quantiles q

        """

        # sort data
        dat_sorted = np.sort(data_series.dropna().values)
        # calculate the proportional values of samples
        per = 1. * np.arange(len(dat_sorted)) / (len(dat_sorted) - 1)
        # map to percentiles
        cdf = np.interp(qt, per, dat_sorted)

        return cdf

    @staticmethod
    def _map_quantile(dat_mod_all, cdf_mod_orig, cdf_dif, qt):
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
        perc_mod = np.interp(dat_mod_all, cdf_mod_orig, qt)
        # correction term for each temperature value in modelled time series
        cor_term = np.interp(perc_mod, qt, cdf_dif)
        # adjust for bias
        dat_mod_adj = dat_mod_all-cor_term

        return pd.Series(dat_mod_adj)

    @classmethod
    def _calc_cdf_ens(cls, dat_mod, qt):
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
        cdf_array = np.zeros((dat_mod.shape[1], len(qt)))
        for i in range(dat_mod.shape[1]):
            # calc cdf per member
            cdf_array[i,:] = cls._calc_cdf(dat_mod.iloc[:,i], qt)

        # average cdf
        cdf_ens = np.nanmean(cdf_array, axis=0)

        return cdf_ens

    @classmethod
    def _map_quantile_ens(cls, dat_mod_all, cdf_mod, cdf_dif, qt):
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
            ens_array[:,i] = cls._map_quantile(dat_mod_all.iloc[:,i],
                                                cdf_mod, cdf_dif, qt)
        dat_mod_corrected = pd.DataFrame(ens_array)
        dat_mod_corrected.columns = dat_mod_all.columns

        return dat_mod_corrected
