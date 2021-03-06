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

Define TCTracks auxiliary methods: BUFR based TC predictions (from ECMWF)
"""

__all__ = ['TCForecast']

# standard libraries
import os
import datetime as dt
import fnmatch
import ftplib
import logging
import tempfile
from pathlib import Path
import collections

# additional libraries
import numpy as np
import pandas as pd
import tqdm
import xarray as xr
import eccodes as ec

# climada dependencies
from climada.hazard.tc_tracks import (
    TCTracks, set_category, DEF_ENV_PRESSURE, CAT_NAMES
)
from climada.util.files_handler import get_file_names

# declare constants
ECMWF_FTP = 'dissemination.ecmwf.int'
ECMWF_USER = 'wmo'
ECMWF_PASS = 'essential'

BASINS = {
    'W': 'W - North West Pacific',
    'C': 'C - North Central Pacific',
    'E': 'E - North East Pacific',
    'P': 'P - South Pacific',
    'L': 'L - North Atlantic',
    'A': 'A - Arabian Sea (North Indian Ocean)',
    'B': 'B - Bay of Bengal (North Indian Ocean)',
    'U': 'U - Australia',
    'S': 'S - South-West Indian Ocean',
    'X': 'X - Undefined Basin'
}
"""Gleaned from the ECMWF wiki at
https://confluence.ecmwf.int/display/FCST/Tropical+Cyclone+tracks+in+BUFR+-+including+genesis
with added basin 'X' to deal with it appearing in operational forecasts
(see e.g. years 2020 and 2021 in the sidebar at https://www.ecmwf.int/en/forecasts/charts/tcyclone/)
and Wikipedia at https://en.wikipedia.org/wiki/Invest_(meteorology)

The BUFR code table is using EMO BUFR table version 35,
available at https://confluence.ecmwf.int/display/ECC/WMO%3D35+element+table?src=contextnavpagetreemode
"""

SAFFIR_MS_CAT = np.array([18, 33, 43, 50, 59, 71, 1000])
"""Saffir-Simpson Hurricane Categories in m/s"""

SIG_CENTRE = 1
"""The BUFR code 008005 significance for 'centre'"""

LOGGER = logging.getLogger(__name__)

MISSING_DOUBLE = ec.CODES_MISSING_DOUBLE
MISSING_LONG = ec.CODES_MISSING_LONG
"""Missing double and integers in ecCodes """

class TCForecast(TCTracks):
    """An extension of the TCTracks construct adapted to forecast tracks
    obtained from numerical weather prediction runs.

    Attributes
    ----------
    data : list of xarray.Dataset
        Same as in parent class, adding the following attributes
        - ensemble_member (int)
        - is_ensemble (bool; if False, the simulation is a high resolution deterministic run
    """

    def fetch_ecmwf(self, path=None, files=None, target_dir=None, remote_dir=None):
        """
        Fetch and read latest ECMWF TC track predictions from the FTP
        dissemination server into instance. Use path or files argument
        to use local files instead.

        Assumes file naming conventions consistent with ECMWF: all files
        are assumed to have 'tropical_cyclone' and 'ECEP' in their name,
        denoting tropical cyclone ensemble forecast files.

        Parameters
        ----------
        path : str, list(str), optional
            A location in the filesystem. Either a
            path to a single BUFR TC track file, or a folder containing
            only such files, or a globbing pattern. Passed to
            climada.util.files_handler.get_file_names
        files : file-like, optional
            An explicit list of file objects, bypassing
            get_file_names
        target_dir : str, optional
            An existing directory in the filesystem. When set, downloaded BUFR
            files will be saved here, otherwise they will be downloaded as
            temporary files.
        remote_dir : str, optional
            If set, search the ECMWF FTP folder for forecast files in the
            directory; otherwise defaults to the latest. Format:
            yyyymmddhhmmss, e.g. 20200730120000
        """
        if path is None and files is None:
            files = self.fetch_bufr_ftp(target_dir, remote_dir)
        elif files is None:
            files = get_file_names(path)
        elif not isinstance(files, list):
            files = [files]


        for i, file in tqdm.tqdm(enumerate(files, 1), desc='Processing',
                                 unit=' files', total=len(files)):
            # Open the bufr file if not already
            if isinstance(file, str) or isinstance(file, Path):
                file = open(file, 'rb')

            if os.name == 'nt' and hasattr(file, 'file'):
                file = file.file # if in windows try accessing the underlying tempfile directly incase variable file is tempfile._TemporaryFileWrapper

            self.read_one_bufr_tc(file, id_no=i)

            file.close()  # discards if tempfile

    @staticmethod
    def fetch_bufr_ftp(target_dir=None, remote_dir=None):
        """
        Fetch and read latest ECMWF TC track predictions from the FTP
        dissemination server. If target_dir is set, the files get downloaded
        persistently to the given location. A list of opened file-like objects
        gets returned.

        Parameters
        ----------
        target_dir : str
            An existing directory to write the files to. If
            None, the files get returned as tempfiles.
        remote_dir : str, optional
            If set, search this ftp folder for
            forecast files; defaults to the latest. Format:
            yyyymmddhhmmss, e.g. 20200730120000

        Returns
        -------
        [filelike]
        """
        con = ftplib.FTP(host=ECMWF_FTP, user=ECMWF_USER, passwd=ECMWF_PASS)

        try:
            if remote_dir is None:
                # Read list of directories on the FTP server
                remote = pd.Series(con.nlst())
                # Identify directories with forecasts initialised as 00 or 12 UTC
                remote = remote[remote.str.contains('120000|000000$')]
                # Select the most recent directory (names are formatted yyyymmddhhmmss)
                remote = remote.sort_values(ascending=False)
                remote_dir = remote.iloc[0]

            # Connect to the directory
            con.cwd(remote_dir)

            # Filter to files with 'tropical_cyclone' in the name: each file is a forecast ensemble for one event
            remotefiles_temp = fnmatch.filter(con.nlst(), '*tropical_cyclone*')
            # Filter to forecast ensemble files only
            remotefiles = fnmatch.filter(remotefiles_temp, '*ECEP*')

            if len(remotefiles) == 0:
                msg = 'No tracks found at ftp://{}/{}'
                msg.format(ECMWF_FTP, remote_dir)
                raise FileNotFoundError(msg)

            localfiles = []

            LOGGER.info('Fetching BUFR tracks:')

            for rfile in tqdm.tqdm(remotefiles, desc='Download', unit=' files'):
                if target_dir:
                    lfile = Path(target_dir, rfile).open('w+b')
                else:
                    lfile = tempfile.TemporaryFile(mode='w+b')

                con.retrbinary('RETR ' + rfile, lfile.write)
                lfile.seek(0)
                localfiles.append(lfile)

        except ftplib.all_errors as err:
            con.quit()
            raise type(err)('Error while downloading BUFR TC tracks: ' + str(err)) from err

        _ = con.quit()

        return localfiles

    def read_one_bufr_tc(self, file, id_no=None):
        """ Read a single BUFR TC track file tailored to the ECMWF TC track
        predictions format.

        Parameters:
            file (str, filelike): Path object, string, or file-like object
            id_no (int): Numerical ID; optional. Else use date + random int.
        """
        # Open the bufr file
        if isinstance(file, str) or isinstance(file, Path):
            # for the case that file is str, try open it
            file = open(file, 'rb')

        bufr = ec.codes_bufr_new_from_file(file)

        # we need to instruct ecCodes to expand all the descriptors
        # i.e. unpack the data values
        ec.codes_set(bufr, 'unpack', 1)

        # get the forecast time
        timestamp_origin = dt.datetime(
            ec.codes_get(bufr, 'year'), ec.codes_get(bufr, 'month'),
            ec.codes_get(bufr, 'day'), ec.codes_get(bufr, 'hour'),
            ec.codes_get(bufr, 'minute'),
        )
        timestamp_origin = np.datetime64(timestamp_origin)

        # get storm identifier
        sid = ec.codes_get(bufr, 'stormIdentifier').strip()

        # number of timesteps (size of the forecast time + initial analysis timestep)
        try:
            n_timestep = ec.codes_get_size(bufr, 'timePeriod') + 1
        except ec.CodesInternalError:
            LOGGER.warning("Track %s has no defined timePeriod. Track is discarded.", sid)
            return None

        # get number of ensemble members
        ens_no = ec.codes_get_array(bufr, "ensembleMemberNumber")
        n_ens = len(ens_no)

        # See documentation for link to ensemble types
        # Sometimes only one value is given instead of an array and it needs to be reproduced across all tracks
        ens_type = ec.codes_get_array(bufr, 'ensembleForecastType')
        if len(ens_type) == 1:
            ens_type = np.repeat(ens_type, n_ens)

        # values at timestep 0 (perturbed from the analysis for each ensemble member)
        lat_init_temp = ec.codes_get_array(bufr, '#2#latitude')
        lon_init_temp = ec.codes_get_array(bufr, '#2#longitude')
        pre_init_temp = ec.codes_get_array(bufr, '#1#pressureReducedToMeanSeaLevel')
        wnd_init_temp = ec.codes_get_array(bufr, '#1#windSpeedAt10M')

        # check dimension of the variables, and replace missing value with NaN
        lat_init = self._check_variable(lat_init_temp, n_ens, varname="Latitude at time 0")
        lon_init = self._check_variable(lon_init_temp, n_ens, varname="Longitude at time 0")
        pre_init = self._check_variable(pre_init_temp, n_ens, varname="Pressure at time 0")
        wnd_init = self._check_variable(wnd_init_temp, n_ens, varname="Maximum 10m wind at time 0")

        # Create dictionaries of lists to store output for each variable.
        # Each dict entry is an ensemble member, and it contains a list of forecast values by timestep
        latitude = {ind_ens: np.array(lat_init[ind_ens]) for ind_ens in range(n_ens)}
        longitude = {ind_ens: np.array(lon_init[ind_ens]) for ind_ens in range(n_ens)}
        pressure = {ind_ens: np.array(pre_init[ind_ens]) for ind_ens in range(n_ens)}
        max_wind = {ind_ens: np.array(wnd_init[ind_ens]) for ind_ens in range(n_ens)}

        # getting the forecasted storms
        timesteps_int = [0 for x in range(n_timestep)]
        for ind_timestep in range(1, n_timestep):
            rank1 = ind_timestep * 2 + 2  # rank for getting storm centre information
            rank3 = ind_timestep * 2 + 3  # rank for getting max wind information

            # Get timestep
            timestep = ec.codes_get_array(bufr, "#%d#timePeriod" % ind_timestep)
            timesteps_int[ind_timestep] = self.get_value_from_bufr_array(timestep)

            # Location of the storm: first check significance value matches what we expect
            sig_values = ec.codes_get_array(bufr, "#%d#meteorologicalAttributeSignificance" % rank1)
            significance = self.get_value_from_bufr_array(sig_values)

            # get lat, lon, and pressure of all ensemble members at ind_timestep
            if significance == 1:
                lat_temp = ec.codes_get_array(bufr, "#%d#latitude" % rank1)
                lon_temp = ec.codes_get_array(bufr, "#%d#longitude" % rank1)
                pre_temp = ec.codes_get_array(bufr, "#%d#pressureReducedToMeanSeaLevel" % (ind_timestep + 1))
            else:
                raise ValueError('unexpected meteorologicalAttributeSignificance=', significance)

            # Location of max wind: check significance value matches what we expect
            sig_values = ec.codes_get_array(bufr, "#%d#meteorologicalAttributeSignificance" % rank3)
            significanceWind = self.get_value_from_bufr_array(sig_values)

            # max_wind of each ensemble members at ind_timestep
            if significanceWind == 3:
                wnd_temp = ec.codes_get_array(bufr, "#%d#windSpeedAt10M" % (ind_timestep + 1))
            else:
                raise ValueError('unexpected meteorologicalAttributeSignificance=', significance)

            # check dimension of the variables, and replace missing value with NaN
            lat = self._check_variable(lat_temp, n_ens, varname="Latitude at time "+str(ind_timestep))
            lon = self._check_variable(lon_temp, n_ens, varname="Longitude at time "+str(ind_timestep))
            pre = self._check_variable(pre_temp, n_ens, varname="Pressure at time "+str(ind_timestep))
            wnd = self._check_variable(wnd_temp, n_ens, varname="Maximum 10m wind at time "+str(ind_timestep))

            # appending values into dictionaries
            for ind_ens in range(n_ens):
                latitude[ind_ens] = np.append(latitude[ind_ens], lat[ind_ens])
                longitude[ind_ens] = np.append(longitude[ind_ens], lon[ind_ens])
                pressure[ind_ens] = np.append(pressure[ind_ens], pre[ind_ens])
                max_wind[ind_ens] = np.append(max_wind[ind_ens], wnd[ind_ens])


        # storing information into a dictionary
        msg = {
            # subset forecast data
            'latitude': latitude,
            'longitude': longitude,
            'wind_10m': max_wind,
            'pressure': pressure,
            'timestamp': timesteps_int,

            # subset metadata
            'wmo_longname': ec.codes_get(bufr, 'longStormName').strip(),
            'storm_id': sid,
            'ens_type': ens_type,
            'ens_number': ens_no,
        }

        if id_no is None:
            id_no = timestamp_origin.item().strftime('%Y%m%d%H') + \
                    str(np.random.randint(1e3, 1e4))

        orig_centre = ec.codes_get(bufr, 'centre')
        if orig_centre == 98:
            provider = 'ECMWF'
        else:
            provider = 'BUFR code ' + str(orig_centre)

        for i in range(n_ens):
            name = msg['wmo_longname']
            track = self._subset_to_track(
                msg, i, provider, timestamp_origin, name, id_no
            )
            if track is not None:
                self.append(track)
            else:
                LOGGER.debug('Dropping empty track %s, subset %d', name, i)


    @staticmethod
    def get_value_from_bufr_array(var):
        if len(var) == 1:
            if var[0] == MISSING_LONG:
                ValueError("Array contained a single, missing value")
            return var[0]
        else:
            for i in range(len(var)):
                if var[i] != MISSING_LONG:
                    return var
                    break
            raise ValueError("Did not find a non-missing value in the array")


    @staticmethod
    def _subset_to_track(msg, index, provider, timestamp_origin, name, id_no):
        """Subroutine to process one BUFR subset into one xr.Dataset"""
        lat = np.array(msg['latitude'][index], dtype='float')
        lon = np.array(msg['longitude'][index], dtype='float')
        wnd = np.array(msg['wind_10m'][index], dtype='float')
        pre = np.array(msg['pressure'][index], dtype='float')

        sid = msg['storm_id'].strip()

        timestep_int = np.array(msg['timestamp']).squeeze()
        timestamp = timestamp_origin + timestep_int.astype('timedelta64[h]')

        # 'ens_type' can take a number of values telling us the source of the track.
        # 0 means the deterministic analysis, which we want to flag.
        # See documentation for link to ensemble types.
        ens_bool = msg['ens_type'][index] != 0

        try:
            track = xr.Dataset(
                data_vars={
                    'max_sustained_wind': ('time', np.squeeze(wnd)),
                    'central_pressure': ('time', np.squeeze(pre)/100),
                    'ts_int': ('time', timestep_int),
                    'lat': ('time', lat),
                    'lon': ('time', lon),
                },
                coords={
                    'time': timestamp,
                },
                attrs={
                    'max_sustained_wind_unit': 'm/s',
                    'central_pressure_unit': 'mb',
                    'name': name,
                    'sid': sid,
                    'orig_event_flag': False,
                    'data_provider': provider,
                    'id_no': (int(id_no) + index / 100),
                    'ensemble_number': msg['ens_number'][index],
                    'is_ensemble': ens_bool,
                    'forecast_time': timestamp_origin,
                }
            )
        except ValueError as err:
            LOGGER.warning(
                'Could not process track %s subset %d, error: %s',
                sid, index, err
                )
            return None

        track = track.dropna('time')

        if track.sizes['time'] == 0:
            return None

        # can only make latlon coords after dropna
        track = track.set_coords(['lat', 'lon'])
        track['time_step'] = track.ts_int - \
            track.ts_int.shift({'time': 1}, fill_value=0)

        track = track.drop_vars(['ts_int'])

        track['radius_max_wind'] = (('time'), np.full_like(
            track.time, np.nan, dtype=float)
        )
        track['environmental_pressure'] = (('time'), np.full_like(
            track.time, DEF_ENV_PRESSURE, dtype=float)
        )

        # according to specs always num-num-letter
        track['basin'] = ('time', np.full_like(track.time, BASINS[sid[2]], dtype=object))

        if sid[2] == 'X':
            LOGGER.info(
                'Undefined basin %s for track name %s ensemble no. %d',
                sid[2], track.attrs['name'], track.attrs['ensemble_number'])

        cat_name = CAT_NAMES[set_category(
            max_sus_wind=track.max_sustained_wind.values,
            wind_unit=track.max_sustained_wind_unit,
            saffir_scale=SAFFIR_MS_CAT
        )]
        track.attrs['category'] = cat_name
        return track


    @staticmethod
    def _check_variable(var, n_ens, varname=None):
        """Check the value and dimension of variable"""
        if len(var) == n_ens:
            var[var == MISSING_DOUBLE] = np.nan
            return var
        elif len(var) == 1 and var[0] == MISSING_DOUBLE:
            return np.repeat(np.nan, n_ens)
        elif len(var) == 1 and var[0] != MISSING_DOUBLE:
            LOGGER.warning('%s: only 1 variable value for %d ensemble members, duplicating value to all members. '
                           'This is only acceptable for lat and lon data at time 0.', varname, n_ens)
            return np.repeat(var[0], n_ens)

        else:
            raise ValueError
