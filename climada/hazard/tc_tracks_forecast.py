"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define TCTracks auxiliary methods: BUFR based TC predictions (from ECMWF)
"""

__all__ = ['TCForecast']

# standard libraries
import datetime as dt
import fnmatch
import ftplib
import logging
import os
import tempfile

# additional libraries
import numpy as np
import pybufrkit
import tqdm
import xarray as xr

# climada dependencies
from climada.hazard.tc_tracks import TCTracks, set_category
from climada.util.files_handler import get_file_names

# declare constants
ECMWF_FTP = 'dissemination.ecmwf.int'
ECMWF_USER = 'wmo'
ECMWF_PASS = 'essential'

BASINS = {
    'W' : 'W - North West Pacific',
    'C' : 'C - North Central Pacific',
    'E' : 'E - North East Pacific',
    'P' : 'P - South Pacific',
    'L' : 'L - North Atlantic',
    'A' : 'A - Arabian Sea (North Indian Ocean)',
    'B' : 'B - Bay of Bengal (North Indian Ocean)',
    'U' : 'U - Australia',
    'S' : 'S - South-West Indian Ocean'
}
"""Gleaned from the ECMWF wiki at
https://confluence.ecmwf.int/display/FCST/Tropical+Cyclone+tracks+in+BUFR+-+including+genesis
and Wikipedia at https://en.wikipedia.org/wiki/Invest_(meteorology)
"""

SIG_CENTRE = 1
"""The 008005 significance for 'centre'"""

LOGGER = logging.getLogger(__name__)

class TCForecast(TCTracks):
    """An extension of the TCTracks construct adapted to forecast tracks
    obtained from numerical weather prediction runs.

    Attributes:
        data (list(xarray.Dataset)): Same as in parent class, adding the
            following attributes
                - ensemble_member (int)
                - is_ensemble (bool)
    """
    def fetch_ecmwf(self, path=None):
        """
        Fetch and read latest ECMWF TC track predictions from the FTP
        dissemination server into instance. Use path argument to use local files
        instead.

        Parameters:
            path (str, list(str)): A location in the filesystem. Either a
                path to a single BUFR TC track file, or a folder containing only
                such files, or a globbing pattern.
        """
        if path is None:
            files = self.fetch_bufr_ftp()
        else:
            files = get_file_names(path)

        for i, file in enumerate(files, 1):
            try:
                file.seek(0) # reset cursor if opened file instance
            except AttributeError:
                pass

            self.read_one_bufr_tc(file, id_no=i)

            try:
                file.close() # discard if tempfile
            except AttributeError:
                pass

    @staticmethod
    def fetch_bufr_ftp(target_dir=None):
        """
        Fetch and read latest ECMWF TC track predictions from the FTP dissemination
        server. If target_dir is set, the files get downloaded persistently to the
        given location. A list of opened file-like objects gets returned.

        Parameters:
            target_dir (str): An existing directory to write the files to. If None,
                the files get returned as tempfiles.
            close_files (bool): Should the returned files be closed?

        Returns:
            [str] or [filelike]
        """
        con = ftplib.FTP(host=ECMWF_FTP, user=ECMWF_USER, passwd=ECMWF_PASS)

        try:
            folders = con.nlst()
            folders.sort(reverse=True)
            con.cwd(folders[0]) # latest folder

            remotefiles = fnmatch.filter(con.nlst(), '*tropical_cyclone*')
            localfiles = []

            LOGGER.info('Fetching BUFR tracks:')
            for rfile in tqdm.tqdm(remotefiles, unit='files'):
                if target_dir:
                    lfile = open(os.path.join(target_dir, rfile), 'w+b')
                else:
                    lfile = tempfile.TemporaryFile(mode='w+b')

                con.retrbinary('RETR ' + rfile, lfile.write)

                if target_dir:
                    localfiles.append(lfile.name)
                    lfile.close()
                else:
                    localfiles.append(lfile)

        except ftplib.all_errors as err:
            con.quit()
            LOGGER.error('Error while downloading BUFR TC tracks.')
            raise err

        _ = con.quit()

        return localfiles

    def read_one_bufr_tc(self, file, id_no=None, fcast_rep=None):
        """ Read a single BUFR TC track file.

        Parameters:
            file (str, filelike): Path object, string, or file-like object
            id_no (int): Numerical ID; optional. Else use date + random int.
            fcast_rep (int): Of the form 1xx000, indicating the delayed
                replicator containing the forecast values; optional.
        """

        decoder = pybufrkit.decoder.Decoder()
        list_out = []

        if hasattr(file, 'read'):
            bufr = decoder.process(file.read())
        elif hasattr(file, 'read_bytes'):
            bufr = decoder.process(file.read_bytes())
        elif os.path.isfile(file):
            with open(file, 'rb') as i:
                bufr = decoder.process(i.read())
        else:
            raise FileNotFoundError('Check file argument')

        # setup parsers and querents
        npparser = pybufrkit.dataquery.NodePathParser()
        dquerent = pybufrkit.dataquery.DataQuerent(npparser)

        meparser = pybufrkit.mdquery.MetadataExprParser()
        mquerent = pybufrkit.mdquery.MetadataQuerent(meparser)

        if fcast_rep is None:
            fcast_rep = self._find_delayed_replicator(
                mquerent.query(bufr, '%unexpanded_descriptors')
            )

        # query the bufr message
        significance_dquery = dquerent.query(bufr, fcast_rep + '> 008005')
        latitude_dquery     = dquerent.query(bufr, fcast_rep + '> 005002')
        longitude_dquery    = dquerent.query(bufr, fcast_rep + '> 006002')
        wind_10m_dquery     = dquerent.query(bufr, fcast_rep + '> 011012')
        pressure_dquery     = dquerent.query(bufr, fcast_rep + '> 010051')
        timestamp_dquery    = dquerent.query(bufr, fcast_rep + '> 004024')
        wmo_longname_dquery = dquerent.query(bufr, '/001027')
        storm_id_dquery     = dquerent.query(bufr, '/001025')

        timestamp_origin = dt.datetime(
            mquerent.query(bufr, '%year'),
            mquerent.query(bufr, '%month'),
            mquerent.query(bufr, '%day'),
            mquerent.query(bufr, '%hour'),
            mquerent.query(bufr, '%minute'),
        )
        timestamp_origin = np.datetime64(timestamp_origin)

        if id_no is None:
            id_no = timestamp_origin.item().strftime('%Y%m%d%H') + \
                    str(np.random.randint(1e3, 1e4))

        orig_centre = mquerent.query(bufr, '%originating_centre')
        if orig_centre == 98:
            provider = 'ECMWF'
        else:
            provider = 'BUFR code ' + str(orig_centre)

        data_subcat = mquerent.query(bufr, '%data_i18n_subcategory')
        n_subsets = mquerent.query(bufr, '%n_subsets')

        if (data_subcat != 0) or (n_subsets == 1):
            is_ensemble = False
        elif (data_subcat == 0) and (n_subsets < 52):
            is_ensemble = True
        else:
            is_ensemble = None

        for i in significance_dquery.subset_indices():
            sig = np.array(significance_dquery.get_values(i), dtype='int')
            lat = np.array(latitude_dquery.get_values(i), dtype='float')
            lon = np.array(longitude_dquery.get_values(i), dtype='float')
            wnd = np.array(wind_10m_dquery.get_values(i), dtype='float')
            pre = np.array(pressure_dquery.get_values(i), dtype='float')

            name = wmo_longname_dquery.get_values(i)[0].decode().strip()
            sid = storm_id_dquery.get_values(i)[0].decode().strip()

            timestep_int = np.array(timestamp_dquery.get_values(i)).squeeze()
            timestamp = timestamp_origin + timestep_int.astype('timedelta64[h]')

            if is_ensemble is None and i < 52:
                is_ensemble = True
            elif is_ensemble is None and i == 52:
                is_ensemble = False
            # in the old format, subset 51 is the ENS control run (unperturbed),
            # see https://www.ecmwf.int/en/forecasts/datasets/set-iii#III-viii

            track = xr.Dataset(
                data_vars={
                    'max_sustained_wind': ('time', np.squeeze(wnd)),
                    'central_pressure': ('time', np.squeeze(pre)),
                    'ts_int': ('time', timestep_int),
                    'lat': ('time', lat[sig == 1]),
                    'lon': ('time', lon[sig == 1]),
                },
                coords={
                    'time': timestamp,
                },
                attrs={
                    'max_sustained_wind_unit': 'm/s',
                    'central_pressure_unit': 'Pa',
                    'name': name,
                    'sid': sid,
                    'orig_event_flag': False,
                    'data_provider': provider,
                    'id_no': (int(id_no) + i / 100),
                    'ensemble_number': i if is_ensemble else None,
                    'is_ensemble': is_ensemble,
                    'forecast_time': timestamp_origin,
                }
            )

            track = track.dropna('time')

            if track.sizes['time'] != 0:
                track = track.set_coords(['lat', 'lon'])
                track['time_step'] = track.ts_int - \
                    track.ts_int.shift({'time': 1}, fill_value=0)
                track = track.drop('ts_int') # TODO use drop_vars after upgrading xarray

                # according to specs always num-num-letter
                track.attrs['basin'] = BASINS[sid[2]]

                saffir_scale = np.array([18, 33, 43, 50, 59, 71, 1000]) # in m/s
                track.attrs['category'] = set_category(
                    max_sus_wind=track.max_sustained_wind.values,
                    wind_unit=track.max_sustained_wind_unit,
                    saffir_scale=saffir_scale
                )

                self.append(track)
            else:
                LOGGER.warning('Dropping empty track %s, subset %d', track.sid, i)

    @staticmethod
    def _find_delayed_replicator(descriptors):
        """The current bufr tc tracks only use one delayed replicator, enclosing
        all forecast values. This finds it.

        Parameters:
            bufr_message: An in-memory pybufrkit BUFR message
        """
        delayed_replicators = [
            d for d in descriptors
            if 100000 < d < 200000 and d % 1000 == 0
        ]

        if len(delayed_replicators) != 1:
            LOGGER.error('Could not find fcast_rep, please set manually.')
            raise ValueError('More than one delayed replicator in BUFR file')

        return str(delayed_replicators[0])
