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

Define the SupplyChain class.
"""

__all__ = ['SupplyChain']

import logging
import datetime as dt
from pathlib import Path
import zipfile
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import zipfile
import warnings

import pymrio
from pymrio.tools.iodownloader import download_exiobase3, download_oecd, download_wiod2013
from pymrio.tools.ioparser import parse_eora26, parse_exiobase1, parse_exiobase2, parse_exiobase3, parse_oecd, parse_wiod
from pymrio.tools.iomath import calc_A, calc_L, calc_x_from_L

from climada import CONFIG
from climada.util import files_handler as u_fh
import climada.util.coordinates as u_coord
from climada.engine import Impact
from climada.entity.exposures.base import Exposures

LOGGER = logging.getLogger(__name__)
WIOD_FILE_LINK = CONFIG.engine.supplychain.resources.wiod16.str()
"""Link to the 2016 release of the WIOD tables."""

MRIOT_DIRECTORY = CONFIG.engine.supplychain.local_data.mriot.dir()
"""Directory where Multi-Regional Input-Output Tables (MRIOT) are downloaded."""

mriot_funcs = {'EORA26': parse_eora26, 'EXIOBASE1': parse_exiobase1, 'EXIOBASE2': parse_exiobase2, 
                'EXIOBASE3': (download_exiobase3, parse_exiobase3), 'OECD': (download_oecd, parse_oecd), 
                'WIOD2013': (download_wiod2013, parse_wiod)}
calc_G = calc_L

def extract_mriot_data(mriot_df=None, col_iso3=None, col_sectors=None,
                       rows_data=None, cols_data=None):

    start_row, end_row = rows_data
    start_col, end_col = cols_data

    sectors = mriot_df.iloc[start_row:end_row, col_sectors].unique()
    regions = mriot_df.iloc[start_row:end_row, col_iso3].unique()
    Z_multiindex = pd.MultiIndex.from_product(
                [regions, sectors], names = ['region', 'sector'])

    Z = mriot_df.iloc[start_row:end_row, start_col:end_col].values.astype(float)
    Z = pd.DataFrame(data = Z,
                    index = Z_multiindex,
                    columns = Z_multiindex
                    )

    Y = mriot_df.iloc[start_row:end_row, end_col:-1].sum(1).values.astype(float)
    Y = pd.DataFrame(data = Y,
                    index = Z_multiindex,
                    columns = ['final demand']
                    )

    # total production
    x = mriot_df.iloc[start_row:end_row, -1].values.astype(float)
    x = pd.DataFrame(data = x,
                    index = Z_multiindex,
                    columns = ['total production']
                    )

    return Z, Y, x

def download_and_extract_wiod13(year, mriot_dir=MRIOT_DIRECTORY):
    # For the moment pymrio cannot download wiod, so we need an ad hoc function
    wiod_dir = mriot_dir / 'WIOD'

    if not wiod_dir.exists():
        wiod_dir.mkdir()

        LOGGER.info('Downloading folder with WIOD tables')

        downloaded_file_name = u_fh.download_file(WIOD_FILE_LINK, download_dir=wiod_dir)
        downloaded_file_zip_path = Path(downloaded_file_name + '.zip')
        Path(downloaded_file_name).rename(downloaded_file_zip_path)

        with zipfile.ZipFile(downloaded_file_zip_path, 'r') as zip_ref:
                zip_ref.extractall(wiod_dir)

    file_name = 'WIOT{}_Nov16_ROW.xlsb'.format(year)
    file_loc = wiod_dir / file_name
    mriot_df = pd.read_excel(file_loc, engine='pyxlsb')

    Z, Y, x = extract_mriot_data(mriot_df, col_iso3=2, col_sectors=1, 
                                rows_data=(5,2469), cols_data=(4,2468))

    return Z, Y, x

def calc_v(Z, x):
    """Calculate value added from the Z and x matrix

    value added (v) = industry output (x) - inter-industry inputs (sum_rows(Z))

    Parameters
    ----------
    Z : pandas.DataFrame or numpy.array
        Symmetric input output table (flows)
    x : pandas.DataFrame or numpy.array
        industry output

    Returns
    -------
    pandas.DataFrame or numpy.array
        Value added v as row vector
        The type is determined by the type of Z. If DataFrame index as Z

    """
    v = np.diff(np.vstack((Z.sum(0), x.T)), axis=0)
    if type(Z) is pd.DataFrame:
        v = pd.DataFrame(v, columns=Z.index, index=["indout"])
    if type(v) is pd.Series:
        v = pd.DataFrame(v)
    if type(v) is pd.DataFrame:
        v.index = ["indout"]
    return v

def calc_B(Z, x):
    """Calculate the B matrix (allocation coefficients matrix) from Z and x

    Parameters
    ----------
    Z : pandas.DataFrame or numpy.array
        Symmetric input output table (flows)
    x : pandas.DataFrame or numpy.array
        Industry output column vector

    Returns
    -------
    pandas.DataFrame or numpy.array
        Symmetric input output table (allocation matrix) B
        The type is determined by the type of Z.
        If DataFrame index/columns as Z

    Notes
    -----
    This function resembles the function "calc_A" in pymrio.tools.iomath which
    is to derive a technical coefficients matrix, A. Here "calc_A" is adapted 
    to compute the allocation coefficients matrix B.

    """
    if (type(x) is pd.DataFrame) or (type(x) is pd.Series):
        x = x.values
    if (type(x) is not np.ndarray) and (x == 0):
        recix = 0
    else:
        with warnings.catch_warnings():
            # catch the divide by zero warning
            # we deal wit that by setting to 0 afterwards
            warnings.simplefilter("ignore")
            recix = 1 / x
        recix[recix == np.inf] = 0
        recix = recix.reshape((-1, 1))

    if type(Z) is pd.DataFrame:
        return pd.DataFrame(Z.values * recix, index=Z.index, columns=Z.columns)
    else:
        return Z * recix

def calc_x_from_G(G, v):
    """Calculate the industry output x from a v vector and G 

    x = vG

    The industry output x is computed from a value-added vector v

    Parameters
    ----------
    v : pandas.DataFrame or numpy.array
        a row vector of the total final added-value
    G : pandas.DataFrame or numpy.array
        Symmetric input output Ghosh table

    Returns
    -------
    pandas.DataFrame or numpy.array
        Industry output x as column vector
        The type is determined by the type of G. If DataFrame index as G

    Notes
    -----
    This function resembles the function "calc_x_from_L" in pymrio.tools.iomath
    and it is adapted for the Ghosh case.

    """
    x = v.dot(G)
    if type(x) is pd.Series:
        x = pd.DataFrame(x)
    if type(x) is pd.DataFrame:
        x.columns = ["indout"]
    return x

class SupplyChain:
    """SupplyChain class.

    The SupplyChain class provides methods for loading Multi-Regional Input-Output
    Tables (MRIOT) and computing direct, indirect and total impacts.

    Attributes
    ----------
    mriot : IOSystem
        The input-output table data.
    direct_impact : pd.DataFrame
        Direct impact array.
    direct_aai_agg : pd.DataFrame
        Average annual direct impact array.
    indirect_impact : pd.DataFrame
        Indirect impact array.
    indirect_aai_agg : pd.DataFrame
        Average annual indirect impact array.
    total_impact : pd.DataFrame
        Total impact array.
    total_aai_agg : pd.DataFrame
        Average annual total impact array.
    """

    def __init__(self,
                mriot = None,
                inverse = None,
                coefficients = None,
                direct_imp_mat = None,
                direct_impt_aai_agg = None,
                indirect_imp_mat = None,
                indirect_impt_aai_agg = None,
                total_imp_mat = None,
                total_impt_aai_agg = None
                ):

        """Initialize SupplyChain."""
        self.mriot = pymrio.IOSystem() if mriot is None else mriot
        self.inverse = pd.DataFrame([]) if inverse is None else inverse
        self.coefficients = pd.DataFrame([]) if coefficients is None else coefficients
        self.direct_imp_mat = pd.DataFrame([]) if direct_imp_mat is None else direct_imp_mat
        self.direct_impt_aai_agg = pd.DataFrame([]) if direct_impt_aai_agg is None else direct_impt_aai_agg
        self.indirect_imp_mat = pd.DataFrame([]) if indirect_imp_mat is None else indirect_imp_mat
        self.indirect_impt_aai_agg = pd.DataFrame([]) if indirect_impt_aai_agg is None else indirect_impt_aai_agg
        self.total_imp_mat = pd.DataFrame([]) if total_imp_mat is None else total_imp_mat
        self.total_impt_aai_agg = pd.DataFrame([]) if total_impt_aai_agg is None else total_impt_aai_agg

    @classmethod
    def read_mriot(cls, mriot_type, mriot_year, mriot_dir=MRIOT_DIRECTORY, file_name=None):
        """Read multi-regional input-output tables
        Describe all available types. EXIOBASE1, EXIOBASE2 and EORA26 needs a registration prior to download, 
        they can be downloaded at .. .. and loaded as user provided IO tables
    
        year : int
        mriot_type : str
        mriot_dir : path it contains the downloaded mriot data either automatically or manually 
        file_name : path path to zip file
        eora26, exiobase1 and 2 need to be downloaded manually. Please make sure they are stored under 
        ./mriot_dir/mriot_type/year.
        'EORA26', 'EXIOBASE1', 'EXIOBASE2', 'EXIOBASE3', 'OECD', 'WIOD2013', 'USER'
    """

        data_dir = mriot_dir / mriot_type / str(mriot_year)
        parsed_data_dir = data_dir / 'pymrio_parsed_data'

        # TODO. TEST 'EORA26', 'EXIOBASE1', 'EXIOBASE2' 'OECD'
        if mriot_type in ['EORA26', 'EXIOBASE1', 'EXIOBASE2']:
            parser = mriot_funcs[mriot_type]

            if not parsed_data_dir.exists():
                if file_name is None:
                    raise ValueError('Missing name of the downloaded mriot zip folder')

                file_path = data_dir / file_name
                mriot = parser(path=file_path)

                # save only the system and not the extension (for now)
                mriot.save(parsed_data_dir)

            else:
                mriot = pymrio.load(path=parsed_data_dir)

        elif mriot_type == 'EXIOBASE3':
            downloader, parser = mriot_funcs[mriot_type]

            if not parsed_data_dir.exists():
                # EXIOBASE3 gets a system argument. This can be ixi (ind x ind matrix) or pxp (prod x prod matrix).
                # By default both are downloaded, we here use only ixi for the time being.
                mriot_meta = downloader(storage_folder=data_dir, system="ixi", years=[mriot_year])
                file_name = str(mriot_meta.history[0]).split(' ')[-1]
                file_path = data_dir / file_name
                mriot = parser(path=file_path)

                # save only the system and not the extension
                mriot.save(parsed_data_dir)

            else:
                mriot = pymrio.load(path=parsed_data_dir)

        elif mriot_type == 'WIOD':
            Z, Y, x = download_and_extract_wiod13(mriot_year)

            mriot = pymrio.IOSystem(Z=Z, Y=Y, x=x)

        # # TODO. It seems OECD_CONFIG in pymrio is outdated and so this crashes too. Check if 
        # # parsing of manually downloaded files work.
        # elif mriot_type == 'OECD':
        #     downloader, parser = mriot_funcs[mriot_type]
        #     if not parsed_data_dir.exists():
        #         # OECD has different versions, 2016, 2018, 2021. pymrio default is v2021 but it treats years differently 
        #         # so here v2018 is passed. 
        #         mriot_meta = downloader(storage_folder=data_dir, version = "v2018"), years=year)
        #         # to check file name
        #         file_name = str(mriot_meta.history[0]).split(' ')[-1]
        #         file_path = data_dir / file_name
        #         mriot = parser(path=file_path)
        #         # save only the system and not the extension (for now)
        #         mriot.save(parsed_data_dir)
        #     else:
        #         mriot = pymrio.load(path=parsed_data_dir)

        mriot.meta.change_meta('description', 'Metadata for pymrio Multi Regional Input-Output Table')
        mriot.meta.change_meta('name', f'{mriot_type}-{mriot_year}')
        mriot.unit = 'M.EUR'

        return cls(mriot=mriot)

    def calc_all_impacts(self, hazard, exposure, imp_fun_set, impacted_secs, io_approach):
        """Calculate direct, indirect and total impacts."""

        self.calc_direct_imp_mat(hazard, exposure, imp_fun_set, impacted_secs)
        self.calc_indirect_imp_mat(io_approach)
        self.calc_total_imp_mat()

    def calc_direct_imp_mat(self, hazard, exposure, imp_fun_set, impacted_secs=None):
        """Calculate direct impacts.

        Parameters
        ----------
        hazard : Hazard
            Hazard object for impact calculation.
        exposure : Exposures
            Exposures object for impact calculation. For WIOD tables, exposure.region_id
            must be country names following ISO3 codes.
        imp_fun_set : ImpactFuncSet
            Set of impact functions.
        selected_subsec : str or list
            Positions of the selected sectors. These positions can be either
            defined by the user by passing a list of values, or by using built-in
            sectors' aggregations for the WIOD data passing a string with possible
            values being "service", "manufacturing", "agriculture" or "mining".
            Default is "service".

        """
        self.frequency = hazard.frequency
        self.event_id = hazard.event_id

        if impacted_secs is None:
            warnings.warn("No impacted sectors were specified. It is assumed that the exposure is representative of all sectors in the IO table")
            impacted_secs = self.mriot.get_sectors().tolist()

        if isinstance(impacted_secs, (range, np.ndarray)):
            impacted_secs = self.mriot.get_sectors()[impacted_secs].tolist()

        self.direct_imp_mat = pd.DataFrame(0, index = self.event_id, columns = self.mriot.Z.columns)

        for exp_regid in exposure.gdf.region_id.unique():
            reg_exp = Exposures(exposure.gdf[exposure.gdf.region_id == exp_regid])
            reg_exp.check()

            # Normalize exposure
            total_reg_value = reg_exp.gdf['value'].sum()
            reg_exp.gdf['value'] /= total_reg_value

            # Calc normalized impact for country
            imp = Impact()
            imp.calc(reg_exp, imp_fun_set, hazard)

            # Extend normalized impact to all subsectors
            imp_sec_event = np.repeat(imp.at_event, len(impacted_secs)
                                    ).reshape(len(hazard.event_id), len(impacted_secs))

            mriot_type = self.mriot.meta.name.split('-')[0]
            mriot_reg_name = self.map_exp_to_mriot(exp_regid, mriot_type)

            # Calculate actual impact multiplying the norm impact by each subsector's total production
            prod_impacted_secs = self.mriot.x.loc[(mriot_reg_name, 
                                                   impacted_secs), :].values.flatten()*self.conv_fac()

            # Sum needed below in case of many ROWs, which are aggregated into
            # one country as per WIOD table.
            self.direct_imp_mat.loc[:, (mriot_reg_name,
                                       impacted_secs)] += np.multiply(imp_sec_event, prod_impacted_secs)

        self.direct_impt_aai_agg = self.direct_imp_mat.T.dot(self.frequency)

    def calc_indirect_imp_mat(self, io_approach):
        """Calculate indirect impacts according to the specified input-output
        appraoch. This function needs to be run after calc_sector_direct_impact.

        Parameters
        ----------
        io_approach : str
            The adopted input-output modeling approach. Possible approaches
            are 'leontief', 'ghosh' and 'eeioa'. Default is 'gosh'.

        References
        ----------
        [1] W. W. Leontief, Output, employment, consumption, and investment,
        The Quarterly Journal of Economics 58, 1944.
        [2] Ghosh, A., Input-Output Approach in an Allocation System,
        Economica, New Series, 25, no. 97: 58-64. doi:10.2307/2550694, 1958.
        [3] Kitzes, J., An Introduction to Environmentally-Extended Input-Output
        Analysis, Resources, 2, 489-503; doi:10.3390/resources2040489, 2013.
        """

        self.calc_matrixes(io_approach=io_approach)
        direct_intensity = self.direct_imp_mat.divide(self.mriot.x.values.flatten()).fillna(0)

        if io_approach == 'leontief':
            degr_demand = direct_intensity*self.mriot.Y.values.flatten()

            self.indirect_imp_mat = pd.concat(
                                            [calc_x_from_L(self.inverse, degr_demand.iloc[i]) 
                                            for i in range(len(self.event_id))],
                                            axis=1).T.set_index(self.event_id)

        elif io_approach == 'ghosh':
            value_added = calc_v(self.mriot.Z, self.mriot.x)
            degr_value_added = direct_intensity*value_added.values

            self.indirect_imp_mat = pd.concat(
                                            [calc_x_from_G(self.inverse, degr_value_added.iloc[i]) 
                                            for i in range(len(self.event_id))],
                                            axis=1).T.set_index(self.event_id)

        elif io_approach == 'eeioa':
            self.indirect_imp_mat = pd.DataFrame(
                direct_intensity.dot(self.inverse) * self.mriot.x.values.flatten()
                )

        self.indirect_impt_aai_agg = self.indirect_imp_mat.T.dot(self.frequency)

    def calc_total_imp_mat(self):
        """Calculate total impacts summing direct and indirect impacts."""

        self.total_imp_mat = self.direct_imp_mat.add(self.indirect_imp_mat)
        self.total_impt_aai_agg = self.total_imp_mat.T.dot(self.frequency)

    def calc_matrixes(self, io_approach):
        io_model = {'leontief': (calc_A, calc_L),
                    'eeioa': (calc_A, calc_L),
                    'ghosh': (calc_B, calc_G)}

        coeff_func, inv_func = io_model[io_approach]

        self.coefficients = coeff_func(self.mriot.Z, self.mriot.x)
        self.inverse = inv_func(self.coefficients)

    def conv_fac(self):
        if self.mriot.unit == 'M.EUR':
            conv_factor = 1e6
        else:
            conv_factor = 1
            warnings.warn("No known unit has been found. It is assumed values do not need a conversion")
        return conv_factor

    def map_exp_to_mriot(self, exp_regid, mriot_type):
        """
        Map regions names in exposure into Input-output regions names.
        exp_regid must be according to ISO 3166 numeric country codes.
        """

        if mriot_type == 'WIOD':
            mriot_reg_name = u_coord.country_to_iso(exp_regid, "alpha3")

            idx_country = np.where(self.mriot.get_regions() == mriot_reg_name)[0]

            if not idx_country.size > 0.:
                mriot_reg_name = 'ROW'

        elif mriot_type == 'EXIOBASE3':
            mriot_reg_name = u_coord.country_to_iso(exp_regid, "alpha2")
            idx_country = np.where(self.mriot.get_regions() == mriot_reg_name)[0]

            if not idx_country.size > 0.:
                # EXIOBASE3 in fact contains five ROW regions, 
                # but for now they are all catagorised as ROW.
                mriot_reg_name = 'ROW'

        # default name in meta for pymrio IOSystem
        elif mriot_type == 'IO':
            warnings.warn("No mriot_type was specified and no name conversion is applied. Formats of regions' names in exposure and IO table must be the same")
            mriot_reg_name = exp_regid

        return mriot_reg_name