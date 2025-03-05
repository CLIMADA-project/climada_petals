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

__all__ = ["SupplyChain"]

import logging
from pathlib import Path
import zipfile
import warnings
import pandas as pd
import numpy as np

import pymrio

from boario.extended_models import ARIOPsiModel
from boario.event import EventKapitalRecover, EventKapitalRebuild
from boario.simulation import Simulation

from climada import CONFIG
from climada.util import files_handler as u_fh
import climada.util.coordinates as u_coord

LOGGER = logging.getLogger(__name__)
WIOD_FILE_LINK = CONFIG.engine.supplychain.resources.wiod16.str()
"""Link to the 2016 release of the WIOD tables."""

MRIOT_DIRECTORY = CONFIG.engine.supplychain.local_data.mriot.dir()
"""Directory where Multi-Regional Input-Output Tables (MRIOT) are downloaded."""

VA_NAME = "value added"
"""Index name for value added"""

def calc_va(Z, x):
    """Calculate value added (v) from Z and x

    value added = industry output (x) - inter-industry inputs (sum_rows(Z))

    Parameters
    ----------
    Z : pandas.DataFrame or numpy.array
        Symmetric multi-regional input output table (flows)
    x : pandas.DataFrame or numpy.array
        industry output

    Returns
    -------
    pandas.DataFrame or numpy.array
        Value added va as row vector

    Notes
    -----
    This function adapts pymrio.tools.iomath.calc_x to compute
    value added (va).
    """

    value_added = np.diff(np.vstack((Z.sum(0), x.T)), axis=0)
    if isinstance(Z, pd.DataFrame):
        value_added = pd.DataFrame(value_added, columns=Z.index, index=[VA_NAME])
    if isinstance(value_added, np.ndarray):
        value_added = pd.DataFrame(value_added, index=[VA_NAME])
    return value_added

def calc_B(Z, x):
    """Calculate the B matrix (allocation coefficients matrix)
    from Z matrix and x vector

    Parameters
    ----------
    Z : pandas.DataFrame or numpy.array
        Symmetric multi-regional input output table (flows)
    x : pandas.DataFrame, pandas.Series, or numpy.array
        Industry output column vector

    Returns
    -------
    pandas.DataFrame or numpy.array
        Allocation coefficients matrix B.
        Same type as input parameter ``Z``.

    Notes
    -----
    This function adapts pymrio.tools.iomath.calc_A to compute
    the allocation coefficients matrix B.
    """
    # Convert x to a NumPy array
    x = np.asarray(x)

    # Handle zero values in x
    with np.errstate(divide='ignore'):
        recix = np.where(x == 0, 0, 1 / x).reshape((1, -1))

    # Calculate B matrix
    if isinstance(Z, pd.DataFrame):
        return pd.DataFrame(np.transpose(Z.values) * recix, index=Z.index, columns=Z.columns)
    else:
        return np.transpose(Z) * recix


def calc_G(B):
    """Calculate the Ghosh inverse matrix G either from B
    G = inverse matrix of (I - B) where I is an identity matrix of same shape as B.
    Note that we define G as the transpose of the Ghosh inverse matrix, so that we can apply the factors of
    production intensities from the left-hand-side for both Leontief and Ghosh attribution. In this way the
    multipliers have the same (vector) dimensions and can be added.
    Parameters
    ----------
    B : pandas.DataFrame or numpy.array
        Symmetric input output table (coefficients)
    Returns
    -------
    pandas.DataFrame or numpy.array
        Ghosh input output table G
        The type is determined by the type of B.
        If DataFrame index/columns as B
    """
    I = np.eye(B.shape[0])
    if isinstance(B, pd.DataFrame):
        return pd.DataFrame(
            np.linalg.inv(I - B), index=B.index, columns=B.columns
        )
    else:
        return np.linalg.inv(I - B)  # G = inverse matrix of (I - B)

def calc_x_from_G(G, va):
    """Calculate the industry output x from a v vector and G matrix

    x = Gva

    The industry output x is computed from a value-added vector v

    Parameters
    ----------
    v : pandas.DataFrame or numpy.array
        a row vector of the total final added-value
    G : pandas.DataFrame or numpy.array
        **Transpose** of Ghosh inverse matrix

    Returns
    -------
    pandas.DataFrame or numpy.array
        Industry output x as column vector.
        Same type as input parameter ``G``.

    Notes
    -----
    This function adapts the function pymrio.tools.iomath.calc_x_from_L to
    compute total output (x) from the Ghosh inverse.
    """

    x = G.dot(va.T)
    if isinstance(x, pd.Series):
        x = pd.DataFrame(x)
    if isinstance(x, pd.DataFrame):
        x.columns = ["indout"]
    return x


def parse_mriot_from_df(
        mriot_df, col_iso3, col_sectors,
        rows_data, cols_data, row_fd_cats=None
    ):
    """Build multi-index dataframes of the transaction matrix, final demand and total
       production from a Multi-Regional Input-Output Table dataframe.

    Parameters
    ----------
    mriot_df : pandas.DataFrame
        The Multi-Regional Input-Output Table
    col_iso3 : int
        Column's position of regions' iso names
    col_sectors : int
        Column's position of sectors' names
    rows_data : (int, int)
        Tuple of integers with positions of rows
        containing the MRIOT data for intermediate demand
        matrix.
        Final demand matrix is assumed to be the remaining columns
        of the DataFrame except the last one (which generally holds
        total output).
    cols_data : (int, int)
        Tuple of integers with positions of columns
        containing the MRIOT data
    row_fd_cats : int
        Integer index of the row containing the
        final demand categories.
    """

    start_row, end_row = rows_data
    start_col, end_col = cols_data

    sectors = mriot_df.iloc[start_row:end_row, col_sectors].unique()
    regions = mriot_df.iloc[start_row:end_row, col_iso3].unique()
    if row_fd_cats is None:
        n_fd_cat = (mriot_df.shape[1] - (end_col+1)) // len(regions)
        fd_cats = [f"fd_cat_{i}" for i in range(n_fd_cat)]
    else:
        fd_cats = mriot_df.iloc[row_fd_cats, end_col:-1].unique()

    multiindex = pd.MultiIndex.from_product(
        [regions, sectors], names=["region", "sector"]
    )

    multiindex_final_demand = pd.MultiIndex.from_product(
                [regions, fd_cats], names = ['region', 'category'])

    Z = mriot_df.iloc[start_row:end_row, start_col:end_col].values.astype(float)
    Z = pd.DataFrame(data=Z, index=multiindex, columns=multiindex)

    Y = mriot_df.iloc[start_row:end_row, end_col:-1].values.astype(float)
    Y = pd.DataFrame(
                    data = Y,
                    index = multiindex,
                    columns = multiindex_final_demand
                    )

    x = mriot_df.iloc[start_row:end_row, -1].values.astype(float)
    x = pd.DataFrame(data=x, index=multiindex, columns=["indout"])

    return Z, Y, x

def mriot_file_name(mriot_type, mriot_year):
    """Retrieve the original EXIOBASE3, WIOD16 or OECD21 MRIOT file name

    Parameters
    ----------
    mriot_type : str
    mriot_year : int

    Returns
    -------
    str
        name of MRIOT file
    """

    if mriot_type == "EXIOBASE3":
        return f"IOT_{mriot_year}_ixi.zip"

    if mriot_type == "WIOD16":
        return f"WIOT{mriot_year}_Nov16_ROW.xlsb"

    if mriot_type == "OECD21":
        return f"ICIO2021_{mriot_year}.csv"

    raise ValueError("Unknown MRIOT type")


def download_mriot(mriot_type, mriot_year, download_dir):
    """Download EXIOBASE3, WIOD16 or OECD21 Multi-Regional Input Output Tables
    for specific years.

    Parameters
    ----------
    mriot_type : str
    mriot_year : int
    download_dir : pathlib.PosixPath

    Notes
    -----
    The download of EXIOBASE3 and OECD21 tables makes use of pymrio functions.
    The download of WIOD16 tables requires ad-hoc functions, since the
    related pymrio functions were broken at the time of implementation
    of this function.
    """

    if mriot_type == "EXIOBASE3":
        pymrio.download_exiobase3(
            storage_folder=download_dir, system="ixi", years=[mriot_year]
        )

    elif mriot_type == "WIOD16":
        download_dir.mkdir(parents=True, exist_ok=True)
        downloaded_file_name = u_fh.download_file(
            WIOD_FILE_LINK,
            download_dir=download_dir,
        )
        downloaded_file_zip_path = Path(downloaded_file_name + ".zip")
        Path(downloaded_file_name).rename(downloaded_file_zip_path)

        with zipfile.ZipFile(downloaded_file_zip_path, "r") as zip_ref:
            zip_ref.extractall(download_dir)

    elif mriot_type == "OECD21":
        years_groups = ["1995-1999", "2000-2004", "2005-2009", "2010-2014", "2015-2018"]
        year_group = years_groups[int(np.floor((mriot_year - 1995) / 5))]

        pymrio.download_oecd(storage_folder=download_dir, years=year_group)


def parse_mriot(mriot_type, downloaded_file):
    """Parse EXIOBASE3, WIOD16 or OECD21 MRIOT for specific years

    Parameters
    ----------
    mriot_type : str
    downloaded_file : pathlib.PosixPath

    Notes
    -----
    The parsing of EXIOBASE3 and OECD21 tables makes use of pymrio functions.
    The parsing of WIOD16 tables requires ad-hoc functions, since the
    related pymrio functions were broken at the time of implementation
    of this function.
    """

    if mriot_type == "EXIOBASE3":
        mriot = pymrio.parse_exiobase3(path=downloaded_file)
        # no need to store A
        mriot.A = None

    elif mriot_type == "WIOD16":
        mriot_df = pd.read_excel(downloaded_file, engine="pyxlsb")

        Z, Y, x = parse_mriot_from_df(
            mriot_df,
            col_iso3=2,
            col_sectors=1,
            row_fd_cats=2,
            rows_data=(5,2469),
            cols_data=(4,2468),
        )

        mriot = pymrio.IOSystem(Z=Z, Y=Y, x=x)
        multiindex_unit = pd.MultiIndex.from_product(
                [mriot.get_regions(), mriot.get_sectors()],
                names = ['region', 'sector']
                )
        mriot.unit = pd.DataFrame(
                    data = np.repeat(["M.USD"], len(multiindex_unit)),
                    index = multiindex_unit,
                    columns = ["unit"]
                    )

    elif mriot_type == "OECD21":
        mriot = pymrio.parse_oecd(path=downloaded_file)
        mriot.x = pymrio.calc_x(mriot.Z, mriot.Y)

    else:
        raise RuntimeError(f"Unknown mriot_type: {mriot_type}")
    return mriot


class SupplyChain:
    """SupplyChain class.

    The SupplyChain class provides methods for loading Multi-Regional Input-Output
    Tables (MRIOT) and computing direct, indirect and total impacts.

    Attributes
    ----------
    mriot : pymrio.IOSystem
            An object containing all MRIOT related info (see also pymrio package):
                mriot.Z : transaction matrix, or interindustry flows matrix
                mriot.Y : final demand
                mriot.x : industry or total output
                mriot.meta : metadata
    secs_exp : pd.DataFrame
            Exposure dataframe of each region/sector in the MRIOT. Columns are the
            same as the chosen MRIOT.
    secs_imp : pd.DataFrame
            Impact dataframe for the directly affected countries/sectors for each event with
            impacts. Columns are the same as the chosen MRIOT and rows are the hazard events ids.
    secs_shock : pd.DataFrame
            Shocks (i.e. impact / exposure) dataframe for the directly affected countries/sectors
            for each event with impacts. Columns are the same as the chosen MRIOT and rows are the
            hazard events ids.
    inverse : dict
            Dictionary with keys being the chosen approach (ghosh, leontief)
            and values the Leontief (L, if approach is leontief) or Ghosh (G, if
            approach is ghosh) inverse matrix.
    coeffs : dict
            Dictionary with keys the chosen approach (ghosh, leontief)
            and values the Technical (A, if approach is leontief) or allocation
            (B, if approach is ghosh) coefficients matrix.
    sim: boario.simulation.Simulation
            Boario's simulation object. Only relevant when io_approach in "boario_aggregated" or
            "boario_disaggregated". Default is None.
    events_date: np.array
            Integer date corresponding to the proleptic Gregorian ordinal, where January 1 of year
            1 has ordinal 1 (ordinal format of datetime library) of events leading to impact.
            Deafult is None.
    supchain_imp : dict
            Dictionary with keys the chosen approach (ghosh, leontief or boario
            and its variations) and values dataframes of production losses (ghosh, leontief)
            or production dynamics (boario and its variations) to countries/sectors for each event.
            For each dataframe, columns are the same as the chosen MRIOT and rows are the
            hazard events' ids.
    """

    def __init__(self, mriot):

        """Initialize SupplyChain.

        Parameters
        ----------
        mriot : pymrio.IOSystem
                An object containing all MRIOT related info (see also pymrio package):
                    mriot.Z : transaction matrix, or interindustry flows matrix
                    mriot.Y : final demand
                    mriot.x : industry or total output
                    mriot.meta : metadata
        """

        self.mriot = mriot
        self.secs_exp = None
        self.secs_imp = None
        self.secs_shock = None
        self.sim = None
        self.events_date = None
        self.inverse = dict()
        self.coeffs = dict()
        self.supchain_imp = dict()

    @classmethod
    def from_mriot(
        cls, mriot_type, mriot_year, mriot_dir=MRIOT_DIRECTORY, del_downloads=True
    ):
        """Download, parse and read WIOD16, EXIOBASE3, or OECD21 Multi-Regional
        Input-Output Tables.

        Parameters
        ----------
        mriot_type : str
            Type of mriot table to use.
            The three possible types are: 'EXIOBASE3', 'WIOD16', 'OECD21'
        mriot_year : int
            Year of MRIOT
        mriot_dir : pathlib.PosixPath
            Path to the MRIOT folder. Default is CLIMADA storing directory.
        del_downloads : bool
            If the downloaded files are deleted after saving the parsed data. Default is
            True. WIOD16 and OECD21 data are downloaded as group of years.

        Notes
        -----
           EXIOBASE3 different world regions (WA, WF, WL, WM and WE) are aggregated
           to a single Rest of the World (ROW) region.

           Sometime, the Change In Inventory (CII) column of final demand lead to
           total final demand being negative, which causes problem with some indirect
           impact computation. Current solution is to set CII to 0 for (region,sector)
           where final demand is negative. In such a case, production vector is
           recomputed accordingly, and a warning is raised.

        Returns
        -------
        mriot : pymrio.IOSystem
            An object containing all MRIOT related info (see also pymrio package):
                mriot.Z : transaction matrix, or interindustry flows matrix
                mriot.Y : final demand
                mriot.x : total output
                mriot.meta : metadata
        """

        # download directory and file of interest
        downloads_dir = mriot_dir / mriot_type / "downloads"
        downloaded_file = downloads_dir / mriot_file_name(mriot_type, mriot_year)

        # parsed data directory
        parsed_data_dir = mriot_dir / mriot_type / str(mriot_year)

        # if data were not downloaded nor parsed: download, parse and save parsed
        if not downloaded_file.exists() and not parsed_data_dir.exists():
            download_mriot(mriot_type, mriot_year, downloads_dir)

            mriot = parse_mriot(mriot_type, downloaded_file)
            mriot.save(parsed_data_dir)

            if del_downloads:
                for dwn in downloads_dir.iterdir():
                    dwn.unlink()
                downloads_dir.rmdir()

        # if data were downloaded but not parsed: parse and save parsed
        elif downloaded_file.exists() and not parsed_data_dir.exists():
            mriot = parse_mriot(mriot_type, downloaded_file)
            mriot.save(parsed_data_dir)

        # if data were parsed and saved: load them
        else:
            mriot = pymrio.load(path=parsed_data_dir)

        # aggregate ROWs for EXIOBASE:
        if mriot_type == 'EXIOBASE3':
            agg_regions = mriot.get_regions().tolist()[:-5] + ['ROW']*5
            mriot = mriot.aggregate(region_agg = agg_regions)

        # Check if negative demand - this happens when the
        # "Changes in Inventory (CII)" demand category is
        # larger than the sum of all other categories
        if (mriot.Y.sum(axis=1) < 0).any():
            warnings.warn("Found negatives values in total final demand, "
                        "setting them to 0 and recomputing production vector")
            mriot.Y.loc[mriot.Y.sum(axis=1) < 0] = mriot.Y.loc[mriot.Y.sum(axis=1) < 0].clip(lower=0)
            mriot.x = pymrio.calc_x(mriot.Z, mriot.Y)

        mriot.meta.change_meta(
            "description", "Metadata for pymrio Multi Regional Input-Output Table"
        )
        mriot.meta.change_meta("name", f"{mriot_type}-{mriot_year}")

        return cls(mriot=mriot)

    def calc_shock_to_sectors(self,
                              exposure,
                              impact,
                              impacted_secs=None,
                              shock_factor=None
                              ):
        """Calculate exposure, impact and shock at the sectorial level.
        This function translate spatially-distributed exposure and impact
        information into exposure and impact in the MRIOT's region/sectors
        typology, for each hazard event.

        Parameters
        ----------
        exposure : climada.entity.Exposure
            CLIMADA Exposure object of direct impact calculation
        impact : climada.engine.Impact
            CLIMADA Impact object of direct impact calculation
        impacted_secs : (range, np.ndarray, str, list)
            Information regarding the impacted sectors. This can be provided
            as positions of the impacted sectors in the MRIOT (as range or np.ndarray)
            or as sector names (as string or list).
        shock_factor : np.array
            Array length should equal the number of sectors. For each sector, it defines
            by which factor the ratio of direct losses over exposure translate into an economic shock
            (on production, final demand, capital stock, depending on the indirect impact method used).
            By default, the value is None, and the factor is 1 for all sectors.
        """

        if impacted_secs is None:
            warnings.warn(
                "No impacted sectors were specified. It is assumed that the exposure is "
                "representative of all sectors in the IO table"
            )
            impacted_secs = self.mriot.get_sectors().tolist()

        elif isinstance(impacted_secs, (range, np.ndarray)):
            impacted_secs = self.mriot.get_sectors()[impacted_secs].tolist()

        elif isinstance(impacted_secs, str):
            impacted_secs = [impacted_secs]

        if shock_factor is None:
            shock_factor = np.repeat(1, self.mriot.x.shape[0])

        events_w_imp_bool = np.asarray(impact.imp_mat.sum(1)!=0).flatten()
        self.events_date = impact.date[events_w_imp_bool]

        self.secs_exp = pd.DataFrame(
            0.0,
            index=["total_value"],
            columns=self.mriot.Z.columns
        )
        self.secs_imp = pd.DataFrame(
            0.0,
            index=impact.event_id[events_w_imp_bool],
            columns=self.mriot.Z.columns
        )
        self.secs_imp.index = self.secs_imp.index.set_names('event_id')

        mriot_type = self.mriot.meta.name.split("-")[0]

        for exp_regid in exposure.gdf.region_id.unique():
            exp_bool = exposure.gdf.region_id == exp_regid
            tot_value_reg_id = exposure.gdf[exp_bool].value.sum()
            tot_imp_reg_id = impact.imp_mat[events_w_imp_bool][:,exp_bool].sum(1)

            mriot_reg_name = self.map_exp_to_mriot(exp_regid, mriot_type)
            secs_prod = self.mriot.x.loc[(mriot_reg_name, impacted_secs), :]

            secs_prod_ratio = (secs_prod / secs_prod.sum()).values.flatten()

            # Overall sectorial stock exposure and impact are distributed among
            # subsectors proportionally to their own contribution to overall
            # sectorial production: Sum needed below in case of many ROWs, which are
            # aggregated into one region as per WIOD table.
            self.secs_exp.loc[:, (mriot_reg_name, impacted_secs)] += (
                tot_value_reg_id * secs_prod_ratio
                ) / self.conversion_factor()

            self.secs_imp.loc[:, (mriot_reg_name, impacted_secs)] += (
                tot_imp_reg_id * secs_prod_ratio
                ) / self.conversion_factor()

        self.secs_shock = self.secs_imp.divide(
            self.secs_exp.values
        ).fillna(0) * shock_factor

        if not np.all(self.secs_shock <= 1):
            warnings.warn(
                "Consider changing the provided stock-to-production losses ratios,"
                "as some of them lead to some sectors' production losses to "
                "exceed the maximum sectorial production. For these sectors, total "
                "production loss is assumed."
            )
            self.secs_shock[self.secs_shock > 1] = 1

    def calc_matrices(self, io_approach):
        """Build technical coefficient and Leontief inverse matrixes
        (if leontief approach) or allocation coefficients and
        Ghosh matrixes (if ghosh approach).

        Parameters
        ----------
        io_approach : str
            The adopted input-output modeling approach.
            Possible choices are 'leontief' or 'ghosh'.
        """

        io_model = {
            "leontief": (pymrio.calc_A, pymrio.calc_L),
            "ghosh": (calc_B, calc_G),
            "boario": (pymrio.calc_A, pymrio.calc_L),
        }
        try:
            coeff_func, inv_func = io_model[io_approach]
        except KeyError as keyerr:
            raise KeyError("The provided approach does not exist") from keyerr

        self.coeffs.update({io_approach: coeff_func(self.mriot.Z, self.mriot.x)})
        self.inverse.update({io_approach: inv_func(self.coeffs[io_approach])})

    def calc_impacts(self,
                    io_approach,
                    exposure=None,
                    impact=None,
                    impacted_secs=None,
                    shock_factor=None,
                    boario_params=dict(),
                    boario_type='recovery',
                    boario_aggregate='agg'
                    ):
        """Calculate indirect production impacts based on to the
        chosen input-output approach.

        Parameters
        ----------
        io_approach : str
            The adopted input-output modeling approach.
            Possible choices are 'leontief', 'ghosh' or 'boario'
            'boario_recovery', 'boario_rebuild' and 'boario_shockprod'.
        exposure : climada.entity.Exposure
            CLIMADA Exposure object of direct impact calculation. Default is None.
        impact : climada.engine.Impact
            CLIMADA Impact object of direct impact calculation. Default is None.
        impacted_secs : (range, np.ndarray, str, list)
            Information regarding the impacted sectors. This can be provided
            as positions of the impacted sectors in the MRIOT (as range or np.ndarray)
            or as sector names (as string or list). Default is None.
        shock_factor : np.array
            It has length equal to the number of sectors. For each sector, it defines to
            what extent the fraction of indirect losses differs from the one of direct
            losses (i.e., impact / exposure). Deafult value is None, which means that shock
            factors for all sectors are equal to 1, i.e., that production and stock losses
            fractions are the same.
        boario_params: dict
            Dictionary containing parameters to instantiate boario's ARIOPsiModel (key 'model'),
            Simulation (key 'sim') and Event (key 'event') classes. Parameters instantiating
            each class need to be stored in a dictionary, e.g., {'model': {}, 'sim': {}, 'event': {}}.
            You can also specify "show_progress=False" to remove the progress bar during simulations.
            Only meangingful when io_approach='boario'. Default is None.
        boario_type: str
            The chosen boario type. Possible choices are 'recovery', 'rebuild' and
            'production_shock'. Only meaningful when io_approach='boario'. Default 'recovery'.
        boario_aggregate: str
            Whether events are aggregated or not. Possible choices are 'agg' or 'sep'.
            Only meaningful when io_approach='boario'. Default is 'agg'.

        Notes
        -----
        * The Leontief approach assumes the shock to degrade the final demand,
          and computes the resulting changed production.
        * The Ghosh approach assumes the shock to impact value added,
          and computes the resulting production.
        * The BoARIO approach assumes the shock to incapacitate productive capital
          (and possibly generate a reconstruction demand with ``boario_type="rebuild"``)
          and computes the change of production over time with the ARIO model.
          See the `BoARIO documentation <https://spjuhel.github.io/BoARIO/>`_ for more details
          (Note that not all features of BoARIO are included yet).

        References
        ----------
        [1] W. W. Leontief, Output, employment, consumption, and investment,
        The Quarterly Journal of Economics 58, 1944.
        [2] Ghosh, A., Input-Output Approach in an Allocation System,
        Economica, New Series, 25, no. 97: 58-64. doi:10.2307/2550694, 1958.
        [3] Kitzes, J., An Introduction to Environmentally-Extended Input-Output
        Analysis, Resources, 2, 489-503; doi:10.3390/resources2040489, 2013.
        """

        self.calc_matrices(io_approach=io_approach)

        # This block is a temporary `fix`, before a deeper rework of the module.
        # The underlying problem is that if secs_shock was set outside this method,
        # impact and exposure are ignored as ``calc_shock_to_sectors`` is not run in the next block.
        # The objective here is to warn user.
        if ((exposure is not None) or (impact is not None)) and self.secs_shock is not None:
            warnings.warn("``impact`` and ``exposure`` given in argument while ``secs_shock`` is already set. They will be ignored")

        if self.secs_shock is None:
            self.calc_shock_to_sectors(exposure, impact, impacted_secs, shock_factor)

        n_events = self.secs_shock.shape[0]
        if io_approach == "leontief":
            degr_demand = (
                self.secs_shock * self.mriot.Y.sum(1)
            )

            self.supchain_imp.update({io_approach : pd.concat(
                [
                    pymrio.calc_x_from_L(self.inverse[io_approach], degr_demand.iloc[i])
                    for i in range(n_events)
                ],
                axis=1,
            ).T.set_index(self.secs_shock.index)})

        elif io_approach == "ghosh":
            value_added = calc_va(self.mriot.Z, self.mriot.x)
            degr_value_added = (
                self.secs_shock * value_added.values
            )

            self.supchain_imp.update({io_approach : pd.concat(
                [
                    calc_x_from_G(self.inverse[io_approach], degr_value_added.iloc[i])
                    for i in range(n_events)
                ],
                axis=1,
            ).T.set_index(self.secs_shock.index)})

        elif io_approach == 'boario':

            # This is `quick and dirty`, a better solution will
            # be brought by the rework of the module.
            if self.secs_exp is None or self.secs_imp is None:
                raise AttributeError("``secs_exp`` and/or ``secs_imp`` attributes were not set properly. This may be caused by ``secs_shock`` not being None, from previous computation, set it to None and rerun the method.")

            self.mriot.A = self.coeffs[io_approach]
            self.mriot.L = self.inverse[io_approach]
            show_progress= boario_params.get("show_progress",True)
            for boario_param_type in ['model', 'sim']:
                if boario_param_type not in boario_params:
                    warnings.warn(f"""BoARIO '{boario_param_type}' parameters were not specified and default values are used. This is not recommended and likely undesired."""
                                  )
                    boario_params.update({f'{boario_param_type}':{}})

            if 'event' not in boario_params:
                if boario_type == 'recovery':
                    boario_params.update({'event': {'recovery_time' : 60}})
                    warnings.warn(f"BoARIO {boario_type} event parameters were not specified."
                              "This is not recommended. Default value for `recovery_time` is 60.")
                elif boario_type == 'rebuild':
                    raise ValueError("""Using the ``boario_type=rebuild`` requires you to define the rebuilding sectors in the ``boario_params`` argument:
                    {"model":{}, "sim":{}, "event":{"rebuilding_sectors={"<sector_name>":reconstruction_share}}}""")

            # call ARIOPsiModel with default params
            model = ARIOPsiModel(self.mriot,
                                 # productive capital vector (i.e. exposure) needs to be in
                                 # MRIOT's unit, this is the case as self.secs_exp was rescaled
                                 # with the conversion_factor upon its construction
                                 productive_capital_vector = self.secs_exp,
                                 # model monetary factor equals the MRIOT's unit
                                 monetary_factor = self.conversion_factor(),
                                 **boario_params['model']
                                 )

            # run simulation up to one year after the last event
            self.sim = Simulation(
                        model,
                        n_temporal_units_to_sim = int(self.events_date.max()-self.events_date.min()+365),
                        **boario_params['sim']
                        )

            if boario_type == 'recovery':

                events_list = [EventKapitalRecover.from_series(
                                        impact=self.secs_imp.iloc[i],
                                        occurrence = int(self.events_date[i]-self.events_date.min()+1),
                                        # event monetary factor equals the impact units. self.secs_imp
                                        # was rescaled by the conversion_factor upon its construction so
                                        # we pass the conversion_factor as unit
                                        event_monetary_factor = self.conversion_factor(),
                                        **boario_params['event']
                            ) for i in range(n_events)
                ]

            elif boario_type == 'rebuild':

                events_list = [EventKapitalRebuild.from_series(
                                        impact=self.secs_imp.iloc[i],
                                        occurrence = (self.events_date[i]-self.events_date.min()+1),
                                        # event monetary factor equal to the impact units. self.secs_imp
                                        # was rescaled by the conversion_factor upon its construction so
                                        # we pass the conversion_factor as unit
                                        event_monetary_factor = self.conversion_factor(),
                                        **boario_params['event']
                            ) for i in range(n_events)
                ]

            # Currently not working in BoARIO.
            # elif boario_type == 'shockprod':
            #     from boario.event import EventArbitraryProd
            #     events_list = [
            #        EventArbitraryProd.from_series(
            #            impact=self.secs_shock.iloc[i],
            #            occurrence = (self.events_date[i]-self.events_date.min()+1),
            #            **boario_params['event']
            #         )
            #         for i in range(n_events)
            #     ]

            else:
                raise RuntimeError(f"Unknown boario type : {boario_type}")

            if boario_aggregate == 'agg':
                self.sim.add_events(events_list)
                self.sim.loop(progress=show_progress)
                self.supchain_imp.update({
                    f'{io_approach}_{boario_type}_{boario_aggregate}' :
                    self.sim.production_realised.copy()[
                                self.secs_imp.columns]
                })

            elif boario_aggregate == 'sep':
                results = []
                for ev in events_list:
                    self.sim.add_event(ev)
                    self.sim.loop(progress=show_progress)
                    results.append(
                        self.sim.production_realised.copy()[
                            self.secs_imp.columns]
                        )
                    self.sim.reset_sim_full()

                self.supchain_imp.update({
                    f'{io_approach}_{boario_type}_{boario_aggregate}' : results
                })

            else:
                raise RuntimeError(f"Unknown boario aggregation type: {boario_aggregate}")

        # The calc_matrices() call at the top fails before so this is not usefull
        # else:
        #    raise RuntimeError(f"Unknown io_approach: {io_approach}")

    def conversion_factor(self):
        """Conversion factor based on unit specified in the Multi-Regional Input-Output Table."""

        unit = None
        if isinstance(self.mriot.unit, pd.DataFrame):
            unit = self.mriot.unit.values[0][0]
        elif isinstance(self.mriot.unit, str):
            unit = self.mriot.unit
        if unit in ["M.EUR", "Million USD", "M.USD"]:
            conversion_factor = 1e6
        else:
            conversion_factor = 1
            warnings.warn(
                "No known unit was provided. It is assumed that values do not need to "
                "be converted."
            )
        return conversion_factor

    def map_exp_to_mriot(self, exp_regid, mriot_type):
        """
        Map regions names in exposure into Input-Output regions names.
        exp_regid must follow ISO 3166 numeric country codes.
        """

        if mriot_type == "EXIOBASE3":
            mriot_reg_name = u_coord.country_to_iso(exp_regid, "alpha2")

        elif mriot_type in ["WIOD16", "OECD21"]:
            mriot_reg_name = u_coord.country_to_iso(exp_regid, "alpha3")

        else:
            warnings.warn(
                "For a correct calculation the format of regions' names in exposure and "
                "the IO table must match."
            )
            return exp_regid

        idx_country = np.where(self.mriot.get_regions() == mriot_reg_name)[0]

        if not idx_country.size > 0.0:
            mriot_reg_name = "ROW"

        return mriot_reg_name
