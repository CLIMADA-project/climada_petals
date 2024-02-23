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

Define RiverFlood class.
"""

__all__ = ['RiverFlood']

import logging
import datetime as dt
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd
import geopandas as gpd
from rasterio.warp import Resampling
from shapely.geometry import Polygon, MultiPolygon

from climada.util.constants import RIVER_FLOOD_REGIONS_CSV
import climada.util.coordinates as u_coord
from climada.hazard.base import Hazard
from climada.hazard.centroids import Centroids

NATID_INFO = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)


LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'RF'
"""Hazard type acronym RiverFlood"""


class RiverFlood(Hazard):
    """Contains flood events
    Flood intensities are calculated by means of the
    CaMa-Flood global hydrodynamic model

    Attributes
    ----------
    fla_event : 1d array(n_events)
        total flooded area for every event
    fla_annual : 1d array (n_years)
        total flooded area for every year
    fla_ann_av : float
        average flooded area per year
    fla_ev_av : float
        average flooded area per event
    fla_ann_centr : 2d array(n_years x n_centroids)
        flooded area in
        every centroid for every event
    fla_ev_centr : 2d array(n_events x n_centroids)
        flooded area in
        every centroid for every event
    """

    def __init__(self, *args, **kwargs):
        """Empty constructor"""
        Hazard.__init__(self, *args, haz_type=HAZ_TYPE, **kwargs)

    @classmethod
    def from_nc(cls, dph_path=None, frc_path=None, origin=False,
                centroids=None, countries=None, reg=None, shape=None, ISINatIDGrid=False,
                years=None):
        """Wrapper to fill hazard from nc_flood file

        Parameters
        ----------
        dph_path : str, optional
            Flood file to read (depth)
        frc_path : str, optional
            Flood file to read (fraction)
        origin : bool, optional
            Historical or probabilistic event. Default: False
        centroids : Centroids, optional
            centroids to extract
        countries : list of str, optional
            If `reg` is None, use this selection of countries (ISO3). Default: None
        reg : list of str, optional
            Use region code to consider whole areas. If not None, countries and centroids
            are ignored. Default: None
        shape : str or Path, optional
            If `reg` and `countries` are None, use the first geometry in this shape file to cut
            out the area of interest. Default: None
        ISINatIDGrid : bool, optional
            Indicates whether ISIMIP_NatIDGrid is used. Default: False
        years : list of int
            Years that are considered. Default: None

        Returns
        -------
        haz : RiverFlood instance

        Raises
        ------
        NameError
        """
        if years is None:
            years = [2000]
        if dph_path is None:
            raise NameError('No flood-depth-path set')
        if frc_path is None:
            raise NameError('No flood-fraction-path set')
        if not Path(dph_path).exists():
            raise NameError('Invalid flood-file path %s' % dph_path)
        if not Path(frc_path).exists():
            raise NameError('Invalid flood-file path %s' % frc_path)

        with xr.open_dataset(dph_path) as flood_dph:
            time = pd.to_datetime(flood_dph["time"].values)

        event_index = np.where(np.isin(time.year, years))[0]
        if len(event_index) == 0:
            raise AttributeError(f'No events found for selected {years}')
        bands = event_index + 1

        if countries or reg:
            # centroids as points
            if ISINatIDGrid:

                dest_centroids = RiverFlood._select_exact_area(countries, reg)[0]
                centroids_meta = dest_centroids.get_meta()

                haz = cls.from_raster(files_intensity=[dph_path],
                                      files_fraction=[frc_path], band=bands.tolist(),
                                      transform=centroids_meta['transform'],
                                      width=centroids_meta['width'],
                                      height=centroids_meta['height'],
                                      resampling=Resampling.nearest)
                haz_centroids_meta = haz.centroids.get_meta()
                x_i = ((dest_centroids.lon - haz_centroids_meta['transform'][2]) /
                       haz_centroids_meta['transform'][0]).astype(int)
                y_i = ((dest_centroids.lat - haz_centroids_meta['transform'][5]) /
                       haz_centroids_meta['transform'][4]).astype(int)

                fraction = haz.fraction[:, y_i * haz_centroids_meta['width'] + x_i]
                intensity = haz.intensity[:, y_i * haz_centroids_meta['width'] + x_i]

                haz.centroids = dest_centroids
                haz.intensity = sp.sparse.csr_matrix(intensity)
                haz.fraction = sp.sparse.csr_matrix(fraction)
            else:
                if reg:
                    iso_codes = u_coord.region2isos(reg)
                    # envelope containing counties
                    cntry_geom = u_coord.get_land_geometry(iso_codes)
                    haz = cls.from_raster(files_intensity=[dph_path],
                                          files_fraction=[frc_path],
                                          band=bands.tolist(),
                                          geometry=cntry_geom.geoms)
                else:
                    cntry_geom = u_coord.get_land_geometry(countries)
                    haz = cls.from_raster(files_intensity=[dph_path],
                                          files_fraction=[frc_path],
                                          band=bands.tolist(),
                                          geometry=cntry_geom.geoms)

        elif shape:
            shapes = gpd.read_file(shape)

            rand_geom = shapes.geometry[0]

            if isinstance(rand_geom, MultiPolygon):
                rand_geom = rand_geom.geoms
            elif isinstance(rand_geom, Polygon) or not isinstance(rand_geom, Iterable):
                rand_geom = [rand_geom]

            haz = cls.from_raster(files_intensity=[dph_path],
                                  files_fraction=[frc_path],
                                  band=bands.tolist(),
                                  geometry=rand_geom)

        elif not centroids:
            # centroids as raster
            haz = cls.from_raster(files_intensity=[dph_path],
                                  files_fraction=[frc_path],
                                  band=bands.tolist())

        else:  # use given centroids
            # if centroids.meta or grid_is_regular(centroids)[0]:
            #TODO: implement case when meta or regulargrid is defined
            #      centroids.meta or grid_is_regular(centroidsxarray)[0]:
            #      centroids>flood --> error
            #      reprojection, resampling.average (centroids< flood)
            #      (transform)
            #      reprojection change resampling"""
            # else:
            metafrc, fraction = u_coord.read_raster(frc_path, band=bands.tolist())
            metaint, intensity = u_coord.read_raster(dph_path, band=bands.tolist())
            x_i = ((centroids.lon - metafrc['transform'][2]) /
                   metafrc['transform'][0]).astype(int)
            y_i = ((centroids.lat - metafrc['transform'][5]) /
                   metafrc['transform'][4]).astype(int)
            fraction = fraction[:, y_i * metafrc['width'] + x_i]
            intensity = intensity[:, y_i * metaint['width'] + x_i]
            haz = cls(
                centroids=centroids,
                intensity=sp.sparse.csr_matrix(intensity),
                fraction=sp.sparse.csr_matrix(fraction),
            )

        haz.units = 'm'
        haz.event_id = np.arange(1, haz.intensity.shape[0] + 1)
        haz.event_name = list(map(str, years))

        if origin:
            haz.orig = np.ones(haz.size, bool)
        else:
            haz.orig = np.zeros(haz.size, bool)

        haz.frequency = np.ones(haz.size) / haz.size

        with xr.open_dataset(dph_path) as flood_dph:
            haz.date = np.array([
                dt.datetime(
                    flood_dph.time.dt.year.values[i],
                    flood_dph.time.dt.month.values[i],
                    flood_dph.time.dt.day.values[i],
                ).toordinal()
                for i in event_index
            ])

        return haz

    def set_from_nc(self, *args, **kwargs):
        """This function is deprecated, use RiverFlood.from_nc instead."""
        LOGGER.warning("The use of RiverFlood.set_from_nc is deprecated."
                       "Use LowFlow.from_nc instead.")
        self.__dict__ = RiverFlood.from_nc(*args, **kwargs).__dict__

    def exclude_trends(self, fld_trend_path, dis):
        """
        Function allows to exclude flood impacts that are caused in areas
        exposed discharge trends other than the selected one. (This function
        is only needed for very specific applications)

        Raises
        ------
        NameError
        """
        if not Path(fld_trend_path).exists():
            raise NameError('Invalid ReturnLevel-file path %s' % fld_trend_path)
        else:
            metafrc, trend_data = u_coord.read_raster(fld_trend_path, band=[1])
            x_i = ((self.centroids.lon - metafrc['transform'][2]) /
                   metafrc['transform'][0]).astype(int)
            y_i = ((self.centroids.lat - metafrc['transform'][5]) /
                   metafrc['transform'][4]).astype(int)

        trend = trend_data[:, y_i * metafrc['width'] + x_i]

        if dis == 'pos':
            dis_map = np.greater(trend, 0)
        else:
            dis_map = np.less(trend, 0)

        new_trends = dis_map.astype(int)

        new_intensity = np.multiply(self.intensity.todense(), new_trends)
        new_fraction = np.multiply(self.fraction.todense(), new_trends)

        self.intensity = sp.sparse.csr_matrix(new_intensity)
        self.fraction = sp.sparse.csr_matrix(new_fraction)

    def exclude_returnlevel(self, frc_path):
        """
        Function allows to exclude flood impacts below a certain return level
        by manipulating flood fractions in a way that the array flooded more
        frequently than the treshold value is excluded. (This function
        is only needed for very specific applications)

        Raises
        ------
        NameError
        """

        if not Path(frc_path).exists():
            raise NameError('Invalid ReturnLevel-file path %s' % frc_path)
        else:
            metafrc, fraction = u_coord.read_raster(frc_path, band=[1])
            x_i = ((self.centroids.lon - metafrc['transform'][2]) /
                   metafrc['transform'][0]).astype(int)
            y_i = ((self.centroids.lat - metafrc['transform'][5]) /
                   metafrc['transform'][4]).astype(int)
            fraction = fraction[:, y_i * metafrc['width'] + x_i]
            new_fraction = np.array(np.subtract(self.fraction.todense(),
                                                fraction))
            new_fraction = new_fraction.clip(0)
            self.fraction = sp.sparse.csr_matrix(new_fraction)

    def set_flooded_area(self, save_centr=False):
        """
        Calculates flooded area for hazard. sets yearly flooded area and
            flooded area per event

        Raises
        ------
        MemoryError
        """
        self.centroids.set_area_pixel()
        area_centr = self.centroids.get_area_pixel()
        event_years = np.array([dt.date.fromordinal(self.date[i]).year
                                for i in range(len(self.date))])
        years = np.unique(event_years)
        year_ev_mk = self._annual_event_mask(event_years, years)

        fla_ann_centr = np.zeros((len(years), len(self.centroids.lon)))
        fla_ev_centr = np.array(np.multiply(self.fraction.todense(),
                                            area_centr))
        self.fla_event = np.sum(fla_ev_centr, axis=1)
        for year_ind in range(len(years)):
            fla_ann_centr[year_ind, :] =\
                np.sum(fla_ev_centr[year_ev_mk[year_ind, :], :],
                       axis=0)
        self.fla_annual = np.sum(fla_ann_centr, axis=1)
        self.fla_ann_av = np.mean(self.fla_annual)
        self.fla_ev_av = np.mean(self.fla_event)
        if save_centr:
            self.fla_ann_centr = sp.sparse.csr_matrix(fla_ann_centr)
            self.fla_ev_centr = sp.sparse.csr_matrix(fla_ev_centr)

    def _annual_event_mask(self, event_years, years):
        """Assignes events to each year

        Returns
        -------
        bool array (columns contain events, rows contain years)
        """
        event_mask = np.full((len(years), len(event_years)), False, dtype=bool)
        for year_ind, year in enumerate(years):
            events = np.where(event_years == year)[0]
            event_mask[year_ind, events] = True
        return event_mask

    def set_flood_volume(self, save_centr=False):
        """Calculates flooded area for hazard. sets yearly flooded area and
            flooded area per event

        Raises
        ------
        MemoryError
        """

        fv_ann_centr = np.multiply(self.fla_ann_centr.todense(), self.intensity.todense())

        if save_centr:
            self.fv_ann_centr = sp.sparse.csr_matrix(self.fla_ann_centr)
        self.fv_annual = np.sum(fv_ann_centr, axis=1)

    @staticmethod
    def _select_exact_area(countries=None, reg=None):
        """Extract coordinates of selected countries or region
        from NatID grid. If countries are given countries are cut,
        if only reg is given, the whole region is cut.

        Parameters
        ----------
        countries :
            List of countries
        reg :
            List of regions

        Raises
        ------
        KeyError

        Returns
        -------
        centroids
        """
        lat, lon = u_coord.get_region_gridpoints(
            countries=countries, regions=reg, basemap="isimip", resolution=150)

        if reg:
            country_isos = u_coord.region2isos(reg)
        else:
            country_isos = countries if countries else []

        natIDs = u_coord.country_iso2natid(country_isos)

        centroids = Centroids.from_lat_lon(lat, lon)
        centroids.id = np.arange(centroids.lon.shape[0])
        # centroids.set_region_id()
        return centroids, country_isos, natIDs
