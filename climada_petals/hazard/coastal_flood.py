import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Polygon

from climada.hazard import Hazard
from climada.util import files_handler as u_fh
import climada.util.coordinates as u_coord

SOURCE_LINK = 'http://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/'
DOWNLOAD_DIRECTORY = Path.cwd() # TODO: we'll rather need something like CONFIG.hazard.coastal_flood.dir()

HAZ_TYPE = 'CF'
"""Hazard type acronym CoastalFlood"""

__all__ = ['CoastalFlood']

class CoastalFlood(Hazard):
    """Contains coastal flood events pulled by
    the Acqueduct project :
    https://www.wri.org/data/aqueduct-floods-hazard-maps
    """

    def __init__(self, **kwargs):
        """Empty constructor"""

        Hazard.__init__(self, HAZ_TYPE, **kwargs)

    @classmethod
    def from_aqueduct_tif(cls, rcps=None, years=None, return_periods=None,
                           subsidence=['nosub'], projection=['0'],
                           centroids=None, countries=None, shape=None,
                           boundaries=None, dwd_dir=DOWNLOAD_DIRECTORY):
        """
        TODO: proper docstring

        rcps : list or str
            RCPs scenarios. Possible values are ...
        years : list, str or int
            simulations' reference years. Possible values are ...
        return_periods : list or str
            simulations' return periods. Possible values are ...
        subsidence : list or str
            If there land subsidence is simulated or not. 
            Possible values are 'nosub' and 'wtsub'.
        projection : list or str
            TODO: investigate meaning.
            EXPLAIN HERE. Possible values are ... 
        centroids : Centroids
            centroids to extract
        countries : list of countries ISO3
            selection of countries
        shape : 

        """

        # assess number of files and from there the event_ids, these should be then passed one
        # by one as attrs in haz.from_raster, otherwise it wont concat properly

        if isinstance(rcps, str):
            rcps = [rcps]
        if isinstance(years, (str, int)):
            years = [years]
        if isinstance(return_periods, str):
            return_periods = [return_periods]
        if isinstance(subsidence, str):
            subsidence = [subsidence]
        if isinstance(projection, str):
            projection = [projection]

        file_names = [
                f'inuncoast_{rcp}_{sub}_{year}_rp{rp.zfill(4)}_{proj}.tif'
                    for rcp in rcps
                    for sub in subsidence
                    for year in years
                    for rp in return_periods
                    for proj in projection
                    # You can't have: 
                    # - year historic with rcp different than historical
                    # - rcp historical, no subsidence and year different than historic
                    if not (
                            ((year == 'hist') & (rcp != 'historical')) |
                            ((rcp == 'historical') & (sub == 'nosub') & (year != 'hist'))
                            )
                    ]

        hazs = []

        for i, file_name in enumerate(file_names):
            link_to_file = "".join([SOURCE_LINK, file_name])
            downloaded_file = dwd_dir / file_name

            if not downloaded_file.exists():
                u_fh.download_file(
                link_to_file,
                download_dir = dwd_dir,
                )

            if countries:
                geom = u_coord.get_land_geometry(countries).geoms

            # TODO: implement shape
            elif shape:
                # shapes = gpd.read_file(shape) # set projection
                # geom = shapes.geometry[0]
                pass

            # TODO: implement centroids
            elif centroids:
                pass

            elif boundaries:
                min_lon, min_lat, max_lon, max_lat = boundaries
                geom = [Polygon([(min_lon, min_lat),
                                (max_lon, min_lat),
                                (max_lon, max_lat),
                                (min_lon, max_lat)])]

            else:
                geom = None
                # TODO: LOGGER(loading for the whole world)

            _, rcp, sub, year, rp, proj = file_name.split('.tif')[0].split('_')
            rp = rp.lstrip('rp0')

            haz = cls().from_raster(files_intensity=file_name,
                                          geometry=geom,
                                          attrs={
                                                'event_id': np.array([i+1]),
                                                'event_name': np.array([f'{rcp}_{sub}_{year}_{rp}rp_{proj}']),
                                                #TODO: check if proper assignment
                                                'frequency': np.array([1 / int(rp)])
                                                })
            haz.units = 'm'
            haz.centroids.set_meta_to_lat_lon()
            hazs.append(haz)
    
        return cls().concat(hazs)