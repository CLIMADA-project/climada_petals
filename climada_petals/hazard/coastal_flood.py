from typing import Iterable, Union, Optional
import numpy as np
from shapely.geometry import Polygon

from climada.hazard import Hazard
from climada.util import files_handler as u_fh
import climada.util.coordinates as u_coord
from climada import CONFIG

AQUEDUCT_SOURCE_LINK = CONFIG.hazard.coastal_flood.resources.aqueduct.str()
DOWNLOAD_DIRECTORY = CONFIG.hazard.coastal_flood.local_data.aqueduct.dir()

HAZ_TYPE = 'CF'
"""Hazard type acronym CoastalFlood"""

__all__ = ['CoastalFlood']

class CoastalFlood(Hazard):
    """
    Contains coastal flood events pulled by the Acqueduct project :
        https://www.wri.org/data/aqueduct-floods-hazard-maps
    """

    def __init__(self, **kwargs):
        """Empty constructor"""

        Hazard.__init__(self, HAZ_TYPE, **kwargs)

    @classmethod
    def from_aqueduct_tif(cls,
                          rcp: str,
                          target_year : str,
                          return_periods: Union[int, Iterable[int]],
                          subsidence: Optional[str] = 'wtsub',
                          percentile: str = '95',
                          countries: Optional[Union[str, Iterable[str]]] = None,
                          boundaries: Iterable[float] = None,
                          dwd_dir=DOWNLOAD_DIRECTORY):
        """
        Download and load coastal flooding events from the Aqueduct dataset.
        For more info on the data see: 
            https://files.wri.org/d8/s3fs-public/aqueduct-floods-methodology.pdf

        Parameters
        ----------
        rcp : str
            RCPs scenario. Possible values are historical, 45 and 85.
        target_year : str
            future target year. Possible values are hist, 2030, 2050 and 2080.
        return_periods : int or list of int
            events' return periods.
            Possible values are 2, 5, 10, 25, 50, 100, 250, 500, 1000.
        subsidence : str
            If land subsidence is simulated or not.
            Possible values are nosub and wtsub. Default wtsub.
        percentile : str
            Sea level rise scenario (in percentile).
            Possible values are 5, 50 and 95.
        countries : str or list of str
            countries ISO3 codes
        boundaries : tuple of floats
            geographical boundaries in the order:
                minimum longitude, minimum latitude,
                maximum longitude, maximum latitude
        Returns
        -------
        haz : CoastalFlood instance
        """

        if (target_year == 'hist') & (rcp != 'historical'):
            raise ValueError(f"RCP{rcp} cannot have hist as target year")

        if (rcp == 'historical') & (subsidence == 'nosub') & (target_year != 'hist'):
            raise ValueError("Historical without subsidence can only have hist as target year")

        if isinstance(return_periods, int):
            return_periods = [return_periods]

        return_periods.sort(reverse=True)

        if isinstance(countries, str):
            countries = [countries]

        rcp_name = f"rcp{rcp[0]}p{rcp[1]}" if rcp in ['45', '85'] else rcp
        rps_name = [f"{str(rp).zfill(4)}" for rp in return_periods]
        perc_name = f"0_perc_{percentile.zfill(2)}" if percentile in ['05', '50'] else '0'

        file_names = [
                f'inuncoast_{rcp_name}_{subsidence}_{target_year}_rp{rp.zfill(4)}_{perc_name}.tif'
                for rp in rps_name
                    ]

        hazs = []
        for i, file_name in enumerate(file_names):
            link_to_file = "".join([AQUEDUCT_SOURCE_LINK, file_name])
            file_path = dwd_dir / file_name

            if not file_path.exists():
                u_fh.download_file(link_to_file, download_dir=dwd_dir)

            if countries:
                geom = u_coord.get_land_geometry(countries).geoms

            elif boundaries:
                min_lon, min_lat, max_lon, max_lat = boundaries
                geom = [Polygon([(min_lon, min_lat),
                                 (max_lon, min_lat),
                                 (max_lon, max_lat),
                                 (min_lon, max_lat)])]
            else:
                geom = None

            event_name = f'1-in-{return_periods[i]}y_{percentile}pct_{rcp_name}_{target_year}_{subsidence}'

            haz = cls().from_raster(files_intensity=file_path,
                                    geometry=geom,
                                    attrs={'event_id': np.array([i+1]),
                                           'event_name': np.array([event_name]),
                                           'frequency': np.array([1 / int(return_periods[i])])})
            haz.units = 'm'
            haz.centroids.set_meta_to_lat_lon()
            hazs.append(haz)

        haz = cls().concat(hazs)
        haz.frequency = np.diff(haz.frequency, prepend=0)

        return haz
