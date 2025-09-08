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

Download openstreetmap data from the Overpass API
"""

import itertools
import logging
import time
import geopandas as gpd
import pandas as pd
import overpy
import shapely

LOGGER = logging.getLogger(__name__)


class OSMApiQuery:
    """
    Queries features directly via the overpass turbo API.

    Parameters
    ----------    
    area: tuple (ymin, xmin, ymax, xmax)
    condition: str
        must be of format '["key"]' or '["key"="value"]', etc.

    Note
    -----
    The area (bounding box) ordering in the overpass query language is different
    from the convention in shapely / geopandas. If you directly pass area as 
    bbox, make sure the order is (ymin, xmin, ymax, xmax).
    If you use a classib bbox from shapely or geopandas, use the f
    rom_bounding_box() method, which reorders the inputs!
    """

    def __init__(self, area, condition):
        self.area = area
        self.condition = condition

    @classmethod
    def from_bounding_box(cls, bbox, condition):
        """
        Parameters
        ----------    
        bbox: tuple
            bbox as given from the standard convention of a shapely / geopandas
        bounding box as (xmin, ymin, xmax, ymax)
        condition: str
            must be of format '["key"]' or '["key"="value"]', etc.
        """
        # Maybe need to make sure that bbox is what you expect it is?
        xmin, ymin, xmax, ymax = bbox
        return cls((ymin, xmin, ymax, xmax), condition)

    @classmethod
    def from_polygon(cls, polygon, condition):
        """
        Parameters
        ----------    
        polygon: shapely.geometry.polygon
        condition: str
            must be of format '["key"]' or '["key"="value"]', etc.
        """
        lon, lat = polygon.exterior.coords.xy
        lat_lon_str = " ".join([str(y)+" "+str(x) for y, x in zip(lat, lon)])
        return cls(area=f'(poly:"{lat_lon_str}")', condition=condition)

    def _insistent_osm_api_query(self, query_clause, read_chunk_size=100000,
                                 end_of_patience=127):
        """Runs a single Overpass API query through overpy.Overpass.query.
        In case of failure it tries again after an ever increasing waiting period.
        If the waiting period surpasses a given limit an exception is raised.

        Parameters:
            query_clause (str): the query
            read_chunk_size (int): paramter passed over to overpy.Overpass.query
            end_of_patience (int): upper limit for the next waiting period to proceed.

        Returns:
            result as returned by overpy.Overpass.query
        """
        api = overpy.Overpass(read_chunk_size=read_chunk_size)
        waiting_period = 1
        while True:
            try:
                return api.query(query_clause)
            except overpy.exception.OverpassTooManyRequests:
                if waiting_period < end_of_patience:
                    LOGGER.warning("""Too many Overpass API requests -
                                   trying again in {waiting_period} seconds """)
                else:
                    raise Exception("Overpass API is consistently unavailable")
            except Exception as exc:
                if waiting_period < end_of_patience:
                    LOGGER.warning(f"""{exc}
                                   Trying again in {waiting_period} seconds""")
                else:
                    raise Exception(
                        "The Overpass API is consistently unavailable")
            time.sleep(waiting_period)
            waiting_period *= 2

    def _overpass_query_string(self):
        return f'[out:json][timeout:180];(nwr{self.condition}{self.area};(._;>;););out;'

    def _assemble_from_relations(self, result):
        """
        pick out those nodes and ways from result instance that belong to
        relations. Assemble relations into gdfs. Keep track of which nodes
        and ways have been "used up" already

        Parameters
        ---------
        result : overpy.Overpass result object

        Returns
        -------
        nodes_taken : list
            node-ids that have been used to construct relations.
            Not "available" anymore for further constructions
        ways_taken : list
            way-ids that have been used to construct relations.
            Not "available" anymore for further constructions
        gdf_rels : gpd.GeoDataFrame
            gdf with relations that were assembled from result object
        """

        nodes_taken = []
        ways_taken = []
        data_geom = []
        data_id = []
        data_tags = []

        for relation in result.relations:
            data_tags.append(relation.tags)
            data_id.append(relation.id)
            roles = []
            relationways = []

            for way in relation.members:
                relationways.append(way.ref)
                roles.append(way.role)

            ways_taken.append(relationways)

            nodes_taken_mp, gdf_polys = self._assemble_from_ways(
                result, relationways, closed_lines_are_polys=True)

            nodes_taken.append(nodes_taken_mp)

            gdf_polys['role'] = roles

            # separate relationways into inner, outer polygons and linestrings,
            # combine them.

            # step 1: polygons to multipolygons
            inner_mp = shapely.geometry.MultiPolygon(
                gdf_polys.geometry[(gdf_polys.geometry.type == 'Polygon') &
                                   (gdf_polys.role == 'inner')].values)
            outer_mp = shapely.geometry.MultiPolygon(
                gdf_polys.geometry[(gdf_polys.geometry.type == 'Polygon') &
                                   (gdf_polys.role == 'outer')].values)

            # step 2: poly from lines --> multiline --> line --> polygon
            lines = gdf_polys.geometry[
                (gdf_polys.geometry.type == 'LineString')].values
            if len(lines) > 0:
                poly = shapely.geometry.Polygon(
                    shapely.ops.linemerge(shapely.geometry.MultiLineString(lines)))
            else:
                poly = shapely.geometry.Polygon([])

            # step 3: combine to one multipoly
            multipoly = shapely.ops.unary_union([outer_mp - inner_mp, poly])
            data_geom.append(multipoly)

            if multipoly.area == 0:
                LOGGER.info('Empty geometry encountered.')

        gdf_rels = gpd.GeoDataFrame(
            data={'osm_id': data_id, 'geometry': data_geom, 'tags': data_tags},
            geometry='geometry', crs='epsg:4326')

        # list of lists into list:
        nodes_taken = list(itertools.chain.from_iterable(nodes_taken))
        ways_taken = list(itertools.chain.from_iterable(ways_taken))

        return nodes_taken, ways_taken, gdf_rels

    def _assemble_from_ways(self, result, ways_avail, closed_lines_are_polys):
        """
        pick out those nodes and ways from result instance that belong to
        ways. Assemble ways into gdfs. Keep track of which nodes
        and ways have been "used up" already

        Parameters
        ---------
        result : overpy.Overpass result object
        ways_avail : list
            way-ids that have not yet been used for relation construction
            and are hence available for way constructions
        closed_lines_are_polys : bool
            whether closed lines are polygons

        Returns
        -------
        nodes_taken : list
            node-ids that have been used to construct relations.
            Not "available" anymore for further constructions
        ways_taken : list
            way-ids that have been used to construct relations.
            Not "available" anymore for further constructions
        gdf_ways : gpd.GeoDataFrame
            gdf with ways that were assembled from result object
        """

        nodes_taken = []
        data_geom = []
        data_id = []
        data_tags = []

        for way in result.ways:
            if way.id in ways_avail:
                node_lat_lons = []
                for node in way.nodes:
                    nodes_taken.append(node.id)
                    node_lat_lons.append((float(node.lat), float(node.lon)))
                data_geom.append(shapely.geometry.LineString(node_lat_lons))
                data_id.append(way.id)
                data_tags.append(way.tags)

            if closed_lines_are_polys:
                data_geom = [shapely.geometry.Polygon(way) if way.is_closed
                             else way for way in data_geom]

        gdf_ways = gpd.GeoDataFrame(
            data={'osm_id': data_id, 'geometry': data_geom, 'tags': data_tags},
            geometry='geometry', crs='epsg:4326')

        return nodes_taken, gdf_ways

    def _assemble_from_nodes(self, result, nodes_avail):
        """
        pick out those nodes and ways from result instance that belong to
        ways. Assemble ways into gdfs. Keep track of which nodes
        and ways have been "used up" already

        Parameters
        ---------
        result : overpy.Overpass result object
        nodes_avail : list
            node-ids that have not yet been used for relation and way
            construction and are hence available for node constructions

        Returns
        -------
        gdf_nodes : gpd.GeoDataFrame
            gdf with nodes that were assembled from result object
        """
        data_geom = []
        data_id = []
        data_tags = []

        for node in result.nodes:
            if node.id in nodes_avail:
                data_geom.append(shapely.geometry.Point(node.lat, node.lon))
                data_id.append(node.id)
                data_tags.append(node.tags)

        gdf_nodes = gpd.GeoDataFrame(
            data={'osm_id': data_id, 'geometry': data_geom, 'tags': data_tags},
            geometry='geometry', crs='epsg:4326')

        return gdf_nodes

    def _update_availability(self, full_set, to_remove):
        """
        update id availabilities from whole result set after using up some for
        geometry construction
        """
        return [item for item in full_set if item not in to_remove]

    def _assemble_results(self, result, closed_lines_are_polys=True):
        """
        assemble an overpass result object with results, ways, nodes etc. from the
        format-specific structure into one geodataframe

        Parameters
        ---------
        result : overpy.Overpass result object
        closed_lines_are_polys : bool
                whether closed lines are polygons. Default is True

        Returns
        ------
        gdf_results : gpd.GeoDataFrame
            Result-gdf from the overpass query.
        """
        gdf_results = gdf_ways = gdf_nodes = gpd.GeoDataFrame(
            columns=['osm_id', 'geometry', 'tags'],
            geometry='geometry', crs='epsg:4326')
        nodes_avail = result.node_ids
        ways_avail = result.way_ids

        if len(result.relations) > 0:
            nodes_taken, ways_taken, gdf_rels = self._assemble_from_relations(
                result)
            gdf_results = pd.concat([gdf_results, gdf_rels], axis=0)
            nodes_avail = self._update_availability(nodes_avail, nodes_taken)
            ways_avail = self._update_availability(ways_avail, ways_taken)

        if len(ways_avail) > 0:
            nodes_taken, gdf_ways = self._assemble_from_ways(
                result,  ways_avail, closed_lines_are_polys)
            gdf_results = pd.concat([gdf_results, gdf_ways], axis=0)
            nodes_avail = self._update_availability(nodes_avail, nodes_taken)

        if len(nodes_avail) > 0:
            gdf_nodes = self._assemble_from_nodes(result, nodes_avail)
            gdf_results = pd.concat([gdf_results, gdf_nodes], axis=0)

        if len(result.nodes) == 0:
            LOGGER.warning('empty result gdf returned.')

        return gdf_results.reset_index(drop=True)

    def _osm_geoms_to_gis(self, gdf):
        """
        convert lat / lon ordering of OSM convention back to conventional
        GIS ordering (x,y) instead of (lat / lon)

        Parameters
        ----------
        gdf : gpd.GeoDataFrame

        Returns
        -------
        gpd.GeoSeries
            Geometry series with swapped coordinates.
        """

        return gdf.geometry.map(lambda geometry:
                                shapely.ops.transform(
                                    lambda x, y: (y, x), geometry))

    def get_data_overpass(self, closed_lines_are_polys=True):
        """
        wrapper for all helper funcs to get & assemble data
        """

        query_clause = self._overpass_query_string()
        result = self._insistent_osm_api_query(query_clause)
        gdf_result = self._assemble_results(result, closed_lines_are_polys)
        gdf_result = gdf_result.set_geometry(
            self._osm_geoms_to_gis(gdf_result))

        return gdf_result
