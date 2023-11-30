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

Make network base classes (data containers)
"""

import logging
import geopandas as gpd
import igraph as ig
import pandas as pd

LOGGER = logging.getLogger(__name__)


class Network:

    def __init__(self,
                 edges=gpd.GeoDataFrame(),
                 nodes=gpd.GeoDataFrame()):
        """
        initialize a network object given edges and nodes dataframes
        """
        if edges.empty:
            edges = gpd.GeoDataFrame(
                columns=['from_id', 'to_id', 'orig_id', 'geometry'],
                geometry='geometry', crs='EPSG:4326')
        if nodes.empty:
            nodes = gpd.GeoDataFrame(
                columns=['name_id', 'orig_id', 'geometry'],
                geometry='geometry', crs='EPSG:4326')

        if not hasattr(edges, 'orig_id'):
            edges['orig_id'] = range(len(edges))
        if not hasattr(nodes, 'orig_id'):
            nodes['orig_id'] = range(len(nodes))
        if not hasattr(edges, 'osm_id'):
            edges['osm_id'] = range(len(edges))

        self.edges = edges
        self.nodes = nodes

    @classmethod
    def from_nws(cls, networks):
        """
        make one network object out of several network objects
        """
        edges = gpd.GeoDataFrame(
            columns=['from_id', 'to_id', 'orig_id', 'geometry'],
            geometry='geometry', crs='EPSG:4326')
        nodes = gpd.GeoDataFrame(
            columns=['name_id', 'orig_id', 'geometry'],
            geometry='geometry', crs='EPSG:4326')

        id_counter_nodes = 0

        for net in networks:
            edge_gdf = net.edges.reset_index(drop=True)
            node_gdf = net.nodes.reset_index(drop=True)
            edge_gdf['from_id'] = edge_gdf['from_id'] + id_counter_nodes
            edge_gdf['to_id'] = edge_gdf['to_id'] + id_counter_nodes
            node_gdf['name_id'] = range(id_counter_nodes,
                                        id_counter_nodes+len(node_gdf))
            id_counter_nodes += len(node_gdf)
            edges = pd.concat([edges, edge_gdf])
            nodes = pd.concat([nodes, node_gdf])
        edges[['from_id', 'to_id']] = edges[['from_id', 'to_id']].astype(int)

        return Network(edges=edges.reset_index(drop=True),
                       nodes=nodes.reset_index(drop=True))

    @classmethod
    def from_graphs(cls, graphs):
        """
        make one network object out of several graph objects
        """
        graph = ig.Graph(directed=graphs[0].is_directed())
        for gra in graphs:
            graph += gra

        edges = gpd.GeoDataFrame(graph.get_edge_dataframe().rename(
            {'source': 'from_id', 'target': 'to_id'}, axis=1),
            geometry='geometry', crs='EPSG:4326')
        nodes = gpd.GeoDataFrame(graph.get_vertex_dataframe().reset_index(
        ).rename({'vertex ID': 'name_id'}, axis=1),
            geometry='geometry', crs='EPSG:4326')

        return Network(edges=edges, nodes=nodes)
