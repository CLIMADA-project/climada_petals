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

Make network base classes and Graph base class (data containers)
"""

import logging
import geopandas as gpd
import igraph as ig
import pandas as pd
from zipfile import ZipFile, ZIP_DEFLATED
import io
from pathlib import Path
from climada_petals.engine.networks.nw_utils import infra_plot, population_plot, dep_plot, access_plot

LOGGER = logging.getLogger(__name__)


class Network:

    # map methods
    plot_infra = infra_plot
    plot_pop = population_plot
    plot_dep = dep_plot
    plot_access = access_plot

    def __init__(self,
                 edges=gpd.GeoDataFrame(),
                 nodes=gpd.GeoDataFrame()):
        """
        initialize a network object given edges and nodes dataframes
        """
        if edges.empty:
            edges = gpd.GeoDataFrame(
                columns=['from_id', 'to_id', 'id', 'orig_id', 'geometry'],
                geometry='geometry', crs='EPSG:4326')
        if nodes.empty:
            nodes = gpd.GeoDataFrame(
                columns=['id', 'orig_id', 'geometry'],
                geometry='geometry', crs='EPSG:4326')

        if not hasattr(edges, 'orig_id'):
            edges['orig_id'] = range(len(edges))
        if not hasattr(nodes, 'orig_id'):
            nodes['orig_id'] = range(len(nodes))

        if not hasattr(edges, 'id'):
            edges['id'] = range(len(edges))
        if not hasattr(nodes, 'id'):
            nodes['id'] = range(len(nodes))

        if not hasattr(edges, 'osm_id'):
            edges['osm_id'] = range(len(edges))

        self.edges = edges
        self.nodes = nodes

    def reproject(self, crs):
        """
        Reproject the network from its current crs to a new one.

        Parameters
        ----------
        crs : str
            The new crs to project to.
        """
        self.nodes = self.nodes.to_crs(crs)
        self.edges = self.edges.to_crs(crs)

    @classmethod
    def from_nws(cls, networks):
        """
        make one network object out of several network objects
        """
        edges = gpd.GeoDataFrame(
            columns=['from_id', 'to_id', 'orig_id', 'geometry'],
            geometry='geometry', crs='EPSG:4326')
        nodes = gpd.GeoDataFrame(
            columns=['id', 'orig_id', 'geometry'],
            geometry='geometry', crs='EPSG:4326')

        id_counter_nodes = 0

        for net in networks:
            edge_gdf = net.edges.reset_index(drop=True)
            node_gdf = net.nodes.reset_index(drop=True)
            edge_gdf['from_id'] = edge_gdf['from_id'] + id_counter_nodes
            edge_gdf['to_id'] = edge_gdf['to_id'] + id_counter_nodes
            node_gdf['id'] = range(id_counter_nodes,
                                   id_counter_nodes+len(node_gdf))
            id_counter_nodes += len(node_gdf)
            edges = pd.concat([edges, edge_gdf])
            nodes = pd.concat([nodes, node_gdf])
        edges[['from_id', 'to_id']] = edges[['from_id', 'to_id']].astype(int)

        return Network(edges=edges.reset_index(drop=True),
                       nodes=nodes.reset_index(drop=True))

    def save_network_zip(self, path_save, savename):
        """
        Save a network's nodes and edges into a single .zip archive
        containing Feather files.

        Args:
            network (Network): Network with GeoDataFrames `.nodes` and `.edges`.
            path_save (str | pathlib.Path): Directory to place the archive.
            savename (str): Base name for the archive (without extension).

        Returns:
            pathlib.Path: Path to the created zip archive.
        """
        path_save = Path(path_save)
        path_save.mkdir(parents=True, exist_ok=True)
        zip_path = path_save / f"{savename}.zip"

        with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zf:
            # Save nodes
            if hasattr(self, "nodes") and not self.nodes.empty:
                buf_nodes = io.BytesIO()
                gpd.GeoDataFrame(self.nodes).to_feather(buf_nodes)
                zf.writestr(f"{savename}_nodes.feather", buf_nodes.getvalue())

            # Save edges (optional if present)
            if hasattr(self, "edges") and not self.edges.empty:
                buf_edges = io.BytesIO()
                gpd.GeoDataFrame(self.edges).to_feather(buf_edges)
                zf.writestr(f"{savename}_edges.feather", buf_edges.getvalue())

        return zip_path

    @classmethod
    def load_network_zip(cls, path_load, savename):
        """
        Load a network's nodes and edges from a .zip archive that
        contains Feather files saved by `save_network_zip`.

        Args:
            path_load (str | pathlib.Path): Directory containing `<savename>.zip`.
            savename (str): Base name used when saving (without extension).

        Returns:
            Network: Network with `nodes` and `edges` GeoDataFrames
                     (empty if not found).
        """
        path_load = Path(path_load)
        zip_path = path_load / f"{savename}.zip"

        nodes = gpd.GeoDataFrame()
        edges = gpd.GeoDataFrame()

        if not zip_path.exists():
            print(f"Archive {zip_path} not found")
            return Network(edges=edges, nodes=nodes)

        with ZipFile(zip_path, mode="r") as zf:
            nodes_name = f"{savename}_nodes.feather"
            edges_name = f"{savename}_edges.feather"

            if nodes_name in zf.namelist():
                with zf.open(nodes_name) as f:
                    nodes = gpd.read_feather(io.BytesIO(f.read()))
            else:
                print(f"Nodes file {nodes_name} not found in archive")

            if edges_name in zf.namelist():
                with zf.open(edges_name) as f:
                    edges = gpd.read_feather(io.BytesIO(f.read()))
            else:
                print(f"Edges file {edges_name} not found in archive")

        return cls(edges=edges, nodes=nodes)
    def update_network_from_graphs(self, graphs):
        """
        update network object from several graph objects
        """

        new_edges = gpd.GeoDataFrame(graphs.get_edge_dataframe().rename(
            {'source': 'from_id', 'target': 'to_id'}, axis=1),
            geometry='geometry', crs='EPSG:4326')
        new_nodes = graphs.get_vertex_dataframe()
        if 'id' in new_nodes.columns:
            new_nodes.pop('id')
        new_nodes = gpd.GeoDataFrame(new_nodes.reset_index().rename(
            {'vertex ID': 'id'}, axis=1),
            geometry='geometry', crs='EPSG:4326')


        self.edges = new_edges
        self.nodes = new_nodes

    def to_graph(self, directed=False):
        """
        network : instance of networks.nw_base.Network
        """
        self.directed = directed
        if not self.edges.empty:
            graph = self._from_es(
                gdf_edges=self.edges, gdf_nodes=self.nodes)
        else:
            graph = self._from_vs(
                gdf_nodes=self.nodes)
        return graph
    def _remove_namecol(self, gdf_nodes):
        if gdf_nodes is not None:
            if hasattr(gdf_nodes, 'name'):
                gdf_nodes = gdf_nodes.drop('name', axis=1)
        return gdf_nodes

    def _from_es(self, gdf_edges, gdf_nodes=None):
        return ig.Graph.DataFrame(
            gdf_edges,
            vertices=self._remove_namecol(gdf_nodes),
            directed=self.directed)

    def _from_vs(self, gdf_nodes):
        gdf_nodes = self._remove_namecol(gdf_nodes)
        vertex_attrs = gdf_nodes.to_dict('list')
        return ig.Graph(
            n=len(gdf_nodes),
            vertex_attrs=vertex_attrs,
            directed=self.directed)

    def initialize_funcstates(self):
        """
        Initialize functional states for a new network
        """
        self.edges[['func_internal','func_tot']] = 1
        self.nodes[['func_internal','func_tot']] = 1
        self.edges['imp_dir'] = 0
        self.nodes['imp_dir'] = 0

    def initialize_capacity(self, dep_table):
        for __, row in dep_table.iterrows():
            source = row['source']
            target = row['target']
            self.nodes[f'capacity_{source}_{target}'] = 0
            self.nodes.loc[self.nodes['ci_type']==f'{source}',f'capacity_{source}_{target}'] = 1
            self.nodes.loc[self.nodes['ci_type']==f'{target}',f'capacity_{source}_{target}'] = -1

    def initialize_supply(self, dep_table):
        for __, row in dep_table.loc[dep_table['type_I'] == 'enduser'].iterrows():
            self.nodes[f'access_state_{row["source"]}_people'] = "undefined"
            self.nodes[f'actual_supply_{row["source"]}_people'] = 0
            #self.nodes.loc[self.nodes['ci_type']=='people',f'actual_supply_{row["source"]}_people'] = 1
            self.nodes.loc[self.nodes['ci_type']=='people',f'access_state_{row["source"]}_people'] = "no base access"