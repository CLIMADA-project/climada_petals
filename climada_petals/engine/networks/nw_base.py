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
                 edges=None,
                 nodes=None):
        """Initialize a network object from edges and nodes GeoDataFrames

        Creates a Network instance with optional edges (line features) and nodes (point features).
        If empty GeoDataFrames are provided, default structures with required columns are created.
        The method automatically adds 'id' and 'orig_id' columns if they don't exist.

        Parameters
        ----------
        edges : gpd.GeoDataFrame, optional
            GeoDataFrame containing network edges (e.g., roads, power lines).
            Must have columns 'from_id', 'to_id', and 'geometry'.
            Defaults to an empty GeoDataFrame with EPSG:4326 CRS.
        nodes : gpd.GeoDataFrame, optional
            GeoDataFrame containing network nodes (e.g., infrastructure facilities, people).
            Must have columns 'id' and 'geometry'.
            Defaults to an empty GeoDataFrame with EPSG:4326 CRS.

        Attributes
        ----------
        edges : gpd.GeoDataFrame
            Network edges with 'from_id', 'to_id', 'id', 'orig_id', and 'geometry' columns
        nodes : gpd.GeoDataFrame
            Network nodes with 'id', 'orig_id', and 'geometry' columns

        Examples
        --------
        >>> # Create empty network
        >>> network = Network()

        >>> # Create network with data
        >>> edges_gdf = gpd.GeoDataFrame(...)
        >>> nodes_gdf = gpd.GeoDataFrame(...)
        >>> network = Network(edges=edges_gdf, nodes=nodes_gdf)
        """
        if edges is None:
            edges = gpd.GeoDataFrame(
                columns=['from_id', 'to_id', 'id', 'orig_id', 'geometry'],
                geometry='geometry', crs='EPSG:4326')
        if nodes is None:
            nodes = gpd.GeoDataFrame(
                columns=['id', 'orig_id', 'geometry'],
                geometry='geometry', crs='EPSG:4326')

        if 'orig_id' not in edges.columns:
            edges['orig_id'] = range(len(edges))
        if 'orig_id' not in nodes.columns:
            nodes['orig_id'] = range(len(nodes))

        if 'id' not in edges.columns:
            edges['id'] = range(len(edges))
        if 'id' not in nodes.columns:
            nodes['id'] = range(len(nodes))

        self.edges = edges
        self.nodes = nodes

    def reproject(self, crs):
        """Reproject the network to a new coordinate reference system

        Transforms both edges and nodes GeoDataFrames to the specified CRS.
        The operation modifies the network in-place.

        Parameters
        ----------
        crs : str or dict or pyproj.CRS
            Target coordinate reference system. Can be anything accepted by
            :py:meth:`geopandas.GeoDataFrame.to_crs`, such as an EPSG code
            (e.g., 'EPSG:3857'), a PROJ string, or a CRS object.

        Examples
        --------
        >>> network.reproject('EPSG:3857')  # Web Mercator
        >>> network.reproject('EPSG:4326')  # WGS84

        See Also
        --------
        geopandas.GeoDataFrame.to_crs : Underlying reprojection method
        """
        self.nodes = self.nodes.to_crs(crs)
        self.edges = self.edges.to_crs(crs)

    @classmethod
    def from_networks(cls, networks):
        """Combine multiple Network objects into a single unified Network

        Concatenates edges and nodes from multiple networks, automatically adjusting
        vertex IDs to ensure uniqueness. The resulting network maintains all edges
        and nodes from input networks with renumbered identifiers.

        Parameters
        ----------
        networks : list of Network
            List of Network instances to combine. Each network's nodes and edges
            will be concatenated with adjusted IDs to prevent conflicts.

        Returns
        -------
        Network
            A new Network instance containing all edges and nodes from input networks
            with renumbered 'id', 'from_id', and 'to_id' fields.

        Examples
        --------
        >>> road_network = Network(edges=road_edges)
        >>> health_network = Network(nodes=health_nodes)
        >>> people_network = Network(nodes=people_nodes)
        >>> combined = Network.from_networks([road_network, health_network, people_network])

        Notes
        -----
        All input networks must have the same CRS (defaults to EPSG:4326).
        Node IDs are renumbered sequentially across all networks.
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
        """Save network to a compressed zip archive with Feather format

        Exports both nodes and edges GeoDataFrames to Apache Feather format within
        a compressed zip archive. This format provides fast I/O and preserves all
        geographic and attribute data. The archive contains two files:
        ``<savename>_nodes.feather`` and ``<savename>_edges.feather``.

        Parameters
        ----------
        path_save : str or pathlib.Path
            Directory where the zip archive will be created. The directory
            is created if it doesn't exist.
        savename : str
            Base name for the archive and internal files (without extension).
            The final archive will be named ``<savename>.zip``.

        Returns
        -------
        pathlib.Path
            Path to the created zip archive.

        Examples
        --------
        >>> network.save_network_zip('/path/to/save', 'my_network')
        PosixPath('/path/to/save/my_network.zip')

        See Also
        --------
        load_network_zip : Load network from saved archive
        geopandas.GeoDataFrame.to_feather : Underlying serialization method
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
        """Load network from a compressed zip archive

        Reads a network previously saved with :py:meth:`save_network_zip` from
        a zip archive containing Feather files. If the archive or individual files
        are not found, returns a Network with empty GeoDataFrames and prints warnings.

        Parameters
        ----------
        path_load : str or pathlib.Path
            Directory containing the ``<savename>.zip`` archive.
        savename : str
            Base name used when the network was saved (without extension).

        Returns
        -------
        Network
            Network instance with loaded nodes and edges GeoDataFrames.
            If files are missing, the corresponding GeoDataFrames will be empty.

        Examples
        --------
        >>> network = Network.load_network_zip('/path/to/archives', 'my_network')

        See Also
        --------
        save_network_zip : Save network to archive
        geopandas.read_feather : Underlying deserialization method
        """
        path_load = Path(path_load)
        zip_path = path_load / f"{savename}.zip"

        nodes = gpd.GeoDataFrame()
        edges = gpd.GeoDataFrame()

        if not zip_path.exists():
            LOGGER.info("Archive %s not found", zip_path)
            return Network(edges=edges, nodes=nodes)

        with ZipFile(zip_path, mode="r") as zf:
            nodes_name = f"{savename}_nodes.feather"
            edges_name = f"{savename}_edges.feather"

            if nodes_name in zf.namelist():
                with zf.open(nodes_name) as f:
                    nodes = gpd.read_feather(io.BytesIO(f.read()))
            else:
                LOGGER.info("Nodes file %s not found in archive", nodes_name)

            if edges_name in zf.namelist():
                with zf.open(edges_name) as f:
                    edges = gpd.read_feather(io.BytesIO(f.read()))
            else:
                LOGGER.info("Edges file %s not found in archive", edges_name)

        return cls(edges=edges, nodes=nodes)
    @classmethod
    def from_graphs(cls, graphs):
        """Create network from an igraph.Graph object

        Creates a new Network instance from an igraph.Graph object. This is typically used after graph-based
        operations (e.g., adding edges, updating attributes) to reflect changes
        back to the Network structure.

        Parameters
        ----------
        graphs : igraph.Graph
            Graph object from which to extract updated edges and vertices.
            Must have 'geometry' attributes for both edges and vertices.

        Notes
        -----
        This method:
        - Renames graph columns 'source'/'target' to 'from_id'/'to_id' for edges
        - Resets node index and renames 'vertex ID' to 'id'
        - Maintains EPSG:4326 CRS
        - Creates new Network instance with updated edges and nodes

        See Also
        --------
        to_graph : Convert Network to igraph.Graph
        """

        edges = gpd.GeoDataFrame(graphs.get_edge_dataframe().rename(
            {'source': 'from_id', 'target': 'to_id'}, axis=1),
            geometry='geometry', crs='EPSG:4326')
        nodes = graphs.get_vertex_dataframe()
        if 'id' in nodes.columns:
            nodes.pop('id')
        nodes = gpd.GeoDataFrame(nodes.reset_index().rename(
            {'vertex ID': 'id'}, axis=1),
            geometry='geometry', crs='EPSG:4326')

        return cls(edges=edges, nodes=nodes)

    def to_graph(self, directed=False):
        """Convert Network to an igraph.Graph object

        Creates an igraph.Graph representation of the network suitable for
        graph-based analysis and operations. If edges exist, constructs the graph
        from the edge list; otherwise, creates a graph from vertices only.

        Parameters
        ----------
        directed : bool, optional
            Whether to create a directed graph. Defaults to False (undirected).

        Returns
        -------
        igraph.Graph
            Graph object with all vertex and edge attributes preserved.
            Vertices have all columns from the nodes GeoDataFrame.
            Edges have all columns from the edges GeoDataFrame.

        Examples
        --------
        >>> graph = network.to_graph(directed=True)
        >>> graph.vcount()  # Number of vertices
        >>> graph.ecount()  # Number of edges

        See Also
        --------
        from_graph : Create network from an igraph.Graph object
        igraph.Graph.DataFrame : Underlying graph construction method
        """

        if not self.edges.empty:
            graph = self._from_es(
                gdf_edges=self.edges, gdf_nodes=self.nodes, directed=directed)
        else:
            graph = self._from_vs(
                gdf_nodes=self.nodes, directed=directed)
        return graph

    def _remove_namecol(self, gdf_nodes):
        """Remove 'name' column from GeoDataFrame to avoid igraph conflicts

        The igraph library uses 'name' as a reserved attribute for vertices.
        This helper removes any existing 'name' column before graph construction
        to prevent conflicts.

        Parameters
        ----------
        gdf_nodes : gpd.GeoDataFrame or None
            GeoDataFrame potentially containing a 'name' column

        Returns
        -------
        gpd.GeoDataFrame or None
            GeoDataFrame with 'name' column removed if it existed, or None
        """
        if gdf_nodes is not None:
            if 'name' in gdf_nodes.columns:
                gdf_nodes = gdf_nodes.drop('name', axis=1)
        return gdf_nodes

    def _from_es(self, gdf_edges, gdf_nodes=None, directed=False):
        """Construct igraph.Graph from edges with optional nodes

        Parameters
        ----------
        gdf_edges : gpd.GeoDataFrame
            Edge data with 'from_id', 'to_id', and other attributes
        gdf_nodes : gpd.GeoDataFrame, optional
            Node data. If None, nodes are inferred from edge endpoints.
        directed : bool, optional
            Whether to create a directed graph. Defaults to False (undirected).

        Returns
        -------
        igraph.Graph
            Graph constructed from edge list
        """
        return ig.Graph.DataFrame(
            gdf_edges,
            vertices=self._remove_namecol(gdf_nodes),
            directed=directed)

    def _from_vs(self, gdf_nodes, directed=False):
        """Construct igraph.Graph from vertices only (no edges)

        Creates a graph with isolated vertices when no edge information is available.

        Parameters
        ----------
        gdf_nodes : gpd.GeoDataFrame
            Node data with all vertex attributes
        directed : bool, optional
            Whether to create a directed graph. Defaults to False (undirected).

        Returns
        -------
        igraph.Graph
            Graph with n vertices and 0 edges, where n = len(gdf_nodes)
        """
        gdf_nodes = self._remove_namecol(gdf_nodes)
        vertex_attrs = gdf_nodes.to_dict('list')
        return ig.Graph(
            n=len(gdf_nodes),
            vertex_attrs=vertex_attrs,
            directed=directed)

    def initialize_funcstates(self):
        """Initialize functional state attributes for network components

        Sets initial functionality and impact attributes for all edges and nodes.
        This should be called before running cascade simulations or impact assessments.

        The method creates the following attributes:
        - ``func_internal``: Internal functionality state (1 = fully functional)
        - ``func_tot``: Total functionality state (1 = fully functional)
        - ``imp_dir``: Direct impact from hazards (0 = no impact)

        Notes
        -----
        Both edges and nodes are initialized to fully functional (value = 1).
        Direct impacts are initialized to zero.

        Examples
        --------
        >>> network.initialize_funcstates()
        >>> network.nodes['func_tot']  # All values are 1
        >>> network.edges['imp_dir']   # All values are 0
        """
        self.edges[['func_internal','func_tot']] = 1
        self.nodes[['func_internal','func_tot']] = 1
        self.edges['imp_dir'] = 0
        self.nodes['imp_dir'] = 0

    def initialize_capacity(self, dep_table):
        """Initialize capacity attributes for dependency relationships

        Creates capacity attributes for each source-target dependency pair defined
        in the dependency table. Capacity values indicate whether a node provides (+1),
        consumes (-1), or is neutral (0) with respect to a particular resource or service.

        Parameters
        ----------
        dep_table : pd.DataFrame
            Dependency table with at least 'source' and 'target' columns.
            Each row defines a dependency relationship between infrastructure types.

        Notes
        -----
        For each dependency, creates a column ``capacity_{source}_{target}`` where:
        - Source nodes (providers) have capacity = 1
        - Target nodes (consumers) have capacity = -1
        - Other nodes have capacity = 0

        Examples
        --------
        >>> dep_table = pd.DataFrame({
        ...     'source': ['road', 'healthcare'],
        ...     'target': ['people', 'people']
        ... })
        >>> network.initialize_capacity(dep_table)
        >>> network.nodes['capacity_road_people']  # 1 for roads, -1 for people, 0 otherwise

        See Also
        --------
        initialize_supply : Initialize supply attributes for enduser dependencies
        """
        for __, row in dep_table.iterrows():
            source = row['source']
            target = row['target']
            self.nodes[f'capacity_{source}_{target}'] = 0
            self.nodes.loc[self.nodes['ci_type']==f'{source}',f'capacity_{source}_{target}'] = 1
            self.nodes.loc[self.nodes['ci_type']==f'{target}',f'capacity_{source}_{target}'] = -1

    def initialize_supply(self, dep_table):
        """Initialize supply and access state attributes for enduser dependencies

        Creates supply and access state tracking attributes for dependencies where
        the target is an enduser (typically 'people'). These attributes track whether
        endusers have access to services and what their supply level is.

        Parameters
        ----------
        dep_table : pd.DataFrame
            Dependency table with at least 'source', 'target', and 'type_I' columns.
            Only rows where ``type_I == 'enduser'`` are processed.

        Notes
        -----
        For each enduser dependency, creates two attributes:
        - ``access_state_{source}_people``: Access state ("undefined" initially,
          "no base access" for people nodes)
        - ``actual_supply_{source}_people``: Current supply level (0 initially)

        Examples
        --------
        >>> dep_table = pd.DataFrame({
        ...     'source': ['healthcare', 'road'],
        ...     'target': ['people', 'people'],
        ...     'type_I': ['enduser', 'enduser']
        ... })
        >>> network.initialize_supply(dep_table)
        >>> network.nodes['access_state_healthcare_people']  # "no base access" for people
        >>> network.nodes['actual_supply_road_people']  # 0 for all nodes

        See Also
        --------
        initialize_capacity : Initialize capacity attributes
        initialize_funcstates : Initialize functional state attributes
        """
        for __, row in dep_table.loc[dep_table['type_I'] == 'enduser'].iterrows():
            self.nodes[f'access_state_{row["source"]}_people'] = "undefined"
            self.nodes[f'actual_supply_{row["source"]}_people'] = 0
            #self.nodes.loc[self.nodes['ci_type']=='people',f'actual_supply_{row["source"]}_people'] = 1
            self.nodes.loc[self.nodes['ci_type']=='people',f'access_state_{row["source"]}_people'] = "no base access"