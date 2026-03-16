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

"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj


import scipy

from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_utils import make_edge_geometries, _ckdnearest
from climada_petals.engine.networks.nw_preps import reset_ids

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.engine import ImpactCalc
from climada.util import lines_polys_handler as u_lp
from climada.util.constants import ONE_LAT_KM

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel("INFO")


class GraphCalcs:
    """Complete graph-based CI network analysis toolkit

    Provides all graph operations for critical infrastructure (CI) networks,
    including:
    - Network construction (linking, clustering)
    - Dependency setup (shortest paths, friction surface, edge conditions)
    - Cascade analysis (failure propagation, access checking, supply updates)

    Users can call methods in any order for maximum flexibility. For common
    workflows, see NetworkCalcs as a convenience wrapper.

    Examples
    --------
    Direct instantiation for custom workflows:

        from climada_petals.engine.networks.nw_base import Network
        from climada_petals.engine.networks.graph_calcs import GraphCalcs

        network = Network(edges=edges_gdf, nodes=nodes_gdf)
        gc = GraphCalcs(network=network, directed=True)

        # Build graph
        gc.build_graph()

        # Link vertices
        gc.link_vertices_closest_k(
            source_attrs={'ci_type': 'road'},
            target_attrs={'ci_type': 'people'},
            dist_thresh=5000,
            k=1,
            link_attrs={'ci_type': 'road_people'}
        )

        # Setup dependencies
        gc.calc_dependencies(
            source_attrs={'ci_type': 'road'},
            target_attrs={'ci_type': 'people'},
            via_attrs={'ci_type': 'road'},
            link_attrs={'ci_type': 'dependency_road_people'},
            link_condition='distance',
            dist_thresh=10000,
            dur_thresh=np.inf,
            k=1,
            bidir_link=False
        )

        # Check access and cascade
        gc._check_access(row, friction_surf=None, rerouting=True)
    """

    def __init__(self, network, directed=True, friction_surf=None):
        """Create graph-calculation helper for a network

        Parameters
        ----------
        network : Network
            Network to perform graph calculations on.
        directed : bool, optional
            Whether to build a directed igraph representation. Default is ``True``.
        friction_surf : object, optional
            Friction surface used for duration-based linking. Default is ``None``.

        Notes
        -----
        The graph is lazily built and cached on first access via `graph`.
        """
        self.network = network
        self._graph = None
        self.directed = directed
        self.friction_surf = friction_surf

    def build_graph(self):
        """Build and cache an igraph representation of the network

        Returns
        -------
        igraph.Graph
            Graph generated from the current network nodes and edges.

        """
        self._graph = self.network.to_graph(directed=self.directed)
        return self._graph

    @property
    def graph(self):
        if self._graph is None:
            return self.build_graph()
        return self._graph

    def full_reset(self):
        """Clear all graph and cache state

        Notes
        -----
        Use when vertices are added/removed or the graph must be rebuilt.
        """
        self._graph = None

    # =============================================================================
    # Making links
    # =============================================================================

    def link_clusters(
        self,
        dist_thresh=np.inf,
        graph_connectivity_mode="weak",
        link_attrs=None,
        dist_auto_convert=True,
    ):
        """Link connected components into a single graph

        For each component, the method connects the nearest node in that
        component to a node in another component, up to a distance threshold.

        Parameters
        ----------
        dist_thresh : float, optional
            Maximum distance (in meters) to allow cluster linking. Default is ``np.inf``.
        graph_connectivity_mode : str, optional
            Connectivity mode for the graph. Default is ``"weak"``.
        link_attrs : dict, optional
            Edge attributes to set for newly created links.
        dist_auto_convert : bool, optional
            If ``True`` and the network is in a geographic CRS, automatically convert
            the distance threshold from meters to degrees. Default is ``True``.

        Notes
        -----
        Distances are approximated by converting meters to degrees using
        :data:`~climada.util.constants.ONE_LAT_KM`.
        """

        gdf_vs = self.graph.get_vertex_dataframe()
        gdf_vs["membership"] = self.graph.connected_components(
            mode=graph_connectivity_mode
        ).membership

        v_ids_source = []
        v_ids_target = []

        if self.network.crs.is_geographic and dist_auto_convert:
            dist_thresh /= ONE_LAT_KM * 1000
            LOGGER.info(
                "Network is in geographic CRS; automatically converting distance threshold to degrees: %f",
                dist_thresh,
            )

        members = np.unique(gdf_vs["membership"])
        if len(members) <= 1:
            LOGGER.info("Graph is already fully connected; no cluster linking needed.")
            return
        for i in range(len(members) - 1):  # last iteration is redundant
            gdf_a = gdf_vs[gdf_vs["membership"] == members[i]]
            gdf_b = gdf_vs[gdf_vs["membership"] != members[i]]
            if gdf_a.empty or gdf_b.empty:
                continue
            try:
                dists, ix_match = _ckdnearest(gdf_a, gdf_b, dist_thresh=dist_thresh)
                min_dist = min(dists)
                source = gdf_a.iloc[np.where(dists == min_dist)[0]].index[0]
                target = gdf_b.loc[ix_match[np.where(dists == min_dist)[0]]].index[0]
                v_ids_source.append(source)
                v_ids_target.append(target)
            except (IndexError, KeyError):
                LOGGER.info(
                    "No valid link found within distance threshold. Minimum distance: %f",
                    min_dist,
                )
                continue

        if len(v_ids_source) > 0:
            self._edges_from_vlists(v_ids_source, v_ids_target, link_attrs)

    def link_vertices_closest_k(
        self, source_attrs, target_attrs, dist_thresh, k, link_attrs=None, bidir=False
    ):
        """Link each target to its closest ``k`` sources

        Parameters
        ----------
        source_attrs : dict
            Vertex attribute filters for source candidates.
        target_attrs : dict
            Vertex attribute filters for target candidates.
        dist_thresh : float
            Maximum distance (in meters) to allow links.
        k : int
            Number of nearest sources per target.
        link_attrs : dict, optional
            Edge attributes for created links. Default is ``None``.
        bidir : bool, optional
            If ``True``, add reverse links as well. Default is ``False``.
        """

        # select only those for which specified attrs apply
        df_vs_target = GraphCalcs._filter_vertices(self.graph, target_attrs)

        # select only those for which specified attrs apply
        df_vs_source = GraphCalcs._filter_vertices(self.graph, source_attrs)

        if not (df_vs_source.empty or df_vs_target.empty):
            v_ids_source, v_ids_target = self._select_closest_k(
                df_vs_source, df_vs_target, dist_thresh, k, self.network.crs, bidir
            )

            self._edges_from_vlists(v_ids_source, v_ids_target, link_attrs)
        else:
            LOGGER.info(
                "No vertices found matching source %s or target %s; no links created.",
                source_attrs,
                target_attrs,
            )

    def link_vertices_edgecond(self, target_attrs, edge_attrs, link_attrs, bidir=False):
        """Link vertices based on existing edge conditions

        Creates dependency edges between vertices if an existing edge with
        specified attributes already connects them.

        Parameters
        ----------
        target_attrs : dict
            Vertex attribute filters for targets.
        edge_attrs : dict
            Edge attributes that must be present on existing edges.
        link_attrs : dict
            Edge attributes for new dependency links.
        bidir : bool, optional
            If ``True``, add reverse links as well. Default is ``False``.
        """
        df_vs_target = GraphCalcs._filter_vertices(self.graph, target_attrs)

        if df_vs_target.empty:
            LOGGER.info(
                "No vertices found matching target %s; no links created.", target_attrs
            )
            return

        vs_target = self.graph.vs[df_vs_target.index.values]

        pot_edges_ids = [
            self.graph.incident(v_target, mode="all") for v_target in vs_target
        ]
        # flatten nested list
        pot_edges_ids = [item for sublist in pot_edges_ids for item in sublist]

        pot_edges = [
            self.graph.es[item]
            for item in pot_edges_ids
            if all(
                self.graph.es[item][key] == value for key, value in edge_attrs.items()
            )
        ]

        # make sure source are indeed of edge_attrs type and targets of target_attrs type
        sources = []
        targets = []
        for edge in pot_edges:
            source_vx = self.graph.vs[edge.source]
            target_vx = self.graph.vs[edge.target]
            if source_vx["ci_type"] == edge_attrs["ci_type"]:
                sources.append(edge.source)
                targets.append(edge.target)
            elif target_vx["ci_type"] == edge_attrs["ci_type"]:
                sources.append(edge.target)
                targets.append(edge.source)
            else:
                raise ValueError("Edge does not connect correct ci_types!")

        self._edges_from_vlists(sources, targets, link_attrs)
        if bidir:
            self._edges_from_vlists(targets, sources, link_attrs)

    def link_vertices_shortest_paths(
        self,
        source_attrs,
        target_attrs,
        via_attrs,
        link_attrs,
        dist_thresh,
        k,
        criterion="distance",
        bidir=False,
    ):
        """Link targets to sources via shortest paths

        Computes shortest-path distances within a subgraph of allowed vertices
        and creates dependency links where distance constraints are satisfied.

        Parameters
        ----------
        source_attrs : dict
            Vertex attribute filters for source candidates.
        target_attrs : dict
            Vertex attribute filters for target candidates.
        via_attrs : dict
            Edge attribute filters for allowable paths.
        link_attrs : dict
            Edge attributes for created dependency links.
        dist_thresh : float
            Maximum path length to allow links.
        k : int
            Number of links per target.
        criterion : str, optional
            Edge weight attribute used for shortest paths. Default is ``"distance"``.
        bidir : bool, optional
            If ``True``, add reverse links as well. Default is ``False``.
        """

        # subgraph containing only "allowed" elements
        subgraph = self._create_subgraph(source_attrs, target_attrs, via_attrs)

        if len(subgraph.vs) == 0 or len(subgraph.es) == 0:
            LOGGER.info(
                "No vertices or edges found matching source %s, target %s, or via %s attributes; no links created.",
                source_attrs,
                target_attrs,
                via_attrs,
            )
            return

        # mapping from subgraph to graph indices (create lookup array instead of dict for speed)
        subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(self.graph, subgraph)

        # Vectorize the dictionary lookup
        subgraph_ids = np.array(sorted(subgraph_graph_vsdict.keys()))
        graph_ids = np.array([subgraph_graph_vsdict[k] for k in subgraph_ids])

        # select only those for which specified attrs apply
        df_vs = subgraph.get_vertex_dataframe()
        df_vs_target = GraphCalcs._filter_vertices(df_vs, target_attrs)
        df_vs_source = GraphCalcs._filter_vertices(df_vs, source_attrs)

        path_dists = subgraph.distances(
            source=df_vs_source.index.values,
            target=df_vs_target.index.values,
            weights=criterion,
            mode="all",
        )
        path_dists = np.array(path_dists)  # dim: (#sources, #targets)

        if len(path_dists) > 0:
            # Select up to k closest sources per target within dist_thresh
            sorted_indices = np.argsort(path_dists, axis=0)[:k, :]
            k_shortest_dists = path_dists[
                sorted_indices, np.arange(path_dists.shape[1])
            ]
            valid_mask = k_shortest_dists <= dist_thresh

            ix_k, ix_target = np.where(valid_mask)
            ix_source = sorted_indices[ix_k, ix_target]

            # Vectorized re-mapping using numpy arrays instead of list comprehension
            source_subgraph_ids = df_vs_source.index.values[ix_source]
            target_subgraph_ids = df_vs_target.index.values[ix_target]

            v_ids_source = np.array(
                [subgraph_graph_vsdict[int(sid)] for sid in source_subgraph_ids]
            )
            v_ids_target = np.array(
                [subgraph_graph_vsdict[int(tid)] for tid in target_subgraph_ids]
            )

            link_attrs["distance"] = path_dists[(ix_source, ix_target)]

            self._edges_from_vlists(
                v_ids_source.tolist(), v_ids_target.tolist(), link_attrs
            )

            if bidir:
                self._edges_from_vlists(
                    v_ids_target.tolist(), v_ids_source.tolist(), link_attrs
                )

    def link_vertices_friction_surf(
        self,
        source_attrs,
        target_attrs,
        link_attrs,
        dur_thresh,
        k,
        dist_thresh=np.inf,
        bidir=False,
    ):
        """Link vertices using a friction surface duration constraint

        Parameters
        ----------
        source_attrs : dict
            Source vertex attributes for filtering.
        target_attrs : dict
            Target vertex attributes for filtering.
        dur_thresh : float
            Maximum travel duration to allow a link.
        k : int
            Number of nearest sources per target to consider.
        dist_thresh : float, optional
            Maximum geographic distance (meters) for candidate links. Default is np.inf.
        link_name : str, optional
            Edge type name to assign. Default creates ``dependency_{source}_{target}``.
        bidir : bool, optional
            If ``True``, add reverse links as well. Default is ``False``.
        """
        if self.friction_surf is None:
            raise AttributeError(
                "Friction surface is required for this linking method."
                " Please provide a friction surface when initializing GraphCalcs or NetworkCalcs."
            )

        gdf_vs_target = GraphCalcs._filter_vertices(self.graph, target_attrs)
        gdf_vs_source = GraphCalcs._filter_vertices(self.graph, source_attrs)

        if not (gdf_vs_source.empty or gdf_vs_target.empty):
            v_ids_source, v_ids_target = self._select_closest_k(
                gdf_vs_source, gdf_vs_target, dist_thresh, k, self.network.crs, bidir
            )

            edge_geoms = make_edge_geometries(
                self.graph.vs[v_ids_source]["geometry"],
                self.graph.vs[v_ids_target]["geometry"],
            )

            friction = self._calc_friction(edge_geoms, self.friction_surf)
            v_ids_source = np.array(v_ids_source)[friction < dur_thresh]
            v_ids_target = np.array(v_ids_target)[friction < dur_thresh]

            self._edges_from_vlists(
                v_ids_source.tolist(), v_ids_target.tolist(), link_attrs
            )
        else:
            LOGGER.info(
                "No vertices found matching source %s or target %s; no links created.",
                source_attrs["ci_type"],
                target_attrs["ci_type"],
            )

    # =============================================================================
    # Helper funcs for making links
    # =============================================================================
    @staticmethod
    def _filter_vertices(graph_or_df, attr_dict):
        """Filter vertices by attribute values

        Parameters
        ----------
        graph_or_df : igraph.Graph or pd.DataFrame
            Graph whose vertices to filter, or a pre-built vertex
            dataframe (from ``graph.get_vertex_dataframe()``).
        attr_dict : dict
            Attribute filters as ``{key: value}`` pairs.

        Returns
        -------
        pd.DataFrame
            Vertex dataframe subset matching all filters.
        """
        if isinstance(graph_or_df, pd.DataFrame):
            df_vs = graph_or_df
        else:
            df_vs = graph_or_df.get_vertex_dataframe()

        # Apply filters using numpy operations for speed
        mask = np.ones(len(df_vs), dtype=bool)
        for key, value in attr_dict.items():
            mask &= (df_vs[key] == value).values
        return df_vs[mask]

    @staticmethod
    def _filter_edges(graph, attr_dict):
        """Filter edges by attribute values

        Parameters
        ----------
        graph : igraph.Graph
            Graph to filter.
        attr_dict : dict
            Attribute filters as ``{key: value}`` pairs.

        Returns
        -------
        pd.DataFrame
            Edge dataframe subset matching all filters.
        """

        df_es = graph.get_edge_dataframe()
        for key, value in attr_dict.items():
            df_es = df_es[df_es[key] == value]
        return df_es

    def _edges_from_vlists(self, v_ids_source, v_ids_target, link_attrs=None):
        """Create edges between source and target vertex lists

        Adds edge geometries and distances if missing in ``link_attrs``.

        Parameters
        ----------
        v_ids_source : list
            Source vertex indices in the graph.
        v_ids_target : list
            Target vertex indices in the graph.
        link_attrs : dict, optional
            Edge attributes for new links.
        """

        # Early return if no edges to add
        if len(v_ids_source) == 0 or len(v_ids_target) == 0:
            return

        pairs = list(zip(v_ids_source, v_ids_target))

        link_attrs["geometry"] = make_edge_geometries(
            self.graph.vs[v_ids_source]["geometry"],
            self.graph.vs[v_ids_target]["geometry"],
        )

        if "distance" not in link_attrs.keys():
            LOGGER.info("Adding edge distances for new links.")
            if self.network.crs is None or self.network.crs.is_geographic:
                geod = pyproj.Geod(ellps="WGS84")
                distances = [
                    geod.geometry_length(edge_geom)
                    for edge_geom in link_attrs["geometry"]
                ]
            else:
                distances = [edge_geom.length for edge_geom in link_attrs["geometry"]]
            link_attrs["distance"] = distances

        self.graph.add_edges(pairs, attributes=link_attrs)

    @staticmethod
    def _select_closest_k(
        gdf_vs_source,
        gdf_vs_target,
        dist_thresh,
        k,
        crs,
        bidir=False,
        dist_auto_convert=True,
    ):
        """Select closest source vertices for each target

        Parameters
        ----------
        gdf_vs_source : pd.DataFrame
            Source vertex dataframe.
        gdf_vs_target : pd.DataFrame
            Target vertex dataframe.
        dist_thresh : float
            Maximum distance (in meters) for matches.
        k : int
            Number of closest sources per target.
        crs : pyproj.CRS
            Coordinate reference system for the data.
        bidir : bool, optional
            If ``True``, append reverse links. Default is ``False``.
        dist_auto_convert : bool, optional
            If ``True`` and the network is in a geographic CRS, automatically convert
            the distance threshold from meters to degrees. Default is ``True``.

        Returns
        -------
        list, list
            Source and target vertex indices for links.
        """

        if crs.is_geographic and dist_auto_convert:
            dist_thresh /= ONE_LAT_KM * 1000
            LOGGER.info(
                "Network is in geographic CRS; automatically converting distance threshold to degrees: %f",
                dist_thresh,
            )

        # index matches, in format (#target vs, k). nans for those without matches
        __, ix_matches = _ckdnearest(
            gdf_vs_target, gdf_vs_source, k=k, dist_thresh=dist_thresh
        )
        # broadcast target indices to same format
        ix_matches = ix_matches.flatten()
        v_ids_target = np.array(
            np.broadcast_to(
                np.array([gdf_vs_target.id]).T, (len(gdf_vs_target), k)
            ).flatten()
        )
        v_ids_target = v_ids_target[~np.isnan(ix_matches)]
        v_ids_source = np.array(gdf_vs_source.loc[ix_matches[~np.isnan(ix_matches)]].id)

        if bidir:
            v_ids_source_orig = v_ids_source.copy()
            v_ids_target_orig = v_ids_target.copy()
            v_ids_source = np.append(v_ids_source_orig, v_ids_target_orig)
            v_ids_target = np.append(v_ids_target_orig, v_ids_source_orig)

        return list(v_ids_source), list(v_ids_target)

    def _create_subgraph(self, source_attrs, target_attrs, via_attrs):
        """Create a subgraph with source, target, and via elements

        Parameters
        ----------
        source_attrs : dict
            Vertex attribute filters for source candidates.
        target_attrs : dict
            Vertex attribute filters for target candidates.
        via_attrs : dict
            Edge attribute filters for allowable paths.

        Returns
        -------
        igraph.Graph
            Induced subgraph containing only relevant vertices and via edges.

        See Also
        --------
        link_vertices_shortest_paths : Link using shortest paths in a subgraph
        """

        # Get vertex dataframe once and reuse for all filters
        df_vs = self.graph.get_vertex_dataframe()

        # select only those for which specified attrs apply
        df_vs_source = GraphCalcs._filter_vertices(df_vs, source_attrs)
        df_vs_target = GraphCalcs._filter_vertices(df_vs, target_attrs)
        df_vs_via = GraphCalcs._filter_vertices(df_vs, via_attrs)

        # Use efficient numpy operations instead of list concatenation
        vs_keep = np.unique(
            np.concatenate(
                (
                    df_vs_source.index.values,
                    df_vs_target.index.values,
                    df_vs_via.index.values,
                )
            )
        ).astype(int)

        # vs_keep has indexing of original graph, subgraph has new indexing. There
        # is no way of keeping track of the re-ordering, other than to have a named
        # attribute!
        self.graph.vs["orig_id"] = range(len(self.graph.vs))
        self.graph.es["orig_id"] = range(len(self.graph.es))
        subgraph = self.graph.induced_subgraph(vs_keep)

        # delete remaining edges that have wrong attributes
        df_es_via = GraphCalcs._filter_edges(subgraph, via_attrs)

        correct_edges = df_es_via.index.values
        wrong_edges = set(range(len(subgraph.es))).difference(set(correct_edges))

        subgraph.delete_edges(wrong_edges)

        return subgraph

    @staticmethod
    def _get_subgraph2graph_vsdict(graph, subgraph):
        """Map subgraph vertex indices to original graph indices

        Parameters
        ----------
        graph : igraph.Graph
            Original graph with ``orig_id`` attributes.
        subgraph : igraph.Graph
            Induced subgraph built from ``graph``.

        Returns
        -------
        dict
            Mapping ``{subgraph_index: graph_index}``.
        """
        # Vectorized attribute access
        subgraph_vs_indices = np.arange(len(subgraph.vs))
        subgraph_orig_ids = np.array(subgraph.vs.get_attribute_values("orig_id"))

        graph_vs_indices = np.arange(len(graph.vs))
        graph_orig_ids = np.array(graph.vs.get_attribute_values("orig_id"))

        # Use numpy argsort for faster mapping
        sort_idx = np.argsort(graph_orig_ids)
        graph_orig_ids_sorted = graph_orig_ids[sort_idx]
        graph_vs_indices_sorted = graph_vs_indices[sort_idx]

        # Use searchsorted to find indices - O(log n) instead of O(n)
        positions = np.searchsorted(graph_orig_ids_sorted, subgraph_orig_ids)
        result = {}
        for i, orig_id in enumerate(subgraph_orig_ids):
            pos = positions[i]  # Use precomputed position
            if (
                pos < len(graph_orig_ids_sorted)
                and graph_orig_ids_sorted[pos] == orig_id
            ):
                result[subgraph_vs_indices[i]] = graph_vs_indices_sorted[pos]

        return result

    @staticmethod
    def _get_subgraph2graph_esdict(graph, subgraph):
        """Map subgraph edge indices to original graph indices

        Parameters
        ----------
        graph : igraph.Graph
            Original graph with ``orig_id`` attributes.
        subgraph : igraph.Graph
            Induced subgraph built from ``graph``.

        Returns
        -------
        dict
            Mapping ``{subgraph_index: graph_index}``.
        """
        # Vectorized attribute access
        subgraph_es_indices = np.arange(len(subgraph.es))
        subgraph_orig_ids = np.array(subgraph.es.get_attribute_values("orig_id"))

        graph_es_indices = np.arange(len(graph.es))
        graph_orig_ids = np.array(graph.es.get_attribute_values("orig_id"))

        # Use numpy argsort for faster mapping
        sort_idx = np.argsort(graph_orig_ids)
        graph_orig_ids_sorted = graph_orig_ids[sort_idx]
        graph_es_indices_sorted = graph_es_indices[sort_idx]

        # Use searchsorted for O(log n) lookup
        positions = np.searchsorted(graph_orig_ids_sorted, subgraph_orig_ids)
        result = {}
        for i, orig_id in enumerate(subgraph_orig_ids):
            pos = positions[i]  # Use precomputed position
            if (
                pos < len(graph_orig_ids_sorted)
                and graph_orig_ids_sorted[pos] == orig_id
            ):
                result[subgraph_es_indices[i]] = graph_es_indices_sorted[pos]

        return result

    @staticmethod
    def _calc_friction(edge_geoms, friction_surf):
        """Compute travel duration along edges using a friction surface

        Parameters
        ----------
        edge_geoms : list
            Edge geometries (LineString) to evaluate.
        friction_surf : object
            Friction surface hazard-like object used for impact calculation.

        Returns
        -------
        np.ndarray
            Aggregated duration per edge geometry.
        """

        # define mapping as impact function.
        impf_fric = ImpactFunc()
        impf_fric.id = 1
        impf_fric.haz_type = ""
        impf_fric.name = "friction surface mapping"
        impf_fric.intensity_unit = "min/m"
        impf_fric.intensity = np.linspace(
            friction_surf.intensity.data.min(),
            friction_surf.intensity.data.max(),
            num=500,
        )
        impf_fric.mdd = np.linspace(
            friction_surf.intensity.data.min(),
            friction_surf.intensity.data.max(),
            num=500,
        )
        impf_fric.paa = np.sort(np.linspace(1, 1, num=500))
        impf_fric.check()
        impf_set = ImpactFuncSet()
        impf_set.append(impf_fric)

        # perform impact calc for mapping.
        exp_links = Exposures(gpd.GeoDataFrame({"geometry": edge_geoms}))
        exp_links.gdf["impf_"] = 1
        # exp_links.gdf["geometry_orig"] = exp_links.gdf.geometry

        # step-by-step to avoid 0 duration sections
        exp_pnt = u_lp.exp_geom_to_pnt(
            exp_links,
            res=100,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=100,
        )

        impact_pnt = ImpactCalc(exp_pnt, impf_set, friction_surf).impact(save_mat=True)
        if impact_pnt.imp_mat.size < len(exp_pnt.gdf):
            imp_arry = np.array(impact_pnt.imp_mat.todense()).flatten()
            imp_arry[imp_arry == 0] = (
                exp_pnt.gdf.value[imp_arry == 0] * friction_surf.intensity.data.min()
            )
            impact_pnt.imp_mat = scipy.sparse.csr_matrix(imp_arry)

        friction = u_lp.impact_pnt_agg(impact_pnt, exp_pnt.gdf, u_lp.AggMethod.SUM)

        return friction.eai_exp

    def calc_dependencies(
        self,
        source_attrs,
        target_attrs,
        via_attrs,
        link_attrs,
        link_condition,
        dist_thresh,
        dur_thresh,
        k,
        bidir_link,
    ):
        """Dispatch dependency creation based on link condition

        Parameters
        ----------
        source_attrs : dict
            Source vertex filters.
        target_attrs : dict
            Target vertex filters.
        via_attrs : dict
            Via edge filters.
        link_attrs : dict
            Attributes assigned to new dependency edges.
        link_condition : str
            Condition type (e.g., ``"distance"``, ``"duration"``, ``"edgecond"``).
        dist_thresh : float
            Threshold for distance or duration (depending on condition).
        dur_thresh : float
            Threshold for duration (depending on condition).
        k : int
            Number of shortest paths to consider.
        bidir_link : bool
            Whether to add reverse links.
        """
        if "ci_type" not in link_attrs:
            link_attrs["ci_type"] = (
                f"dependency_{source_attrs['ci_type']}_{target_attrs['ci_type']}"
            )
            LOGGER.info(
                "No ci_type specified for links; defaulting to %s",
                link_attrs["ci_type"],
            )
        if "distance" in link_condition:
            self.link_vertices_shortest_paths(
                source_attrs=source_attrs,
                target_attrs=target_attrs,
                via_attrs=via_attrs,
                link_attrs=link_attrs,
                dist_thresh=dist_thresh,
                k=k,
                bidir=bidir_link,
            )
        elif "duration" in link_condition:
            self.link_vertices_friction_surf(
                source_attrs=source_attrs,
                target_attrs=target_attrs,
                link_attrs=link_attrs,
                dist_thresh=dist_thresh,
                dur_thresh=dur_thresh,
                k=k,
                bidir=bidir_link,
            )
        elif "edgecond" in link_condition:
            self.link_vertices_edgecond(
                target_attrs=target_attrs,
                edge_attrs=source_attrs,
                link_attrs=link_attrs,
                bidir=bidir_link,
            )
        else:
            raise NotImplementedError

    # =============================================================================
    # Propagation functions
    # =============================================================================

    def _propagate_check_fail(self, source, target, type_I, thresh_func):
        """Propagate functional failures for a source-target dependency

        Parameters
        ----------
        source : str
            Source infrastructure type.
        target : str
            Target infrastructure type.
        type_I : str
            Type of dependency (enduser vs functional).
        thresh_func : float
            Functional threshold for target nodes.

        Notes
        -----
        Updates ``func_tot`` or ``actual_supply`` on target nodes in-place.
        """
        # Vectorized vertex selection by ci_type
        ci_types = np.array(self.graph.vs["ci_type"])
        source_ids = np.where(ci_types == source)[0]
        target_ids = np.where(ci_types == target)[0]
        all_ids = np.concatenate([source_ids, target_ids])
        all_ids_list = all_ids.tolist()

        # Use direct adjacency matrix slicing instead of subgraph
        adj_full = self.graph.get_adjacency_sparse()
        adj_sub = adj_full[all_ids, :][:, all_ids]

        # Vectorized capacity & func_tot reads via batch attribute access
        func_tots = np.array(self.graph.vs[all_ids_list]["func_tot"], dtype=float)
        capacities = np.array(
            self.graph.vs[all_ids_list][f"capacity_{source}_{target}"], dtype=float
        )

        func_capa = func_tots * capacities

        # Matrix multiplication using sparse operations
        capa_rec = scipy.sparse.csr_matrix(func_capa).dot(adj_sub).toarray().squeeze()
        if capa_rec.ndim == 0:
            capa_rec = np.array([capa_rec])

        # Vectorized threshold check
        is_target = np.array(self.graph.vs[all_ids_list]["ci_type"]) == target
        func_thresh = np.where(is_target, thresh_func, 0)
        capa_suff = (capa_rec >= func_thresh).astype(int)

        # Extract target-only arrays for batch updates
        target_graph_ids = all_ids[is_target].tolist()
        target_capa_suff = capa_suff[is_target]

        # Batch update graph attributes using vectorized operations
        supply_attr = f"actual_supply_{source}_{target}"
        access_attr = f"access_state_{source}_{target}"

        if type_I == "enduser":
            # Read previous supply in batch
            prev_supply = np.array(
                self.graph.vs[target_graph_ids][supply_attr], dtype=float
            )

            # Write new supply in batch
            self.graph.vs[target_graph_ids][supply_attr] = target_capa_suff.tolist()

            # Determine access states vectorized:
            # - sufficient capacity now -> 'access undisrupted'
            # - insufficient now, but had supply before -> 'access disrupted'
            # - insufficient now, never had supply -> 'no base access'
            access_states = np.where(
                target_capa_suff >= 1,
                "access undisrupted",
                np.where(
                    prev_supply >= thresh_func, "access disrupted", "no base access"
                ),
            )
            self.graph.vs[target_graph_ids][access_attr] = access_states.tolist()
        else:
            # For functional dependencies: func_tot = min(capa_suff, orig_func)
            orig_func = np.array(
                self.graph.vs[target_graph_ids]["func_tot"], dtype=float
            )
            new_func = np.minimum(target_capa_suff, orig_func)
            self.graph.vs[target_graph_ids]["func_tot"] = new_func.tolist()

    def funcstates_sum(self):
        """Sum functional states across vertices and edges

        Returns
        -------
        tuple
            ``(sum_vertices, sum_edges)`` of ``func_tot``.
        """
        return (
            sum(self.graph.vs.get_attribute_values("func_tot")),
            sum(self.graph.es.get_attribute_values("func_tot")),
        )

    def update_internal_dependencies(self, p_source, p_sink, source_var, demand_var):
        """Update internal dependencies for networked CI types

        Parameters
        ----------
        p_source : str
            Power source type (e.g., ``"power_plant"``).
        p_sink : str
            Power sink type (e.g., ``"power_line"``).
        source_var : str
            Attribute name for source generation.
        demand_var : str
            Attribute name for demand consumption.
        """

        # specifically for roads: if edge is dysfunctional, render its target vertex dysfunctional
        if {"road"}.issubset(set(self.graph.vs["ci_type"])):
            LOGGER.info("Updating roads")
            sources_targets_dys = [
                [edge.source, edge.target]
                for edge in self.graph.es.select(ci_type="road").select(func_tot_eq=0)
            ]
            sources_targets_dys = (
                np.array(sources_targets_dys).flatten().tolist()
            )  # flatten array
            self.graph.vs.select(sources_targets_dys).select(ci_type="road")[
                "func_tot"
            ] = 0

        # specifically for powerlines: check power clusters
        if {p_source, p_sink}.issubset(set(self.graph.vs["ci_type"])):
            LOGGER.info("Updating power clusters")
            # For another version using pandapower, see nw_utils.py
            # Since powerlines are directed in a directed graph,
            # make sure 'reverse' lines are also down

            edges_dys = self.graph.es.select(ci_type="power_line").select(func_tot_eq=0)
            reverse_edges = [(edge.target, edge.source) for edge in edges_dys]
            eids = self.graph.get_eids(
                pairs=reverse_edges, path=None, directed=True, error=True
            )
            self.graph.es[eids]["func_tot"] = 0
            LOGGER.info(
                "Using updated power line algorithm: dysfunc edges before: \
                  %i, after: %i",
                len(edges_dys),
                len(self.graph.es.select(ci_type="power_line").select(func_tot_eq=0)),
            )
            self.powercap_from_clusters(
                p_source=p_source,
                p_sink=p_sink,
                demand_ci="people",
                source_var=source_var,
                demand_var=demand_var,
            )

    def update_functional_dependencies(self, df_dependencies):
        """Update functional CI-to-CI dependencies

        Parameters
        ----------
        df_dependencies : pd.DataFrame
            Dependency table with ``type_I == 'functional'``.
        """

        for __, row in df_dependencies[
            df_dependencies["type_I"] == "functional"
        ].iterrows():

            if row.access_cnstr:
                # TODO: Implement
                LOGGER.warning(
                    "Road access condition for CI-CI deps not yet implemented"
                )

            self._propagate_check_fail(
                row.source, row.target, row.type_I, row.thresh_func
            )

    def update_enduser_dependencies(
        self,
        df_dependencies,
        friction_surf,
        access_check_method="routing",
        rerouting=True,
    ):
        """Update end-user dependencies for the cascade

        Parameters
        ----------
        df_dependencies : pd.DataFrame
            Dependency table with ``type_I == 'enduser'``.
        access_check_method : str
            Method to check access either "routing" or "propagation". Default is ``"routing"``.,
        friction_surf : object or None
            Friction surface used for routing when applicable.
        rerouting : bool, optional
            Whether to allow rerouting to alternative sources. Default is ``True``.
        """

        for __, row in df_dependencies[
            df_dependencies["type_I"] == "enduser"
        ].iterrows():

            if access_check_method == "routing":
                self._check_access(row, friction_surf, rerouting=rerouting)
            elif access_check_method == "propagation":
                if row.access_cnstr:
                    LOGGER.warning(
                        "Propagation method does not account for via-link "
                        "access constraints (access_cnstr=True) for "
                        "%s->%s. Road disruptions between "
                        "source and target will not be detected. "
                        'Use access_check_method="routing" for accurate results.',
                        row.source,
                        row.target,
                    )
                self._propagate_check_fail(
                    row.source, row.target, row.type_I, row.thresh_func
                )
            else:
                raise ValueError("Invalid access check method specified!")

    def _get_former_access_info(self, dependency_name):
        """Retrieve former access status for a dependency

        Parameters
        ----------
        dependency_name : str
            Name of dependency edges to check.

        Returns
        -------
        tuple
            ``(es_access_base, ppl_former_access, ppl_former_access_source_failed)``.
        """
        es_access_base = self.graph.es.select(ci_type=dependency_name)
        ppl_former_access = [edge.target for edge in es_access_base]
        ppl_former_access_source_failed = [
            edge.target
            for edge in es_access_base
            if self.graph.vs[edge.source]["func_tot"] < 1
        ]
        return es_access_base, ppl_former_access, ppl_former_access_source_failed

    def _recompute_dependencies_with_rerouting(self, row, dependency_name):
        """Recompute dependencies with rerouting allowed

        Parameters
        ----------
        row : pd.Series
            Dependency configuration row.
        dependency_name : str
            Name of dependency edges.

        Returns
        -------
        tuple
            ``(es_access_new, ppl_new_access, ppl_access_all_via)``.
        """
        # Delete existing dependencies to recompute from scratch
        self.graph.delete_edges(ci_type=dependency_name)

        # If access_cnstr is False, compute dependencies without requiring functional via edges
        via_attrs_dict = {"ci_type": row["via_link"]}
        if row.access_cnstr:
            via_attrs_dict["func_tot"] = 1

        self.calc_dependencies(
            source_attrs={"ci_type": row["source"], "func_tot": 1},
            target_attrs={"ci_type": row["target"]},
            via_attrs=via_attrs_dict,
            link_attrs={"ci_type": dependency_name},
            link_condition=row["link_condition"],
            dist_thresh=row["thresh_dist"],
            dur_thresh=row["thresh_dur"],
            k=row["n_links"],
            bidir_link=row["bidir_link"],
        )

        # Check if could have access if links were not broken
        if row.access_cnstr:
            # Compute dependencies without requiring functional via edges to identify
            # people who could have access if via links were functional
            self.calc_dependencies(
                source_attrs={"ci_type": row["source"], "func_tot": 1},
                target_attrs={"ci_type": row["target"]},
                via_attrs={"ci_type": row["via_link"]},  # No func_tot requirement
                link_attrs={"ci_type": "new_" + dependency_name},
                link_condition=row["link_condition"],
                dist_thresh=row["thresh_dist"],
                dur_thresh=row["thresh_dur"],
                k=row["n_links"],
                bidir_link=row["bidir_link"],
            )

            # People having access regardless of the state of the via link
            ppl_access_all_via = [
                edge.target
                for edge in self.graph.es.select(ci_type="new_" + dependency_name)
            ]
            # Delete temporary edges
            self.graph.delete_edges(ci_type="new_" + dependency_name)
        else:
            ppl_access_all_via = []

        # Re-query after all edge deletions to get fresh edge objects
        es_access_new = self.graph.es.select(ci_type=dependency_name)
        ppl_new_access = [edge.target for edge in es_access_new]

        if not row.access_cnstr:
            ppl_access_all_via = ppl_new_access

        return ppl_new_access, ppl_access_all_via

    def _validate_dependency_paths(self, edge_pairs, row, subgraph):
        """Validate which dependency edges still have valid paths

        Parameters
        ----------
        edge_pairs : list of tuple
            ``(source, target)`` vertex index pairs to validate.
        row : pd.Series
            Dependency configuration row.
        graph_subgraph_vsdict : dict
            Mapping from graph vertex ids to subgraph vertex ids.
        subgraph : igraph.Graph
            Subgraph containing only source, target, and via vertices.

        Returns
        -------
        list of tuple
            ``(source, target)`` pairs that still have valid paths.
        """
        # map graph vertex ids to subgraph vertex ids for quick lookup
        # Map from original graph ids to subgraph ids
        subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(self.graph, subgraph)
        graph_subgraph_vsdict = {
            int(v): int(k) for k, v in subgraph_graph_vsdict.items()
        }

        pairs_to_keep = []
        for source, target in edge_pairs:
            source_sub = graph_subgraph_vsdict.get(source)
            target_sub = graph_subgraph_vsdict.get(target)

            if source_sub is not None and target_sub is not None:
                try:
                    dist = subgraph.distances(
                        source=source_sub,
                        target=target_sub,
                        weights="distance",
                        mode="all",  # Treat as undirected for connectivity check
                    )
                    # If path exists and is within threshold, keep the edge
                    if dist[0][0] < row["thresh_dist"]:
                        pairs_to_keep.append((source, target))
                except (IndexError, ValueError):
                    # No path exists, edge should be removed
                    pass

        return pairs_to_keep

    def _validate_dependencies_without_rerouting(
        self, row, dependency_name, es_access_base
    ):
        """Validate dependencies without rerouting

        Parameters
        ----------
        row : pd.Series
            Dependency configuration row.
        dependency_name : str
            Name of dependency edges.
        es_access_base : list
            Base access edges.
        ppl_former_access : list
            People who had former access.

        Returns
        -------
        tuple
            ``(es_access_new, ppl_new_access, ppl_access_all_via)``.
        """

        # Extract all data from edge objects BEFORE any deletions,
        # because igraph invalidates all edge objects when edges are deleted.
        func_source_pairs = []
        failed_edge_indices = []
        for edge in es_access_base:
            src, tgt = edge.source, edge.target
            if self.graph.vs[src]["func_tot"] >= 1:
                func_source_pairs.append((src, tgt))
            else:
                failed_edge_indices.append(edge.index)

        # People having access regardless of the state of the via link
        ppl_access_all_via = [tgt for _, tgt in func_source_pairs]

        # Remove dependency edges from failed sources (they can't provide access)
        if failed_edge_indices:
            self.graph.delete_edges(failed_edge_indices)

        if row.access_cnstr:
            # Need to check if via edges used in former dependencies have failed
            # Keep edges where source is functional and path through functional via edges exists

            if len(func_source_pairs) > 0:
                # create subgraph
                subgraph = self._create_subgraph(
                    source_attrs={"ci_type": row["source"], "func_tot": 1},
                    target_attrs={"ci_type": row["target"]},
                    via_attrs={"ci_type": row["via_link"], "func_tot": 1},
                )

                # Check which former dependency edges still have valid paths
                pairs_to_keep = self._validate_dependency_paths(
                    func_source_pairs, row, subgraph
                )
                pairs_to_keep_set = set(pairs_to_keep)

                # Find and delete dependency edges that no longer have valid paths
                pairs_to_remove = [
                    pair for pair in func_source_pairs if pair not in pairs_to_keep_set
                ]
                if pairs_to_remove:
                    # Find current edge indices by source-target lookup
                    eids_to_remove = self.graph.get_eids(
                        pairs=pairs_to_remove, directed=True, error=False
                    )
                    eids_to_remove = [eid for eid in eids_to_remove if eid >= 0]
                    if eids_to_remove:
                        self.graph.delete_edges(eids_to_remove)

            # Get updated list of access edges after validation
            es_access_new = self.graph.es.select(ci_type=dependency_name)
            ppl_new_access = [edge.target for edge in es_access_new]
        else:
            # No access constraints, so no need to check via edges
            ppl_new_access = ppl_access_all_via

        return ppl_new_access, ppl_access_all_via

    def _mark_access_states_and_supply(
        self,
        row,
        ppl_former_access,
        ppl_former_access_source_failed,
        ppl_access_all_via,
        ppl_new_access,
    ):
        """Mark access states and supply for enduser nodes

        Parameters
        ----------
        row : pd.Series
            Dependency configuration row.
        ppl_former_access : list
            Enduser who had former access.
        ppl_former_access_source_failed : list
            Enduser whose former source failed.
        ppl_access_all_via : list
            Enduser who could have access if via links were functional.
        ppl_new_access : list
            Enduser who have current access.
        """
        # Convert to sets for O(1) membership checking
        ppl_former_access_source_failed_set = set(ppl_former_access_source_failed)
        ppl_new_access_set = set(ppl_new_access)

        # If init source was failed but ppl still have access, then they have access to a new source
        ppl_access_new_source = [
            ppl_node
            for ppl_node in ppl_new_access
            if ppl_node in ppl_former_access_source_failed_set
        ]
        if ppl_access_new_source:
            self.graph.vs[ppl_access_new_source][
                f"access_state_{row.source}_{row.target}"
            ] = "access new source"

        # If people have access only when no functional via is required, then access is disrupted via
        ppl_access_broken_via = [
            ppl_node
            for ppl_node in ppl_access_all_via
            if ppl_node not in ppl_new_access_set
        ]
        if ppl_access_broken_via:
            self.graph.vs[ppl_access_broken_via][
                f"access_state_{row.source}_{row.target}"
            ] = "access disrupted via"

        # If endusers do not have access due to via constraints, then the access is disrupted at source
        ppl_access_broken_via_set = set(ppl_access_broken_via)
        ppl_no_reaccess = [
            ppl_node
            for ppl_node in ppl_former_access
            if (
                ppl_node not in ppl_new_access_set
                and ppl_node not in ppl_access_broken_via_set
            )
        ]
        if ppl_no_reaccess:
            self.graph.vs[ppl_no_reaccess][
                f"access_state_{row.source}_{row.target}"
            ] = "access disrupted source"

        # Remaining accesses are undisrupted
        ppl_access_undisrupted = [
            ppl_node
            for ppl_node in ppl_new_access
            if ppl_node not in ppl_former_access_source_failed_set
        ]
        if ppl_access_undisrupted:
            self.graph.vs[ppl_access_undisrupted][
                f"access_state_{row.source}_{row.target}"
            ] = "access undisrupted"

        # Add boolean array of actual supply
        # Endusers with access get supply=1 (includes undisrupted, all_via, and new_source)
        # Use set to avoid duplicates
        ppl_with_supply = list(
            set(ppl_access_undisrupted + ppl_access_all_via + ppl_access_new_source)
        )
        if ppl_with_supply:
            self.graph.vs[ppl_with_supply][
                f"actual_supply_{row.source}_{row.target}"
            ] = 1

        ppl_without_supply = ppl_no_reaccess + ppl_access_broken_via
        if ppl_without_supply:
            self.graph.vs[ppl_without_supply][
                f"actual_supply_{row.source}_{row.target}"
            ] = 0

    def _check_access(self, row, friction_surf, rerouting=True, initial=False):
        """Check and update access states for end-user dependencies

        Parameters
        ----------
        row : pd.Series
            Dependency configuration row containing source, target, and via settings.
        friction_surf : object or None
            Friction surface for duration-based routing (if used).
        rerouting : bool, optional
            Whether to allow rerouting to alternative sources. Default is ``True``.
        initial : bool, optional
            Whether this is an initial cascade. Default is ``False``.
        """
        dependency_name = f"dependency_{row.source}_{row.target}"

        # Get former access information
        es_access_base, ppl_former_access, ppl_former_access_source_failed = (
            self._get_former_access_info(dependency_name)
        )

        # Recheck access based on rerouting setting
        if rerouting:
            ppl_new_access, ppl_access_all_via = (
                self._recompute_dependencies_with_rerouting(row, dependency_name)
            )
        else:
            ppl_new_access, ppl_access_all_via = (
                self._validate_dependencies_without_rerouting(
                    row, dependency_name, es_access_base
                )
            )

        # Mark access states and supply
        self._mark_access_states_and_supply(
            row,
            ppl_former_access,
            ppl_former_access_source_failed,
            ppl_access_all_via,
            ppl_new_access,
        )

    @DeprecationWarning
    def _recheck_access(
        self,
        source_ci,
        target_ci,
        via_ci,
        friction_surf,
        dist_thresh,
        dur_thresh,
        criterion="distance",
        link_name=None,
        bidir=False,
    ):
        """Recheck access for constrained links (deprecated)

        Parameters
        ----------
        source_ci : str
            Source infrastructure type.
        target_ci : str
            Target infrastructure type.
        via_ci : str
            Via-link infrastructure type.
        friction_surf : object
            Friction surface for duration-based checks.
        dist_thresh : float
            Distance threshold for path validity.
        dur_thresh : float
            Duration threshold for friction surface.
        criterion : str, optional
            Edge weight attribute for path search. Default is ``"distance"``.
        link_name : str, optional
            Edge type name. Default is ``None``.
        bidir : bool, optional
            Whether to add reverse links. Default is ``False``.
        """
        es_check = self.graph.es.select(ci_type=f"dependency_{source_ci}_{target_ci}")

        bools_check = [self.graph.vs[edge.source]["func_tot"] > 0 for edge in es_check]

        es_check = [
            edge for edge, bool_check in zip(es_check, bools_check) if bool_check
        ]

        if len(es_check) > 0:

            edge_geoms = [edge["geometry"] for edge in es_check]
            v_ids_target = [edge.target for edge in es_check]
            v_ids_source = [edge.source for edge in es_check]
            v_ids_via = [vs.index for vs in self.graph.vs.select(ci_type=f"{via_ci}")]

            # first check friction
            friction = self._calc_friction(edge_geoms, friction_surf)
            bool_keep = friction < dur_thresh

            # then check shortest paths
            v_seq = self.graph.vs(
                list(np.unique([*v_ids_target, *v_ids_source, *v_ids_via]))
            )

            subgraph = self.graph.induced_subgraph(v_seq)
            # subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(v_seq)
            subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(
                self.graph, subgraph
            )

            graph_subgraph_vsdict = {
                int(v): int(k) for k, v in subgraph_graph_vsdict.items()
            }
            subgraph.delete_edges(subgraph.es.select(func_tot_lt=1))
            wrong_edges = set(subgraph.es["ci_type"]).difference({via_ci})
            subgraph.delete_edges(subgraph.es.select(ci_type_in=wrong_edges))

            for ix, source, target, bool_f in zip(
                np.arange(len(bool_keep)), v_ids_source, v_ids_target, bool_keep
            ):
                if not bool_f:
                    dist = subgraph.distances(
                        source=graph_subgraph_vsdict[source],
                        target=graph_subgraph_vsdict[target],
                        weights="distance",
                    )
                    if dist[0][0] < dist_thresh:
                        bool_keep[ix] = True
                        es_check[ix]["distance"] = dist[0][0]
            self.graph.delete_edges(
                [edge.index for edge, bool_f in zip(es_check, bool_keep) if not bool_f]
            )
