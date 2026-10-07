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

"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj


import scipy

from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.graph_calcs import GraphCalcs, _dependency_name
from climada_petals.engine.networks.nw_utils import make_edge_geometries, _ckdnearest

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.engine import ImpactCalc
from climada.util import lines_polys_handler as u_lp
from climada.util.constants import ONE_LAT_KM

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel("INFO")


class NetworkCalcs:
    """Wrapper for network preparation and cascade execution

    High-level convenience wrapper for common CI network workflows.
    Uses GraphCalcs internally for all graph operations. For advanced users
    seeking flexibility, GraphCalcs can be used directly to compose custom
    cascades and dependency setups.
    """

    def __init__(self, network, dep_table=None, friction_surf=None):
        """Create a network calculator

        Parameters
        ----------
        network : Network
            Network on which links, dependencies and cascades are computed.
        dep_table : pd.DataFrame, optional
            Dependency table with one row per source-target dependency. Required
            by ``initialize_base_state``, ``setup_dependencies`` and ``cascade``.
            Default is ``None``.
        friction_surf : object, optional
            Friction surface used for duration-based linking. Default is ``None``.
        """
        self._network = network
        self.dep_table = dep_table
        self._graph_calc = GraphCalcs(network=network, friction_surf=friction_surf)

    @property
    def network(self):
        """Access the current network"""
        return self._network

    @network.setter
    def network(self, new_network):
        """Keep graph_calc in sync whenever network is updated"""
        self._network = new_network
        self._graph_calc.network = new_network

    @property
    def graph(self):
        """Return cached igraph representation"""
        return self._graph_calc.graph

    def merge_clusters(
        self, ci_type, max_iter, dist_thresh=30000, graph_connectivity_mode="weak"
    ):
        """Iteratively merge disconnected clusters

        In each iteration, every cluster (connected component) is linked to its
        closest other cluster within ``dist_thresh``. Iterations stop when the
        network forms a single cluster or after ``max_iter`` iterations, so the
        network can remain disconnected if clusters are further apart than
        ``dist_thresh``. The network is updated in place.

        Parameters
        ----------
        ci_type : str
            Edge type assigned to new links.
        max_iter : int
            Maximum number of merge iterations.
        dist_thresh : float, optional
            Maximum distance (meters) for cluster linking. Default is ``30000``.
        graph_connectivity_mode : str, optional
            Connectivity mode used to identify clusters (``"weak"`` or
            ``"strong"``). Default is ``"weak"``, which ignores edge directions.
        """
        iter_count = 0
        n_clusters = len(self.graph.connected_components(mode=graph_connectivity_mode))
        LOGGER.info("Number of clusters in the network before merging: %i", n_clusters)
        # dist_thresh = cntry_shape.area / nclusters
        while (n_clusters > 1) and (iter_count < max_iter):
            self._graph_calc.link_clusters(
                dist_thresh=dist_thresh,
                graph_connectivity_mode=graph_connectivity_mode,
                link_attrs={"ci_type": ci_type},
            )
            iter_count += 1
            self.network = Network.from_graphs(self.graph, crs=self.network.crs)
            self._graph_calc.full_reset()
            n_clusters = len(
                self.graph.connected_components(mode=graph_connectivity_mode)
            )
        LOGGER.info("Number of clusters in the network after merging: %i", n_clusters)

    def add_physical_links(self, physical_dependencies):
        """Add physical links based on a physical dependency table

        Each target is linked to its ``n_links`` closest sources (within
        ``thresh_dist``) with a link of type ``link``. Physical links represent
        infrastructure without a direction (routing and connectivity ignore
        the edge direction), so a single link per pair is created unless the
        optional column ``bidir_link`` is ``True``.

        Parameters
        ----------
        physical_dependencies : pd.DataFrame
            Table with the columns ``source``, ``target``, ``link``,
            ``thresh_dist``, ``n_links`` and optionally ``bidir_link``
            (default ``False``).
        """

        # create "missing physical structures" - needed for real world flows
        # syntax: each target is connected to max k sources given constraints

        for i, row in physical_dependencies.iterrows():
            self._graph_calc.link_vertices_closest_k(
                source_attrs={"ci_type": row["source"]},
                target_attrs={"ci_type": row["target"]},
                link_attrs={"ci_type": row["link"]},
                dist_thresh=row["thresh_dist"],
                bidir=bool(row.get("bidir_link", False)),
                k=row["n_links"],
            )

        ##update network
        self.network = Network.from_graphs(self.graph, crs=self.network.crs)

        # Invalidate cached graph
        self._graph_calc.full_reset()

    def initialize_base_state(self):
        """Initialize functional, capacity, and supply base state

        Sets all nodes and edges to fully functional, and creates the capacity,
        supply and access state attributes of the dependencies in the dependency
        table.

        Raises
        ------
        ValueError
            If no dependency table was provided.

        Notes
        -----
        The method should be called after all physical links have been added (``merge_clusters``,
        ``add_physical_links``), so that these also receive a functional state, and
        before ``setup_dependencies``.

        See Also
        --------
        Network.initialize_funcstates, Network.initialize_capacity,
        Network.initialize_supply
        """

        # base state
        # do it after build up of physical dependencies so that created edge also receive
        # functionality states

        if self.dep_table is None:
            raise ValueError(
                "Cannot initialize base state without a dependency table. Please provide a dependency table at initialization."
            )
        self.network.initialize_funcstates()
        self.network.initialize_capacity(self.dep_table)
        self.network.initialize_supply(self.dep_table)

    def setup_dependencies(self):
        """Create the dependency links of the dependency table

        For each row of the dependency table, targets are linked to their sources
        with edges of type ``dependency_{source}_{target}``, directed from source
        to target, following the row's ``link_condition`` (``"distance"``,
        ``"duration"`` or ``"edgecond"``), thresholds and ``n_links``. The network
        is updated in place.

        Raises
        ------
        ValueError
            If no dependency table was provided.

        See Also
        --------
        GraphCalcs.calc_dependencies
        """

        if self.dep_table is None:
            raise ValueError(
                "Cannot setup dependencies without a dependency table. Please provide a dependency table at initialization."
            )

        for i, row in self.dep_table.iterrows():
            dependency_name = _dependency_name(row["source"], row["target"])
            self._graph_calc.calc_dependencies(
                source_attrs={"ci_type": row["source"]},
                target_attrs={"ci_type": row["target"]},
                via_attrs={"ci_type": row["via_link"]},
                link_attrs={"ci_type": dependency_name},
                link_condition=row["link_condition"],
                dist_thresh=row["thresh_dist"],
                dur_thresh=row["thresh_dur"],
                k=row["n_links"],
                bidir_link=False,  # dependencies are directed from source to target
            )

        # update network
        self.network = Network.from_graphs(self.graph, crs=self.network.crs)
        # Invalidate cached graph
        self._graph_calc.full_reset()

    def cascade(
        self,
        p_source="power_plant",
        p_sink="power_line",
        source_var="el_generation",
        demand_var="el_consumption",
        friction_surf=None,
        rerouting=True,
        access_check_method="routing",
    ):
        """Perform cascade failure propagation on the network

        Iteratively updates the functional states of network components until
        convergence, then updates end-user dependencies. The cascade models how
        failures propagate through the network based on internal and functional
        dependencies. The network is updated in place.

        Parameters
        ----------
        p_source : str, optional
            Type of power source nodes. Default is ``"power_plant"``.
        p_sink : str, optional
            Type of power sink nodes. Default is ``"power_line"``.
        source_var : str, optional
            Attribute name for source generation. Default is ``"el_generation"``.
        demand_var : str, optional
            Attribute name for demand consumption. Default is ``"el_consumption"``.
        friction_surf : object, optional
            Friction surface for duration-based access checks. Default is ``None``.
        rerouting : bool, optional
            If ``True``, end-users whose source or path failed can be linked to
            another source. Default is ``True``.
        access_check_method : str, optional
            Method to check end-user access, ``"routing"`` or ``"propagation"``.
            Default is ``"routing"``.

        Raises
        ------
        ValueError
            If no dependency table was provided.

        Notes
        -----
        - Internal and functional dependencies are updated until the functional
          states no longer change.
        - End-user dependencies are updated once, after convergence.
        - The network is then rebuilt from the graph, so that edges created or
          removed during the cascade are reflected, and the cached graph is reset.
        """

        if self.dep_table is None:
            raise ValueError(
                "Cannot propagate cascade failure without a dependency table. Please provide a dependency table at initialization."
            )

        delta = -1
        cycles = 0
        while delta != 0:
            LOGGER.info("Updating functional states. Current delta: %i", delta)
            func_states_vs, func_states_es = self._graph_calc.funcstates_sum()
            self._graph_calc.update_internal_dependencies(
                p_source=p_source,
                p_sink=p_sink,
                source_var=source_var,
                demand_var=demand_var,
            )

            self._graph_calc.update_functional_dependencies(self.dep_table)
            func_states_vs2, func_states_es2 = self._graph_calc.funcstates_sum()
            delta = max(
                abs(func_states_vs - func_states_vs2),
                abs(func_states_es - func_states_es2),
            )
            cycles += 1

        LOGGER.info(
            "Ended functional state update." + " Proceeding to end-user update."
        )
        self._graph_calc.update_enduser_dependencies(
            self.dep_table,
            friction_surf,
            rerouting=rerouting,
            access_check_method=access_check_method,
        )

        # update network
        self.network = Network.from_graphs(self.graph, crs=self.network.crs)
        # Invalidate cached graph
        self._graph_calc.full_reset()
