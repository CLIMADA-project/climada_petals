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
from climada_petals.engine.networks.graph_calcs import GraphCalcs
from climada_petals.engine.networks.nw_utils import make_edge_geometries, _ckdnearest
from climada_petals.engine.networks.nw_preps import reset_ids

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.engine import ImpactCalc
from climada.util import lines_polys_handler as u_lp
from climada.util.constants import ONE_LAT_KM

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel("INFO")

# constants
PHYSICAL_SOURCES = ["road", "rail"]


class NetworkCalcs:
    """Wrapper for network preparation and cascade execution"""

    def __init__(self, network, dep_table, friction_surf=None, directed=True):
        self.network = network
        self.dep_table = dep_table
        self._graph_calc = GraphCalcs(
            network_calc=self, directed=directed, friction_surf=friction_surf
        )

    @property
    def graph(self):
        """Return cached igraph representation"""
        return self._graph_calc.graph

    def merge_clusters(
        self, ci_type, max_iter, dist_thresh=30000, graph_connectivity_mode="weak"
    ):
        """Iteratively merge disconnected clusters

        Parameters
        ----------
        ci_type : str
            Edge type assigned to new links.
        max_iter : int
            Maximum number of merge iterations.
        dist_thresh : float, optional
            Maximum distance (meters) for cluster linking. Default is ``30000``.
        graph_connectivity_mode : str, optional
            Connectivity mode for the graph. Default is ``"weak"``.
        """
        iter_count = 0
        n_clusters = len(self.graph.connected_components())
        LOGGER.info(
            print("Number of clusters in the network before merging: %i", n_clusters)
        )
        # dist_thresh = cntry_shape.area / nclusters
        while (n_clusters > 1) and (iter_count < max_iter):
            self._graph_calc.link_clusters(
                dist_thresh=dist_thresh,
                graph_connectivity_mode=graph_connectivity_mode,
                link_attrs={"ci_type": ci_type},
            )
            iter_count += 1
            self.network = Network.from_graphs(self.graph, crs=self.network.crs)
            self.network = reset_ids(self.network)
            self._graph_calc.full_reset()
        n_clusters = len(self.graph.connected_components())
        LOGGER.info(
            print("Number of clusters in the network after merging: %i", n_clusters)
        )

    def add_physical_links(self):
        """Add physical links based on dependency table"""

        # create "missing physical structures" - needed for real world flows
        # syntax: each target is connected to max k sources given constraints
        physical_dependencies = self.dep_table.loc[
            (self.dep_table["source"].isin(PHYSICAL_SOURCES))
        ]
        for i, row in physical_dependencies.iterrows():
            self._graph_calc.link_vertices_closest_k(
                source_attrs={"ci_type": row["source"]},
                target_attrs={"ci_type": row["target"]},
                link_attrs={"ci_type": row["source"]},
                dist_thresh=row["thresh_dist"],
                bidir=True,
                k=row["n_links"],
            )

        ##update network
        self.network = Network.from_graphs(self.graph, crs=self.network.crs)

        ##need to have all ids reset after new road edges have been added
        self.network = reset_ids(self.network)

        # Invalidate cached graph
        self._graph_calc.full_reset()

    def initialize_base_state(self):
        """Initialize functional, capacity, and supply base state"""
        # base state
        # do it after build up of physical dependencies so that created edge also receive
        # functionality states
        self.network.initialize_funcstates()
        self.network.initialize_capacity(self.dep_table)
        self.network.initialize_supply(self.dep_table)

    def setup_dependencies(self):
        """Create dependency links and initialize end-user access"""
        for i, row in self.dep_table.iterrows():
            dependency_name = f'dependency_{row["source"]}_{row["target"]}'
            self._graph_calc.calc_dependencies(
                source_attrs={"ci_type": row["source"]},
                target_attrs={"ci_type": row["target"]},
                via_attrs={"ci_type": row["via_link"]},
                link_attrs={"ci_type": dependency_name},
                link_condition=row["link_condition"],
                dist_thresh=row["thresh_dist"],
                dur_thresh=row["thresh_dur"],
                k=row["n_links"],
                bidir_link=row["bidir_link"],
            )
        # initialize base access and supply for enduser dependencies
        enduser_rows = self.dep_table[self.dep_table["type_I"] == "enduser"]
        for __, row in enduser_rows.iterrows():
            dependency_name = f'dependency_{row["source"]}_{row["target"]}'
            dep_edges = self.graph.es.select(ci_type=dependency_name)
            if len(dep_edges) == 0:
                continue
            targets = [edge.target for edge in dep_edges]
            self.graph.vs[targets][
                f"access_state_{row.source}_{row.target}"
            ] = "access undisrupted"
            self.graph.vs[targets][f"actual_supply_{row.source}_{row.target}"] = 1
        # reset ids as new edges have been created
        self.network = reset_ids(self.network)
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
        initial=False,
        friction_surf=None,
        rerouting=True,
        access_check_method="routing",
    ):
        """
        Perform cascade failure analysis on the network.
                This method iteratively updates the functional states of network components
                until convergence, then updates end-user dependencies. The cascade process
                models how failures propagate through the network based on internal and
                functional dependencies.
                Parameters
                ----------
                p_source : str, optional
                    Type of source nodes (default is 'power_plant').
                p_sink : str, optional
                    Type of sink nodes (default is 'power_line').
                source_var : str, optional
                    Variable name for source generation (default is 'el_generation').
                demand_var : str, optional
                    Variable name for demand consumption (default is 'el_consumption').
                initial : bool, optional
                    If True, forces end-user dependency update even if convergence occurs
                    in first cycle (default is False).
                friction_surf : optional
                    Friction surface data for routing calculations (default is None).
                rerouting : bool, optional
                    If True, enables rerouting for end-user dependencies (default is True).
                access_check_method : str, optional
                    Method to use for checking access (default is "routing").
                Returns
                -------
                None
                    Updates the network in place.
                Notes
                -----
                - The method iterates until functional states converge (delta = 0)
                - Updates both internal and functional dependencies during iteration
                - After convergence, updates end-user dependencies
                - Resets network IDs to account for newly created edges
                - Invalidates cached graph data after completion
        """
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

        # reset ids as new edges may have been created
        self.network = reset_ids(self.network)
        # update network
        self.network = Network.from_graphs(self.graph, crs=self.network.crs)
        # Invalidate cached graph
        self._graph_calc.full_reset()
