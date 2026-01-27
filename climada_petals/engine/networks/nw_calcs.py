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
import igraph as ig
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
from tqdm import tqdm
import timeit
import gc


import scipy

from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_utils import (make_edge_geometries,
                                                     _ckdnearest)
from climada_petals.engine.networks.nw_preps import (reset_ids,
                                                     ordered_network)

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.engine import ImpactCalc
from climada.util import lines_polys_handler as u_lp
from climada.util.constants import ONE_LAT_KM

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')

#constants
PHYSICAL_SOURCES = ['road', 'rail']

class GraphCalcs():
    def __init__(self, parent, directed=False):
        """
        network : instance of networks.nw_base.Network
        """
        self.parent = parent #parent nw calc object
        self._graph = None
        self.directed = directed
    @property
    def network(self):
        return self.parent.network

    def build_graph(self):
        self._graph = self.network.to_graph(directed=self.directed)
        return self._graph
    @property
    def graph(self):
        if self._graph is None:
            return self.build_graph()
        return self._graph

    def invalidate(self):
        self._graph = None
    # =============================================================================
    # Making links
    # =============================================================================

    def link_clusters(self, dist_thresh=np.inf, link_attrs=None):
        """
        link nodes from different clusters to their nearest nodes in other
        clusters to generate one connected graph.

        Parameters
        ----------
        graph : nw_base.Graph object
        dist_thresh : float
            distance threshold up to where clusters can be linked
        metres : bool
            whether distance is in metres

        Returns
        -------
        graph
        """

        gdf_vs = self.graph.get_vertex_dataframe()
        #if ci_type is not None:#filter to ci_type
        #    gdf_vs = gdf_vs[gdf_vs.ci_type==ci_type]
        # Use 'weak' mode for directed graphs to treat them as undirected for connectivity
        mode = 'weak' if self.graph.is_directed() else None
        gdf_vs['membership'] = self.graph.connected_components(mode=mode).membership

        v_ids_source = []
        v_ids_target = []

        # very rough conversion from metres to degrees
        dist_thresh /= (ONE_LAT_KM*1000)

        members = np.unique(gdf_vs['membership'])
        if len(members) <= 1:
            LOGGER.info("Graph is already fully connected; no cluster linking needed.")
            return
        for i in range(len(members)-1):# last iteration is redundant
            gdf_a = gdf_vs[gdf_vs['membership'] == members[i]]
            gdf_b = gdf_vs[gdf_vs['membership'] != members[i]]
            if gdf_a.empty or gdf_b.empty:
                continue
            try:
                dists, ix_match = _ckdnearest(
                    gdf_a, gdf_b, dist_thresh=dist_thresh)
                source = gdf_a.iloc[np.where(dists == min(dists))[
                    0]].index[0]
                target = gdf_b.loc[ix_match[np.where(dists == min(dists))[
                    0]]].index[0]
                v_ids_source.append(source)
                v_ids_target.append(target)
            except (IndexError, KeyError):
                # if no match within given distance
                continue

        if len(v_ids_source) > 0:
            self._edges_from_vlists(
                v_ids_source, v_ids_target, link_attrs)

        #self.invalidate()
    def link_vertices_closest_k(self, source_attrs, target_attrs, link_attrs=None,
                                dist_thresh=np.inf, bidir=False, k=5):
        """
        find k nearest source vertices for each target vertex,
        given distance constraints and identifying attributes

        Parameters
        ----------
        graph : nw_base.Graph object
        source_attrs : dict {attr_name_s1 : attr_val_s1, ..., }
        target_attrs : dict {attr_name_t1 : attr_val_t1, ..., }


        Returns
        -------
        graph
        """

        # select only those for which specified attrs apply
        df_vs_target = GraphCalcs._filter_vertices(self.graph, target_attrs)

        # select only those for which specified attrs apply
        df_vs_source = GraphCalcs._filter_vertices(self.graph, source_attrs)

        v_ids_source, v_ids_target = self._select_closest_k(
            df_vs_source, df_vs_target, dist_thresh, bidir, k)

        self._edges_from_vlists(v_ids_source, v_ids_target, link_attrs)

        #self.invalidate()
    def link_vertices_edgecond(self, target_attrs, edge_attrs, link_attrs,
                               bidir=False):
        """
        make a dependency edge between two vertices if another edge with a
        certain attribute (specified in edge_attrs) already exists between those
        two.
        Primarily intended for dependency_road_people, given that a road exists
        directly at people node.

        Parameters
        ----------
        graph : nw_base.Graph object
        target_attrs : dict
        edge_attrs : dict
        link_attrs : dict
        bidir : bool

        Returns
        -------
        graph
        """
        df_vs_target = GraphCalcs._filter_vertices(self.graph, target_attrs)

        vs_target = self.graph.vs[df_vs_target.index.values]

        pot_edges_ids = [self.graph.incident(v_target, mode='all')
                     for v_target in vs_target]
        # flatten nested list
        pot_edges_ids = [item for sublist in pot_edges_ids for item in sublist]

        # select those edges which fulfill edge_attrs
        #for key, value in edge_attrs.items():
        #    pot_edges = [self.graph.es[item] for item in pot_edges_ids
        #                 if self.graph.es[item][key] == value]
        pot_edges = [
            self.graph.es[item]
            for item in pot_edges_ids
            if all(self.graph.es[item][key] == value for key, value in edge_attrs.items())
            ]

        #make sure source are indeed of edge_attrs type and targets of target_attrs type
        sources = []
        targets = []
        for edge in pot_edges:
            source_vx = self.graph.vs[edge.source]
            target_vx = self.graph.vs[edge.target]
            if source_vx['ci_type'] == edge_attrs['ci_type']:
                sources.append(edge.source)
                targets.append(edge.target)
            elif target_vx['ci_type'] == edge_attrs['ci_type']:
                sources.append(edge.target)
                targets.append(edge.source)
            else:
                raise ValueError("Edge does not connect correct ci_types!")

        self._edges_from_vlists(sources, targets, link_attrs)
        if bidir:
            self._edges_from_vlists(targets, sources, link_attrs)

        #self.invalidate()
    def link_vertices_shortest_paths(self, source_attrs, target_attrs, via_attrs,
                                     link_attrs, dist_thresh=10e6, criterion='distance',
                                     k=1, bidir=False):
        """
        Per target, choose single shortest path to source which is
        below dist_thresh.

        Parameters
        ----------
        graph : nw_base.Graph object
        source_attrs : dict
        target_attrs : dict
        via_attrs : dict
        link_attrs : dict
        single_shortest : bool
            Whether to make a link between all sources and targets for which the
            shortest path is < dist_thresh, or whether to only make a link for the
            shortest of all.
        bidir : bool

        Returns
        -------
        graph
        """

        # subgraph containing only "allowed" elements
        subgraph = self._create_subgraph(source_attrs, target_attrs, via_attrs)

        # mapping from subgraph to graph indices
        subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(self.graph, subgraph)

        # select only those for which specified attrs apply
        df_vs_target = GraphCalcs._filter_vertices(subgraph, target_attrs)

        # select only those for which specified attrs apply
        df_vs_source = GraphCalcs._filter_vertices(subgraph, source_attrs)

        path_dists = subgraph.distances(
            source=df_vs_source.index.values, target=df_vs_target.index.values,
            weights=criterion,
            mode='all')
        path_dists = np.array(path_dists)  # dim: (#sources, #targets)

        if not len(path_dists) == 0:

            if k==1:#single_shortest
                ix_source, ix_target = np.where(
                    ((path_dists == path_dists.min(axis=0)) &
                     (path_dists <= dist_thresh)))  # min dist. per target
            else:
                ix_source, ix_target = np.where(path_dists < dist_thresh)

            # Get the indices of the k shortest distances per target
            #sorted_indices = np.argsort(path_dists, axis=0)[:k, :]  # Indices of k smallest distances
            #valid_mask = path_dists[sorted_indices, np.arange(path_dists.shape[1])] <= dist_thresh  # Apply threshold
    #
            ## Extract source and target indices
            #ix_source, ix_target = np.where(valid_mask)
            #ix_source = sorted_indices[ix_source, ix_target]  # Map to original indices

            ## re-map sources to original graph
            #v_ids_source = df_vs_source.index.values[list(ix_source)]
            v_ids_source = [subgraph_graph_vsdict[v_id_source] for v_id_source
                            in df_vs_source.index.values[list(ix_source)]]

            ## re-map targets to original graph
            #v_ids_target = df_vs_target.index.values[list(ix_target)]
            v_ids_target = [subgraph_graph_vsdict[v_id_target] for v_id_target
                            in df_vs_target.index.values[list(ix_target)]]

            link_attrs['distance'] = path_dists[(ix_source, ix_target)]

            self._edges_from_vlists(v_ids_source, v_ids_target, link_attrs)

            if bidir:
               self._edges_from_vlists(v_ids_target, v_ids_source, link_attrs)

        #self.invalidate()
    def link_vertices_friction_surf(self, source_ci, target_ci, friction_surf,
                                        link_name=None, dist_thresh=None,
                                        bidir=False, k=5, dur_thresh=None):

            gdf_vs = self.graph.get_vertex_dataframe()
            gdf_vs_target = gdf_vs[gdf_vs.ci_type==target_ci]
            gdf_vs_source = gdf_vs[(gdf_vs.ci_type==source_ci) &
                                   (gdf_vs.func_tot==1)]
            del gdf_vs

            if not (gdf_vs_source.empty or gdf_vs_target.empty):
                v_ids_source, v_ids_target = self._select_closest_k(
                    gdf_vs_source, gdf_vs_target, dist_thresh, bidir, k)

                edge_geoms = make_edge_geometries(
                    self.graph.vs[v_ids_source]['geometry'],
                    self.graph.vs[v_ids_target]['geometry'])

                friction = self._calc_friction(edge_geoms, friction_surf)
                v_ids_source = np.array(v_ids_source)[friction<dur_thresh]
                v_ids_target = np.array(v_ids_target)[friction<dur_thresh]

                if not link_name:
                    link_name = f'dependency_{source_ci}_{target_ci}'

                self._edges_from_vlists(list(v_ids_source), list(v_ids_target), {'ci_type': link_name})

            #self.invalidate()

    # =============================================================================
    # Helper funcs for making links
    # =============================================================================
    @staticmethod
    def _filter_vertices(graph, attr_dict):
        """
        get vertices of graph to which given attributes apply

        Parameters
        ----------
        graph : igraph.Graph object

        Returns
        -------
        df_vs : pd.Dataframe
        """

        df_vs = graph.get_vertex_dataframe()
        for key, value in attr_dict.items():
            df_vs = df_vs[df_vs[key] == value]
        return df_vs

    @staticmethod
    def _filter_edges(graph, attr_dict):
        """
        get edges of graph to which given attributes apply

        Parameters
        ----------
        graph : igraph.Graph object

        Returns
        -------
        df_es : pd.Dataframe
        """

        df_es = graph.get_edge_dataframe()
        for key, value in attr_dict.items():
            df_es = df_es[df_es[key] == value]
        return df_es


    def _edges_from_vlists(self, v_ids_source, v_ids_target, link_attrs=None):
        """
        add edges to graph given source and target vertex lists
        adds geometries, edge lengths, edge names and func states as attributes

        Parameters
        ----------
        graph : nw_base.Graph object

        Returns
        -------
        graph : nw_base.Graph object
        """

        pairs = list(zip(v_ids_source, v_ids_target))

        link_attrs['geometry'] = make_edge_geometries(
            self.graph.vs[v_ids_source]['geometry'],
            self.graph.vs[v_ids_target]['geometry'])

        if 'distance' not in link_attrs.keys():
            print("! adding distance")
            link_attrs['distance'] = [
                pyproj.Geod(ellps='WGS84').geometry_length(edge_geom)
                for edge_geom in link_attrs['geometry']
            ]
        ## check if save to add new orig_id here as they might replace existing ones
        #add orig_id to new edges
        #if 'orig_id' not in link_attrs.keys():
        #    link_attrs['orig_id'] = np.max(self.graph.es['orig_id'])+np.arange(len(pairs))

        self.graph.add_edges(pairs, attributes=link_attrs)

    @staticmethod
    def _select_closest_k(gdf_vs_source, gdf_vs_target, dist_thresh,
                          bidir=False, k=5):
        """
        Parameters
        ----------

        Returns
        -------
        list, list
        """

        # crappy conversion of metres to degrees
        dist_thresh /= (ONE_LAT_KM*1000)

        # index matches, in format (#target vs, k). nans for those without matches
        __, ix_matches = _ckdnearest(gdf_vs_target, gdf_vs_source, k=k,
                                     dist_thresh=dist_thresh)
        # broadcast target indices to same format
        ix_matches = ix_matches.flatten()
        v_ids_target = np.array(np.broadcast_to(
            np.array([gdf_vs_target.id]).T, (len(gdf_vs_target), k)).flatten())
        v_ids_target = v_ids_target[~np.isnan(ix_matches)]
        v_ids_source = np.array(
            gdf_vs_source.loc[ix_matches[~np.isnan(ix_matches)]].id)

        if bidir:
            v_ids_target = np.append(v_ids_target, v_ids_source)
            v_ids_source = np.append(v_ids_source, v_ids_target)

        return list(v_ids_source), list(v_ids_target)


    def _create_subgraph(self, source_attrs, target_attrs, via_attrs):
        """
        Create a subgraph from the original graph. Includes only vertices and edges
        from source, target and via types.

        Parameters
        ----------
        graph : nw_base.Graph object
        source_attrs : dict
        target_attrs : dict
        via_attrs : dict
        link_attrs : dict


        Returns
        -------
        vs_keep : list
            vertex ids of original graph that is kept in subgraph

        subgraph : iself.graph
            induced subgraph of graph, given v_seq


        See also
        --------
        link_vertices_shortest_paths(), link_vertices_shortest_path()
        """

        # select only those for which specified attrs apply
        df_vs_source = GraphCalcs._filter_vertices(self.graph, source_attrs)
        df_vs_target = GraphCalcs._filter_vertices(self.graph, target_attrs)
        df_vs_via = GraphCalcs._filter_vertices(self.graph, via_attrs)

        vs_keep = np.concatenate((df_vs_source.index.values,
                                  df_vs_target.index.values,
                                  df_vs_via.index.values))

        # vs_keep has indexing of original graph, subgraph has new indexing. There
        # is no way of keeping track of the re-ordering, other than to have a named
        # attribute!
        self.graph.vs['orig_id'] = range(len(self.graph.vs))
        self.graph.es['orig_id'] = range(len(self.graph.es))
        subgraph = self.graph.induced_subgraph(vs_keep)

        #map graph ids to subgraph ids
        #subgraph_graph_esdict = GraphCalcs._get_subgraph2graph_esdict(self.graph, subgraph)
        #graph_to_subgraph_esdict = {v: k for k, v in subgraph_graph_esdict.items()}

        # delete remaining edges that have wrong attributes
        #df_es_target = GraphCalcs._filter_edges(subgraph, target_attrs)
        #df_es_source = GraphCalcs._filter_edges(subgraph, source_attrs)
        df_es_via = GraphCalcs._filter_edges(subgraph, via_attrs)

        correct_edges = df_es_via.index.values
        #correct_edges = np.concatenate((df_es_target.index.values,
        #                               df_es_source.index.values,
        #                               df_es_via.index.values))

        #map correct edge ids back to subgraph ids
        #correct_edges = [subgraph_graph_esdict[id_corr_edg] for id_corr_edg
        #                in correct_edges]

        wrong_edges = set(range(len(subgraph.es))).difference(set(correct_edges))

        subgraph.delete_edges(wrong_edges)

        return subgraph

    @staticmethod
    def _get_subgraph2graph_vsdict(graph, subgraph):
        """
        Keep track of which vertices in induced subgraph represent which vertices
        in original graph. dict[subgraph_vs_ind] = graph_vs_ind
        Goes via the named attribute 'orig_id' created before making the subgraph.

        Parameters
        ----------
        graph : igraph.Graph
        subgraph : igraph.Graph
            induced subgraph of graph

        Returns
        -------
        dict
            mapping from subgraph to graph indices.
        """
        subgraph_vs_indices = [subvx.index for subvx in subgraph.vs]
        subgraph_orig_ids = subgraph.vs.get_attribute_values('orig_id')
        df_subg = pd.DataFrame(
            subgraph_vs_indices, index=subgraph_orig_ids, columns=['index_sub'])

        graph_vs_indices = [vx.index for vx in graph.vs]
        graph_orig_ids = graph.vs.get_attribute_values('orig_id')
        df_g = pd.DataFrame(
            graph_vs_indices, index=graph_orig_ids,  columns=['index_g'])

        df_conc = pd.concat([df_subg, df_g], axis=1)
        result = df_conc.groupby('index_sub')['index_g'].first().to_dict()
        #result = dict((k, v) for k, v in zip(df_conc['index_sub'], df_conc['index_g'])) #previous version, very slow
        return result

    @staticmethod
    def _get_subgraph2graph_esdict(graph, subgraph):
        """
        Keep track of which edges in induced subgraph represent which edges
        in original graph. dict[subgraph_vs_ind] = graph_vs_ind
        Goes via the named attribute 'orig_id' created before making the subgraph.

        Parameters
        ----------
        graph : igraph.Graph
        subgraph : igraph.Graph
            induced subgraph of graph

        Returns
        -------
        dict
            mapping from subgraph to graph indices.
        """
        subgraph_es_indices = [subvx.index for subvx in subgraph.es]
        subgraph_orig_ids = subgraph.es.get_attribute_values('orig_id')
        df_subg = pd.DataFrame(
            subgraph_es_indices, index=subgraph_orig_ids, columns=['index_sub'])

        graph_es_indices = [vx.index for vx in graph.es]
        graph_orig_ids = graph.es.get_attribute_values('orig_id')
        df_g = pd.DataFrame(
            graph_es_indices, index=graph_orig_ids,  columns=['index_g'])

        df_conc = pd.concat([df_subg, df_g], axis=1)
        result = df_conc.groupby('index_sub')['index_g'].first().to_dict()
        #result = dict((k, v) for k, v in zip(df_conc['index_sub'], df_conc['index_g'])) #previous version, very slow
        return result
    @staticmethod
    def _calc_friction(edge_geoms, friction_surf):

            # define mapping as impact function.
            impf_fric = ImpactFunc()
            impf_fric.id = 1
            impf_fric.haz_type = ''
            impf_fric.name = 'friction surface mapping'
            impf_fric.intensity_unit = 'min/m'
            impf_fric.intensity = np.linspace(friction_surf.intensity.data.min(),
                                              friction_surf.intensity.data.max(),
                                              num=500)
            impf_fric.mdd = np.linspace(friction_surf.intensity.data.min(),
                                        friction_surf.intensity.data.max(),
                                        num=500)
            impf_fric.paa = np.sort(np.linspace(1, 1, num=500))
            impf_fric.check()
            impf_set = ImpactFuncSet()
            impf_set.append(impf_fric)

            # perform impact calc for mapping.
            exp_links = Exposures(gpd.GeoDataFrame({'geometry': edge_geoms}))
            exp_links.gdf['impf_'] = 1
            #exp_links.gdf["geometry_orig"] = exp_links.gdf.geometry

            # step-by-step to avoid 0 duration sections
            exp_pnt = u_lp.exp_geom_to_pnt(
                exp_links, res=100, to_meters=True,
                disagg_met=u_lp.DisaggMethod.FIX, disagg_val=100)

            impact_pnt = ImpactCalc(exp_pnt, impf_set, friction_surf).impact(save_mat=True)
            if impact_pnt.imp_mat.size < len(exp_pnt.gdf):
                imp_arry = np.array(impact_pnt.imp_mat.todense()).flatten()
                imp_arry[imp_arry==0] = \
                    exp_pnt.gdf.value[imp_arry==0]*friction_surf.intensity.data.min()
                impact_pnt.imp_mat = scipy.sparse.csr_matrix(imp_arry)

            friction = u_lp.impact_pnt_agg(
                impact_pnt, exp_pnt.gdf, u_lp.AggMethod.SUM)

            return friction.eai_exp

    def _calc_dependencies(self, source_attrs, target_attrs, via_attrs, link_attrs, link_condition, dist_thresh, bidir_link, friction_surf=None):
        if "distance" in link_condition:
            self.link_vertices_shortest_paths(
                source_attrs=source_attrs,
                target_attrs=target_attrs,
                via_attrs=via_attrs,
                link_attrs=link_attrs,
                dist_thresh=dist_thresh,
                bidir=bidir_link
            )
        elif "duration" in link_condition:
            self.link_vertices_friction_surf(
                source_ci=source_attrs,
                target_ci=target_attrs,
                friction_surf=friction_surf,
                link_name=link_attrs,
                dist_thresh=dist_thresh,
                bidir=bidir_link
            )
        elif "edgecond" in link_condition:
            self.link_vertices_edgecond(
                target_attrs=target_attrs,
                edge_attrs=source_attrs,
                link_attrs=link_attrs,
                bidir=bidir_link
            )
        else:
            raise NotImplementedError

    # =============================================================================
    # Propagation functions
    # =============================================================================

    def _propagate_check_fail(self, source, target, thresh_func):
        """
        propagate capacities from source vertices to target vertices
        on the subgraph via the adjacency matrix.
        check whether capacity enough.
        fail target if not.
        """
        v_seq = self.graph.vs.select(ci_type_in=[source, target])
        subgraph = self.graph.induced_subgraph(v_seq)

        #take vertices seq from subgraph and do operation on it to be sure that order is the same
        v_seq_sub = subgraph.vs.select(ci_type_in=[source, target])

        #keep track of original ids
        subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(self.graph, subgraph)
        #subgraph_graph_vsdict = _get_subgraph2graph_vsdict_old(graph, v_seq)
        v_seq_orig_id = [subgraph_graph_vsdict[v_id] for v_id in v_seq_sub.indices]

        try:
            adj_sub = subgraph.get_adjacency_sparse()
        except (TypeError, ValueError):
            # treats case where empty adjacency matrix!
            adj_sub = scipy.sparse.csr_matrix(subgraph.get_adjacency().data)

        # Hadamard product func_tot (*) capacity
        func_capa = np.multiply(v_seq['func_tot'],
                                v_seq[f'capacity_{source}_{target}'])
        # propagate capacities down from source --> target along adj
        capa_rec = scipy.sparse.csr_matrix(func_capa).dot(adj_sub)

        # functionality thesholds for received capacity
        func_thresh = np.array([thresh_func if vx['ci_type'] == target
                                else 0 for vx in v_seq])

        # boolean vector whether received capacity great enough to supply endusers
        capa_suff = (np.array(capa_rec.todense()).squeeze()
                     >= func_thresh).astype(int)
        # This is under the assumption that subgraph retains the same
        # relative ordering of vertices as in v_seq extracted from graph!
        # This further assumes that any operation on a VertexSeq equally modifies its graph.
        # Both should be the case, but the igraph doc is always a bit ambiguous

        if target == 'people':
            self.graph.vs[v_seq_orig_id][f'actual_supply_{source}_{target}'] = capa_suff
        else:
            # Only update TARGET vertices, not source vertices
            # Get indices of target vertices only
            target_mask = np.array([vx['ci_type'] == target for vx in v_seq])
            target_orig_ids = [v_seq_orig_id[i] for i, is_target in enumerate(target_mask) if is_target]
            func_tot_targets = np.minimum(capa_suff[target_mask], v_seq.select(ci_type=target)['func_tot'])
            self.graph.vs[target_orig_ids]['func_tot'] = func_tot_targets

        #delete large objects to avoid memory issues
        del capa_rec, func_capa, capa_suff, adj_sub, func_thresh, subgraph
        gc.collect()

    def _funcstates_sum(self):
        """
        return the total funcstate sum func_tot across all vertices and
        edges

        Parameters
        ----------
        graph : nw_base.Graph object

        Returns
        -------
        tuple (int, int) : sum of vertex func_tot, sum of edges func_tot
        """
        return (sum(self.graph.vs.get_attribute_values('func_tot')),
                sum(self.graph.es.get_attribute_values('func_tot')))


    def _update_internal_dependencies(self, p_source, p_sink, source_var,
                                      demand_var):
        """
        for ci-types with an internally networked structure (e.g. roads and
        power lines which consist in edges + nodes), update those ci networks
        internally
        """

        # specifically for roads: if edge is dysfunctional, render its target vertex dysfunctional
        if {'road'}.issubset(set(self.graph.vs['ci_type'])):
            LOGGER.info('Updating roads')
            sources_targets_dys = [[edge.source, edge.target] for edge in self.graph.es.select(
                ci_type='road').select(func_tot_eq=0)]
            sources_targets_dys = np.array(sources_targets_dys).flatten().tolist()#flatten array
            self.graph.vs.select(sources_targets_dys).select(
                ci_type='road')['func_tot'] = 0

        # specifically for powerlines: check power clusters
        if {p_source, p_sink}.issubset(set(self.graph.vs['ci_type'])):
            LOGGER.info('Updating power clusters')
            # For another version using pandapower, see nw_utils.py
            # Since powerlines are directed in a directed graph,
            # make sure 'reverse' lines are also down

            edges_dys = self.graph.es.select(ci_type='power_line'
                                             ).select(func_tot_eq=0)
            reverse_edges = [(edge.target, edge.source) for edge in edges_dys]
            eids = self.graph.get_eids(pairs=reverse_edges, path=None,
                                       directed=True, error=True)
            self.graph.es[eids]['func_tot'] = 0
            LOGGER.info(f"""Using updated power line algorithm: dysfunc edges before:
                  {len(edges_dys)}, after: {len(self.graph.es.select(ci_type='power_line'
                                             ).select(func_tot_eq=0))}""")
            self.powercap_from_clusters(p_source=p_source, p_sink=p_sink,
                                        demand_ci='people', source_var=source_var, demand_var=demand_var)

    def _update_functional_dependencies(self, df_dependencies):

        for __, row in df_dependencies[
                df_dependencies['type_I'] == 'functional'].iterrows():

            if row.access_cnstr:
                # TODO: Implement
                LOGGER.warning(
                    'Road access condition for CI-CI deps not yet implemented')

            self._propagate_check_fail(row.source, row.target, row.thresh_func)


    def _update_enduser_dependencies(self, df_dependencies,
                                     friction_surf, rerouting=True):

        for __, row in df_dependencies[
                df_dependencies['type_I'] == 'enduser'].iterrows():


            if (row.target == 'people'):
                self._check_access(row, friction_surf, rerouting=rerouting)
            else:
                self._propagate_check_fail(row.source, row.target, row.thresh_func)


    def _get_former_access_info(self, dependency_name):
        """
        Get information about former access from base state.

        Parameters
        ----------
        dependency_name : str
            Name of dependency edges to check

        Returns
        -------
        tuple
            (es_access_base, ppl_former_access, ppl_former_access_source_failed)
        """
        es_access_base = self.graph.es.select(ci_type=dependency_name)
        ppl_former_access = [edge.target for edge in es_access_base]
        ppl_former_access_source_failed = [
            edge.target for edge in es_access_base
            if self.graph.vs[edge.source]["func_tot"] < 1
        ]
        return es_access_base, ppl_former_access, ppl_former_access_source_failed

    def _recompute_dependencies_with_rerouting(self, row, dependency_name):
        """
        Recompute dependencies when rerouting is allowed.

        Parameters
        ----------
        row : pd.Series
            Dependency configuration row
        dependency_name : str
            Name of dependency edges

        Returns
        -------
        tuple
            (es_access_new, ppl_new_access, ppl_access_all_via)
        """
        # Delete existing dependencies to recompute from scratch
        self.graph.delete_edges(ci_type=dependency_name)

        # If access_cnstr is False, compute dependencies without requiring functional via edges
        via_attrs_dict = {'ci_type': row['via_link']}
        if row.access_cnstr:
            via_attrs_dict['func_tot'] = 1

        self._calc_dependencies(
            source_attrs={'ci_type': row['source'], 'func_tot': 1},
            target_attrs={'ci_type': row['target']},
            via_attrs=via_attrs_dict,
            link_attrs={'ci_type': dependency_name},
            link_condition=row['link_condition'],
            dist_thresh=row['thresh_dist'],
            bidir_link=row['bidir_link']
        )

        es_access_new = self.graph.es.select(ci_type=dependency_name)
        ppl_new_access = [edge.target for edge in es_access_new]

        # Check if could have access if links were not broken
        if row.access_cnstr:
            # Compute dependencies without requiring functional via edges to identify
            # people who could have access if via links were functional
            self._calc_dependencies(
                source_attrs={'ci_type': row['source'], 'func_tot': 1},
                target_attrs={'ci_type': row['target']},
                via_attrs={'ci_type': row['via_link']},  # No func_tot requirement
                link_attrs={'ci_type': "new_"+dependency_name},
                link_condition=row['link_condition'],
                dist_thresh=row['thresh_dist'],
                bidir_link=row['bidir_link']
            )

            # People having access regardless of the state of the via link
            ppl_access_all_via = [
                edge.target for edge in self.graph.es.select(ci_type="new_"+dependency_name)
            ]
            # Delete temporary edges
            self.graph.delete_edges(ci_type="new_"+dependency_name)
        else:
            ppl_access_all_via = ppl_new_access

        return es_access_new, ppl_new_access, ppl_access_all_via

    def _validate_dependency_paths(self, es_check, row, graph_subgraph_vsdict, subgraph):
        """
        Validate which dependency edges still have valid paths through functional via edges.

        Parameters
        ----------
        es_check : list
            Edges to validate
        row : pd.Series
            Dependency configuration row
        graph_subgraph_vsdict : dict
            Mapping from graph vertex ids to subgraph vertex ids
        subgraph : igraph.Graph
            Subgraph containing only source, target, and via vertices

        Returns
        -------
        list
            Edges that still have valid paths
        """
        edges_to_keep = []
        for edge in es_check:
            source_sub = graph_subgraph_vsdict.get(edge.source)
            target_sub = graph_subgraph_vsdict.get(edge.target)

            if source_sub is not None and target_sub is not None:
                try:
                    dist = subgraph.distances(
                        source=source_sub,
                        target=target_sub,
                        weights='distance',
                        mode='all'  # Treat as undirected for connectivity check
                    )
                    # If path exists and is within threshold, keep the edge
                    if dist[0][0] < row['thresh_dist']:
                        edges_to_keep.append(edge)
                except (IndexError, ValueError):
                    # No path exists, edge should be removed
                    pass

        return edges_to_keep

    def _validate_dependencies_without_rerouting(self, row, dependency_name,
                                                  es_access_base, ppl_former_access):
        """
        Validate existing dependencies when rerouting is not allowed.

        Parameters
        ----------
        row : pd.Series
            Dependency configuration row
        dependency_name : str
            Name of dependency edges
        es_access_base : list
            Base access edges
        ppl_former_access : list
            People who had former access

        Returns
        -------
        tuple
            (es_access_new, ppl_new_access, ppl_access_all_via)
        """
        es_access_new = self.graph.es.select(ci_type=dependency_name)

        if row.access_cnstr:
            # Need to check if via edges used in former dependencies have failed
            # Keep edges where source is functional and path through functional via edges exists
            es_check = [
                edge for edge in es_access_base
                if self.graph.vs[edge.source]['func_tot'] >= 1
            ]

            if len(es_check) > 0:
                # Check which edges still have valid paths through functional via edges
                v_ids_source = [edge.source for edge in es_check]
                v_ids_target = [edge.target for edge in es_check]
                v_ids_via = [v.index for v in self.graph.vs.select(ci_type=row['via_link'])]

                # Create subgraph with only source, target, and via vertices
                v_seq = list(np.unique([*v_ids_source, *v_ids_target, *v_ids_via]))
                self.graph.vs['orig_id'] = range(len(self.graph.vs))
                subgraph = self.graph.induced_subgraph(v_seq)

                # Map from original graph ids to subgraph ids
                subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(self.graph, subgraph)
                graph_subgraph_vsdict = {int(v): int(k) for k, v in subgraph_graph_vsdict.items()}

                # Delete failed edges and non-via edges from subgraph
                edges_to_delete = []
                for e in subgraph.es:
                    if 'func_tot' in e.attributes() and e['func_tot'] is not None and e['func_tot'] < 1:
                        edges_to_delete.append(e.index)
                subgraph.delete_edges(edges_to_delete)

                wrong_edges = set(subgraph.es['ci_type']).difference({row['via_link']})
                subgraph.delete_edges(subgraph.es.select(ci_type_in=wrong_edges))

                # Check which former dependency edges still have valid paths
                edges_to_keep = self._validate_dependency_paths(
                    es_check, row, graph_subgraph_vsdict, subgraph
                )

                # Delete edges that no longer have valid paths
                edges_to_remove = [edge.index for edge in es_check if edge not in edges_to_keep]
                self.graph.delete_edges(edges_to_remove)

            # Get updated list of access edges after validation
            es_access_new = self.graph.es.select(ci_type=dependency_name)
            ppl_new_access = [edge.target for edge in es_access_new]
            ppl_access_all_via = ppl_former_access
        else:
            # No access constraints, so no need to check via edges
            ppl_new_access = [edge.target for edge in es_access_new]
            ppl_access_all_via = ppl_new_access

        return es_access_new, ppl_new_access, ppl_access_all_via

    def _mark_access_states_and_supply(self, row, es_access_new, ppl_former_access,
                                       ppl_former_access_source_failed, ppl_access_all_via,
                                       ppl_new_access):
        """
        Mark access states and actual supply for all nodes.

        Parameters
        ----------
        row : pd.Series
            Dependency configuration row
        es_access_new : list
            New access edges after validation
        ppl_former_access : list
            People who had former access
        ppl_former_access_source_failed : list
            People whose former source failed
        ppl_access_all_via : list
            People who could have access if via links were functional
        ppl_new_access : list
            People who have current access
        """
        # If init source was failed but ppl still have access, then they have access to a new source
        ppl_access_new_source = [
            edge.target for edge in es_access_new
            if edge.target in ppl_former_access_source_failed
        ]
        self.graph.vs[ppl_access_new_source][f'access_state_{row.source}_people'] = "access new source"

        # If people have access only when no functional via is required, then access is disrupted via
        ppl_access_broken_via = [
            ppl_node for ppl_node in ppl_access_all_via
            if ppl_node not in ppl_new_access
        ]
        self.graph.vs[ppl_access_broken_via][f'access_state_{row.source}_people'] = "access disrupted via"

        # If people do not have access due to via constraints, then the access is disrupted at source
        ppl_no_reaccess = [
            ppl_node for ppl_node in ppl_former_access
            if (ppl_node not in ppl_new_access and ppl_node not in ppl_access_broken_via)
        ]
        self.graph.vs[ppl_no_reaccess][f'access_state_{row.source}_people'] = "access disrupted source"

        # Remaining accesses are undisrupted
        ppl_access_undisrupted = [
            edge.target for edge in es_access_new
            if edge.target not in ppl_former_access_source_failed
        ]
        self.graph.vs[ppl_access_undisrupted][f'access_state_{row.source}_people'] = "access undisrupted"

        # Add boolean array of actual supply
        # People with access get supply=1 (includes undisrupted, all_via, and new_source)
        # Use set to avoid duplicates
        ppl_with_supply = list(set(ppl_access_undisrupted + ppl_access_all_via + ppl_access_new_source))
        self.graph.vs[ppl_with_supply][f'actual_supply_{row.source}_{row.target}'] = 1
        self.graph.vs[ppl_no_reaccess + ppl_access_broken_via][f'actual_supply_{row.source}_{row.target}'] = 0

    def _check_access(self, row, friction_surf, rerouting=True, initial=False):
        """
        Check and update access states for end-user dependencies.

        This is the main function that orchestrates the access checking process.
        It delegates to helper functions for specific tasks.

        Parameters
        ----------
        row : pd.Series
            Dependency configuration row containing source, target, via_link, etc.
        friction_surf : object or None
            Friction surface for calculating travel times (not currently used)
        rerouting : bool, optional
            Whether to allow rerouting to alternative sources. Default is True.
        initial : bool, optional
            Whether this is an initial cascade (no former access state). Default is False.
        """
        dependency_name = f'dependency_{row.source}_{row.target}'

        # Get former access information
        es_access_base, ppl_former_access, ppl_former_access_source_failed = \
                self._get_former_access_info(dependency_name)

        # Recheck access based on rerouting setting
        if rerouting:
            es_access_new, ppl_new_access, ppl_access_all_via = \
                self._recompute_dependencies_with_rerouting(row, dependency_name)
        else:
            es_access_new, ppl_new_access, ppl_access_all_via = \
                self._validate_dependencies_without_rerouting(
                    row, dependency_name, es_access_base, ppl_former_access
                )

        # Mark access states and supply
        self._mark_access_states_and_supply(
            row, es_access_new, ppl_former_access, ppl_former_access_source_failed,
            ppl_access_all_via, ppl_new_access
        )


    @DeprecationWarning
    def _recheck_access(self, source_ci, target_ci, via_ci, friction_surf,
                       dist_thresh, dur_thresh, criterion='distance',
                       link_name=None, bidir=False):
        """
        for links with access constraints, re-check those with functional
        sources where paths may however be broken now.
        Those with dysfunctional sources don't need to be checked, since
        dysfunctionality will anyways propagate to target later.
        """
        es_check = self.graph.es.select(
            ci_type=f'dependency_{source_ci}_{target_ci}')

        bools_check = [self.graph.vs[edge.source]['func_tot'] > 0
                       for edge in es_check]

        es_check = [edge for edge, bool_check in zip(es_check, bools_check)
                    if bool_check]

        if len(es_check) > 0:

            edge_geoms = [edge['geometry'] for edge in es_check]
            v_ids_target = [edge.target for edge in es_check]
            v_ids_source = [edge.source for edge in es_check]
            v_ids_via = [vs.index for vs in
                         self.graph.vs.select(ci_type=f'{via_ci}')]

            # first check friction
            friction = self._calc_friction(edge_geoms, friction_surf)
            bool_keep = friction < dur_thresh

            # then check shortest paths
            v_seq = self.graph.vs(list(np.unique([*v_ids_target, *v_ids_source,
                                                  *v_ids_via])))

            subgraph = self.graph.induced_subgraph(v_seq)
            #subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(v_seq)
            subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(self.graph, subgraph)

            graph_subgraph_vsdict = {int(v): int(k) for k,
                                     v in subgraph_graph_vsdict.items()}
            subgraph.delete_edges(subgraph.es.select(func_tot_lt=1))
            wrong_edges = set(subgraph.es['ci_type']).difference(
                {via_ci})
            subgraph.delete_edges(subgraph.es.select(ci_type_in=wrong_edges))

            for ix, source, target, bool_f in (zip(np.arange(len(bool_keep)),
                                                   v_ids_source, v_ids_target,
                                                   bool_keep)):
                if not bool_f:
                    dist = subgraph.distances(
                        source=graph_subgraph_vsdict[source],
                        target=graph_subgraph_vsdict[target],
                        weights='distance')
                    if dist[0][0] < dist_thresh:
                        bool_keep[ix] = True
                        es_check[ix]['distance'] = dist[0][0]
            self.graph.delete_edges([edge.index for edge, bool_f in
                                     zip(es_check, bool_keep)
                                     if not bool_f])

    def powercap_from_clusters(self, p_source, p_sink, demand_ci, source_var,
                               demand_var):

        capacity_vars = [var for var in self.graph.vs.attributes()
                         if f'capacity_{p_sink}_' in var]
        power_vs = self.graph.vs.select(
            ci_type_in=['power_line', p_source, p_sink, demand_ci])
        # make subgraph spanning all nodes, but only functional edges
        # Subgraph operations do not modify original graph.
        power_subgraph = self.graph.induced_subgraph(power_vs)
        power_subgraph.delete_edges(func_tot_lt=0.1)

        subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(self.graph, power_vs)

        for cluster in power_subgraph.clusters(mode='weak'):

            sources = power_subgraph.vs[cluster].select(ci_type=p_source)
            sinks = power_subgraph.vs[cluster].select(ci_type=p_sink)
            demands = power_subgraph.vs[cluster].select(ci_type=demand_ci)

            psupply = sum([source[source_var]*source['func_tot']
                           for source in sources])
            pdemand = sum([demand[demand_var] for demand in demands])

            try:
                sd_ratio = min(1, psupply/pdemand)
            except ZeroDivisionError:
                sd_ratio = 1

            for var in capacity_vars:
                self.graph.vs[
                    [subgraph_graph_vsdict[sink.index] for sink in sinks]
                ].set_attribute_values(var, sd_ratio)

class NetworkCalcs():
    """Gathers wrapper for network preparation"""
    def __init__(self, network, dep_table):
        self.network = network
        self.dep_table = dep_table
        self.graph_calc = GraphCalcs(parent=self)

    @property
    def graph(self):
        """Convenience proxy"""
        return self.graph_calc.graph

    def merge_clusters(self, ci_type, max_iter, dist_thresh=30000):
        iter_count = 0
        n_clusters = len(self.graph_calc.graph.connected_components())
        LOGGER.info(print(f'Number of clusters in the network before merging: {n_clusters}'))
        #dist_thresh = cntry_shape.area / nclusters
        while (n_clusters>1) and (iter_count<max_iter):
            self.graph_calc.link_clusters(dist_thresh=dist_thresh, link_attrs={'ci_type':ci_type})
            iter_count+=1
            self.network.update_network_from_graphs(self.graph)
            self.network = reset_ids(self.network)
            self.graph_calc.invalidate()
        n_clusters = len(self.graph_calc.graph.connected_components())
        LOGGER.info(print(f'Number of clusters in the network after merging: {n_clusters}'))

        #update orig_id field (required for building subgraphes)
        self.network.edges['orig_id'] = self.network.edges['id']
        self.network.nodes['orig_id'] = self.network.nodes['id']
        self.network = ordered_network(self.network)


    def add_physical_links(self):
        """Wrapper function to add physical links."""

        # create "missing physical structures" - needed for real world flows
        # syntax: each target is connected to max k sources given constraints
        physical_dependencies = self.dep_table.loc[
            (self.dep_table['source'].isin(PHYSICAL_SOURCES))
        ]
        for i, row in physical_dependencies.iterrows():
            self.graph_calc.link_vertices_closest_k(
                                         source_attrs={
                                             'ci_type': row['source']},
                                         target_attrs={
                                             'ci_type': row['target']},
                                         link_attrs={
                                             'ci_type': row['source']},
                                         dist_thresh=row['thresh_dist'],
                                         bidir=True,
                                         k=row['n_links'])

        ##TODO refactor the reformating of the network
        self.network.update_network_from_graphs(self.graph)

        ##need to have all ids reset after new road edges have been added
        self.network = reset_ids(self.network)
        #update orig_id field (required for building subgraphes)
        self.network.edges['orig_id'] = self.network.edges['id']
        self.network.nodes['orig_id'] = self.network.nodes['id']
        self.network = ordered_network(self.network)

        # Invalidate cached graph
        self.graph_calc.invalidate()
    def initialize_base_state(self):
        #base state
        #do it after build up of physical dependencies so that created edge also receive
        #functionality states
        self.network.initialize_funcstates()
        self.network.initialize_capacity(self.dep_table)
        self.network.initialize_supply(self.dep_table)

    def setup_dependencies(self):
        for i, row in self.dep_table.iterrows():
            dependency_name = f'dependency_{row["source"]}_{row["target"]}'
            self.graph_calc._calc_dependencies(
                source_attrs={
                    'ci_type': row['source'],
                    'func_tot': 1},
                target_attrs={
                    'ci_type': row['target']},
                via_attrs={
                    'ci_type': row['via_link'],
                    'func_tot': 1},
                link_attrs={
                    'ci_type': dependency_name},
                link_condition=row['link_condition'],
                dist_thresh=row['thresh_dist'],
                bidir_link=row['bidir_link']
            )
        #update network
        self.network.update_network_from_graphs(self.graph)
        # Invalidate cached graph
        self.graph_calc.invalidate()


    def cascade(self, p_source='power_plant',
                p_sink='power_line', source_var='el_generation', demand_var='el_consumption',
                  initial=False, friction_surf=None, rerouting=True):
        """
        entire cascade wrapper for internal state update, functional dependency iterations,
        enduser dependency updates. CI-specific. Writing more generically does not
        work atm, as there are too many CI-specific functionality assumptions.
        """
        delta = -1
        cycles = 0
        while delta != 0:
            LOGGER.info(
                f'Updating functional states. Current delta: {delta}')
            func_states_vs, func_states_es = self.graph_calc._funcstates_sum()
            self.graph_calc._update_internal_dependencies(
                p_source=p_source, p_sink=p_sink, source_var=source_var, demand_var=demand_var)

            self.graph_calc._update_functional_dependencies(self.dep_table)
            func_states_vs2, func_states_es2 = self.graph_calc._funcstates_sum()
            delta = max(abs(func_states_vs-func_states_vs2),
                        abs(func_states_es-func_states_es2))
            cycles += 1

        LOGGER.info('Ended functional state update.' +
                    ' Proceeding to end-user update.')
        if (cycles > 1) or initial:
            self.graph_calc._update_enduser_dependencies(
                self.dep_table, friction_surf, rerouting=rerouting)

        #update network
        self.network.update_network_from_graphs(self.graph)
        # Invalidate cached graph
        self.graph_calc.invalidate()