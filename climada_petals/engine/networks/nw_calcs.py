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
import igraph as ig
import logging
import numpy as np
import geopandas as gpd
import pandas as pd
import pyproj
import shapely
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from tqdm import tqdm


from climada_petals.engine.networks.base import Network
from climada_petals.engine.networks.nw_flows import PowerCluster

from climada.util.constants import ONE_LAT_KM

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')


class GraphCalcs():
    
    def __init__(self, graph):
        """
        nw_or_graph : instance of networks.base.Network or .MultiNetwork or
            igraph.Graph
        """
        self.graph = graph
    
    def link_clusters(self):
        """
        recursively link clusters to giant component of graph, by closest nodes
        """
        while len(self.graph.clusters()) > 1:
            
            giant = self.graph.clusters().giant()
            next_cluster = self.graph.clusters().subgraphs()[1]
            
            gdf_vs_source = giant.get_vertex_dataframe()
            gdf_vs_assign = next_cluster.get_vertex_dataframe()
            dists, ix_match = self._ckdnearest(gdf_vs_assign, gdf_vs_source)
            
            source = gdf_vs_assign.iloc[np.where(dists==min(dists))[0]]
            target = gdf_vs_source.iloc[ix_match[np.where(dists==min(dists))[0]]]

            edge_geom = self.make_edge_geometries([source.geometry.values[0]],[target.geometry.values[0]])[0]
            
            dist = pyproj.Geod(ellps='WGS84').geometry_length(edge_geom)
            self.graph.add_edge(target.orig_id.values[0], source.orig_id.values[0], 
                                geometry=edge_geom, ci_type=gdf_vs_assign.ci_type.iloc[0],
                                distance=dist)
            
    def link_vertices_closest_k(self, from_ci, to_ci, 
                                link_name=None, dist_thresh=None,
                                bidir=False, k=5):
        """
        match all vertices of graph_assign to closest vertices in graph_base.
        Updated in vertex attributes (vID of graph_base, geometry & distance)
        
        """
        gdf_vs = self.graph.get_vertex_dataframe()
        gdf_vs_target = gdf_vs[gdf_vs.ci_type == to_ci]
        gdf_vs_source = gdf_vs[gdf_vs.ci_type == from_ci]

        # shape: (target vs, k)
        dists, ix_matches = self._ckdnearest(gdf_vs_target, gdf_vs_source, k=k)
        if k>1:
            dists = dists.flatten()
            ix_matches = ix_matches.flatten()
        # TODO: dist_thresh condition only holds vaguely for m input & lat/lon coordinates (EPSG4326)
        # shape: (target vs, k)
        dists_thresh_bool = [((not dist_thresh) or np.isnan(dist_thresh) or (dist < (dist_thresh/(ONE_LAT_KM*1000))))
                            for dist in dists]
        
        # shape: (target vs, ..)
        if k > 1:
            gdf_target_old = gdf_vs_target.copy()
            gdf_vs_target = pd.DataFrame(columns=gdf_target_old.columns)
            for __, row in gdf_target_old.iterrows():
                gdf_vs_target = gdf_vs_target.append([row]*k)
           
        edge_geoms = self.make_edge_geometries(gdf_vs_source.loc[ix_matches].geometry[dists_thresh_bool],
                                               gdf_vs_target.geometry[dists_thresh_bool])
        if not link_name:
            link_name = f'dependency_{from_ci}_{to_ci}'

        self.graph.add_edges(zip(ix_matches[dists_thresh_bool],
                                 gdf_vs_target.index[dists_thresh_bool]), 
                                         attributes =
                                          {'geometry' : edge_geoms,
                                           'ci_type' : [link_name],
                                           'distance' : dists[dists_thresh_bool]*(ONE_LAT_KM*1000),
                                           'func_internal' : 1,
                                           'func_tot' : 1,
                                           'imp_dir' : 0})
        if bidir:
            self.graph.add_edges(zip(gdf_vs_target.index[dists_thresh_bool],
                                     ix_matches[dists_thresh_bool]), 
                                         attributes =
                                          {'geometry' : edge_geoms,
                                           'ci_type' : [link_name],
                                           'distance' : dists[dists_thresh_bool]*(ONE_LAT_KM*1000),
                                           'func_internal' : 1,
                                           'func_tot' : 1,
                                           'imp_dir' : 0})

    def link_vertices_shortest_paths(self, from_ci, to_ci, via_ci, 
                                    dist_thresh=100000, criterion='distance',
                                    link_name=None, bidir=False):
        """
        make all links below certain dist_thresh along shortest paths length
        between all possible sources & targets --> doesnt matter whether search
        is done from source to all targets or from target to all sources
        
        """
        
        vs_subgraph = self.graph.vs.select(
            ci_type_in=[from_ci, to_ci, via_ci])
        subgraph =  vs_subgraph.subgraph()
        subgraph.delete_edges(subgraph.es.select(func_tot_lt=1))
        subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(
            vs_subgraph)
        vs_source = subgraph.vs.select(ci_type=from_ci)
        vs_target = subgraph.vs.select(ci_type=to_ci)
        
        # TODO: don't hardcode metres to degree conversion assumption
        ix_matches = self._preselect_destinations(
            vs_source, vs_target, dist_thresh/(ONE_LAT_KM*1000))       
        for vx, indices in tqdm(zip(vs_source, ix_matches), 
                                desc=f'paths from {from_ci}', total=len(vs_source)):
            paths = subgraph.get_shortest_paths(
                vx, vs_target[indices], 
                subgraph.es.get_attribute_values(criterion),
                mode='out',
                output='epath')
            for index, path in zip(indices, paths):
                if path:
                    dist = np.array(subgraph.es.get_attribute_values(criterion))[path].sum()
                    if dist < dist_thresh:
                        edge_geom = self.make_edge_geometries(
                            [self.graph.vs[subgraph_graph_vsdict[vx.index]]['geometry']],
                            [self.graph.vs[subgraph_graph_vsdict[vs_target[index].index]]['geometry']])[0]
                        if not link_name:
                            link_name = f'dependency_{from_ci}_{to_ci}'
                        #TODO: collect all edges to be added and assign in one add_edges call
                        self.graph.add_edge(self.graph.vs[subgraph_graph_vsdict[vx.index]],
                                            self.graph.vs[subgraph_graph_vsdict[vs_target[index].index]],
                                            geometry=edge_geom,
                                            ci_type=link_name, 
                                            distance=dist,func_internal=1, 
                                            func_tot=1, imp_dir=0)  
                        if bidir:
                            self.graph.add_edge(self.graph.vs[subgraph_graph_vsdict[vs_target[index].index]], 
                                                self.graph.vs[subgraph_graph_vsdict[vx.index]],
                                                geometry=edge_geom,
                                                ci_type=link_name, 
                                                distance=dist,func_internal=1, 
                                                func_tot=1, imp_dir=0)
                        
                        
    def link_vertices_shortest_path(self, from_ci, to_ci, via_ci, 
                                    dist_thresh=100000, criterion='distance',
                                    link_name=None, bidir=False):
        """
        only single shortest path to target (!) 
        below dist_thresh is chosen
        """
        
        vs_subgraph = self.graph.vs.select(
            ci_type_in=[from_ci, to_ci, via_ci])
        subgraph =  vs_subgraph.subgraph()
        subgraph.delete_edges(subgraph.es.select(func_tot_lt=1))
        
        subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(
            vs_subgraph)
        vs_source = subgraph.vs.select(ci_type=from_ci)
        vs_target = subgraph.vs.select(ci_type=to_ci)
        
        # TODO: don't hardcode metres to degree conversion assumption
        ix_matches = self._preselect_destinations(
            vs_target, vs_source, dist_thresh/(ONE_LAT_KM*1000))       
        
        for indices, vx in tqdm(zip(ix_matches, vs_target), desc=f'Paths to {to_ci}', total=len(vs_target)):
            # can only go from single vertex to several, hence syntax target--> sources
            # doesnt matter, since roads are bidirectional anyways
            paths = subgraph.get_shortest_paths(
                vx, vs_source[indices],
                subgraph.es.get_attribute_values(criterion),
                mode='out', output='epath')
            min_dist = dist_thresh
            
            for index, path in zip(indices, paths):
                if path:
                    dist = np.array(subgraph.es.get_attribute_values(criterion))[path].sum()
                    if dist < min_dist:
                        min_dist = dist
                        min_index = index
            
            if min_dist < dist_thresh:
                edge_geom = self.make_edge_geometries(
                            [self.graph.vs[subgraph_graph_vsdict[vs_source[min_index].index]]['geometry']],
                            [self.graph.vs[subgraph_graph_vsdict[vx.index]]['geometry']])[0]

                if not link_name:
                            link_name = f'dependency_{from_ci}_{to_ci}'
                #TODO: collect all edges to be added and assign in one add_edges call
                self.graph.add_edge(
                    self.graph.vs[subgraph_graph_vsdict[vs_source[min_index].index]],
                    self.graph.vs[subgraph_graph_vsdict[vx.index]],
                    geometry=edge_geom,
                    ci_type=link_name, distance=min_dist,
                    func_internal=1, func_tot=1, imp_dir=0)
                if bidir:
                    self.graph.add_edge(
                        self.graph.vs[subgraph_graph_vsdict[vx.index]], 
                        self.graph.vs[subgraph_graph_vsdict[vs_source[min_index].index]], 
                        geometry=edge_geom,
                        ci_type=link_name, distance=min_dist,
                        func_internal=1, func_tot=1, imp_dir=0)


    def place_dependency(self, source, target, single_link=True, 
                         access_cnstr=False, dist_thresh=None):
        """
        source : supporting infra
        target : dependent infra / ppl
        single_link : bool
            Whether there is (max.) a single link allowable between a target 
            vertice and a source vertice or whether target vertices can have several
            dependencies of the same type.
            Whether the link cis unchangeable between the specific S & T or 
            if it could be re-routable upon invalidity at a later stage.
        access_cnstr : bool
            Whether the link requires physical (road)-access to be possible 
            between S & T or a direct (logical) connection can be made.
        dist_thresh : float
            Whether there is a maximum link length allowed for a link length 
            to be established (in metres!). Default is None.
        
        """
        LOGGER.info(f'Placing dependency between {source} and {target}')
        dep_name = f'dependency_{source}_{target}'
        
        # make links
        if not access_cnstr:
            if single_link:
                self.link_vertices_closest_k(source, target, link_name=dep_name,
                                           dist_thresh=dist_thresh, k=1)
            else:
                self.link_vertices_closest_k(source, target, link_name=dep_name,
                                             dist_thresh=dist_thresh, k=5)
        else:
            if single_link:
                self.link_vertices_shortest_path(source, target, via_ci='road', 
                                    dist_thresh=dist_thresh, criterion='distance',
                                    link_name=dep_name)
            else:
                self.link_vertices_shortest_paths(source, target, via_ci='road', 
                                    dist_thresh=dist_thresh, criterion='distance',
                                    link_name=dep_name)
        
    def get_shortest_paths(self, from_vs, to_vs, edgeweights, mode='out', 
                           output='epath'):
        
        return self.graph.get_shortest_paths(from_vs, to_vs, edgeweights, 
                                             mode, output)
        
    def _make_edgeweights(self, criterion, from_ci, to_ci, via_ci):
        
        allowed_edges= [f'{via_ci}']
        
        return np.array([1*edge[criterion] if (
            (edge['ci_type'] in allowed_edges) & (edge['func_internal']==1))
                        else 10e16 for edge in self.graph.es])
    
    def _preselect_destinations(self, vs_assign, vs_base, dist_thresh):
        points_base = np.array([(x.x, x.y) for x in vs_base['geometry']])
        point_tree = cKDTree(points_base)
        
        points_assign = np.array([(x.x, x.y) for x in vs_assign['geometry']])
        ix_matches = []
        for assign_loc in points_assign:
            ix_matches.append(point_tree.query_ball_point(assign_loc, dist_thresh))
        return ix_matches
    
    def _funcstates_sum(self):
        """return the total funcstate sum func_tot across all vertices and edges
        
        Parameters
        ---------
        
        Returns
        -------
        tuple (int, int) :
            sum of vertex func_tot, sum of edges func_tot
        
        """
        return (sum(self.graph.vs.get_attribute_values('func_tot')), 
                sum(self.graph.es.get_attribute_values('func_tot')))

    def _update_cis_internally(self, df_dependencies, p_source, p_sink, per_cap_cons, source_var='el_gen_mw'):
        ci_types = np.unique(np.append(
            np.unique(df_dependencies.source), 
            np.unique(df_dependencies.target)))
        
        targets = [edge.target for edge in self.graph.es.select(func_tot_eq=0)]
        self.graph.vs.select(targets)['func_tot'] = 0
        
        for ci_type in ci_types:
            if (ci_type=='substation') or (ci_type=='power line'):
                LOGGER.info('Updating power clusters')
                # TODO: This is poorly hard-coded and poorly assigned.
                self = PowerCluster.set_capacity_from_sd_ratio(
                    self, per_cap_cons=per_cap_cons, source_ci=p_source,
                    sink_ci=p_sink, source_var=source_var)
            else:
                LOGGER.info(f'Updating {ci_type}')
                targets = [edge.target for edge in 
                           self.graph.es.select(ci_type_eq=f'{ci_type}'
                                                ).select(func_tot_eq=0)]
                self.graph.vs.select(
                    targets).select(ci_type_eq=f'{ci_type}')['func_tot'] = 0
    
    def _update_functional_dependencies(self, df_dependencies):
    
        for __, row in df_dependencies[
                df_dependencies['type_I']=='functional'].iterrows():
            
            vs_source_target = self.graph.vs.select(
                ci_type_in=[row.source, row.target])
                        
            #subgraph_graph_vsdict = self._get_subgraph2graph_vsdict(vs_source_target)

            # adjacency matrix of selected subgraph
            adj_sub = csr_matrix(np.array(vs_source_target.subgraph(
                ).get_adjacency().data))
            
            # Hadamard product func_tot (*) capacity
            func_capa = csr_matrix(np.multiply(
                vs_source_target.get_attribute_values('func_tot'),
                vs_source_target.get_attribute_values(
                    f'capacity_{row.source}_{row.target}')))
            
            # for dependencies w/ access_cnstr, set (x,y) in adj to 0 if length of  
            # path x-->y now > thresh_dist
            if row.access_cnstr:
                pass
                # vs_subgraph_rd = self.graph.vs.select(
                #             ci_type_in=[row.source, row.target, 'road'])
                # subgraph_rd =  vs_subgraph_rd.subgraph()
                # subgraph_rd_graph_vsdict = self._get_subgraph2graph_vsdict(
                #     vs_subgraph_rd)
                # vs_source_rd = subgraph.vs.select(ci_type=row.source)
                # vs_target_rd = subgraph.vs.select(ci_type=row.target)

                # for source_ix_sub, target_ix_sub in zip(*np.where(adj_sub==1)):
                # paths = subgraph.get_shortest_paths(
                #                 vx, vs_target[indices], 
                #                 subgraph.es.get_attribute_values(criterion)*subgraph.es.get_attribute_values('func_tot'),
                #                 mode='out',
                #                 output='epath')
                #             for index, path in zip(indices, paths):
                #                 if path:
                #                     dist = np.array(subgraph.es.get_attribute_values(criterion))[path].sum()
                #                     if dist < dist_thresh:
                    
                #     distance = row.thresh_dist
                    
                #     if paths:
                #         for path in paths:
                #             dist = weight[path].sum()
                #             if dist < distance:
                #                 distance = dist
                                
                #     if distance >= row.thresh_dist:
                #         adj_sub[source_ix_sub, target_ix_sub] = 0
                        
            # propagate capacities down from source --> target along adj
            capa_rec = func_capa.dot(adj_sub).toarray().squeeze()
            
            # functionality thesholds for recieved capacity
            func_thresh = np.array([row.thresh_func if vx['ci_type'] == row.target 
                                    else -9999 for vx in vs_source_target])
            
            # boolean vector whether received capacity great enough to supply endusers
            capa_suff = (capa_rec >= func_thresh).astype(int)
            
            # update funcstates
            # unclear if less faulty than just assigning directly to 
            # vs_source_target, which is a view of that vs.

            self.graph.vs[[vx.index for vx in vs_source_target]
                          ]['func_tot'] = [np.min([capa, func]) for capa, func
                                           in zip(capa_suff, vs_source_target['func_tot'])]  
         
    def _update_enduser_dependencies(self, df_dependencies):

        for __, row in df_dependencies[
                df_dependencies['type_I']=='enduser'].iterrows():
            
            vs_source_target = self.graph.vs.select(
                ci_type_in=[row.source, row.target])

            # adjacency matrix of selected subgraph
            adj_sub = np.array(vs_source_target.subgraph().get_adjacency().data)
            
            # Hadamard product func_tot (*) capacity
            func_capa = np.multiply(
                vs_source_target.get_attribute_values('func_tot'),
                vs_source_target.get_attribute_values(
                    f'capacity_{row.source}_{row.target}'))
            
            # for dependencies w/ access_cnstr, set (x,y) in adj to 0 if length of  
            # path x-->y now > thresh_dist
            if row.access_cnstr:
                
                # path search on subgraph with roads additionally
                # TODO: don't hard-code via-ci
                
                vlist_source_target = list(vs_source_target)
                vlist_via = list(self.graph.vs.select(ci_type='road'))
                vlist_source_target.extend(vlist_via)
                
                subgraph_2 = self.graph.subgraph(vlist_source_target)
                subgraph_2.delete_edges(subgraph_2.es.select(func_tot_lt=1))

                for source_ix_sub, target_ix_sub in tqdm(zip(*np.where(adj_sub==1)),
                                                         desc=f'checking road access {row.source}-{row.target}',
                                                         total=len(np.where(adj_sub==1)[0])):
                    # TODO: d
                    paths = subgraph_2.get_shortest_paths(
                        subgraph_2.vs[source_ix_sub],subgraph_2.vs[target_ix_sub],
                        subgraph_2.es.get_attribute_values('distance'),
                        mode='out', output='epath')
                    
                    distance = row.thresh_dist
                    
                    if paths:
                        for path in paths:
                            dist = np.array(subgraph_2.es.get_attribute_values('distance'))[path].sum()
                            if dist < distance:
                                distance = dist
                                
                    if distance >= row.thresh_dist:
                        adj_sub[source_ix_sub, target_ix_sub] = 0
                        
            # propagate capacities down from source --> target along adj
            capa_rec = np.dot(func_capa, adj_sub)   
            
            # functionality thesholds for recieved capacity
            func_thresh = np.array([row.thresh_func if vx['ci_type'] == row.target 
                                    else 0 for vx in vs_source_target])
            
            # boolean vector whether received capacity great enough to supply endusers
            capa_suff = (capa_rec >= np.array(func_thresh)).astype(int)
            
            # unclear if less faulty than just assigning directly to 
            # vs_source_target, which is a view of that vs.
            self.graph.vs[[vx.index for vx in vs_source_target]
                          ][f'actual_supply_{row.source}_{row.target}'] = capa_suff

    def _get_subgraph2graph_vsdict(self, vertex_seq):
        return dict((k,v) for k, v in zip(
            [subvx.index for subvx in self.graph.subgraph(vertex_seq).vs],
            [vx.index for vx in vertex_seq]))

    def cascade(self, df_dependencies, p_source='power plant', p_sink='substation', per_cap_cons=0.000079, source_var='el_gen_mw'):
        """
        wrapper for internal state update, functional dependency iterations,
        enduser dependency updates
        """
        delta = -1
        while delta != 0:
            func_state_tot_vs, func_state_tot_es = self._funcstates_sum()
            self._update_cis_internally(df_dependencies, p_source=p_source,
                                        p_sink=p_sink, per_cap_cons=per_cap_cons,
                                        source_var=source_var)
            LOGGER.info('Updating functional states.'+
                        f' Current delta before: {delta}')
            self._update_functional_dependencies(df_dependencies)
            func_state_tot_vs2, func_state_tot_es2 = self._funcstates_sum()
            delta_vs = func_state_tot_vs-func_state_tot_vs2
            delta_es = func_state_tot_es-func_state_tot_es2
            delta = max(abs(delta_vs), abs(delta_es))
            LOGGER.info(f' Current delta after: {delta}')

        self._update_enduser_dependencies(df_dependencies)

    def get_path_distance(self, epath):
        return sum(self.graph.es[epath]['distance'])
        
    def graph_style(self, ecol='red', vcol='red', vsize=2, ewidth=3, *kwargs):
        visual_style = {}
        visual_style["edge_color"] = ecol
        visual_style["vertex_color"] = vcol
        visual_style["vertex_size"] = vsize
        visual_style["edge_width"] = ewidth
        visual_style["layout"] = self.graph.layout("fruchterman_reingold")
        visual_style["edge_arrow_size"] = 0
        visual_style["edge_curved"] = 1
        if kwargs:
            pass
        return visual_style

    def plot_graph(self, *kwargs):
        return ig.plot(self.graph, **self.graph_style(*kwargs))
      
    def plot_multigraph(self,layer_dict,layout):    
        visual_style = {}
        visual_style["vertex_size"] = [layer_dict['vsize'][attr] for attr in
                                       self.graph.vs["ci_type"]]
        visual_style["edge_arrow_size"] = 1
        visual_style["edge_color"] = [layer_dict['edge_col'][attr] for attr in
                                      self.graph.vs["ci_type"]]
        visual_style["vertex_color"] = [layer_dict['vertex_col'][attr] for attr
                                        in self.graph.vs["ci_type"]]
        if layout == "fruchterman_reingold":
            visual_style["layout"] = self.graph.layout("fruchterman_reingold")
        elif layout == 'sugiyama':
            visual_style["layout"] = self.graph.layout_sugiyama(
                layers=[layer_dict['layers'][attr] for attr in 
                        self.graph.vs["ci_type"]])
        visual_style["edge_curved"] = 0.2
        visual_style["edge_width"] = 1
        
        return ig.plot(self.graph, **visual_style) 

    def _ckdnearest(self, vs_assign, gdf_base, k=1):
        """
        see https://gis.stackexchange.com/a/301935

        Parameters
        ----------
        vs_assign : gpd.GeoDataFrame or Point

        gdf_base : gpd.GeoDataFrame

        Returns
        ----------

        """
        # TODO: this should be a util function
        # TODO: this mixed input options (1 vertex vs gdf) is not nicely solved
        if (isinstance(vs_assign, gpd.GeoDataFrame) 
            or isinstance(vs_assign, pd.DataFrame)):
            n_assign = np.array(list(vs_assign.geometry.apply(lambda x: (x.x, x.y))))
        else:
            n_assign = np.array([(vs_assign.geometry.x, vs_assign.geometry.y)])
        n_base = np.array(list(gdf_base.geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(n_base)
        dist, idx = btree.query(n_assign, k=k)
        return dist, np.array(gdf_base.iloc[idx.flatten()].index).reshape(dist.shape)


    def make_edge_geometries(self, vs_geoms_from, vs_geoms_to):
        """
        create straight shapely LineString geometries between lists of 
        from and to nodes, to be added to newly created edges as attributes
        """
        return [shapely.geometry.LineString([geom_from, geom_to]) for
                                             geom_from, geom_to in
                                            zip(vs_geoms_from, vs_geoms_to)]
    
    def return_network(self):
        return Network.from_graphs([self.graph])
    

class Graph(GraphCalcs):
    """
    create an igraph Graph from network components
    """
    
    def __init__(self, network, directed=False):
        """
        network : instance of networks.base.Network"""
        if network.edges is not None:
            self.graph = self.graph_from_es(
                gdf_edges=network.edges, gdf_nodes=network.nodes, 
                directed=directed)
        else:
            self.graph = self.graph_from_vs(
                gdf_nodes=network.nodes, directed=directed)
    
    @staticmethod
    def graph_from_es(gdf_edges, gdf_nodes=None, directed=False):
        return ig.Graph.DataFrame(
            gdf_edges, vertices=gdf_nodes, directed=directed)
   
    @staticmethod
    def graph_from_vs(gdf_nodes, directed=False):
        vertex_attrs = gdf_nodes.to_dict('list')
        return ig.Graph(
            n=len(gdf_nodes),vertex_attrs=vertex_attrs, directed=directed)

