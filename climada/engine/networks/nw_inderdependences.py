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
"""

"""Implement interdependences between graphs"""

from collections import OrderedDict
import geopandas as gpd
import igraph as ig
import numpy as np
from operator import itemgetter
import pandas as pd
from scipy.spatial import cKDTree
import shapely

from .nw_preprocessing import (pygeos_to_shapely,shapely_to_pygeos)


def assign_vs_geoms(graph):
    """
    For a graph where edges represent spatial lines, retrieve point 
    geometries of their vertices
    
    Parameters
    ----------
    graph : igraph.Graph
    
    Returns
    --------
    graph : igraph.Graph
        w/ updated vertex attribute "geometry"
    
    """
    
    gdf_es = graph.get_edge_dataframe()
    gdf_vs = graph.get_vertex_dataframe()

    gdf_es[['geometry_from','geometry_to']] = pd.DataFrame(gdf_es.apply(
        lambda row: (row.geometry.coords[0], 
                     row.geometry.coords[-1]), axis=1
        ).tolist(), index=gdf_es.index)
    
    vs_dict = OrderedDict(gdf_es[['source','geometry_from']].values.tolist())
    vs_dict.update(OrderedDict(gdf_es[['target','geometry_to']].values.tolist()))
    gdf_vs['geometry'] = itemgetter(*gdf_vs.index.values.tolist())(vs_dict)
    
    graph.vs['geometry'] = gdf_vs.apply(
        lambda row: shapely.geometry.Point(row.geometry), axis=1)

    return graph


def _ckdnearest(gdf_assign, gdf_base, ci_base):
    """ 
    see https://gis.stackexchange.com/a/301935 
    
    Parameters
    ----------
    gdf_base : gpd.GeoDataFrame
        
    gdf_assign : gpd.GeoDataFrame
    
    Returns
    ----------
    gpd.GeoDataFrame
    """
    n_assign = np.array(list(gdf_assign.geometry.apply(lambda x: (x.x, x.y))))
    n_base = np.array(list(gdf_base.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(n_base)
    dist, idx = btree.query(n_assign, k=1)
    return pd.concat([gdf_base.iloc[idx].reset_index(),
                      pd.Series(dist, name='distance')], axis=1)


def assign_closest_vs(graph_assign, graph_base):
    """
    match all vertices of graph_assign to closest vertices in graph_base.
    Updated in vertex attributes (vID of graph_base, geometry & distance)
    """
    gdf_vs_assign = graph_assign.get_vertex_dataframe()
    gdf_vs_base = graph_base.get_vertex_dataframe()
    
    ci_base = gdf_vs_base.ci_type.iloc[0]
    gdf_match = _ckdnearest(gdf_vs_assign, gdf_vs_base, ci_base)
    gdf_match = gdf_match[['geometry', 'orig_id', 'distance']].rename(
        {"geometry":"geometry_nearest_"+ci_base, 
          'orig_id':'orig_id_nearest_'+ci_base},axis=1)
    
    for col in gdf_match.columns:
        graph_assign.vs[col] = gdf_match[col]
    
    return graph_assign

def edges_from_closest_vs(graph_combined, from_ci, to_ci):
    """
    """
    vsseq_from_ci = graph_combined.vs.select(ci_type=from_ci)
    for vs1 in vsseq_from_ci:
        vs2 = graph_combined.vs.select(ci_type=to_ci, orig_id=vs1['orig_id_nearest_'+to_ci])[0]
        graph_combined.add_edge(vs1,vs2,directed=False,
                                geometry=shapely.geometry.LineString(
                                    [vs1['geometry'],vs2['geometry']]),
                                 ci_type='dependency',
                                 distance=vs1['distance'])
    return graph_combined


def plot_multigraph(graph,layer_dict,layout):    
    visual_style = {}
    visual_style["vertex_size"] = [layer_dict['vsize'][attr] for attr in graph.vs["ci_type"]]
    visual_style["edge_arrow_size"] = 1
    visual_style["edge_color"] = [layer_dict['edge_col'][attr] for attr in graph.vs["ci_type"]]
    visual_style["vertex_color"] = [layer_dict['vertex_col'][attr] for attr in graph.vs["ci_type"]]
    if layout == "fruchterman_reingold":
        visual_style["layout"] = graph.layout("fruchterman_reingold")
    elif layout == 'sugiyama':
        visual_style["layout"] = graph.layout_sugiyama(layers=[layer_dict['layers'][attr] for attr in graph.vs["ci_type"]])#
    visual_style["edge_curved"] = 0.2
    visual_style["edge_width"] = 1
    
    return ig.plot(graph, **visual_style) 

