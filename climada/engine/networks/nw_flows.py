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

"""Implement flows / paths through graphs """
import geopandas as gpd
import pandas as pd
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import contextily as ctx


def make_ci_weights(graph, ci_types=None):
    """make additional weight attributes per edge-type"""
    if not ci_types:
        ci_types = np.unique(graph.es['ci_type'])
        # TODO: delete "dependency" from list
       
    for ci_type in ci_types:
        graph.es['weight_'+ci_type] = [1 if ci in (ci_type,'dependency') 
                                      else 10e8 for ci in graph.es['ci_type']]
    return graph
    
def shortest_path(graph, orig, dest, mode='out', weights=['distance'], output='epath'):
    if len(weights)==2:
        weights = [a*b for a,b in zip(graph.es[weights[0]],graph.es[weights[1]])]
    
    return graph.get_shortest_paths(orig, dest, weights, mode, output)
    

def shortest_paths(graph, from_ci, to_ci, via_ci=None, mode='out', output='epath'):
    vseq_from_ci = graph.vs.select(ci_type=from_ci)
    vseq_to_ci = graph.vs.select(ci_type=to_ci)
    pathlist=list()
    
    if via_ci:
        for vs in vseq_from_ci:
            pathlist.append(
                shortest_path(graph, vs, vseq_to_ci, 
                              weights=['distance','weight_'+via_ci],mode=mode,
                              output=output))
    else:
        for vs in vseq_from_ci:
            pathlist.append(
                shortest_path(graph, vs, vseq_to_ci, weights=['distance'], 
                              mode=mode, output=output))
    return pathlist

def make_df_pathsummary(graph,from_ci, to_ci, pathlist,output='epath'):
    if output == 'epath':
        return pd.DataFrame(pathlist,index=graph.vs.select(ci_type=from_ci)['orig_id'],
                            columns=graph.vs.select(ci_type=to_ci)['orig_id'])
    if output == 'distance':
        # TODO: fully vectorize this
        #np.broadcast_to(np.array(graph.es['distance']),(len(pathlist), len(pathlist[0])))
        path_dist = np.array([])
        dist_array = np.array(graph.es['distance'])       
        for i in pathlist:
            for j in i:
                path_dist = np.append(path_dist, dist_array[j].sum())
        path_dist = path_dist.reshape((len(pathlist), len(pathlist[0])))
        path_dist[path_dist == 0] = np.nan
        return pd.DataFrame(path_dist,
                            index=graph.vs.select(ci_type=from_ci)['orig_id'],
                            columns=graph.vs.select(ci_type=to_ci)['orig_id'])
    
def select_single_shortest_path(df_pathdists, df_epaths):
    dest_ix = df_pathdists.columns.values
    shortest_path = pd.DataFrame(index=df_pathdists.index, columns = ['path','distance'])
    for ix, row in df_pathdists.iterrows():
        min_dist = min(row)
        selected_dest = [(dist == min_dist) for dist in row]
        shortest_path.iloc[ix]['distance'] = min_dist
        shortest_path.iloc[ix]['path'] =  df_epaths.iloc[ix][selected_dest]       
    return shortest_path

         
def plot_shortest_paths(graph, from_ci, to_ci, via_ci, single_shortest_path):
    
    fig, ax = plt.subplots(figsize=(15, 15))    
    # via
    gpd.GeoDataFrame(geometry=graph.es.select(ci_type=via_ci)['geometry'],
                     crs=4326).to_crs(epsg=3857).plot(ax=ax, label=via_ci, color='black')
        
    for ix in single_shortest_path.index:
        gpd.GeoDataFrame(
            geometry=ig.EdgeSeq(graph, single_shortest_path.iloc[ix]['path'].values[0])['geometry']
            ,crs=4326).to_crs(epsg=3857).plot(ax=ax, color='green') #label=f'Shortest path from {from_ci} {ix} to {to_ci}'
    # origin
    gpd.GeoDataFrame(geometry=graph.vs.select(ci_type=from_ci)['geometry']
                     ,crs=4326).to_crs(epsg=3857).plot(ax=ax, label=from_ci, color='red')
    # destination
    gpd.GeoDataFrame(geometry=graph.vs.select(ci_type=to_ci)['geometry']
                     ,crs=4326).to_crs(epsg=3857).plot(ax=ax, label=to_ci, color='blue')
    ax.legend()
    ax.set(title=f'Shortest paths from {from_ci} to {to_ci} via {via_ci}')
    ctx.add_basemap(ax)
    plt.show()
    
graph = make_ci_weights(graph, ci_types=None)
pathlist = shortest_paths(graph,'health', 'power_plants', via_ci='power_lines')
df_health_to_pp_epath = make_df_pathsummary(graph,'health','power_plants',pathlist)
df_health_to_pp_dist = make_df_pathsummary(graph,'health','power_plants',pathlist,output='distance')
single_shortest_path = select_single_shortest_path(df_health_to_pp_dist,df_health_to_pp_epath)
plot_shortest_paths(graph, 'health', 'power_plants', 'power_lines', single_shortest_path)