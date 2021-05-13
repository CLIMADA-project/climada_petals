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

"""Make graph base class"""

import igraph as ig
import numpy as np

def make_graph_from_es(gdf_edges, directed=False, vertices=None):
    # TODO: don't hard-code vs attribute
    # from_id and to_id need to be first two columns
    graph = ig.Graph.DataFrame(gdf_edges, directed=False, 
                              vertices=vertices)
    graph.vs['ci_type'] = gdf_edges.ci_type.iloc[0]
    graph.vs['orig_id'] = np.arange(len(graph.vs))
    
    return graph

def make_graph_from_vs(gdf_vertices):
    vertex_attrs = gdf_vertices.to_dict('list')
    vertex_attrs['orig_id'] = list(np.arange(len(gdf_vertices)))
    return ig.Graph(n=len(gdf_vertices),
                     vertex_attrs=vertex_attrs)

def graph_style(gdf, color, vsize=2, ewidth=3, *kwargs):
    visual_style = {}
    visual_style["edge_color"] = color
    visual_style["vertex_color"] = color
    visual_style["vertex_size"] = vsize
    visual_style["edge_width"] = ewidth
    visual_style["layout"] = gdf.layout("fruchterman_reingold")
    visual_style["edge_arrow_size"] = 0
    visual_style["edge_curved"] = 0
    if kwargs:
        pass    
    return visual_style