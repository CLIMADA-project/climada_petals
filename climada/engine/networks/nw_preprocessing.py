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
import shapely
import pygeos
import pandas as pd
import geopandas as gpd

# =============================================================================
# Util funcs
# =============================================================================

def _pygeos_to_shapely(geom):
    """geometry conversion"""
    return shapely.wkt.loads(pygeos.io.to_wkt(geom))

def _shapely_to_pygeos(geom):
    """geometry conversion"""
    return pygeos.io.from_wkt(geom.wkt)

    
def pygeos_to_shapely(df, colname='geometry'):
    """dataframe conversion"""
    gdf = gpd.GeoDataFrame(df)
    gdf[colname] = gdf.apply(lambda row: _pygeos_to_shapely(row[colname]), axis=1)
    return gdf

def shapely_to_pygeos(gdf, colname='geometry'):
    """dataframe conversion"""
    df = pd.DataFrame(gdf)
    df[colname] = df.apply(lambda row: _shapely_to_pygeos(row[colname]), axis=1)
    return df

# =============================================================================
# DF operations to add nodes & edges info, other relevant attr info to gdf
# =============================================================================

# instantiate nw class?

def consolidate_ci_attrs(gdf, ci_type=None, to_drop=None, to_keep=None):
    if ci_type:
        gdf['ci_type'] = ci_type
    if to_drop:
        gdf = gdf.drop(to_drop, axis=1)
    if to_keep:
        gdf = gdf[to_keep]
    return gdf

# =============================================================================
# DF operations to clean and simplify 
# =============================================================================

# adding endpoints
# splitting and merging lines where sensible
# simplifying structures (loops, curves, deg 2 nodes, etc.)


# =============================================================================
# DF operations to add topology
# =============================================================================

# from id / to id


# =============================================================================
# Wrappers tailored to CI types
# =============================================================================

""" individual steps, from flow_model.load_network(): for roads """
# nw_beirut_roads = simplify.Network(edges=df_beirut_roads)
# nw_beirut_roads = simplify.add_endpoints(nw_beirut_roads)
# nw_beirut_roads = simplify.split_edges_at_nodes(nw_beirut_roads)
# nw_beirut_roads = simplify.clean_roundabouts(nw_beirut_roads)
# nw_beirut_roads = simplify.add_ids(nw_beirut_roads)
# nw_beirut_roads = simplify.add_topology(nw_beirut_roads)
# nw_beirut_roads = simplify.drop_hanging_nodes(nw_beirut_roads)
# nw_beirut_roads = simplify.merge_edges(nw_beirut_roads)
# nw_beirut_roads = simplify.reset_ids(nw_beirut_roads) 
# nw_beirut_roads = simplify.add_distances(nw_beirut_roads)
# nw_beirut_roads = simplify.merge_multilinestrings(nw_beirut_roads)
# nw_beirut_roads = simplify.fill_attributes(nw_beirut_roads)
# nw_beirut_roads = simplify.add_travel_time(nw_beirut_roads)   

""" individual steps, from flow_model.load_network(): for power lines """
# nw_lbn_hvmv = simplify.Network(edges=df_lbn_hvmv)
# nw_lbn_hvmv = simplify.add_endpoints(nw_lbn_hvmv)
# nw_lbn_hvmv = simplify.split_edges_at_nodes(nw_lbn_hvmv)
# nw_lbn_hvmv = simplify.add_ids(nw_lbn_hvmv)
# nw_lbn_hvmv = simplify.add_topology(nw_lbn_hvmv)
# nw_lbn_hvmv = simplify.drop_hanging_nodes(nw_lbn_hvmv)
# nw_lbn_hvmv = simplify.reset_ids(nw_lbn_hvmv) 
# nw_lbn_hvmv = simplify.add_distances(nw_lbn_hvmv)