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

# shortest paths summary table
df_shortest_paths = pd.DataFrame(columns=seq_pp['pp_name'],
                                 index=pd.Index(range(len(seq_hosp))
                                                ).set_names('hospitalID'))
for ix in range(len(seq_hosp)):
    df_shortest_paths.loc[ix] = graph.get_shortest_paths(
        seq_hosp[ix], to=seq_pp, mode='out', output='epath', weights="distance")

for orig in df_shortest_paths.index:
    for dest in df_shortest_paths.columns:
        edges['shortest_path_'+str(orig)+'_'+str(dest)] = \
            [x in df_shortest_paths.iloc[orig][dest] for x in edges.index]

df_shortest_dists = pd.DataFrame(columns=seq_pp['pp_name'],
                                 index=pd.Index(range(len(seq_hosp))
                                                ).set_names('hospitalID'))
for orig in df_shortest_dists.index:
    for dest in df_shortest_dists.columns:
        df_shortest_dists.loc[orig][dest] = edges[edges['shortest_path_'+str(orig)+'_'+str(dest)]]['distance'].sum()


# plot
edges = graph.get_edge_dataframe()
edges = pygeos_to_shapely(edges)

plot_shortest_paths(edges, gdf_beirut_health, gdf_lbn_pp, df_shortest_paths, 
                    [1,2,3,4,9], ['Baalback', 'Zouk 1'])