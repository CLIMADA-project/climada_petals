import numpy as np
import geopandas as gpd
from shapely.geometry import box, shape
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from shapely.ops import unary_union
from rasterio.features import shapes, rasterize
from rasterio.transform import from_bounds
from sklearn.neighbors import NearestNeighbors
import cartopy.crs as ccrs
import networkx as nx

import logging

LOGGER = logging.getLogger(__name__)


# specify resultion to change exposure layer into country polygons
#tc_bound_resolution = 1000


class Subareas:

    '''Class to handle subareas for CAT bonds.
    
    Attributes
        ----------
        hazard : climada.Hazard
            Hazard object containing hazard data.
        vulnerability : climada.Vulnerability
            Vulnerability object containing vulnerability data.
        exposure : climada.Exposure
            Exposure object containing monetary data.
        resolution : float
            Resolution for grid cells to create subareas.
        crs : str, optional
            Coordinate reference system for spatial data (default: "EPSG:3857").
        subareas_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing the subareas as polygons.
        exp_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing the exposure perimeter as a polygon.
    '''
    

    def __init__(
        self,
        hazard,
        vulnerability,
        exposure,
        resolution,
    ):

        self.hazard = hazard
        self.vulnerability = vulnerability
        self._exposure = exposure
        self._resolution = resolution
        self._build_subareas()

    def _build_subareas(self):
        """Calculate subareas and islands."""
        self.subareas_gdf = self._init_subareas()

    # --- Properties with auto-rebuild ---
    @property
    def exposure(self):
        return self._exposure

    @property
    def resolution(self):
        return self._resolution

    def plot(self):
        if self.subareas_gdf is None:
            raise ValueError("Subareas have not been generated yet.")
        else:
            # Let plot_raster() create the correct cartopy GeoAxes
            ax = self._exposure.plot_raster()

            # Overlay subareas directly with the correct CRS transform
            self.subareas_gdf.plot(
                ax=ax,
                facecolor="none",
                edgecolor="red",
                lw=2,
                transform=ccrs.PlateCarree(),  # CLIMADA rasters use this by default
                zorder=5,
            )

            xmin1, ymin1, xmax1, ymax1 = self._exposure.gdf.total_bounds
            xmin2, ymin2, xmax2, ymax2 = self.subareas_gdf.total_bounds

            xmin = min(xmin1, xmin2)
            xmax = max(xmax1, xmax2)
            ymin = min(ymin1, ymin2)
            ymax = max(ymax1, ymax2)

            # 4️⃣ Add padding (e.g. 10% wider and 5% taller)
            pad_x = (xmax - xmin) * 0.1   # 10% horizontal padding
            pad_y = (ymax - ymin) * 0.05  # 5% vertical padding

            ax.set_extent(
                [xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y],
                crs=ccrs.PlateCarree()
            )

            # Add legend
            handles = [Line2D([0], [0], color="red", lw=2, label="Subareas")]
            ax.legend(handles=handles, loc="upper right")

            plt.show()

    def count_subareas(self):
        if self.subareas_gdf is None:
            raise ValueError("Subareas have not been generated yet.")
        else:
            return len(self.subareas_gdf)

    def _init_subareas(self):

        """
        Divides the exposure set into subareas and returns a geodataframe for the perimeter of exposed assets.

        Parameters
        ----------
        self : class instance
            Instance of the Subareas class.

        Returns
        -------
        subareas_gdf : GeoDataFrame
            Geodataframe of subareas covering the exposure perimeter.
        exp_gdf : GeoDataFrame
            Geodataframe of the exposure perimeter.
        """
        exp_gdf = self._create_exp_gdf()
        logging.info("Number of polygons in exposure perimeter: %d", len(exp_gdf))
        exp_gdf = exp_gdf.explode(ignore_index=True, index_parts=True)

        subareas_gdf = self._crop_grid_cells_to_polygon(
            exp_gdf
        )

        subareas_gdf["subarea_letter"] = [chr(65 + i) for i in range(len(subareas_gdf))]

        return subareas_gdf

    def _crop_grid_cells_to_polygon(self, exp_gdf):

        """
        Generates subareas based on exposure perimeter stored in a GeoDataFrame.
        This function takes a GeoDataFrame of polygons and, for each polygon, generates a grid of rectangular cells
        within its bounding box. Each grid cell is then cropped to the polygon's boundary using geometric intersection.
        For polygons smaller than a specified minimum area, the polygon itself is retained without cropping.
        The resulting grid cells are the subareas of the CAT bond.

        Parameters
        ----------
        self : class instance
            Instance of the Subareas class.
        exp_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing polygon geometries to be cropped into subareas.

        Returns
        -------
        subareas : geopandas.GeoDataFrame
            GeoDataFrame containing the cropped grid cells for all polygons, with empty geometries removed.
        """

        LOGGER.info("Creating subareas from exposure perimeter polygon.")
        cropped_cells = []

        LOGGER.info(f"Number of polygons to process: {len(exp_gdf)}")

        # Loop through each polygon in the GeoDataFrame
        for idx, polygon in exp_gdf.iterrows():
            
            # Pad the geometry bounds by 2% of width/height for better coverage
            minx, miny, maxx, maxy = polygon.geometry.bounds
            pad_x = (maxx - minx) * 0.02
            pad_y = (maxy - miny) * 0.02
            minx -= pad_x
            maxx += pad_x
            miny -= pad_y
            maxy += pad_y

            LOGGER.info(
                f"Processing polygon with bounds: {minx}, {miny}, {maxx}, {maxy}"
            )
            if maxx - minx < self._resolution or maxy - miny < self._resolution:
                LOGGER.info(
                    "Polygon smaller than resolution; adding polygon bounding box."
                )
                # Add a rectangle (bounding box) with 2% buffer around the polygon
                buffered_bbox = box(minx, miny, maxx, maxy)
                cropped_cells.append(
                    gpd.GeoDataFrame(geometry=[buffered_bbox], crs=exp_gdf.crs)
                )
                continue

            num_cells_x = int((maxx - minx) / self._resolution) + 1
            num_cells_y = int((maxy - miny) / self._resolution) + 1
            n_cols = int(np.ceil((maxx - minx) / self._resolution))
            n_rows = int(np.ceil((maxy - miny) / self._resolution))

            LOGGER.info(
                f"Number of cells in x direction: {num_cells_x}, y direction: {num_cells_y}"
            )   

            grid_cells = []
            for x in range(n_cols):
                for y in range(n_rows):

                    x1 = minx + x * self._resolution
                    y1 = miny + y * self._resolution
                    x2 = x1 + self._resolution
                    y2 = y1 + self._resolution

                    grid_cell = box(
                        x1, y1, x2, y2
                    )

                    # Only keep grid cell if at least one exposure point is inside
                    if any(p.within(grid_cell) for p in self.exposure.gdf.geometry):
                        grid_cells.append(grid_cell)

            grid_gdf = gpd.GeoDataFrame(
                grid_cells, columns=["geometry"], crs=exp_gdf.crs
            )

            cropped_cells.append(grid_gdf)

        grids = gpd.GeoDataFrame(
            pd.concat(cropped_cells, ignore_index=True), crs=exp_gdf.crs
        )
        # Merge overlapping grid cells into single polygons
        merged_grids = self.merge_overlapping_grids_nx(grids)
        merged_grids.reset_index(drop=True, inplace=True)
        subareas = merged_grids[~merged_grids.is_empty]
        subareas = subareas.reset_index(drop=True)

        LOGGER.info("Subareas created.")

        return subareas

    def _create_exp_gdf(self):

        """
        Generates a merged polygon representing the geometric extent of the exposed assets.
        This function rasterizes the geometries in the input exposure object, identifies contiguous regions
        where the exposure value is greater than zero, and merges these regions into a single polygon.
        The result is returned as a GeoDataFrame with the specified coordinate reference system.

        Parameters
        ----------
        self : class instance
            Instance of the Subareas class.

        Returns
        -------
        exp_gdf : geopandas.GeoDataFrame
            A GeoDataFrame containing a single merged polygon geometry representing the geometric extent of
            the country in the specified CRS.
        """
        LOGGER.info("Creating exposure perimeter polygon from exposure data.")
        exp_gdf = self._exposure.gdf
        minx, miny, maxx, maxy = exp_gdf.total_bounds

        LOGGER.info(f"Exposure total bounds: {minx}, {miny}, {maxx}, {maxy}")
        coords = np.vstack((exp_gdf.geometry.x, exp_gdf.geometry.y)).T
        nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        res = distances[:, 1].mean() * 1.2

        LOGGER.info(f"Approximate resolution: {res} CRS units")

        width = max(int((maxx - minx) / res), 1)
        height = max(int((maxy - miny) / res),1)
        LOGGER.info(f"Rasterizing exposure with width: {width}, height: {height}")
        
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        shapes_gen = (
            (geom, value) for geom, value in zip(exp_gdf.geometry, exp_gdf["value"])
        )

        raster = rasterize(
            shapes=shapes_gen,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="float32",
        )

        mask = raster > 0
        shapes_gen = list(shapes(raster, mask=mask, transform=transform))
        polygons = [shape(geom) for geom, value in shapes_gen if value > 0]
        exp_gdf_sep = gpd.GeoDataFrame(geometry=polygons, crs=exp_gdf.crs)
        merged_exp_gdf_sep = unary_union(exp_gdf_sep.geometry)
        exp_gdf = gpd.GeoDataFrame(geometry=[merged_exp_gdf_sep], crs=exp_gdf.crs)
        LOGGER.info("Exposure perimeter polygon created.")

        return exp_gdf


    def merge_overlapping_grids(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

        """
        Merges overlapping grid cells in a GeoDataFrame into single polygons.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing grid cell geometries.

        Returns
        -------
        merged_gdf : geopandas.GeoDataFrame
            GeoDataFrame with overlapping grid cells merged into single polygons.
        """
      
        LOGGER.info("Merging overlapping grid cells into single polygons.")

        geoms = gdf.geometry.tolist()
        to_delete = []
        for idx, geom in enumerate(geoms):
            for idx_inner, candidate in enumerate(geoms):
                if idx >= idx_inner:
                    continue
                else:
                    # Example grid cells
                    cell1 = box(geom.bounds[0], geom.bounds[1], geom.bounds[2], geom.bounds[3])
                    cell2 = box(candidate.bounds[0], candidate.bounds[1], candidate.bounds[2], candidate.bounds[3])
                    is_contained_1 = cell1.contains(cell2)
                    is_contained_2 = cell2.contains(cell1)
                    if is_contained_1:
                        to_delete.append(idx_inner)
                        continue
                    elif is_contained_2:
                        geoms[idx] = geoms[idx_inner]
                        to_delete.append(idx_inner)
                        continue
                    else:
                        # Calculate intersection area
                        overlap = cell1.intersection(cell2).area
                        print("Overlapping area:", overlap)
                        if overlap > 0:
                            geoms[idx] = geoms[idx].union(geoms[idx_inner])
                            to_delete.append(idx_inner)
                            continue
                        else:
                            continue
        geoms = [geom for i, geom in enumerate(geoms) if i not in to_delete]
        merged_gdf = gpd.GeoDataFrame(geometry=geoms, crs=gdf.crs)
        LOGGER.info("Merging completed.")
        return merged_gdf
                    
    def merge_overlapping_grids_nx(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        LOGGER.info("Merging overlapping grid cells into single polygons.")

        geoms = gdf.geometry.tolist()
        # Step 1: Remove polygons strictly within others
        to_remove = set()
        for i, geom in enumerate(geoms):
            for j, candidate in enumerate(geoms):
                if i == j or j in to_remove:
                    continue
                if geom.within(candidate):
                    to_remove.add(i)
                elif candidate.within(geom):
                    to_remove.add(j)
        geoms_filtered = [geom for i, geom in enumerate(geoms) if i not in to_remove]

        # Step 2: Merge polygons that overlap with positive area
        G = nx.Graph()
        G.add_nodes_from(range(len(geoms_filtered)))
        for i, geom in enumerate(geoms_filtered):
            for j, candidate in enumerate(geoms_filtered):
                if i >= j:
                    continue
                if geom.intersection(candidate).area > 1e-9:
                    G.add_edge(i, j)

        merged_polys = [
            gpd.GeoSeries([geoms_filtered[idx] for idx in comp]).unary_union
            for comp in nx.connected_components(G)
        ]

        merged_gdf = gpd.GeoDataFrame(geometry=merged_polys, crs=gdf.crs)
        LOGGER.info("Merging completed.")
        return merged_gdf
