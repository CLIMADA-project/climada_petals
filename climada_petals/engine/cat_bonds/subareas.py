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

from climada_petals.util.config import LOGGER


class Subareas:
    """
    Handle subareas for CAT bond spatial aggregation.

    Parameters
    ----------
    hazard : climada.Hazard
        Hazard object containing event data. Attention: The date attribute is used
        in the CAT bond simulation. Date must include a year and month.
    vulnerability : climada.Vulnerability
        Vulnerability object containing vulnerability functions.
    exposure : climada.Exposure
        Exposure object containing monetary data and geometry.
    subareas_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing subarea polygons. Must cover the exposure
        perimeter and include (or will be assigned) a `subarea_letter` column.

    Attributes
    ----------
    hazard : climada.Hazard
        Hazard object containing event data. Attention: The date attribute is used
        in the CAT bond simulation. Date must include a year and month.
    vulnerability : climada.Vulnerability
        Vulnerability object containing vulnerability functions.
    exposure : climada.Exposure
        Exposure object containing monetary data and geometry.
    subareas_gdf : geopandas.GeoDataFrame
        GeoDataFrame of subarea polygons with `subarea_letter` labels.
    """
    

    def __init__(
        self,
        hazard,
        vulnerability,
        exposure,
        subareas_gdf: gpd.GeoDataFrame,
    ):
        """
        Initialize a Subareas instance.

        Parameters
        ----------
        hazard : climada.Hazard
            Hazard object containing event data. Attention: The date attribute is used
        in the CAT bond simulation. Date must include a year and month.
        vulnerability : climada.Vulnerability
            Vulnerability object containing vulnerability functions.
        exposure : climada.Exposure
            Exposure object containing monetary data and geometry.
        subareas_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing subarea polygons.
        """

        self.hazard = hazard
        self.vulnerability = vulnerability
        self._exposure = exposure
        self.subareas_gdf = subareas_gdf

    @classmethod
    def from_resolution(cls, hazard, vulnerability, exposure, resolution: float, subareas_gdf: gpd.GeoDataFrame | None = None):
        """
        Create subareas from a grid resolution and an exposure layer.

        Parameters
        ----------
        hazard : climada.Hazard
            Hazard object containing event data. Attention: The date attribute is used
        in the CAT bond simulation. Date must include a year and month.
        vulnerability : climada.Vulnerability
            Vulnerability object containing vulnerability functions.
        exposure : climada.Exposure
            Exposure object containing monetary data and geometry.
        resolution : float
            Grid cell size used to generate subareas.
        subareas_gdf : geopandas.GeoDataFrame, optional
            Unused placeholder for compatibility.

        Returns
        -------
        Subareas
            Initialized instance with generated subareas.
        """
        subareas_gdf = cls._init_subareas(exposure, resolution)

        return cls(hazard, vulnerability, exposure, subareas_gdf)
    
    @classmethod
    def from_geodataframe(cls, hazard, vulnerability, exposure, gdf: gpd.GeoDataFrame): 
        """
        Create subareas from an existing GeoDataFrame.

        Parameters
        ----------
        hazard : climada.Hazard
            Hazard object containing event data. Attention: The date attribute is used
        in the CAT bond simulation. Date must include a year and month.
        vulnerability : climada.Vulnerability
            Vulnerability object containing vulnerability functions.
        exposure : climada.Exposure
            Exposure object containing monetary data and geometry.
        gdf : geopandas.GeoDataFrame
            GeoDataFrame of polygon subareas covering the exposure perimeter.

        Returns
        -------
        Subareas
            Initialized instance with the provided subareas.
        """
        if (gdf.geometry.type != 'Polygon').any():
            raise ValueError("All geometries in the GeoDataFrame must be of type 'Polygon'.")
        exp_gdf = _create_exp_gdf(exposure)
        LOGGER.info("Number of polygons in exposure perimeter: %d", len(exp_gdf))
        if gdf.contains(exp_gdf.unary_union).all() is False:
            raise ValueError("The provided GeoDataFrame does not fully cover the exposure perimeter.")
        if 'subarea_letter' not in gdf.columns:
            gdf = gdf.copy()
            gdf["subarea_letter"] = [chr(65 + i) for i in range(len(gdf))]
        subareas_gdf = gdf.to_crs(exposure.gdf.crs)
        return cls(hazard, vulnerability, exposure, subareas_gdf)

    # --- Properties ---
    @property
    def exposure(self):
        """
        Return the exposure object.

        Returns
        -------
        climada.Exposure
            Exposure object containing monetary data and geometry.
        """
        return self._exposure

    def plot(self):
        """
        Plot exposure raster with subarea boundaries overlaid.

        Returns
        -------
        None
        """
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

            # Add padding (e.g. 5% wider and taller)
            pad_x = (xmax - xmin) * 0.05   # 10% horizontal padding
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
        """
        Return the number of subareas.

        Returns
        -------
        int
            Number of subarea polygons.
        """
        if self.subareas_gdf is None:
            raise ValueError("Subareas have not been generated yet.")
        else:
            return len(self.subareas_gdf)

    @staticmethod
    def _init_subareas(exposure, resolution: float):
        """
        Divides the exposure set into subareas and returns a geodataframe for the perimeter of exposed assets.

        Parameters
        ----------
        exposure : climada.Exposure
            Exposure object containing monetary data.
        resolution : float
            Resolution for grid cells to create subareas.

        Returns
        -------
        subareas_gdf : geopandas.GeoDataFrame
            Geodataframe of subareas covering the exposure perimeter.
        """
        exp_gdf = _create_exp_gdf(exposure)
        LOGGER.info("Number of polygons in exposure perimeter: %d", len(exp_gdf))
        subareas_gdf = _crop_grid_cells_to_polygon(resolution, exp_gdf, exposure)
        subareas_gdf["subarea_letter"] = [chr(65 + i) for i in range(len(subareas_gdf))]

        return subareas_gdf

def _crop_grid_cells_to_polygon(resolution: float, exp_gdf: gpd.GeoDataFrame, exposure):
    """
    Generate subarea grid cells from exposure perimeter polygons.

    For each polygon in `exp_gdf`, build a padded bounding box, create a regular
    grid of `resolution` cells over that box, and keep only cells that contain
    at least one exposure point. If the bounding box is smaller than one cell in
    either dimension, keep a single buffered bounding box cell. Finally, merge
    overlapping grid cells and drop empty geometries.

    Parameters
    ----------
    resolution : float
        Grid cell size used to generate subareas (CRS units).
    exp_gdf : geopandas.GeoDataFrame
        GeoDataFrame of perimeter polygons.
    exposure : climada.Exposure
        Exposure object with point geometries in `exposure.gdf`.

    Returns
    -------
    subareas : geopandas.GeoDataFrame
        GeoDataFrame of merged grid cells (subareas) covering exposure points.
    """
    cropped_cells = []
    exp_points = exposure.gdf.geometry
    spatial_index = exp_points.sindex
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
        if maxx - minx < resolution or maxy - miny < resolution:
            LOGGER.info(
                "Polygon smaller than resolution; adding polygon bounding box."
            )
            # Add a rectangle (bounding box) with 2% buffer around the polygon
            buffered_bbox = box(minx, miny, maxx, maxy)
            cropped_cells.append(
                gpd.GeoDataFrame(geometry=[buffered_bbox], crs=exp_gdf.crs)
            )
            continue
        
        n_cols = int(np.ceil((maxx - minx) / resolution))
        n_rows = int(np.ceil((maxy - miny) / resolution))

        grid_cells = []
        for x in range(n_cols):
            for y in range(n_rows):
                x1 = minx + x * resolution
                y1 = miny + y * resolution
                x2 = x1 + resolution
                y2 = y1 + resolution
                grid_cell = box(x1, y1, x2, y2)

                if len(spatial_index.query(grid_cell, predicate='intersects')) > 0:
                    grid_cells.append(grid_cell)
        grid_gdf = gpd.GeoDataFrame(
            grid_cells, columns=["geometry"], crs=exp_gdf.crs
        )

        cropped_cells.append(grid_gdf)

    grids = gpd.GeoDataFrame(
        pd.concat(cropped_cells, ignore_index=True), crs=exp_gdf.crs
    )

    # Merge overlapping grid cells into single polygons
    merged_grids = _merge_overlapping_grids(grids)
    merged_grids.reset_index(drop=True, inplace=True)
    subareas = merged_grids[~merged_grids.is_empty]
    subareas = subareas.reset_index(drop=True)
    LOGGER.info("Subareas created.")

    return subareas

def _create_exp_gdf(exposure) -> gpd.GeoDataFrame:
    """
    Build a polygonal footprint of exposed assets from an exposure GeoDataFrame.

    Rasterizes the exposure points on a grid whose resolution is inferred from
    nearest-neighbor spacing, then extracts contiguous regions where rasterized
    values are > 0. The regions are unioned and exploded into individual polygons.

    Parameters
    ----------
    exposure : climada.Exposure
        Exposure object containing point geometries and a `value` column.

    Returns
    -------
    exp_gdf : geopandas.GeoDataFrame
        GeoDataFrame of one or more polygons representing the exposure footprint,
        in the exposure CRS.
    """

    exp_gdf = exposure.gdf
    minx, miny, maxx, maxy = exp_gdf.total_bounds
    coords = np.vstack((exp_gdf.geometry.x, exp_gdf.geometry.y)).T
    nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    res = distances[:, 1].mean() * 1.2  
    width = max(int((maxx - minx) / res), 1)
    height = max(int((maxy - miny) / res),1)
    LOGGER.info(f"Rasterizing exposure with width: {width}, height: {height}, resolution: {res:.2f}")
    
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
    exp_gdf = gpd.GeoDataFrame(geometry=[merged_exp_gdf_sep], crs=exp_gdf.crs).explode(ignore_index=True, index_parts=True)
    LOGGER.info("Exposure perimeter polygon created.")

    return exp_gdf

def _merge_overlapping_grids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merges overlapping grid cells in a GeoDataFrame into single polygons using NetworkX.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing grid cell geometries.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with merged polygons.
    """

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
    return merged_gdf
