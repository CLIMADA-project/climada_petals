import numpy as np
import geopandas as gpd
from shapely.geometry import box, shape
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from shapely.ops import unary_union
from rasterio.features import shapes, rasterize
from rasterio.transform import from_bounds


# specify resultion to change exposure layer into country polygons
resolution = 1000


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
        grid_specs : dict or object
            Specifications for the spatial grid (number of rows and columns). Defines count of subaraeas.
        buffer_grid_size : int, optional
            Size of the buffer around input country. Resulting geometry is used to derive subareas (in km; default is 5).
        min_pol_size : int, optional
            Minimum polygon size for subareas (default: 1000 square meters).
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
        grid_specs,
        buffer_grid_size=5.0,
        min_pol_size=1000,
        crs="EPSG:3857",
    ):

        self.hazard = hazard
        self.vulnerability = vulnerability
        self._exposure = exposure
        self._grid_specs = grid_specs
        self._buffer_grid_size = buffer_grid_size
        self._min_pol_size = min_pol_size
        self._crs = crs
        self._build_subareas()

    def _build_subareas(self):
        """Recalculate subareas and islands."""
        self.subareas_gdf, self.exp_gdf = self._init_subareas()

    # --- Properties with auto-rebuild ---
    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, value):
        self._exposure = value
        self._build_subareas()

    @property
    def grid_specs(self):
        return self._grid_specs

    @grid_specs.setter
    def grid_specs(self, value):
        self._grid_specs = value
        self._build_subareas()

    @property
    def buffer_grid_size(self):
        return self._buffer_grid_size

    @buffer_grid_size.setter
    def buffer_grid_size(self, value):
        self._buffer_grid_size = value
        self._build_subareas()

    @property
    def min_pol_size(self):
        return self._min_pol_size

    @min_pol_size.setter
    def min_pol_size(self, value):
        self._min_pol_size = value
        self._build_subareas()

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, value):
        self._crs = value
        self._build_subareas()

    def plot(self):
        if self.subareas_gdf is None:
            raise ValueError("Subareas have not been generated yet.")
        else:
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            self.exp_gdf.plot(ax=ax, color="green", label="Exposure")
            self.subareas_gdf.plot(
                ax=ax, facecolor="none", edgecolor="red", lw=2, label="Subarea"
            )
            handles = [
                Line2D([0], [0], color="green", lw=4, label="Exposure"),
                Line2D([0], [0], color="red", lw=2, label="Subareas"),
            ]
            ax.legend(handles=handles, loc="upper right")
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.set_yticks(ax.get_yticks()[1:])
            ax.set_xticks(ax.get_xticks())
            xlabel = ax.get_xticks()
            new_xlabel = []
            for label in xlabel:
                new_xlabel.append(str(round(-label, 1)) + "°W")
            ax.set_xticklabels(new_xlabel)
            ylabel = ax.get_yticks()
            new_ylabel = []
            for label in ylabel:
                new_ylabel.append(str(round(-label, 1)) + "°S")
            ax.set_yticklabels(new_ylabel)
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
        exp_gdf = exp_gdf.explode(ignore_index=True, index_parts=True)
        buffered_geometries = exp_gdf.geometry.buffer(self._buffer_grid_size * 1000)
        exp_gdf = unary_union(buffered_geometries)
        exp_gdf = gpd.GeoDataFrame({"geometry": [exp_gdf]}, crs=self._crs).explode(
            index_parts=True
        )
        subareas_gdf = self._crop_grid_cells_to_polygon(
            exp_gdf
        )
        if self._crs == "EPSG:3857":
            exposure_crs = self._exposure.crs
            exp_gdf = exp_gdf.to_crs(exposure_crs)
            subareas_gdf = subareas_gdf.to_crs(exposure_crs)
        subareas_gdf["subarea_letter"] = [chr(65 + i) for i in range(len(subareas_gdf))]

        return subareas_gdf, exp_gdf

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

        cropped_cells = []

        # Loop through each polygon in the GeoDataFrame
        for idx, polygon in exp_gdf.iterrows():
            polygon_area_km2 = polygon.geometry.area / 1e6
            if polygon_area_km2 < self._min_pol_size:
                grid_gdf = gpd.GeoDataFrame(
                    {"geometry": [polygon.geometry]}, crs=exp_gdf.crs
                )
                cropped_cells.append(grid_gdf)
            else:
                minx, miny, maxx, maxy = polygon.geometry.bounds

                num_cells_x = self._grid_specs[0]
                num_cells_y = self._grid_specs[1]
                x_coords = np.linspace(minx, maxx, num_cells_x + 1)
                y_coords = np.linspace(miny, maxy, num_cells_y + 1)

                grid_cells = []
                for i in range(num_cells_x):
                    for j in range(num_cells_y):
                        grid_cell = box(
                            x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1]
                        )
                        cell_cropped = grid_cell.intersection(polygon.geometry)
                        grid_cells.append(cell_cropped)

                grid_gdf = gpd.GeoDataFrame(
                    grid_cells, columns=["geometry"], crs=exp_gdf.crs
                )

                cropped_cells.append(grid_gdf)

        grids = gpd.GeoDataFrame(
            pd.concat(cropped_cells, ignore_index=True), crs=exp_gdf.crs
        )
        grids.reset_index(drop=True, inplace=True)
        subareas = grids[~grids.is_empty]
        subareas = subareas.reset_index(drop=True)

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

        exp_crs = self._exposure.gdf.to_crs(self._crs)
        minx, miny, maxx, maxy = exp_crs.total_bounds

        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)

        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        shapes_gen = (
            (geom, value) for geom, value in zip(exp_crs.geometry, exp_crs["value"])
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
        exp_gdf_sep = gpd.GeoDataFrame(geometry=polygons, crs=self._crs)
        merged_exp_gdf_sep = unary_union(exp_gdf_sep.geometry)
        exp_gdf = gpd.GeoDataFrame(geometry=[merged_exp_gdf_sep], crs=self._crs)

        return exp_gdf
