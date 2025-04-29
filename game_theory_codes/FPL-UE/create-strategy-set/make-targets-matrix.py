import os
from osgeo import gdal
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
from scipy.ndimage import distance_transform_edt






class LandUseRewards:

    def __init__(self, raster_path):
        self.raster_path = raster_path
        self.lulc_data = None
        self.geotransform = None
        self.load_raster()

    def load_raster(self):
        """Read and store LULC raster data"""
        try:
            ds = gdal.Open(self.raster_path)
            if ds is None:
                raise ValueError("Could not open raster file")

            self.lulc_data = ds.ReadAsArray()
            self.geotransform = ds.GetGeoTransform()
            self.projection = ds.GetProjection()
            ds = None

        except Exception as e:
            raise Exception(f"Error reading raster: {str(e)}")

    def _detect_forest_plantation_border(self):

        ds = os.path.join("game_theory_codes/FPL-UE/create-strategy-set/outputs/first_reachable_targets.tif")
        
        self.forest_agriculture_fringe = gdal.Open(ds).ReadAsArray()

        fig, ax = plt.subplots(figsize=(8, 8))

        img = ax.imshow(self.forest_agriculture_fringe, cmap="Greys_r", alpha=1)

        plt.colorbar(img, ax=ax, shrink=0.5)
        
        ax.set_axis_off()

        fig.savefig(os.path.join("game_theory_codes/FPL-UE/create-strategy-set/outputs/forest-agricultural-fringe.png"), dpi=300, bbox_inches='tight')

        return self.forest_agriculture_fringe
    
    def interpolate_lulc_matrix(self, target_shape):
        """Interpolate matrix to new dimensions"""
        if self.lulc_data is None:
            raise ValueError("Raster data not loaded")

        y, x = np.mgrid[0 : self.lulc_data.shape[0], 0 : self.lulc_data.shape[1]]
        points = np.column_stack((y.flat, x.flat))
        values = self.lulc_data.flat

        grid_y, grid_x = np.mgrid[0 : target_shape[0], 0 : target_shape[1]]
        scaling_y = self.lulc_data.shape[0] / target_shape[0]
        scaling_x = self.lulc_data.shape[1] / target_shape[1]

        grid_y = grid_y * scaling_y
        grid_x = grid_x * scaling_x

        return griddata(points, values, (grid_y, grid_x), method="nearest")

    def interpolate_forest_agriculture_fringe_matrix(self, target_shape):
    
        high_rows, high_cols = self.forest_agriculture_fringe.shape
        low_rows, low_cols = target_shape
        
        low_res_matrix = np.zeros(target_shape, dtype=self.forest_agriculture_fringe.dtype)
        
        row_block_size = high_rows / low_rows
        col_block_size = high_cols / low_cols
        
        for i in range(low_rows):
            for j in range(low_cols):
                row_start = int(i * row_block_size)
                row_end = int((i + 1) * row_block_size)
                col_start = int(j * col_block_size)
                col_end = int((j + 1) * col_block_size)

                block = self.forest_agriculture_fringe[row_start:row_end, col_start:col_end]
                
                if np.any(block == 1):
                    low_res_matrix[i, j] = 1
        
        return low_res_matrix

    def create_indexed_mapping(self, high_res_matrix, target_shape):
        
        high_rows, high_cols = high_res_matrix.shape
        low_rows, low_cols = target_shape
        
        low_res_binary = np.zeros(target_shape, dtype=high_res_matrix.dtype)
        
        row_block_size = high_rows / low_rows
        col_block_size = high_cols / low_cols

        for i in range(low_rows):
            for j in range(low_cols):
                row_start = int(i * row_block_size)
                row_end = int((i + 1) * row_block_size)
                col_start = int(j * col_block_size)
                col_end = int((j + 1) * col_block_size)
                
                block = high_res_matrix[row_start:row_end, col_start:col_end]
                
                if np.any(block == 1):
                    low_res_binary[i, j] = 1
        
        low_res_indexed = np.zeros_like(low_res_binary, dtype=int)
        high_res_indexed = np.zeros(high_res_matrix.shape, dtype=int)
        
        index = 1 
        
        for i in range(low_rows):
            for j in range(low_cols):
                if low_res_binary[i, j] == 1:

                    low_res_indexed[i, j] = index
                    
                    row_start = int(i * row_block_size)
                    row_end = int((i + 1) * row_block_size)
                    col_start = int(j * col_block_size)
                    col_end = int((j + 1) * col_block_size)

                    block_mask = high_res_matrix[row_start:row_end, col_start:col_end] == 1
                    high_res_indexed[row_start:row_end, col_start:col_end][block_mask] = index
                    
                    index += 1
        
        return low_res_binary, low_res_indexed, high_res_indexed

    def save_interpolated_matrix(self, interpolated_data, output_path, target_shape):
        """
        Save interpolated matrix as a GeoTIFF file
        
        Parameters:
        interpolated_data: numpy.ndarray - The interpolated data to save
        output_path: str - Path where the file will be saved
        target_shape: tuple - Target shape (rows, cols) used for interpolation
        """

        new_geotransform = list(self.geotransform)
        
        scaling_x = self.lulc_data.shape[1] / target_shape[1]
        scaling_y = self.lulc_data.shape[0] / target_shape[0]
        
        new_geotransform[1] = self.geotransform[1] * scaling_x  
        new_geotransform[5] = self.geotransform[5] * scaling_y  
        
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            output_path, 
            target_shape[1],
            target_shape[0],
            1,                
            gdal.GDT_Float32 
        )
        
        if out_ds is None:
            raise ValueError("Could not create output file")
        
        out_ds.SetGeoTransform(tuple(new_geotransform))
        out_ds.SetProjection(self.projection)
        
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(interpolated_data)
        
        out_ds = None

    def get_cell_value(self, row, col):
        """Get LULC code for specific cell"""
        if self.lulc_data is None:
            raise ValueError("Raster data not loaded")
        return self.lulc_data[row, col]

    def plot_lulc_matrices(self, interpolated=None):
        """Plot original and interpolated matrices side by side"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        data_value_map = {1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 9, 7: 10, 8: 14, 9: 15}

        data_LULC = self.lulc_data.copy()

        for i in range(1, 10):
            data_LULC[data_LULC == data_value_map[i]] = i

        levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        clrs = [
            "greenyellow",
            "mediumpurple",
            "turquoise",
            "plum",
            "black",
            "blue",
            "yellow",
            "mediumseagreen",
            "forestgreen",
        ]
        cmap, norm = colors.from_levels_and_colors(levels, clrs)

        im1 = ax1.imshow(data_LULC, cmap=cmap, norm=norm)
        ax1.set_title(f"Original Matrix {data_LULC.shape}")
        ax1.set_xticks([])
        ax1.set_yticks([])

        cbar = plt.colorbar(
            im1, ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9], fraction=0.046, pad=0.04, ax=ax1
        )
        cbar.ax.set_yticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        cbar.ax.set_yticklabels(
            [
                "Deciduous Broadleaf Forest",
                "Built-up Land",
                "Mixed Forest",
                "Shrubland",
                "Barren Land",
                "Water Bodies",
                "Plantations",
                "Grassland",
                "Broadleaf evergreen forest",
            ]
        )

        if interpolated is not None:

            interpolated_data = interpolated.copy()

            for i in range(1, 10):
                interpolated_data[interpolated_data == data_value_map[i]] = i

        if interpolated_data is not None:
            im2 = ax2.imshow(interpolated_data, cmap=cmap, norm=norm)
            ax2.set_title(f"Interpolated Matrix {interpolated_data.shape}")
            ax2.set_xticks([])
            ax2.set_yticks([])

            cbar = plt.colorbar(
                im2, ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9], fraction=0.046, pad=0.04, ax=ax2
            )
            cbar.ax.set_yticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9])
            cbar.ax.set_yticklabels(
                [
                    "Deciduous Broadleaf Forest",
                    "Built-up Land",
                    "Mixed Forest",
                    "Shrubland",
                    "Barren Land",
                    "Water Bodies",
                    "Plantations",
                    "Grassland",
                    "Broadleaf evergreen forest",
                ]
            )

        plt.tight_layout()
        plt.savefig(
            "game_theory_codes/FPL-UE/create-strategy-set/outputs/interpolated_LULC_matrix.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def plot_forest_agricultural_fringe_matrices(self, interpolated=None):
        """Plot original and interpolated matrices side by side"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        data = self.forest_agriculture_fringe.copy()

        im1 = ax1.imshow(data, cmap="Greys")
        ax1.set_title(f"Original Matrix {data.shape}")
        ax1.set_xticks([])
        ax1.set_yticks([])

        if interpolated is not None:

            interpolated_data = interpolated.copy()

        if interpolated_data is not None:
            im2 = ax2.imshow(interpolated_data,  cmap="Greys")
            ax2.set_title(f"Interpolated Matrix {interpolated_data.shape}")
            ax2.set_xticks([])
            ax2.set_yticks([])

        plt.tight_layout()
        plt.savefig(
            "game_theory_codes/FPL-UE/create-strategy-set/outputs/interpolated_forest_agricultural_fringe_matrix.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def get_matrix_indices(self, lat, lon):
        """Convert lat/lon to matrix indices"""
        x = int((lon - self.geotransform[0]) / self.geotransform[1])
        y = int((lat - self.geotransform[3]) / self.geotransform[5])
        return y, x

    def get_value_at_coords(self, lat, lon, interpolated=None):
        """Get value from original or interpolated matrix at coordinates"""
        y, x = self.get_matrix_indices(lat, lon)
        if interpolated is not None:
            scale_y = interpolated.shape[0] / self.lulc_data.shape[0]
            scale_x = interpolated.shape[1] / self.lulc_data.shape[1]
            y_interp = int(y * scale_y)
            x_interp = int(x * scale_x)
            return interpolated[y_interp, x_interp]
        else:
            return self.lulc_data[y, x]

    def plot_coords(self, lat, lon, interpolated=None):
        """Plot matrices with highlighted coordinates"""
        y, x = self.get_matrix_indices(lat, lon)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        data_value_map = {1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 9, 7: 10, 8: 14, 9: 15}

        data_LULC = self.lulc_data.copy()

        for i in range(1, 10):
            data_LULC[data_LULC == data_value_map[i]] = i

        levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        clrs = [
            "greenyellow",
            "mediumpurple",
            "turquoise",
            "plum",
            "black",
            "blue",
            "yellow",
            "mediumseagreen",
            "forestgreen",
        ]
        cmap, norm = colors.from_levels_and_colors(levels, clrs)

        im1 = ax1.imshow(data_LULC, cmap=cmap, norm=norm)
        ax1.axhline(y=y, color="r", linestyle="--", alpha=0.5)
        ax1.axvline(x=x, color="r", linestyle="--", alpha=0.5)
        ax1.plot(x, y, "r*", markersize=10)
        ax1.set_title("Original Matrix")
        ax1.set_xticks([])
        ax1.set_yticks([])

        cbar = plt.colorbar(
            im1, ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9], fraction=0.046, pad=0.04, ax=ax1
        )
        cbar.ax.set_yticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        cbar.ax.set_yticklabels(
            [
                "Deciduous Broadleaf Forest",
                "Built-up Land",
                "Mixed Forest",
                "Shrubland",
                "Barren Land",
                "Water Bodies",
                "Plantations",
                "Grassland",
                "Broadleaf evergreen forest",
            ]
        )

        if interpolated is not None:

            interpolated_data = interpolated.copy()

            for i in range(1, 10):
                interpolated_data[interpolated_data == data_value_map[i]] = i

            scale_y = interpolated_data.shape[0] / self.lulc_data.shape[0]
            scale_x = interpolated_data.shape[1] / self.lulc_data.shape[1]
            y_interp = int(y * scale_y)
            x_interp = int(x * scale_x)

            im2 = ax2.imshow(interpolated_data, cmap=cmap, norm=norm)
            ax2.axhline(y=y_interp, color="r", linestyle="--", alpha=0.5)
            ax2.axvline(x=x_interp, color="r", linestyle="--", alpha=0.5)
            ax2.plot(x_interp, y_interp, "r*", markersize=10)
            ax2.set_title(f"Interpolated Matrix")
            ax2.set_xticks([])
            ax2.set_yticks([])

            cbar = plt.colorbar(
                im2, ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9], fraction=0.046, pad=0.04, ax=ax2
            )
            cbar.ax.set_yticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9])
            cbar.ax.set_yticklabels(
                [
                    "Deciduous Broadleaf Forest",
                    "Built-up Land",
                    "Mixed Forest",
                    "Shrubland",
                    "Barren Land",
                    "Water Bodies",
                    "Plantations",
                    "Grassland",
                    "Broadleaf evergreen forest",
                ]
            )

        plt.tight_layout()
        plt.savefig(
            "game_theory_codes/FPL-UE/create-strategy-set/outputs/interpolated_LULC_matrix_value_at_coordinatex_x_"
            + str(x)
            + "_y_"
            + str(y)
            + ".png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def assign_target_rewards_and_penalties_random(
        self, targets_matrix):
        """
        Assign targetID, rewards and penalties for landuse cells.
        Ensures U_c_i > U_u_i for each target i, with values in [-0.5, 0.5]
        """
        
        target_cells = np.where(targets_matrix > 0)

        n_targets = len(target_cells[0])
        
        uncovered_utilities = np.random.uniform(-0.5, 0.45, n_targets)
        
        covered_utilities = np.array([
            np.random.uniform(uncovered_utilities[i] + 0.0001, 0.5)
            for i in range(n_targets)
        ])

        df = pd.DataFrame(
            {
                "targetID": range(1, n_targets + 1),
                "row": target_cells[0],
                "col": target_cells[1],
                "reward": covered_utilities, 
                "penalty": uncovered_utilities,  
            }
        )

        df.to_csv(
            "game_theory_codes/FPL-UE/create-strategy-set/outputs/target_rewards_penalties.csv",
            index=False,
        )
        return df





shape_of_coverage_matrix = (1069, 1070)

raster_path = os.path.join(
    "mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"
)

assign_rewards_and_penalties = LandUseRewards(raster_path)

lulc_interpolated = assign_rewards_and_penalties.interpolate_lulc_matrix(shape_of_coverage_matrix)

assign_rewards_and_penalties.save_interpolated_matrix(
    lulc_interpolated, "game_theory_codes/FPL-UE/create-strategy-set/outputs/interpolated_LULC_matrix.tif",
    shape_of_coverage_matrix
)

assign_rewards_and_penalties.plot_lulc_matrices(lulc_interpolated)

shape_of_coverage_matrix = (20, 20)

forest_agriculture_fringe = assign_rewards_and_penalties._detect_forest_plantation_border()

forest_agriculture_fringe_interpolated = assign_rewards_and_penalties.interpolate_forest_agriculture_fringe_matrix(shape_of_coverage_matrix)

assign_rewards_and_penalties.plot_forest_agricultural_fringe_matrices(forest_agriculture_fringe_interpolated)

print("Total number of targets:", np.sum(forest_agriculture_fringe_interpolated))

low_res_binary, low_res_indexed, high_res_indexed = assign_rewards_and_penalties.create_indexed_mapping(forest_agriculture_fringe, shape_of_coverage_matrix)

assign_rewards_and_penalties.save_interpolated_matrix(
    high_res_indexed, "game_theory_codes/FPL-UE/create-strategy-set/outputs/high_res_indexed_forest_agricultural_fringe.tif",
    (1069, 1070)
)

assign_rewards_and_penalties.save_interpolated_matrix(
    low_res_indexed, "game_theory_codes/FPL-UE/create-strategy-set/outputs/low_res_indexed_forest_agricultural_fringe.tif",
    shape_of_coverage_matrix
)

targets_df = assign_rewards_and_penalties.assign_target_rewards_and_penalties_random(
    low_res_indexed
)