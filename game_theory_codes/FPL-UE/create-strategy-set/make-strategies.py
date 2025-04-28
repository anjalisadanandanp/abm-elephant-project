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

    def _detect_forest_plantation_border(self, data, buffer=1, border_x=5,  border_y=5):
        forest_mask = (data == 15)  | (data == 5) | (data == 4) 
        plantation_mask = data == 10
        
        distance_from_forest = distance_transform_edt(~forest_mask)
        
        self.forest_agriculture_fringe = plantation_mask & (distance_from_forest <= buffer)

        #st all the values below and above a row value as zero
        self.forest_agriculture_fringe[:border_y, :] = 0
        self.forest_agriculture_fringe[-border_y:, :] = 0
        self.forest_agriculture_fringe[:, :border_x] = 0
        self.forest_agriculture_fringe[:, -border_x:] = 0

        fig, ax = plt.subplots(figsize=(8, 8))

        img = ax.imshow(self.forest_agriculture_fringe, cmap="Greys_r", alpha=1)

        plt.colorbar(img, ax=ax, shrink=0.5)
        
        ax.set_axis_off()

        fig.savefig(os.path.join("game_theory_codes/FPL-UE/create-strategy-set/outputs/forest-agricultural-fringe.png"), dpi=300, bbox_inches='tight')

        return
    
    def interpolate_matrix(self, target_shape):
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
        
        new_geotransform[1] = self.geotransform[1] * scaling_x  # Pixel width
        new_geotransform[5] = self.geotransform[5] * scaling_y  # Pixel height 
        
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            output_path, 
            target_shape[1],  # Width (cols)
            target_shape[0],  # Height (rows)
            1,                # Number of bands
            gdal.GDT_Float32  # Data type (change as needed)
        )
        
        if out_ds is None:
            raise ValueError("Could not create output file")
        
        out_ds.SetGeoTransform(tuple(new_geotransform))
        out_ds.SetProjection(self.projection)
        
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(interpolated_data)
        
        out_ds = None
        # print(f"Interpolated matrix saved to {output_path}")

    def get_cell_value(self, row, col):
        """Get LULC code for specific cell"""
        if self.lulc_data is None:
            raise ValueError("Raster data not loaded")
        return self.lulc_data[row, col]

    def plot_matrices(self, interpolated=None):
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
        self, target_value=10, interpolated=None):
        """
        Assign targetID, rewards and penalties for landuse cells.
        Ensures U_c_i > U_u_i for each target i, with values in [-0.5, 0.5]
        """
        if interpolated is None:
            raise ValueError("Interpolated matrix required")

        # target_cells = np.where(interpolated == target_value) 

        target_cells = np.where(self.forest_agriculture_fringe == 1)

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
                "reward": covered_utilities,  # U_c_i
                "penalty": uncovered_utilities,  # U_u_i
            }
        )

        df.to_csv(
            "game_theory_codes/FPL-UE/create-strategy-set/outputs/target_rewards_penalties.csv",
            index=False,
        )
        return df

    def plot_with_rewards(self, targets_df, interpolated=None, name=None):
        """Plot matrices with rewards and penalties as text overlays"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        if interpolated is not None:

            data_value_map = {1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 9, 7: 10, 8: 14, 9: 15}
            interpolated_data = interpolated.copy()

            for i in range(1, 10):
                interpolated_data[interpolated_data == data_value_map[i]] = i

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

            im1 = ax1.imshow(interpolated_data, cmap=cmap, norm=norm)
            im2 = ax2.imshow(interpolated_data, cmap=cmap, norm=norm)

            for _, row in targets_df.iterrows():
                ax1.text(
                    row["col"],
                    row["row"],
                    f'{row["reward"]:.1f}',
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=4,
                )

                ax2.text(
                    row["col"],
                    row["row"],
                    f'{row["penalty"]:.1f}',
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=4,
                )

            ax1.set_title("Rewards Distribution")
            ax2.set_title("Penalties Distribution")

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

            cbar2 = plt.colorbar(
                im2, ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9], fraction=0.046, pad=0.04, ax=ax2
            )
            cbar2.ax.set_yticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9])
            cbar2.ax.set_yticklabels(
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
                "game_theory_codes/FPL-UE/create-strategy-set/outputs/"
                + name
                + "_rewards_penalties.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        return






shape_of_coverage_matrix = (250, 250)

raster_path = os.path.join(
    "mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"
)

assign_rewards_and_penalties = LandUseRewards(raster_path)

interpolated = assign_rewards_and_penalties.interpolate_matrix(shape_of_coverage_matrix)

assign_rewards_and_penalties.save_interpolated_matrix(
    interpolated, "game_theory_codes/FPL-UE/create-strategy-set/outputs/interpolated_LULC_matrix.tif",
    shape_of_coverage_matrix
)

assign_rewards_and_penalties.plot_matrices(interpolated)

assign_rewards_and_penalties._detect_forest_plantation_border(interpolated, buffer=1)

assign_rewards_and_penalties.save_interpolated_matrix(
    assign_rewards_and_penalties.forest_agriculture_fringe, "game_theory_codes/FPL-UE/create-strategy-set/outputs/potential_targets_matrix.tif",
    shape_of_coverage_matrix
)

targets_df = assign_rewards_and_penalties.assign_target_rewards_and_penalties_random(
    target_value=10, interpolated=interpolated
)

assign_rewards_and_penalties.plot_with_rewards(
    targets_df, interpolated, name="defender"
)