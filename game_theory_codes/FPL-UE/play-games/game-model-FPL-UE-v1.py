import os
from osgeo import gdal
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import importlib
from itertools import combinations
from typing import Set
from typing import Iterator
import pathlib
import yaml
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time
import random
from math import comb

import warnings
warnings.filterwarnings("ignore")


fontsize = 8
plt.rcParams.update(
    {
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "figure.titlesize": fontsize,
    }
)


import sys
sys.path.append(os.getcwd())

module = importlib.import_module('game_theory_codes.FPL-UE.play-games.abm_model_HEC_with_landscape_deterrent_policies')
batch_run_model = module.batch_run_model


    


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

    def _detect_forest_plantation_border(self, data, buffer=1, border_x=25,  border_y=25):
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

        fig.savefig(os.path.join("game_theory_codes/FPL-UE/outputs/forest-agricultural-fringe.png"), dpi=300, bbox_inches='tight')

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
            "game_theory_codes/FPL-UE/outputs/interpolated_LULC_matrix.png",
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
            "game_theory_codes/FPL-UE/outputs/interpolated_LULC_matrix_value_at_coordinatex_x_"
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
            "game_theory_codes/FPL-UE/outputs/target_rewards_penalties.csv",
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
                "game_theory_codes/FPL-UE/outputs/"
                + name
                + "_rewards_penalties.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        return







def create_defender_coverage_matrix(defender_strategy, grid_shape):

    coverage_matrix = np.zeros(grid_shape, dtype=np.int8)

    lulc_data = gdal.Open("game_theory_codes/FPL-UE/outputs/potential_targets_matrix.tif").ReadAsArray()
    plantation_rows, plantation_cols = np.where(lulc_data == 1)

    for i, (row, col) in enumerate(zip(plantation_rows, plantation_cols)):
        if defender_strategy[i] == 1:
            coverage_matrix[row, col] = 1
        else:   
            coverage_matrix[row, col] = 0

    return coverage_matrix

def plot_and_save_defender_coverage(coverage_matrix, output_folder, figsize=(8, 8), 
                          protected_color='red', unprotected_color='white'):





    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = mcolors.ListedColormap([unprotected_color, protected_color])
    
    im = ax.imshow(coverage_matrix, cmap=cmap, vmin=0, vmax=1)
    
    ax.set_xticks([])
    ax.set_yticks([])

    legend_elements = [
        Patch(facecolor=protected_color, edgecolor='black', label='Protected'),
        Patch(facecolor=unprotected_color, edgecolor='black', label='Unprotected')
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.savefig(
        os.path.join(output_folder, "defender_coverage_matrix.png"),
        bbox_inches="tight",
        dpi=300,
    )





    source_file = gdal.Open("game_theory_codes/FPL-UE/outputs/interpolated_LULC_matrix.tif")

    cols = source_file.RasterXSize
    rows = source_file.RasterYSize
    projection = source_file.GetProjection()
    geotransform = source_file.GetGeoTransform()

    output_file = "game_theory_codes/FPL-UE/outputs/coverage_matrix.tif"

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(coverage_matrix.astype(np.uint8))

    source_file = None
    output_dataset = None





    output_file = os.path.join(output_folder, "defender_coverage_matrix.tif")

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(coverage_matrix.astype(np.uint8))

    source_file = None
    output_dataset = None

    return 

def combination_to_binary_vector(combination, num_landscape_cells):

    binary_vector = np.zeros(num_landscape_cells, dtype=int)
    binary_vector[list(combination)] = 1
    return binary_vector

def generate_defender_strategies_using_attack_probabilities(output_folder, n_attack_locations) -> Iterator[np.ndarray]:

    potential_coverage_matrix = gdal.Open(os.path.join("game_theory_codes/FPL-UE/outputs/potential_targets_matrix.tif")).ReadAsArray()

    game_step_folders = os.listdir(output_folder)
    game_step_folders.sort()

    attacker_strategy_matrix_history = np.zeros_like(potential_coverage_matrix)

    for game_step_folder in game_step_folders:

        attacker_strategy_matrix = gdal.Open(os.path.join(output_folder, game_step_folder, "attacker_strategy_matrix.tif")).ReadAsArray()

        mask = potential_coverage_matrix == 1
        attacker_strategy_matrix_history[mask] += attacker_strategy_matrix[mask]

    attack_location_indices = np.where(attacker_strategy_matrix_history > 0)
    attack_locations = [(row, col) for row, col in zip(attack_location_indices[0], attack_location_indices[1])]
    
    print("Number of attack locations:", len(attack_locations))
    
    # probabilities = [attacker_strategy_matrix_history[row, col] for row, col in attack_locations]

    # probabilities = probabilities / np.sum(probabilities)

    # sampled_attack_locations = np.random.choice(
    #     len(attack_locations), 
    #     size=n_attack_locations, 
    #     replace=False,
    #     p=probabilities
    # )

    # sampled_attack_locations = [attack_locations[i] for i in sampled_attack_locations]

    # sampled_attack_locations = np.argsort(probabilities)[-n_attack_locations:]

    return attack_locations

def calculate_reward_for_strategy(attacked_locations, NUM_LANDSCAPE_CELLS, perturbed_reward):
    v = combination_to_binary_vector(attacked_locations, NUM_LANDSCAPE_CELLS)
    v = np.array(v)
    total_reward = np.dot(v, perturbed_reward)
    return total_reward, v

def find_best_strategy_parallel(attack_locations, budget_k, NUM_LANDSCAPE_CELLS, perturbed_reward, n_processes=16):
    
    process_func = partial(
        calculate_reward_for_strategy,
        NUM_LANDSCAPE_CELLS=NUM_LANDSCAPE_CELLS,
        perturbed_reward=perturbed_reward
    )
    
    max_reward = float('-inf')
    best_strategy = None
    
    with mp.Pool(processes=n_processes) as pool:

        combination_generator = combinations(attack_locations, budget_k)

        for total_reward, v in tqdm(pool.imap(process_func, combination_generator, chunksize=512)):
            if total_reward > max_reward:
                max_reward = total_reward
                best_strategy = v
    
    return best_strategy

def select_defender_strategy(
    attack_locations,
    estimated_reward: np.ndarray,
    eta: float,
    gamma,
    NUM_LANDSCAPE_CELLS,
    budget_k
    ) -> np.ndarray:


    flag = np.random.random() < gamma 

    if flag: 

        potential_coverage_matrix = gdal.Open(os.path.join("game_theory_codes/FPL-UE/outputs/potential_targets_matrix.tif")).ReadAsArray()
        target_location_indices = np.where(potential_coverage_matrix == 1)
        target_locations = [(row, col) for row, col in zip(target_location_indices[0], target_location_indices[1])]

        random_sample = random.sample(target_locations, budget_k)

        v_t = combination_to_binary_vector(random_sample, NUM_LANDSCAPE_CELLS)

    else:  

        n = len(estimated_reward)
        z = np.random.exponential(scale=1/eta, size=n)
        
        perturbed_reward = estimated_reward + z

        v_t = find_best_strategy_parallel(attack_locations, budget_k, NUM_LANDSCAPE_CELLS, perturbed_reward)

    return v_t

def evaluate_strategy(strategy, perturbed_reward):
    strategy = np.array(strategy)
    total_reward = np.dot(strategy, perturbed_reward)
    return (total_reward, strategy)

def run_abm(model_params, experiment_name, output_folder, shape_of_coverage_matrix):

    with open(os.path.join(output_folder, "model_parameters.yaml"), "w") as configfile:
        yaml.dump(model_params, configfile, default_flow_style=False)




    batch_run_model(model_params, experiment_name, output_folder)




    matrix = np.zeros(shape_of_coverage_matrix, dtype=np.uint8)

    for simulation_folder in os.listdir(output_folder):

        try:

            df = pd.read_csv(os.path.join(output_folder, simulation_folder, "output_files/agent_data.csv"))
            df.dropna(subset=['ROW', 'COL'], inplace=True)
        
            rows = df['ROW'].astype(int).values
            cols = df['COL'].astype(int).values
        
            mask = (0 <= rows) & (0 <= cols)
            valid_rows = rows[mask]
            valid_cols = cols[mask]
            
            matrix[valid_rows, valid_cols] += 1
        
        except Exception as e:
            pass

    interpolated = gdal.Open("game_theory_codes/FPL-UE/outputs/potential_targets_matrix.tif").ReadAsArray()

    mask = interpolated != 1
    matrix[mask] = 0

    total_attacks = np.sum(matrix)
    matrix = matrix/total_attacks



    fig, ax = plt.subplots(figsize=(8, 8))
    
    cmap = 'coolwarm'
    
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=np.max(matrix))
    
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = plt.colorbar(im, shrink=0.5)

    ticks = np.linspace(0, np.max(matrix), num=5) 
    cbar.set_label("Attack Probability", rotation=90)
    cbar.set_ticks(ticks)

    plt.savefig(
        os.path.join(output_folder, "attacker_strategy_matrix.png"),
        bbox_inches="tight",
        dpi=300,
    )




    source_file = gdal.Open("game_theory_codes/FPL-UE/outputs/interpolated_LULC_matrix.tif")

    cols = source_file.RasterXSize
    rows = source_file.RasterYSize
    projection = source_file.GetProjection()
    geotransform = source_file.GetGeoTransform()

    output_file = os.path.join(output_folder, "attacker_strategy_matrix.tif")

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Float32)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(matrix.astype(np.float32))

    source_file = None
    output_dataset = None




    lulc_data = gdal.Open("game_theory_codes/FPL-UE/outputs/interpolated_LULC_matrix.tif").ReadAsArray()
    coverage_matrix = gdal.Open("game_theory_codes/FPL-UE/outputs/potential_targets_matrix.tif").ReadAsArray()

    plantation_rows, plantation_cols = np.where(lulc_data == 10)

    attacker_strategy = []
    for row, col in zip(plantation_rows, plantation_cols):
        if coverage_matrix[row, col] == 1:
            if matrix[row, col] > 0:
                attacker_strategy.append(1)
            else:
                attacker_strategy.append(0)

    return attacker_strategy

def step_utility_defender(attacker_strategy_i, defender_strategy_i, targets_df):

    attacker_strategy = np.array(attacker_strategy_i)
    defender_strategy = np.array(defender_strategy_i)

    r = (targets_df['reward'] - targets_df['penalty']).values
    r_t = [a * b for a, b in zip(attacker_strategy, r)]

    reward_01 = np.dot(defender_strategy, r_t)
    reward_02 = np.dot(attacker_strategy, targets_df['penalty'].values)

    return reward_01 + reward_02

def calculate_reward_for_strategy_best(location, NUM_LANDSCAPE_CELLS, attacker_strategy_history, targets_df):

    v = combination_to_binary_vector(location, NUM_LANDSCAPE_CELLS)
    v = np.array(v)
    
    total_strategy_utility = 0
    for attacker_strategy in attacker_strategy_history:
        step_utility = step_utility_defender(attacker_strategy, v, targets_df)
        total_strategy_utility += step_utility
    
    return total_strategy_utility, v
    
def calculate_best_strategy_v2(NUM_LANDSCAPE_CELLS, attack_locations, attacker_strategy_history, targets_df, budget_k, n_processes=16):

    process_func = partial(
        calculate_reward_for_strategy_best,
        NUM_LANDSCAPE_CELLS=NUM_LANDSCAPE_CELLS,
        attacker_strategy_history=attacker_strategy_history,
        targets_df=targets_df
    )
    
    max_reward = float('-inf')
    best_strategy = None
    
    with mp.Pool(processes=n_processes) as pool:
        combination_generator = combinations(attack_locations, budget_k)
        
        for total_reward, v in tqdm(pool.imap(process_func, combination_generator, chunksize=512)):
            if total_reward > max_reward:
                max_reward = total_reward
                best_strategy = v
    
    return best_strategy

def GR_algorithm(attack_locations,
                 eta: float, 
                 gamma,
                 M: int, 
                 estimated_reward: np.ndarray, 
                 NUM_LANDSCAPE_CELLS, 
                 budget_k) -> np.ndarray:
    """
    Implements the GR (Geometric Resampling) Algorithm.
    """
    n = len(estimated_reward)
    K = np.zeros(n, dtype=int)
    k = 1
    
    while k <= M:

        v_tilde = select_defender_strategy(attack_locations, estimated_reward, eta, gamma, NUM_LANDSCAPE_CELLS, budget_k)
        
        for i in range(n):
            if k < M and v_tilde[i] == 1 and K[i] == 0:
                K[i] = k
            elif k == M and K[i] == 0:
                K[i] = M
        
        if np.all(K > 0):
            break
            
        k += 1
    
    return K

def update_estimated_reward(
    estimated_reward: np.ndarray,
    K: np.ndarray,
    attacker_strategy: np.ndarray,
    defender_strategy: np.ndarray,
    targets_df: pd.DataFrame) -> np.ndarray:

    attacker_strategy = np.array(attacker_strategy)
    defender_strategy = np.array(defender_strategy)

    r = (targets_df['reward'] - targets_df['penalty']).values
    r_t = [a * b for a, b in zip(attacker_strategy, r)]

    updated_reward = estimated_reward.copy()

    protected_cells = np.where((defender_strategy == 1))[0]
    
    for idx in protected_cells:
        updated_reward[idx] += K[idx] * r_t[idx]
    
    return updated_reward

def plot_defender_regret(defender_regret_values):

    regret_values = np.array(defender_regret_values)
    steps = np.arange(1, len(regret_values) + 1)
    
    plt.figure(figsize=(6, 6))
    plt.plot(steps, regret_values, 'b-', label='FPL-UE')
    
    plt.xlabel('Step')
    plt.ylabel('Regret Value')
    plt.title('Defender Regret Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()

    plt.savefig('game_theory_codes/FPL-UE/outputs/defender_regret_plot.png', dpi=300, bbox_inches='tight')
    
    plt.close()

    return

def calculate_defender_regret(defender_strategy_history, attacker_strategy_history, best_hindsight_strategy, targets_df):
    
    assert len(defender_strategy_history) == len(attacker_strategy_history)
    max_steps = len(defender_strategy_history)

    regret_i_hindsight = 0
    regret_i = 0

    for step in range(max_steps):
        attacker_strategy_i = attacker_strategy_history[step]
        defender_strategy_i = defender_strategy_history[step]

        r = (targets_df['reward'] - targets_df['penalty']).values
        r_t = [a * b for a, b in zip(attacker_strategy_i, r)]

        regret_i_hindsight += np.dot(best_hindsight_strategy, r_t)

    for step in range(max_steps):
        attacker_strategy_i = attacker_strategy_history[step]
        defender_strategy_i = defender_strategy_history[step]

        r = (targets_df['reward'] - targets_df['penalty']).values
        r_t = [a * b for a, b in zip(attacker_strategy_i, r)]

        regret_i += np.dot(defender_strategy_i, r_t)

    REGRET = (regret_i_hindsight - regret_i)/max_steps

    return REGRET
     
def run_single_play(model_params, experiment_name, output_folder, MAX_GAME_STEPS, NUM_LANDSCAPE_CELLS, BUDGET_K, M, gamma, eta, targets_df, shape_of_coverage_matrix):

    estimated_reward = np.zeros(NUM_LANDSCAPE_CELLS)

    defender_strategy_history = []
    attacker_strategy_history = []

    defender_regret_values = []
    
    for i in range(MAX_GAME_STEPS):

        print("\n----- GameStep", i + 1,"-----")

        attack_locations = list(generate_defender_strategies_using_attack_probabilities(output_folder, BUDGET_K))

        print("Attack locations:", attack_locations)
        
        defender_strategy_i = select_defender_strategy(attack_locations, estimated_reward, eta, gamma, NUM_LANDSCAPE_CELLS, BUDGET_K)

        coverage_matrix = create_defender_coverage_matrix(defender_strategy_i, shape_of_coverage_matrix)

        path = pathlib.Path(os.path.join(output_folder, "game_step_" + str(i + 1)))
        path.mkdir(parents=True, exist_ok=True)

        plot_and_save_defender_coverage(coverage_matrix, os.path.join(output_folder, "game_step_" + str(i + 1)))

        attacker_strategy_i = run_abm(model_params, experiment_name, os.path.join(output_folder, "game_step_" + str(i + 1)), shape_of_coverage_matrix)

        print(f"Protected target IDs: {targets_df['targetID'].loc[np.where(np.array(defender_strategy_i) == 1)[0]].tolist()}")
        print(f"Attacked target IDs: {targets_df['targetID'].loc[np.where(np.array(attacker_strategy_i) == 1)[0]].tolist()}")

        defender_strategy_history.append(defender_strategy_i)
        attacker_strategy_history.append(attacker_strategy_i)

        best_defender_strategy_t = calculate_best_strategy_v2(NUM_LANDSCAPE_CELLS, attack_locations, attacker_strategy_history, targets_df, BUDGET_K)

        K = GR_algorithm(attack_locations, eta, gamma, M, estimated_reward, NUM_LANDSCAPE_CELLS, BUDGET_K)

        estimated_reward = update_estimated_reward(estimated_reward, K, attacker_strategy_i, defender_strategy_i, targets_df)

        print("step utility for defender:", step_utility_defender(attacker_strategy_i, defender_strategy_i, targets_df))

        regret_i = calculate_defender_regret(defender_strategy_history, attacker_strategy_history, best_defender_strategy_t, targets_df)

        print("Defender regret:", regret_i)

        defender_regret_values.append(regret_i)

    plot_defender_regret(defender_regret_values)

    return  







def optimise_strategy(model_params, experiment_name, output_folder):

    raster_path = os.path.join(
        "mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"
    )

    assign_rewards_and_penalties = LandUseRewards(raster_path)

    shape_of_coverage_matrix = (125, 125)

    interpolated = assign_rewards_and_penalties.interpolate_matrix(shape_of_coverage_matrix)

    assign_rewards_and_penalties.save_interpolated_matrix(
        interpolated, "game_theory_codes/FPL-UE/outputs/interpolated_LULC_matrix.tif",
        shape_of_coverage_matrix
    )

    assign_rewards_and_penalties.plot_matrices(interpolated)

    assign_rewards_and_penalties._detect_forest_plantation_border(interpolated, buffer=1)

    assign_rewards_and_penalties.save_interpolated_matrix(
        assign_rewards_and_penalties.forest_agriculture_fringe, "game_theory_codes/FPL-UE/outputs/potential_targets_matrix.tif",
        shape_of_coverage_matrix
    )

    targets_df = assign_rewards_and_penalties.assign_target_rewards_and_penalties_random(
        target_value=10, interpolated=interpolated
    )

    assign_rewards_and_penalties.plot_with_rewards(
        targets_df, interpolated, name="defender"
    )




    #--------------------GAME STEP AT T=0--------------------------#
    output_folder_step_0 = os.path.join(output_folder, "game_step_0")


    #------------------create the coverage matrix------------------#
    path = pathlib.Path(output_folder_step_0)
    path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_folder_step_0, "model_parameters.yaml"), "w") as configfile:
        yaml.dump(model_params, configfile, default_flow_style=False)

    coverage_matrix = np.zeros(shape_of_coverage_matrix, dtype=np.int8)
    #------------------create the coverage matrix------------------#



    #------------------plot the coverage matrix------------------#
    fig, ax = plt.subplots(figsize=(8, 8))
    
    cmap = mcolors.ListedColormap(["white", "red"])
    
    im = ax.imshow(coverage_matrix, cmap=cmap, vmin=0, vmax=1)
    
    ax.set_xticks([])
    ax.set_yticks([])

    legend_elements = [
        Patch(facecolor="red", edgecolor='black', label='Protected'),
        Patch(facecolor="white", edgecolor='black', label='Unprotected')
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.savefig(
        os.path.join(output_folder_step_0, "defender_coverage_matrix.png"),
        bbox_inches="tight",
        dpi=300,
        
    )

    plt.close()
    #------------------plot the coverage matrix------------------#


    #------------------save the coverage matrix------------------#  
    source_file = gdal.Open("game_theory_codes/FPL-UE/outputs/interpolated_LULC_matrix.tif")

    cols = source_file.RasterXSize
    rows = source_file.RasterYSize
    projection = source_file.GetProjection()
    geotransform = source_file.GetGeoTransform()

    output_file = "game_theory_codes/FPL-UE/outputs/coverage_matrix.tif"

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(coverage_matrix.astype(np.uint8))

    source_file = None
    output_dataset = None


    output_file = os.path.join(output_folder_step_0, "defender_coverage_matrix.tif")

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(coverage_matrix.astype(np.uint8))

    source_file = None
    output_dataset = None
    #------------------save the coverage matrix------------------#  





    batch_run_model(model_params, experiment_name, output_folder_step_0)





    #------------------find the attacker strategy------------------#
    matrix = np.zeros(shape_of_coverage_matrix, dtype=np.uint8)

    for simulation_folder in os.listdir(output_folder_step_0):

        try:

            df = pd.read_csv(os.path.join(output_folder_step_0, simulation_folder, "output_files/agent_data.csv"))
            df.dropna(subset=['ROW', 'COL'], inplace=True)
        
            rows = df['ROW'].astype(int).values
            cols = df['COL'].astype(int).values
        
            mask = (0 <= rows) & (0 <= cols)
            valid_rows = rows[mask]
            valid_cols = cols[mask]
            
            matrix[valid_rows, valid_cols] += 1
        
        except Exception as e:
            pass


    potential_targets = gdal.Open("game_theory_codes/FPL-UE/outputs/potential_targets_matrix.tif").ReadAsArray()

    mask = potential_targets != 1
    matrix[mask] = 0


    total_attacks = np.sum(matrix)
    matrix = matrix/total_attacks

    fig, ax = plt.subplots(figsize=(8, 8))
    
    cmap = 'coolwarm'
    
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=np.max(matrix))
    
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = plt.colorbar(im, shrink=0.5)

    ticks = np.linspace(0, np.max(matrix), num=5) 
    cbar.set_label("Attack Probability", rotation=90)
    cbar.set_ticks(ticks)

    plt.savefig(
        os.path.join(output_folder_step_0, "attacker_strategy_matrix.png"),
        bbox_inches="tight",
        dpi=300,
    )

    source_file = gdal.Open("game_theory_codes/FPL-UE/outputs/interpolated_LULC_matrix.tif")

    cols = source_file.RasterXSize
    rows = source_file.RasterYSize
    projection = source_file.GetProjection()
    geotransform = source_file.GetGeoTransform()

    output_file = os.path.join(output_folder_step_0, "attacker_strategy_matrix.tif")

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Float32)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(matrix.astype(np.float32))

    source_file = None
    output_dataset = None
    #------------------find the attacker strategy------------------#






    NUM_LANDSCAPE_CELLS = len(targets_df)  # Total number of landscape cells within the simulation extent
    BUDGET_K = 5  # Maximum number of cells that can be protected by the defenders at every time-step
    MAX_GAME_STEPS = 100  # Maximum number of time-steps in the game
    gamma = 0.25  # Exploration/Exploitation Trade-off parameter
    eta = 10  #reward perturbation parameter
    M = 25      #PARAMETER IN THE ALGORITHM

    run_single_play(model_params, experiment_name, output_folder, MAX_GAME_STEPS, NUM_LANDSCAPE_CELLS, BUDGET_K, M, gamma, eta, targets_df, shape_of_coverage_matrix)












if __name__ == "__main__":

    model_params = {
            "year": 2010,
            "month": "Mar",
            "num_bull_elephants": 1,
            "area_size": 1100,
            "spatial_resolution": 30,
            "max_food_val_cropland": 100,
            "max_food_val_forest": 10,
            "prob_food_forest": 0.10,
            "prob_food_cropland": 0.10,
            "prob_water_sources": 0.00,
            "thermoregulation_threshold": 28,
            "num_days_agent_survives_in_deprivation": 10,
            "knowledge_from_fringe": 1500,
            "prob_crop_damage": 0.05,
            "prob_infrastructure_damage": 0.01,
            "percent_memory_elephant": 0.375,
            "radius_food_search": 750,
            "radius_water_search": 750,
            "radius_forest_search": 1500,
            "fitness_threshold": 0.4,
            "terrain_radius": 750,
            "slope_tolerance": 30,
            "num_processes": 4,
            "iterations": 4,
            "max_time_steps": 288 * 30,
            "aggression_threshold_enter_cropland": 1.0,
            "human_habituation_tolerance": 1.0,
            "elephant_agent_visibility_radius": 1000,
            "plot_stepwise_target_selection": False,
            "threshold_days_of_food_deprivation": 0,
            "threshold_days_of_water_deprivation": 3,
            "number_of_feasible_movement_directions": 3,
            "track_in_mlflow": False,
            "elephant_starting_location": "user_input",
            "elephant_starting_latitude": 1049237,
            "elephant_starting_longitude": 8570917,
            "elephant_aggression_value": 0.8,
            "elephant_crop_habituation": False
        }
    

    experiment_name = "mitigation-measures-within-plantations-FPL-UE/" 

    elephant_category = "solitary_bulls"

    starting_location = (
        "latitude-"
        + str(model_params["elephant_starting_latitude"])
        + "-longitude-"
        + str(model_params["elephant_starting_longitude"])
    )

    landscape_food_probability = (
        "landscape-food-probability-forest-"
        + str(model_params["prob_food_forest"])
        + "-cropland-"
        + str(model_params["prob_food_cropland"])
    )

    water_holes_probability = "water-holes-within-landscape-" + str(
        model_params["prob_water_sources"]
    )

    memory_matrix_type = "random-memory-matrix-model"

    num_days_agent_survives_in_deprivation = (
        "num_days_agent_survives_in_deprivation-"
        + str(model_params["num_days_agent_survives_in_deprivation"])
    )

    maximum_food_in_a_forest_cell = "maximum-food-in-a-forest-cell-" + str(
        model_params["max_food_val_forest"]
    )

    elephant_thermoregulation_threshold = (
        "thermoregulation-threshold-temperature-"
        + str(model_params["thermoregulation_threshold"])
    )

    threshold_food_derivation_days = "threshold_days_of_food_deprivation-" + str(
        model_params["threshold_days_of_food_deprivation"]
    )

    threshold_water_derivation_days = "threshold_days_of_water_deprivation-" + str(
        model_params["threshold_days_of_water_deprivation"]
    )

    slope_tolerance = "slope_tolerance-" + str(model_params["slope_tolerance"])
    
    num_days_agent_survives_in_deprivation = (
        "num_days_agent_survives_in_deprivation-"
        + str(model_params["num_days_agent_survives_in_deprivation"])
    )

    elephant_aggression_value = "elephant_aggression_value_" + str(
        model_params["elephant_aggression_value"]
    )

    output_folder = os.path.join(
        os.getcwd(),
        "model_runs",
        experiment_name,
        starting_location,
        elephant_category,
        landscape_food_probability,
        water_holes_probability,
        memory_matrix_type,
        num_days_agent_survives_in_deprivation,
        maximum_food_in_a_forest_cell,
        elephant_thermoregulation_threshold,
        threshold_food_derivation_days,
        threshold_water_derivation_days,
        slope_tolerance,
        num_days_agent_survives_in_deprivation,
        elephant_aggression_value,
        str(model_params["year"]),
        str(model_params["month"])
    )

    optimise_strategy(
        model_params=model_params,
        experiment_name=experiment_name,
        output_folder=output_folder,
    )


