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
import random
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from tqdm import tqdm

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

from experiments.ranger_deployment.experiment_names import FancyNameGenerator

    


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

        target_cells = np.where(interpolated == target_value)
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
    """
    Creates a 2D matrix showing which cells are covered by the defender
    
    Args:
        defender_strategy: 1D array with 0s and 1s, where 1 means cell is protected
        grid_shape: Tuple of (rows, cols) representing the landscape dimensions
        
    Returns:
        2D matrix where 1 indicates a cell protected by defender, 0 indicates unprotected
    """
    coverage_matrix = np.zeros(grid_shape, dtype=np.int8)

    lulc_data = gdal.Open("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif").ReadAsArray()
    plantation_rows, plantation_cols = np.where(lulc_data == 10)

    for i, (row, col) in tqdm(enumerate(zip(plantation_rows, plantation_cols))):
        if defender_strategy[i] == 1:
            coverage_matrix[row, col] = 1
        else:   
            coverage_matrix[row, col] = 0

    return coverage_matrix

def plot_and_save_defender_coverage(coverage_matrix, figsize=(8, 8), 
                          protected_color='red', unprotected_color='white'):
    """
    Plot the defender coverage matrix as a heatmap.
    
    Args:
        coverage_matrix: 2D numpy array where 1 represents protected cells and 0 unprotected cells
        title: Title for the plot
        figsize: Size of the figure as (width, height)
        protected_color: Color for protected cells
        unprotected_color: Color for unprotected cells
        grid_color: Color for the grid lines
    """
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
        "game_theory_codes/FPL-UE/outputs/defender_coverage_matrix.png",
        bbox_inches="tight",
        dpi=300,
    )

    source_file = gdal.Open("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif")

    cols = source_file.RasterXSize
    rows = source_file.RasterYSize
    projection = source_file.GetProjection()
    geotransform = source_file.GetGeoTransform()
    band = source_file.GetRasterBand(1)

    data = band.ReadAsArray()

    output_file = "game_theory_codes/FPL-UE/outputs/coverage_matrix.tif"

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(coverage_matrix.astype(np.uint8))

    source_file = None
    output_dataset = None

    print(f"Coverage matrix saved as {output_file}")

    return 

def generate_defender_strategies_v1(num_landscape_cells: int, budget_k: int) -> Set[np.ndarray]:
    """
    Generates all possible defender pure strategies given the landscape constraints.
    
    Args:
        num_landscape_cells: Total number of cells in the landscape
        budget_k: Maximum number of cells that can be protected
        
    Returns:
        Each strategy is a binary vector of length num_landscape_cells where:
        - 1 indicates a protected cell
        - 0 indicates an unprotected cell
        - Sum of 1s in each strategy is less than or equal to budget_k
    """

    strategies = set()
    
    for num_protected in range(budget_k, budget_k + 1):

        for protected_cells in tqdm(combinations(range(num_landscape_cells), num_protected)):
            strategy = np.zeros(num_landscape_cells, dtype=np.int8)
            strategy[list(protected_cells)] = 1
            strategies.add(tuple(strategy)) 

    print(f"Number of possible defender strategies: {len(strategies)}")
            
    return strategies

def generate_defender_strategies_v2(num_landscape_cells: int, budget_k: int) -> Iterator[np.ndarray]:
    """
    Generates all possible defender pure strategies given the landscape constraints.
    
    Args:
        num_landscape_cells: Total number of cells in the landscape
        budget_k: Maximum number of cells that can be protected
        
    Returns:
        An iterator of strategies, where each strategy is a binary vector of length 
        num_landscape_cells where:
        - 1 indicates a protected cell
        - 0 indicates an unprotected cell
        - Sum of 1s in each strategy is less than or equal to budget_k
    """
    
    for num_protected in tqdm(range(budget_k, budget_k + 1)):
        for protected_cells in combinations(range(num_landscape_cells), num_protected):
            strategy = np.zeros(num_landscape_cells, dtype=np.int8)  # Use int8 instead of float64
            strategy[list(protected_cells)] = 1
            yield strategy   # Yield each strategy instead of storing in a set

def select_defender_strategy_V1(
    E: Set[np.ndarray],  # Set of exploration strategies
    estimated_reward: np.ndarray,  # Current estimated reward vector
    gamma: float,  # Exploration probability
    eta: float
    ) -> np.ndarray:
    """
    Selects a strategy based on the exploration-exploitation trade-off.
    """

    flag = np.random.random() < gamma 

    if flag:  # Exploration of strategies
        strategies = list(E)
        v_t = strategies[np.random.randint(len(strategies))]

    else:  # Exploitation of learned strategies

        n = len(estimated_reward)
        z = np.random.exponential(scale=1/eta, size=n)
        
        perturbed_reward = estimated_reward + z

        max_reward = float('-inf')
        best_strategy = None
        
        for v in E:

            v = np.array(v)
            
            total_reward = np.dot(v, perturbed_reward) 

            if total_reward > max_reward:
                max_reward = total_reward
                best_strategy = v
        
        v_t = best_strategy
    
    return v_t

def optimized_selection(estimated_reward, eta, BUDGET_K):

    n = len(estimated_reward)
    z = np.random.exponential(scale=1/eta, size=n)
    
    perturbed_reward = estimated_reward + z
    top_indices = np.argsort(perturbed_reward)[-BUDGET_K:]
    best_strategy = np.zeros(n)
    best_strategy[top_indices] = 1

    return best_strategy

def select_defender_strategy_V2(
    NUM_LANDSCAPE_CELLS, 
    BUDGET_K,
    estimated_reward: np.ndarray, 
    gamma: float,
    eta: float
    ) -> np.ndarray:
    """
    Selects a strategy based on the exploration-exploitation trade-off.
    Works with a generator of strategies instead of a set.
    """
    flag = np.random.random() < gamma 

    if flag:  

        print("EXPLORATION")

        selected_indices = np.random.choice(NUM_LANDSCAPE_CELLS, BUDGET_K, replace=False)
        selected_strategy = np.zeros(NUM_LANDSCAPE_CELLS)
        selected_strategy[selected_indices] = 1

        v_t = selected_strategy

    else: 

        print("EXPLOITATION")

        best_strategy = optimized_selection(estimated_reward, eta, BUDGET_K)
        v_t = best_strategy
    
    return v_t

def run_abm(model_params, experiment_name, output_folder):

    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_folder, "model_parameters.yaml"), "w") as configfile:
        yaml.dump(model_params, configfile, default_flow_style=False)

    batch_run_model(model_params, experiment_name, output_folder)

    return

def run_single_play(model_params, experiment_name, output_folder, MAX_GAME_STEPS, NUM_LANDSCAPE_CELLS, BUDGET_K, E, M, gamma, eta, targets_df):

    estimated_reward = np.zeros(NUM_LANDSCAPE_CELLS)

    defender_strategy_history = []
    attacker_strategy_history = []

    defender_regret_values = []
    
    for i in range(MAX_GAME_STEPS):

        print("\n----- GameStep", i + 1,"-----")

        defender_strategy_i = select_defender_strategy_V2(NUM_LANDSCAPE_CELLS, BUDGET_K, estimated_reward, gamma, eta)
        print(f"Selected strategy: {defender_strategy_i}")

        coverage_matrix = create_defender_coverage_matrix(defender_strategy_i, (1069, 1070))

        plot_and_save_defender_coverage(coverage_matrix)

        run_abm(model_params, experiment_name, output_folder)

    #     print(f"Number of cells protected: {int(sum(defender_strategy_i))}")
    #     print(f"Protected target IDs: {targets_df['targetID'].loc[np.where(np.array(defender_strategy_i) == 1)[0]].tolist()}")

    #     defender_strategy_history.append(defender_strategy_i)
    #     attacker_strategy_history.append(attacker_strategy_i)

    #     best_defender_strategy_i = calculate_best_strategy_v1(E, attacker_strategy_i)

    #     print("Best strategy for defender for the current step:", best_defender_strategy_i)

    #     best_defender_strategy_t = calculate_best_strategy_v2(E, attacker_strategy_history)

    #     print("Best strategy for defender considering all attacker histories:", best_defender_strategy_t)

    #     K = GR_algorithm(eta, M, estimated_reward, E=E, gamma=gamma)

    #     estimated_reward = update_estimated_reward(estimated_reward, K, attacker_strategy_i, defender_strategy_i, targets_df)

    #     print("step utility for defender:", step_utility_defender(attacker_strategy_i, defender_strategy_i, targets_df))

    #     regret_i = calculate_defender_regret(defender_strategy_history, attacker_strategy_history, best_defender_strategy_t)

    #     print("Defender regret:", regret_i)

    #     defender_regret_values.append(regret_i)

    # plot_defender_regret(defender_regret_values)

    return  




def optimise_strategy(model_params, experiment_name, output_folder):

    raster_path = os.path.join(
        "mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"
    )

    assign_rewards_and_penalties = LandUseRewards(raster_path)

    interpolated = assign_rewards_and_penalties.interpolate_matrix((1069, 1070))

    # assign_rewards_and_penalties.plot_matrices(interpolated)

    targets_df = assign_rewards_and_penalties.assign_target_rewards_and_penalties_random(
        target_value=10, interpolated=interpolated
    )

    # assign_rewards_and_penalties.plot_with_rewards(
    #     targets_df, interpolated, name="defender"
    # # )

    NUM_LANDSCAPE_CELLS = len(targets_df)  # Total number of landscape cells within the simulation extent
    BUDGET_K = 10000  # Maximum number of cells that can be protected by the defenders at every time-step
    MAX_GAME_STEPS = 1000  # Maximum number of time-steps in the game
    gamma = 0.25  # Exploration/Exploitation Trade-off parameter
    eta = 10  #reward perturbation parameter
    M = 10      #PARAMETER IN THE ALGORITHM

    print(f"Generating all strategies for the defender for {NUM_LANDSCAPE_CELLS} landscape cells and {BUDGET_K} budget")

    # Generate all valid defender strategies
    # E = generate_defender_strategies_v2(NUM_LANDSCAPE_CELLS, BUDGET_K)

    # print("Example strategies:")
    # for i, strategy in enumerate(E):  
    #     if i >= 5:
    #         break
    #     print(f"Strategy {i + 1}: {tuple(strategy)}")

    run_single_play(model_params, experiment_name, output_folder, MAX_GAME_STEPS, NUM_LANDSCAPE_CELLS, BUDGET_K, generate_defender_strategies_v2, M, gamma, eta, targets_df)












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
            "num_processes": 8,
            "iterations": 8,
            "max_time_steps": 288 * 5,
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
            "elephant_crop_habituation": False,
            "num_guards": 3,
            "ranger_visibility_radius": 1000,
        }
    
    generator = FancyNameGenerator()
    run_name = generator.generate_name()

    experiment_name = "mitigation-measures-within-plantations-FPL-UE/" + run_name

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
        "without_rangers",
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


