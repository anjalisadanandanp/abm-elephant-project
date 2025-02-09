import numpy as np
from typing import List, Set
from itertools import combinations
import os
from osgeo import gdal
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import pathlib
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib import colors
from pyproj import Proj, transform
import yaml





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

    def assign_defender_target_rewards_and_penalties_random(
        self, target_value=10, interpolated=None):

        """Assign targetID, rewards and penalties for landuse cells"""
        if interpolated is None:
            raise ValueError("Interpolated matrix required")

        target_cells = np.where(interpolated == target_value)
        n_targets = len(target_cells[0])

        df = pd.DataFrame(
            {
                "targetID": range(1, n_targets + 1),
                "row": target_cells[0],
                "col": target_cells[1],
                "reward": np.random.uniform(1, 10, n_targets),
                "penalty": np.random.uniform(-10, 0, n_targets),
            }
        )

        df.to_csv(
            "game_theory_codes/FPL-UE/outputs/defender_rewards_penalties.csv",
            index=False,
        )
        return df

    def assign_attacker_target_rewards_and_penalties_random(
        self, target_value=10, interpolated=None
    ):
        """Assign targetID, rewards and penalties for landuse cells"""
        if interpolated is None:
            raise ValueError("Interpolated matrix required")

        target_cells = np.where(interpolated == target_value)
        n_targets = len(target_cells[0])

        df = pd.DataFrame(
            {
                "targetID": range(1, n_targets + 1),
                "row": target_cells[0],
                "col": target_cells[1],
                "reward": np.random.uniform(1, 10, n_targets),
                "penalty": np.random.uniform(-10, 0, n_targets),
            }
        )

        df.to_csv(
            "game_theory_codes/FPL-UE/outputs/attacker_rewards_penalties.csv",
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
                    f'{row["reward"]:.0f}',
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=4,
                )

                ax2.text(
                    row["col"],
                    row["row"],
                    f'{row["penalty"]:.0f}',
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





def generate_defender_strategies(num_landscape_cells: int, budget_k: int) -> Set[np.ndarray]:
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
    
    for num_protected in range(1, budget_k + 1):

        for protected_cells in combinations(range(num_landscape_cells), num_protected):
            strategy = np.zeros(num_landscape_cells)
            strategy[list(protected_cells)] = 1
            strategies.add(tuple(strategy)) 

    print(f"Number of possible defender strategies: {len(strategies)}")
            
    return strategies

def plot_rewards_comparison(estimated_reward, z, perturbed_reward, eta):
    """
    Creates a visualization comparing estimated rewards, perturbation, and perturbed rewards.
    
    Args:
        estimated_reward: Original estimated reward vector
        z: Perturbation vector (exponential noise)
        perturbed_reward: Final perturbed reward (estimated_reward + z)
        eta: Learning rate parameter used for perturbation
    """

    indices = np.arange(len(estimated_reward)) 
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    fig.suptitle(f'Reward Perturbation Analysis (η={eta})', fontsize=14)
    
    width = 0.25
    ax1.bar(indices - width, estimated_reward, width, label='Estimated Reward', color='blue', alpha=0.6)
    ax1.bar(indices, perturbed_reward, width, label='Perturbed Reward', color='red', alpha=0.6)
    ax1.bar(indices + width, z, width, label='Perturbation (z)', color='green', alpha=0.6)
    
    ax1.set_xlabel('Target Index')
    ax1.set_ylabel('Value')
    ax1.set_title('Comparison of Rewards and Perturbation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for i in indices:
        ax1.text(i - width, estimated_reward[i], f'{estimated_reward[i]:.2f}', 
                ha='center', va='bottom', rotation=0)
        ax1.text(i, perturbed_reward[i], f'{perturbed_reward[i]:.2f}', 
                ha='center', va='bottom', rotation=0)
        ax1.text(i + width, z[i], f'{z[i]:.2f}', 
                ha='center', va='bottom', rotation=0)
    
    ax2.plot(indices, estimated_reward, 'b-o', label='Estimated Reward', alpha=0.6)
    ax2.plot(indices, perturbed_reward, 'r-o', label='After Perturbation', alpha=0.6)
    ax2.fill_between(indices, estimated_reward, perturbed_reward, 
                     color='gray', alpha=0.2, label='Perturbation Effect')
    
    ax2.set_xlabel('Target Index')
    ax2.set_ylabel('Value')
    ax2.set_title('Effect of Perturbation on Estimated Rewards')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    plt.savefig('game_theory_codes/FPL-UE/outputs/reward_perturbation.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 

def select_defender_strategy(
    E: Set[np.ndarray],  # Set of exploration strategies
    estimated_reward: np.ndarray,  # Current estimated reward vector
    gamma: float,  # Exploration probability
    eta: float,  
    targets_df_defender: pd.DataFrame  # DataFrame containing target information
    ) -> np.ndarray:
    """
    Selects a strategy based on the exploration-exploitation trade-off.
    """

    flag = np.random.random() < gamma 

    # print("EXPLORATION PHASE" if flag else "EXPLOITATION PHASE")
    
    if flag:  # Exploration of strategies
        strategies = list(E)
        v_t = strategies[np.random.randint(len(strategies))]

    else:  # Exploitation of learned strategies

        n = len(estimated_reward)
        z = np.random.exponential(scale=1/eta, size=n)
        
        perturbed_reward = estimated_reward + z

        fig = plot_rewards_comparison(estimated_reward, z, perturbed_reward, eta)

        max_reward = float('-inf')
        best_strategy = None
        
        for v in E:

            v = np.array(v)
            protected_indices = np.where(v == 1)[0]
            unprotected_indices = np.where(v == 0)[0]

            protected_reward = sum(targets_df_defender.iloc[protected_indices]['reward'])
            unprotected_penalty = sum(targets_df_defender.iloc[unprotected_indices]['penalty'])
            
            total_reward = np.dot(v, perturbed_reward) 

            if total_reward > max_reward:
                max_reward = total_reward
                best_strategy = v
        
        v_t = best_strategy
    
    return v_t

def update_estimated_reward(
    estimated_reward: np.ndarray,
    K: np.ndarray,
    attacker_strategy: np.ndarray,
    defender_strategy: np.ndarray,
    targets_df_defender: pd.DataFrame
) -> np.ndarray:
    """
    Updates estimated rewards based on chosen strategy and actual rewards/penalties.
    """

    updated_reward = estimated_reward.copy()

    attacker_strategy = np.array(attacker_strategy)
    defender_strategy = np.array(defender_strategy)

    protected_cells = np.where((defender_strategy == 1))[0]
    
    for idx in protected_cells:
        reward = targets_df_defender.iloc[idx]['reward'] + targets_df_defender.iloc[idx]['penalty']
        updated_reward[idx] += K[idx] * reward
    
    return updated_reward

def select_random_strategy(E: Set[np.ndarray]) -> np.ndarray:
    """
    Randomly selects a strategy from the set of strategies E.
    """
    strategies = list(E)
    return np.array(strategies[np.random.randint(len(strategies))])

def GR_algorithm(eta: float, 
                 M: int, 
                 estimated_reward: np.ndarray, 
                 E: Set[np.ndarray], 
                 gamma: float, 
                 targets_df_defender: pd.DataFrame) -> np.ndarray:
    """
    Implements the GR (Geometric Resampling) Algorithm.
    """
    n = len(estimated_reward)
    K = np.zeros(n, dtype=int)
    k = 1
    
    while k <= M:

        v_tilde = select_defender_strategy(E, estimated_reward, gamma, eta, targets_df_defender)
        
        for i in range(n):
            if k < M and v_tilde[i] == 1 and K[i] == 0:
                K[i] = k
            elif k == M and K[i] == 0:
                K[i] = M
        
        if np.all(K > 0):
            break
            
        k += 1
    
    return K











# Test the implementation
if __name__ == "__main__":

    raster_path = os.path.join(
        "mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"
    )

    assign_rewards_and_penalties = LandUseRewards(raster_path)

    interpolated = assign_rewards_and_penalties.interpolate_matrix((10, 10))
    assign_rewards_and_penalties.plot_matrices(interpolated)

    targets_df_defender = assign_rewards_and_penalties.assign_defender_target_rewards_and_penalties_random(
        target_value=10, interpolated=interpolated
    )
    assign_rewards_and_penalties.plot_with_rewards(
        targets_df_defender, interpolated, name="defender"
    )

    targets_df_attacker = assign_rewards_and_penalties.assign_attacker_target_rewards_and_penalties_random(
        target_value=10, interpolated=interpolated
    )
    assign_rewards_and_penalties.plot_with_rewards(
        targets_df_attacker, interpolated, name="attacker"
    )

    assert len(targets_df_defender) == len(targets_df_attacker)

    NUM_LANDSCAPE_CELLS = len(targets_df_attacker)  # Total number of landscape cells within the simulation extent
    BUDGET_K = 2  # Maximum number of cells that can be protected by the defenders at every time-step
    MAX_GAME_STEPS = 10  # Maximum number of time-steps in the game
    gamma = 0.1  # Exploration/Exploitation Trade-off parameter
    eta = 0.80  #reward perturbation parameter
    M = 100

    # Generate all valid defender strategies
    E = generate_defender_strategies(NUM_LANDSCAPE_CELLS, BUDGET_K)

    print("Example strategies:")
    for i, strategy in enumerate(list(E)[:5]):  
        print(f"Strategy {i + 1}: {strategy}")

    estimated_reward = np.zeros(NUM_LANDSCAPE_CELLS)
    
    for i in range(MAX_GAME_STEPS):

        print("\n----- GameStep", i + 1,"-----")

        attacker_strategy_i = select_random_strategy(E)
        print("Attacker strategy:", attacker_strategy_i)
        
        defender_strategy_i = select_defender_strategy(E, estimated_reward, gamma, eta, targets_df_defender)

        print(f"Selected strategy: {defender_strategy_i}")
        print(f"Number of cells protected: {int(sum(defender_strategy_i))}")
        print(f"Protected target IDs: {targets_df_defender['targetID'].loc[np.where(np.array(defender_strategy_i) == 1)[0]].tolist()}")

        K = GR_algorithm(eta, M, estimated_reward, E=E, gamma=gamma, targets_df_defender=targets_df_defender)
        print("K values:", K)

        estimated_reward = update_estimated_reward(estimated_reward, K, attacker_strategy_i, defender_strategy_i, targets_df_defender)
        