import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import pathlib
import yaml
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import random
import itertools
from pyproj import Proj, transform  
from mpl_toolkits.basemap import Basemap    
import rasterio
from rasterio.features import shapes
import fiona
import geojson
import matplotlib.cm as cm
import shutil


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

module = importlib.import_module('game_theory_codes.OUR-MODEL.abm_model_HEC_with_landscape_deterrent_policies')
batch_run_model = module.batch_run_model



def combination_to_binary_vector(combination, NUM_LANDSCAPE_CELLS):

    binary_vector = np.zeros(NUM_LANDSCAPE_CELLS, dtype=int)
    indices = [int(index - 1) for index in combination]
    binary_vector[indices] = 1
    return binary_vector

def generate_defender_strategies(BUDGET_K, NUM_LANDSCAPE_CELLS, targets_df):

    def select_cells_at_distance(target_lat, target_lon, distance_cells, distance_type='euclidean'):

        dataset = gdal.Open("create-strategy-matrix/boundary_raster_discretised.tif")
        geotransform = dataset.GetGeoTransform()
        original_array = dataset.ReadAsArray()

        original_array[275:350, 550:650] = 0
        
        x_origin = geotransform[0]  
        y_origin = geotransform[3] 
        pixel_width = geotransform[1]
        pixel_height = geotransform[5] 
        
        target_col = int((target_lon - x_origin) / pixel_width)
        target_row = int((target_lat - y_origin) / pixel_height)
    
        
        rows, cols = original_array.shape
        
        if not (0 <= target_row < rows and 0 <= target_col < cols):
            raise ValueError(f"Target coordinates are outside raster bounds. "
                            f"Pixel coordinates: ({target_row}, {target_col}), "
                            f"Raster shape: ({rows}, {cols})")
        
        row_indices, col_indices = np.ogrid[:rows, :cols]
        
        if distance_type == 'euclidean':
            distances = np.sqrt((row_indices - target_row)**2 + (col_indices - target_col)**2)
        elif distance_type == 'manhattan':
            distances = np.abs(row_indices - target_row) + np.abs(col_indices - target_col)
        elif distance_type == 'chebyshev':
            distances = np.maximum(np.abs(row_indices - target_row), np.abs(col_indices - target_col))
        else:
            raise ValueError("distance_type must be 'euclidean', 'manhattan', or 'chebyshev'")

        if isinstance(distance_cells, (int, float)):
            mask = (distances <= distance_cells)
        else:
            raise ValueError("distance_cells must be a number or a tuple of (min_distance, max_distance)")
        
        result_array = np.where(mask, original_array, 0)
        
        return result_array
    
    potential_coverage_matrix = gdal.Open(os.path.join("create-strategy-matrix/boundary_raster_discretised.tif")).ReadAsArray()

    potential_targets = targets_df["boundary_patch_id"].tolist()

    potential_coverage_matrix = select_cells_at_distance(
        target_lat=1049237,
        target_lon=8570917,
        distance_cells=175,
        distance_type='euclidean'
    )

    potential_coverage_matrix = potential_coverage_matrix.astype(int)

    mask = np.isin(potential_coverage_matrix, potential_targets)
    potential_coverage_matrix[~mask] = 0
    potential_coverage_matrix = potential_coverage_matrix.astype(int)

    unique_values = np.unique(potential_coverage_matrix)
    non_zero_unique_values = unique_values[unique_values != 0]

    print("boundary patches:", non_zero_unique_values, "total numbers:", len(non_zero_unique_values))

    combinations_of_size_k = itertools.combinations(non_zero_unique_values, BUDGET_K)

    defender_strategies = []

    for combination in tqdm(combinations_of_size_k):
        strategy_vector = combination_to_binary_vector(combination, NUM_LANDSCAPE_CELLS)
        defender_strategies.append(strategy_vector)

    return defender_strategies

def calculate_reward_for_strategy(defender_strategy, perturbed_reward):
    v = np.array(defender_strategy)
    total_reward = np.dot(v, perturbed_reward)
    return total_reward, v

def find_defender_strategy(
    abm_run_folder,
    NUM_LANDSCAPE_CELLS
    ):

    coverage_matrix = gdal.Open(os.path.join(abm_run_folder, "defender_coverage_matrix.tif")).ReadAsArray()

    unique_targets = np.unique(coverage_matrix)
    unique_targets = unique_targets[unique_targets != 0]  
    
    # print(f"Found targets in coverage matrix: {unique_targets}")
    
    v_t = np.zeros(NUM_LANDSCAPE_CELLS)
    
    for target_id in unique_targets:
        v_t[target_id - 1] = 1  
    
    return v_t

def calculate_attacker_strategy_from_abm_runs(abm_run_folder, coverage_matrix_folder):

    potential_coverage_matrix = gdal.Open(os.path.join(coverage_matrix_folder, "potential_coverage_matrix.tif")).ReadAsArray()
    
    trajectory_matrix = np.zeros_like(potential_coverage_matrix, dtype=np.uint8)

    num_simulations = 0

    dict_of_attacked_targets = {}

    for simulation_folder in os.listdir(abm_run_folder):

        try:

            df = pd.read_csv(os.path.join(abm_run_folder, simulation_folder, "output_files/agent_data.csv"))
            df.dropna(subset=['ROW', 'COL'], inplace=True)

            unique_targets = df["target_attacked"].dropna().unique()

            for target in unique_targets:
                if target not in dict_of_attacked_targets:
                    dict_of_attacked_targets[target] = 0
                dict_of_attacked_targets[target] += 1
        
            rows = df['ROW'].astype(int).values
            cols = df['COL'].astype(int).values
        
            mask = (0 <= rows) & (0 <= cols)
            valid_rows = rows[mask]
            valid_cols = cols[mask]
            
            trajectory_matrix[valid_rows, valid_cols] = 1

            num_simulations += 1
        
        except Exception as e:
            pass


    mask = potential_coverage_matrix == 0
    trajectory_matrix[mask] = 0



    trajectory_matrix = trajectory_matrix/num_simulations



    for target, count in dict_of_attacked_targets.items():
        print(f"Covered target {target} was attacked {count} times.")   \



    targets_matrix = gdal.Open("create-strategy-matrix/boundary_raster_discretised.tif").ReadAsArray()
    unique_values = np.unique(targets_matrix)
    non_zero_unique_values = unique_values[unique_values != 0]
    total_targets = len(non_zero_unique_values)
    attacker_strategy_covered_targets = [0 for i in range(total_targets - 1)]

    for target in dict_of_attacked_targets:
        attacker_strategy_covered_targets[int(target - 1)] = 1


    targets_matrix = gdal.Open("create-strategy-matrix/boundary_raster_discretised.tif").ReadAsArray()
    unique_values = np.unique(targets_matrix)
    non_zero_unique_values = unique_values[unique_values != 0]
    total_targets = len(non_zero_unique_values)
    attacker_strategy_uncovered_targets = [0 for i in range(total_targets - 1)]

    coverage_matrix = gdal.Open("create-strategy-matrix/boundary_raster_discretised.tif").ReadAsArray()
    plantation_rows, plantation_cols = np.where(coverage_matrix != 0)
    unique_values, counts = np.unique(coverage_matrix, return_counts=True)
    value_counts = dict(zip(unique_values, counts))

    for row, col in zip(plantation_rows, plantation_cols):
        if trajectory_matrix[row, col] > 0:
            attacker_strategy_uncovered_targets[int(coverage_matrix[row,col]-1)] += trajectory_matrix[row, col]

    def probabilistic_binary_conversion(original_list):

        filtered_list = [value for value in original_list if value != 0]
        percentile_value = np.percentile(filtered_list, 50)

        binary_list = []
        
        for value in original_list:

            probability = min(1, value / percentile_value)
            binary_value = np.random.binomial(1, probability)
            binary_list.append(binary_value)
        
        return binary_list
    
    attacker_strategy_uncovered_targets = probabilistic_binary_conversion(attacker_strategy_uncovered_targets)



    attacker_strategy = [min(1, v1+v2) for v1, v2 in zip(attacker_strategy_covered_targets, attacker_strategy_uncovered_targets)]
 


    return attacker_strategy

def step_utility_defender(attacker_strategy_i, defender_strategy_i, targets_df):

    attacker_strategy = np.array(attacker_strategy_i)
    defender_strategy = np.array(defender_strategy_i)

    r = (targets_df['reward'] - targets_df['penalty']).values
    r_t = [a * b for a, b in zip(attacker_strategy, r)]

    reward_01 = np.dot(defender_strategy, r_t)
    reward_02 = np.dot(attacker_strategy, targets_df['penalty'].values)

    return reward_01 + reward_02

def calculate_reward_for_strategy_best(defender_strategy, attacker_strategy_history, targets_df):

    v = np.array(defender_strategy)
    
    total_strategy_utility = 0
    for attacker_strategy in attacker_strategy_history:
        step_utility = step_utility_defender(attacker_strategy, v, targets_df)
        total_strategy_utility += step_utility
    
    return total_strategy_utility, v
    
def calculate_best_strategy(defender_strategies, attacker_strategy_history, targets_df, n_processes=4):

    process_func = partial(
        calculate_reward_for_strategy_best,
        attacker_strategy_history=attacker_strategy_history,
        targets_df=targets_df
    )
    
    max_reward = float('-inf')
    best_strategy = None
    
    with mp.Pool(processes=n_processes) as pool:
        
        for total_reward, v in tqdm(pool.imap(process_func, defender_strategies, chunksize=512)):
            if total_reward > max_reward:
                max_reward = total_reward
                best_strategy = v

    return best_strategy

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

    plt.savefig('game_theory_codes/OUR-MODEL/defender_regret_plot.png', dpi=300, bbox_inches='tight')
    
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
     
def run_single_play(abm_run_folder, coverage_matrix_folder, MAX_GAME_STEPS, BUDGET_K, NUM_LANDSCAPE_CELLS, targets_df):

    defender_strategy_history = []
    attacker_strategy_history = []

    defender_regret_values = []

    defender_strategies = generate_defender_strategies(BUDGET_K, NUM_LANDSCAPE_CELLS, targets_df)
    
    for i in range(1, MAX_GAME_STEPS+1):

        print("\n----- GameStep", i ,"-----")

        try:

            defender_strategy_i = find_defender_strategy(abm_run_folder=os.path.join(abm_run_folder, "game_step_" + str(i + 1)), NUM_LANDSCAPE_CELLS=NUM_LANDSCAPE_CELLS)

            print("Defender strategy:", defender_strategy_i)

            attacker_strategy_i = calculate_attacker_strategy_from_abm_runs(abm_run_folder = os.path.join(abm_run_folder, "game_step_" + str(i)),
                                                                            coverage_matrix_folder = os.path.join(coverage_matrix_folder))

            print("Attacker strategy:", attacker_strategy_i)

            print(f"Protected target IDs: {targets_df['boundary_patch_id'].loc[np.where(np.array(defender_strategy_i) == 1)[0]].tolist()}")
            print(f"Attacked target IDs: {targets_df['boundary_patch_id'].loc[np.where(np.array(attacker_strategy_i) == 1)[0]].tolist()}")

            defender_strategy_history.append(defender_strategy_i)
            attacker_strategy_history.append(attacker_strategy_i)

            best_defender_strategy_t = calculate_best_strategy(defender_strategies, attacker_strategy_history, targets_df)

            print("Best defender strategy:", best_defender_strategy_t)

            print("step utility for defender:", step_utility_defender(attacker_strategy_i, defender_strategy_i, targets_df))

            regret_i = calculate_defender_regret(defender_strategy_history, attacker_strategy_history, best_defender_strategy_t, targets_df)

            print("Defender regret:", regret_i)

            defender_regret_values.append(regret_i)

        except:
            pass

    plot_defender_regret(defender_regret_values)

    return  







def optimise_strategy(abm_run_folder, coverage_matrix_folder, MAX_GAME_STEPS, BUDGET_K):

    NUM_LANDSCAPE_CELLS = 238     

    targets_df = pd.read_csv("assign_rewards_and_penalties/boundary_patch_reward_penalty_matrix.csv")

    run_single_play(abm_run_folder, coverage_matrix_folder, MAX_GAME_STEPS, BUDGET_K, NUM_LANDSCAPE_CELLS, targets_df)












if __name__ == "__main__":

    model_params = {
            "year": 2010,
            "month": "Mar",
            "num_bull_elephants": 1,
            "area_size": 1100,
            "spatial_resolution": 30,
            "max_food_val_cropland": 100,
            "max_food_val_forest": 5,
            "prob_food_forest": 0.10,
            "prob_food_cropland": 0.10,
            "prob_water_sources": 1.0,
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
            "slope_tolerance": 35,
            "num_processes": 8,
            "iterations": 24,
            "max_time_steps": 288 * 30,
            "aggression_threshold_enter_cropland": 1.0,
            "human_habituation_tolerance": 1.0,
            "elephant_agent_visibility_radius": 500,
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

    BUDGET_K = 10                               # Maximum number of cells that can be protected by the defenders at every time-step
    MAX_GAME_STEPS = 35                         # Maximum number of time-steps in the game
    max_gamma = 1.0                             # Exploration/Exploitation Trade-off parameter
    min_gamma = 0.20                            # Exploration/Exploitation Trade-off parameter
    num_steps_gamma_decay = 10                  # Exploration/Exploitation Trade-off parameter
    eta = 0.0                                   # reward perturbation parameter
    M = 30                                      # parameter in the GR algorithm

    experiment_name = "mitigation-measures-within-plantations-FPL-UE_v2_1/" 

    FPL_UE_params = (
        "budget_k_"
        + str(BUDGET_K)
        + "-max_game_steps_"
        + str(MAX_GAME_STEPS) 
        + "-max_gamma_"
        + str(max_gamma)
        + "-min_gamma_"
        + str(min_gamma)
        + "-num_steps_gamma_decay_"
        + str(num_steps_gamma_decay)
        + "-eta_"
        + str(eta)
        + "-M_"
        + str(M)
    )

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

    food_availability_sceanario = "random-food-distribition-within-agricultural-plots-and-other-plantation-cells"

    water_availability_sceanario = "water-source-rivers-landscape-" + str(model_params["prob_water_sources"])

    food_memory_matrix_type = "random-memory-forest-and_plantation-fringe-model"
    
    water_memory_matrix_type = "full-memory-forest-and_plantation-model"

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

    elephant_aggression_value = "elephant_aggression_value_" + str(
        model_params["elephant_aggression_value"]
    )

    game_params = "budget_k_" + str(BUDGET_K) + "_MAX_GAME_STEPS_" + str(MAX_GAME_STEPS)

    abm_run_folder = os.path.join(
        os.getcwd(),
        "game_theory_codes/OUR-MODEL",
        experiment_name,
        FPL_UE_params,
        starting_location,
        elephant_category,
        food_availability_sceanario,
        landscape_food_probability,
        water_availability_sceanario,
        food_memory_matrix_type,
        water_memory_matrix_type,
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

    coverage_matrix_folder = os.path.join(
        os.getcwd(),
        "game_theory_codes/OUR-MODEL",
        experiment_name,
        FPL_UE_params,
        "coverage_matrix_init")

    optimise_strategy(
        abm_run_folder=abm_run_folder,
        coverage_matrix_folder = coverage_matrix_folder,
        MAX_GAME_STEPS = MAX_GAME_STEPS,
        BUDGET_K = BUDGET_K
    )


