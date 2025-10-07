import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import pathlib
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import random
import itertools

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

module = importlib.import_module('guarding-policies.FPL-UE_no-memory-model-v1_1.abm_model_HEC_with_landscape_deterrent_policies_without_ranger_proximity')
batch_run_model = module.batch_run_model



def combination_to_binary_vector(combination, NUM_LANDSCAPE_CELLS):

    binary_vector = np.zeros(NUM_LANDSCAPE_CELLS, dtype=int)
    indices = [int(index - 1) for index in combination]
    binary_vector[indices] = 1
    return binary_vector

def select_cells_at_distance(target_lat, target_lon, distance_cells, distance_type='euclidean'):

    dataset = gdal.Open(coverage_matrix_path)
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
    
def generate_defender_strategies(coverage_matrix_path, BUDGET_K, NUM_LANDSCAPE_CELLS, targets_df):

    potential_coverage_matrix = gdal.Open(os.path.join(coverage_matrix_path)).ReadAsArray()

    potential_targets = targets_df["boundary_patch_id"].tolist()

    potential_coverage_matrix = select_cells_at_distance(
        target_lat=1049000,
        target_lon=8570800,
        distance_cells=145,
        distance_type='euclidean'
    )

    potential_coverage_matrix = potential_coverage_matrix.astype(int)

    mask = np.isin(potential_coverage_matrix, potential_targets)
    potential_coverage_matrix[~mask] = 0
    potential_coverage_matrix = potential_coverage_matrix.astype(int)

    unique_values = np.unique(potential_coverage_matrix)
    non_zero_unique_values = unique_values[unique_values != 0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = mcolors.ListedColormap(['white', 'black'])
    im = ax.imshow(potential_coverage_matrix, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    legend_elements = [
        Patch(facecolor='black', edgecolor='black', label='Potential Coverage'),
        Patch(facecolor='white', edgecolor='black', label='No Coverage')
    ]

    ax.legend(handles=legend_elements, loc="upper right")
    plt.savefig(
        os.path.join(OUTPUT_FOLDER, "coverage_matrix_init", "potential_coverage_matrix.png"),
        bbox_inches="tight",
        dpi=500,
    )
    plt.close(fig)

    combinations_of_size_k = itertools.combinations(non_zero_unique_values, BUDGET_K)

    defender_strategies = []

    for combination in tqdm(combinations_of_size_k):
        strategy_vector = combination_to_binary_vector(combination, NUM_LANDSCAPE_CELLS)
        defender_strategies.append(strategy_vector)

    source_file = gdal.Open("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif")

    cols = source_file.RasterXSize
    rows = source_file.RasterYSize
    projection = source_file.GetProjection()
    geotransform = source_file.GetGeoTransform()

    output_file = os.path.join(OUTPUT_FOLDER, "coverage_matrix_init/potential_coverage_matrix.tif")

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(potential_coverage_matrix.astype(np.uint8))

    source_file = None
    output_dataset = None

    return defender_strategies

def calculate_reward_for_strategy(defender_strategy, perturbed_reward):
    v = np.array(defender_strategy)
    total_reward = np.dot(v, perturbed_reward)
    return total_reward, v

def find_best_strategy_parallel(defender_strategies, perturbed_reward, n_processes=16):
    
    process_func = partial(
        calculate_reward_for_strategy,
        perturbed_reward=perturbed_reward
    )
    
    max_reward = float('-inf')
    best_strategy = None
    
    with mp.Pool(processes=n_processes) as pool:

        for total_reward, v in tqdm(pool.imap(process_func, defender_strategies, chunksize=4096)):
            if total_reward > max_reward:
                max_reward = total_reward
                best_strategy = v
    
    return best_strategy

def select_defender_strategy(
    defender_strategies,
    estimated_reward: np.ndarray,
    eta: float,
    gamma,
    NUM_LANDSCAPE_CELLS,
    budget_k
    ) -> np.ndarray:


    flag = np.random.random() < gamma 

    if flag: 

        # print("Random Strategy Selected")

        potential_coverage_matrix = gdal.Open(os.path.join(OUTPUT_FOLDER, "coverage_matrix_init/potential_coverage_matrix.tif")).ReadAsArray()
        unique_values = np.unique(potential_coverage_matrix)
        non_zero_unique_values = list(unique_values[unique_values != 0])
        random_sample = random.sample(non_zero_unique_values, budget_k)

        v_t = combination_to_binary_vector(random_sample, NUM_LANDSCAPE_CELLS)

    else:  

        # print("Optimal Strategy Selected")
        
        n = len(estimated_reward)
        z = np.random.exponential(scale=1/eta, size=n)
        perturbed_reward = estimated_reward + z

        perturbed_reward = estimated_reward

        v_t = find_best_strategy_parallel(defender_strategies, perturbed_reward)

    return v_t

def select_defender_strategy_v2(
    defender_strategies_k,
    estimated_reward: np.ndarray,
    eta: float,
    gamma,
    NUM_LANDSCAPE_CELLS,
    budget_k,
    TARGETS
    ) -> np.ndarray:

    exploration_k = int(np.floor(gamma * budget_k))
    
    n = len(estimated_reward)
    z = np.random.exponential(scale=1/eta, size=n)
    perturbed_reward = estimated_reward + z

    perturbed_reward = estimated_reward

    v_t_exploitation = find_best_strategy_parallel(defender_strategies_k, perturbed_reward)
    
    all_indices = set(TARGETS)
    
    exploitation_indices = set(np.where(v_t_exploitation == 1)[0])
    exploitation_indices = set(np.array(list(exploitation_indices)) + 1)
    
    # print("exploitation indices:", exploitation_indices)
    
    available_for_exploration = list(all_indices - exploitation_indices)
    
    exploration_indices_list = random.sample(available_for_exploration, exploration_k)
    
    # print("exploration_indices_list: ", exploration_indices_list)

    final_indices = exploitation_indices.union(set(exploration_indices_list))

    assert len(final_indices) == budget_k
    
    v_t = np.zeros(NUM_LANDSCAPE_CELLS, dtype=int)
    indices_list = list(final_indices)
    zero_based_indices = np.array(indices_list) - 1
    v_t[zero_based_indices] = 1
    
    return v_t

def run_abm(output_folder, NUM_LANDSCAPE_CELLS):


    dict_of_attacked_targets = {}
    
    simulation_folders = [
    item for item in os.listdir(output_folder) 
    if os.path.isdir(os.path.join(output_folder, item))
    ]

    for simulation_folder in simulation_folders:

        try:

            df = pd.read_csv(os.path.join(output_folder, simulation_folder, "output_files/agent_data.csv"))
            df.dropna(subset=['ROW', 'COL'], inplace=True)

            unique_targets = df["target_attacked"].dropna().unique()

            for target in unique_targets:
                if target not in dict_of_attacked_targets:
                    dict_of_attacked_targets[target] = 0
                dict_of_attacked_targets[target] += 1
        
        except Exception as e:
            pass

    for target, count in dict_of_attacked_targets.items():
        # print(f"Covered target {target} was attacked {count} times.")  
        pass

    df_dict_of_attacked_targets = pd.DataFrame.from_dict(dict_of_attacked_targets, orient='index', columns=['count'])

    df_dict_of_attacked_targets.reset_index(inplace=True)
    df_dict_of_attacked_targets.rename(columns={'index': 'target'}, inplace=True)
                                            
    df_dict_of_attacked_targets.to_csv(os.path.join(output_folder, "df_of_attacked_targets.csv"))
    
    attacker_strategy = [0 for i in range(NUM_LANDSCAPE_CELLS)]

    for target in dict_of_attacked_targets:
        attacker_strategy[int(target - 1)] = 1




    def lat_lon_to_pixel(lats, lons, xmin, ymax, xres, yres):
        """Convert lat/lon coordinates to pixel coordinates"""
        rows = ((ymax - lats) / -yres).astype(int)
        cols = ((lons - xmin) / xres).astype(int)
        return rows, cols

    def find_cropland_use_indices(landuse_sequence):
        indices = []
        start = None
        
        for i, value in enumerate(landuse_sequence):
            if value == 10:
                if start is None:
                    start = i
            else:
                if start is not None:
                    # End subsequence if current value is not 10, 3, 9, or 6
                    if value not in [10, 3, 9, 6]:
                        indices.append((start, i))
                        start = None
        
        # Handle case where sequence ends while in a subsequence starting with 10
        # Only add if the last value is not 10, 3, 9, or 6
        if start is not None:
            last_value = landuse_sequence[-1]
            if last_value not in [10, 3, 9, 6]:
                indices.append((start, len(landuse_sequence)))
        
        return indices
    
    set_attacked_unguarded_boundary = set()
    
    for simulation_folder in simulation_folders:

        try:
            agent_data = pd.read_csv(os.path.join(output_folder, simulation_folder, "output_files/agent_data.csv"))

            geotransform = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").GetGeoTransform()
            ag_xmin, ag_xres, ag_xskew, ag_ymax, ag_yskew, ag_yres = geotransform

            landuse_matrix = gdal.Open(os.path.join(output_folder, simulation_folder, "env", "LULC.tif")).ReadAsArray()
            boundary_patches_guarded = gdal.Open(os.path.join(output_folder, simulation_folder, "env", "defender_coverage_matrix_0.tif")).ReadAsArray()

            boundary_patches = gdal.Open("guarding-policies/FPL-UE_no-memory-model-v1_1/model-runs/game_model-v3_1-run-model-with-intervention-v2-no-knowledge/mitigation-measures-within-plantations/latitude-[[1049000]]-longitude-[[8570800]]/solitary_bulls/random-food-distribition-within-agricultural-plots/landscape-food-probability-forest-0.1-cropland-1.0/water-source-rivers-landscape-0.05/random-memory-forest-and_plantation-fringe-model/full-memory-forest-and_plantation-model/num_days_agent_survives_in_deprivation-10/maximum-food-in-a-forest-cell-25/thermoregulation-threshold-temperature-28/threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/slope_tolerance-30/num_days_agent_survives_in_deprivation-10/elephant_aggression_value_0.8/2010/Mar/num_protected_targets_3/num_strategic_traj_12_num_iterations_12/boundary_raster_discretised_1200m/budget_k_3-max_game_steps_50-eta_10-M_12/coverage_matrix_init/potential_coverage_matrix.tif").ReadAsArray()
            boundary_patches_unguarded = boundary_patches - boundary_patches_guarded

            rows, cols = lat_lon_to_pixel(
                agent_data["latitude"].values, agent_data["longitude"].values, 
                ag_xmin, ag_ymax, ag_xres, ag_yres
            )
            
            landuse_values = landuse_matrix[rows, cols]

            indices = find_cropland_use_indices(landuse_values)
            
            for sequence in indices:

                if (sequence[1] - sequence[0]) >= 6:
                    
                    if boundary_patches_unguarded[rows[sequence[0]], cols[sequence[0]]] != 0:
                        
                        set_attacked_unguarded_boundary.add(boundary_patches_unguarded[rows[sequence[0]], cols[sequence[0]]])
                        
        except Exception as e:
            # print(e)
            pass
        
    for target in set_attacked_unguarded_boundary:
        attacker_strategy[int(target - 1)] = 1

    return attacker_strategy

def step_utility_defender(attacker_strategy_i, defender_strategy_i, targets_df):

    attacker_strategy = np.array(attacker_strategy_i)
    defender_strategy = np.array(defender_strategy_i)

    r = (targets_df['reward'] - targets_df['penalty']).values
    r_t = [a * b for a, b in zip(attacker_strategy, r)]

    reward_01 = np.dot(defender_strategy, r_t)
    reward_02 = np.dot(attacker_strategy, targets_df['penalty'].values)

    return reward_01 + reward_02

def calculate_reward_for_single_defender_strategy(
    defender_strategy,
    attacker_strategy_history_full, 
):

    total_cumulative_reward = 0.0

    num_game_steps = len(attacker_strategy_history_full)

    for i in range(num_game_steps):
        coveragepath = pathlib.Path(os.path.join(
            OUTPUT_FOLDER, 
            "coverage_matrix_init", 
            "game_step_" + str(i+1)
        ))
        
        targets_df = pd.read_csv(os.path.join(coveragepath, "boundary_patch_reward_penalty_matrix.csv"))

        attacker_strategy = attacker_strategy_history_full[i]
        
        v = np.array(defender_strategy)
    
        step_reward = step_utility_defender(attacker_strategy, v, targets_df)
    
        total_cumulative_reward += step_reward

    return total_cumulative_reward, defender_strategy
     
def calculate_best_strategy(defender_strategies, attacker_strategy_history, n_processes=16):

    process_func = partial(
        calculate_reward_for_single_defender_strategy,
        attacker_strategy_history_full=attacker_strategy_history,
    )

    max_reward = float('-inf')
    best_strategy = None
    total_iterations = len(defender_strategies) # Parallelize over the defender strategies

    with mp.Pool(processes=n_processes) as pool:
        for total_reward, v in tqdm(
            pool.imap(process_func, defender_strategies, chunksize=512), 
            total=total_iterations
        ):
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

    plt.savefig(os.path.join(OUTPUT_FOLDER, 'coverage_matrix_init/defender_regret_plot.png'), dpi=300, bbox_inches='tight')
    
    plt.close()

    return

def calculate_defender_regret(defender_strategy_history, attacker_strategy_history, best_hindsight_strategy):
    
    assert len(defender_strategy_history) == len(attacker_strategy_history)
    max_steps = len(defender_strategy_history)

    regret_i_hindsight = 0
    regret_i = 0

    for i, step in enumerate(range(max_steps)):
        attacker_strategy_i = attacker_strategy_history[step]
        defender_strategy_i = defender_strategy_history[step]

        coveragepath = pathlib.Path(os.path.join(
            OUTPUT_FOLDER, 
            "coverage_matrix_init", 
            "game_step_" + str(i+1)
        ))
        
        targets_df = pd.read_csv(os.path.join(coveragepath, "boundary_patch_reward_penalty_matrix.csv"))
        
        r = (targets_df['reward'] - targets_df['penalty']).values
        r_t = [a * b for a, b in zip(attacker_strategy_i, r)]

        regret_i_hindsight += np.dot(best_hindsight_strategy, r_t)

    for i, step in enumerate(range(max_steps)):
        attacker_strategy_i = attacker_strategy_history[step]
        defender_strategy_i = defender_strategy_history[step]

        coveragepath = pathlib.Path(os.path.join(
            OUTPUT_FOLDER, 
            "coverage_matrix_init", 
            "game_step_" + str(i+1)
        ))
        
        targets_df = pd.read_csv(os.path.join(coveragepath, "boundary_patch_reward_penalty_matrix.csv"))
        
        r = (targets_df['reward'] - targets_df['penalty']).values
        r_t = [a * b for a, b in zip(attacker_strategy_i, r)]

        regret_i += np.dot(defender_strategy_i, r_t)

    REGRET = (regret_i_hindsight - regret_i)/max_steps

    return REGRET

def run_single_play(output_folder, MAX_GAME_STEPS, NUM_LANDSCAPE_CELLS, BUDGET_K):

    defender_strategy_history = []
    attacker_strategy_history = []

    coveragepath = pathlib.Path(os.path.join(OUTPUT_FOLDER, "coverage_matrix_init", "game_step_" + str(1)))
    targets_df = pd.read_csv(os.path.join(coveragepath, "boundary_patch_reward_penalty_matrix.csv"))
        
    defender_strategies = generate_defender_strategies(coverage_matrix_path, BUDGET_K, NUM_LANDSCAPE_CELLS, targets_df)
    
    defender_regret_values = []
    
    for i in range(1, MAX_GAME_STEPS+1):
        
        print("\n----- GameStep", i ,"-----")
        
        coveragepath = pathlib.Path(os.path.join(OUTPUT_FOLDER, "coverage_matrix_init", "game_step_" + str(i)))
        
        defender_strategy_matrix = gdal.Open(os.path.join(coveragepath, "defender_coverage_matrix.tif")).ReadAsArray()
        targets_df = pd.read_csv(os.path.join(coveragepath, "boundary_patch_reward_penalty_matrix.csv"))

        v_t = np.zeros(NUM_LANDSCAPE_CELLS, dtype=int)
        indices_list = list(np.unique(defender_strategy_matrix))
        indices_list = [index for index in indices_list if index != 0]
        zero_based_indices = np.array(indices_list) - 1
        v_t[zero_based_indices] = 1
            
        defender_strategy_i = v_t

        # print("Defender strategy:", defender_strategy_i)

        attacker_strategy_i = run_abm(os.path.join(output_folder, "game_step_" + str(i)), NUM_LANDSCAPE_CELLS)

        # print("Attacker strategy:", attacker_strategy_i)

        print(f"Protected target IDs: {targets_df['boundary_patch_id'].loc[np.where(np.array(defender_strategy_i) == 1)[0]].tolist()}")
        print(f"Attacked target IDs: {targets_df['boundary_patch_id'].loc[np.where(np.array(attacker_strategy_i) == 1)[0]].tolist()}")

        defender_strategy_history.append(defender_strategy_i)
        attacker_strategy_history.append(attacker_strategy_i)

        best_defender_strategy_t = calculate_best_strategy(defender_strategies, attacker_strategy_history)

        regret_i = calculate_defender_regret(defender_strategy_history, attacker_strategy_history, best_defender_strategy_t)

        print("Defender regret:", regret_i)
        
        defender_regret_values.append(regret_i)

    plot_defender_regret(defender_regret_values)

    return  







def optimise_strategy(output_folder, NUM_LANDSCAPE_CELLS, BUDGET_K, MAX_GAME_STEPS):

    run_single_play(output_folder, MAX_GAME_STEPS, NUM_LANDSCAPE_CELLS, BUDGET_K)












if __name__ == "__main__":

    boundary_raster_discretised = "boundary_raster_discretised_1200m"

    global coverage_matrix_path   
    
    coverage_matrix_path = "guarding-policies/dynamic-guarding-model-v1/create-strategy-matrix-v2/" + boundary_raster_discretised + "/boundary_raster_discretised.tif"

    potential_coverage_matrix = select_cells_at_distance(
        target_lat=1049000,
        target_lon=8570800,
        distance_cells=145,
        distance_type='euclidean'
    )

    source_file = gdal.Open(coverage_matrix_path)

    cols = source_file.RasterXSize
    rows = source_file.RasterYSize
    projection = source_file.GetProjection()
    geotransform = source_file.GetGeoTransform()

    output_file = os.path.join("guarding-policies/FPL-UE_no-memory-model-v1_1/defender_coverage_matrix.tif")

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(potential_coverage_matrix.astype(np.uint8))

    source_file = None
    output_dataset = None
    
    
    potential_coverage_matrix = gdal.Open(output_file).ReadAsArray()
    
    total_num_targets = np.unique(potential_coverage_matrix)[-1]
    
    print("length of strategy matrix:", total_num_targets)

    TARGETS  = np.unique(potential_coverage_matrix)

    TARGETS = [x for x in TARGETS if x != 0]

    print("Total number of targets to protect:", len(TARGETS), "\n", "TARGETS:", TARGETS)

    coverage_matrix_path = os.path.join(output_file)

    proximity_filter_parameter = [0.999]
    cost_function_threshold_parameter = [0]
    
    parameter_combinations = list(itertools.product(proximity_filter_parameter, cost_function_threshold_parameter))

    num_resources_k = [6]
    
    for ranger_proximity_threshold, cost_ranger_proximity_threshold in parameter_combinations:

        for k in num_resources_k: 

            BUDGET_K = k                   # Maximum number of cells that can be protected by the defenders at every time-step
            MAX_GAME_STEPS = 50                         # Maximum number of time-steps in the game
            eta = 10                                   # reward perturbation parameter
            M = 12                                      # parameter in the GR algorithm

            FPL_UE_params = (
                "budget_k_"
                + str(BUDGET_K)
                + "-max_game_steps_"
                + str(MAX_GAME_STEPS) 
                + "-eta_"
                + str(eta)
                + "-M_"
                + str(M)
            )
    

            model_params = {
                    "year": 2010,
                    "month": "Mar",
                    "num_bull_elephants": 1,
                    "area_size": 1100,
                    "spatial_resolution": 30,
                    "max_food_val_cropland": 100,
                    "max_food_val_forest": 25,
                    "prob_food_forest": 0.10,
                    "prob_food_cropland": 1.0,
                    "prob_water_sources": 0.05,
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
                    "num_processes": 12,
                    "iterations": 12,
                    "max_time_steps": 288 * 10,
                    "aggression_threshold_enter_cropland": 1.0,
                    "human_habituation_tolerance": 1.0,
                    "elephant_agent_visibility_radius": 500,
                    "plot_stepwise_target_selection": False,
                    "threshold_days_of_food_deprivation": 0,
                    "threshold_days_of_water_deprivation": 3,
                    "number_of_feasible_movement_directions": 4,
                    "track_in_mlflow": False,
                    "elephant_starting_location": "user_input",
                    "elephant_starting_latitude": [[1049000]],
                    "elephant_starting_longitude": [[8570800]],
                    "elephant_aggression_value": 0.8,
                    "elephant_crop_habituation": True,
                    "ranger_proximity_threshold": None,
                    "cost_ranger_proximity_threshold": None,
                    "num_protected_targets": k
                }
            
            NUM_STRATEGIC_TRAJECTORIES = 12

            experiment_name = "mitigation-measures-within-plantations"

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

            food_availability_sceanario = "random-food-distribition-within-agricultural-plots"

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

            target_folder = f"num_protected_targets_" + str(model_params["num_protected_targets"])

            simulation_repeats = f'num_strategic_traj_{NUM_STRATEGIC_TRAJECTORIES}_num_iterations_{model_params["iterations"]}'



            output_folder = os.path.join(
                "guarding-policies/FPL-UE_no-memory-model-v1_1/model-runs/game_model-v3_1-run-model-with-intervention-v2-no-knowledge/",
                experiment_name,
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
                str(model_params["month"]),
                target_folder,
                simulation_repeats,
                boundary_raster_discretised,
                FPL_UE_params,
                "agent-based-model-runs"
            )
            
            global OUTPUT_FOLDER
            
            OUTPUT_FOLDER = os.path.join(
                "guarding-policies/FPL-UE_no-memory-model-v1_1/model-runs/game_model-v3_1-run-model-with-intervention-v2-no-knowledge/",
                experiment_name,
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
                str(model_params["month"]),
                target_folder,
                simulation_repeats,
                boundary_raster_discretised,
                FPL_UE_params
            )
            
            optimise_strategy(
                output_folder=output_folder,
                NUM_LANDSCAPE_CELLS = total_num_targets,
                BUDGET_K = BUDGET_K,
                MAX_GAME_STEPS = MAX_GAME_STEPS
            )
