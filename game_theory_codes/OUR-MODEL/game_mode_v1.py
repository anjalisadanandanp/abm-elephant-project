import numpy as np
from tqdm import tqdm
from itertools import combinations
import random
import matplotlib.pyplot as plt
import os
import yaml
from osgeo import gdal
import pandas as pd
import pathlib
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

import sys
sys.path.append(os.getcwd())

import importlib
module = importlib.import_module('game_theory_codes.FPL-UE.play-games.abm_model_HEC_with_landscape_deterrent_policies')
batch_run_model = module.batch_run_model

class RepeatedStackelbergGame:

    def __init__(
            
        self,
        num_targets,
        num_defender_resources,
        defender_payoffs,
        defender_penalties
    ):
        
        self.num_targets = num_targets
        self.num_defender_resources = num_defender_resources

        self.defender_payoffs = np.array(defender_payoffs)
        self.defender_penalties = np.array(defender_penalties)

        self.defender_strategy_history = []
        self.attacker_strategy_history = []

        self.defender_strategies = []
        for targets_to_cover in tqdm(combinations(range(self.num_targets), self.num_defender_resources)):
            self.defender_strategies.append([1 if i in targets_to_cover else 0 for i in range(self.num_targets)])

    def adversary_response_model_v1(self):

        if self.attacker_strategy_history == []:
            return [1/self.num_targets for target in range(self.num_targets)]

        else:
            attack_probability = np.sum(np.array(self.attacker_strategy_history), axis=0)/self.step
            return attack_probability
        
    def expected_step_utility_defender(self, attacker_strategy, defender_strategy):

        r = self.defender_payoffs - self.defender_penalties
        r_t = [a * b for a, b in zip(attacker_strategy, r)]

        reward_01 = np.dot(defender_strategy, r_t)
        reward_02 = np.dot(attacker_strategy, self.defender_penalties)

        U = reward_01 + reward_02

        return U
    
    def take_action_sequential(self):
        
        if self.defender_strategy_history == []:

            targets_to_cover = random.sample(range(self.num_targets), self.num_defender_resources)
            strategy = np.array([1 if i in targets_to_cover else 0 for i in range(self.num_targets)])

            return strategy

        else:

            best_utility = float('-inf')
            best_strategy = np.zeros(self.num_targets)

            attack_probability = self.adversary_response_model_v1()

            for strategy in self.defender_strategies:

                utility = self.expected_step_utility_defender(attacker_strategy=attack_probability, defender_strategy=strategy)
                
                if utility > best_utility:
                    best_utility = utility
                    best_strategy = strategy

            return best_strategy

    def take_action_parallel(self):
        
        if self.defender_strategy_history == []:

            targets_to_cover = random.sample(range(self.num_targets), self.num_defender_resources)
            strategy = np.array([1 if i in targets_to_cover else 0 for i in range(self.num_targets)])

            return strategy

        else:

            best_utility = float('-inf')
            best_strategy = np.zeros(self.num_targets)

            attack_probability = self.adversary_response_model_v1()

            calculate_utility = partial(self.expected_step_utility_defender, 
                                    attacker_strategy=attack_probability)
            
            best_utility = float('-inf')
            best_strategy = None
            
            with Pool(processes=8) as pool:
                for strategy, utility in tqdm(zip(self.defender_strategies, 
                                        pool.imap(calculate_utility, self.defender_strategies))):
                    if utility > best_utility:
                        best_utility = utility
                        best_strategy = strategy
            
            return best_strategy
        
    def calculate_reward_for_strategy_best(self, defender_strategy):

        total_strategy_utility = 0

        for attacker_strategy in self.attacker_strategy_history:
            step_utility = self.expected_step_utility_defender(attacker_strategy, defender_strategy)
            total_strategy_utility += step_utility
        
        return total_strategy_utility

    def find_best_strategy_sequential(self):

        max_reward = float('-inf')
        best_strategy = None
        
        for strategy in self.defender_strategies:
            total_reward  = self.calculate_reward_for_strategy_best(strategy)
            if total_reward > max_reward:
                max_reward = total_reward
                best_strategy = strategy
        
        return best_strategy

    def find_best_strategy_parallel(self):

        process_strategy = partial(self.calculate_reward_for_strategy_best)
        
        max_reward = float('-inf')
        best_strategy = None
        
        with Pool(processes=8) as pool:
            for strategy, reward in tqdm(zip(self.defender_strategies, 
                                    pool.imap(process_strategy, self.defender_strategies))):
                if reward > max_reward:
                    max_reward = reward
                    best_strategy = strategy
        
        return best_strategy, max_reward

    def calculate_defender_regret(self, best_hindsight_strategy):
        
        regret_i_hindsight = 0
        regret_i = 0

        for step in range(self.step):
            attacker_strategy_i = self.attacker_strategy_history[step]
            defender_strategy_i = self.defender_strategy_history[step]

            r = self.defender_payoffs - self.defender_penalties
            r_t = [a * b for a, b in zip(attacker_strategy_i, r)]

            regret_i_hindsight += np.dot(best_hindsight_strategy, r_t)

        for step in range(self.step):
            attacker_strategy_i = self.attacker_strategy_history[step]
            defender_strategy_i = self.defender_strategy_history[step]

            r = self.defender_payoffs - self.defender_penalties
            r_t = [a * b for a, b in zip(attacker_strategy_i, r)]

            regret_i += np.dot(defender_strategy_i, r_t)

        REGRET = (regret_i_hindsight - regret_i)/(self.step+1)

        return REGRET

    def run_simulation(self, num_steps, model_params, experiment_name, output_folder):

        defender_regret_values = []

        for step in tqdm(range(num_steps)):

            self.step = step 

            self.defender_strategy_i = self.take_action_sequential()
            self.defender_strategy_history.append(self.defender_strategy_i)

            coverage_matrix = create_defender_coverage_matrix(self.defender_strategy_i)

            path = pathlib.Path(os.path.join(output_folder, "game_step_" + str(step + 1)))
            path.mkdir(parents=True, exist_ok=True)

            plot_and_save_defender_coverage(coverage_matrix, os.path.join(output_folder, "game_step_" + str(step + 1)))

            self.attacker_strategy_i = run_abm(model_params, experiment_name, os.path.join(output_folder, "game_step_" + str(step + 1)))
            self.attacker_strategy_history.append(self.attacker_strategy_i)

            print("step:", step, "defender strategy:", self.defender_strategy_i, "attacker strategy:", self.attacker_strategy_i)
        
            best_defender_strategy_t = self.find_best_strategy_sequential()

            regret_i = self.calculate_defender_regret(best_defender_strategy_t)

            print("Defender regret:", regret_i)

            if self.step != 0:
                defender_regret_values.append(regret_i)

        self.plot_defender_regret(defender_regret_values)

        return

    def plot_defender_regret(self, defender_regret_values):

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

        plt.savefig('defender_regret_plot.png', dpi=300, bbox_inches='tight')
        
        plt.close()

        return




def create_defender_coverage_matrix(defender_strategy):

    potential_coverage_matrix = gdal.Open(os.path.join("game_theory_codes/FPL-UE/create-strategy-set/outputs/high_res_indexed_forest_agricultural_fringe.tif")).ReadAsArray()
    
    coverage_matrix = np.zeros_like(potential_coverage_matrix)

    target_ids = [index + 1 for index, value in enumerate(defender_strategy) if value != 0]

    for target_id in target_ids:
        mask = potential_coverage_matrix == target_id
        coverage_matrix[mask] = 1

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





    source_file = gdal.Open("game_theory_codes/FPL-UE/create-strategy-set/outputs/high_res_indexed_forest_agricultural_fringe.tif")

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

def run_abm(model_params, experiment_name, output_folder):

    with open(os.path.join(output_folder, "model_parameters.yaml"), "w") as configfile:
        yaml.dump(model_params, configfile, default_flow_style=False)


    batch_run_model(model_params, experiment_name, output_folder)


    #--------------------calculate attacker strategy matrix--------------------#
    potential_coverage_matrix = gdal.Open(os.path.join("game_theory_codes/FPL-UE/create-strategy-set/outputs/high_res_indexed_forest_agricultural_fringe.tif")).ReadAsArray()
    
    trajectory_matrix = np.zeros_like(potential_coverage_matrix, dtype=np.uint8)

    num_simulations = 0

    for simulation_folder in os.listdir(output_folder):

        try:

            df = pd.read_csv(os.path.join(output_folder, simulation_folder, "output_files/agent_data.csv"))
            df.dropna(subset=['ROW', 'COL'], inplace=True)
        
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
    #--------------------calculate attacker strategy matrix--------------------#


    #--------------------save attacker strategy matrix--------------------#
    fig, ax = plt.subplots(figsize=(8, 8))
    
    cmap = 'coolwarm'
    
    im = ax.imshow(trajectory_matrix, cmap=cmap, vmin=0, vmax=np.max(trajectory_matrix))
    
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = plt.colorbar(im, shrink=0.5)

    ticks = np.linspace(0, np.max(trajectory_matrix), num=5) 
    cbar.set_label("Attack Probability", rotation=90)
    cbar.set_ticks(ticks)

    plt.savefig(
        os.path.join(output_folder, "attacker_strategy_matrix.png"),
        bbox_inches="tight",
        dpi=300,
        )
    #--------------------save attacker strategy matrix--------------------#


    #--------------------save attacker strategy matrix--------------------#
    source_file = gdal.Open("game_theory_codes/FPL-UE/create-strategy-set/outputs/high_res_indexed_forest_agricultural_fringe.tif")

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
    output_band.WriteArray(trajectory_matrix.astype(np.float32))

    source_file = None
    output_dataset = None
    #--------------------save attacker strategy matrix--------------------#



    targets_matrix = gdal.Open("game_theory_codes/FPL-UE/create-strategy-set/outputs/low_res_indexed_forest_agricultural_fringe.tif").ReadAsArray()
    unique_values = np.unique(targets_matrix)
    non_zero_unique_values = unique_values[unique_values != 0]
    total_targets = len(non_zero_unique_values)
    attacker_strategy = [0 for i in range(total_targets)]


    coverage_matrix = gdal.Open("game_theory_codes/FPL-UE/create-strategy-set/outputs/high_res_indexed_forest_agricultural_fringe.tif").ReadAsArray()
    plantation_rows, plantation_cols = np.where(coverage_matrix != 0)

    for row, col in zip(plantation_rows, plantation_cols):
        if trajectory_matrix[row, col] > 0:
            attacker_strategy[int(coverage_matrix[row,col]-1)] += trajectory_matrix[row, col]

    def probabilistic_binary_conversion(original_list):

        filtered_list = [value for value in original_list if value != 0]
        percentile_value = np.percentile(filtered_list, 50)

        binary_list = []
        
        for value in original_list:

            probability = min(1, value / percentile_value)
            binary_value = np.random.binomial(1, probability)
            binary_list.append(binary_value)
        
        return binary_list
    
    attacker_strategy = probabilistic_binary_conversion(attacker_strategy)

    return attacker_strategy













if __name__ == "__main__":

    model_params = {
            "year": 2010,
            "month": "Mar",
            "num_bull_elephants": 1,
            "area_size": 1100,
            "spatial_resolution": 30,
            "max_food_val_cropland": 100,
            "max_food_val_forest": 15,
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
            "elephant_crop_habituation": True
        }

    NUM_DEFENDER_RESOURCES = 7
    MAX_GAME_STEPS = 10

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

    game_params = "budget_k_" + str(NUM_DEFENDER_RESOURCES) + "_MAX_GAME_STEPS_" + str(MAX_GAME_STEPS)

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


    targets_df = pd.read_csv("game_theory_codes/FPL-UE/create-strategy-set/outputs/target_rewards_penalties.csv")



    defender_payoffs = targets_df["reward"].values.tolist()
    defender_penalties = targets_df["penalty"].values.tolist()


    game = RepeatedStackelbergGame(
        num_targets=len(targets_df),
        num_defender_resources=NUM_DEFENDER_RESOURCES,
        defender_payoffs=defender_payoffs,
        defender_penalties = defender_penalties
    )
    
    game.run_simulation(num_steps=MAX_GAME_STEPS, 
                        model_params=model_params, 
                        experiment_name=experiment_name,
                        output_folder=output_folder,)
    