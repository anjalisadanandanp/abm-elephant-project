import numpy as np
import pandas as pd
from osgeo import gdal
import os
import itertools
from tqdm import tqdm
from pyproj import Proj, transform    

import warnings
warnings.filterwarnings("ignore")


def find_food_value_association():

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
            "slope_tolerance": 35,
            "num_processes": 47,
            "iterations": 47,
            "max_time_steps": 288 * 30,
            "aggression_threshold_enter_cropland": 1.0,
            "human_habituation_tolerance": 1.0,
            "elephant_agent_visibility_radius": 500,
            "plot_stepwise_target_selection": False,
            "threshold_days_of_food_deprivation": 0,
            "threshold_days_of_water_deprivation": 3,
            "number_of_feasible_movement_directions": 4,
            "track_in_mlflow": False,
            "elephant_starting_location": "user_input",
            "elephant_starting_latitude": [[1052166]],
            "elephant_starting_longitude": [[8572829]],
            "elephant_aggression_value": 0.8,
            "elephant_crop_habituation": True,
            "ranger_proximity_threshold": None,
            "cost_ranger_proximity_threshold": None,
        }

    NUM_STRATEGIC_TRAJECTORIES = 188

    experiment_name = "mitigation-measures-within-plantations-FPL-UE_v3_1"

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

    targets_to_cover = [0]

    target_folder = f"protected_targets_{'_'.join(map(str, targets_to_cover))}"

    simulation_repeats = f'num_strategic_traj_{NUM_STRATEGIC_TRAJECTORIES}_num_iterations_{model_params["iterations"]}'

    folder = os.path.join(
        "guarding-policies/static-guarding-model-v1/model-runs/exploratory_search_on_evading_trajectories/game_model-v3_1-run-model-without-intervention-v2/",
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
    )

    # agricultural_plots = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").ReadAsArray()

    # food_matrix = gdal.Open(os.path.join("guarding-policies/static-guarding-model-v1/model-runs/exploratory_search_on_evading_trajectories/game_model-v3_1-run-model-without-intervention-v2/mitigation-measures-within-plantations-FPL-UE_v3_1/latitude-[[1052166]]-longitude-[[8572829]]/solitary_bulls/random-food-distribition-within-agricultural-plots-and-other-plantation-cells/landscape-food-probability-forest-0.1-cropland-1.0/water-source-rivers-landscape-0.05/random-memory-forest-and_plantation-fringe-model/full-memory-forest-and_plantation-model/num_days_agent_survives_in_deprivation-10/maximum-food-in-a-forest-cell-25/thermoregulation-threshold-temperature-28/threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/slope_tolerance-35/num_days_agent_survives_in_deprivation-10/elephant_aggression_value_0.8/2010/Mar/protected_targets_0/num_strategic_traj_188_num_iterations_47/game_step_1/0ae61fb2-7e21-479c-b272-88f066f7e68d/env/food_matrix_0.1_1.0_.tif")).ReadAsArray()

    for max_cells_per_group in tqdm([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]):

        # boundary_patches = gdal.Open(os.path.join("guarding-policies/dynamic-guarding-model-v1/create-strategy-matrix-v2/boundary_raster_discretised_" + str(int(max_cells_per_group*30)) + "m/boundary_raster_discretised.tif")).ReadAsArray()

        save_folder = os.path.join("guarding-policies/dynamic-guarding-model-v1/find_boundary_patch_reward_penalty_values-v2_3/boundary_raster_discretised_" + str(int(max_cells_per_group*30)) + "m")

        df = pd.read_csv(os.path.join(save_folder, "association_matrix_num_visiting_trajs.csv"))
        
        rewards = df["num_visiting_trajectories"].values.tolist()

        rewards = np.array(rewards)

        global_min = np.min(rewards)
        global_max = np.max(rewards)

        normalized_all = (rewards - global_min) / (global_max - global_min) / 2

        rewards = normalized_all.flatten().tolist()

        print("reward update:", rewards)

        penalties = [-r for r in rewards]

        df_reward_penalty = pd.DataFrame({
            "boundary_patch_id": df["boundary_patch_id"],
            "reward": rewards,
            "penalty": penalties
        })
        df_reward_penalty.to_csv(os.path.join(save_folder, "boundary_patch_reward_penalty_matrix.csv"), index=False)

find_food_value_association()