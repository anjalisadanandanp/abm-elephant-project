import numpy as np
import pandas as pd
from osgeo import gdal
import os
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from pyproj import Proj, transform    
from mpl_toolkits.basemap import Basemap    
import matplotlib.colors as mcolors   

import warnings
warnings.filterwarnings("ignore")




def generate_parameter_combinations(model_params_all):

    month = model_params_all["month"]
    max_food_val_forest = model_params_all["max_food_val_forest"]
    prob_food_forest = model_params_all["prob_food_forest"]
    prob_food_cropland = model_params_all["prob_food_cropland"]
    thermoregulation_threshold = model_params_all["thermoregulation_threshold"]
    threshold_days_food = model_params_all["threshold_days_of_food_deprivation"]
    threshold_days_water = model_params_all["threshold_days_of_water_deprivation"]
    prob_water_sources = model_params_all["prob_water_sources"]
    num_days_agent_survives_in_deprivation = model_params_all["num_days_agent_survives_in_deprivation"]
    slope_tolerance = model_params_all["slope_tolerance"]
    elephant_aggression_value = model_params_all["elephant_aggression_value"]

    combinations = list(itertools.product(
        month,
        max_food_val_forest,
        prob_food_forest,
        prob_food_cropland,
        thermoregulation_threshold,
        threshold_days_food,
        threshold_days_water,
        prob_water_sources,
        num_days_agent_survives_in_deprivation,
        slope_tolerance,
        elephant_aggression_value
    ))

    all_param_dicts = []
    for combo in combinations:
        params_dict = model_params_all.copy()
        
        params_dict.update({
            "month": combo[0],
            "max_food_val_forest": combo[1],
            "prob_food_forest": combo[2],
            "prob_food_cropland": combo[3],
            "thermoregulation_threshold": combo[4],
            "threshold_days_of_food_deprivation": combo[5],
            "threshold_days_of_water_deprivation": combo[6],
            "prob_water_sources": combo[7],
            "num_days_agent_survives_in_deprivation": combo[8],
            "slope_tolerance": combo[9],
            "elephant_aggression_value": combo[10]
        })
        
        all_param_dicts.append(params_dict)
    
    return all_param_dicts


def return_output_folder(experiment_name, model_params):

    elephant_category = "solitary_bulls"
    starting_location = "latitude-" + str(model_params["elephant_starting_latitude"]) + "-longitude-" + str(model_params["elephant_starting_longitude"])
    landscape_food_probability = "landscape-food-probability-forest-" + str(model_params["prob_food_forest"]) + "-cropland-" + str(model_params["prob_food_cropland"])
    food_availability_sceanario = "random-food-distribition-within-within-plantation-cells"
    water_availability_sceanario = "water-source-rivers-landscape-" + str(model_params["prob_water_sources"])
    food_memory_matrix_type = "random-memory-forest-and_plantation-fringe-model"
    water_memory_matrix_type = "full-memory-forest-and_plantation-model"
    num_days_agent_survives_in_deprivation = "num_days_agent_survives_in_deprivation-" + str(model_params["num_days_agent_survives_in_deprivation"])
    maximum_food_in_a_forest_cell = "maximum-food-in-a-forest-cell-" + str(model_params["max_food_val_forest"])
    elephant_thermoregulation_threshold = "thermoregulation-threshold-temperature-" + str(model_params["thermoregulation_threshold"])
    threshold_food_derivation_days = "threshold_days_of_food_deprivation-" + str(model_params["threshold_days_of_food_deprivation"])
    threshold_water_derivation_days = "threshold_days_of_water_deprivation-" + str(model_params["threshold_days_of_water_deprivation"])
    slope_tolerance = "slope_tolerance-" + str(model_params["slope_tolerance"])
    num_days_agent_survives_in_deprivation = "num_days_agent_survives_in_deprivation-" + str(model_params["num_days_agent_survives_in_deprivation"])
    elephant_aggression_value = "elephant_aggression_value_" + str(model_params["elephant_aggression_value"])

    output_folder = os.path.join(experiment_name, starting_location, elephant_category, food_availability_sceanario, landscape_food_probability, 
                                 water_availability_sceanario, food_memory_matrix_type, water_memory_matrix_type, num_days_agent_survives_in_deprivation, maximum_food_in_a_forest_cell, 
                                 elephant_thermoregulation_threshold, threshold_food_derivation_days, threshold_water_derivation_days, 
                                 slope_tolerance, num_days_agent_survives_in_deprivation, elephant_aggression_value,
                                 str(model_params["year"]), str(model_params["month"]))
    
    return output_folder
    

def find_food_value_association():

    model_params_all = {
        "year": 2010,
        "month": ["Mar", "Aug"],
        "num_bull_elephants": 1, 
        "area_size": 1100,              
        "spatial_resolution": 30, 
        "max_food_val_cropland": 100,
        "max_food_val_forest": [5],
        "prob_food_forest": [0.10],
        "prob_food_cropland": [0.10],
        "prob_water_sources": [1.0],
        "thermoregulation_threshold": [28, 32],
        "num_days_agent_survives_in_deprivation": [10],     
        "knowledge_from_fringe": 1500,   
        "prob_crop_damage": 0.05,           
        "prob_infrastructure_damage": 0.01,
        "percent_memory_elephant": 0.375,   
        "radius_food_search": 750,     
        "radius_water_search": 750, 
        "radius_forest_search": 1500,
        "fitness_threshold": 0.4,   
        "terrain_radius": 750,       
        "slope_tolerance": [30, 32.5, 35, 37.5, 40],
        "num_processes": 32,
        "iterations": 128,
        "max_time_steps": 288*30,
        "aggression_threshold_enter_cropland": 1.0,
        "elephant_agent_visibility_radius": 500,
        "plot_stepwise_target_selection": False,
        "threshold_days_of_food_deprivation": [0],
        "threshold_days_of_water_deprivation": [3],
        "number_of_feasible_movement_directions": 3,
        "track_in_mlflow": False,
        "elephant_starting_location": "user_input",
        "elephant_starting_latitude": 1049237,
        "elephant_starting_longitude": 8570917,
        "elephant_aggression_value": [0.2, 0.8],
        "elephant_crop_habituation": False
        }

    param_dicts = generate_parameter_combinations(model_params_all)
    experiment_name = "model-without-intervention"

    output_folders = [return_output_folder(experiment_name, param_dict) for param_dict in param_dicts]

    agricultural_plots = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").ReadAsArray()
    boundary_patches = gdal.Open("create-strategy-matrix/boundary_raster_discretised.tif").ReadAsArray()

    geotransform = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").GetGeoTransform()
    ag_xmin, ag_xres, ag_xskew, ag_ymax, ag_yskew, ag_yres = geotransform
    ag_rows, ag_cols = agricultural_plots.shape

    row_size, col_size = boundary_patches.shape
    xmin, xres, xskew, ymax, yskew, yres = gdal.Open("create-strategy-matrix/boundary_raster_discretised.tif").GetGeoTransform()
    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
    LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
    LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

    for folder in tqdm(output_folders):

        run_folder = os.path.join("/mnt/qdata/abm-elephant-project/aryabhata-runs/", folder)
        expts = os.listdir(run_folder)

        landuse_matrix = gdal.Open(os.path.join(run_folder, expts[-1], "env", "LULC.tif")).ReadAsArray()
        food_matrix = gdal.Open(os.path.join(run_folder, expts[-1], "env", "food_matrix_" + str(folder.split("/")[4].split("-")[4]) + "_" + str(folder.split("/")[4].split("-")[6]) + "_.tif")).ReadAsArray()

        save_folder = os.path.join(os.getcwd(), "create-boundary-agricultural-patch-association-matrix-v2/outputs/", folder)

        df = pd.read_csv(os.path.join(save_folder, "association_matrix_num_visiting_trajs.csv"))

        column_boundary_ids = []
        column_food_value = []

        for index, row in df.iterrows():

            if index == 0:
                pass

            else:

                non_zero_items = [(col, val) for col, val in row.items() if val != 0]

                row_name = row['Unnamed: 0']
                col_names = [col for col in df.columns if col != 'Unnamed: 0']

                boundary_patch_id = int(row_name.split("_")[-1])

                boundarymask = boundary_patches == boundary_patch_id
                associated_plots = []
                food_within_plots = []

                flag = False
                
                for col, val in non_zero_items:

                    try:
                        agricultural_plot_id = int(col.split("_")[-1])
                        ag_mask = agricultural_plots == agricultural_plot_id
                        if np.any(ag_mask):
                            flag = True
                            associated_plots.append(agricultural_plot_id)

                            foodmask = food_matrix[ag_mask]
                            food_within_plots.append(np.sum(foodmask))
                    except:
                        pass

                if flag:
                    print(f"Boundary Patch ID: {boundary_patch_id}, Associated Agricultural Plots: {associated_plots}, Food within Plots: {food_within_plots}")
                    total_food = np.sum(food_within_plots)

                else:
                    total_food = 0

                column_boundary_ids.append(boundary_patch_id)
                column_food_value.append(total_food)

        df_new = pd.DataFrame({
            "boundary_patch_id": column_boundary_ids,
            "total_food_value": column_food_value
        })
        df_new.to_csv(os.path.join(save_folder, "boundary_patch_association_matrix.csv"), index=False)



        rewards = []

        num_trajs_threatening = pd.read_csv(os.path.join(save_folder, "association_matrix_num_visiting_trajs.csv"))
        num_food_resources_under_attack = pd.read_csv(os.path.join(save_folder, "boundary_patch_association_matrix.csv"))

        for boundary_id in num_food_resources_under_attack["boundary_patch_id"].unique():

            reward = 0.0

            rows = num_trajs_threatening[num_trajs_threatening["Unnamed: 0"] == f"boundary_patch_{boundary_id}"]

            num_intersecting_trajs = rows.iloc[0, 1:].sum()

            if num_intersecting_trajs == 0:
                reward = 0.0
                rewards.append(reward)

            else:
                reward = num_food_resources_under_attack[num_food_resources_under_attack["boundary_patch_id"] == boundary_id]["total_food_value"].values[0]*num_intersecting_trajs

                print(f"Boundary Patch ID: {boundary_id}, Reward: {reward}")

                rewards.append(reward)

        rewards = np.array(rewards).flatten()

        print(rewards.shape)

        global_min = np.min(rewards)
        global_max = np.max(rewards)

        normalized_all = (rewards - global_min) / (global_max - global_min) / 2

        rewards = normalized_all.flatten().tolist()

        print("reward update:", rewards)

        penalties = [-r for r in rewards]

        df_reward_penalty = pd.DataFrame({
            "boundary_patch_id": df_new["boundary_patch_id"],
            "reward": rewards,
            "penalty": penalties
        })
        df_reward_penalty.to_csv(os.path.join(save_folder, "boundary_patch_reward_penalty_matrix.csv"), index=False)



find_food_value_association()