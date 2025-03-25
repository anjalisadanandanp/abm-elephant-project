#------------importing libraries----------------#
import os

import sys
sys.path.append(os.getcwd())

import pathlib
import yaml

import mlflow

from mesageo_elephant_project.elephant_project.model.abm_model_HEC_v2 import batch_run_model
#------------importing libraries----------------#



model_params = {
    "year": 2010,
    "month": "Jan",
    "num_bull_elephants": 1, 
    "area_size": 1100,              
    "spatial_resolution": 30, 
    "max_food_val_cropland": 100,
    "max_food_val_forest": 25,
    "prob_food_forest": 0.10,
    "prob_food_cropland": 0.10,
    "prob_water_sources": 0.0,
    "thermoregulation_threshold": 32,
    "movement_fitness_depreceation": -0.000347222,        
    "knowledge_from_fringe": 1500,   
    "prob_crop_damage": 0.05,           
    "prob_infrastructure_damage": 0.01,
    "percent_memory_elephant": 0.375,   
    "radius_food_search": 750,     
    "radius_water_search": 750, 
    "radius_forest_search": 1500,
    "fitness_threshold": 0.4,   
    "terrain_radius": 750,       
    "slope_tolerance": 32,   
    "num_processes": 8,
    "iterations": 8,
    "max_time_steps": 288*1,
    "aggression_threshold_enter_cropland": 0.5,
    "elephant_agent_visibility_radius": 500,
    "plot_stepwise_target_selection": False,
    "threshold_days_of_food_deprivation": 0,
    "threshold_days_of_water_deprivation": 0,
    "number_of_feasible_movement_directions": 3,
    "track_in_mlflow": True
    }





def run_model():

    experiment_name = "exploratory-search-landscape-food-probability"
    elephant_category = "solitary_bulls"
    landscape_food_probability = "landscape-food-probability-forest-" + str(model_params["prob_food_forest"]) + "-cropland-" + str(model_params["prob_food_cropland"])
    water_holes_probability = "water-holes-within-landscape-" + str(model_params["prob_water_sources"])
    memory_matrix_type = "random-memory-matrix-model"
    elephant_thermoregulation_threshold = "thermoregulation-threshold-temperature-" + str(model_params["thermoregulation_threshold"])
    threshold_food_derivation_days = "threshold_days_of_food_deprivation-" + str(model_params["threshold_days_of_food_deprivation"])
    threshold_water_derivation_days = "threshold_days_of_water_deprivation-" + str(model_params["threshold_days_of_water_deprivation"])
    
    output_folder = os.path.join(os.getcwd(), "model_runs/", experiment_name, elephant_category, landscape_food_probability, water_holes_probability, memory_matrix_type, elephant_thermoregulation_threshold, threshold_food_derivation_days, threshold_water_derivation_days, str(model_params["year"]), str(model_params["month"]))

    if os.path.exists(output_folder):
        os.system("rm -r " + output_folder)
    
    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_folder, 'model_parameters.yaml'), 'w') as configfile:
        yaml.dump(model_params, configfile, default_flow_style=False)

    batch_run_model(model_params, experiment_name, output_folder)

    return



if __name__ == "__main__":
    run_model()