#------------importing libraries----------------#
import os

import sys
sys.path.append(os.getcwd())

import pathlib
import yaml

from mesageo_elephant_project.elephant_project.model.abm_model_HEC_v2 import batch_run_model
#------------importing libraries----------------#



model_params = {
    "year": 2010,
    "month": "Jan",
    "num_bull_elephants": 1, 
    "area_size": 1100,              
    "spatial_resolution": 30, 
    "max_food_val_cropland": 60,
    "max_food_val_forest": 30,
    "prob_food_forest": 0.01,
    "prob_food_cropland": 0.01,
    "prob_water_sources": 0.0,
    "thermoregulation_threshold": 28,
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
    "num_processes": 1,
    "iterations": 1,
    "max_time_steps": 288*30,
    "aggression_threshold_enter_cropland": 0.5,
    "elephant_agent_visibility_radius": 750,
    "plot_stepwise_target_selection": False,
    "threshold_days_of_food_deprivation": 1,
    "threshold_days_of_water_deprivation": 1,
    "number_of_feasible_movement_directions": 3
    }





def run_model():

    expt_name = "mitigation_measures"
    elephant_category = "solitary_bulls"
    
    output_folder = os.path.join(os.getcwd(), "model_runs/", expt_name, elephant_category, str(model_params["year"]), str(model_params["month"]))

    if os.path.exists(output_folder):
        os.system("rm -r " + output_folder)
    
    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_folder, 'model_parameters.yaml'), 'w') as configfile:
        yaml.dump(model_params, configfile, default_flow_style=False)

    batch_run_model(model_params, output_folder)

    return



if __name__ == "__main__":
    run_model()