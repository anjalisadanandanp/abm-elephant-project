#------------importing libraries----------------#
import os

import sys
sys.path.append(os.getcwd())

import pathlib
import yaml

from mesageo_elephant_project.elephant_project.model.abm_model_HEC import batch_run_model
#------------importing libraries----------------#



model_params = {
    "year": 2010,
    "month": "Jan",
    "num_bull_elephants": 1, 
    "area_size": 1100,              
    "spatial_resolution": 30, 
    "max_food_val_cropland": 60,
    "max_food_val_forest": 30,
    "prob_food_forest": 0.85,
    "prob_food_cropland": 0.85,
    "prob_water_sources": 0.1,
    "thermoregulation_threshold": 28,
    "prob_drink_water_dry": 0.3333,   
    "prob_drink_water_wet": 0.0333, 
    "movement_fitness_depreceation": -0.000347222,        
    "fitness_increment_when_eats_food": 0.05,      
    "fitness_increment_when_drinks_water_dry": 0.05,    
    "fitness_increment_when_drinks_water_wet": 0.005,    
    "knowledge_from_fringe": 1500,   
    "prob_crop_damage": 0.05,           
    "prob_infrastructure_damage": 0.01,
    "percent_memory_elephant": 0.8,   
    "radius_food_search": 750,     
    "radius_water_search": 750, 
    "fitness_threshold": 0.4,   
    "terrain_radius": 750,       
    "tolerance": 1000,   
    "num_processes": 8,
    "iterations": 8,
    "max_time_steps": 2880,
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