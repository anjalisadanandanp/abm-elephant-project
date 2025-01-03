#------------importing libraries----------------#
import os

import sys
sys.path.append(os.getcwd())

import json
import pathlib
import calendar
import multiprocessing

from mesageo_elephant_project.elephant_project.model.abm_model_HEC import batch_run_model
#------------importing libraries----------------#


food_val_cropland = 100
food_val_forest = 30
prob_food_forest = 0.10
prob_food_cropland = 0.30
prob_water = 0.01
thermoregulation_threshold = 28
year = 2010
month = "Jan"
number_processes = 8
iterations = 8

model_init_files = os.path.join(os.getcwd(), "mesageo_elephant_project/elephant_project/data/model_init_files_humans_and_bulls/model_init_values/init_files")

GA_params_init_file = open(os.path.join(model_init_files,"GA_tuned_params.json"))
GA_params_init_data = json.load(GA_params_init_file)

population_init_file = open(os.path.join(model_init_files,"population_init.json"))
population_init_data = json.load(population_init_file)

fixed_params = {
    "num_bull_elephants": 1, 
    "area_size": 1100,              
    "spatial_resolution": 30, 
    "temporal_resolution": 5,
    "prob_drink_water_dry": 0.3333,   
    "prob_drink_water_wet": 0.0000, 
    "percent_memory_elephant": GA_params_init_data["percent_memory_elephant"],   
    "radius_food_search": GA_params_init_data["radius_food_search"],     
    "radius_water_search": GA_params_init_data["radius_water_search"], 
    "movement_fitness_depreceation": -0.000347,        
    "fitness_increment_when_eats_food": 0.05,      
    "fitness_increment_when_drinks_water_dry": 0.05,    
    "fitness_increment_when_drinks_water_wet": 0.005,    
    "fitness_threshold": GA_params_init_data["fitness_threshold"],   
    "terrain_radius": GA_params_init_data["terrain_radius"],       
    "tolerance": GA_params_init_data["tolerance"],   
    "knowledge_from_fringe": 1500,   
    "prob_crop_damage": 0.05,           
    "prob_infrastructure_damage": 0.01
    }





def run_model(year, month, mon_len, prob_food_forest, prob_food_cropland, prob_water, food_val_forest, food_val_cropland, thermoregulation_threshold):

    variable_params =  { 
        "year": year,
        "month": month,    
        "max_time": 100,                
        "prob_food_forest": prob_food_forest,  
        "prob_food_cropland": prob_food_cropland,  
        "prob_water_sources": prob_water,  
        "max_food_val_forest": food_val_forest,
        "max_food_val_cropland": food_val_cropland,
        "thermoregulation_threshold": thermoregulation_threshold,
        }

    model_params = fixed_params
    model_params.update(variable_params)

    expt_name = "mitigation_measures"
    elephant_category = "solitary_bulls"
    
    output_folder = os.path.join(os.getcwd(), "model_runs/", expt_name, elephant_category, month)

    #delete the folder if it already exists
    if os.path.exists(output_folder):
        os.system("rm -r "+output_folder)
    
    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    
    batch_run_model(model_params, number_processes, iterations, output_folder)

    return


def run_abm():

    month_idx = list(calendar.month_abbr).index(month)
    num_days = calendar.monthrange(year, month_idx)[1]

    p1 = multiprocessing.Process(target=run_model, 
                                 args=(year, 
                                       month, 
                                       num_days, 
                                       prob_food_forest, 
                                       prob_food_cropland, 
                                       prob_water, 
                                       food_val_forest, 
                                       food_val_cropland, 
                                       thermoregulation_threshold))

    p1.start()
    p1.join()

run_abm()