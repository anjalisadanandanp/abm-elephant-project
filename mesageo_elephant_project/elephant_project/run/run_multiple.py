import os

import sys
sys.path.append(os.getcwd())

import json
import pathlib
import calendar
import multiprocessing

from mesageo_elephant_project.elephant_project.model.abm_model_HEC_EBF_and_MF_as_refuge import batch_run_model

food_val_cropland = 100
prob_food_forest = 0.05
prob_food_cropland = 0.30
prob_water = 0.01
threshold = 28
aggression_values = [0.2]
year = 2010
number_processes = 3
iterations = 3

#model init files
model = "01"
model_init = os.path.join(os.getcwd(), "mesageo_elephant_project/elephant_project/data", "model_init_files_humans_and_bulls", "model_run_" + model)

init_file = open(os.path.join(model_init, "init_files","model_thresholds.json"))
init_data = json.load(init_file)

population_init_file = open(os.path.join(model_init, "init_files","population_init.json"))
population_init_data = json.load(population_init_file)

GA_params_init_file = open(os.path.join(model_init, "init_files","GA_tuned_params.json"))
GA_params_init_data = json.load(GA_params_init_file)

fixed_params = {
    "temporal_scale": 5,
    "num_bull_elephants": 1, 
    "num_herds": 0,  
    "area_size": 1100,              
    "resolution": 30, 
    "prob_drink_water_dry": GA_params_init_data["prob_drink_water_dry"],   
    "prob_drink_water_wet": GA_params_init_data["prob_drink_water_wet"], 
    "percent_memory_elephant": GA_params_init_data["percent_memory_elephant"],   
    "radius_food_search": GA_params_init_data["radius_food_search"],     
    "radius_water_search": GA_params_init_data["radius_water_search"], 
    "movement_fitness_depreceation": -0.000576,        
    "fitness_increment_when_eats_food": 0.05,      
    "fitness_increment_when_drinks_water_dry": 0,    
    "fitness_increment_when_drinks_water_wet": 0,    
    "fitness_threshold": GA_params_init_data["fitness_threshold"],   
    "terrain_radius": GA_params_init_data["terrain_radius"],  
    "discount": GA_params_init_data["discount"],     
    "tolerance": GA_params_init_data["tolerance"],   
    "AW_onward_time_start": population_init_data["AW_onward_time_start"],      
    "AW_onward_time_end": population_init_data["AW_onward_time_end"],       
    "AW_return_time_start": population_init_data["AW_return_time_start"],    
    "AW_return_time_end": population_init_data["AW_return_time_end"],      
    "OW_onward_time_start": population_init_data["OW_onward_time_start"],      
    "OW_onward_time_end": population_init_data["OW_onward_time_end"],      
    "OW_return_time_start": population_init_data["OW_return_time_start"],  
    "OW_return_time_end": population_init_data["OW_return_time_end"],    
    "RW_onward_time_start": population_init_data["RW_onward_time_start"],      
    "RW_onward_time_end": population_init_data["RW_onward_time_end"],     
    "RW_return_time_start": population_init_data["RW_return_time_start"],    
    "RW_return_time_end": population_init_data["RW_return_time_end"],      
    "speed_walking_start": population_init_data["speed_walking_start"],   
    "speed_walking_end": population_init_data["speed_walking_end"],        
    "speed_vehicle_start": population_init_data["speed_vehicle_start"], 
    "speed_vehicle_end": population_init_data["speed_vehicle_end"],  
    "knowledge_from_fringe": 1500,   
    "human_agent_visibility": 250,      
    "elephant_agent_visibility": 500, 
    "prob_crop_damage": 0.05,           
    "prob_infrastructure_damage": 0, 
    "fitness_fn_decrement_humans": 0.1,     
    "fitness_fn_decrement_elephants": 0.05,   
    "aggression_fn_decrement_elephants": 0, 
    "aggression_fn_increment_elephants": 0,
    "escape_radius_humans": 500,     
    "radius_forest_search": 1000,   
    "action_probability": 0.5, 
    "aggress_threshold_inflict_damage": init_data["aggress_threshold_inflict_damage"],                 
    "aggress_threshold_enter_cropland": init_data["aggress_threshold_enter_cropland"],           
    "food_habituation_threshold": init_data["food_habituation_threshold"],              
    "human_habituation_tolerance": init_data["human_habituation_tolerance"],                    
    "elephant_habituation_tolerance": init_data["elephant_habituation_tolerance"],  
    "disturbance_tolerance": init_data["disturbance_tolerance"],
    "num_guard_agents": 0,
    }





def run_model_01(year, month, mon_len, prob_food_forest, prob_food_cropland, prob_water, food_val_forest, food_val_cropland, threshold, aggression, model):

    variable_params =  { 
        "year": year,
        "month": month,    
        "max_time": mon_len*288,                
        "prob_food_forest": prob_food_forest,  
        "prob_food_cropland": prob_food_cropland,  
        "prob_water": prob_water,  
        "max_food_val_forest": 5,
        "max_food_val_cropland": food_val_cropland,
        "THRESHOLD": threshold,
        "elephant_aggression": aggression
        }

    model_params = variable_params
    model_params.update(fixed_params)

    expt_name = "_01_aggression_and_crop_raiding_incidents"
    output = "prob_water__" + str(prob_water) + "__output_files"
    food_val = "food_value_forest__" + str(food_val_forest) + "__food_value_cropland__" + str(food_val_cropland)
    temp_threshold = "THRESHOLD_" + str(model_params["THRESHOLD"])
    elephant_category = "solitary_bulls"
    expt_id = "aggression:" + str(model_params["elephant_aggression"])
    
    output_folder = os.path.join( os.getcwd(), "mesageo_elephant_project/elephant_project/outputs/batch_run/", expt_name, output, food_val, temp_threshold, elephant_category, expt_id, "model_" + model, month)
    
    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    
    batch_run_model(model_init, model_params, number_processes, iterations, output_folder, model)

    return


def run_model_02(year, month, mon_len, prob_food_forest, prob_food_cropland, prob_water, food_val_forest, food_val_cropland, threshold, aggression, model):

    variable_params =  { 
        "year": year,
        "month": month,    
        "max_time": mon_len*288,                
        "prob_food_forest": prob_food_forest,  
        "prob_food_cropland": prob_food_cropland,  
        "prob_water": prob_water,  
        "max_food_val_forest": 10,
        "max_food_val_cropland": food_val_cropland,
        "THRESHOLD": threshold,
        "elephant_aggression": aggression
        }

    model_params = variable_params
    model_params.update(fixed_params)

    expt_name = "_01_aggression_and_crop_raiding_incidents"
    output = "prob_water__" + str(prob_water) + "__output_files"
    food_val = "food_value_forest__" + str(food_val_forest) + "__food_value_cropland__" + str(food_val_cropland)
    temp_threshold = "THRESHOLD_" + str(model_params["THRESHOLD"])
    elephant_category = "solitary_bulls"
    expt_id = "aggression:" + str(model_params["elephant_aggression"])
    
    output_folder = os.path.join( os.getcwd(), "mesageo_elephant_project/elephant_project/outputs/batch_run/", expt_name, output, food_val, temp_threshold, elephant_category, expt_id, "model_" + model, month)
    
    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    
    batch_run_model(model_init, model_params, number_processes, iterations, output_folder, model)

    return
                

def run_model_03(year, month, mon_len, prob_food_forest, prob_food_cropland, prob_water, food_val_forest, food_val_cropland, threshold, aggression, model):

    variable_params =  { 
        "year": year,
        "month": month,    
        "max_time": mon_len*288,                
        "prob_food_forest": prob_food_forest,  
        "prob_food_cropland": prob_food_cropland,  
        "prob_water": prob_water,  
        "max_food_val_forest": 15,
        "max_food_val_cropland": food_val_cropland,
        "THRESHOLD": threshold,
        "elephant_aggression": aggression
        }

    model_params = variable_params
    model_params.update(fixed_params)

    expt_name = "_01_aggression_and_crop_raiding_incidents"
    output = "prob_water__" + str(prob_water) + "__output_files"
    food_val = "food_value_forest__" + str(food_val_forest) + "__food_value_cropland__" + str(food_val_cropland)
    temp_threshold = "THRESHOLD_" + str(model_params["THRESHOLD"])
    elephant_category = "solitary_bulls"
    expt_id = "aggression:" + str(model_params["elephant_aggression"])
    
    output_folder = os.path.join( os.getcwd(), "mesageo_elephant_project/elephant_project/outputs/batch_run/", expt_name, output, food_val, temp_threshold, elephant_category, expt_id, "model_" + model, month)
    
    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    
    batch_run_model(model_init, model_params, number_processes, iterations, output_folder, model)

    return


def run_model_04(year, month, mon_len, prob_food_forest, prob_food_cropland, prob_water, food_val_forest, food_val_cropland, threshold, aggression, model):

    variable_params =  { 
        "year": year,
        "month": month,    
        "max_time": mon_len*288,                
        "prob_food_forest": prob_food_forest,  
        "prob_food_cropland": prob_food_cropland,  
        "prob_water": prob_water,  
        "max_food_val_forest": 20,
        "max_food_val_cropland": food_val_cropland,
        "THRESHOLD": threshold,
        "elephant_aggression": aggression
        }

    model_params = variable_params
    model_params.update(fixed_params)

    expt_name = "_01_aggression_and_crop_raiding_incidents"
    output = "prob_water__" + str(prob_water) + "__output_files"
    food_val = "food_value_forest__" + str(food_val_forest) + "__food_value_cropland__" + str(food_val_cropland)
    temp_threshold = "THRESHOLD_" + str(model_params["THRESHOLD"])
    elephant_category = "solitary_bulls"
    expt_id = "aggression:" + str(model_params["elephant_aggression"])
    
    output_folder = os.path.join( os.getcwd(), "mesageo_elephant_project/elephant_project/outputs/batch_run/", expt_name, output, food_val, temp_threshold, elephant_category, expt_id, "model_" + model, month)
    
    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    
    batch_run_model(model_init, model_params, number_processes, iterations, output_folder, model)

    return


def run_model_05(year, month, mon_len, prob_food_forest, prob_food_cropland, prob_water, food_val_forest, food_val_cropland, threshold, aggression, model):

    variable_params =  { 
        "year": year,
        "month": month,    
        "max_time": mon_len*288,                
        "prob_food_forest": prob_food_forest,  
        "prob_food_cropland": prob_food_cropland,  
        "prob_water": prob_water,  
        "max_food_val_forest": 25,
        "max_food_val_cropland": food_val_cropland,
        "THRESHOLD": threshold,
        "elephant_aggression": aggression
        }

    model_params = variable_params
    model_params.update(fixed_params)

    expt_name = "_01_aggression_and_crop_raiding_incidents"
    output = "prob_water__" + str(prob_water) + "__output_files"
    food_val = "food_value_forest__" + str(food_val_forest) + "__food_value_cropland__" + str(food_val_cropland)
    temp_threshold = "THRESHOLD_" + str(model_params["THRESHOLD"])
    elephant_category = "solitary_bulls"
    expt_id = "aggression:" + str(model_params["elephant_aggression"])
    
    output_folder = os.path.join(os.getcwd(), "mesageo_elephant_project/elephant_project/outputs/batch_run/", expt_name, output, food_val, temp_threshold, elephant_category, expt_id, "model_" + model, month)
    
    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    
    batch_run_model(model_init, model_params, number_processes, iterations, output_folder, model)

    return
                



def run_models():

    for month_idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        month = calendar.month_abbr[month_idx]
        num_days = calendar.monthrange(year, month_idx)[1]

        if month_idx == 12:
            num_days = 30

        #launch two functions run_model_01 and run_model_02 in parallel
        p1 = multiprocessing.Process(target=run_model_01, args=(year, month, num_days, prob_food_forest, prob_food_cropland, prob_water, 5, food_val_cropland, threshold, aggression, model))
        p2 = multiprocessing.Process(target=run_model_02, args=(year, month, num_days, prob_food_forest, prob_food_cropland, prob_water, 10, food_val_cropland, threshold, aggression, model))
        p3 = multiprocessing.Process(target=run_model_03, args=(year, month, num_days, prob_food_forest, prob_food_cropland, prob_water, 15, food_val_cropland, threshold, aggression, model))
        p4 = multiprocessing.Process(target=run_model_04, args=(year, month, num_days, prob_food_forest, prob_food_cropland, prob_water, 20, food_val_cropland, threshold, aggression, model))
        p5 = multiprocessing.Process(target=run_model_05, args=(year, month, num_days, prob_food_forest, prob_food_cropland, prob_water, 25, food_val_cropland, threshold, aggression, model))

        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()

        print("Finished running: ", "month:", month, "num days:", num_days, "aggression:",  aggression, "food_val_forest:", 5)
        print("Finished running: ", "month:", month, "num days:", num_days, "aggression:",  aggression, "food_val_forest:", 10)
        print("Finished running: ", "month:", month, "num days:", num_days, "aggression:",  aggression, "food_val_forest:", 15)
        print("Finished running: ", "month:", month, "num days:", num_days, "aggression:",  aggression, "food_val_forest:", 20)
        print("Finished running: ", "month:", month, "num days:", num_days, "aggression:",  aggression, "food_val_forest:", 25)

for aggression in aggression_values:
    run_models()