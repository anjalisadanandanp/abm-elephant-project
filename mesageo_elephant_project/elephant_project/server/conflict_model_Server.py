import os
import pathlib

import sys
sys.path.append(os.getcwd())

# PRINTING CURRENT WORKING DIRECTORY
# print(os.getcwd())

# importing the server run model function
from mesageo_elephant_project.elephant_project.model.abm_model_HEC import server_run_model

#importing model class
from mesageo_elephant_project.elephant_project.model.abm_model_HEC import Conflict_model

#importing visualization tools 
from mesa_geo.visualization.ModularVisualization import ModularServer

# from mesa_geo.visualization.MapModule import MapModule        #This is the original import statement
from mesageo_elephant_project.elephant_project.experiment_setup_files.Mesa_MapVisualization import MapModule


from mesa.visualization.modules import TextElement

#importing agent classes
from mesageo_elephant_project.elephant_project.model.abm_model_HEC import Elephant    

from mesageo_elephant_project.elephant_project.model.abm_model_HEC import landuse_classes


import os
import json
import calendar


#model init files
model = "01"
model_init = os.path.join(os.getcwd(), "mesageo_elephant_project/elephant_project/data/", "model_init_files_humans_and_bulls", "model_run_" + model)

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


user_init = {
    "SIMULATION_MONTH": "Jan",
    "SIMULATION_YEAR": 2010,
    "PROB_FOOD_FOREST": 0.05,
    "PROB_FOOD_CROPLAND": 0.30,
    "PROB_WATER": 0.05,
    "FOOD_VAL_CROPLAND": 100,
    "FOOD_VAL_FOREST": 25,
    "THRESHOLD": 28,
}

#use dictionary to map month to month index
month_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
month_idx = month_dict[user_init["SIMULATION_MONTH"]]

month = calendar.month_abbr[month_idx]
num_days = calendar.monthrange(user_init["SIMULATION_YEAR"], month_idx)[1]

print("Month: ", month, "Number of simulation days: ", num_days)

variable_params =  { 
    "year": user_init["SIMULATION_YEAR"],
    "month": month,    
    "max_time": num_days*288,                #each day has 288 time steps
    "prob_food_forest": user_init["PROB_FOOD_FOREST"],  
    "prob_food_cropland": user_init["PROB_FOOD_CROPLAND"],  
    "prob_water": user_init["PROB_WATER"],  
    "max_food_val_forest": user_init["FOOD_VAL_FOREST"],
    "max_food_val_cropland": user_init["FOOD_VAL_CROPLAND"],
    "THRESHOLD": user_init["THRESHOLD"],
    "elephant_aggression": 0.8
    }

model_params = variable_params
model_params.update(fixed_params)

expt_name = "_01_aggression_and_crop_raiding_incidents"
output = "prob_water__" + str(user_init["PROB_WATER"]) + "__output_files"
food_val = "food_value_forest__" + str(variable_params["max_food_val_forest"]) + "__food_value_cropland__" + str(user_init["FOOD_VAL_CROPLAND"])
temp_threshold = "THRESHOLD_" + str(model_params["THRESHOLD"])
elephant_category = "solitary_bulls"
expt_id = "aggression:" + str(model_params["elephant_aggression"])

output_folder = os.path.join( os.getcwd(), "mesageo_elephant_project/elephant_project/outputs/server_run/", expt_name, output, food_val, temp_threshold, elephant_category, expt_id, "model_" + model, month)

path = pathlib.Path(output_folder)
path.mkdir(parents=True, exist_ok=True)

server_run_model(model_init, model_params, output_folder, model)

#-------agent visualisation-------------------------#
def agent_visualization(agent):

    if agent is None:
        return

    agent_portrayal = {}

    if isinstance(agent, Elephant):
        agent_portrayal ["layer"] = 2

    else:
        agent_portrayal ["layer"] = 1

    return agent_portrayal
#------------------------------------------------------#


#-------TEXT ELEMENTS FOR VISUALIZATION----------------#
class Model_time_display(TextElement):
    """ 
    Display the model time
    """
    def __init__(self):
        pass

    def render(self, model):
        return "Time elapsed in Hours: " + str(model.model_hour) + str("  ") + "Day: " + str(model.model_day)
#------------------------------------------------------#


#-------TEXT ELEMENTS FOR VISUALIZATION----------------#
class day_or_night_display(TextElement):
    """
    Display the model time
    """
    def __init__(self):
        pass

    def render(self, model):

        if model.hour_in_day >= 6 and model.hour_in_day < 18:
            return "Day, " + "HOUR: " + str(model.hour_in_day) + ":00"
        else:
            return "Night, " + "HOUR: " + str(model.hour_in_day) + ":00"
#------------------------------------------------------#


#-------TEXT ELEMENTS FOR VISUALIZATION----------------#
class display_elephant_agent_mode(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        for a in model.schedule.agents:
            if "bull" in a.unique_id:
                return "MODE:" + str(a.mode) + ", FITNESS:" + str(round(a.fitness, 4)) + ", AGGRESSION:" + str(round(a.aggress_factor, 4))
#------------------------------------------------------#

model_time_display = Model_time_display()
display_elephant_mode = display_elephant_agent_mode()
day_night_display = day_or_night_display()

map = MapModule(agent_visualization, Conflict_model.MAP_COORDS, 12, 600, 600)

server = ModularServer(
    Conflict_model,[map, model_time_display, day_night_display, display_elephant_mode], "Simulation model for Human-Elephant Conflict", variable_params
)

server.launch()