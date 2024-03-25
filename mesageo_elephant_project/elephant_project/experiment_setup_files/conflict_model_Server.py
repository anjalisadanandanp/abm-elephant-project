#importing model class
from experiment_setup_files.conflict_model import Conflict_model

#importing visualization tools 
from mesa_geo.visualization.ModularVisualization import ModularServer

#from mesa_geo.visualization.MapModule import MapModule
from experiment_setup_files.Mesa_MapVisualization import MapModule

from mesa.visualization.modules import ChartModule, TextElement

#importing agent classes
from experiment_setup_files.conflict_model_Elephant_agent import Elephant    

from experiment_setup_files.initilialize_food_water_matrix_server_run import environment

import os
from osgeo import gdal
import json
import shutil

init_file = open(os.path.join("experiment_setup_files", "init_files","model_thresholds.json"))
init_data = json.load(init_file)

model_params =  {     
        "temporal_scale": 5,
        "num_bull_elephants": 1,    
        "max_time": 2880,              
        "area_size": 1100,              
        "resolution": 30,    
        "prob_food_cropland": 0.2,      
        "prob_food_forest": 0.2,  
        "prob_water": 0.0,  
        "prob_drink_water": 0.001,   
        "percent_memory_elephant": 0.3,   
        "radius_food_search": 480,     
        "radius_water_search": 480, 
        "movement_fitness_depreceation": -0.000576,        
        "fitness_increment_when_eats_food": 0.01,      
        "fitness_increment_when_drinks_water": 0.05,     
        "fitness_threshold": 0.8,   
        "terrain_radius": 480,  
        "discount": 0.92,     
        "tolerance": 182,   
        "AW_onward_time_start": 5,      
        "AW_onward_time_end": 10,       
        "AW_return_time_start": 16,    
        "AW_return_time_end": 21,      
        "OW_onward_time_start": 7,      
        "OW_onward_time_end": 11,      
        "OW_return_time_start": 16,  
        "OW_return_time_end": 21,    
        "RW_onward_time_start": 5,      
        "RW_onward_time_end": 10,     
        "RW_return_time_start": 16,    
        "RW_return_time_end": 21,      
        "speed_walking_start": 0.5555,   
        "speed_walking_end": 1.1111,        
        "speed_vehicle_start": 4.16667, 
        "speed_vehicle_end": 25,  
        "knowledge_from_fringe": 1000,   
        "human_agent_visibility": 500,      
        "elephant_agent_visibility": 500, 
        "prob_crop_damage": 0.01,           
        "prob_infrastructure_damage": 0.01, 
        "fitness_fn_decrement_humans": 0.05,     
        "fitness_fn_decrement_elephants": 0.05,   
        "escape_radius_humans": 200,     
        "radius_forest_search": 1000,   
        "action_probability": 0.5, 
        "aggress_threshold_inflict_damage": init_data["aggress_threshold_inflict_damage"],                 
        "aggress_threshold_enter_cropland": init_data["aggress_threshold_enter_cropland"],           
        "food_habituation_threshold": init_data["food_habituation_threshold"],              
        "human_habituation_tolerance": init_data["human_habituation_tolerance"],                    
        "elephant_habituation_tolerance": init_data["elephant_habituation_tolerance"],  
        "disturbance_tolerance": init_data["disturbance_tolerance"]
        }


#OVERWRITING MODEL CLASS STEP FUNCTION
#######################################################################################################################
class Conflict_model(Conflict_model):

    #-----------------------------------------------------------------------------------------------------
    def FOOD_MATRIX(self, prob_food_in_forest, prob_food_in_cropland):
        """ Returns the food matrix model of the study area"""

        #fid = os.path.join("Environment","Raster_Files_Seethathode_Derived",self.area[self.area_size], self.reso[self.resolution],"Food_matrix_"+ str(Prob_food_in_forest) +"_.tif")
        fid = os.path.join("experiment_setup_files","simulation_results_server_run", "food_and_water_matrix","food_matrix_"+ str(prob_food_in_forest) + "_" + str(prob_food_in_cropland) + "_.tif")

        FOOD = gdal.Open(fid).ReadAsArray()  
        return FOOD.tolist()  
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
    def WATER_MATRIX(self, prob_water):
        """ Returns the water matrix model of the study area"""

        fid = os.path.join("experiment_setup_files","simulation_results_server_run", "food_and_water_matrix","water_matrix_"+ str(prob_water) +"_.tif")

        WATER = gdal.Open(fid).ReadAsArray()  
        return WATER.tolist() 
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------

path_to_folder = os.path.join("experiment_setup_files","simulation_results_server_run")
if os.path.isdir(path_to_folder):
    shutil.rmtree(path_to_folder)
os.mkdir(path_to_folder)

path_to_folder = os.path.join("experiment_setup_files","simulation_results_server_run", "food_and_water_matrix")
if os.path.isdir(path_to_folder):
    shutil.rmtree(path_to_folder)
os.mkdir(path_to_folder)

env = environment(Prob_food_in_forest = model_params["prob_food_forest"],
                    Prob_food_in_cropland = model_params["prob_food_cropland"],
                    Prob_water = model_params["prob_water"], area_size = 1100, resolution = 30)
env.main()

#Agent visualization
def Agent_Visualization(agent):

    if agent is None:
        return

    portrayal_agent = {}

    if type(agent) is Elephant:
        portrayal_agent ["layer"] = 1
        portrayal_agent["fitness"] = round(agent.fitness , 2)
        portrayal_agent["aggression"] = round(agent.aggress_factor , 2)

    else:
        portrayal_agent ["layer"] = 1
        portrayal_agent["fitness"] = round(agent.fitness , 2)

    return portrayal_agent




#Text visualization
class Model_time_display(TextElement):
    """ 
    Display the model time
    """
    def __init__(self):
        pass

    def render(self, model):
        return "Time elapsed in Hours: " + str(model.model_hour) 


#Text visualization
class day_or_night_display(TextElement):
    """
    Display the model time
    """
    def __init__(self):
        pass

    def render(self, model):

        if model.hour_in_day >= 6 and model.hour_in_day < 18:
            return "Day" 
        else:
            return "Night" 


class display_elephant_agent_mode(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        for a in model.schedule.agents:
            if "bull" in a.unique_id:
                return "MODE:" + str(a.mode) + "   FITNESS:" + str(round(a.fitness, 4)) + "   AGGRESSION:" + str(round(a.aggress_factor, 4))


chart1 = ChartModule(
    [{"Label": "elephant_deaths", "Color": "Black"}], data_collector_name="datacollector"
)
chart2 = ChartModule(
    [{"Label": "human_deaths", "Color": "Black"}], data_collector_name="datacollector"
)
chart3 = ChartModule(
    [{"Label": "disturbance", "Color": "Black"}], data_collector_name="datacollector"
)



model_time_display = Model_time_display()
display_elephant_mode = display_elephant_agent_mode()
day_night_display = day_or_night_display()

map=MapModule(Agent_Visualization,Conflict_model.MAP_COORDS,12,600,600)

server = ModularServer(
    Conflict_model,[map, model_time_display, day_night_display, display_elephant_mode, chart1, chart2, chart3],"Forest model with elephant agents", model_params
)

server.launch()