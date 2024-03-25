#importing model class
from experiment_setup_files.conflict_model import Conflict_model

#importing agent classes
from experiment_setup_files.conflict_model_Elephant_agent import Elephant    
from experiment_setup_files.conflict_model_Human_agent import Cultivators_Agricultural_labourers, HomeBound
from experiment_setup_files.conflict_model_Human_agent import Other_workers
from experiment_setup_files.conflict_model_Human_agent import RandomWalkers_homebound

from experiment_setup_files.initilialize_food_water_matrix_batch_run import environment

from multiprocessing import freeze_support
import pandas as pd
import os
import shutil
import numpy as np
import rasterio as rio
from osgeo import gdal
import json

import uuid

from experiment_setup_files.Mesa_BatchRunner_class_v1_2 import batch_run


#OVERWRITING MODEL CLASS STEP FUNCTION
#######################################################################################################################
class Conflict_model(Conflict_model):
    #-----------------------------------------------------------------------------------------------------
    def FOOD_MATRIX(self, prob_food_in_forest, prob_food_in_cropland):
        """ Returns the food matrix model of the study area"""
        fid = os.path.join("experiment_setup_files","simulation_results_batch_run", "food_and_water_matrix","food_matrix_"+ str(prob_food_in_forest) + "_" + str(prob_food_in_cropland) + "_.tif")
        FOOD = gdal.Open(fid).ReadAsArray()  
        return FOOD.tolist()  
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
    def WATER_MATRIX(self, prob_water):
        """ Returns the water matrix model of the study area"""
        fid = os.path.join("experiment_setup_files","simulation_results_batch_run", "food_and_water_matrix","water_matrix_"+ str(prob_water) +"_.tif")
        WATER = gdal.Open(fid).ReadAsArray()  
        return WATER.tolist() 
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
    def step(self):

        self.update_human_disturbance_explict()

        #SIMULATION OF ONE TICK
        self.schedule.step()
        self.running = 'True'
        self.grid._recreate_rtree()

        #REMOVING DEAD AGENTS
        for x in self.agents_to_remove:
            self.schedule.remove(x)     #REMOVING FROM THE SCHEDULE
            self.grid.remove_agent(x)   #REMOVING FROM THE GRID
            self.dead_agents.add(x)
        self.agents_to_remove = set()

        self.num_elephant_deaths = 0 
        self.num_human_deaths = 0

        for agent in self.dead_agents:
            if "bull" in agent.unique_id:
                self.num_elephant_deaths += 1

            elif "Human" in agent.unique_id:
                self.num_human_deaths += 1

        #COLLECT DATA 
        self.datacollector.collect(self)

        #UPDATE TIME
        self.model_time = self.model_time + 1      
        self.model_minutes = self.model_time * 5
        self.model_hour = int(self.model_minutes/60)
        self.model_day = int(self.model_hour/24)
        self.hour_in_day =  self.model_hour - self.model_day*24

        #TERMINATE IF NO AGENTS ARE REMAINING OR MAX STEPS HAS REACHED
        if len(self.schedule.agents) == 0 or self.model_time == self.max_time:
            self.running = False

       #TERMINATE THE SIMULATION AND WRITE RESULTS TO FILE
        if self.running == False:

            now = uuid.uuid1()

            source = os.path.join(self.folder_root, "LULC.tif")
            with rio.open(source) as src:
                ras_meta = src.profile

            loc = os.path.join("experiment_setup_files","simulation_results_batch_run", "infrastructure_damage__" + str(now) + "__" + ".tif")
            with rio.open(loc, 'w', **ras_meta) as dst:
                dst.write(np.array(self.infrastructure_damage).reshape(self.row_size, self.col_size).astype('int'), 1)

            loc = os.path.join("experiment_setup_files","simulation_results_batch_run", "crop_damage__" + str(now) + "__" + ".tif")
            with rio.open(loc, 'w', **ras_meta) as dst:
                dst.write(np.array(self.crop_damage).reshape(self.row_size, self.col_size).astype('int'), 1)
    #---------------------------------------------------------------------------------------------------------
#######################################################################################################################





if __name__ == '__main__':

    freeze_support()

    init_file = open(os.path.join("experiment_setup_files","init_files","model_thresholds.json"))
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

    path_to_folder = os.path.join("experiment_setup_files","simulation_results_batch_run")

    if os.path.isdir(path_to_folder):
        shutil.rmtree(path_to_folder)
    os.mkdir(path_to_folder)

    path_to_folder = os.path.join("experiment_setup_files","simulation_results_batch_run", "food_and_water_matrix")

    if os.path.isdir(path_to_folder):
        shutil.rmtree(path_to_folder)
    os.mkdir(path_to_folder)

    env = environment(Prob_food_in_forest = model_params["prob_food_forest"],
                        Prob_food_in_cropland = model_params["prob_food_cropland"],
                        Prob_water = model_params["prob_water"], area_size = 1100, resolution = 30)
    env.main()

    code_runner = batch_run(Conflict_model, parameters = model_params, number_processes = 10, iterations = 10,
                        max_steps = 8640, data_collection_period=1, display_progress=True)

    results_df = pd.DataFrame(code_runner)
    results_df.to_csv(os.path.join("experiment_setup_files","simulation_results_batch_run","Batch_run.csv"))