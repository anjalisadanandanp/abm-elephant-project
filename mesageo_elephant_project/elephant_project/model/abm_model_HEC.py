
#---------------imports-------------------#
import os                               # for file operations
import sys                              # for system operations 
import shutil                           # to copy files
import json                             # to read json files
import math                             # for mathematical operations
import numpy as np                      # for numerical operations
import pandas as pd                     # for data manipulation and analysis
import pickle                           # to save and load pickle objects
from osgeo import gdal                  # for raster operations
import rasterio as rio                  # for raster operations
import osmnx as ox                      # for street network analysis
import networkx as nx                   # for network analysis  
from shapely import geometry, ops       # for manipulation georeferenced data 
from shapely.geometry import Point      # for creating point geometries
from multiprocessing import freeze_support      # for multiprocessing
from math import radians, sin, cos, acos        # for mathematical operations
from pyproj import Proj, transform              # for coordinate transformations
import matplotlib.pyplot as plt                 # for plotting
from mpl_toolkits.basemap import Basemap        # for plotting
from matplotlib import colors                   # for plotting
import matplotlib.cm as cm                      # for plotting
import matplotlib.colors as mcolors             # for plotting
from scipy.ndimage.interpolation import map_coordinates     # for interpolation
import calendar                         # for date and time operations
import datetime                         # for date and time operations
import warnings                         # to ignore warnings
import geopandas as gpd                 # for geospatial operations
import uuid                             # for generating unique ids
#---------------imports-------------------#


warnings.filterwarnings('ignore')       # to ignore warnings
sys.path.append(os.getcwd())            # add the current working directory to the system path



#----------------model imports-------------------#
#MODEL CLASS
from mesageo_elephant_project.elephant_project.experiment_setup_files.Mesa_Model_class import Model      #modified implementation (with bug fix)

#AGENT CLASS
from mesa_geo.geoagent import GeoAgent

#AGENT SCHEDULES
from mesageo_elephant_project.elephant_project.experiment_setup_files.Mesa_agent_scheduling import RandomActivation    #Random activation

#GEOSPACE
from mesageo_elephant_project.elephant_project.experiment_setup_files.Mesa_geospace import GeoSpace          #modified implementation

#AGENT_CREATOR
from mesa_geo.geoagent import AgentCreator          #Default implementation


#DATA_COLLECTOR
#from mesa.datacollection import DataCollector           #Default implementation 
from mesageo_elephant_project.elephant_project.experiment_setup_files.datacollector_codes.Mesa_Datacollector_v1_0 import DataCollector       #all agent data collected
#from experiment_setup_files.Mesa_Datacollector_v1_1 import DataCollector        #ONLY ELEPHANT AGENT DATA COLLECTED

from mesageo_elephant_project.elephant_project.experiment_setup_files.batch_runner_codes.Mesa_BatchRunner_class_v1_1 import batch_run      #runnning multiple simulations
#-------------------------------------------------#






#--------------------------------------------------------------------------------------------------------------------------------
class Elephant(GeoAgent):
    """Class to define the ELEPHANT agent model"""

    def __init__(self,unique_id,model,shape):
        super().__init__(unique_id,model,shape) 


        #agent initialisation
        init_file = open(os.path.join(input_folder, "init_files", "elephant_init.json"))
        init_data = json.load(init_file)


        #INITIALIZE BEHAVIOURAL PARAMETERS: FITNESS AND AGGRESSION
        #-------------------------------------------------------------------
        self.fitness = init_data["fitness_init"]    #fitness value of the individual at the start of the simulation
        self.aggress_factor = self.model.elephant_aggression   #aggression value of the individual at the start of the simulation
        self.food_habituation = init_data["food_habituation_init"]     #how much the elephant is used to the agricultural crops
        self.human_habituation = init_data["human_habituation_init"]      #how much the elephant agent is used to the presence of humans
        #-------------------------------------------------------------------



        #-------------------------------------------------------------------
        #INITIALIZE FOOD AND WATER MEMORY MATRICES
        #-------------------------------------------------------------------
        #Initialize the food and water memory matrices of the elephant agents depending on the model parameter "percent_memory_elephant"
        self.Percent_memory_elephant = self.model.percent_memory_elephant
        self.knowledge_from_fringe = self.model.knowledge_from_fringe         #how much the agent knows about the food availability at the fringe
        
        #Sub-models:
        #food_memory, water_memory = self.initialize_memory_matrix()
        #food_memory, water_memory = self.initialize_memory_matrix_only_forest()
        food_memory, water_memory = self.initialize_memory_matrix_with_knowledge_from_fringe()

        #To have Json serializable agent attributes, we convert the memory matrices to list
        self.food_memory = food_memory.tolist()         
        self.water_memory = water_memory.tolist()
        #-------------------------------------------------------------------
    


        #-------------------------------------------------------------------
        #OTHER VARIABLES
        #-------------------------------------------------------------------
        self.mode = self.model.random.choice(["RandomWalk", "TargetedWalk"])  
        self.food_consumed = 0                      
        self.num_days_water_source_visit = 0        
        self.num_days_food_depreceation = 0        
        self.visit_water_source = True            
        self.heading = self.model.random.uniform(0,360)
        self.ROW, self.COL = self.update_grid_index()  
        self.target_present = False                
        self.target_lat = None                     
        self.target_lon = None                      
        self.distance_to_water_source = None
        self.target_name = None
        self.radius_food_search = self.model.radius_food_search
        self.radius_water_search = self.model.radius_water_search
        self.slope = self.model.SLOPE[self.ROW][self.COL]

        self.radius_forest_search =  self.model.radius_forest_search    
        self.danger_to_life = False
        self.inconflictwith = []    

        self.used_firecracker = False

        self.prob_thermoregulation = self.model.temp[self.ROW,self.COL]
        self.landuse = self.model.LANDUSE[self.ROW][self.COL]
        self.hour = self.model.hour_in_day
        self.water_source_proximity = self.model.water_proximity[self.ROW][self.COL]

        self.num_thermoregulation_steps = 0
        self.num_steps_thermoregulated = 0
    #-------------------------------------------------------------------
    def move_point(self,xnew,ynew): 
        """
        Function to move the agent in the geo-space"""    
        #Function to move the agent in the geo-space 
        #coordinates are in EPSG:3857
        #xnew: longitude
        #ynew: latitude
        return Point(xnew,ynew)
    #-----------------------------------------------------------------------------------------------------
    def initialize_memory_matrix(self):
        """ Function that assigns memory matrix to elephants"""

        #food_memory, water_memory
        food_memory=np.zeros_like(self.model.FOOD)
        water_memory=np.zeros_like(self.model.WATER)

        for i in range(0,self.model.row_size):
            for j in range(0,self.model.col_size):
                if self.model.random.uniform(0,1) < self.Percent_memory_elephant:
                    food_memory[i,j] = self.model.FOOD[i][j]
                    water_memory[i,j] = self.model.WATER[i][j]

        # saving the memory matrix as .tif file 
        source = os.path.join(self.model.folder_root, "LULC.tif")
        with rio.open(source) as src:
            ras_meta = src.profile
        memory_loc = os.path.join(self.model.folder_root, "food_memory_" + str(self.unique_id) + ".tif")
                    
        with rio.open(memory_loc, 'w', **ras_meta) as dst:
            dst.write(food_memory.astype('float32'), 1)
        memory_loc = os.path.join(self.model.folder_root, "water_memory_" + str(self.unique_id) + ".tif")
        with rio.open(memory_loc, 'w', **ras_meta) as dst:
            dst.write(water_memory.astype('float32'), 1)

        return food_memory, water_memory
    #-----------------------------------------------------------------------------------------------------
    def initialize_memory_matrix_only_forest(self):
        """ Function that assigns memory matrix to elephants. The elephant agent has only knowledge of the forests."""

        #food_memory, water_memory
        food_memory=np.zeros_like(self.model.FOOD)
        water_memory=np.zeros_like(self.model.WATER)

        for i in range(0,self.model.row_size):
            for j in range(0,self.model.col_size):
                if self.model.random.uniform(0,1) < self.Percent_memory_elephant and self.model.plantation_proximity[i][j] > 5:
                    food_memory[i,j] = self.model.FOOD[i][j]
                    water_memory[i,j] = self.model.WATER[i][j]

        #saving the memory matrix as .tif file 
                    
        source = os.path.join(self.model.folder_root, "LULC.tif")
        with rio.open(source) as src:
            ras_meta = src.profile
        memory_loc = os.path.join(self.model.folder_root, "food_memory_" + str(self.unique_id) + ".tif")
                    
        with rio.open(memory_loc, 'w', **ras_meta) as dst:
            dst.write(food_memory.astype('float32'), 1)
        memory_loc = os.path.join(self.model.folder_root, "water_memory_" + str(self.unique_id) + ".tif")
        with rio.open(memory_loc, 'w', **ras_meta) as dst:
            dst.write(water_memory.astype('float32'), 1)

        return food_memory, water_memory
    #-----------------------------------------------------------------------------------------------------
    def initialize_memory_matrix_with_knowledge_from_fringe(self):
        """ Function that assigns memory matrix to elephants. The elephant agent has knowledge of the fringe areas."""

        #self.knowlege_from_fringe : unit is in metres
        no_of_cells = self.model.random.randint(0, int(self.knowledge_from_fringe/self.model.xres))     #spatial resolution 
        
        memory=np.zeros_like(self.model.LANDUSE)
        food_memory=np.zeros_like(self.model.LANDUSE)
        water_memory=np.zeros_like(self.model.LANDUSE)

        for i in range(0,self.model.row_size):
            for j in range(0,self.model.col_size):
                if self.model.LANDUSE[i][j]==self.model.landusetypes["Evergreen Broadlead Forest"] or self.model.LANDUSE[i][j]==self.model.landusetypes["Water Bodies"]:
                    memory[i,j]=1
                elif self.model.LANDUSE[i][j]!=self.model.landusetypes["Evergreen Broadlead Forest"]:
                    try:
                        if self.model.LANDUSE[i+no_of_cells][j+no_of_cells]==self.model.landusetypes["Evergreen Broadlead Forest"] or self.model.LANDUSE[i-no_of_cells][j+no_of_cells]==self.model.landusetypes["Evergreen Broadlead Forest"] or self.model.LANDUSE[i-no_of_cells][j-no_of_cells]==self.model.landusetypes["Evergreen Broadlead Forest"] or self.model.LANDUSE[i+no_of_cells][j-no_of_cells]==self.model.landusetypes["Evergreen Broadlead Forest"]:
                            #we are assuming that the elephant agent has some knowledge about the fringe areas
                            memory[i,j]=1
                    except:
                        pass

        for i in range(0,self.model.row_size):
            for j in range(0,self.model.col_size):
                if memory[i,j]==1:
                    if self.model.random.uniform(0,1) < self.Percent_memory_elephant:
                        food_memory[i,j] = self.model.FOOD[i][j]
                        water_memory[i,j] = self.model.WATER[i][j]

        self.memory = memory.tolist()

        #saving the memory matrix as .tif file 
        import rasterio as rio
        source = os.path.join(self.model.folder_root, "LULC.tif")
        with rio.open(source) as src:
            ras_meta = src.profile
        memory_loc = os.path.join(self.model.folder_root, "food_memory_"+ str(self.unique_id) + ".tif")

        with rio.open(memory_loc, 'w', **ras_meta) as dst:
            dst.write(food_memory.astype('float32'), 1)
        memory_loc = os.path.join(self.model.folder_root, "water_memory_"+ str(self.unique_id) + ".tif")
        with rio.open(memory_loc, 'w', **ras_meta) as dst:
            dst.write(water_memory.astype('float32'), 1)

        return food_memory, water_memory
    #-----------------------------------------------------------------------------------------------------
    def distance_calculator_epsg3857(self,slat,elat,slon,elon):  
        """
        Returns the distance between current position and target position"""
        #Returns the distance between current position and target position
        #Input CRS: epsg:3857 

        if slat==elat and slon==elon:
            return 0
            
        dist = np.sqrt((slat-elat)**2+(slon-elon)**2)
        return dist   #returns distance in metres
    #----------------------------------------------------------------------------------------------------- 
    def update_grid_index(self):
        """
        Update the grid index of the agent"""
        # calculate indices and index array
        row, col = self.model.get_indices(self.shape.x, self.shape.y)

        if row == (self.model.row_size - 1) or row == 0 or col == (self.model.col_size - 1) or col == 0:      #if agent is at the border
            lon, lat = self.model.pixel2coord(row, col)
            self.shape = self.move_point(lon, lat)

        return row, col
    #-----------------------------------------------------------------------------------------------------  
    def next_step_to_move(self):
        """how the elephant agent moves from the current co-ordinates to the next"""

        self.mode = self.current_mode_of_the_agent()   

        if self.mode == "RandomWalk":
            next_lon, next_lat = self.correlated_random_walk_without_terrain_factor()   
            row, col = self.model.get_indices(next_lon, next_lat)
            if self.model.LANDUSE[row][col] == 10 and self.aggress_factor < self.model.aggress_threshold_enter_cropland:
                next_lon, next_lat = self.shape.x, self.shape.y     
            self.shape = self.move_point(next_lon, next_lat)

        elif self.mode == "TargetedMove":

            if self.target_present == False:

                row_start, row_end, col_start, col_end = self.return_feasible_direction_to_move()

                rand_num = self.model.random.uniform(0,1)  
                if rand_num <= self.model.prob_drink_water:
                    self.target_to_drink_water(row_start, row_end, col_start, col_end)    
                    self.radius_water_search = self.model.radius_water_search
                    self.target_name = "water"    

                else:
                    self.target_for_foraging(row_start, row_end, col_start, col_end)     
                    self.target_name = "food" 
                    
            row, col = self.model.get_indices(self.target_lon, self.target_lat)
            
            if self.aggress_factor < self.model.aggress_threshold_enter_cropland:       #models:01-16
                #print("aggression is low")
                if self.model.LANDUSE[self.ROW][self.COL] == 10:      #target is a human habitated area, check for the disturbance 
                    if self.model.human_disturbance < self.model.disturbance_tolerance:     #proceed to forage
                        #print("in cropland: ", self.model.hour_in_day)
                        next_lon, next_lat = self.targeted_walk()
                        self.shape = self.move_point(next_lon, next_lat)

                    else:
                        #print("in cropland: ", self.model.hour_in_day)
                        self.target_for_escape()    #move to the forest
                        next_lon, next_lat = self.targeted_walk()
                        self.shape = self.move_point(next_lon, next_lat)

                else:   #target is not a human habitated area, proceed to forage
                    #print("in forest: ", self.model.hour_in_day)
                    next_lon, next_lat = self.targeted_walk()
                    self.shape = self.move_point(next_lon, next_lat)

            else:               #forage irrespective of disturbance     #models:17-32
                next_lon, next_lat = self.targeted_walk()
                self.shape = self.move_point(next_lon, next_lat)

        elif self.mode == "EscapeMode":      #When in conflict with humans and aggression is low
            self.target_for_escape()    #set the target to move to incase there is no target
            next_lon, next_lat = self.targeted_walk()
            self.shape = self.move_point(next_lon, next_lat)

        elif self.mode == "InflictDamage":       #When in conflict with humans and aggression is high
            self.InflictDamage()   

        elif self.mode == "Thermoregulation":
            if self.target_present == False:
                row_start, row_end, col_start, col_end = self.return_feasible_direction_to_move()
                self.target_thermoregulation(row_start, row_end, col_start, col_end)
                self.target_name = "thermoregulation"
            next_lon, next_lat = self.targeted_walk()
            self.shape = self.move_point(next_lon, next_lat)

            # print("proximity to a water source:", self.model.water_proximity[self.ROW][self.COL]*33.33)
            if self.model.water_proximity[self.ROW][self.COL]*33.33 < 250:
                self.num_steps_thermoregulated += 1
   
        #Consume food and water if available
        self.eat_food()
        self.drink_water()

        #crop-infrastructure damage
        self.crop_infrastructure_damage() 

        #Update fitness based on the cost of movement
        self.update_fitness_value(self.model.movement_fitness_depreceation)

        return 
    #--------------------------------------------------------------------------------------------------
    def current_mode_of_the_agent(self):
        """function returns the mode of the agent depending on its energy levels and interaction with human agents"""

        #From the transition probability matrix 
        state1 = 0.8775307
        state2 = 0.9096085
        state1_to_state2 = 0.1224693
        state2_to_state1 = 0.0903915

        ROW, COL = self.model.return_indices_temperature_matrix(self.shape.y, self.shape.x)
        self.prob_thermoregulation = self.model.temp[ROW,COL]

        if self.prob_thermoregulation > 0.5:
            self.num_thermoregulation_steps += 1

        if self.model.elephant_agent_visibility != None:
            for agent in self.model.schedule.agents:
                conflict_neighbors = []
                if isinstance(agent,  Humans):
                    agent.dist_to_bull = agent.distance_calculator_epsg3857(self.shape.y, agent.shape.y, self.shape.x, agent.shape.x)
                    if agent.dist_to_bull <= self.model.elephant_agent_visibility:
                        conflict_neighbors.append(agent)

                if len(conflict_neighbors) > 0:
                    self.conflict_neighbor = conflict_neighbors
                else:
                    self.conflict_neighbor = None

            self.update_danger_to_life()       #update danger_to_life

        if (self.danger_to_life==True and self.aggress_factor < self.model.aggress_threshold_inflict_damage) or (self.danger_to_life==True and self.fitness < self.model.fitness_threshold) or (self.used_firecracker == True):      
            #Supercedes all other requirements --> This state is encountered when in conflict with Human agents
            mode="EscapeMode"

        elif self.danger_to_life==True and self.aggress_factor >= self.model.aggress_threshold_inflict_damage:
            #Supercedes all other requirements --> This state is encountered when in conflict with Human agents
            mode="InflictDamage"

        else:

            if  self.prob_thermoregulation > 0.5:
                mode = "Thermoregulation"

            else:
                num = self.model.random.uniform(0,1)

                if self.fitness < self.model.fitness_threshold:
                    mode = "TargetedMove"

                elif self.mode == "RandomWalk":
                    if num <= state1:
                        mode = "RandomWalk"
                    else:
                        mode = "TargetedMove"
                
                elif self.mode == "TargetedMove":

                    if num <= state2:
                            mode = "TargetedMove"
                    else:
                        mode = "RandomWalk"

                else:
                    mode = self.model.random.choice(["TargetedMove", "RandomWalk"])

        self.used_firecracker = False

        return mode
    #-------------------------------------------------------------------------------------------------
    def update_danger_to_life(self):
        """Update danger_to_life"""
            
        if self.conflict_neighbor != None:     #There are human agents in the viscinity
            if self.human_habituation < self.model.human_habituation_tolerance:     #Not used to human agents
                for agent in self.conflict_neighbor:
                    self.danger_to_life = True
                    self.conflict_with_humans = True 
                    self.model.CONFLICT_LOCATIONS.append([self.unique_id, self.shape.x, self.shape.y, agent.unique_id, agent.shape.x, agent.shape.y])

            else:  #Used to human agents
                for agent in self.conflict_neighbor:
                    self.danger_to_life = False  
                    self.model.INTERACTION_LOCATIONS.append([self.unique_id, self.shape.x, self.shape.y, agent.unique_id, agent.shape.x, agent.shape.y])  

        else:
            self.danger_to_life = False   

        return 
    #-------------------------------------------------------------------------------------------------
    def correlated_random_walk_without_terrain_factor(self):
        """Correlated random walk without terrain factor, used when the agent is in RandomWalk mode"""

        #from fitted HMM model
        mean = -3.0232348
        kappa = 0.3336909
        turning_angle = np.random.vonmises(mean, kappa, 1)  #radians

        movement_direction = self.heading + turning_angle*57.2958       #57.2958 -> to convert radians to degrees

        #from fitted HMM model
        mean = 0.004031308
        std_dv = 0.003406922  

        shape = (mean/std_dv)**2
        rate =  mean/std_dv**2
        step_length = np.random.gamma(shape, (1/rate), 1)       #unit is kilometres

        theta = movement_direction * 0.0174533      #0.0174533 -> To convert degrees to radians 
        dx = step_length * math.cos(theta) * 1000      #factor 1000: to convert distance to metres
        dy = step_length * math.sin(theta) * 1000  

        new_lat = dy + self.shape.y
        new_lon = dx + self.shape.x

        #update heading direction
        co_ord = complex(dx,dy)
        direction = (-1*np.angle([co_ord],deg=True)).tolist()
        self.heading = direction     #unit: degrees

        return new_lon, new_lat
    #-----------------------------------------------------------------------------------------------------
    def targeted_walk(self):
        """" Function to simulate the targeted movement of agents """

        distance = self.distance_calculator_epsg3857(self.shape.y, self.target_lat, self.shape.x, self.target_lon)

        if distance < self.model.xres/2:    #Move to the target
            self.target_present = False
            self.mode = "RandomWalk"        #switch mode

        #The distance an elephant agent moves in 5 minutes. 
        #The step length is calculated from the gamma distribution fitted to the movement data
        #The unit of step length is in kilometers
        mean = 0.03988893 
        std_dv = 0.03784399

        shape = (mean/std_dv)**2
        rate =  mean/std_dv**2
        step_length = np.random.gamma(shape, (1/rate), 1)

        dx = (self.target_lon - self.shape.x)/1000
        dy = (self.target_lat - self.shape.y)/1000

        co_ord = complex(dx,dy)
        direction = np.angle([co_ord], deg=True)

        for _ in range(0,1):
            theta = self.model.random.uniform(direction-30,direction+30) * 0.0174533    #30 degrees to induce randomness in targeted walk

        dx = step_length * math.cos(theta) * 1000       #factor 1000: to convert to metres
        dy = step_length * math.sin(theta) * 1000  
        new_lat = dy + self.shape.y 
        new_lon = dx + self.shape.x 
    
        #update heading direction
        co_ord = complex(dx,dy)
        direction = -np.angle([co_ord],deg=True)   
        self.heading = direction.tolist()        #unit:degrees

        return new_lon,new_lat
    #-----------------------------------------------------------------------------------------------------
    def targeted_walk_modified(self):
        """" Function to simulate the targeted movement of agents """
        #moving one cell at a time

        distance = self.distance_calculator_epsg3857(self.shape.y, self.target_lat, self.shape.x, self.target_lon)

        if distance < self.model.xres/2:    #Move to the target
            self.target_present = False
            self.mode = "RandomWalk"        #switch mode to random walk
            return self.shape.x, self.shape.y

        dx = self.target_lon - self.shape.x
        dy = self.target_lat - self.shape.y
        if dx == 0 and dy == 0:
            # print("same location")
            return self.shape.x, self.shape.y

        co_ord = complex(dx/1000,dy/1000)
        direction = np.angle([co_ord], deg=True)

        #convert direction to degrees between 0 and 360 if it is negative
        if direction < 0:
            direction = 360 + direction

        # print("direction: ", direction)

        #direction between -22.5 and 22.5: move right
        if (direction >= 337.5 and direction <= 360) or (direction >= 0 and direction <= 22.5):
            # print("should move right")
            new_lon = self.shape.x + self.model.xres
            new_lat = self.shape.y

        #direction between 22.5 and 67.5: move up and right 
        elif direction > 22.5 and direction <= 67.5:
            # print("should move up and right")
            new_lon = self.shape.x + self.model.xres
            new_lat = self.shape.y - self.model.yres
        
        #direction between 67.5 and 112.5: move up
        elif direction > 67.5 and direction <= 112.5:
            # print("should move up")
            new_lon = self.shape.x
            new_lat = self.shape.y - self.model.yres

        #direction between 112.5 and 157.5: move up and left
        elif direction > 112.5 and direction <= 157.5:
            # print("should move up and left")
            new_lon = self.shape.x - self.model.xres
            new_lat = self.shape.y - self.model.yres

        #direction between 157.5 and 202.5: move left
        elif direction > 157.5 and direction <= 202.5:
            # print("should move left")
            new_lon = self.shape.x - self.model.xres
            new_lat = self.shape.y

        #direction between 202.5 and 247.5: move down and left
        elif direction > 202.5 and direction <= 247.5:
            # print("should move down and left")
            new_lon = self.shape.x - self.model.xres
            new_lat = self.shape.y + self.model.yres

        #direction between 247.5 and 292.5: move down
        elif direction > 247.5 and direction <= 292.5:
            # print("should move down")
            new_lon = self.shape.x
            new_lat = self.shape.y + self.model.yres

        #direction between 292.5 and 337.5: move down and right
        elif direction > 292.5 and direction <= 337.5:
            # print("should move down and right")
            new_lon = self.shape.x + self.model.xres
            new_lat = self.shape.y + self.model.yres

        # print("cur_lon: ", self.shape.x, "cur_lat: ", self.shape.y)
        # print("new_lon: ", new_lon, "new_lat: ", new_lat)
    
        #update heading direction
        co_ord = complex(dx/1000,dy/1000)
        direction = -np.angle([co_ord],deg=True)   
        self.heading = direction.tolist()        #unit:degrees

        return new_lon, new_lat
    #-----------------------------------------------------------------------------------------------------
    def return_feasible_direction_to_move(self):
        """Return feasible direction to move based on the terrain cost"""

        radius = int(self.model.terrain_radius/self.model.xres)     #spatial resolution: xres

        rad_1 = radius
        rad_2 = radius
        rad_3 = radius
        rad_4 = radius

        # Handling edge cases
        if self.ROW < radius:      
            rad_1 = self.ROW

        elif self.ROW > self.model.row_size-1-radius:
            rad_2 = self.model.row_size - self.ROW - 1

        if self.COL < radius:
            rad_3 = self.COL

        elif self.COL > self.model.col_size-radius-1:
            rad_4 = self.model.col_size - self.COL - 1

        row_start = self.ROW-rad_1    #searches in the viscinity of the current location
        row_end = self.ROW+rad_2
        col_start = self.COL-rad_3
        col_end = self.COL+rad_4

        discount_factor = self.model.discount

        #Movement cost computation
        cost_0 = [discount_factor**i*(self.model.DEM[self.ROW-i][self.COL-i]-self.model.DEM[self.ROW][self.COL]) for i in range(1, np.min([rad_1, rad_3]))]
        cost_2 = [discount_factor**i*(self.model.DEM[self.ROW-i][self.COL+i]-self.model.DEM[self.ROW][self.COL]) for i in range(1, np.min([rad_1, rad_4]))]
        cost_6 = [discount_factor**i*(self.model.DEM[self.ROW+i][self.COL-i]-self.model.DEM[self.ROW][self.COL]) for i in range(1, np.min([rad_3, rad_2]))]
        cost_8 = [discount_factor**i*(self.model.DEM[self.ROW+i][self.COL+i]-self.model.DEM[self.ROW][self.COL]) for i in range(1, np.min([rad_2, rad_4]))]

        cost_1 = [discount_factor**i*(self.model.DEM[self.ROW-i][self.COL]-self.model.DEM[self.ROW][self.COL]) for i in range(1,rad_1)]
        cost_3 = [discount_factor**i*(self.model.DEM[self.ROW][self.COL-i]-self.model.DEM[self.ROW][self.COL]) for i in range(1,rad_3)]
        cost_5 = [discount_factor**i*(self.model.DEM[self.ROW][self.COL+i]-self.model.DEM[self.ROW][self.COL]) for i in range(1,rad_4)]
        cost_7 = [discount_factor**i*(self.model.DEM[self.ROW+i][self.COL]-self.model.DEM[self.ROW][self.COL]) for i in range(1,rad_2)]
        
        cost_0 = sum(x for x in cost_0 if x > 0)
        cost_1 = sum(x for x in cost_1 if x > 0)
        cost_2 = sum(x for x in cost_2 if x > 0)
        cost_3 = sum(x for x in cost_3 if x > 0)
        cost_5 = sum(x for x in cost_5 if x > 0)
        cost_6 = sum(x for x in cost_6 if x > 0)
        cost_7 = sum(x for x in cost_7 if x > 0)
        cost_8 = sum(x for x in cost_8 if x > 0)

        cost = [cost_0, cost_1, cost_2, cost_3, cost_5, cost_6, cost_7, cost_8]
        direction = [135, 90, 45, 180, 0, 225, 270, 315]

        #Adding high movement cost for edges
        if rad_1 < radius:
            cost[0] = 10000
            cost[1] = 10000
            cost[2] = 10000

        if rad_3 < radius:
            cost[0] = 10000
            cost[3] = 10000
            cost[5] = 10000

        if rad_4 < radius:
            cost[2] = 10000
            cost[4] = 10000
            cost[7] = 10000

        if rad_2 < radius:
            cost[5] = 10000
            cost[6] = 10000
            cost[7] = 10000

        #Generate steps in those directions with minimum cost of movement. Discard other directions
        theta = []
        for i in range(0, len(direction)):
            if cost[i] < self.model.tolerance:
                theta.append(direction[i]) 

        #if no direction available, choose direction with minimum movement cost
        if theta == []:    
            min_cost =  cost[0]
            theta = [direction[0]]
            for i in range(0, len(direction)):
                if cost[i] <= min_cost:
                    min_cost = cost[i]
                    theta = [direction[i]]

        #choose a direction to move
        movement_direction = np.random.choice(theta)

        if movement_direction == 0:
            col_start = self.COL + 1
            col_end = self.COL + self.model.radius_food_search/self.model.xres + 1
            row_start = int(self.ROW - self.model.radius_food_search/self.model.xres/2)
            row_end = int(self.ROW + self.model.radius_food_search/self.model.xres/2 + 1)

        if movement_direction == 180:
            col_start = self.COL - self.model.radius_food_search/self.model.xres
            col_end = self.COL 
            row_start = int(self.ROW - self.model.radius_food_search/self.model.xres/2)
            row_end = int(self.ROW + self.model.radius_food_search/self.model.xres/2 + 1)

        if movement_direction == 90:
            col_start = int(self.COL - self.model.radius_food_search/self.model.xres/2 )
            col_end = int(self.COL + self.model.radius_food_search/self.model.xres/2 + 1)
            row_start = self.ROW - self.model.radius_food_search/self.model.xres
            row_end = self.ROW 

        if movement_direction == 270:
            col_start = int(self.COL - self.model.radius_food_search/self.model.xres/2 )
            col_end = int(self.COL + self.model.radius_food_search/self.model.xres/2 + 1)
            row_end = self.ROW + self.model.radius_food_search/self.model.xres + 1
            row_start = self.ROW + 1

        if movement_direction == 45:
            col_start = int(self.COL + self.model.radius_food_search/self.model.xres/2 )
            col_end = self.COL + self.model.radius_food_search/self.model.xres + 1
            row_end = int(self.ROW - self.model.radius_food_search/self.model.xres/2)
            row_start = self.ROW - self.model.radius_food_search/self.model.xres 

        if movement_direction == 135:
            col_start = self.COL - self.model.radius_food_search/self.model.xres 
            row_end = int(self.ROW - self.model.radius_food_search/self.model.xres/2)
            row_start = self.ROW - self.model.radius_food_search/self.model.xres  

        if movement_direction == 225:
            col_end = int(self.COL - self.model.radius_food_search/self.model.xres/2 )
            col_start = self.COL - self.model.radius_food_search/self.model.xres 
            row_start = int(self.ROW + self.model.radius_food_search/self.model.xres/2)
            row_end = self.ROW + self.model.radius_food_search/self.model.xres  + 1
            
        if movement_direction == 315:
            col_start = int(self.COL + self.model.radius_food_search/self.model.xres/2 )
            col_end = self.COL + self.model.radius_food_search/self.model.xres
            row_start = int(self.ROW + self.model.radius_food_search/self.model.xres/2)
            row_end = self.ROW + self.model.radius_food_search/self.model.xres  + 1

        if row_start < 0:
            row_start = 0

        if row_start >= self.model.row_size:
            row_start = self.model.row_size - self.model.radius_food_search/self.model.xres

        if row_end < 0:
            row_end = self.model.radius_food_search/self.model.xres

        if row_end >= self.model.row_size:
            row_end = self.model.row_size - 1

        if col_start < 0:
            col_start = 0

        if col_start >= self.model.col_size:
            col_start = self.model.col_size - self.model.radius_food_search/self.model.xres

        if col_end >= self.model.col_size:
            col_end = self.model.col_size - 1

        if col_end <0:
            col_end = 0

        row_start = int(row_start)
        row_end = int(row_end)
        col_start = int(col_start)
        col_end = int(col_end)

        #print(row_start, row_end, col_start, col_end)

        return row_start, row_end, col_start, col_end
    #-----------------------------------------------------------------------------------------------------
    def return_feasible_direction_to_move_modified(self):

        radius = int(self.model.terrain_radius*2/self.model.xres) + 1   #spatial resolution: xres

        #create a n*n numpy array to store the data
        data = np.zeros((radius, radius), dtype=object)

        #fill the array with theta values calculated on the basis of row and column index with respect to the center of the array
        for i in range(0, radius):
            for j in range(0, radius):
                data[i][j] = np.arctan2(((i-(radius//2))*np.pi),((j-(radius//2))*np.pi))

        #convert the array to degrees from radians
        data = np.rad2deg(data.astype(float))

        #find min and max of the array
        min_val = np.amin(data)
        max_val = np.amax(data)

        #set center value to -1
        data[radius//2][radius//2] = -500

        #discretize the array into 8 bins
        data = np.digitize(data, np.linspace(min_val, max_val, 17))

        #map values in array based on the following mapping
        map = {0:0, 1:8, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3, 8:4, 9:4, 10:5, 11:5, 12:6, 13:6, 14:7, 15:7, 16:8, 17:8}

        for i in range(0, radius):
            for j in range(0, radius):
                data[i][j] = map[data[i][j]]

        data[radius//2][radius//2] = 9
        slope = np.array(self.model.SLOPE)[self.ROW - radius//2:self.ROW + radius//2 + 1, self.COL - radius//2:self.COL + radius//2 + 1]

        #sum the values in each direction based on the data array
        direction_0 = slope[data == 1]
        direction_1 = slope[data == 2]
        direction_2 = slope[data == 3]
        direction_3 = slope[data == 4]
        direction_4 = slope[data == 5]
        direction_5 = slope[data == 6]
        direction_6 = slope[data == 7]
        direction_7 = slope[data == 8]

        #find cells greater than 30 degrees in each direction
        direction_0 = [x for x in direction_0.flatten() if x > 30]
        direction_1 = [x for x in direction_1.flatten() if x > 30]
        direction_2 = [x for x in direction_2.flatten() if x > 30]
        direction_3 = [x for x in direction_3.flatten() if x > 30]
        direction_4 = [x for x in direction_4.flatten() if x > 30]
        direction_5 = [x for x in direction_5.flatten() if x > 30]
        direction_6 = [x for x in direction_6.flatten() if x > 30]
        direction_7 = [x for x in direction_7.flatten() if x > 30]

        #calculate the cost of movement in each direction as sum of the direction cells
        cost_0 = sum(x for x in direction_0 if x > 0)
        cost_1 = sum(x for x in direction_1 if x > 0)
        cost_2 = sum(x for x in direction_2 if x > 0)
        cost_3 = sum(x for x in direction_3 if x > 0)
        cost_4 = sum(x for x in direction_4 if x > 0)
        cost_5 = sum(x for x in direction_5 if x > 0)
        cost_6 = sum(x for x in direction_6 if x > 0)
        cost_7 = sum(x for x in direction_7 if x > 0)

        cost = [cost_0, cost_1, cost_2, cost_3, cost_4, cost_5, cost_6, cost_7]
        direction = [135, 90, 45, 0, 315, 270, 225, 180]

        # print("cost: ", cost)

        #Generate steps in those directions with minimum cost of movement. Discard other directions
        theta = []
        for i in range(0, len(direction)):
            if cost[i] <= self.model.tolerance:
                theta.append(direction[i]) 

        #if no direction available, choose direction with minimum movement cost
        if theta == []:    
            min_cost =  cost[0]
            theta = [direction[0]]
            for i in range(1, len(direction)):
                if cost[i] <= min_cost:
                    min_cost = cost[i]
                    theta = [direction[i]]

        #choose a direction to move
        movement_direction = np.random.choice(theta)

        if movement_direction == 135:
            #create an array with 0 when data array != 1
            filter = np.zeros_like(data)
            filter[data == 1] = 1

        elif movement_direction == 90:
            #create an array with 0 when data array != 2
            filter = np.zeros_like(data)
            filter[data == 2] = 1

        elif movement_direction == 45:
            #create an array with 0 when data array != 3
            filter = np.zeros_like(data)
            filter[data == 3] = 1

        elif movement_direction == 0:
            #create an array with 0 when data array != 4
            filter = np.zeros_like(data)
            filter[data == 4] = 1

        elif movement_direction == 315:
            #create an array with 0 when data array != 5
            filter = np.zeros_like(data)
            filter[data == 5] = 1

        elif movement_direction == 270:
            #create an array with 0 when data array != 6
            filter = np.zeros_like(data)
            filter[data == 6] = 1

        elif movement_direction == 225:
            #create an array with 0 when data array != 7
            filter = np.zeros_like(data)
            filter[data == 7] = 1

        elif movement_direction == 180:
            #create an array with 0 when data array != 8
            filter = np.zeros_like(data)
            filter[data == 8] = 1

        #center value is set to -1
        filter[radius//2][radius//2] = 2

        # print("\n")
        # print("Movement Direction:", movement_direction)
        # print("filter: \n", filter)

        self.direction = movement_direction

        return filter
    #-----------------------------------------------------------------------------------------------------
    def target_for_foraging(self, row_start, row_end, col_start, col_end):
        """ Function returns the foraging target for the elephant agent to move."""

        if self.target_present == True:     #If target already exists
            return
        
        ##########################################################################

        coord_list=[]

        val = self.model.plantation_proximity[self.ROW][self.COL]

        prob = self.aggress_factor

        if self.food_habituation < self.model.food_habituation_threshold:  #Not food habituated

            #print("not food habituated")

            for i in range(row_start,row_end):
                for j in range(col_start,col_end):
                    if i == self.ROW and j == self.COL:
                        pass

                    elif self.food_memory[i][j] > 0:
                        coord_list.append([i,j])

        else:  

            #print("food habituated")                                                     #food habituated

            mask2 = (np.array(self.model.LANDUSE) != 10)  #others
            mask1 = (np.array(self.model.LANDUSE) == 10)  #cropland
            food_memory = np.array(self.food_memory)
            food_forest = sum(food_memory[mask2])
            food_cropland = sum(food_memory[mask1])

            if self.num_days_food_depreceation >= 3:    #FOOD DEPRECEATED

                if np.random.uniform(0,1) < prob:
                    #print("move closer to plantation")
                    for i in range(row_start,row_end):
                        for j in range(col_start,col_end):

                            if self.model.plantation_proximity[i][j] <= val:
                                coord_list.append([i,j]) 

                else:
                    #print("target in forest")
                    for i in range(row_start,row_end):
                        for j in range(col_start,col_end):
                            if i == self.ROW and j == self.COL:
                                pass

                            elif self.food_memory[i][j] > 0:
                                coord_list.append([i,j])     

            else:

                if food_cropland > 1.5*food_forest:    #HIGH FOOD AVAILABILITY IN CROPLAND
                    #print("High food in cropland")
                    #print("move closer to plantation")
                    for i in range(row_start,row_end):
                        for j in range(col_start,col_end):

                            if self.model.plantation_proximity[i][j] <= val:
                                coord_list.append([i,j]) 
  
                else:
                    #print("Low food in cropland")
                    #print("target in forest")
                    for i in range(row_start,row_end):
                        for j in range(col_start,col_end):
                            if i == self.ROW and j == self.COL:
                                pass

                            elif self.food_memory[i][j] > 0:
                                coord_list.append([i,j])      

        ##########################################################################     

        if coord_list == []:
            coord_list.append([self.ROW,self.COL])

        x, y = self.model.random.choice(coord_list)
        lon = self.model.xres * 0.5  + self.model.xmin + y * self.model.xres
        lat = self.model.yres * 0.5  + self.model.ymax + x * self.model.yres
        self.target_lon, self.target_lat = lon, lat
        self.target_present = True

        return
    #-----------------------------------------------------------------------------------------------------
    def target_to_drink_water(self, row_start, row_end, col_start, col_end):

        """ Function returns the target for the elephant agent to move.
        The target is selected from the memory matrix, where the elephant agent thinks it can find water.
        Barrier to movement is considered while selecting the target to move."""

        if self.target_present == True:     #If target already exists
            return

        coord_list=[]

        for i in range(row_start,row_end):
            for j in range(col_start,col_end):
                if i == self.ROW and j == self.COL:
                    pass

                elif self.water_memory[i][j]>0:
                    coord_list.append([i, j])

        #model : move closer to the water area
        if coord_list==[]:
            val = self.model.water_proximity[self.ROW][self.COL]
            coord_list.append([self.ROW,self.COL])
            for _ in range(10):
                i = self.model.random.randint(row_start, row_end)
                j = self.model.random.randint(col_start, col_end)
                if self.model.water_proximity[i][j] <= val:
                    coord_list.append([i,j])

        x, y = self.model.random.choice(coord_list)
        lon = self.model.xres * 0.5  + self.model.xmin + y * self.model.xres
        lat = self.model.yres * 0.5  + self.model.ymax + x * self.model.yres
        self.target_lon, self.target_lat = lon, lat
        self.target_present = True

        return
    #-----------------------------------------------------------------------------------------------------
    def target_thermoregulation(self, row_start, row_end, col_start, col_end):

        """ Function returns the target for the elephant agent to move.
        The target is selected from the memory matrix, where the elephant agent thinks it can find water or shade.
        Barrier to movement is considered while selecting the target to move."""

        if self.target_present == True:     #If target already exists
            return

        coord_list=[]

        for i in range(row_start,row_end):
            for j in range(col_start,col_end):
                if i == self.ROW and j == self.COL:
                    pass

                elif self.model.temp[i,j] <= self.model.temp[self.ROW,self.COL]:
                    coord_list.append([i, j])

        if coord_list != []:
            x, y = self.model.random.choice(coord_list)
            lon = self.model.xres * 0.5  + self.model.xmin + y * self.model.xres
            lat = self.model.yres * 0.5  + self.model.ymax + x * self.model.yres
            self.target_lon, self.target_lat = lon, lat
            self.target_present = True

        elif coord_list == []:
            self.target_present = False
            if self.model.random.uniform(0,1) < 0.5:
                self.target_to_drink_water(row_start, row_end, col_start, col_end)
            else:
                self.target_for_escape()

        return
    #-----------------------------------------------------------------------------------------------------
    def target_for_escape(self):
        """ Function returns the target for the elephant agent to move in case of danger to life. """

        radius = int(self.radius_forest_search/self.model.xres)     #spatial resolution

        row_start = self.ROW-radius    #searches in the viscinity of the current location
        row_end = self.ROW+radius+1
        col_start = self.COL-radius
        col_end = self.COL+radius+1
                        
        #To handle edge cases
        if self.ROW < radius:
            row_start = 0

        elif self.ROW > self.model.row_size-1-radius:
            row_end = self.model.row_size-1

        if self.COL < radius:
            col_start = 0

        elif self.COL > self.model.col_size-radius-1:
            col_end = self.model.col_size-1

        coord_list=[]
        
        for i in range(row_start,row_end):
            for j in range(col_start,col_end):
                if i == self.ROW and j == self.COL:
                    pass

                elif self.model.LANDUSE[i][j] == 15 and self.model.LANDUSE[i][j] == 4:
                    coord_list.append([i, j])

        #model: move closer to the forest
        if coord_list==[]:
            val = self.model.forest_proximity[self.ROW][self.COL]
            coord_list.append([self.ROW,self.COL])
            for _ in range(10):
                i = self.model.random.randint(row_start, row_end)
                j = self.model.random.randint(col_start, col_end)
                if self.model.forest_proximity[i][j] < val:
                    coord_list.append([i,j])

        x, y = self.model.random.choice(coord_list)
        lon = self.model.xres * 0.5  + self.model.xmin + y * self.model.xres
        lat = self.model.yres * 0.5  + self.model.ymax + x * self.model.yres
        self.target_lon, self.target_lat = lon, lat
        self.target_present = True

        return
    #-----------------------------------------------------------------------------------------------------
    def target_for_foraging_modified(self, filter):

        # print("FORAGING")

        if self.target_present == True:     #If target already exists
            # print("target already exists")
            return
        
        ##########################################################################

        coord_list=[]
        radius = int(self.model.terrain_radius*2/self.model.xres) + 1   #spatial resolution: xres

        row_start = self.ROW - radius//2
        col_start = self.COL - radius//2
        row_end = self.ROW + radius//2 + 1
        col_end = self.COL + radius//2 + 1

        val = self.model.plantation_proximity[self.ROW][self.COL]
        prob = self.aggress_factor

        if self.food_habituation < self.model.food_habituation_threshold:  #Not food habituated

            # print("not food habituated")
            # print("target in forest")
            for i in range(row_start,row_end):
                for j in range(col_start,col_end):
                    if i == self.ROW and j == self.COL:
                        pass

                    elif self.food_memory[i][j] > 0 and filter[i - row_start][j - col_start] == 1:
                        coord_list.append([i,j])

        else:  

            # print("food habituated")                                                     #food habituated
            mask2 = (np.array(self.model.LANDUSE) != 10)  #others
            mask1 = (np.array(self.model.LANDUSE) == 10)  #cropland
            food_memory = np.array(self.food_memory)
            food_forest = sum(food_memory[mask2])
            food_cropland = sum(food_memory[mask1])

            if self.num_days_food_depreceation >= 3:    #FOOD DEPRECEATED

                if self.model.random.uniform(0,1) < prob:
                    #proximity_vals = []
                    # print("move closer to plantation")
                    for i in range(row_start,row_end):
                        for j in range(col_start,col_end):

                            if self.model.plantation_proximity[i][j] <= val and filter[i - row_start][j - col_start] == 1:
                                coord_list.append([i,j]) 

                else:
                    # print("target in forest")
                    for i in range(row_start,row_end):
                        for j in range(col_start,col_end):
                            if i == self.ROW and j == self.COL:
                                pass

                            elif self.food_memory[i][j] > 0 and filter[i - row_start][j - col_start] == 1:
                                coord_list.append([i,j])     

            else:

                if food_cropland > 1.25*food_forest:    #HIGH FOOD AVAILABILITY IN CROPLAND
                    # print("High food in cropland")
                    # print("move closer to plantation")
                    for i in range(row_start,row_end):
                        for j in range(col_start,col_end):

                            if self.model.plantation_proximity[i][j] <= val and filter[i - row_start][j - col_start] == 1:
                                coord_list.append([i,j]) 
  
                else:
                    # print("Low food in cropland")
                    # print("target in forest")
                    for i in range(row_start,row_end):
                        for j in range(col_start,col_end):
                            if i == self.ROW and j == self.COL:
                                pass

                            elif self.food_memory[i][j] > 0 and filter[i - row_start][j - col_start] == 1:
                                coord_list.append([i,j])      
        ##########################################################################     

        if coord_list == []:
            coord_list.append([self.ROW,self.COL])
        x, y = self.model.random.choice(coord_list)
        lon, lat = self.model.pixel2coord(x, y)
        self.target_lon, self.target_lat = lon, lat
        self.target_present = True

        return
    #-----------------------------------------------------------------------------------------------------
    def target_to_drink_water_modified(self, filter):

        # print("DRINKING WATER")

        """ Function returns the target for the elephant agent to move.
        The target is selected from the memory matrix, where the elephant agent thinks it can find water.
        Barrier to movement is considered while selecting the target to move."""

        if self.target_present == True:     #If target already exists
            return

        coord_list=[]

        radius = int(self.radius_water_search*2/self.model.xres)     #spatial resolution

        row_start = self.ROW - radius//2
        col_start = self.COL - radius//2
        row_end = self.ROW + radius//2 + 1
        col_end = self.COL + radius//2 + 1

        for i in range(row_start,row_end):
            for j in range(col_start,col_end):
                if i == self.ROW and j == self.COL:
                    pass

                elif self.water_memory[i][j] > 0 and filter[i - row_start][j - col_start] == 1:
                    coord_list.append([i, j])

        #model : move closer to the water area
        if coord_list==[]:
            val = self.model.water_proximity[self.ROW][self.COL]
            coord_list.append([self.ROW,self.COL])
            for _ in range(10):
                i = self.model.random.randint(row_start, row_end - 1)
                j = self.model.random.randint(col_start, col_end - 1)
                if self.model.water_proximity[i][j] <= val and filter[i - row_start][j - col_start] == 1:
                    coord_list.append([i,j])

        x, y = self.model.random.choice(coord_list)
        lon, lat = self.model.pixel2coord(x, y)
        self.target_lon, self.target_lat = lon, lat
        self.target_present = True

        return
    #-----------------------------------------------------------------------------------------------------
    def target_thermoregulation_modified(self, filter):

        # print("THERMOREGULATION")

        """ Function returns the target for the elephant agent to move.
        The target is selected from the memory matrix, where the elephant agent thinks it can find water.
        Barrier to movement is considered while selecting the target to move."""

        if self.target_present == True:     #If target already exists
            return

        coord_list=[]

        radius = int(self.radius_water_search*2/self.model.xres)     #spatial resolution

        row_start = self.ROW - radius//2
        col_start = self.COL - radius//2
        row_end = self.ROW + radius//2 + 1
        col_end = self.COL + radius//2 + 1

        for i in range(row_start,row_end):
            for j in range(col_start,col_end):
                if i == self.ROW and j == self.COL:
                    pass

                elif self.model.temp[i,j] <= self.model.temp[self.ROW,self.COL] and filter[i - row_start][j - col_start] == 1:
                    coord_list.append([i, j])

        if coord_list != []:
            x, y = self.model.random.choice(coord_list)
            # lon = self.model.xres * 0.5  + self.model.xmin + y * self.model.xres
            # lat = self.model.yres * 0.5  + self.model.ymax + x * self.model.yres
            lon, lat = self.model.pixel2coord(x, y)
            self.target_lon, self.target_lat = lon, lat
            self.target_present = True

        elif coord_list == []:
            self.target_present = False
            if self.model.random.uniform(0,1) < 0.85:
                self.target_to_drink_water(filter)
            else:
                self.target_for_foraging(filter)

        return
    #-----------------------------------------------------------------------------------------------------
    def target_for_escape_modified(self):

        radius = int(self.radius_water_search/self.model.xres)     #spatial resolution

        row_start = self.ROW-radius    #searches in the viscinity of the current location
        row_end = self.ROW+radius+1
        col_start = self.COL-radius
        col_end = self.COL+radius+1
                        
        #To handle edge cases
        if self.ROW < radius:
            row_start = 0

        elif self.ROW > self.model.row_size-1-radius:
            row_end = self.model.row_size-1

        if self.COL < radius:
            col_start = 0

        elif self.COL > self.model.col_size-radius-1:
            col_end = self.model.col_size-1

        coord_list=[]
        
        for i in range(row_start,row_end):
            for j in range(col_start,col_end):
                if i == self.ROW and j == self.COL:
                    pass

                elif self.model.LANDUSE[i][j] == 15:
                    coord_list.append([i, j])

        #model: move closer to the forest
        if coord_list==[]:
            val = self.model.forest_proximity[self.ROW][self.COL]
            coord_list.append([self.ROW,self.COL])
            for _ in range(10):
                i = self.model.random.randint(row_start, row_end)
                j = self.model.random.randint(col_start, col_end)
                if self.model.forest_proximity[i][j] < val:
                    coord_list.append([i,j])

        x, y = self.model.random.choice(coord_list)
        lon = self.model.xres * 0.5  + self.model.xmin + y * self.model.xres
        lat = self.model.yres * 0.5  + self.model.ymax + x * self.model.yres
        self.target_lon, self.target_lat = lon, lat
        self.target_present = True

        return
    #-----------------------------------------------------------------------------------------------------
    def InflictDamage(self):
        """Function to inflict damage on the human agents"""
        
        for neighbor in self.conflict_neighbor:
            neighbor.fitness = neighbor.fitness - self.model.random.uniform(0, self.model.fitness_fn_decrement_humans)

        return
    #----------------------------------------------------------------------------------------------------
    def drink_water(self):
        """ The elephant agent consumes water from the current cell it is located in"""

        row, col = self.update_grid_index()

        if "dry" in self.model.season:

            if self.model.WATER[row][col]>0:
                self.visit_water_source = True

                #update fitness value
                # self.update_fitness_value(self.model.fitness_increment_when_drinks_water_dry)

        else: 

            if self.model.WATER[row][col]>0:
                self.visit_water_source = True

                #update fitness value
                # self.update_fitness_value(self.model.fitness_increment_when_drinks_water_wet)

        return
    #---------------------------------------------------------------------------------------------------
    def eat_food(self):

        """ The elephant agent consumes food from the current cell it is located in"""

        row, col = self.update_grid_index()

        if self.model.FOOD[row][col] > 0:
            food_consumed = self.model.random.uniform(0,self.model.FOOD[row][col])
            self.food_consumed += food_consumed
            self.model.FOOD[row][col] -= food_consumed
            self.food_memory[row][col] -= food_consumed

            if self.model.FOOD[row][col] < 0:
                self.model.FOOD[row][col] = 0
                self.food_memory[row][col] = 0

            #update fitness value
            # self.update_fitness_value(self.model.fitness_increment_when_eats_food)

        return
    #----------------------------------------------------------------------------------------------------
    def crop_infrastructure_damage(self):
        """Function to simulate the crop and infrastructure damage by the elephant agents"""

        #print("CROP AND INFRASTRUCTURE DAMAGE!")
        row, col = self.update_grid_index()

        status = self.model.crop_status[row][col] 

        num_1 = np.random.uniform(0,1)
        num_2 = np.random.uniform(0,1)

        if self.model.INFRASTRUCTURE[row][col] != 0:
            if num_1 < self.model.prob_infrastructure_damage:
                #print("infrastructure damage")
                self.model.infrastructure_damage_incidents = (self.shape.x, self.shape.y)

            else:
                self.model.infrastructure_damage_incidents = None

        else:
            self.model.infrastructure_damage_incidents = None

        if self.model.AGRI_PLOTS[row][col] != 0 and self.model.FOOD[row][col] > 0:
            if num_2 < self.model.prob_crop_damage: 
                #print("crop damage")
                if status == 1:
                    self.model.crop_damage_incidents_fixed = (self.shape.x, self.shape.y)
                elif status == 2:
                    self.model.crop_damage_incidents_variable = (self.shape.x, self.shape.y)

            else:
                self.model.crop_damage_incidents_fixed = None
                self.model.crop_damage_incidents_variable = None

        else:
            self.model.crop_damage_incidents_fixed = None
            self.model.crop_damage_incidents_variable = None
        return
    #----------------------------------------------------------------------------------------------------
    def update_aggression_factor(self, val):
        """The function updates the aggression factor of the agent"""

        aggress_factor = self.aggress_factor
        aggress_factor += val

        if aggress_factor < 0:
            self.aggress_factor = 0

        elif aggress_factor > 1:
            self.aggress_factor = 1

        else:
            self.aggress_factor = aggress_factor

        return 
    #----------------------------------------------------------------------------------------------------
    def update_fitness_value(self, val):
        """The function updates the fitness value of the agent"""

        # print("UPDATE FITNESS VALUE:", val)

        fitness = self.fitness
        fitness += val

        if fitness <= 0:
            self.fitness = 0
        elif fitness > 1:
            self.fitness = 1
        else:
            self.fitness = fitness
        return
    #----------------------------------------------------------------------------------------------------
    def update_fitness_thermoregulation(self, num_thermoregulation_steps, num_steps_thermoregulated):
        """Function to update the fitness value of the agent as per thermoregulation criteria"""
        #num_thermoregulation_steps: number of steps the agent has to thermoregulate
        #num_steps_thermoregulated: number of steps the agent has thermoregulated

        # print("UPDATE FITNESS: THERMOREGULATION")

        try:

            fitness_increment = (1/10)*(num_thermoregulation_steps/288)*(num_steps_thermoregulated/num_thermoregulation_steps)
            self.update_fitness_value(fitness_increment)

        except:
            pass

        # print("Fitness Increment: ", fitness_increment)

        return
    #----------------------------------------------------------------------------------------------------
    def update_fitness_foraging(self, num_thermoregulation_steps, food_consumed):
        """Function to update the fitness value of the agent as per foraging criteria"""
        #num_thermoregulation_steps: number of steps the agent has to thermoregulate
        #food_consumed: amount of food consumed by the agent

        # print("UPDATE FITNESS: FORAGING")

        fitness_increment = (1/10)*((288-num_thermoregulation_steps)/288)*(min(food_consumed, self.daily_dry_matter_intake)/self.daily_dry_matter_intake)
        self.update_fitness_value(fitness_increment)

        if food_consumed > self.daily_dry_matter_intake and self.fitness < 0:
            self.update_fitness_value(0.1)

        # print("Fitness Increment: ", fitness_increment)

        return
    #----------------------------------------------------------------------------------------------------
    def update_memory_matrix(self):
        """Function to update the memory matrix of the agent"""
        food = np.array(self.model.FOOD)
        food_mem = np.array(self.memory)
        food_memory = np.zeros_like(food_mem)
        food_memory[(food_mem > 0)] = food[(food_mem > 0)]
        self.food_memory = food_memory
    #----------------------------------------------------------------------------------------------------
    def update_age(self):
        """Function to update the age of the agent"""
        self.age += 1
    #----------------------------------------------------------------------------------------------------
    def elephant_cognition(self):
        """Function to simulate the cognition of the elephant agent"""

        if (self.model.model_time%288) == 0 and self.model.model_time>=288:

            print("X---------------------------------X")

            print("FITNESS BEFORE THERMOREGULATION AND FORAGING UPDATE: ", self.fitness)

            print("updating the day!")
            print("Day: ", self.model.model_day, "num_thermoregulation_steps: ", self.num_thermoregulation_steps, "num_steps_thermoregulated: ", self.num_steps_thermoregulated, "food_consumed: ", self.food_consumed, "daily_dry_matter_intake: ", self.daily_dry_matter_intake)

            #update the fitness value as per the thermoregulation criteria and the foraging criteria
            self.update_fitness_thermoregulation(self.num_thermoregulation_steps, self.num_steps_thermoregulated)
            self.update_fitness_foraging(self.num_thermoregulation_steps, self.food_consumed)

            print("FITNESS AFTER THERMOREGULATION AND FORAGING UPDATE: ", self.fitness)

            self.food_goal = self.daily_dry_matter_intake   

            if self.visit_water_source == False:    
                self.num_days_water_source_visit += 1     
                self.update_fitness_value(0)     
                self.update_aggression_factor(self.model.aggression_fn_increment_elephants)     
            else:
                if self.num_days_water_source_visit == 0:
                    self.update_fitness_value(0)        
                    self.update_aggression_factor(0)        
                else:                   
                    self.update_fitness_value(0)        
                    self.update_aggression_factor(0)     

                self.visit_water_source = False
                self.num_days_water_source_visit = 0    

            if self.food_consumed < self.daily_dry_matter_intake:
                self.num_days_food_depreceation += 1
                self.update_fitness_value(0)   
                self.update_aggression_factor(self.model.aggression_fn_increment_elephants)     
            else:
                if self.num_days_food_depreceation == 0:
                    self.update_fitness_value(0)     
                    self.update_aggression_factor(0)     
                else:                  
                    self.update_fitness_value(0)     
                    self.update_aggression_factor(0)   

                self.num_days_food_depreceation = 0
            
            self.food_consumed = 0 
            self.num_steps_thermoregulated = 0
            self.num_thermoregulation_steps = 0

        self.next_step_to_move()

        return
    #----------------------------------------------------------------------------------------------------
    def step(self):     
        """ Function to simulate the movement of the elephant agent"""

        self.elephant_cognition()

        self.ROW, self.COL = self.update_grid_index()
        self.update_aggression_factor(0)

        if (self.model.model_time%2880) == 0 and self.model.model_time>=288:
            self.update_memory_matrix()

        if self.fitness <= 0:
            self.model.schedule.remove(self)     #REMOVING FROM THE SCHEDULE
            self.model.grid.remove_agent(self)   #REMOVING FROM THE GRID
            self.model.num_elephant_deaths += 1

        self.slope = self.model.SLOPE[self.ROW][self.COL]
        self.hour = self.model.hour_in_day
        self.landuse = self.model.LANDUSE[self.ROW][self.COL]
        self.water_source_proximity = self.model.water_proximity[self.ROW][self.COL]

        self.conflict_neighbor = None
    #----------------------------------------------------------------------------------------------------





class environment():

    def __init__(self, prob_food_in_forest, prob_food_in_cropland, prob_water_sources, max_food_val_forest, max_food_val_cropland, output_folder):
        self.prob_food_in_forest =  prob_food_in_forest
        self.prob_food_in_cropland = prob_food_in_cropland
        self.prob_water_sources = prob_water_sources
        self.max_food_val_forest = max_food_val_forest
        self.max_food_val_cropland = max_food_val_cropland
        self.output_folder = output_folder
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def initialize_food_matrix(self):
        """Function returns a food matrix with values 0-num, 0 being no food avavilability and num being high food availability
        """

        folder_path = os.path.join("mesageo_elephant_project/elephant_project/", "experiment_setup_files","environment_seethathode","Raster_Files_Seethathode_Derived", "area_1100sqKm/reso_30x30")
        fid = os.path.join(folder_path, "LULC.tif")

        Plantation = gdal.Open(fid).ReadAsArray()
        m,n=Plantation.shape

        food_matrix = np.zeros_like(Plantation)
        landscape_cell_status = np.zeros_like(Plantation)

        for i in range(0,m):
            for j in range(0,n):
                if np.random.uniform(0,1) < self.prob_food_in_cropland and Plantation[i,j] == 10:
                    landscape_cell_status[i,j] = 2

                elif np.random.uniform(0,1) < self.prob_food_in_forest and Plantation[i,j] == 15:   
                    landscape_cell_status[i,j] = 1

        forest_mask = (Plantation == 15) & (landscape_cell_status == 1)
        cropland_mask = (Plantation == 10) & (landscape_cell_status == 2)

        food_matrix[forest_mask] = np.random.uniform(0, self.max_food_val_forest, size=(m,n))[forest_mask]
        food_matrix[cropland_mask] = np.random.uniform(0, self.max_food_val_forest, size=(m,n))[cropland_mask]

        #saving the food matrix
        fid = os.path.join(folder_path, "LULC.tif")

        with rio.open(fid) as src:
            ras_data = src.read()
            ras_meta = src.profile

        ras_meta['dtype'] = "float64"
        ras_meta['nodata'] = -99

        fid = os.path.join(self.output_folder, "food_matrix_"+ str(self.prob_food_in_forest) + "_" + str(self.prob_food_in_cropland) + "_.tif")

        with rio.open(fid, 'w', **ras_meta) as dst:
            dst.write(food_matrix.astype(float), 1)

        fid = os.path.join(self.output_folder, "landscape_cell_status.tif")
        with rio.open(fid, 'w', **ras_meta) as dst:
            dst.write(landscape_cell_status.astype(float), 1)

        return 
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def initialize_water_matrix(self):
        
        """The function initializes water matrix based on the simulation parameters"""
        #Prob_water: probability of water being available in a given cell 
        
        #Reading the LULC and storing the plantation area details
        folder_path = os.path.join("mesageo_elephant_project/elephant_project/", "experiment_setup_files","environment_seethathode","Raster_Files_Seethathode_Derived", "area_1100sqKm/reso_30x30")
        fid = os.path.join(folder_path, "LULC.tif")

        LULC = gdal.Open(fid).ReadAsArray()
        rowmax, colmax = LULC.shape

        water_matrix = np.zeros_like(LULC)

        for i in range(0,rowmax):
            for j in range(0,colmax):
                if LULC[i,j]==9:
                    water_matrix[i,j]=1

                if np.random.uniform(0,1) < self.prob_water_sources:
                    water_matrix[i,j]=1

        #saving the water matrix
        fid = os.path.join(folder_path, "LULC.tif")

        with rio.open(fid) as src:
            ras_data = src.read()
            ras_meta = src.profile

        # make any necessary changes to raster properties, e.g.:
        ras_meta['dtype'] = "int32"
        ras_meta['nodata'] = -99

        fid = os.path.join(self.output_folder, "water_matrix_" + str(self.prob_water_sources) +"_.tif")

        with rio.open(fid, 'w', **ras_meta) as dst:
            dst.write(water_matrix.astype(int), 1)

        return
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def main(self):

        self.initialize_food_matrix()
        self.initialize_water_matrix()

        return
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------





class Conflict_model(Model):
    """ 
    Model class: Elephant-Human interaction model
    """

    #Model Initialization
    def __init__(self,
        num_bull_elephants, #number of solitary bull elephants in the simulation
        year,
        month,
        max_time, #maximum simulation time (in ticks)
        area_size, #simulation area in sq. km
        spatial_resolution, #spatial resolution of the simulation area
        temporal_resolution, #temporal resolution of one tick in minutes
        prob_food_forest, #probability of food in the forest
        prob_food_cropland, #probability of food in the forest
        prob_water_sources, #probability of water in the forest
        max_food_val_forest, #maximum food value in the forest cell
        max_food_val_cropland,  #maximum food value in the cropland cell
        prob_drink_water_dry, #probability of the elephant agent drinking water in each tick: dry season
        prob_drink_water_wet, #probability of the elephant agent drinking water in each tick: wet season
        percent_memory_elephant, #percentage memory of the landscape cells by the elephant agents at the start of the simulation
        radius_food_search, #radius within which the elephant agent searches for food
        radius_water_search, #radius within which the elephant agent searches for water
        movement_fitness_depreceation, #fitness depreceation in each tick
        fitness_increment_when_eats_food, #fitness increment when consumes food
        fitness_increment_when_drinks_water_dry, #fitness increment when drinks water:dry season
        fitness_increment_when_drinks_water_wet, #fitness increment when drinks water:wet season
        fitness_threshold, #fitness threshold below which the elephant agent engages only in "TargetedWalk" mode
        discount, #parameter in terrain cost function
        tolerance, #parameter in terrain cost function
        terrain_radius, #parameter in terrain cost function
        knowledge_from_fringe,  #distance from the fringe where elephants knows food availability
        prob_crop_damage, #probability of damaging crop if entered an agricultural field
        prob_infrastructure_damage #probability of damaging infrastructure if entered a settlement area
        ):



        MAP_COORDS=[9.3245, 76.9974]   



        #Folders to read the data files from depending upon the area and resolution
        self.area = {800:"area_800sqKm", 900:"area_900sqKm", 1000:"area_1000sqKm", 1100:"area_1100sqKm"}
        self.reso = {30: "reso_30x30", 60:"reso_60x60", 90:"reso_90x90", 120:"reso_120x120", 150:"reso_150x150"}
        self.delta = {800:14142, 900:15000, 1000:15811, 1100: 16583}  #distance in meteres from the center of the polygon area



        #-------------------------------------------------------------------
        self.num_bull_elephants = num_bull_elephants    
        self.year = year
        self.month = month
        self.max_time = max_time
        self.area_size = area_size
        self.spatial_resolution = spatial_resolution  
        self.temporal_resolution = temporal_resolution
        self.prob_food_forest = prob_food_forest
        self.prob_food_cropland = prob_food_cropland
        self.prob_water_sources = prob_water_sources
        self.max_food_val_forest = max_food_val_forest
        self.max_food_val_cropland = max_food_val_cropland
        self.prob_drink_water_dry = prob_drink_water_dry
        self.prob_drink_water_wet = prob_drink_water_wet
        self.percent_memory_elephant = percent_memory_elephant
        self.radius_food_search = radius_food_search
        self.radius_water_search = radius_water_search
        self.movement_fitness_depreceation = movement_fitness_depreceation
        self.fitness_increment_when_eats_food = fitness_increment_when_eats_food
        self.fitness_increment_when_drinks_water_dry = fitness_increment_when_drinks_water_dry
        self.fitness_increment_when_drinks_water_wet = fitness_increment_when_drinks_water_wet
        self.fitness_threshold = fitness_threshold
        self.discount = discount
        self.tolerance = tolerance
        self.terrain_radius = terrain_radius
        self.knowledge_from_fringe = knowledge_from_fringe
        self.prob_crop_damage = prob_crop_damage
        self.prob_infrastructure_damage = prob_infrastructure_damage
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        #Geographical extend of the study area
        #-------------------------------------------------------------------
        latitude_center = MAP_COORDS[0]
        longitude_center = MAP_COORDS[1]
        inProj, outProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857') 
        self.center_lon, self.center_lat  = transform(inProj, outProj, longitude_center, latitude_center)

        latextend,lonextend = [self.center_lat-self.delta[area_size],self.center_lat+self.delta[area_size]],[self.center_lon-self.delta[area_size],self.center_lon+self.delta[area_size]]
        self.LAT_MIN_epsg3857,self.LAT_MAX_epsg3857 = latextend
        self.LON_MIN_epsg3857,self.LON_MAX_epsg3857 = lonextend

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857') 
        self.LON_MIN,self.LAT_MIN = transform(inProj, outProj, self.LON_MIN_epsg3857, self.LAT_MIN_epsg3857)
        self.LON_MAX,self.LAT_MAX = transform(inProj, outProj, self.LON_MAX_epsg3857, self.LAT_MAX_epsg3857)
        #-------------------------------------------------------------------



        #-------------------------------------------------------------------
        #MULTIPROCESSING
        #-------------------------------------------------------------------
        self.now = "PID" + str(os.getpid())     #creating a unique folder for each process using PID
        self.now = str(uuid.uuid4())            #creating a unique folder for each process using UUID

        if os.path.isdir(os.path.join(folder, self.now)):
            shutil.rmtree(os.path.join(folder, self.now))

        os.mkdir(os.path.join(folder, self.now))
        os.mkdir(os.path.join(folder, self.now, "env"))
        os.mkdir(os.path.join(folder, self.now, "output_files"))


        env_folder_seethathode = os.path.join("mesageo_elephant_project/elephant_project/", "experiment_setup_files","environment_seethathode", "Raster_Files_Seethathode_Derived", self.area[area_size], self.reso[spatial_resolution])
        shutil.copy(os.path.join(env_folder_seethathode, "DEM.tif"), os.path.join(folder, self.now, "env"))
        shutil.copy(os.path.join(env_folder_seethathode, "LULC.tif"), os.path.join(folder, self.now, "env"))
        shutil.copy(os.path.join(env_folder_seethathode, "population.tif"), os.path.join(folder, self.now, "env"))
        #-------------------------------------------------------------------




        self.folder_root = os.path.join(folder, self.now, "env")




        #-------------------------------------------------------------------
        #INITIALIZING ENVIRONMENT
        #-------------------------------------------------------------------
        # MODEL ENVIRONMENT VARIBLES
        self.DEM = self.DEM_study_area()
        self.LANDUSE = self.LANDUSE_study_area()
        self.FOOD = self.FOOD_MATRIX(self.prob_food_forest, self.prob_food_cropland)
        self.WATER = self.WATER_MATRIX()
        self.SLOPE = self.SLOPE_study_area()

        #self.initialize_road_network() 
        file = open(os.path.join(self.folder_root,'road_network.pkl'), 'rb')
        self.road_network = pickle.load(file)

        file = open(os.path.join(self.folder_root,'nodes_proj.pkl'), 'rb')
        self.nodes_proj = pickle.load(file)

        file = open(os.path.join(self.folder_root,'edges_proj.pkl'), 'rb')
        self.edges_proj = pickle.load(file)

        self.food_init = gdal.Open(os.path.join(self.folder_root, "food_matrix_"+ str(self.prob_food_forest) + "_" + str(self.prob_food_cropland) + "_.tif")).ReadAsArray()
        fid = os.path.join(self.folder_root, "crop_status.tif")
        self.crop_status = gdal.Open(fid).ReadAsArray().tolist() 

        self.plantation_proximity = self.proximity_from_plantation()
        self.forest_proximity = self.proximity_from_forest()
        self.water_proximity = self.proximity_from_water()
        self.AGRI_PLOTS, self.INFRASTRUCTURE = self.PROPERTY_MATRIX()

        #FOOD ATTRIBUTES
        self.food_in_forest_fixed = None
        self.food_in_cropland_fixed = None
        self.food_in_forest_variable = None
        self.food_in_cropland_variable = None

        #DAMAGE 
        self.crop_damage_incidents_fixed = None
        self.crop_damage_incidents_variable = None
        self.infrastructure_damage_incidents = None
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        #MODEL TIME VARIABLES
        #-------------------------------------------------------------------
        self.model_time = 1       #Tick counter
        self.model_day = 0          #Time elapsed in days
        self.model_hour = 0         #Time elapsed in hours
        self.model_minutes = 0      #Time elapsed in minutes
        self.hour_in_day = 0        #hour in a day(0 to 23)
        #-------------------------------------------------------------------




        self.abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}
        today = datetime.date(self.year, self.abbr_to_num[self.month], self.model_day + 1)
        start_day = datetime.date(self.year, 1, 1)
        self.num_days_elapsed = (today - start_day).days



        #Raster file details for agent coordinate tracking
        #-------------------------------------------------------------------
        LULC = os.path.join(self.folder_root,"LULC.tif")
        ds = gdal.Open(LULC, 0)
        arr = ds.ReadAsArray()
        self.xmin, self.xres, self.xskew, self.ymax, self.yskew, self.yres = ds.GetGeoTransform()
        self.row_size, self.col_size =  arr.shape
        #-------------------------------------------------------------------




        #For keeping track of Dead Human and Elephant agents
        #-------------------------------------------------------------------
        self.num_human_deaths = 0
        self.num_elephant_deaths = 0
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        #Landuse labels used in LULC
        self.landusetypes={"Deciduous Broadleaf Forest":1,"Cropland":2,"Built-up Land":3,"Mixed Forest":4,
                       "Shrubland":5,"Barren Land":6,"Fallow Land":7,"Wasteland":8,"Water Bodies":9,
                       "Plantations":10,"Aquaculture":11,"Mangrove Forest":12,"Salt Pan":13,"Grassland":14,
                       "Evergreen Broadlead Forest":15,"Deciduous Needleleaf Forest":16,
                       "Permanent Wetlands":17, "Snow and ice":18, "Evergreen Needleleaf Forest":19}
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        #self.schedule = BaseScheduler(self)        #Activation of agents in the order they were added 
        self.schedule = RandomActivation(self)      #Random activation of agents
        self.grid = GeoSpace()
        self.running='True'
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        if self.num_bull_elephants >= 1:
            with open('mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/temperature/hourly_temp_2010_without_juveniles_' + str(THRESHOLD) + '.pkl','rb') as f:
                self.hourly_temp = pickle.load(f)

        else:
            with open('mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/temperature/hourly_temp_2010_with_juveniles_' + str(THRESHOLD) + '.pkl','rb') as f:
                self.hourly_temp = pickle.load(f)   

        #print(len(self.hourly_temp))        

        self.update_hourly_temp()
        #-------------------------------------------------------------------




        #AGENT INITIALIZATION
        #-----------------------------------------------------------
        #SEQUENTIAL PROCESSING
        #-----------------------------------------------------------
        #start = timeit.default_timer()
        self.initialize_bull_elephants()
        #stop = timeit.default_timer()
        #execution_time = stop - start
        #print("Elephant agent initialization (bulls): "+str(execution_time)+" seconds") # It returns time in seconds

        #start = timeit.default_timer()
        self.initialize_herd_elephants()
        #stop = timeit.default_timer()
        #execution_time = stop - start
        #print("Elephant agent initialization (herds): "+str(execution_time)+" seconds") # It returns time in seconds
        
        #start = timeit.default_timer()
        self.initialize_human_agents()
        #stop = timeit.default_timer()
        #execution_time = stop - start
        #print("Human agent initialization: "+str(execution_time)+" seconds") # It returns time in seconds
        #-----------------------------------------------------------




        self.datacollector = DataCollector(model_reporters={
                                                "num_elephant_deaths": "num_elephant_deaths",
                                                "num_human_deaths": "num_human_deaths",
                                                "prob_food_forest": "prob_food_forest",
                                                "prob_food_cropland": "prob_food_cropland",
                                                "max_food_val_forest": "max_food_val_forest",
                                                "max_food_val_cropland": "max_food_val_cropland",
                                                "food_in_forest_fixed": "food_in_forest_fixed",
                                                "food_in_cropland_fixed": "food_in_cropland_fixed",
                                                "food_in_forest_variable": "food_in_forest_variable",
                                                "food_in_cropland_variable": "food_in_cropland_variable",
                                                "crop_damage_incidents_fixed": "crop_damage_incidents_fixed",
                                                "crop_damage_incidents_variable": "crop_damage_incidents_variable",
                                                "infrastructure_damage_incidents": "infrastructure_damage_incidents"},
                                            agent_reporters={
                                                "longitude": "shape.x", 
                                                "latitude": "shape.y",
                                                "fitness": "fitness",
                                                "aggression": "aggress_factor",
                                                "prob_thermo_reg": "prob_thermoregulation",
                                                "mode": "mode",
                                                "hour": "hour",
                                                "landuse": "landuse",
                                                "food_consumed": "food_consumed" ,
                                                "daily_dry_matter_intake": "daily_dry_matter_intake",
                                                "num_days_food_depreceation": "num_days_food_depreceation",
                                                "water_source_proximity": "water_source_proximity",
                                                "num_days_water_source_visit": "num_days_water_source_visit"
                                                })

        
        #Collect data at the start of the simulation
        self.datacollector.collect(self)
        self.create_traj_plot()

        #ADDITIONAL DATA COLLECTION
        self.CONFLICT_LOCATIONS = []
        self.INTERACTION_LOCATIONS = []

        # #use geopands to read geojson file
        # self.landuse_types = gpd.read_file("/home/anjalip/Documents/GitHub/abm-elephant-project/mesageo_elephant_project/elephant_project/geojson_files/lulc.geojson")
        # landuse_class_creator = AgentCreator(landuse_classes, {"model":self})

        # self.landuse_types["unique_id"] = self.landuse_types.index

        # for landuse in [6]:

        #     # save to geojson file
        #     # self.landuse_types[self.landuse_types["DN"] == landuse].to_file("/home/anjalip/Documents/GitHub/abm-elephant-project/mesageo_elephant_project/elephant_project/geojson_files/landuse_" + str(landuse) + ".geojson", driver='GeoJSON')

        #     newagents = landuse_class_creator.from_GeoDataFrame(self.landuse_types[self.landuse_types["DN"] == landuse])

        #     for newagent in newagents:
        #         newagent.unique_id = "landuse_" + str(landuse) + "_" + str(newagent.unique_id)
        #         print(newagent.unique_id)
        #         self.grid.add_agents(newagent)
                # self.schedule.add(newagent)
    #-------------------------------------------------------------------
    def initialize_bull_elephants(self, **kwargs):
        """Initialize the elephant agents"""

        # initialization_function = "random initialization"

        # if initialization_function == "random initialization":
        #     #random initialization
        #     coord_lat, coord_lon = self.elephant_distribution_random_init_forest()

        # elif initialization_function == "close to fringe":
        #     #close to fringe
        #     coord_lat, coord_lon = self.elephant_distribution_close_to_fringe(distance_l=kwargs["lower"], distance_u=kwargs["upper"])

        # else:
        #     #distribution considering landuse, elevation and proximity to fringe
        #     coord_lat, coord_lon = self.elephant_distribution_modified()

        # Writing the init points to a file
        # coord_lat, coord_lon = coord_lat.reset_index(drop=True), coord_lon.reset_index(drop=True)
        # dict = {}
        # for i in range(1,11):
        #     id = "id_" + str(i)
        #     dict[id]= [coord_lat[i-1], coord_lon[i-1]]
        # import json
        # json.dump(dict, open("initialisation_elephants/elev800prox2000.json", 'w'))

        #close to fringe
        coord_lat = [1048779.5968364885]
        coord_lon = [8575881.616495656]

        for i in range(0, self.num_bull_elephants):    #initializing bull elephants

            elephant=AgentCreator(Bull_Elephant,{"model":self})  
            this_x = np.array(coord_lon)[i] + self.random.randint(-10,10)
            this_y = np.array(coord_lat)[i] + self.random.randint(-10,10)
            newagent = elephant.create_agent(Point(this_x,this_y), "bull_"+str(i))
            newagent.herdID = 0   #variables for assigning the social structure of the elephants. herdID = 0 corresponds to solitary bulls.
            newagent.Leader = True   #bull elephants are always leaders
            newagent.sex = "Male"
            newagent.age = self.random.randrange(15,60) 
            newagent.conflict_neighbor = None
            
            #Assign body weight of the elephant agent depending on the sex and age
            newagent.body_weight = self.assign_body_weight_elephants(newagent.age, newagent.sex)

            #Assign daily dry matter intake depending on the body weight
            newagent.daily_dry_matter_intake = self.assign_daily_dietary_requiremnt(newagent.body_weight)

            self.grid.add_agents(newagent)
            self.schedule.add(newagent)
    #-------------------------------------------------------------------
    def initialize_herd_elephants(self, **kwargs):
        """Initialize the elephant agents"""

        #close to fringe
        coord_lat = [1048779.5968364885]
        coord_lon = [8575881.616495656]

        for herd_id in range(1, self.num_herds + 1):    #initializing herds

            #print("Herd ID:", herd_id)

            num_elephants = self.random.randint(5,20)
            #print("Number of elephants in the herd:", num_elephants)

            this_x = np.array(coord_lon)[0] + self.random.randint(-10,10)
            this_y = np.array(coord_lat)[0] + self.random.randint(-10,10)

            agent_list = []
            agent_age_list = []

            for ele_id in range(num_elephants):

                elephant = AgentCreator(Herd_Elephant,{"model":self})  
                newagent = elephant.create_agent(Point(this_x,this_y), "herd_"+str(herd_id)+"_ele_"+str(ele_id))
                newagent.herdID = herd_id   #variables for assigning the social structure of the elephants. herdID = 0 corresponds to solitary bulls.
                newagent.age = self.random.randrange(1,60) 
                newagent.conflict_neighbor = None
                agent_list.append(newagent)
                agent_age_list.append(newagent.age)

            #print(agent_age_list, agent_list)
            agent_age_list = np.array(agent_age_list)
            agent_list = np.array(agent_list)
            idx = np.argsort(agent_age_list)
            #print(agent_age_list[idx], agent_list[idx])

            agent_age_list = agent_age_list[idx]
            agent_list = agent_list[idx]

            agent_list[-1].Leader = True
            agent_list[-1].sex = "Female"
            #Assign body weight of the elephant agent depending on the sex and age
            agent_list[-1].body_weight = self.assign_body_weight_elephants(agent_list[-1].age, agent_list[-1].sex)
            #Assign daily dry matter intake depending on the body weight
            agent_list[-1].daily_dry_matter_intake = self.assign_daily_dietary_requiremnt(agent_list[-1].body_weight)
            agent_list[-1].follow = None
            agent_list[-1].followers = agent_list[:-1]
            self.grid.add_agents(agent_list[-1])
            self.schedule.add(agent_list[-1])

            for agt in agent_list[:-1]:
                if agt.age < 15:
                    if self.random.uniform(0,1) < 0.5:
                        agt.sex = "Female"
                    else:
                        agt.sex = "Male"
                else:
                    agt.sex = "Female"

                agt.Leader = False
                #Assign body weight of the elephant agent depending on the sex and age
                agt.body_weight = self.assign_body_weight_elephants(agt.age, agt.sex)
                #Assign daily dry matter intake depending on the body weight
                agt.daily_dry_matter_intake = self.assign_daily_dietary_requiremnt(agt.body_weight)  

            food_reqmt_herd = 0

            for agt in agent_list:
                food_reqmt_herd += agt.daily_dry_matter_intake

            #print("Total food requirement:", food_reqmt_herd, "Average food requirement: ", food_reqmt_herd/len(agent_list))
            agent_list[-1].daily_dry_matter_intake = food_reqmt_herd

        return
    #-------------------------------------------------------------------
    def elephant_distribution_random_init_forest(self):
        """ Function to return the distribution of elephants within the study area"""

        #The function is used for the distribution of elephant agents into the area classified as the Evergreen forest within the study area
        #The function returns a dataframe with latitude and longitudes in the EPSG:3857 reference system to be used in ABM directly

        lat=[]
        lon=[]

        for i in range(0,self.row_size):
            for j in range(0,self.col_size):
                if self.LANDUSE[i][j]==self.landusetypes["Evergreen Broadlead Forest"]: 
                    x,y=self.pixel2coord(i,j)
                    lon.append(x)
                    lat.append(y)

        coordinates=pd.DataFrame(np.concatenate((np.array(lat).reshape(-1,1),np.array(lon).reshape(-1,1)),axis=1))
        indices = self.random.sample(range(0, len(coordinates)),self.num_bull_elephants)
        return coordinates.iloc[indices][0],coordinates.iloc[indices][1]
    #----------------------------------------------------------------------------------------------------- 
    def elephant_distribution_close_to_fringe(self, distance_l=10, distance_u=50):
        """ Function to return the distribution of elephants within the study area"""

        proximity_map_1 = gdal.Open(os.path.join(self.folder_root, "proximity_from_population.tif")).ReadAsArray()
        #proximity_map_2 = gdal.Open(os.path.join(self.folder_root, "proximity_from_plantation_builtup_shrubland.tif")).ReadAsArray()
        proximity_map_2 = gdal.Open(os.path.join(self.folder_root, "proximity_from_plantation.tif")).ReadAsArray()

        lat=[]
        lon=[]

        for i in range(0,self.row_size):
            for j in range(0,self.col_size):
                if proximity_map_1[i,j] > distance_l/self.xres and proximity_map_1[i,j] < distance_u/self.xres:
                    if proximity_map_2[i,j] > distance_l/self.xres and proximity_map_2[i,j] < distance_u/self.xres:
                        if self.LANDUSE[i][j] != 3 or self.LANDUSE[i][j] != 10 or self.LANDUSE[i][j] != 5:
                            x,y=self.pixel2coord(i,j)
                            lon.append(x)
                            lat.append(y)

        coordinates=pd.DataFrame(np.concatenate((np.array(lat).reshape(-1,1),np.array(lon).reshape(-1,1)),axis=1))
        indices = self.random.sample(range(0, len(coordinates)),self.numberofbullelephants)
        return coordinates.iloc[indices][0],coordinates.iloc[indices][1]
    #----------------------------------------------------------------------------------------------------- 
    def elephant_distribution_modified(self, elevation_l = 600, elevation_u = 1200, landuse_codes = [15], proximity_from_fringe_l = 2000, proximity_from_fringe_u = 4000):
        """ Function to return the distribution of elephants within the study area"""

        #elevation ---> elephants agents are not distributed in locations with altitude above this value
        #landuse_codes ---> elephants agents are distributed in locations with these landuse codes
        #proximity_from_fringe (unit: metres) ---> Proximity maps have been created from Land use types: "Plantations": 10, "Built-up Land": 3, "Shrubland": 5.

        #The function is used for the distribution of elephant agents within the study area
        #The function returns a dataframe with latitude and longitudes in the EPSG:3857 reference system to be used in ABM directly

        num_cells_l = int(proximity_from_fringe_l/self.xres)
        num_cells_u = int(proximity_from_fringe_u/self.xres)

        path = os.path.join(self.folder_root, "DEM.tif")
        DEM = gdal.Open(path).ReadAsArray()

        LOCATIONS_DEM = np.zeros_like(DEM)
        LOCATIONS_PROXIMITY = np.zeros_like(DEM)

        DEM = gdal.Open(path).ReadAsArray().tolist()

        for i in range(0, self.row_size):
            for j in range(0, self.col_size):
                if DEM[i][j] > elevation_l and DEM[i][j] < elevation_u:
                    LOCATIONS_DEM[i,j] = 1

        path = os.path.join(self.folder_root, "proximity_from_plantation_builtup_shrubland.tif")
        proximity = gdal.Open(path).ReadAsArray()

        for i in range(0, self.row_size):
            for j in range(0, self.col_size):
                if proximity[i][j] > num_cells_l*self.xres and proximity[i][j] < num_cells_u*self.xres:
                    LOCATIONS_PROXIMITY[i,j] = 1

        lat=[]
        lon=[]

        for k in landuse_codes:
            for i in range(0, self.row_size):
                for j in range(0, self.col_size):
                    if self.LANDUSE[i][j]==k: 
                        if LOCATIONS_DEM[i, j] == 1 and LOCATIONS_PROXIMITY[i, j] == 1:
                            x,y=self.pixel2coord(i,j)
                            lon.append(x)
                            lat.append(y)

        coordinates=pd.DataFrame(np.concatenate((np.array(lat).reshape(-1,1),np.array(lon).reshape(-1,1)),axis=1))

        try:
            indices = np.random.choice(range(0, len(coordinates)-1), self.num_bull_elephants)

        except:
            print("Error: check parameters passed. No coordinates available to distribute elephant agents")
            return None, None

        return coordinates.iloc[indices][0],coordinates.iloc[indices][1]
    #-----------------------------------------------------------------------------------------------------
    def assign_body_weight_elephants(self,age,sex):
        """ The function returns the body weight of the elephant agents depending on the age and sex"""
        #Growth in the Asian elephant. R SUKUMAR, N V JOSHI and V KRISHNAMURTHY. Proc. Indian Acad. Sci. (Anim. Sci.), VoL 97. No.6. November 1988, pp. 561-571
        #von Bertalanffy functions
        #Body weight is calculayted in kg

        if sex=="Male":
            body_weight = 4000*(1-np.exp(-0.149*(age+3.16)))**3

        else:
            body_weight = 3055*(1-np.exp(-0.092*(age+6.15)))**3

        return body_weight
    #-----------------------------------------------------------------------------------------------------
    def assign_daily_dietary_requiremnt(self,body_weight):
        """ The function assigns the daily dietary requirement based on the body weight"""

        #The wild adult Asian elephant's daily dry matter intake: 1.5% to 1.9% of body weight
        #Source: Nutrition adivisary group handbook. Elephants: nutrition and dietary husbandry

        daily_dry_matter_intake = np.random.uniform(1.5,1.9)*body_weight/100

        return daily_dry_matter_intake
    #-----------------------------------------------------------------------------------------------------
    def initialize_human_agents(self):
        """ The function initializes human agents"""

        coord = self.co_ordinates_residential()
        coord = np.array(coord[["0", "1", "2"]])

        coord_non_residential = self.co_ordinates_non_residential()
        coord_non_residential = np.array(coord_non_residential[["0", "1"]])

        coord_agricultural = self.co_ordinates_agricultural_plots()
        coord_agricultural = np.array(coord_agricultural[["0", "1"]])

        init_file = open(os.path.join(os.getcwd(), "mesageo_elephant_project/elephant_project/data/", "model_init_files_humans_and_bulls", "model_run_" + _model_id_, "init_files", "population_init.json"))
        init_data = json.load(init_file)
        num_CAL = init_data["num_CAL"]    
        num_OW = init_data["num_OW"] 
        num_HB = init_data["num_HB"] 
        num_RWHB = init_data["num_RWHB"] 
        num_RWP = init_data["num_RWP"] 
        num_CP = init_data["num_CP"] 

        for k in range(0,num_CAL):
            num=self.random.choices(coord[:], weights=coord[:,2])  
            this_x=num[0][1]    #longitude
            this_y=num[0][0]    #latitude
            humans=AgentCreator(Cultivators_Agricultural_labourers,{"model":self})
            num=self.random.choices(coord_agricultural[:])
            target_x=num[0][1]
            target_y=num[0][0]
            newagent=humans.create_agent(Point(this_x,this_y),"CAL:"+str(k))
            newagent.target_lon = target_x
            newagent.target_lat= target_y
            newagent.initialize_target_destination_nodes()
            newagent.initialize_distance_to_target()
            newagent.conflict_neighbor = None
            self.grid.add_agents(newagent)
            self.schedule.add(newagent)

        for k in range(0,num_OW):
            num=self.random.choices(coord[:], weights=coord[:,2])  
            this_x=num[0][1]    #longitude
            this_y=num[0][0]    #latitude
            humans=AgentCreator(Other_workers,{"model":self}) 
            num=self.random.choices(coord_non_residential[:])
            target_x=num[0][1]
            target_y=num[0][0]
            newagent=humans.create_agent(Point(this_x,this_y),"OW:"+str(k))
            newagent.target_lon = target_x
            newagent.target_lat= target_y
            newagent.initialize_target_destination_nodes()
            newagent.initialize_distance_to_target()
            newagent.conflict_neighbor = None
            self.grid.add_agents(newagent)
            self.schedule.add(newagent)

        for k in range(0,num_HB):
            num=self.random.choices(coord[:], weights=coord[:,2])  
            this_x=num[0][1]    #longitude
            this_y=num[0][0]    #latitude
            humans=AgentCreator(HomeBound,{"model":self}) 
            target_x=None
            target_y=None
            newagent=humans.create_agent(Point(this_x,this_y),"HB:"+str(k))
            newagent.target_lon = target_x
            newagent.target_lat= target_y
            newagent.initialize_target_destination_nodes()
            newagent.initialize_distance_to_target()
            newagent.conflict_neighbor = None
            self.grid.add_agents(newagent)
            self.schedule.add(newagent)

        for k in range(0,num_RWHB):
            num=self.random.choices(coord[:], weights=coord[:,2])  
            this_x=num[0][1]    #longitude
            this_y=num[0][0]    #latitude
            humans=AgentCreator(RandomWalkers_homebound,{"model":self}) 
            target_x=None
            target_y=None
            newagent=humans.create_agent(Point(this_x,this_y),"RWHB:"+str(k))
            newagent.target_lon = target_x
            newagent.target_lat= target_y
            newagent.initialize_target_destination_nodes()
            newagent.initialize_distance_to_target()
            newagent.conflict_neighbor = None
            self.grid.add_agents(newagent)
            self.schedule.add(newagent)

        for k in range(0,num_RWP):
            num=self.random.choices(coord[:], weights=coord[:,2])  
            this_x=num[0][1]    #longitude
            this_y=num[0][0]    #latitude
            humans=AgentCreator(RandomWalkers_perpetual,{"model":self}) 
            target_x=None
            target_y=None
            newagent=humans.create_agent(Point(this_x,this_y),"RWP:"+str(k))
            newagent.target_lon = target_x
            newagent.target_lat= target_y
            newagent.initialize_target_destination_nodes()
            newagent.initialize_distance_to_target()
            newagent.conflict_neighbor = None
            self.grid.add_agents(newagent)
            self.schedule.add(newagent)
            
        for k in range(0,num_CP):
            num=self.random.choices(coord[:], weights=coord[:,2])  
            this_x=num[0][1]    #longitude
            this_y=num[0][0]    #latitude
            humans=AgentCreator(commuters_perpetual,{"model":self}) 
            num=self.random.choices(coord_non_residential[:])
            target_x=num[0][1]
            target_y=num[0][0]
            newagent=humans.create_agent(Point(this_x,this_y),"CP:"+str(k))
            newagent.target_lon = target_x
            newagent.target_lat= target_y
            newagent.initialize_target_destination_nodes()
            newagent.initialize_distance_to_target()
            newagent.conflict_neighbor = None
            self.grid.add_agents(newagent)
            self.schedule.add(newagent)

        coord_guard = self.guard_agent_dist_coords()
        coord_guard = np.array(coord_guard[["lat", "lon"]])

        # def return_landuse_map():
        #     ds = gdal.Open("experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif")
        #     data = ds.ReadAsArray()
        #     data = np.flip(data, axis=0)
        #     row_size, col_size = data.shape
        #     xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

        #     data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

        #     for i in range(1,10):
        #         data[data == data_value_map[i]] = i

        #     fig_background, ax_background = plt.subplots(figsize = (10,10))
        #     ax_background.yaxis.set_inverted(True)

        #     outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        #     LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
        #     LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

        #     #map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, projection='merc', resolution='l')
        #     map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        #     #setting cmap
        #     levels = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]
        #     clrs = ["greenyellow","mediumpurple","turquoise","plum","black", "blue", "yellow", "forestgreen", "mediumseagreen"] 
        #     cmap, norm = colors.from_levels_and_colors(levels, clrs)

        #     map.imshow(data, cmap = cmap, norm=norm, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])

        #     map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
        #     map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])
        #     return map, ax_background

        # landuse_background, ax_background = return_landuse_map()

        for k in range(0,self.num_guard_agents):
            num=self.random.choices(coord_guard[:])  
            this_x=num[0][1]    #longitude
            this_y=num[0][0]    #latitude
            humans=AgentCreator(Guard_agents,{"model":self}) 
            target_x=None
            target_y=None
            newagent=humans.create_agent(Point(this_x,this_y),"guard:"+str(k))
            newagent.target_lon = target_x
            newagent.target_lat = target_y
            newagent.initialize_target_destination_nodes()
            newagent.initialize_distance_to_target()
            newagent.conflict_neighbor = None
            self.grid.add_agents(newagent)
            self.schedule.add(newagent)

            # outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
            # longitude, latitude = transform(inProj, outProj, this_x, this_y)
            # x, y = landuse_background(longitude,latitude)
            # ax_background.scatter(x, y, 25, marker='o', color='red', zorder=2) 

        #plt.savefig(os.path.join(folder, batch_folder, self.now,"Guard_agent_locations.png"), dpi=300)

        return
    #-----------------------------------------------------------------------------------------------------
    def co_ordinates_residential(self):
        """ Function returns the locations of the buildings that are designated as residential in the study area"""

        villages=["Thannithodu","Perunad","Chittar-Seethathodu"]

        lat=[]
        lon=[]
        num_household=[]

        for village in villages:
            path = os.path.join("mesageo_elephant_project/elephant_project", "experiment_setup_files","environment_seethathode", "shape_files", village+"_residential.tif")

            distribution = gdal.Open(path).ReadAsArray()

            ds = gdal.Open(path)
            xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

            m,n = distribution.shape

            for i in range(0,m):
                for j in range(0,n):
                    if distribution[i,j] > 0:  #LULC code 15: Evergreen forests
                        x, y = self.pixel2coord_raster(i, j, xmin, ymax, xres, yres)
                        lon.append(x)
                        lat.append(y)
                        num_household.append(distribution[i,j])


        lat = np.array(lat).reshape(-1,1)
        lon = np.array(lon).reshape(-1,1)
        num_household = np.array(num_household).reshape(-1,1)
        coords = np.concatenate((lat,lon),axis=1)
        coords = np.concatenate((coords,num_household),axis=1)
        coords = pd.DataFrame(coords, columns = ["0", "1", "2"])
        return coords
    #---------------------------------------------------------------------------------------------------------
    def co_ordinates_non_residential(self):
        """ Function returns the locations of the buildings that are designated as non-residential in the study area"""

        villages=["Thannithodu","Perunad","Chittar-Seethathodu"]
        lat=[]
        lon=[]
        num_household=[]

        for village in villages:

            path = os.path.join("mesageo_elephant_project/elephant_project", "experiment_setup_files","environment_seethathode", "shape_files", village+"_non_residential.tif")

            distribution = gdal.Open(path).ReadAsArray()

            ds = gdal.Open(path)
            xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

            m,n = distribution.shape

            for i in range(0,m):
                for j in range(0,n):
                    if distribution[i,j]>0:  
                        x,y=self.pixel2coord_raster(i, j, xmin, ymax, xres, yres)
                        lon.append(x)
                        lat.append(y)
                        num_household.append(distribution[i,j])

        lat = np.array(lat).reshape(-1,1)
        lon = np.array(lon).reshape(-1,1)
        num_household = np.array(num_household).reshape(-1,1)
        coords = np.concatenate((lat,lon),axis=1)
        coords = np.concatenate((coords,num_household),axis=1)
        coords = pd.DataFrame(coords, columns = ["0", "1", "2"])
        return coords
    #---------------------------------------------------------------------------------------------------------
    def co_ordinates_agricultural_plots(self):
        """ Function returns the ID and location of the agricultural plots """

        lat=[]
        lon=[]
        ID=[]

        path = os.path.join("mesageo_elephant_project/elephant_project", "experiment_setup_files","environment_seethathode","Raster_Files_Seethathode_Derived","area_1100sqKm","reso_30x30","LULC.tif")

        distribution = gdal.Open(path).ReadAsArray()

        m,n = distribution.shape

        for i in range(0,m):
            for j in range(0,n):
                if distribution[i,j]==10:  
                    x,y=self.pixel2coord(i,j)
                    lon.append(x)
                    lat.append(y)
                    ID.append(distribution[i,j])

        lat = np.array(lat).reshape(-1,1)
        lon = np.array(lon).reshape(-1,1)
        ID = np.array(ID).reshape(-1,1)
        coords = np.concatenate((lat,lon),axis=1)
        coords = np.concatenate((coords,ID),axis=1)
        coords = pd.DataFrame(coords, columns = ["0", "1", "2"])
        return coords
    #-----------------------------------------------------------------------------------------------------
    def guard_agent_dist_coords(self):

        lat=[]
        lon=[]

        path = os.path.join("mesageo_elephant_project/elephant_project", "experiment_setup_files","environment_seethathode","Raster_Files_Seethathode_Derived","area_1100sqKm","reso_30x30","proximity_map_4_5_15.tif")
        proximity_map = gdal.Open(path).ReadAsArray()

        path = os.path.join("mesageo_elephant_project/elephant_project", "experiment_setup_files","environment_seethathode","Raster_Files_Seethathode_Derived","area_1100sqKm","reso_30x30","LULC.tif")
        LULC = gdal.Open(path).ReadAsArray()

        dist_coords = np.zeros_like(LULC)
        row_size, col_size = LULC.shape

        for i in range(0, row_size):
            for j in range(0, col_size):
                if LULC[i,j] == 10 and proximity_map[i,j]<3:
                    dist_coords[i,j] = 1

        lat = []
        lon = []

        raster = rio.open(os.path.join("mesageo_elephant_project/elephant_project", "experiment_setup_files","environment_seethathode","Raster_Files_Seethathode_Derived","area_1100sqKm","reso_30x30","proximity_map_4_5_15.tif"))

        for i in range(0,row_size):
            for j in range(0,col_size):
                if dist_coords[i,j] == 1:  #LULC code 15: Evergreen forests
                    x, y = raster.xy(i,j)
                    lon.append(x)
                    lat.append(y)

        #print(len(lon), len(lat))

        lat = np.array(lat).reshape(-1,1)
        lon = np.array(lon).reshape(-1,1)
        coords = np.concatenate((lat,lon),axis=1)
        coords = pd.DataFrame(coords, columns = ["lat", "lon"])
        coords = coords[coords["lon"] > 8566000]
        return coords
    #-----------------------------------------------------------------------------------------------------
    def DEM_study_area(self):
        """ Returns the digital elevation model of the study area"""

        fid = os.path.join(self.folder_root, "DEM.tif")

        DEM = gdal.Open(fid).ReadAsArray()  
        return DEM.tolist()  #Conversion to list so that the object becomes json serializable
    #-----------------------------------------------------------------------------------------------------
    def LANDUSE_study_area(self):
        """ Returns the landuse model of the study area"""

        fid = os.path.join(self.folder_root, "LULC.tif")

        LULC = gdal.Open(fid).ReadAsArray()  
        return LULC.tolist()  #Conversion to list so that the object becomes json serializable
    #-----------------------------------------------------------------------------------------------------
    def FOOD_MATRIX(self, prob_food_in_forest, prob_food_in_cropland):
        """ Returns the food matrix model of the study area"""
        fid = os.path.join(self.folder_root, "food_matrix_"+ str(prob_food_in_forest) + "_" + str(prob_food_in_cropland) + "_.tif")
        FOOD = gdal.Open(fid).ReadAsArray()  
        return FOOD.tolist()  
    #-----------------------------------------------------------------------------------------------------
    def WATER_MATRIX(self):
        """ Returns the water matrix model of the study area"""
        fid = os.path.join(self.folder_root ,"water_matrix_"+ str(self.prob_water) +"_.tif")
        WATER = gdal.Open(fid).ReadAsArray()  
        return WATER.tolist() 
    #-----------------------------------------------------------------------------------------------------
    def SLOPE_study_area(self):
        """ Returns the slope model of the study area"""

        fid = os.path.join(self.folder_root, "slope.tif")

        slope = gdal.Open(fid).ReadAsArray()  
        return slope.tolist()  #Conversion to list so that the object becomes json serializable
    #-----------------------------------------------------------------------------------------------------
    def initialize_road_network(self):

        #Road network in the study area: used for commute by the human agents

        north, south, east, west = self.LAT_MIN,self.LAT_MAX,self.LON_MIN,self.LON_MAX
        self.road_network=ox.graph.graph_from_bbox(north, south, east, west, network_type='all', simplify=True, retain_all=False, truncate_by_edge=False, clean_periphery=True, custom_filter=None)
        self.road_network = ox.project_graph(self.road_network,"epsg:3857")
        self.nodes_proj, self.edges_proj = ox.graph_to_gdfs(self.road_network, nodes=True, edges=True)

        # file = open('model_init_files_01/model_run_00/osmnx/road_network.pkl', 'wb')
        # pickle.dump(self.road_network, file)

        # file = open('model_init_files_01/model_run_00/osmnx/nodes_proj.pkl', 'wb')
        # pickle.dump(self.nodes_proj, file)

        # file = open('model_init_files_01/model_run_00/osmnx/edges_proj.pkl', 'wb')
        # pickle.dump(self.edges_proj, file)

        return
    #-----------------------------------------------------------------------------------------------------
    def proximity_from_plantation(self):
        """ Returns the proximity matrix from the plantations"""

        fid = os.path.join(self.folder_root, "proximity_from_plantation.tif")

        plantation_proximity = gdal.Open(fid).ReadAsArray()  
        return plantation_proximity.tolist() #Conversion to list so that the object becomes json serializable
    #-----------------------------------------------------------------------------------------------------
    def proximity_from_forest(self):
        """ Returns the proximity matrix from the evergreen broadleaf forest"""

        fid = os.path.join(self.folder_root, "proximity_from_forest.tif")

        forest_proximity = gdal.Open(fid).ReadAsArray()  
        return forest_proximity.tolist() #Conversion to list so that the object becomes json serializable
    #-----------------------------------------------------------------------------------------------------
    def proximity_from_water(self):
        """ Returns the proximity matrix from the plantations"""

        fid = os.path.join(self.folder_root, "prox_water_matrix_" + str(self.prob_water) + "_.tif")

        water_proximity = gdal.Open(fid).ReadAsArray()
        return water_proximity.tolist() 
    #-----------------------------------------------------------------------------------------------------
    def PROPERTY_MATRIX(self):
        """ Returns the infrastructure and crop matrix of the study area"""

        population = gdal.Open(os.path.join(self.folder_root, "population.tif")).ReadAsArray()
        path = os.path.join(self.folder_root, "crop_status.tif")
        distribution = gdal.Open(path).ReadAsArray()
        m,n = distribution.shape

        plots = np.zeros_like(distribution)
        infrastructure = np.zeros_like(distribution)

        for i in range(0,m):
            for j in range(0,n):
                if distribution[i,j] == 1 or distribution[i,j] == 2:  
                    plots[i, j] = 1

                if population[i,j] != 0:  
                    infrastructure[i, j] = 1

        return plots.tolist(), infrastructure.tolist()
    #-----------------------------------------------------------------------------------------------------
    def get_indices(self, lon, lat):
        """
        Gets the row (i) and column (j) indices in an array for a given set of coordinates.
        return:    row (i) and column (j) indices
        """

        i = int(np.floor((self.ymax-lat) / -self.yres))
        j = int(np.floor((lon-self.xmin) / self.xres))

        if i > self.row_size - 1:
            i = self.row_size - 1

        elif i < 0:
            i = 0

        if j > self.col_size - 1:
            j = self.col_size - 1

        elif j < 0:
            j = 0
            
        #reurns row, col
        return i, j
    #-----------------------------------------------------------------------------------------------------
    def pixel2coord(self, row, col):
        """
        Gets the lon and lat corrsponding to the row and column indices.
        """

        lon = self.xres * 0.5  + self.xmin + col * self.xres
        lat = self.yres * 0.5  + self.ymax + row * self.yres

        return(lon, lat)
    #-----------------------------------------------------------------------------------------------------
    def pixel2coord_raster(self, row, col, xmin, ymax, xres, yres):
        """
        Gets the lon and lat corrsponding to the row and column indices.
        """

        lon = xres * 0.5  + xmin + col * xres
        lat = yres * 0.5  + ymax + row * yres

        return(lon, lat)
    #-----------------------------------------------------------------------------------------------------
    def no_human_disturbance(self):
        #Simulation devoid of human disturbance
        self.human_disturbance = 0
        return
    #-----------------------------------------------------------------------------------------------------
    def update_human_disturbance_explict(self):
        #6am to 6pm: high disturbance
        #6pm to 6am: low disturbance
        if self.hour_in_day >= 6 and self.hour_in_day <= 18:
            self.human_disturbance = self.disturbance_tolerance + np.random.randint(5,10)

        else:
            self.human_disturbance = self.disturbance_tolerance - np.random.randint(5,10)

        return
    #-----------------------------------------------------------------------------------------------------
    def update_human_disturbance_implicit(self):
        #model:from the number of active agents

        self.human_disturbance = 0

        for agent in self.schedule.agents:
            if type(agent) is Cultivators_Agricultural_labourers:
                dist = agent.distance(agent.shape.y, agent.home_lat, agent.shape.x, agent.home_lon)

            elif type(agent) is Other_workers:
                dist = agent.distance(agent.shape.y, agent.home_lat, agent.shape.x, agent.home_lon)

            elif type(agent) is RandomWalkers_homebound:
                dist = agent.distance(agent.shape.y, agent.home_lat, agent.shape.x, agent.home_lon)

            else:
                dist = 0

            if dist > 100:
                self.human_disturbance += 1

        return
    #-----------------------------------------------------------------------------------------------------
    def update_food_matrix_constant(self):

        FOOD = np.array(self.FOOD)
        crop_status = np.array(self.crop_status)

        self.food_in_forest_fixed = sum(FOOD[crop_status == 3])
        self.food_in_cropland_fixed = sum(FOOD[crop_status == 1])
        self.food_in_forest_variable = sum(FOOD[crop_status == 4]) 
        self.food_in_cropland_variable = sum(FOOD[crop_status == 2])

        self.FOOD = FOOD.tolist() 
        self.crop_status = crop_status.tolist()

        #-------------------save the food matrix-------------------#
        # source = os.path.join(self.folder_root, "LULC.tif")
        # with rio.open(source) as src:
        #     ras_meta = src.profile

        # loc = os.path.join(folder, self.now, "env", "food_matrix__" + str( self.model_day) + "__" + ".tif")
        # with rio.open(loc, 'w', **ras_meta) as dst:
        #     dst.write(np.array(self.FOOD).reshape(self.row_size, self.col_size), 1)          
        #---------------------------------------------------------#

        return
    #-----------------------------------------------------------------------------------------------------
    def update_prob_drink_water(self):

        if self.season == "dry":
            self.prob_drink_water = self.prob_drink_water_dry
        
        else:
            self.prob_drink_water = self.prob_drink_water_wet

        return
    #----------------------------------------------------------------------------------------------------
    def update_season(self):

        if self.model_day >= 0 and self.model_day <= 120:
            self.season = "dry"

        return
    #----------------------------------------------------------------------------------------------------
    def update_hourly_temp(self):

        if self.model_hour == 0:
            self.temp = self.hourly_temp[self.model_hour + self.num_days_elapsed*24]
            # print("days elapsed:", self.num_days_elapsed)
            # print("model hour:", self.model_hour)
            # print("index:", self.model_hour + self.num_days_elapsed*24)

        elif self.model_time%12 == 0:
            self.temp = self.hourly_temp[self.model_hour + self.num_days_elapsed*24]
            # print("days elapsed:", self.num_days_elapsed)
            # print("model hour:", self.model_hour)
            # print("index:", self.model_hour + self.num_days_elapsed*24)

        new_dims = []
        for original_length, new_length in zip((7,7), (self.row_size, self.col_size)):
            new_dims.append(np.linspace(0, original_length-1, new_length))

        coords = np.meshgrid(*new_dims, indexing='ij')
        self.temp = map_coordinates(self.temp, coords)

        return
    #----------------------------------------------------------------------------------------------------
    def return_indices_temperature_matrix(self, lat, lon):
        """
        Gets the row (i) and column (j) indices in an array for a given set of coordinates.
        return:    row (i) and column (j) indices
        """

        xmin, xres, xskew, ymax, yskew, yres = (8554728.360406002, 4738.0, 0.0, 1059194.12664853, 0.0, -4738.000000000016)
        row_size = 7
        col_size = 7

        i = int(np.floor((ymax-lat) / -yres))
        j = int(np.floor((lon-xmin) / xres))

        if i > row_size - 1:
            i = row_size - 1

        elif i < 0:
            i = 0

        if j > col_size - 1:
            j = col_size - 1

        elif j < 0:
            j = 0
            
        #reurns row, col
        return i, j
    #----------------------------------------------------------------------------------------------------
    def return_landuse_map(self):

        ds = gdal.Open("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif")
        data = ds.ReadAsArray()
        data = np.flip(data, axis=0)
        row_size, col_size = data.shape
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

        data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

        for i in range(1,10):
            data[data == data_value_map[i]] = i

        fig_background, ax_background = plt.subplots(figsize = (10,10))
        ax_background.yaxis.set_inverted(True)

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
        LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

        #map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, projection='merc', resolution='l')
        map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        #setting cmap
        levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        clrs = ["greenyellow","mediumpurple","turquoise", "plum", "black", "blue", "yellow", "mediumseagreen", "forestgreen"] 
        cmap, norm = colors.from_levels_and_colors(levels, clrs)

        map.imshow(data, cmap = cmap, norm=norm, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])

        map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
        map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])
        
        cbar = plt.colorbar(ticks=[1,2,3,4,5,6,7,8,9],fraction=0.046, pad=0.04)
        cbar.ax.set_yticks(ticks=[1.5,2.5,3,4,5,6,7,8,9]) 
        cbar.ax.set_yticklabels(["Deciduous Broadleaf Forest","Built-up Land","Mixed Forest","Shrubland","Barren Land","Water Bodies","Plantations","Grassland","Broadleaf evergreen forest"])
        
        return map, fig_background, ax_background
    #----------------------------------------------------------------------------------------------------
    def create_traj_plot(self):
        self.landuse_background, self.fig_background_01, self.ax_background_01  = self.return_landuse_map()
    #----------------------------------------------------------------------------------------------------
    def plot_ele_traj(self, longitude, latitude):
        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        longitude, latitude = transform(inProj, outProj, longitude, latitude)
        x_new, y_new = self.landuse_background(longitude,latitude)
        C = np.arange(len(x_new))
        nz = mcolors.Normalize()
        nz.autoscale(C)

        self.ax_background_01.quiver(x_new[:-1], y_new[:-1], x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1], 
                                        scale_units='xy', angles='xy', scale=1, zorder=1, color = cm.jet(nz(C)), width=0.0025)
        
        self.ax_background_01.scatter(x_new[0], y_new[0], 25, marker='o', color='blue', zorder=2) 
        self.ax_background_01.scatter(x_new[-1], y_new[-1], 25, marker='^', color='red', zorder=2) 
    #-----------------------------------------------------------------------------------------------------
    def plot_hum_traj(self, longitude, latitude):
        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        longitude, latitude = transform(inProj, outProj, longitude, latitude)
        x_new, y_new = self.landuse_background(longitude,latitude)
        self.ax_background_01.plot(x_new, y_new, linewidth=0.50, color="black")
    #-----------------------------------------------------------------------------------------------------     
    def step(self):

        #-----------------------------------------------------------
        # print("Model time:", self.model_time, "Model day:", self.model_day, "Model hour:", self.model_hour)
        #-----------------------------------------------------------

        self.slon = []
        self.slat = []
        self.elephant_agents = []

        for agent in self.schedule.agents:
            if isinstance(agent , Elephant):
                self.slon.append(agent.shape.x)
                self.slat.append(agent.shape.y)
                self.elephant_agents.append(agent)
        
        #print(self.elephant_agents)

        self.update_hourly_temp()
        self.update_season()
        self.update_prob_drink_water()

        self.update_human_disturbance_explict()

        if (self.model_time%288) == 0 and self.model_time >= 288:
            #print("PID:", os.getpid(), "Day:", self.model_day)
            self.update_food_matrix_constant()

        self.schedule.step()

        #COLLECT DATA 
        self.datacollector.collect(self)

        #UPDATE TIME
        self.model_time = self.model_time + 1      
        self.model_minutes = self.model_time * 5
        self.model_hour = int(self.model_minutes/60)
        self.model_day = int(self.model_hour/24)
        self.hour_in_day =  self.model_hour - self.model_day*24

        if self.model_time == self.max_time or self.num_elephant_deaths == (self.num_bull_elephants + self.num_herds):
            self.running = False

       #TERMINATE THE SIMULATION AND WRITE RESULTS TO FILE
        if self.running == False:
            print("Simulation completed")

            #print("Elephant deaths:", self.num_elephant_deaths, "Human deaths:", self.num_human_deaths)
            data_agents = self.datacollector.get_agent_vars_dataframe()


            #------------------------------------------------------------------------
            #PLOT ELEPHANT TRAJECTORY
            #------------------------------------------------------------------------
            agents = data_agents["AgentID"].unique()
            for agent in agents:
                if "herd" in agent or "bull" in agent:
                    ele_data = data_agents[data_agents["AgentID"] == agent]
                    self.plot_ele_traj(ele_data["longitude"].values, ele_data["latitude"].values)

            plt.title("Elephant agent trajectories")
            plt.savefig(os.path.join(folder, self.now, "output_files", 'ele_traj.png'), dpi=300)
            #------------------------------------------------------------------------


            #------------------------------------------------------------------------
            #PLOT ALL AGENT TRAJECTORIES
            #------------------------------------------------------------------------
            agents = data_agents["AgentID"].unique()
            for agent in agents:
                if "herd" in agent or "bull" in agent:
                    pass
                else:
                    hum_data = data_agents[data_agents["AgentID"] == agent]
                    self.plot_hum_traj(hum_data["longitude"].values, hum_data["latitude"].values)

            plt.title("Agent trajectories")
            plt.savefig(os.path.join(folder, self.now, "output_files", 'all_traj.png'), dpi=300)
            #------------------------------------------------------------------------


            #------------------------------------------------------------------------
            #SAVE CONFLICT LOCATIONS
            #------------------------------------------------------------------------
            with open(os.path.join(folder, self.now, "output_files", "conflict_locations.pkl"), 'wb') as f:
                pickle.dump(self.CONFLICT_LOCATIONS, f)

            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857') 
            arr = np.array(self.CONFLICT_LOCATIONS)

            if arr.size != 0:
                lon = arr[:, 1]
                lat = arr[:, 2]
                longitude, latitude = transform(inProj, outProj, lon, lat)
                x_new, y_new = self.landuse_background(longitude,latitude)
                plt.scatter(x_new, y_new, 50, marker='o', color='red', zorder=3)
            #------------------------------------------------------------------------


            #------------------------------------------------------------------------
            #SAVE INTERACTION LOCATIONS
            #------------------------------------------------------------------------
            with open(os.path.join(folder, self.now, "output_files", "interaction_locations.pkl"), 'wb') as f:
                pickle.dump(self.INTERACTION_LOCATIONS, f)

            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857') 
            arr = np.array(self.INTERACTION_LOCATIONS)
            if arr.size != 0:
                lon = arr[:, 1]
                lat = arr[:, 2]
                longitude, latitude = transform(inProj, outProj, lon, lat)
                x_new, y_new = self.landuse_background(longitude,latitude)
                plt.scatter(x_new, y_new, 50, marker='o', color='blue', zorder=3)
            #------------------------------------------------------------------------


            #------------------------------------------------------------------------
            plt.title("Conflict and interaction locations")
            plt.savefig(os.path.join(folder, self.now, "output_files", 'conflict_and_interactions.png'), dpi=300, bbox_inches='tight')
            #------------------------------------------------------------------------


            data_model = self.datacollector.get_model_vars_dataframe()


            data_agents.to_csv(os.path.join(folder, self.now, "output_files", "agent_data.csv"))
            data_model.to_csv(os.path.join(folder, self.now, "output_files", "model_data.csv"))
    #----------------------------------------------------------------------------------------------------






















def batch_run_model(model_params, number_processes, iterations, output_folder):

    freeze_support()

    global folder 
    folder = output_folder

    path_to_folder = os.path.join(folder)
    os.makedirs(path_to_folder, exist_ok=True)

    env = environment(prob_food_in_forest = model_params["prob_food_forest"],
                        prob_food_in_cropland = model_params["prob_food_cropland"],
                        prob_water_sources = model_params["prob_water_sources"], 
                        max_food_val_forest = model_params["max_food_val_forest"],
                        max_food_val_cropland = model_params["max_food_val_cropland"],
                        output_folder=output_folder).main()

    code_runner = batch_run(model_cls = Conflict_model, 
                            parameters = model_params, 
                            number_processes = number_processes, 
                            iterations = iterations,
                            max_steps = model_params["max_time"], 
                            data_collection_period=1, 
                            display_progress=True)

    return


