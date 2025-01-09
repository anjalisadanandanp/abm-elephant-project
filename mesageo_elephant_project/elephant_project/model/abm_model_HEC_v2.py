
#---------------imports-------------------#
import os                               # for file operations
import sys                              # for system operations 
import shutil                           # to copy files
import math                             # for mathematical operations
import numpy as np                      # for numerical operations
import pandas as pd                     # for data manipulation and analysis
import pickle                           # to save and load pickle objects
from osgeo import gdal                  # for raster operations
import rasterio as rio                  # for raster operations
from shapely.geometry import Point, LineString      # for creating point geometries and line geometries
from multiprocessing import freeze_support      # for multiprocessing
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
from scipy.ndimage import distance_transform_edt   # for distance transformations     
import mlflow     
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

# from mesageo_elephant_project.elephant_project.experiment_setup_files.batch_runner_codes.Mesa_BatchRunner_class_v1_1 import batch_run      #runnning multiple simulations
from mesa.batchrunner import batch_run
#-------------------------------------------------#






#--------------------------------------------------------------------------------------------------------------------------------
class Elephant(GeoAgent):
    """Class to define the ELEPHANT agent model"""

    def __init__(self,unique_id,model,shape):
        super().__init__(unique_id,model,shape) 


        #-------------------------------------------------------------------
        self.mode = self.model.random.choice(["RandomWalk", "TargetedWalk"])  
        self.fitness = 1
        self.aggression = 1
        self.disturbance_tolerance = 0.5
        self.crop_habituated = True
        self.food_consumed = 0                      
        self.visit_water_source = False            
        self.heading = self.model.random.uniform(0,360)
        self.ROW, self.COL = self.update_grid_index()  
        self.target_present = False                
        self.target_lat = None                     
        self.target_lon = None                      
        self.target_name = None
        self.radius_food_search = self.model.radius_food_search
        self.radius_water_search = self.model.radius_water_search
        self.num_days_water_source_visit = 0        
        self.num_days_food_depreceation = 0 
        self.num_thermoregulation_steps = 0
        self.num_steps_thermoregulated = 0
        self.distance_to_target = None
        self.danger_to_life = False
        self.conflict_with_humans = False

        self.proximity_to_plantations = self.model.calculate_proximity_map(landscape_matrix=self.model.LANDUSE, target_class=10, name="plantations")
        self.proximity_to_forests = self.model.calculate_proximity_map(landscape_matrix=self.model.LANDUSE, target_class=15, name="forests")

        #----------------hoose the type of memory matrix initialization-------------------#
        # self.initialize_memory_matrix_only_forest()
        self.initialize_memory_matrix_random()
        # self.initialize_memory_matrix_with_knowledge_from_fringe()
        #---------------------------------------------------------------------------------#

        self.proximity_to_water_sources = self.model.calculate_proximity_map(landscape_matrix=self.model.WATER, target_class=1, name="water_sources")
        self.proximity_to_food_sources = self.model.calculate_proximity_map(landscape_matrix=self.food_memory_cells, target_class=1, name="food_sources")

        self.crop_damage_matrix = np.zeros_like(self.model.LANDUSE)
        self.infrastructure_damage_matrix = np.zeros_like(self.model.LANDUSE)
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
    def initialize_memory_matrix_random(self):
        """ Function that assigns memory matrix to elephants"""

        #food_memory, water_memory
        food_memory=np.zeros_like(self.model.FOOD)
        food_memory_cells=np.zeros_like(self.model.FOOD)

        for i in range(0,self.model.row_size):
            for j in range(0,self.model.col_size):
                if self.model.random.uniform(0,1) < self.model.percent_memory_elephant:
                    food_memory[i,j] = self.model.FOOD[i][j]
                    if self.model.FOOD[i][j] > 0:
                        food_memory_cells[i,j] = 1

        self.food_memory = food_memory.tolist() 
        self.food_memory_cells = food_memory_cells.tolist()

        source = os.path.join(self.model.folder_root, "env", "LULC.tif")
        with rio.open(source) as src:
            ras_meta = src.profile

        memory_loc = os.path.join(self.model.folder_root, "env", "food_memory_" + str(self.unique_id) + ".tif")
        with rio.open(memory_loc, 'w', **ras_meta) as dst:
            dst.write(food_memory_cells.astype('float32'), 1)

        return 
    #-----------------------------------------------------------------------------------------------------
    def initialize_memory_matrix_with_knowledge_from_fringe(self):
        """ Function that assigns memory matrix to elephants. The elephant agent has knowledge of the fringe areas."""

        #self.knowlege_from_fringe : unit is in metres
        no_of_cells = self.model.random.randint(0, int(self.model.knowledge_from_fringe/self.model.xres))     #spatial resolution 
        
        food_memory=np.zeros_like(self.model.LANDUSE)
        food_memory_cells=np.zeros_like(self.model.FOOD)

        for i in range(0,self.model.row_size):
            for j in range(0,self.model.col_size):
                if self.model.LANDUSE[i][j]==self.model.landusetypes["Evergreen Broadlead Forest"] or self.model.LANDUSE[i][j]==self.model.landusetypes["Water Bodies"]:
                    if self.model.FOOD[i][j] > 0:
                        food_memory_cells[i,j] = 1
                elif self.model.LANDUSE[i][j]!=self.model.landusetypes["Evergreen Broadlead Forest"]:
                    try:
                        if self.model.LANDUSE[i+no_of_cells][j+no_of_cells]==self.model.landusetypes["Evergreen Broadlead Forest"] or self.model.LANDUSE[i-no_of_cells][j+no_of_cells]==self.model.landusetypes["Evergreen Broadlead Forest"] or self.model.LANDUSE[i-no_of_cells][j-no_of_cells]==self.model.landusetypes["Evergreen Broadlead Forest"] or self.model.LANDUSE[i+no_of_cells][j-no_of_cells]==self.model.landusetypes["Evergreen Broadlead Forest"]:
                            if self.model.FOOD[i][j] > 0:
                                food_memory_cells[i,j] = 1
                    except:
                        pass

        for i in range(0,self.model.row_size):
            for j in range(0,self.model.col_size):
                if food_memory_cells[i,j] == 1:
                    if self.model.random.uniform(0,1) < self.model.percent_memory_elephant:
                        food_memory[i,j] = self.model.FOOD[i][j]

        self.food_memory = food_memory.tolist() 
        self.food_memory_cells = food_memory_cells.tolist()

        source = os.path.join(self.model.folder_root, "env", "LULC.tif")
        with rio.open(source) as src:
            ras_meta = src.profile

        memory_loc = os.path.join(self.model.folder_root, "env", "food_memory_" + str(self.unique_id) + ".tif")
        with rio.open(memory_loc, 'w', **ras_meta) as dst:
            dst.write(food_memory_cells.astype('float32'), 1)

        return 
    #-----------------------------------------------------------------------------------------------------
    def initialize_memory_matrix_only_forest(self):
        """ Function that assigns memory matrix to elephants."""

        food_memory=np.zeros_like(self.model.LANDUSE)
        food_memory_cells=np.zeros_like(self.model.FOOD)

        for i in range(0,self.model.row_size):
            for j in range(0,self.model.col_size):
                if self.model.random.uniform(0,1) < self.model.percent_memory_elephant and self.proximity_to_plantations[i][j] > 1:
                    if self.model.FOOD[i][j] > 0:
                        food_memory_cells[i,j] = 1
                    
        self.food_memory = food_memory.tolist() 
        self.food_memory_cells = food_memory_cells.tolist()

        source = os.path.join(self.model.folder_root, "env", "LULC.tif")
        with rio.open(source) as src:
            ras_meta = src.profile

        memory_loc = os.path.join(self.model.folder_root, "env", "food_memory_" + str(self.unique_id) + ".tif")
        with rio.open(memory_loc, 'w', **ras_meta) as dst:
            dst.write(food_memory_cells.astype('float32'), 1)

        return
    #-------------------------------------------------------------------------------------------
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
    def next_step_to_move_v1(self):
        """how the elephant agent moves from the current co-ordinates to the next"""

        self.mode = self.current_mode_of_the_agent()   

        if self.mode == "RandomWalk":
            next_lon, next_lat = self.correlated_random_walk_without_terrain_factor()   
            row, col = self.model.get_indices(next_lon, next_lat)
            if self.model.LANDUSE[row][col] == 10 and self.aggression > self.model.aggression_threshold_enter_cropland:
                next_lon, next_lat = self.shape.x, self.shape.y     
            self.shape = self.move_point(next_lon, next_lat)
            self.target_name = None 

        elif self.mode == "ForagingMode":

            if self.target_present == False:
                print("step:0------foraging mode: target not present, choosing a food target------")
                filter = self.return_feasible_direction_to_move_v1()
                self.target_for_foraging_v1(filter)     
                self.target_name = "food:foraging" 

            else:
                print("step:0------foraging mode: target is present, moving towards the food target------")
                row, col = self.model.get_indices(self.target_lon, self.target_lat)

                if self.aggression < self.model.aggression_threshold_enter_cropland:
                    print("step:1------foraging mode: the aggression level is low, checking for the type of landuse------")

                    if self.model.LANDUSE[self.ROW][self.COL] == 10 or self.model.LANDUSE[row][col] == 10: 
                        print("step:2------foraging mode: target present is in cropland, check for the disturbance------")

                        if self.model.human_disturbance < self.disturbance_tolerance:   
                            print("step:3------foraging mode: target present is in cropland, night time, move towards target------")
                            next_lon, next_lat = self.targeted_walk_v0()
                            self.shape = self.move_point(next_lon, next_lat)
                            self.target_name = "cropland:foraging"
                        else:
                            print("step:3------foraging mode: target present is in cropland, day time, escape to forest------")
                            filter = self.return_feasible_direction_to_move_v1()
                            self.target_for_escape_v1(filter)    #move to the forest
                            self.target_name = "forest:escaping"
                            next_lon, next_lat = self.targeted_walk_v0()
                            self.shape = self.move_point(next_lon, next_lat)

                    else:
                        print("step:2------foraging mode: target present is in forest, move towards target------")
                        next_lon, next_lat = self.targeted_walk_v0()
                        self.shape = self.move_point(next_lon, next_lat)
                        self.target_name = "forest:foraging"

                else:
                    print("step:1------foraging mode: aggression level is high, move towards target------")
                    next_lon, next_lat = self.targeted_walk_v0()
                    self.shape = self.move_point(next_lon, next_lat)
                    self.target_name = "food:foraging"
            
            print("food source proximity (number of cells):", self.proximity_to_food_sources[self.ROW][self.COL])

        elif self.mode == "Thermoregulation":

            if self.target_present == False:
                print("step:0------thermoregulation mode: target not present, choosing a water source------")
                filter = self.return_feasible_direction_to_move_v1()
                self.target_thermoregulation_v1(filter)
                self.target_name = "thermoregulation"
            
            else:
                print("step:0------thermoregulation mode: target is present, moving towards the target------")
                next_lon, next_lat = self.targeted_walk_v0()
                self.shape = self.move_point(next_lon, next_lat)

        elif self.mode == "EscapeMode":      
            self.target_for_escape_v1()  
            next_lon, next_lat = self.targeted_walk_v0()
            self.shape = self.move_point(next_lon, next_lat)
            self.target_name = "forest:escaping"

        elif self.mode == "InflictDamage":     
            self.InflictDamage()   

        self.eat_food()
        self.drink_water()

        self.crop_and_infrastructure_damage() 

        self.update_fitness_value(-1/(288*self.model.num_days_agent_survives_in_deprivation))

        if  self.prob_thermoregulation > 0.5:
            print("water source proximity (number of cells):", self.proximity_to_water_sources[self.ROW][self.COL])
            if self.proximity_to_water_sources[self.ROW][self.COL]*33.33 < self.model.radius_water_search:
                self.num_steps_thermoregulated += 1

        return 
    #--------------------------------------------------------------------------------------------------
    def update_danger_to_life(self):
        """Update danger_to_life"""
            
        if self.conflict_neighbor != None:    
            if self.human_habituation < self.model.human_habituation_tolerance:  
                for agent in self.conflict_neighbor:
                    self.danger_to_life = True
                    self.conflict_with_humans = True 
                    return  

            else: 
                for agent in self.conflict_neighbor:
                    self.danger_to_life = False 
                    self.conflict_with_humans = True
                    return 
        else:
            self.danger_to_life = False   
            self.conflict_with_humans = False
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
        if  self.prob_thermoregulation > 0.5:
            self.num_thermoregulation_steps += 1
        
        if self.model.elephant_agent_visibility_radius != None:
            for agent in self.model.schedule.agents:
                conflict_neighbors = []
                if isinstance(agent,  Humans):
                    distance = agent.distance_calculator_epsg3857(self.shape.y, agent.shape.y, self.shape.x, agent.shape.x)
                    if distance <= self.model.elephant_agent_visibility_radius:
                        conflict_neighbors.append(agent)
                if len(conflict_neighbors) > 0:
                    self.conflict_neighbor = conflict_neighbors
                else:
                    self.conflict_neighbor = None

        self.update_danger_to_life()

        if (self.danger_to_life==True and self.conflict_with_humans==True):      
            #Supercedes all other requirements --> This state is encountered when in conflict with Human agents
            mode="EscapeMode"

        elif self.danger_to_life==True and self.conflict_with_humans==False:
            #Supercedes all other requirements --> This state is encountered when in conflict with Human agents
            mode="InflictDamage"

        elif self.fitness < self.model.fitness_threshold:
            mode = "ForagingMode"

        elif  self.prob_thermoregulation > 0.5:
            mode = "Thermoregulation"

        else:

            num = self.model.random.uniform(0,1)

            if self.mode == "RandomWalk":
                if num <= state1:
                    mode = "RandomWalk"
                else:
                    mode = "ForagingMode"
            
            elif self.mode == "ForagingMode":

                if num <= state2:
                        mode = "ForagingMode"
                else:
                    mode = "RandomWalk"

            else:
                mode = self.model.random.choice(["ForagingMode", "RandomWalk"])

        return mode
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
    def targeted_walk_v0(self):
        """" Function to simulate the targeted movement of agents """

        self.distance_to_target = self.distance_calculator_epsg3857(self.shape.y, self.target_lat, self.shape.x, self.target_lon)
        print("distance to target:", self.distance_to_target)

        if self.distance_to_target < self.model.xres:    #Move to the target
            print("----REACHED TARGET!----:", self.target_name)
            self.target_name = None
            self.target_present = False
            self.target_lon = None
            self.target_lat = None
            self.mode = "RandomWalk"        #switch mode to random walk
            self.distance_to_target = None
            return self.shape.x, self.shape.y

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
    def targeted_walk_v1(self):
        """" Function to simulate the targeted movement of agents """
        #moving one cell at a time

        self.distance_to_target = self.distance_calculator_epsg3857(self.shape.y, self.target_lat, self.shape.x, self.target_lon)
        print("distance to target:", self.distance_to_target)

        if self.distance_to_target < self.model.xres:    #Move to the target
            print("---Reached the target----", self.target_name)
            self.target_name = None
            self.target_present = False
            self.target_lon = None
            self.target_lat = None
            self.mode = "RandomWalk"        #switch mode to random walk
            self.distance_to_target = None
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
    
        #update heading direction
        co_ord = complex(dx/1000,dy/1000)
        direction = -np.angle([co_ord],deg=True)   
        self.heading = direction.tolist()        #unit:degrees

        return new_lon, new_lat
    #-----------------------------------------------------------------------------------------------------
    def return_feasible_direction_to_move_v1(self):

        radius = int(self.model.terrain_radius*2/self.model.xres) + 1   

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
        direction_0 = [1 for x in direction_0.flatten() if x > self.model.slope_tolerance]
        direction_1 = [1 for x in direction_1.flatten() if x > self.model.slope_tolerance]
        direction_2 = [1 for x in direction_2.flatten() if x > self.model.slope_tolerance]
        direction_3 = [1 for x in direction_3.flatten() if x > self.model.slope_tolerance]
        direction_4 = [1 for x in direction_4.flatten() if x > self.model.slope_tolerance]
        direction_5 = [1 for x in direction_5.flatten() if x > self.model.slope_tolerance]
        direction_6 = [1 for x in direction_6.flatten() if x > self.model.slope_tolerance]
        direction_7 = [1 for x in direction_7.flatten() if x > self.model.slope_tolerance]

        #calculate the cost of movement in each direction as sum of the direction cells
        cost_0 = sum(x for x in direction_0)
        cost_1 = sum(x for x in direction_1)
        cost_2 = sum(x for x in direction_2)
        cost_3 = sum(x for x in direction_3)
        cost_4 = sum(x for x in direction_4)
        cost_5 = sum(x for x in direction_5)
        cost_6 = sum(x for x in direction_6)
        cost_7 = sum(x for x in direction_7)

        cost = [cost_0, cost_1, cost_2, cost_3, cost_4, cost_5, cost_6, cost_7]
        direction = [135, 90, 45, 0, 315, 270, 225, 180]

        idx = np.argsort(cost)
        theta = [direction[i] for i in idx[0:self.model.number_of_feasible_movement_directions]]

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

        self.direction = movement_direction

        # print("choosen direction to move:", self.direction)

        if self.model.plot_stepwise_target_selection == True:

            #plot the slope matrix and the filter matrix to visualize the movement direction
            fig, ax = plt.subplots(1,2, figsize=(10,5))
            img1 = ax[0].imshow(slope, cmap='coolwarm', vmin=0, vmax=60)
            ax[0].set_title("Slope Matrix")
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            plt.colorbar(img1, ax=ax[0], orientation='vertical', shrink=0.5)

            img2 = ax[1].imshow(filter, cmap='gray')
            ax[1].set_title("Filter Matrix")
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            plt.colorbar(img2, ax=ax[1], orientation='vertical', shrink=0.5)

            plt.savefig(os.path.join(folder, self.model.now, "output_files", self.unique_id + "_step_" + str(self.model.schedule.steps) + "_feasible_move_direction_.png"), dpi=300, bbox_inches='tight')

        return filter
    #-----------------------------------------------------------------------------------------------------
    def target_for_foraging_v1(self, filter):

        if self.target_present == True:  
            return

        coord_list = []

        radius = int(self.model.radius_food_search*2/self.model.xres) + 1   #spatial resolution: xres

        row_start = self.ROW - radius//2
        col_start = self.COL - radius//2
        row_end = self.ROW + radius//2 + 1
        col_end = self.COL + radius//2 + 1

        #To handle edge cases
        if self.ROW < radius:
            row_start = 0

        elif self.ROW > self.model.row_size-1-radius:
            row_end = self.model.row_size-1

        if self.COL < radius:
            col_start = 0

        elif self.COL > self.model.col_size-radius-1:
            col_end = self.model.col_size-1

        if self.num_days_food_depreceation >= self.model.threshold_days_of_food_deprivation or self.crop_habituated == True:

            print("step:0------target for foraging:food deprecated agent or crop habituated agent------")

            if np.random.uniform(0,1) < self.aggression:

                print("step:1------target for foraging:move closer to plantations")

                for i in range(row_start,row_end):
                    for j in range(col_start,col_end):
                        if self.proximity_to_plantations[i][j] < self.proximity_to_plantations[self.ROW][self.COL] and filter[i - row_start][j - col_start] == 1:

                            if self.food_memory[i][j] > 0:
                                coord_list.append([i,j])     

                if coord_list == []:

                    for i in range(row_start,row_end):
                        for j in range(col_start,col_end):

                            if self.proximity_to_food_sources[i][j] < self.proximity_to_food_sources[self.ROW][self.COL] and filter[i - row_start][j - col_start] == 1:
                                    coord_list.append([i,j]) 

            else:

                print("step:1------target for foraging:choose food target from memory")

                for i in range(row_start,row_end):
                    for j in range(col_start,col_end):
                        if i == self.ROW and j == self.COL:
                            pass

                        elif self.food_memory[i][j] > 0 and filter[i - row_start][j - col_start] == 1:
                            coord_list.append([i,j])

                if coord_list == []:
                    
                    for i in range(row_start,row_end):
                        for j in range(col_start,col_end):

                            if self.proximity_to_food_sources[i][j] < self.proximity_to_food_sources[self.ROW][self.COL] and filter[i - row_start][j - col_start] == 1:
                                    coord_list.append([i,j]) 

                        
        else:

            print("step:0------target for foraging:not a food food deprecated agent or crop habituated agent------")
            print("step:1------target for foraging:choose food target from memory")
            for i in range(row_start,row_end):
                for j in range(col_start,col_end):
                    if i == self.ROW and j == self.COL:
                        pass

                    elif self.food_memory[i][j] > 0 and filter[i - row_start][j - col_start] == 1:
                        coord_list.append([i,j])

        if coord_list == []:
            print("step:2------target not available within search radius:move to a random cell------")

            for i in range(row_start,row_end):
                for j in range(col_start,col_end):
                    if filter[i - row_start][j - col_start] == 1:
                        coord_list.append([i,j]) 

        if coord_list != []:

            x, y = self.model.random.choice(coord_list)
            lon, lat = self.model.pixel2coord(x, y)
            self.target_lon, self.target_lat = lon, lat
            self.target_present = True

            if self.model.plot_stepwise_target_selection == True:

                #plot the temperature matrix and the filter matrix to visualize the movement direction, highlighting the target
                fig, ax = plt.subplots(1,2, figsize=(10,5))
                food_memory = np.array(self.food_memory)[row_start:row_end, col_start:col_end]
                img1 = ax[0].imshow(food_memory, cmap='gray', vmin=0, vmax=max(self.model.max_food_val_forest, self.model.max_food_val_cropland))
                ax[0].set_title("Food Memory Matrix: " + str(x - row_start) + " " + str(y - col_start))
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                plt.colorbar(img1, ax=ax[0], orientation='vertical', shrink=0.5)

                img2 = ax[1].imshow(filter, cmap='gray')
                ax[1].set_title("Filter Matrix")
                ax[1].set_xticks([])
                ax[1].set_yticks([])
                plt.colorbar(img2, ax=ax[1], orientation='vertical', shrink=0.5)

                #highlight the target cell 
                ax[0].scatter(y - col_start, x - row_start, color='red', s=100, marker='x', label='Target')
                ax[1].scatter(y - col_start, x - row_start, color='red', s=100, marker='x', label='Target')

                plt.savefig(os.path.join(folder, self.model.now, "output_files", self.unique_id + "_step_" + str(self.model.schedule.steps) + "_foraging_target_.png"), dpi=300, bbox_inches='tight')

        return
    #-----------------------------------------------------------------------------------------------------
    def target_thermoregulation_v1(self, filter):

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

        #To handle edge cases
        if self.ROW < radius:
            row_start = 0

        elif self.ROW > self.model.row_size-1-radius:
            row_end = self.model.row_size-1

        if self.COL < radius:
            col_start = 0

        elif self.COL > self.model.col_size-radius-1:
            col_end = self.model.col_size-1

        if self.num_days_water_source_visit >= self.model.threshold_days_of_water_deprivation:
            print("step:0------target for thermoregulation:water deprecated agent, check for water sources------")
            for i in range(row_start,row_end):
                for j in range(col_start,col_end):

                    if self.model.WATER[i][j] > 0 and filter[i - row_start][j - col_start] == 1:
                        coord_list.append([i,j]) 

                    elif self.proximity_to_water_sources[i][j] < self.proximity_to_water_sources[self.ROW][self.COL] and filter[i - row_start][j - col_start] == 1:
                        coord_list.append([i,j]) 

        else:
            print("step:0------target for thermoregulation:not a water deprecated agent, move to a random cell------")
            for i in range(row_start,row_end):
                for j in range(col_start,col_end):
                    if filter[i - row_start][j - col_start] == 1:
                        coord_list.append([i,j]) 

        if coord_list == []:
            print("step:1------target not available within search radius:move to a random cell------")
            for i in range(row_start,row_end):
                for j in range(col_start,col_end):
                    if filter[i - row_start][j - col_start] == 1:
                        coord_list.append([i,j]) 

        if coord_list != []:

            x, y = self.model.random.choice(coord_list)
            lon, lat = self.model.pixel2coord(x, y)
            self.target_lon, self.target_lat = lon, lat
            self.target_present = True

            if self.model.plot_stepwise_target_selection == True:

                #plot the temperature matrix and the filter matrix to visualize the movement direction, highlighting the target
                fig, ax = plt.subplots(1,2, figsize=(10,5))
                water_matrix = np.array(self.model.WATER)[row_start:row_end, col_start:col_end]
                img1 = ax[0].imshow(water_matrix, cmap='coolwarm', vmin=0, vmax=1)
                ax[0].set_title("Water Sources")
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                plt.colorbar(img1, ax=ax[0], orientation='vertical', shrink=0.5)

                img2 = ax[1].imshow(filter, cmap='gray')
                ax[1].set_title("Filter Matrix")
                ax[1].set_xticks([])
                ax[1].set_yticks([])
                plt.colorbar(img2, ax=ax[1], orientation='vertical', shrink=0.5)

                #highlight the target cell 
                ax[0].scatter(y - col_start, x - row_start, color='red', s=100, marker='x', label='Target')
                ax[1].scatter(y - col_start, x - row_start, color='red', s=100, marker='x', label='Target')

                plt.savefig(os.path.join(folder, self.model.now, "output_files", self.unique_id + "_step_" + str(self.model.schedule.steps) + "_thermoregulation_target_.png"), dpi=300, bbox_inches='tight')
    
        return
    #-----------------------------------------------------------------------------------------------------
    def target_for_escape_v1(self, filter):
        """ Function returns the target for the elephant agent to move in case of danger to life. """

        if self.target_present == True:     #If target already exists
            return
        
        radius = int(self.radius_forest_search*2/self.model.xres)     
        row_start = self.ROW - radius//2
        col_start = self.COL - radius//2
        row_end = self.ROW + radius//2 + 1
        col_end = self.COL + radius//2 + 1

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
        
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                if i == self.ROW and j == self.COL:
                    pass

                elif self.model.LANDUSE[i][j] == 15:
                    coord_list.append([i, j])

        if coord_list==[]:
            coord_list.append([self.ROW,self.COL])
            for _ in range(25):

                radius = int(self.terrain_radius*2/self.model.xres)   

                row_start = self.ROW - radius//2
                col_start = self.COL - radius//2
                row_end = self.ROW + radius//2 + 1
                col_end = self.COL + radius//2 + 1

                i = self.model.random.randint(row_start, row_end)
                j = self.model.random.randint(col_start, col_end)

                if self.proximity_to_forests[i][j] <= self.proximity_to_forests[self.ROW][self.COL] and filter[i - row_start][j - col_start] == 1:
                    coord_list.append([i,j])

        x, y = self.model.random.choice(coord_list)
        lon = self.model.xres * 0.5  + self.model.xmin + y * self.model.xres
        lat = self.model.yres * 0.5  + self.model.ymax + x * self.model.yres
        self.target_lon, self.target_lat = lon, lat
        self.target_present = True

        return
    #-----------------------------------------------------------------------------------------------------
    def drink_water(self):
        """ The elephant agent consumes water from the current cell it is located in"""

        row, col = self.update_grid_index()

        if "dry" in self.model.season:

            if self.model.WATER[row][col] > 0:
                self.visit_water_source = True

        elif "wet" in self.model.season:

            if self.model.WATER[row][col] > 0:
                self.visit_water_source = True

        else:
            pass

        return
    #---------------------------------------------------------------------------------------------------
    def eat_food(self):

        """ The elephant agent consumes food from the current cell it is located in"""

        row, col = self.update_grid_index()

        if self.model.FOOD[row][col] > 0:
            food_consumed = self.model.FOOD[row][col]
            self.food_consumed += food_consumed     #update the food consumed by the agent
            self.model.FOOD[row][col] -= food_consumed      #update the food matrix of the model
            self.food_memory[row][col] -= food_consumed     #update the memory matrix of the agent

            if self.model.FOOD[row][col] <= 0:    #if food is less than 0 in the cell
                self.model.FOOD[row][col] = 0
                self.food_memory[row][col] = 0
                if self.food_memory_cells[row][col] == 1:
                    self.food_memory_cells[row][col] = 0
                    self.update_food_memory_proximity_map()

            self.update_memory_matrix()
            

        return
    #---------------------------------------------------------------------------------------------------
    def update_memory_matrix(self):
        """Function to update the memory matrix of the agent"""
        food = np.array(self.model.FOOD)
        food_mem = np.array(self.food_memory)
        food_memory = np.zeros_like(food_mem)
        food_memory[(food_mem > 0)] = food[(food_mem > 0)]
        self.food_memory = food_memory
    #----------------------------------------------------------------------------------------------------
    def update_fitness_value(self, val):
        """Updates agent's fitness value, keeping it within [0,1]"""

        self.fitness = max(0, min(1, self.fitness + val))

        return
    #----------------------------------------------------------------------------------------------------
    def update_fitness_thermoregulation(self, num_thermoregulation_steps, num_steps_thermoregulated):
        """Function to update the fitness value of the agent as per thermoregulation criteria"""
        #num_thermoregulation_steps: number of steps the agent has to thermoregulate
        #num_steps_thermoregulated: number of steps the agent has thermoregulated

        # print("UPDATE FITNESS: THERMOREGULATION")

        try:
            fitness_increment = (1/self.model.num_days_agent_survives_in_deprivation)*(num_thermoregulation_steps/288)*(num_steps_thermoregulated/num_thermoregulation_steps)
            self.update_fitness_value(fitness_increment)
        except:
            pass

        return
    #----------------------------------------------------------------------------------------------------
    def update_fitness_foraging(self, num_thermoregulation_steps, food_consumed):
        """Function to update the fitness value of the agent as per foraging criteria"""
        #num_thermoregulation_steps: number of steps the agent has to thermoregulate
        #food_consumed: amount of food consumed by the agent

        # print("UPDATE FITNESS: FORAGING")

        fitness_increment = (1/self.model.num_days_agent_survives_in_deprivation)*((288-num_thermoregulation_steps)/288)*(min(food_consumed, self.daily_dry_matter_intake)/self.daily_dry_matter_intake)
        self.update_fitness_value(fitness_increment)

        if food_consumed > self.daily_dry_matter_intake and self.fitness < self.model.fitness_threshold:
            self.update_fitness_value(0.1)

        return
    #----------------------------------------------------------------------------------------------------
    def update_food_memory_proximity_map(self):
        
        self.proximity_to_food_sources = self.model.calculate_proximity_map(landscape_matrix=self.food_memory_cells, target_class=1, name="food_sources")

        source = os.path.join(self.model.folder_root, "env", "LULC.tif")
        with rio.open(source) as src:
            ras_meta = src.profile

        proximity_loc = os.path.join(self.model.folder_root, "env", "proximity_to_food_sources_" + str(self.unique_id) + "_" + str(self.model.schedule.steps) + ".tif")
        with rio.open(proximity_loc, 'w', **ras_meta) as dst:
            dst.write(np.array(self.proximity_to_food_sources).astype('float32'), 1)

        return
    #----------------------------------------------------------------------------------------------------
    def crop_and_infrastructure_damage(self):
        """Function to simulate the crop and infrastructure damage by the elephant agents"""

        row, col = self.update_grid_index()

        if self.model.INFRASTRUCTURE_MATRIX[row][col] != 0:
            if np.random.uniform(0,1) < self.model.prob_infrastructure_damage:
                self.infrastructure_damage_matrix[row][col] = 1

                source = os.path.join(self.model.folder_root, "env", "LULC.tif")

                with rio.open(source) as src:
                    ras_meta = src.profile

                infrastructure_loc = os.path.join(self.model.folder_root, "env", "infrastructure_damage_" + str(self.unique_id) + "_.tif")
                with rio.open(infrastructure_loc, 'w', **ras_meta) as dst:
                    dst.write(np.array(self.infrastructure_damage_matrix).astype('float32'), 1)

        if self.model.AGRICULTURAL_PLOTS[row][col] != 0 and self.model.FOOD[row][col] > 0:
            if np.random.uniform(0,1) < self.model.prob_crop_damage: 
                self.crop_damage_matrix[row][col] = 1

                #save the infrastructure and crop damage matrix
                source = os.path.join(self.model.folder_root, "env", "LULC.tif")

                with rio.open(source) as src:
                    ras_meta = src.profile

                crop_loc = os.path.join(self.model.folder_root, "env", "crop_damage_" + str(self.unique_id) + "_.tif")
                with rio.open(crop_loc, 'w', **ras_meta) as dst:
                    dst.write(np.array(self.crop_damage_matrix).astype('float32'), 1)

        return
    #----------------------------------------------------------------------------------------------------
    def elephant_cognition(self):

        """Function to simulate the cognition of the elephant agent"""

        if (self.model.model_time%288) == 0 and self.model.model_day > 0:         #start of a new day

            self.update_fitness_thermoregulation(self.num_thermoregulation_steps, self.num_steps_thermoregulated)
            self.update_fitness_foraging(self.num_thermoregulation_steps, self.food_consumed)

            if self.model.track_in_mlflow == True:
                mlflow.log_metric("fitness of the agent", self.fitness, self.model.model_day)
                mlflow.log_metric("daily food consumption", self.food_consumed, self.model.model_day)
                mlflow.log_metric("number of days of food deprivation", self.num_days_food_depreceation, self.model.model_day)
                mlflow.log_metric("number of days since water source visit", self.num_days_water_source_visit, self.model.model_day)

            if self.visit_water_source == False:    
                self.num_days_water_source_visit += 1     
   
            else:
                self.visit_water_source = False
                self.num_days_water_source_visit = 0    

            if self.food_consumed < self.daily_dry_matter_intake:
                self.num_days_food_depreceation += 1   
            else:
                self.num_days_food_depreceation = 0
            
            self.food_consumed = 0 
            self.num_steps_thermoregulated = 0
            self.num_thermoregulation_steps = 0

        self.next_step_to_move_v1()
        self.ROW, self.COL = self.update_grid_index()

        # print("Fitness of the elephant agent:", self.fitness, "mode:", self.mode, "num_days_food_depreceation:", self.num_days_food_depreceation, "num_days_water_source_visit:", self.num_days_water_source_visit)

        return
    #----------------------------------------------------------------------------------------------------
    def step(self):     
        """ Function to simulate the movement of the elephant agent"""

        self.elephant_cognition()
    #----------------------------------------------------------------------------------------------------





#--------------------------------------------------------------------------------------------------------------------------------
#Human agent class: serves as superclass
class Humans(GeoAgent):
    """ Human agents class"""

    def __init__(self,unique_id, model, shape):
        super().__init__(unique_id, model, shape)

        self.home_lon = self.shape.x
        self.home_lat = self.shape.y
    #----------------------------------------------------------------------------------------------------
    def step(self):
        return
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





class conflict_model(Model):
    """ 
    Model class: Elephant-Human interaction model
    """

    #Model Initialization
    def __init__(self,
        year,
        month,
        num_bull_elephants,                         #number of solitary bull elephants in the simulation
        area_size,                                  #simulation area in sq. km
        spatial_resolution,                         #spatial resolution of the landscape cells
        max_food_val_cropland,                      #maximum food value in a cropland cell
        max_food_val_forest,                        #maximum food value in a forest cell
        prob_food_forest,                           #probability of food in the forest
        prob_food_cropland,                         #probability of food in the cropland
        prob_water_sources,                         #probability of water holes in the landscape
        thermoregulation_threshold,                 #threshold temperature for thermoregulation for the elephant agents
        num_days_agent_survives_in_deprivation,          
        knowledge_from_fringe,                      #distance from the fringe where elephants knows food availability within the crop fields
        prob_crop_damage,                           #probability of damaging crop if entered an agricultural field
        prob_infrastructure_damage,                 #probability of damaging infrastructure if entered a settlement area
        percent_memory_elephant,                    #percentage memory of the landscape cells by the elephant agents at the start of the simulation
        radius_food_search,                         #radius within which the elephant agent searches for food
        radius_water_search,                        #radius within which the elephant agent searches for water
        radius_forest_search,                       #radius within which the elephant agent searches for forest
        fitness_threshold,                          #fitness threshold below which the elephant agent engages only in foraging activities
        terrain_radius,                             #parameter in terrain cost function
        slope_tolerance,                                  #parameter in terrain cost function
        num_processes,                              #number of processes to run the simulation
        iterations,                                 #number of iterations to run the simulation
        max_time_steps,                             #maximum simulation time (in ticks)
        aggression_threshold_enter_cropland,        #aggression threshold for entering a cropland cell
        elephant_agent_visibility_radius,           
        plot_stepwise_target_selection,
        threshold_days_of_food_deprivation,
        threshold_days_of_water_deprivation,
        number_of_feasible_movement_directions,
        track_in_mlflow
        ):



        self.MAP_COORDS=[9.3245, 76.9974]   



        #Folders to read the data files from depending upon the area and resolution
        self.area = {800:"area_800sqKm", 900:"area_900sqKm", 1000:"area_1000sqKm", 1100:"area_1100sqKm"}
        self.reso = {30: "reso_30x30", 60:"reso_60x60", 90:"reso_90x90", 120:"reso_120x120", 150:"reso_150x150"}
        self.delta = {800:14142, 900:15000, 1000:15811, 1100: 16583}  #distance in meteres from the center of the polygon area



        #-------------------------------------------------------------------
        self.year = year
        self.month = month
        self.num_bull_elephants = num_bull_elephants    
        self.area_size = area_size
        self.spatial_resolution = spatial_resolution  
        self.max_food_val_forest = max_food_val_forest
        self.max_food_val_cropland = max_food_val_cropland
        self.prob_food_forest = prob_food_forest
        self.prob_food_cropland = prob_food_cropland
        self.prob_water_sources = prob_water_sources
        self.thermoregulation_threshold = thermoregulation_threshold
        self.num_days_agent_survives_in_deprivation = num_days_agent_survives_in_deprivation
        self.knowledge_from_fringe = knowledge_from_fringe
        self.prob_crop_damage = prob_crop_damage
        self.prob_infrastructure_damage = prob_infrastructure_damage
        self.percent_memory_elephant = percent_memory_elephant
        self.radius_food_search = radius_food_search
        self.radius_water_search = radius_water_search
        self.radius_forest_search = radius_forest_search
        self.fitness_threshold = fitness_threshold
        self.terrain_radius = terrain_radius
        self.slope_tolerance = slope_tolerance
        self.max_time_steps = max_time_steps
        self.aggression_threshold_enter_cropland = aggression_threshold_enter_cropland
        self.elephant_agent_visibility_radius = elephant_agent_visibility_radius
        self.plot_stepwise_target_selection = plot_stepwise_target_selection
        self.threshold_days_of_food_deprivation = threshold_days_of_food_deprivation
        self.threshold_days_of_water_deprivation = threshold_days_of_water_deprivation
        self.number_of_feasible_movement_directions = number_of_feasible_movement_directions
        self.track_in_mlflow = track_in_mlflow
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        #Geographical extend of the study area
        #-------------------------------------------------------------------
        latitude_center = self.MAP_COORDS[0]
        longitude_center = self.MAP_COORDS[1]
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
        self.folder_root = os.path.join(folder, self.now)

        os.mkdir(os.path.join(folder, self.now, "env"))
        os.mkdir(os.path.join(folder, self.now, "output_files"))
        
        environment(prob_food_in_forest = self.prob_food_forest,
                            prob_food_in_cropland = self.prob_food_cropland,
                            prob_water_sources = self.prob_water_sources,
                            max_food_val_forest = self.max_food_val_forest,
                            max_food_val_cropland = self.max_food_val_cropland,
                            output_folder=os.path.join(self.folder_root, "env")).main()
        
        env_folder_seethathode = os.path.join("mesageo_elephant_project/elephant_project/", "experiment_setup_files","environment_seethathode", "Raster_Files_Seethathode_Derived", self.area[area_size], self.reso[spatial_resolution])
        shutil.copy(os.path.join(env_folder_seethathode, "DEM.tif"), os.path.join(self.folder_root, "env"))
        shutil.copy(os.path.join(env_folder_seethathode, "LULC.tif"), os.path.join(self.folder_root, "env"))
        shutil.copy(os.path.join(env_folder_seethathode, "population.tif"), os.path.join(self.folder_root, "env"))
        # shutil.copy(os.path.join(folder, "food_matrix_" + str(self.prob_food_forest) + "_" + str(self.prob_food_cropland) + "_.tif"), os.path.join(self.folder_root, "env"))
        # shutil.copy(os.path.join(folder, "water_matrix_" + str(self.prob_water_sources) + "_.tif"), os.path.join(self.folder_root, "env"))
        # shutil.copy(os.path.join(folder, "landscape_cell_status.tif"), os.path.join(self.folder_root, "env"))

        self.DEM = self.DEM_study_area()
        self.SLOPE = self.SLOPE_study_area()
        self.LANDUSE = self.LANDUSE_study_area()
        self.FOOD = self.FOOD_MATRIX()
        self.WATER = self.WATER_MATRIX()
        self.LANDSCAPE_STATUS = self.LANDSCAPE_CELL_STATUS()
        self.AGRICULTURAL_PLOTS, self.INFRASTRUCTURE_MATRIX = self.PROPERTY_MATRIX()
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        #MODEL TIME VARIABLES
        #-------------------------------------------------------------------
        self.model_time = 0       #Tick counter
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
        LULC = os.path.join(self.folder_root, "env", "LULC.tif")
        ds = gdal.Open(LULC, 0)
        arr = ds.ReadAsArray()
        self.xmin, self.xres, self.xskew, self.ymax, self.yskew, self.yres = ds.GetGeoTransform()
        self.row_size, self.col_size =  arr.shape
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        self.landusetypes={"Deciduous Broadleaf Forest":1,"Cropland":2,"Built-up Land":3,"Mixed Forest":4,
                       "Shrubland":5,"Barren Land":6,"Fallow Land":7,"Wasteland":8,"Water Bodies":9,
                       "Plantations":10,"Aquaculture":11,"Mangrove Forest":12,"Salt Pan":13,"Grassland":14,
                       "Evergreen Broadlead Forest":15,"Deciduous Needleleaf Forest":16,
                       "Permanent Wetlands":17, "Snow and ice":18, "Evergreen Needleleaf Forest":19}
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        self.schedule = RandomActivation(self)      #Random activation of agents
        self.grid = GeoSpace()
        self.running='True'
        #-------------------------------------------------------------------


        if track_in_mlflow == True:
            mlflow.start_run()
            mlflow.log_params(
                            {"year": self.year,
                            "month": self.month,
                            "num_bull_elephants": self.num_bull_elephants, 
                            "area_size": self.area_size,              
                            "spatial_resolution": self.spatial_resolution, 
                            "max_food_val_cropland": self.max_food_val_cropland,
                            "max_food_val_forest": self.max_food_val_forest,
                            "prob_food_forest": self.prob_food_forest,
                            "prob_food_cropland": self.prob_food_cropland,
                            "prob_water_sources": self.prob_water_sources,
                            "thermoregulation_threshold": self.thermoregulation_threshold,
                            "num_days_agent_survives_in_deprivation": self.num_days_agent_survives_in_deprivation,        
                            "knowledge_from_fringe": self.knowledge_from_fringe,   
                            "prob_crop_damage": self.prob_crop_damage,           
                            "prob_infrastructure_damage": self.prob_infrastructure_damage,
                            "percent_memory_elephant": self.percent_memory_elephant,   
                            "radius_food_search": self.radius_food_search,     
                            "radius_water_search": self.radius_water_search, 
                            "radius_forest_search": self.radius_forest_search,
                            "fitness_threshold": self.fitness_threshold,   
                            "terrain_radius": self.terrain_radius,       
                            "slope_tolerance": self.slope_tolerance,   
                            "max_time_steps": self.max_time_steps,
                            "aggression_threshold_enter_cropland": self.aggression_threshold_enter_cropland,
                            "elephant_agent_visibility_radius": self.elephant_agent_visibility_radius,
                            "threshold_days_of_food_deprivation": self.threshold_days_of_food_deprivation,
                            "threshold_days_of_water_deprivation": self.threshold_days_of_water_deprivation,
                            "number_of_feasible_movement_directions": self.number_of_feasible_movement_directions
                            })
            

        #-------------------------------------------------------------------
        with open('mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/temperature/hourly_temp_2010_without_juveniles_' + str(self.thermoregulation_threshold) + '.pkl','rb') as f:
            self.hourly_temp = pickle.load(f)
  
        self.update_hourly_temp()
        #-------------------------------------------------------------------

        self.initialize_bull_elephants()

        #-------------------------------------------------------------------
        self.datacollector = DataCollector(model_reporters={},
                                           agent_reporters={
                                                "longitude": "shape.x", 
                                                "latitude": "shape.y",
                                                "mode": "mode",
                                                "fitness": "fitness",
                                                "daily_dry_matter_intake": "daily_dry_matter_intake",
                                                "food_consumed": "food_consumed",
                                                "visit_water_source": "visit_water_source",
                                                "target_present": "target_present",
                                                "target_lon": "target_lon",
                                                "target_lat": "target_lat",
                                                "distance_to_target": "distance_to_target",
                                                "target_name": "target_name",
                                                "num_days_water_source_visit": "num_days_water_source_visit",
                                                "num_days_food_depreceation": "num_days_food_depreceation",
                                                "num_thermoregulation_steps": "num_thermoregulation_steps",
                                                "num_steps_thermoregulated": "num_steps_thermoregulated",
                                                })

        self.datacollector.collect(self)
    #-------------------------------------------------------------------
    def initialize_bull_elephants(self, **kwargs):
        """Initialize the elephant agents"""

        coord_lon = [8577680]
        coord_lat = [1045831]

        if self.track_in_mlflow == True:
            mlflow.log_params({"starting longitude": coord_lon[0],
                               "starting latitude": coord_lat[0]})

        # coord_lat, coord_lon = self.elephant_distribution_random_init_forest()

        ds = gdal.Open(os.path.join(self.folder_root, "env", "LULC.tif"))
        data = ds.ReadAsArray()
        data = np.flip(data, axis=0)
        row_size, col_size = data.shape
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

        data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

        for i in range(1,10):
            data[data == data_value_map[i]] = i

        fig, ax = plt.subplots(figsize = (10,10))
        ax.yaxis.set_inverted(True)

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
        LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

        map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        clrs = ["greenyellow","mediumpurple","turquoise", "plum", "black", "blue", "yellow", "mediumseagreen", "forestgreen"] 
        cmap, norm = colors.from_levels_and_colors(levels, clrs)

        map.imshow(data, cmap = cmap, norm=norm, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 0.75)

        map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
        map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])
        
        cbar = plt.colorbar(ticks=[1,2,3,4,5,6,7,8,9],fraction=0.046, pad=0.04)
        cbar.ax.set_yticks(ticks=[1,2,3,4,5,6,7,8,9]) 
        cbar.ax.set_yticklabels(["Deciduous Broadleaf Forest","Built-up Land","Mixed Forest","Shrubland","Barren Land","Water Bodies","Plantations","Grassland","Broadleaf evergreen forest"])
        
        for i in range(0, self.num_bull_elephants):    #initializing bull elephants

            elephant=AgentCreator(Elephant,{"model":self})  
            this_x = np.array(coord_lon)[i] + self.random.randint(-10,10)
            this_y = np.array(coord_lat)[i] + self.random.randint(-10,10)
            newagent = elephant.create_agent(Point(this_x,this_y), "bull_"+str(i))
            newagent.Leader = True   #bull elephants are always leaders
            newagent.sex = "Male"
            newagent.age = self.random.randrange(15, 60) 
            
            #Assign body weight of the elephant agent depending on the sex and age
            newagent.body_weight = self.assign_body_weight_elephants(newagent.age, newagent.sex)

            #Assign daily dry matter intake depending on the body weight
            newagent.daily_dry_matter_intake = self.assign_daily_dietary_requiremnt(newagent.body_weight)

            self.grid.add_agents(newagent)
            self.schedule.add(newagent)

            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
            longitude, latitude = transform(inProj, outProj, this_x, this_y)
            x_new, y_new = map(longitude,latitude)
            ax.scatter(x_new, y_new, 30, marker='s', color='red') 

        plt.title("Elephant agent initialisation")
        plt.savefig(os.path.join(folder, self.now, "output_files", "elephant_agent_init_coords.png"), dpi = 300, bbox_inches = 'tight')
        
        if self.track_in_mlflow == True:
            mlflow.log_figure(fig, "elephant_agent_init_coords.png")

        plt.close()
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
    def DEM_study_area(self):
        """ Returns the digital elevation model of the study area"""

        fid = os.path.join(self.folder_root, "env", "DEM.tif")

        DEM = gdal.Open(fid).ReadAsArray()  
        return DEM.tolist() 
    #-----------------------------------------------------------------------------------------------------
    def SLOPE_study_area(self):
        """ Returns the slope model of the study area"""

        # Evaluate gradient in two dimensions
        px, py = np.gradient(np.array(self.DEM), 20)
        slope = np.sqrt(px ** 2 + py ** 2)

        # If needed in degrees, convert using some trig functions
        slope = np.degrees(np.arctan(slope))

        source = os.path.join(self.folder_root, "env", "LULC.tif")
        with rio.open(source) as src:
            ras_meta = src.profile

        loc = os.path.join(folder, self.now, "env", "slope_matrix.tif")
        with rio.open(loc, 'w', **ras_meta) as dst:
            dst.write(slope, 1)


        return slope.tolist()  
    #-----------------------------------------------------------------------------------------------------
    def LANDUSE_study_area(self):
        """ Returns the landuse model of the study area"""

        fid = os.path.join(self.folder_root, "env", "LULC.tif")

        LULC = gdal.Open(fid).ReadAsArray()  
        return LULC.tolist() 
    #-----------------------------------------------------------------------------------------------------
    def FOOD_MATRIX(self):
        """ Returns the food matrix model of the study area"""
        fid = os.path.join(self.folder_root, "env", "food_matrix_"+ str(self.prob_food_forest) + "_" + str(self.prob_food_cropland) + "_.tif")
        FOOD = gdal.Open(fid).ReadAsArray()  
        return FOOD.tolist()  
    #-----------------------------------------------------------------------------------------------------
    def WATER_MATRIX(self):
        """ Returns the water matrix model of the study area"""
        fid = os.path.join(self.folder_root , "env", "water_matrix_"+ str(self.prob_water_sources) + "_.tif")
        WATER = gdal.Open(fid).ReadAsArray()  
        return WATER.tolist() 
    #-----------------------------------------------------------------------------------------------------
    def LANDSCAPE_CELL_STATUS(self):
        """ Returns the slope model of the study area"""

        fid = os.path.join(self.folder_root, "env", "landscape_cell_status.tif")

        slope = gdal.Open(fid).ReadAsArray()  
        return slope.tolist() 
    #-----------------------------------------------------------------------------------------------------
    def PROPERTY_MATRIX(self):
        """ Returns the infrastructure and crop matrix of the study area"""

        population = gdal.Open(os.path.join(self.folder_root, "env", "population.tif")).ReadAsArray()
        landscape_cell_status = os.path.join(self.folder_root, "env", "landscape_cell_status.tif")
        landscape_cell_status = gdal.Open(landscape_cell_status).ReadAsArray()
        m,n = landscape_cell_status.shape

        agricultural_plots = np.zeros_like(landscape_cell_status)
        infrastructure = np.zeros_like(landscape_cell_status)

        for i in range(0,m):
            for j in range(0,n):
                if landscape_cell_status[i,j] == 2:  
                    agricultural_plots[i, j] = 1

                if population[i,j] != 0:  
                    infrastructure[i, j] = 1

        source = os.path.join(self.folder_root, "env", "LULC.tif")
        with rio.open(source) as src:
            ras_meta = src.profile

        loc = os.path.join(folder, self.now, "env", "infrastructure_matrix.tif")
        with rio.open(loc, 'w', **ras_meta) as dst:
            dst.write(infrastructure, 1)

        loc = os.path.join(folder, self.now, "env", "agricultural_plots_matrix.tif")
        with rio.open(loc, 'w', **ras_meta) as dst:
            dst.write(agricultural_plots, 1)

        return agricultural_plots.tolist(), infrastructure.tolist()
    #-----------------------------------------------------------------------------------------------------
    def calculate_proximity_map(self, landscape_matrix, target_class, name):
        """
        Calculate proximity matrix for a given land use class
        
        Args:
            landscape_matrix: 2D numpy array with land use classes
            target_class: integer representing the land use class to calculate proximity for
        
        Returns:
            2D numpy array with distances to nearest target class cell
        """
        landscape_matrix = np.array(landscape_matrix)
        binary_matrix = (landscape_matrix == target_class).astype(int)
        proximity_matrix = distance_transform_edt(1 - binary_matrix)

        # save proximity matrix
        source = os.path.join(self.folder_root, "env", "LULC.tif")
        with rio.open(source) as src:
            ras_meta = src.profile

        loc = os.path.join(folder, self.now, "env", "proximity_to_" + name + ".tif")
        with rio.open(loc, 'w', **ras_meta) as dst:
            dst.write(proximity_matrix, 1)
        
        return proximity_matrix.tolist()
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
    def update_season(self):

        if self.model_day >= 15 and self.model_day <= 120:
            self.season = "dry"
        else:
            self.season = "wet"

        return
    #----------------------------------------------------------------------------------------------------
    def update_hourly_temp(self):

        if self.model_hour == 0:
            self.temp = self.hourly_temp[self.model_hour + self.num_days_elapsed*24]

        elif self.model_time%12 == 0:
            self.temp = self.hourly_temp[self.model_hour + self.num_days_elapsed*24]

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
            
        return i, j
    #---------------------------------------------------------------------------------------------------- 
    def update_human_disturbance_explict(self):
        #6am to 6pm: high disturbance
        #6pm to 6am: low disturbance
        if self.hour_in_day >= 6 and self.hour_in_day <= 18:
            self.human_disturbance = 1

        else:
            self.human_disturbance = 0

        return
    #----------------------------------------------------------------------------------------------------
    def plot_ele_traj_on_LULC(self, longitude, latitude, agent_id):

        ds = gdal.Open(os.path.join(self.folder_root, "env", "LULC.tif"))
        data = ds.ReadAsArray()
        data = np.flip(data, axis=0)
        row_size, col_size = data.shape
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

        data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

        for i in range(1,10):
            data[data == data_value_map[i]] = i

        fig, ax = plt.subplots(figsize = (10,10))
        ax.yaxis.set_inverted(True)

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
        LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

        map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        clrs = ["greenyellow","mediumpurple","turquoise", "plum", "black", "blue", "yellow", "mediumseagreen", "forestgreen"] 
        cmap, norm = colors.from_levels_and_colors(levels, clrs)

        map.imshow(data, cmap = cmap, norm=norm, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 0.75)

        map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
        map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])
        
        cbar = plt.colorbar(ticks=[1,2,3,4,5,6,7,8,9],fraction=0.046, pad=0.04)
        cbar.ax.set_yticks(ticks=[1,2,3,4,5,6,7,8,9]) 
        cbar.ax.set_yticklabels(["Deciduous Broadleaf Forest","Built-up Land","Mixed Forest","Shrubland","Barren Land","Water Bodies","Plantations","Grassland","Broadleaf evergreen forest"])
        
        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        longitude, latitude = transform(inProj, outProj, longitude, latitude)
        x_new, y_new = map(longitude,latitude)
        C = np.arange(len(x_new))
        nz = mcolors.Normalize()
        nz.autoscale(C)

        ax.quiver(x_new[:-1], y_new[:-1], 
                  x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1], 
                  scale_units='xy', angles='xy', 
                  scale=1, zorder=1, color = cm.jet(nz(C)), 
                  width=0.0025)
        
        ax.scatter(x_new[0], y_new[0], 25, marker='o', color='blue', zorder=2) 
        ax.scatter(x_new[-1], y_new[-1], 25, marker='^', color='red', zorder=2) 

        plt.title("Elephant agent trajectory: " + agent_id)
        plt.savefig(os.path.join(folder, self.now, "output_files", "trajectory_on_LULC_" + agent_id + "_v1.png"), dpi = 300, bbox_inches = 'tight')
        
        if self.track_in_mlflow == True:
            mlflow.log_figure(fig, "trajectory_on_LULC_" + agent_id + "_v1.png")

        plt.close()
    #----------------------------------------------------------------------------------------------------
    def plot_ele_traj_on_slope(self, longitude, latitude, agent_id):
        
        ds = gdal.Open(os.path.join(self.folder_root, "env", "slope_matrix.tif"))
        data = ds.ReadAsArray()
        data = np.flip(data, axis=0)
        row_size, col_size = data.shape
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

        fig, ax = plt.subplots(figsize = (10,10))
        ax.yaxis.set_inverted(True)

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
        LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

        map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        map.imshow(data, cmap = "coolwarm", extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 0.75, vmin = 0, vmax = 2*self.slope_tolerance)

        map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
        map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        longitude, latitude = transform(inProj, outProj, longitude, latitude)
        x_new, y_new = map(longitude,latitude)
        C = np.arange(len(x_new))
        nz = mcolors.Normalize()
        nz.autoscale(C)

        ax.quiver(x_new[:-1], y_new[:-1],
                    x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1], 
                    scale_units='xy', angles='xy', 
                    scale=1, zorder=1, color = cm.jet(nz(C)), 
                    width=0.0025)
        
        ax.scatter(x_new[0], y_new[0], 25, marker='o', color='blue', zorder=2)
        ax.scatter(x_new[-1], y_new[-1], 25, marker='^', color='red', zorder=2)

        plt.title("Elephant agent trajectory: " + agent_id)
        plt.savefig(os.path.join(folder, self.now, "output_files", "trajectory_on_slope_" + agent_id + "_v1.png"), dpi = 300, bbox_inches = 'tight')

        if self.track_in_mlflow == True:
            mlflow.log_figure(fig, "trajectory_on_slope_" + agent_id + "_v1.png")

        plt.close()
    #----------------------------------------------------------------------------------------------------
    def create_trajectory_shapefile(self, agent_id, longitude, latitude):

        longitude = [float(x) for x in longitude]
        latitude = [float(x) for x in latitude]

        coords = list(zip(longitude, latitude))
        
        # Create LineString geometry from coordinates
        line = LineString(coords)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {
                'agent_id': [agent_id],
                'num_points': [len(coords)],
                'geometry': [line]
            },
            crs="EPSG:3857"  # WGS84 coordinate system
        )
        
        gdf.to_file(os.path.join(folder, self.now, "output_files", "trajectory_" + agent_id + "_.shp"), driver='ESRI Shapefile')
        return  
    #----------------------------------------------------------------------------------------------------
    def step(self):

        print("\n")
        print("day:", self.model_day, "hour:", self.hour_in_day, "minutes elapsed:", self.model_minutes, "time step:", self.model_time)

        self.update_hourly_temp()
        self.update_season()
        self.update_human_disturbance_explict()

        try:
            self.schedule.step()
        except:
            self.running = False

        self.datacollector.collect(self)

        #UPDATE TIME
        self.model_time = self.model_time + 1      
        self.model_minutes = self.model_time * 5
        self.model_hour = int(self.model_minutes/60)
        self.model_day = int(self.model_hour/24)
        self.hour_in_day =  self.model_hour - self.model_day*24
            
        for agent in self.schedule.agents:
            if isinstance(agent , Elephant):
                if agent.fitness <= 0:
                    self.schedule.remove(agent)
                    self.grid.remove_agent(agent)
                    self.running = False

        if self.model_time == self.max_time_steps:
            self.running = False

        if self.running == False:

            data_agents = self.datacollector.get_agent_vars_dataframe()

            #------------------------------------------------------------------------
            #PLOT ELEPHANT TRAJECTORY
            #------------------------------------------------------------------------
            agents = data_agents["AgentID"].unique()
            for agent in agents:
                if "herd" in agent or "bull" in agent:
                    ele_data = data_agents[data_agents["AgentID"] == agent]
                    self.create_trajectory_shapefile(agent, ele_data["longitude"].values, ele_data["latitude"].values)
                    self.plot_ele_traj_on_LULC(ele_data["longitude"].values, ele_data["latitude"].values, agent)
                    self.plot_ele_traj_on_slope(ele_data["longitude"].values, ele_data["latitude"].values, agent)
            #------------------------------------------------------------------------

            data_agents.to_csv(os.path.join(folder, self.now, "output_files", "agent_data.csv"), index = False)

            if self.track_in_mlflow == True:
                mlflow.log_metric("maximum time step", self.model_time)
                mlflow.log_artifact(os.path.join(folder, self.now, "output_files", "agent_data.csv"), "agent_data.csv")
                mlflow.end_run()

        return
    #----------------------------------------------------------------------------------------------------






















def batch_run_model(model_params, experiment_name, output_folder):

    freeze_support()

    global folder 
    folder = output_folder

    mlflow.set_experiment(experiment_name)

    batch_run(model_cls = conflict_model, 
                parameters = model_params, 
                number_processes = model_params["num_processes"], 
                iterations = model_params["iterations"],
                max_steps = model_params["max_time_steps"], 
                data_collection_period=1, 
                display_progress=True)

    return


