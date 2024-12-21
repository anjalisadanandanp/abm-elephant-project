
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
from shapely.geometry import Point      # for creating point geometries
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
from matplotlib.patches import Rectangle # for plotting
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


        #-------------------------------------------------------------------
        self.initialize_memory_matrix_with_knowledge_from_fringe()
        self.mode = self.model.random.choice(["RandomWalk", "TargetedWalk"])  
        self.food_consumed = 0                      
        self.visit_water_source = True            
        self.heading = self.model.random.uniform(0,360)
        self.ROW, self.COL = self.update_grid_index()  
        self.target_present = False                
        self.target_lat = None                     
        self.target_lon = None                      
        self.target_name = None
        self.radius_food_search = self.model.radius_food_search
        self.radius_water_search = self.model.radius_water_search
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
        water_memory=np.zeros_like(self.model.WATER)

        for i in range(0,self.model.row_size):
            for j in range(0,self.model.col_size):
                if self.model.random.uniform(0,1) < self.model.percent_memory_elephant:
                    food_memory[i,j] = self.model.FOOD[i][j]
                    water_memory[i,j] = self.model.WATER[i][j]

        self.food_memory = food_memory.tolist() 
        self.water_memory = water_memory.tolist()

        source = os.path.join(self.model.folder_root, "LULC.tif")
        with rio.open(source) as src:
            ras_meta = src.profile

        memory_loc = os.path.join(self.model.folder_root, "food_memory_" + str(self.unique_id) + ".tif")
        with rio.open(memory_loc, 'w', **ras_meta) as dst:
            dst.write(food_memory.astype('float32'), 1)

        memory_loc = os.path.join(self.model.folder_root, "water_memory_" + str(self.unique_id) + ".tif")
        with rio.open(memory_loc, 'w', **ras_meta) as dst:
            dst.write(water_memory.astype('float32'), 1)

        return 
    #-----------------------------------------------------------------------------------------------------
    def initialize_memory_matrix_with_knowledge_from_fringe(self):
        """ Function that assigns memory matrix to elephants. The elephant agent has knowledge of the fringe areas."""

        #self.knowlege_from_fringe : unit is in metres
        no_of_cells = self.model.random.randint(0, int(self.model.knowledge_from_fringe/self.model.xres))     #spatial resolution 
        
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
                    if self.model.random.uniform(0,1) < self.model.percent_memory_elephant:
                        food_memory[i,j] = self.model.FOOD[i][j]

        self.food_memory = food_memory.tolist()
        self.water_memory = water_memory.tolist()

        #saving the memory matrix as .tif file 
        import rasterio as rio
        source = os.path.join(self.model.folder_root, "env", "LULC.tif")
        with rio.open(source) as src:
            ras_meta = src.profile

        memory_loc = os.path.join(self.model.folder_root, "env", "food_memory_"+ str(self.unique_id) + ".tif")
        with rio.open(memory_loc, 'w', **ras_meta) as dst:
            dst.write(food_memory.astype('float32'), 1)

        memory_loc = os.path.join(self.model.folder_root, "env", "water_memory_"+ str(self.unique_id) + ".tif")
        with rio.open(memory_loc, 'w', **ras_meta) as dst:
            dst.write(water_memory.astype('float32'), 1)

        return 
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

            next_lon, next_lat = self.shape.x, self.shape.y     
            self.shape = self.move_point(next_lon, next_lat)

        elif self.mode == "TargetedMove":

            if self.target_present == False:
                filter = self.return_feasible_direction_to_move()
                self.target_for_foraging(filter)     
                self.target_name = "food" 

            next_lon, next_lat = self.targeted_walk()
            self.shape = self.move_point(next_lon, next_lat)

        elif self.mode == "Thermoregulation":
            if self.target_present == False:
                filter = self.return_feasible_direction_to_move()
                self.target_thermoregulation(filter)
                self.target_name = "thermoregulation"

            next_lon, next_lat = self.targeted_walk()
            self.shape = self.move_point(next_lon, next_lat)

        self.eat_food()
        self.drink_water()

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
            mode = "Thermoregulation"

        else:
            num = self.model.random.uniform(0,1)

            if self.mode == "RandomWalk":
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
    def targeted_walk(self):
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
    def target_for_foraging(self, filter):

        if self.target_present == True:  
            return

        coord_list=[]

        radius = int(self.model.terrain_radius*2/self.model.xres) + 1   #spatial resolution: xres

        row_start = self.ROW - radius//2
        col_start = self.COL - radius//2
        row_end = self.ROW + radius//2 + 1
        col_end = self.COL + radius//2 + 1

        for i in range(row_start,row_end):
            for j in range(col_start,col_end):
                if i == self.ROW and j == self.COL:
                    pass

                elif self.food_memory[i][j] > 0 and filter[i - row_start][j - col_start] == 1:
                    coord_list.append([i,j])

        if coord_list != []:
            x, y = self.model.random.choice(coord_list)
            lon, lat = self.model.pixel2coord(x, y)
            self.target_lon, self.target_lat = lon, lat
            self.target_present = True

        elif coord_list == []:
            self.target_present = False

        return
    #-----------------------------------------------------------------------------------------------------
    def target_thermoregulation(self, filter):

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
            lon, lat = self.model.pixel2coord(x, y)
            self.target_lon, self.target_lat = lon, lat
            self.target_present = True

        elif coord_list == []:
            self.target_present = False

        return
    #-----------------------------------------------------------------------------------------------------
    def drink_water(self):
        """ The elephant agent consumes water from the current cell it is located in"""

        row, col = self.update_grid_index()

        if "dry" in self.model.season:

            if self.model.WATER[row][col]>0:
                self.visit_water_source = True

        else: 

            if self.model.WATER[row][col]>0:
                self.visit_water_source = True

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
    def elephant_cognition(self):
        """Function to simulate the cognition of the elephant agent"""

        self.next_step_to_move()

        return
    #----------------------------------------------------------------------------------------------------
    def step(self):     
        """ Function to simulate the movement of the elephant agent"""

        self.elephant_cognition()

        self.ROW, self.COL = self.update_grid_index()
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
        tolerance, #parameter in terrain cost function
        terrain_radius, #parameter in terrain cost function
        knowledge_from_fringe,  #distance from the fringe where elephants knows food availability
        prob_crop_damage, #probability of damaging crop if entered an agricultural field
        prob_infrastructure_damage, #probability of damaging infrastructure if entered a settlement area
        thermoregulation_threshold, #threshold temperature for thermoregulation
        ):



        self.MAP_COORDS=[9.3245, 76.9974]   



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
        self.tolerance = tolerance
        self.terrain_radius = terrain_radius
        self.knowledge_from_fringe = knowledge_from_fringe
        self.prob_crop_damage = prob_crop_damage
        self.prob_infrastructure_damage = prob_infrastructure_damage
        self.thermoregulation_threshold = thermoregulation_threshold
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
        

        env_folder_seethathode = os.path.join("mesageo_elephant_project/elephant_project/", "experiment_setup_files","environment_seethathode", "Raster_Files_Seethathode_Derived", self.area[area_size], self.reso[spatial_resolution])
        shutil.copy(os.path.join(env_folder_seethathode, "DEM.tif"), os.path.join(self.folder_root, "env"))
        shutil.copy(os.path.join(env_folder_seethathode, "LULC.tif"), os.path.join(self.folder_root, "env"))
        shutil.copy(os.path.join(env_folder_seethathode, "population.tif"), os.path.join(self.folder_root, "env"))
        shutil.copy(os.path.join(folder, "food_matrix_" + str(self.prob_food_forest) + "_" + str(self.prob_food_cropland) + "_.tif"), os.path.join(self.folder_root, "env"))
        shutil.copy(os.path.join(folder, "water_matrix_" + str(self.prob_water_sources) + "_.tif"), os.path.join(self.folder_root, "env"))
        shutil.copy(os.path.join(folder, "landscape_cell_status.tif"), os.path.join(self.folder_root, "env"))



        self.DEM = self.DEM_study_area()
        self.SLOPE = self.SLOPE_study_area()
        self.LANDUSE = self.LANDUSE_study_area()
        self.FOOD = self.FOOD_MATRIX()
        self.WATER = self.WATER_MATRIX()
        self.LANDSCAPE_STATUS = self.LANDSCAPE_CELL_STATUS()
        self.AGRI_PLOTS, self.INFRASTRUCTURE = self.PROPERTY_MATRIX()
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
        LULC = os.path.join(self.folder_root, "env", "LULC.tif")
        ds = gdal.Open(LULC, 0)
        arr = ds.ReadAsArray()
        self.xmin, self.xres, self.xskew, self.ymax, self.yskew, self.yres = ds.GetGeoTransform()
        self.row_size, self.col_size =  arr.shape
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
            with open('mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/temperature/hourly_temp_2010_without_juveniles_' + str(self.thermoregulation_threshold) + '.pkl','rb') as f:
                self.hourly_temp = pickle.load(f)

        else:
            with open('mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/temperature/hourly_temp_2010_with_juveniles_' + str(self.thermoregulation_threshold) + '.pkl','rb') as f:
                self.hourly_temp = pickle.load(f)   
        self.update_hourly_temp()
        #-------------------------------------------------------------------





        self.initialize_bull_elephants()



    

        self.datacollector = DataCollector(model_reporters={},
                                           agent_reporters={
                                                "longitude": "shape.x", 
                                                "latitude": "shape.y",
                                                "mode": "mode",
                                                })

        
        #Collect data at the start of the simulation
        self.datacollector.collect(self)
    #-------------------------------------------------------------------
    def initialize_bull_elephants(self, **kwargs):
        """Initialize the elephant agents"""

        # coord_lat, coord_lon = self.elephant_distribution_random_init_forest()

        coord_lat = np.random.uniform(self.center_lat-10000, self.center_lat+10000, self.num_bull_elephants)
        coord_lon = np.random.uniform(self.center_lon-10000, self.center_lon+10000, self.num_bull_elephants)

        for i in range(0, self.num_bull_elephants):    #initializing bull elephants

            elephant=AgentCreator(Elephant,{"model":self})  
            this_x = np.array(coord_lon)[i] + self.random.randint(-10,10)
            this_y = np.array(coord_lat)[i] + self.random.randint(-10,10)
            newagent = elephant.create_agent(Point(this_x,this_y), "bull_"+str(i))
            newagent.Leader = True   #bull elephants are always leaders
            newagent.sex = "Male"
            newagent.age = self.random.randrange(15,60) 
            
            #Assign body weight of the elephant agent depending on the sex and age
            newagent.body_weight = self.assign_body_weight_elephants(newagent.age, newagent.sex)

            #Assign daily dry matter intake depending on the body weight
            newagent.daily_dry_matter_intake = self.assign_daily_dietary_requiremnt(newagent.body_weight)

            self.grid.add_agents(newagent)
            self.schedule.add(newagent)
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

        fid = os.path.join(self.folder_root, "env", "DEM.tif")

        DEM = gdal.Open(fid).ReadAsArray()  
        return DEM.tolist()  #Conversion to list so that the object becomes json serializable
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


        return slope.tolist()  #Conversion to list so that the object becomes json serializable
    #-----------------------------------------------------------------------------------------------------
    def LANDUSE_study_area(self):
        """ Returns the landuse model of the study area"""

        fid = os.path.join(self.folder_root, "env", "LULC.tif")

        LULC = gdal.Open(fid).ReadAsArray()  
        return LULC.tolist()  #Conversion to list so that the object becomes json serializable
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
        return slope.tolist()  #Conversion to list so that the object becomes json serializable
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
    def update_prob_drink_water(self):

        if self.season == "dry":
            self.prob_drink_water = self.prob_drink_water_dry
        
        else:
            self.prob_drink_water = self.prob_drink_water_wet

        return
    #----------------------------------------------------------------------------------------------------
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
            
        #reurns row, col
        return i, j
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

        fig_background, ax_background = plt.subplots(figsize = (10,10))
        ax_background.yaxis.set_inverted(True)

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
        LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

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
        
        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        longitude, latitude = transform(inProj, outProj, longitude, latitude)
        x_new, y_new = map(longitude,latitude)
        C = np.arange(len(x_new))
        nz = mcolors.Normalize()
        nz.autoscale(C)

        ax_background.quiver(x_new[:-1], y_new[:-1], x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1], 
                                        scale_units='xy', angles='xy', scale=1, zorder=1, color = cm.jet(nz(C)), width=0.0025)
        
        ax_background.scatter(x_new[0], y_new[0], 25, marker='o', color='blue', zorder=2) 
        ax_background.scatter(x_new[-1], y_new[-1], 25, marker='^', color='red', zorder=2) 

        plt.title("Elephant agent trajectory:" + agent_id)
        plt.savefig(os.path.join(folder, self.now, "output_files", "trajectory_on_LULC_" + agent_id + "_.png"), dpi = 300, bbox_inches = 'tight')
    #----------------------------------------------------------------------------------------------------
    def step(self):

        print("day:", self.model_day, "hour:", self.model_hour, "minutes:", self.model_minutes)

        self.update_hourly_temp()
        self.update_season()
        self.update_prob_drink_water()
        self.schedule.step()
        self.datacollector.collect(self)

        #UPDATE TIME
        self.model_time = self.model_time + 1      
        self.model_minutes = self.model_time * 5
        self.model_hour = int(self.model_minutes/60)
        self.model_day = int(self.model_hour/24)
        self.hour_in_day =  self.model_hour - self.model_day*24

        if self.model_time == self.max_time:
            self.running = False

            data_agents = self.datacollector.get_agent_vars_dataframe()

            #------------------------------------------------------------------------
            #PLOT ELEPHANT TRAJECTORY
            #------------------------------------------------------------------------
            agents = data_agents["AgentID"].unique()
            for agent in agents:
                if "herd" in agent or "bull" in agent:
                    ele_data = data_agents[data_agents["AgentID"] == agent]
                    self.plot_ele_traj_on_LULC(ele_data["longitude"].values, ele_data["latitude"].values, agent)
            #------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------






















def batch_run_model(model_params, number_processes, iterations, output_folder):

    freeze_support()

    global folder 
    folder = output_folder

    path_to_folder = os.path.join(folder)
    os.makedirs(path_to_folder, exist_ok=True)

    environment(prob_food_in_forest = model_params["prob_food_forest"],
                        prob_food_in_cropland = model_params["prob_food_cropland"],
                        prob_water_sources = model_params["prob_water_sources"], 
                        max_food_val_forest = model_params["max_food_val_forest"],
                        max_food_val_cropland = model_params["max_food_val_cropland"],
                        output_folder=output_folder).main()

    batch_run(model_cls = conflict_model, 
                parameters = model_params, 
                number_processes = number_processes, 
                iterations = iterations,
                max_steps = model_params["max_time"], 
                data_collection_period=1, 
                display_progress=True)

    return


