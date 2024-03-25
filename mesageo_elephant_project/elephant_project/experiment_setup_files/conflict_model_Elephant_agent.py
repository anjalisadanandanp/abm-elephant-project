#importing packages
#########################################

#importing agent class
from mesa_geo.geoagent import GeoAgent

from shapely.geometry import Point
import numpy as np
import math
import json
import os



#To supress warning commands
import warnings
warnings.filterwarnings('ignore')
#########################################




#--------------------------------------------------------------------------------------------------------------------------------
#Each tick corresponds to a temporal resolution of 5 minutes
#The movement and cognition of the agents has been modelled for the corresponding temporal resolution
#--------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------------------------
class Elephant(GeoAgent):
    """ Foraging agent """
    def __init__(self,unique_id,model,shape):
        super().__init__(unique_id,model,shape) 


        #agent initialisation
        init_file = open(os.path.join("experiment_setup_files","init_files","elephant_init.json"))
        init_data = json.load(init_file)


        #INITIALIZE BEHAVIOURAL PARAMETERS: FITNESS AND AGGRESSION
        #-------------------------------------------------------------------
        self.fitness = init_data["fitness_init"]    #fitness value of the individual at the start of the simulation
        self.aggress_factor = init_data["aggression_init"]   #aggression value of the individual at the start of the simulation
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
        self.mode = self.model.random.choice(["RandomWalk", "TargetedWalk"])  #Initial behavioural state

        self.num_days_water_source_visit = 0        #days since last water source visit
        self.num_days_food_depreceation = 0         #days since dietary requirement was satisfied
        self.visit_water_source = True            #if water source was visited the previous day
        self.food_consumed = 0                      #Food consumed since the beginning of the day

        self.heading = 0                            #Direction of movement
        self.ROW, self.COL = self.update_grid_index()   #Grid indices

        self.target_present = False                 #if target exists for movement 
        self.target_lat = None                      #target latitude
        self.target_lon = None                      #target longitude

        self.radius_food_search = self.model.radius_food_search
        self.radius_water_search = self.model.radius_water_search
        self.radius_forest_search =  model.radius_forest_search    #The search radius within which the agent searches for forest to escape to when threatened
        
        self.danger_to_life = False                 #if there is a danger to life for the agent

        self.distance_to_water_source = 0
    #-------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------------
    def move_point(self,xnew,ynew):     
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

        #saving the memory matrix as .tif file 
        # import rasterio as rio
        # source = os.path.join(self.model.folder_root, "LULC.tif")
        # with rio.open(source) as src:
        #     ras_meta = src.profile
        # memory_loc = os.path.join("simulation_results_server_run", "food_memory_"+ str(self.unique_id) + ".tif")
        # with rio.open(memory_loc, 'w', **ras_meta) as dst:
        #     dst.write(food_memory.astype('float32'), 1)
        # memory_loc = os.path.join("simulation_results_server_run", "water_memory_"+ str(self.unique_id) + ".tif")
        # with rio.open(memory_loc, 'w', **ras_meta) as dst:
        #     dst.write(water_memory.astype('float32'), 1)

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

        # #saving the memory matrix as .tif file 
        # import rasterio as rio
        # source = os.path.join(self.model.folder_root, "LULC.tif")
        # with rio.open(source) as src:
        #     ras_meta = src.profile
        # memory_loc = os.path.join("simulation_results_server_run", "food_memory_"+ str(self.unique_id) + ".tif")
        # with rio.open(memory_loc, 'w', **ras_meta) as dst:
        #     dst.write(food_memory.astype('float32'), 1)
        # memory_loc = os.path.join("simulation_results_server_run", "water_memory_"+ str(self.unique_id) + ".tif")
        # with rio.open(memory_loc, 'w', **ras_meta) as dst:
        #     dst.write(water_memory.astype('float32'), 1)

        return food_memory, water_memory

    #-----------------------------------------------------------------------------------------------------
    def initialize_memory_matrix_with_knowledge_from_fringe(self):
        """ Function that assigns memory matrix to elephants"""

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


        # #saving the memory matrix as .tif file 
        # import rasterio as rio
        # source = os.path.join(self.model.folder_root, "LULC.tif")
        # with rio.open(source) as src:
        #     ras_meta = src.profile
        # memory_loc = os.path.join("simulation_results_server_run", "food_memory_"+ str(self.unique_id) + ".tif")
        # with rio.open(memory_loc, 'w', **ras_meta) as dst:
        #     dst.write(food_memory.astype('float32'), 1)
        # memory_loc = os.path.join("simulation_results_server_run", "water_memory_"+ str(self.unique_id) + ".tif")
        # with rio.open(memory_loc, 'w', **ras_meta) as dst:
        #     dst.write(water_memory.astype('float32'), 1)

        return food_memory, water_memory

    #-----------------------------------------------------------------------------------------------------
    def distance_calculator_epsg3857(self,slat,elat,slon,elon):  
        #Returns the distance between current position and target position
        #Input CRS: epsg:3857 

        if slat==elat and slon==elon:
            return 0
            
        dist = np.sqrt((slat-elat)**2+(slon-elon)**2)
        return dist   #returns distance in metres

    #----------------------------------------------------------------------------------------------------- 
    def update_grid_index(self):

        # calculate indices and index array
        row, col = self.model.get_indices(self.shape.x, self.shape.y)

        if row == (self.model.row_size - 1) or row == 0 or col == (self.model.col_size - 1) or col == 0:      #if agent is at the border
            lon, lat = self.model.pixel2coord(row, col)
            self.shape = self.move_point(lon, lat)

        return row, col

    #-----------------------------------------------------------------------------------------------------  
    def next_step_to_move(self):
        """how the elephant agent moves from the current co-ordinates to the next"""

        mode = self.current_mode_of_the_agent()   
        self.mode = mode

        if mode == "RandomWalk":
            next_lon, next_lat = self.correlated_random_walk_without_terrain_factor()   
            row, col = self.model.get_indices(next_lon, next_lat)
            if self.model.LANDUSE[row][col] == 10 and self.aggress_factor < self.model.aggress_threshold_enter_cropland:
                next_lon, next_lat = self.shape.x, self.shape.y     
            self.shape = self.move_point(next_lon, next_lat)

        elif mode == "TargetedMove":
            rand_num = np.random.uniform(0,1)
            if rand_num <= self.model.prob_drink_water:
                self.target_to_drink_water(discount_factor=self.model.discount, tolerance=self.model.tolerance)    
                self.distance_to_water_source = self.distance_calculator_epsg3857(self.shape.y, self.target_lat, self.shape.x, self.target_lon)      #distance is in meters     
                row, col = self.model.get_indices(self.target_lon, self.target_lat)
                self.radius_water_search = self.model.radius_water_search

            else:
                self.target_for_foraging(discount_factor=self.model.discount, tolerance=self.model.tolerance)     
                row, col = self.model.get_indices(self.target_lon, self.target_lat)

            ################################################################################################################
            if self.aggress_factor < self.model.aggress_threshold_enter_cropland:
                if self.model.LANDUSE[row][col] == 10:      #target is a human habitated area, check for the disturbance 
                    if self.model.human_disturbance < self.model.disturbance_tolerance:     #proceed to forage
                        next_lon, next_lat = self.targeted_walk()
                        self.shape = self.move_point(next_lon, next_lat)

                    else:
                        self.target_for_escape()    #move to the forest
                        next_lon, next_lat = self.targeted_walk()
                        self.shape = self.move_point(next_lon, next_lat)

                else:   #target is not a human habitated area, proceed to forage
                    next_lon, next_lat = self.targeted_walk()
                    self.shape = self.move_point(next_lon, next_lat)

            else:
                next_lon, next_lat = self.targeted_walk()
                self.shape = self.move_point(next_lon, next_lat)
            ################################################################################################################

        elif mode == "EscapeMode":      #When in conflict with humans and aggression is low
            self.target_for_escape()    #set the target to move to incase there is no target
            next_lon, next_lat = self.targeted_walk()
            self.shape = self.move_point(next_lon, next_lat)

        elif mode == "InflictDamage":       #When in conflict with humans and aggression is high
            self.InflictDamage()      

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
        #state1_to_state2 = 0.1224693
        #state2_to_state1 = 0.0903915

        if self.model.elephant_agent_visibility != None:
            neighbors = self.model.grid.get_neighbors_within_distance(agent=self, distance=self.model.elephant_agent_visibility)  #neighbors is a generator object
            conflict_neighbor = [neighbor if "Human" in neighbor.unique_id else None for neighbor in neighbors]
            conflict_neighbor = set(conflict_neighbor)
            conflict_neighbor.remove(None)
            self.update_danger_to_life(conflict_neighbor)       #update danger_to_life

        if self.danger_to_life==True and self.aggress_factor < self.model.aggress_threshold_inflict_damage:    
            #Supercedes all other requirements --> This state is encountered when in conflict with Human agents
            mode="EscapeMode"

        elif self.danger_to_life==True and self.aggress_factor >= self.model.aggress_threshold_inflict_damage:
            #Supercedes all other requirements --> This state is encountered when in conflict with Human agents
            mode="InflictDamage"

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

        return mode

    #-------------------------------------------------------------------------------------------------
    def update_danger_to_life(self, conflict_neighbor):
            
        if len(conflict_neighbor) >= 1:     #There are human agents in the viscinity
            if self.human_habituation < self.model.human_habituation_tolerance:     #Not used to human agents
                self.danger_to_life = True
                self.model.conflict_location.add((self.shape.x, self.shape.y))
                return

            else:  #Used to human agents
                self.danger_to_life = False    

        else:
            self.danger_to_life = False    

        return 

    #-------------------------------------------------------------------------------------------------
    def correlated_random_walk_without_terrain_factor(self):

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
    def target_for_foraging(self, discount_factor, tolerance):

        # discount_factor: varies from 0 to 1. Use in calculating the cost of movement.
        # Tolerance: For selecting the directions (Directions are selected with the cost of movement less than the tolerance). 

        radius = int(self.model.terrain_radius/self.model.xres)     #spatial resolution: xres

        if self.target_present == True:     #If target already exists
            return

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
            if cost[i] < tolerance:
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

        coord_list=[]

        row_start = int(row_start)
        row_end = int(row_end)
        col_start = int(col_start)
        col_end = int(col_end)
        
        ##########################################################################
        if self.food_habituation < self.model.food_habituation_threshold:
            for i in range(row_start,row_end):
                for j in range(col_start,col_end):
                    if i == self.ROW and j == self.COL:
                        pass

                    elif self.food_memory[i][j]>0:
                        coord_list.append([i, j])

        else:
            if self.num_days_food_depreceation > 3:
                val = self.model.plantation_proximity[self.ROW][self.COL]
                coord_list.append([self.ROW,self.COL])

                for i in range(row_start,row_end):
                    for j in range(col_start,col_end):
                        if i == self.ROW and j == self.COL:
                            pass

                        elif self.model.plantation_proximity[i][j] <= val:
                            coord_list.append([i,j]) 

            else:
                for i in range(row_start,row_end):
                    for j in range(col_start,col_end):
                        if i == self.ROW and j == self.COL:
                            pass

                        elif self.food_memory[i][j]>0:
                            coord_list.append([i, j])                  
        ##########################################################################     

        #move to a random cell within the search radius
        if coord_list==[]:
            for _ in range(4):
                i = self.model.random.randint(row_start, row_end)
                j = self.model.random.randint(col_start, col_end)
                coord_list.append([i,j])


        x, y = self.model.random.choice(coord_list)
        lon = self.model.xres * 0.5  + self.model.xmin + y * self.model.xres
        lat = self.model.yres * 0.5  + self.model.ymax + x * self.model.yres
        self.target_lon, self.target_lat = lon, lat
        self.target_present = True

        return

    #-----------------------------------------------------------------------------------------------------
    def target_to_drink_water(self, discount_factor, tolerance):

        """ Function returns the target for the elephant agent to move.
        The target is selected from the memory matrix, where the elephant agent thinks it can find water.
        Barrier to movement is considered while selecting the target to move.
        if there is no cell with water available, moves to a random extremity of the search radius"""

        # discount_factor: varies from 0 to 1. Use in calculating the cost of movement.
        # Tolerance: For selecting the directions (Directions are selected with the cost of movement less than the tolerance). 

        radius = int(self.model.terrain_radius/self.model.xres)     #spatial resolution: xres

        if self.target_present == True:     #If target already exists
            return

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
            if cost[i] < tolerance:
                theta.append(direction[i]) 

        #if no direction available, choose direction with minimum movement cost
        if theta == []:    
            min_cost =  cost[0]
            for i in range(1, len(direction)):
                if cost[i] < min_cost:
                    min_cost = cost[i]
                    theta = direction[i] 

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

        coord_list=[]

        row_start = int(row_start)
        row_end = int(row_end)
        col_start = int(col_start)
        col_end = int(col_end)
        
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
    def target_for_escape(self):

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

        neighbors = self.model.grid.get_neighbors_within_distance(agent=self, distance=self.model.Elephant_agent_visibility)  #neighbors is a generator object
        conflict_neighbor = [neighbor if "Human" in neighbor.unique_id else None for neighbor in neighbors]
        conflict_neighbor = set(conflict_neighbor)
        conflict_neighbor.remove(None)
        
        for neighbor in conflict_neighbor:
            neighbor.fitness = neighbor.fitness - self.model.random.uniform(0, self.model.fitness_fn_decrement_humans)

        return

    #----------------------------------------------------------------------------------------------------
    def drink_water(self):
        """ The elephant agent consumes water from the current cell it is located in"""

        row, col = self.update_grid_index()

        if self.model.WATER[row][col]>0:
            self.visit_water_source = True

            #update fitness value
            self.update_fitness_value(self.model.fitness_increment_when_drinks_water)

        return

    #---------------------------------------------------------------------------------------------------
    def eat_food(self):

        """ The elephant agent consumes food from the current cell it is located in"""

        row, col = self.update_grid_index()

        if self.model.FOOD[row][col]>0:
            num = self.model.FOOD[row][col]
            self.food_consumed += self.model.random.uniform(0,self.model.FOOD[row][col])
            self.model.FOOD[row][col] -= self.food_consumed
            self.food_memory[row][col] -= self.food_consumed

            if self.model.FOOD[row][col]<0:
                self.model.FOOD[row][col] = 0
                self.food_memory[row][col] = 0

            #update fitness value
            self.update_fitness_value(self.model.fitness_increment_when_eats_food*num)

        return

    #----------------------------------------------------------------------------------------------------
    def crop_infrastructure_damage(self):

        row, col = self.update_grid_index()

        num_1 = np.random.uniform(0,1)
        num_2 = np.random.uniform(0,1)

        if self.model.INFRASTRUCTURE[row][col] != 0:
            if num_1 < self.model.prob_infrastructure_damage:
                self.model.infrastructure_damage[row][col] = 1

        if self.model.AGRI_PLOTS[row][col] != 0:
            if num_2 < self.model.prob_crop_damage: 
                self.model.crop_damage[row][col] = 1

        return

    #----------------------------------------------------------------------------------------------------
    def update_aggression_factor(self, val):
        """The function updates the aggression factor of the agent"""

        aggress_factor = self.aggress_factor
        aggress_factor += val

        if aggress_factor<0:
            self.aggress_factor = 0

        elif aggress_factor>1:
            self.aggress_factor = 1

        else:
            self.aggress_factor = aggress_factor

        return 

    #----------------------------------------------------------------------------------------------------
    def update_fitness_value(self, val):
        """The function updates the fitness value of the agent"""

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
    def elephant_cognition(self):

        #behaviour of elephant in each time step

        #start of the day: set the goals to be attained ---> food and water intake  
        # ---> update aggression and fitness based on the food and water intake in the previous day

        #movement simulation:
        #If forage is available in the surroundings ---> perform random walk and consume food
        #If forage is absent ---> move to a target from the memory within the search radius
        #Update fitness function and aggression factor

        if (self.model.model_time%288) == 0 and self.model.model_time>288:
            #print("start of the day")            #we are at the start of the day

            #update the parameters at the start of the day

            self.food_goal = self.daily_dry_matter_intake    #set the food goal

            #Updating parameters based on whether the agent visited water source the previous day
            if self.visit_water_source == False:    
                self.num_days_water_source_visit += 1     #set the number of days since the agent visited water source
                self.update_fitness_value(-0.05)     #Update fitness value  ---> fitness depreceation
                self.update_aggression_factor(0)      #Update aggression value  ---> agression increase

            else:
                if self.num_days_water_source_visit == 0:
                    self.update_fitness_value(0)        #Update fitness value  ---> Fitness improvement
                    self.update_aggression_factor(0)        #Update aggression value: aggression remains unaffected because the agent drank water the previous day
                
                else:                   #if its been days since agent drank water
                    self.update_fitness_value(0)        #Update fitness value ---> Fitness improvement
                    self.update_aggression_factor(0)        #Update aggression value: counter the increase in aggression due to water deprivation
                
                self.visit_water_source == False
                self.num_days_water_source_visit = 0      #set the number of days since the agent visited water source

            #Updating parameters based on whether the agent satisfied dietary intake the previous day
            if self.food_consumed < self.daily_dry_matter_intake:
                self.num_days_food_depreceation += 1
                self.update_fitness_value(-0.05)     #Update fitness value
                self.update_aggression_factor(0)      #Update aggression value
    
            else:
                if self.num_days_food_depreceation == 0:
                    self.update_fitness_value(0)     #Update fitness value
                    self.update_aggression_factor(0)      #Update aggression value: aggression remains unaffected because the agent satisfied its dietary needs the previous day
                
                else:                   #if its been days since agent satisfied its dietary requirement
                    self.update_fitness_value(0)     #Update fitness value
                    self.update_aggression_factor(0)      #Update aggression value: counter the increase in aggression due to food deprivation
                
                self.num_days_food_depreceation = 0

            self.food_consumed = 0   #set the food consumed as zero at the start of the day

        #movement simulation
        self.next_step_to_move()
        return

    #----------------------------------------------------------------------------------------------------
    def step(self):        

        #start = timeit.default_timer()
        self.elephant_cognition()
        #stop = timeit.default_timer()
        #execution_time = stop - start
        #print("Elephant agent step: "+str(execution_time)+" seconds") # It returns time in seconds

        #update the ROW and COLumn indices
        self.ROW, self.COL = self.update_grid_index()

        #Remove Dead Agents
        if self.fitness <= 0:
            self.model.agents_to_remove.add(self)