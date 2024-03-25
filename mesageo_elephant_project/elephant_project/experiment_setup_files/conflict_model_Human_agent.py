#importing packages
###########################################################################
#importing the agent class
from mesa_geo.geoagent import GeoAgent

from shapely import geometry, ops
from shapely.geometry import Point    
from math import radians, sin, cos, acos
from haversine import haversine
import networkx as nx
import osmnx as ox
from pyproj import Proj, transform
import numpy as np 
import os
import json

#To supress warning messages
import warnings
warnings.filterwarnings('ignore')

###########################################################################



#--------------------------------------------------------------------------------------------------------------------------------
#Each tick correspons to a temporal discretization of 5 minutes
#The movement and cognition of the agents has been modelled for the corresponding temporal discretization
#--------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------
#Human agent class: serves as superclass
class Humans(GeoAgent):
    """ Human agents"""

    def __init__(self,unique_id,model,shape):
        super().__init__(unique_id,model,shape)

        #agent initialisation
        init_file = open(os.path.join("experiment_setup_files","init_files","human_init.json"))
        init_data = json.load(init_file)


        #INITIALIZE BEHAVIOURAL PARAMETERS: FITNESS AND AGGRESSION
        #-------------------------------------------------------------------
        self.fitness = init_data["fitness_init"]    #fitness value of the individual at the start of the simulation
        self.elephant_habituation = init_data["elephant_habituation_init"]   #aggression value of the individual at the start of the simulation
        #-------------------------------------------------------------------

        #Home latitude and longitude of the agent
        self.home_lon = self.shape.x
        self.home_lat = self.shape.y
        self.escape_target_present = False

    def initialize_target_destination_nodes(self):     
        # To initialize target latitude and longitude
        return

    def initialize_distance_to_target(self):        
        # To initialize distance from home location to target location
        return

    def move_point(self,xnew,ynew):     #To move the agents in the geo-space
        """ Moves the agent in space"""
        #xnew: longitude
        #ynew: latitude
        return Point(xnew,ynew)
        
    def human_cognition(self):
        """The human agents cognition"""
        return

    def InflictDamage(self, conflict_agent):
        #Reduce the fitness of the elephant agent
        conflict_agent[0].fitness = conflict_agent[0].fitness - self.model.random.uniform(0, self.model.fitness_fn_decrement_elephants)
        return

    def EscapeMode(self, conflict_lon, conflict_lat):

        #set the target for escape
        self.target_for_escape(conflict_lon, conflict_lat)

        #walk to the target 
        self.walk_to_target_for_escape()
        
        return

    def distance(self,slat,elat,slon,elon):  
        #Returns the distance between current position and target position
        #Input CRS: epsg:3857 

        if slat==elat and slon==elon:
            return 0
            
        dist = np.sqrt((slat-elat)**2+(slon-elon)**2)
        return dist   #returns distance in metres

    def target_for_escape(self, conflict_lon, conflict_lat):
        """ Function returns the target for the human agent to move in case of threat to life.
        The agent moves to a random extremity of the search radius"""

        #sets the target to move for foraging
        #conflict_lon --- epsg:3857 ---
        #conflict_lat --- epsg:3857 ---

        if self.escape_target_present == True:      #If target to escape is already assigned
            return

        dx = conflict_lon - self.shape.x
        dy = conflict_lat - self.shape.y

        #setting target latitude and longitude for escape away from the conlict location
        if dx >= 0 and dy >= 0:
            self.target_lat_escape = self.shape.y - self.model.random.uniform(0, self.model.escape_radius_humans)
            self.target_lon_escape = self.shape.x - self.model.random.uniform(0, self.model.escape_radius_humans)

        if dx >= 0 and dy <= 0:
            self.target_lat_escape = self.shape.y + self.model.random.uniform(0, self.model.escape_radius_humans)
            self.target_lon_escape = self.shape.x - self.model.random.uniform(0, self.model.escape_radius_humans)

        if dx <= 0 and dy >= 0:
            self.target_lat_escape = self.shape.y - self.model.random.uniform(0, self.model.escape_radius_humans)
            self.target_lon_escape = self.shape.x + self.model.random.uniform(0, self.model.escape_radius_humans)

        if dx <= 0 and dy <= 0:
            self.target_lat_escape = self.shape.y + self.model.random.uniform(0, self.model.escape_radius_humans)
            self.target_lon_escape = self.shape.x + self.model.random.uniform(0, self.model.escape_radius_humans)

        self.escape_target_present = True
        return

    def walk_to_target_for_escape(self):
        self.shape = self.move_point(self.target_lon_escape,self.target_lat_escape)
        return

    def step(self):

        #start = timeit.default_timer()
        self.human_cognition()
        #stop = timeit.default_timer()
        #execution_time = stop - start
        #print("Human agent step: "+str(execution_time)+" seconds") # It returns time in seconds

        #Remove Dead Agents
        if self.fitness <= 0:
            self.model.agents_to_remove.add(self)
        return

    def __repr__(self):
        return "Agent_Human_ " + str(self.unique_id)
#--------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------------------------
#Commuters agent class: serves as superclass
class Commuters(Humans):
    """ Commuters: Both agricultural labourers/ Cultivators and other workers who commutes to their job location daily"""
    
    #agricultural labourers/Cultivators: commutes to agricultural plots
    #other workers: commutes to commercial centres

    def __init__(self,unique_id,model,shape):
        super().__init__(unique_id,model,shape)
        self.target_reached = False     
        self.dist_to_target = None
        self.next = 0
        self.on_ward = True
        self.return_journey = False

    def distance_calculator(self,slat,elat,slon,elon):  
        # returns the distance between current position and target position
        # slat: current latitude  --- epsg:4326 ---
        # elat: target latitude  --- epsg:4326 ---
        # slon: current longitude   --- epsg:4326 ---
        # elon: target longitude   --- epsg:4326 ---

        if slat==elat and slon==elon:
            return 0

        slat = radians(slat)   
        slon = radians(slon)
        elat = radians(elat)
        elon = radians(elon)
        dist = 6371.01 * acos(sin(slat)*sin(elat)  + cos(slat)*cos(elat)*cos(slon - elon))   #unit of distance: Km
        return dist*1000    #returns distnce in metres

    def distance_calculator_epsg3857(self, source_lat, source_lon, target_lat, target_lon):
        #returns the distance between current position and target position
        #the latitude and longitude are in EPSG 3857
        
        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        source_lon, source_lat  = transform(inProj, outProj, source_lat, source_lon)
        target_lon, target_lat  = transform(inProj, outProj, target_lat, target_lon)
        distance = haversine((source_lon, source_lat),(target_lon, target_lat))*1000
        return distance    #Distance returned is in metres

    def walk_to_target(self, target_lat, target_lon):
        #The coordinates are in EPSG 3857

        dx=self.shape.x-target_lon
        dy=self.shape.y-target_lat

        if dx < self.model.xres/2 and self.model.xres/2 :   #The agent is close to the target
            self.target_reached = True
            self.move_point(target_lon,target_lat)
            return

        co_ord = complex(dx,dy)
        direction = np.angle([co_ord],deg=True)
        theta=self.model.random.uniform(direction-5,direction+5)
        dx=self.speed_walking*self.model.temporal_scale*60*np.cos(theta*0.0174533)    #speed is in m/s and Time is in minutes
        dy=self.speed_walking*self.model.temporal_scale*60*np.sin(theta*0.0174533)
        new_lat = dy+self.shape.y
        new_lon = dx+self.shape.x
        self.move_point(new_lon,new_lat)
        return

    def initialize_target_destination_nodes(self):
        self.orig_node = ox.get_nearest_node(self.model.road_network, (self.shape.y,self.shape.x), method='euclidean')   #pass co-ordinates in UTM 
        self.dest_node = ox.get_nearest_node(self.model.road_network, (self.target_lat,self.target_lon), method='euclidean')   #pass co-ordinates in UTM
        self.find_route() 

    def initialize_distance_to_target(self):
        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        longitude_x, latitude_y  = transform(inProj, outProj, self.shape.x, self.shape.y)
        target_lon,target_lat = transform(inProj, outProj, self.target_lon, self.target_lat)
        self.dist_to_target = self.distance_calculator(latitude_y,target_lat,longitude_x,target_lon)

    def find_route(self):
        """ Function returns the route for commute via road depending on the travel speed"""

        route = nx.shortest_path(self.model.road_network, source=self.orig_node, target=self.dest_node, weight='length')
        line =[]

        # the length of each edge traversed along the path
        lengths = ox.utils_graph.get_route_edge_attributes(self.model.road_network, route, 'length')

        # the total length of the path
        path_length = sum(lengths)

        if path_length==0:
            self.route_x = []
            self.route_y = []
            return

        dist = self.speed_vehicle*self.model.temporal_scale*60

        for i in range(0,len(route)-1):
            line.append(self.model.edges_proj.loc[(route[i],route[i+1],0)].geometry)
            
        multi_line = geometry.MultiLineString(line)
        merged_line = ops.linemerge(multi_line)
        distance_delta = dist
        distances = np.arange(0, merged_line.length, distance_delta)
        route = [merged_line.interpolate(distance) for distance in distances] + [merged_line.boundary[1]]

        self.route_x = []       #longitude
        self.route_y = []       #latitude

        self.route_x.append(self.home_lon)
        self.route_y.append(self.home_lat)

        for i in range(0,len(route)):
            self.route_x.append(route[i].x)
            self.route_y.append(route[i].y)

        self.route_x.append(self.target_lon)
        self.route_y.append(self.target_lat)
        return

    def MoveViaVehicle(self):

        if len(self.route_x) < 2:
            return

        if self.on_ward == True:        #Travelling to the destination
            x_new=self.route_x[self.next]
            y_new=self.route_y[self.next]

            self.shape=self.move_point(x_new,y_new)
            self.next=self.next+1

            if (x_new==self.route_x[-1]) and (y_new==self.route_y[-1]):     #End of the journey
                self.on_ward = False
                self.return_journey = False
                self.next=len(self.route_x)-1

        elif self.return_journey == True:       #returning back home
            x_new=self.route_x[self.next]
            y_new=self.route_y[self.next]

            self.shape=self.move_point(x_new,y_new)
            self.next=self.next-1

            if (x_new==self.route_x[0]) and (y_new==self.route_y[0]):     #End of the journey
                self.on_ward = False
                self.return_journey = False
                self.next=0
        return

    def InflictDamage(self, conflict_agent):
        #Reduce the fitness of the elephant agent
        conflict_agent[0].fitness = conflict_agent[0].fitness - self.model.random.uniform(0, self.model.fitness_fn_decrement_elephants)

        return

    def human_cognition(self):
        """The human agents cognition"""
        #If the distance is less than 2km, the agents walks to the target location
        #Else the agents commutes via vehicle

        #Checks for nearby agents withing some viscinity
        neighbors = self.model.grid.get_neighbors_within_distance(agent=self, distance=self.model.human_agent_visibility)  #neighbors is a generator object
        conflict_neighbor = [neighbor if "bull" in neighbor.unique_id else None for neighbor in neighbors]
        conflict_neighbor = set(conflict_neighbor)
        conflict_neighbor.remove(None)      #conflict_neighbor is a set object

        if len(conflict_neighbor) != 0:     #In conflict
            #Conflict Cognition

            #if not habituated to elephants
            if self.elephant_habituation < self.model.elephant_habituation_tolerance:
                #either Inflict Damage or Escape to Safety
                rand = self.model.random.uniform(0,1)

                if rand < self.model.action_probability:          #Inflict Damage with a probability of 0.5
                    self.InflictDamage(self.model.random.sample(conflict_neighbor, 1))      #Inflict damage on a random elephant agent in the viscinity  

                else:           #Escape to safety with a probability of 0.5
                    neighbor = self.model.random.sample(conflict_neighbor, 1)[0]
                    self.EscapeMode(neighbor.shape.x, neighbor.shape.y)
            
            else:

                if self.target_reached ==  True:
                    #At the target, move randomly
                    self.shape=self.move_point(self.shape.x+self.model.random.uniform(-10,10),self.shape.y+self.model.random.uniform(-10,10))

                #Going for work
                elif (self.model.hour_in_day >= self.onward_time and self.model.hour_in_day < self.return_time) and (self.target_reached == False):       #Time for the onward journey
                    if self.dist_to_target < 2000:
                        #walks to the target location
                        self.walk_to_target(self.target_lat, self.target_lon)

                    else:
                        #commutes via the road
                        if self.model.hour_in_day == self.onward_time:
                            self.on_ward = True 
                            self.return_journey =  False

                        if self.on_ward == True and self.return_journey ==  False:
                            self.next_target_lat = self.target_lat
                            self.next_target_lon = self.target_lon
                            self.MoveViaVehicle()

                        else:
                            self.shape=self.move_point(self.shape.x+self.model.random.uniform(-10,10),self.shape.y+self.model.random.uniform(-10,10))

                #Returning home
                elif (self.model.hour_in_day > self.onward_time and self.model.hour_in_day >= self.return_time) and (self.target_reached == False):
                    if self.dist_to_target < 2000:
                        #walks to the target location
                        self.walk_to_target(self.home_lat,self.home_lon)

                    else:

                        if self.model.hour_in_day == self.return_time:
                            self.on_ward = False
                            self.return_journey =  True

                        if self.on_ward == False and self.return_journey ==  True:
                            self.next_target_lat = self.home_lat
                            self.next_target_lon = self.home_lon

                        else:
                            self.shape=self.move_point(self.shape.x+self.model.random.uniform(-10,10),self.shape.y+self.model.random.uniform(-10,10))
                            
                        self.MoveViaVehicle()


        else:       #Not in conflict

            if self.target_reached ==  True:
                #At the target, move randomly
                self.shape=self.move_point(self.shape.x+self.model.random.uniform(-10,10),self.shape.y+self.model.random.uniform(-10,10))

            #Going for work
            elif (self.model.hour_in_day >= self.onward_time and self.model.hour_in_day < self.return_time) and (self.target_reached == False):       #Time for the onward journey
                if self.dist_to_target < 2000:
                    #walks to the target location
                    self.walk_to_target(self.target_lat, self.target_lon)

                else:
                    #commutes via the road
                    if self.model.hour_in_day == self.onward_time:
                        self.on_ward = True 
                        self.return_journey =  False

                    if self.on_ward == True and self.return_journey ==  False:
                        self.next_target_lat = self.target_lat
                        self.next_target_lon = self.target_lon
                        self.MoveViaVehicle()

                    else:
                        self.shape=self.move_point(self.shape.x+self.model.random.uniform(-10,10),self.shape.y+self.model.random.uniform(-10,10))

            #Returning home
            elif (self.model.hour_in_day > self.onward_time and self.model.hour_in_day >= self.return_time) and (self.target_reached == False):
                if self.dist_to_target < 2000:
                    #walks to the target location
                    self.walk_to_target(self.home_lat,self.home_lon)

                else:

                    if self.model.hour_in_day == self.return_time:
                        self.on_ward = False
                        self.return_journey =  True

                    if self.on_ward == False and self.return_journey ==  True:
                        self.next_target_lat = self.home_lat
                        self.next_target_lon = self.home_lon

                    else:
                        self.shape=self.move_point(self.shape.x+self.model.random.uniform(-10,10),self.shape.y+self.model.random.uniform(-10,10))
                        
                    self.MoveViaVehicle()
#--------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------------------------
class Cultivators_Agricultural_labourers(Commuters):

    #agricultural labourers/ Cultivators commutes to agricultural plots

    def __init__(self,unique_id,model,shape):
        super().__init__(unique_id,model,shape)

        #Initializing the onward and return time of the agents
        #We assume a uniform distribution 
        self.onward_time = self.model.random.randint(self.model.AW_onward_time_start,self.model.AW_onward_time_end)
        self.return_time = self.model.random.randint(self.model.AW_return_time_start,self.model.AW_return_time_end)
        
        #speed of commute
        #We assume a uniform distribution 
        self.speed_walking = self.model.random.uniform(self.model.speed_walking_start,self.model.speed_walking_end)
        self.speed_vehicle = self.model.random.uniform(self.model.speed_vehicle_start,self.model.speed_vehicle_end)

#--------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------------------------
class Other_workers(Commuters):
    """ workers commuting to commercial centres"""

    def __init__(self,unique_id,model,shape):
        super().__init__(unique_id,model,shape)

        #Initializing the onward and return time of the agents
        #We assume a uniform distribution 
        self.onward_time = self.model.random.randint(self.model.OW_onward_time_start,self.model.OW_onward_time_end)
        self.return_time = self.model.random.randint(self.model.OW_return_time_start,self.model.OW_return_time_end)
        
        #speed of commute
        #We assume a uniform distribution 
        self.speed_walking = self.model.random.uniform(self.model.speed_walking_start,self.model.speed_walking_end)
        self.speed_vehicle = self.model.random.uniform(self.model.speed_vehicle_start,self.model.speed_vehicle_end)

#--------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------------------------
class HomeBound(Humans):
    """ Stays at home """

    def initialize_target_destination_nodes(self):
        return

    def human_cognition(self):
        """The human agents cognition"""
        #Does nothing, remains at home always

        #Checks for nearby agents withing some viscinity
        neighbors = self.model.grid.get_neighbors_within_distance(agent=self, distance=self.model.human_agent_visibility)  #neighbors is a generator object
        conflict_neighbor = [neighbor if "bull" in neighbor.unique_id else None for neighbor in neighbors]
        conflict_neighbor = set(conflict_neighbor)
        conflict_neighbor.remove(None)      #conflict_neighbor is a set object

        if len(conflict_neighbor) != 0:     #In conflict
            #Conflict Cognition

            if self.elephant_habituation < self. model.elephant_habituation_tolerance:

                #either Inflict Damage or Escape to Safety
                rand = self.model.random.uniform(0,1)

                if rand < self.model.action_probability:          #Inflict Damage with a probability of 0.5
                    self.mode = "inflict_damage"
                    self.InflictDamage(self.model.random.sample(conflict_neighbor, 1))      #Inflict damage on a random elephant agent in the viscinity  

                else:           #Escape to safety with a probability of 0.5
                    self.mode = "escape_mode"
                    neighbor = self.model.random.sample(conflict_neighbor, 1)[0]
                    self.EscapeMode(neighbor.shape.x, neighbor.shape.y)

            else:   #Not in conflict --> stay at home
                self.shape = self.move_point(self.home_lon, self.home_lat)

        else:   #Not in conflict --> stay at home
            self.shape = self.move_point(self.home_lon, self.home_lat)

        return

#--------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------------------------
class RandomWalkers_homebound(Humans):
    """ Does random walk everyday near the home, but returns home at the end of the day"""

    def __init__(self,unique_id,model,shape):
        super().__init__(unique_id,model,shape)
        self.onward_time = self.model.random.randint(self.model.RW_onward_time_start,self.model.RW_onward_time_end)
        self.return_time = self.model.random.randint(self.model.RW_return_time_start,self.model.RW_return_time_end)

    def human_cognition(self):
        """The human agents cognition"""

        #Checks for nearby agents withing some viscinity
        neighbors = self.model.grid.get_neighbors_within_distance(agent=self, distance=self.model.human_agent_visibility)  #neighbors is a generator object
        conflict_neighbor = [neighbor if "bull" in neighbor.unique_id else None for neighbor in neighbors]
        conflict_neighbor = set(conflict_neighbor)
        conflict_neighbor.remove(None)      #conflict_neighbor is a set object

        if len(conflict_neighbor) != 0:     #In conflict
            #Conflict Cognition

            if self.elephant_habituation <= self.model.elephant_habituation_tolerance:

                #either Inflict Damage or Escape to Safety
                rand = self.model.random.uniform(0,1)

                if rand < self.model.action_probability:          #Inflict Damage with a probability of 0.5
                    self.mode = "inflict_damage"
                    self.InflictDamage(self.model.random.sample(conflict_neighbor, 1))      #Inflict damage on a random elephant agent in the viscinity  

                else:           #Escape to safety with a probability of 0.5
                    self.mode = "escape_mode"
                    neighbor = self.model.random.sample(conflict_neighbor, 1)[0]
                    self.EscapeMode(neighbor.shape.x, neighbor.shape.y)

            else:
                if self.model.hour_in_day >= self.return_time or self.model.hour_in_day <= self.onward_time:
                    self.shape=self.move_point(self.home_lon, self.home_lat)
                    self.mode = "at home"

                else:
                    shape_x,shape_y = self.shape.x+self.model.random.uniform(-100,100),self.shape.y+self.model.random.uniform(-100,100)
                    self.shape=self.move_point(shape_x,shape_y)

        else:       #Not in conflict

            if self.model.hour_in_day >= self.return_time or self.model.hour_in_day <= self.onward_time:
                self.shape=self.move_point(self.home_lon, self.home_lat)
                self.mode = "at home"

            else:
                shape_x,shape_y = self.shape.x+self.model.random.uniform(-100,100),self.shape.y+self.model.random.uniform(-100,100)
                self.shape=self.move_point(shape_x,shape_y)

        return

#--------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------------------------
class RandomWalkers_perpetual(Humans):
    """ Does random walk everyday, doesnt return home"""

    def __init__(self,unique_id,model,shape):
        super().__init__(unique_id,model,shape)

    def human_cognition(self):
        """The human agents cognition"""

        #Checks for nearby agents withing some viscinity
        neighbors = self.model.grid.get_neighbors_within_distance(agent=self,distance=self.model.human_agent_visibility)  #neighbors is a generator object
        conflict_neighbor = [neighbor if "bull" in neighbor.unique_id else None for neighbor in neighbors]
        conflict_neighbor = set(conflict_neighbor)
        conflict_neighbor.remove(None)      #conflict_neighbor is a set object

        if len(conflict_neighbor) != 0:     #In conflict
            #Conflict Cognition

            if self.elephant_habituation <= self.model.elephant_habituation_tolerance:

                #either Inflict Damage or Escape to Safety
                rand = self.model.random.uniform(0,1)

                if rand < self.model.action_probability:          #Inflict Damage with a probability of 0.5
                    self.mode = "inflict_damage"
                    self.InflictDamage(self.model.random.sample(conflict_neighbor, 1))      #Inflict damage on a random elephant agent in the viscinity  

                else:           #Escape to safety with a probability of 0.5
                    self.mode = "escape_mode"
                    neighbor = self.model.random.sample(conflict_neighbor, 1)[0]
                    self.EscapeMode(neighbor.shape.x, neighbor.shape.y)

            else:
                shape_x,shape_y = self.shape.x+self.model.random.uniform(-10,10),self.shape.y+self.model.random.uniform(-10,10)
                self.shape=self.move_point(shape_x,shape_y)

        else:       #Not in conlict
            shape_x,shape_y = self.shape.x+self.model.random.uniform(-10,10),self.shape.y+self.model.random.uniform(-10,10)
            self.shape=self.move_point(shape_x,shape_y)

        return

#--------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------------------------
class commuters_perpetual(Humans):
    """ Always on road"""
    #To simulate traffic flow on the roads

    def __init__(self,unique_id,model,shape):
        super().__init__(unique_id,model,shape)
        self.dist_to_target = None
        self.next = 0
        self.on_ward = True

    def distance_calculator(self,slat,elat,slon,elon):  
        # returns the distance between current position and target position
        # slat: current latitude  --- epsg:4326 ---
        # elat: target latitude  --- epsg:4326 ---
        # slon: current longitude   --- epsg:4326 ---
        # elon: target longitude   --- epsg:4326 ---

        if slat==elat and slon==elon:
            return 0
        slat = radians(slat)   
        slon = radians(slon)
        elat = radians(elat)
        elon = radians(elon)
        dist = 6371.01 * acos(sin(slat)*sin(elat)  + cos(slat)*cos(elat)*cos(slon - elon))   #unit of distance: Km
        return dist*1000    #returns distnce in metres

    def initialize_target_destination_nodes(self):
        self.orig_node = ox.get_nearest_node(self.model.road_network, (self.shape.y,self.shape.x), method='euclidean')   #pass co-ordinates in UTM 
        self.dest_node = ox.get_nearest_node(self.model.road_network, (self.target_lat,self.target_lon), method='euclidean')   #pass co-ordinates in UTM
        self.find_route() 

    def initialize_distance_to_target(self):
        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        longitude_x, latitude_y  = transform(inProj, outProj, self.shape.x, self.shape.y)
        target_lon,target_lat = transform(inProj, outProj, self.target_lon, self.target_lat)
        self.dist_to_target = self.distance_calculator(latitude_y,target_lat,longitude_x,target_lon)

    def find_route(self):
        """ Function returns the route for commute via road depending on the travel speed"""

        route = nx.shortest_path(self.model.road_network, source=self.orig_node, target=self.dest_node, weight='length')
        line =[]

        # the length of each edge traversed along the path
        lengths = ox.utils_graph.get_route_edge_attributes(self.model.road_network, route, 'length')

        # the total length of the path
        path_length = sum(lengths)

        if path_length == 0:
            self.route_x = []
            self.route_y = []
            return

        dist = self.model.random.uniform(50,500)

        for i in range(0,len(route)-1):
            line.append(self.model.edges_proj.loc[(route[i],route[i+1],0)].geometry)
            
        multi_line = geometry.MultiLineString(line)
        merged_line = ops.linemerge(multi_line)
        distance_delta = dist
        distances = np.arange(0, merged_line.length, distance_delta)
        route = [merged_line.interpolate(distance) for distance in distances] + [merged_line.boundary[1]]

        self.route_x = []       #longitude
        self.route_y = []       #latitude

        for i in range(0,len(route)):
            self.route_x.append(route[i].x)
            self.route_y.append(route[i].y)
        return

    def MoveViaVehicle(self):

        if self.on_ward == True:        #Travelling to the destination
            x_new = self.route_x[self.next]
            y_new = self.route_y[self.next]

            self.shape=self.move_point(x_new,y_new)
            self.next = self.next+1

            if (x_new == self.route_x[-1]) and (y_new == self.route_y[-1]):     #End of the journey
                self.on_ward = False 

        return

    def human_cognition(self):
        """The human agents cognition"""

        #Checks for nearby agents withing some viscinity
        neighbors = self.model.grid.get_neighbors_within_distance(agent=self, distance=self.model.human_agent_visibility)  #neighbors is a generator object
        conflict_neighbor = [neighbor if "bull" in neighbor.unique_id else None for neighbor in neighbors]
        conflict_neighbor = set(conflict_neighbor)
        conflict_neighbor.remove(None)      #conflict_neighbor is a set object

        if len(conflict_neighbor) != 0:     #In conflict
            #Conflict Cognition

            if self.elephant_habituation <= self.model.elephant_habituation_tolerance:

                #either Inflict Damage or Escape to Safety
                rand = self.model.random.uniform(0,1)

                if rand < self.model.action_probability:          #Inflict Damage with a probability of 0.5
                    self.mode = "inflict_damage"
                    self.InflictDamage(self.model.random.sample(conflict_neighbor, 1))      #Inflict damage on a random elephant agent in the viscinity  

                else:           #Escape to safety with a probability of 0.5
                    self.mode = "escape_mode"
                    neighbor = self.model.random.sample(conflict_neighbor, 1)[0]
                    self.EscapeMode(neighbor.shape.x, neighbor.shape.y)

            else:
                if self.on_ward == True:
                    self.next_target_lat = self.target_lat
                    self.next_target_lon = self.target_lon

                else:
                    self.target_lat = self.model.random.uniform(self.model.LAT_MIN_epsg3857,self.model.LAT_MAX_epsg3857)
                    self.target_lon = self.model.random.uniform(self.model.LON_MIN_epsg3857,self.model.LON_MAX_epsg3857)
                    self.initialize_target_destination_nodes()
                    self.initialize_distance_to_target()     
                    self.on_ward = True    
                    self.next = 0 

                try:
                    self.MoveViaVehicle()  

                except:
                    self.on_ward = False        

        else:       #Not in conflict

            if self.on_ward == True:
                self.next_target_lat = self.target_lat
                self.next_target_lon = self.target_lon

            else:
                self.target_lat = self.model.random.uniform(self.model.LAT_MIN_epsg3857,self.model.LAT_MAX_epsg3857)
                self.target_lon = self.model.random.uniform(self.model.LON_MIN_epsg3857,self.model.LON_MAX_epsg3857)
                self.initialize_target_destination_nodes()
                self.initialize_distance_to_target()  
                self.on_ward = True 
                self.next = 0  

            try:
                self.MoveViaVehicle()  

            except:
                self.on_ward = False  
            
        return

#--------------------------------------------------------------------------------------------------------------------------------