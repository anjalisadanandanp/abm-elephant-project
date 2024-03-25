"""
The model class initialization, scheduling, and datacollection.

1. INITIALISATION 
---------------------------------------------------------------------------------------------------------------
Elephant agent initialization: Three functions are implemented
1. Random initialization in the evergreen broadleaf forests (elephant_distribution_random_init_forest)
2. Initialization in the evergreen broadleaf forests close to the fringe areas (elephant_distribution_close_to_fringe)
3. Distribution considering elevation, proximity and landuse types (elephant_distribution_modified)
---------------------------------------------------------------------------------------------------------------
2. AGENT SCHEDULING: Random activation
---------------------------------------------------------------------------------------------------------------
3. DATACOLLECTION: at every step
---------------------------------------------------------------------------------------------------------------
"""




#importing packages
##################################################################

#MESA modules
#----------------------------------------
#MODEL CLASS
from experiment_setup_files.Mesa_Model_class import Model      #modified implementation (with bug fix)

#AGENT SCHEDULES
from experiment_setup_files.Mesa_agent_scheduling import RandomActivation    #Random activation

#GEOSPACE
from experiment_setup_files.Mesa_geospace import GeoSpace          #modified implementation

#AGENT_CREATOR
from mesa_geo.geoagent import AgentCreator          #Default implementation

#DATA_COLLECTOR
#from mesa.datacollection import DataCollector           #Default implementation  
#from experiment_setup_files.Mesa_Datacollector_v1_0 import DataCollector       #all agent data collected
from experiment_setup_files.Mesa_Datacollector_v1_1 import DataCollector       #only elephant agent data is collected
#----------------------------------------

#Raster manipulation tools    
from osgeo import gdal
import rasterio as rio
from shapely.geometry import Point 
from pyproj import Proj, transform      
import pandas as pd  
import  numpy as np
import os
import json
import osmnx as ox       

#to supress warnings
import warnings     
warnings.filterwarnings('ignore')   

#importing agent classes
from experiment_setup_files.conflict_model_Elephant_agent import Elephant    
from experiment_setup_files.conflict_model_Human_agent import Cultivators_Agricultural_labourers
from experiment_setup_files.conflict_model_Human_agent import Other_workers
from experiment_setup_files.conflict_model_Human_agent import HomeBound
from experiment_setup_files.conflict_model_Human_agent import RandomWalkers_homebound
from experiment_setup_files.conflict_model_Human_agent import RandomWalkers_perpetual
from experiment_setup_files.conflict_model_Human_agent import commuters_perpetual

#-------------------------------------------------------------------




#THE MODEL CLASS IMPLEMENTATION
class Conflict_model(Model):
    """ 
    Model class: Elephant Human interaction
    """

    #Study area co-ordinates
    MAP_COORDS=[9.3245, 76.9974]       

    #Model Initialization
    def __init__(self,
        #Following parameters corresponds with simulation set up:
        temporal_scale, #temporal resolution of one tick in minutes
        num_bull_elephants, #number of solitary bull elephants in the simulation
        max_time, #maximum simulation time (in ticks)
        area_size, #simulation area in sq. km
        resolution, #spatial resolution of the simulation area

        #Following parameters corresponds with resource(food/water) set up in the environment:
        prob_food_forest, #probability of food in the forest
        prob_food_cropland, #probability of food in the forest
        prob_water, #probability of water in the forest

        #Elephant agent initialisation at the start of the simulation
        prob_drink_water, #probability of the elephant agent drinking water in each tick
        percent_memory_elephant, #percentage memory of the landscape cells by the elephant agents at the start of the simulation
        radius_food_search, #radius within which the elephant agent searches for food
        radius_water_search, #radius within which the elephant agent searches for water
        movement_fitness_depreceation, #fitness depreceation in each tick
        fitness_increment_when_eats_food, #fitness increment when consumes food
        fitness_increment_when_drinks_water, #fitness increment when drinks water
        fitness_threshold, #fitness threshold below which the elephant agent engages only in "TargetedWalk" mode
        discount, #parameter in terrain cost function
        tolerance, #parameter in terrain cost function
        terrain_radius, #parameter in terrain cost function

        #Human agent initialisation at the start of the smulation
        AW_onward_time_start, #Cultivators_Agricultural_labourers onward journey start time
        AW_onward_time_end, #Cultivators_Agricultural_labourers onward journey end time
        AW_return_time_start, #Cultivators_Agricultural_labourers return journey start time
        AW_return_time_end, #Cultivators_Agricultural_labourers return journey end time
        OW_onward_time_start, #Other_workers onward journey start time
        OW_onward_time_end, #Other_workers onward journey end time
        OW_return_time_start, #Other_workers return journey start time
        OW_return_time_end, #Other_workers return journey end time
        RW_onward_time_start, #RandomWalkers_homebound onward journey start time
        RW_onward_time_end, #RandomWalkers_homebound onward journey end time
        RW_return_time_start, #RandomWalkers_homebound return journey start time
        RW_return_time_end, #RandomWalkers_homebound return journey end time
        speed_walking_start, #minimum walking speed
        speed_walking_end, #maximum walking speed
        speed_vehicle_start, #minimum vehicular speed
        speed_vehicle_end, #maximum vehicular speed

        #Conflict parameters
        knowledge_from_fringe,  #distance from the fringe where elephants knows food availability
        human_agent_visibility, #distance within which human agents can perceive the presence of elephant agents
        elephant_agent_visibility, #distance within which elephant agents can perceive the presence of human agents
        prob_crop_damage, #probability of damaging crop if entered an agricultural field
        prob_infrastructure_damage, #probability of damaging infrastructure if entered a settlement area
        fitness_fn_decrement_humans, #the magnitude by which the elephant agents depreciates the fitness value of human agents in case of encounter
        fitness_fn_decrement_elephants, #the magnitude by which the human agents depreciates the fitness value of elephant agents in case of encounter
        escape_radius_humans, #the distance to which the human agent recedes in case of conflict with elephant agents
        radius_forest_search, #radius within which agent remembers the forest boundary, to escape in case of conflict with humans
        aggress_threshold_inflict_damage, #the aggression threshold used to simulate conflict with humans
        aggress_threshold_enter_cropland, #the aggression threshold above which the elephant agent dares to enter a cropland
        food_habituation_threshold, #the threshold above which the elephants are habituated towards crops
        human_habituation_tolerance, #the threshold above which the elephants are habituated towards humans
        elephant_habituation_tolerance, #the threshold above which the humans are habituated towards elephants
        disturbance_tolerance,
        action_probability
        ):

        #NOTE: Some parameter listed above are used to parameterize the submodels
        #They are not necessarily simulation parameters, but instead can be fixed

        #Folders to read the data files from depending upon the area and resolution
        self.area = {800:"area_800sqKm", 900:"area_900sqKm", 1000:"area_1000sqKm", 1100:"area_1100sqKm"}
        self.reso = {30: "reso_30x30", 60:"reso_60x60", 90:"reso_90x90", 120:"reso_120x120", 150:"reso_150x150"}
        self.delta = {800:14142, 900:15000, 1000:15811, 1100: 16583}  #distance in meteres from the center of the polygon area

        folder_path = os.path.join("experiment_setup_files","environment_seethathode","Raster_Files_Seethathode_Derived", self.area[area_size])
        self.folder_root = os.path.join(folder_path, self.reso[resolution])     #To read the required files




        #-------------------------------------------------------------------
        #Model: UserSettlable parameters
        #-------------------------------------------------------------------
        #SIMULATION PARAMETERS:
        self.temporal_scale = temporal_scale
        self.num_bull_elephants = num_bull_elephants     
        self.max_time = max_time
        self.area_size = area_size
        self.resolution = resolution  
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        #Environment: UserSettlable parameters
        #-------------------------------------------------------------------
        self.prob_food_forest = prob_food_forest
        self.prob_food_cropland = prob_food_cropland
        self.prob_water = prob_water
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        #Elephant agents: UserSettlable parameters
        #-------------------------------------------------------------------
        self.prob_drink_water = prob_drink_water
        self.percent_memory_elephant = percent_memory_elephant
        self.radius_food_search = radius_food_search
        self.radius_water_search = radius_water_search
        self.movement_fitness_depreceation = movement_fitness_depreceation
        self.fitness_increment_when_eats_food = fitness_increment_when_eats_food
        self.fitness_increment_when_drinks_water = fitness_increment_when_drinks_water
        self.fitness_threshold = fitness_threshold
        self.discount = discount
        self.tolerance = tolerance
        self.terrain_radius = terrain_radius
        self.action_probability = action_probability
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        #Human agents: UserSettlable parameters
        #-------------------------------------------------------------------
        #parameters for Cultivators_Agricultural_labourers
        self.AW_onward_time_start = AW_onward_time_start
        self.AW_onward_time_end = AW_onward_time_end
        self.AW_return_time_start = AW_return_time_start
        self.AW_return_time_end = AW_return_time_end

        #parameters for Other_workers
        self.OW_onward_time_start = OW_onward_time_start
        self.OW_onward_time_end = OW_onward_time_end
        self.OW_return_time_start = OW_return_time_start
        self.OW_return_time_end = OW_return_time_end

        #parameters for RandomWalkers_homebound
        self.RW_onward_time_start = RW_onward_time_start
        self.RW_onward_time_end = RW_onward_time_end
        self.RW_return_time_start = RW_return_time_start
        self.RW_return_time_end = RW_return_time_end

        #Human agent commute speed
        self.speed_walking_start = speed_walking_start
        self.speed_walking_end =  speed_walking_end
        self.speed_vehicle_start =  speed_vehicle_start
        self.speed_vehicle_end = speed_vehicle_end
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        #Conflict cognition: UserSettlable parameters
        #-------------------------------------------------------------------
        self.knowledge_from_fringe = knowledge_from_fringe
        self.human_agent_visibility = human_agent_visibility
        self.elephant_agent_visibility = elephant_agent_visibility
        self.prob_crop_damage = prob_crop_damage
        self.prob_infrastructure_damage = prob_infrastructure_damage
        self.fitness_fn_decrement_humans = fitness_fn_decrement_humans
        self.fitness_fn_decrement_elephants = fitness_fn_decrement_elephants
        self.escape_radius_humans = escape_radius_humans
        self.radius_forest_search = radius_forest_search
        self.aggress_threshold_inflict_damage = aggress_threshold_inflict_damage
        self.aggress_threshold_enter_cropland = aggress_threshold_enter_cropland
        self.food_habituation_threshold = food_habituation_threshold
        self.human_habituation_tolerance = human_habituation_tolerance
        self.elephant_habituation_tolerance = elephant_habituation_tolerance
        self.disturbance_tolerance = disturbance_tolerance
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        #Geographical extend of the study area
        #-------------------------------------------------------------------
        latitude_center = 9.3245
        longitude_center = 76.9974
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
        #INITIALIZING ENVIRONMENT
        #-------------------------------------------------------------------
        # MODEL ENVIRONMENT VARIBLES
        self.DEM = self.DEM_study_area()
        self.LANDUSE = self.LANDUSE_study_area()
        self.FOOD = self.FOOD_MATRIX(self.prob_food_forest, self.prob_food_cropland)
        self.WATER = self.WATER_MATRIX(self.prob_water)
        self.SLOPE = self.SLOPE_study_area()
        self.initialize_road_network()
        self.plantation_proximity = self.proximity_from_plantation()
        self.forest_proximity = self.proximity_from_forest()
        self.water_proximity = self.proximity_from_water()
        self.AGRI_PLOTS, self.INFRASTRUCTURE = self.PROPERTY_MATRIX()
        #-------------------------------------------------------------------




        #-------------------------------------------------------------------
        # DAMAGE MATRICES
        #-------------------------------------------------------------------
        self.infrastructure_damage = np.zeros_like(self.INFRASTRUCTURE).tolist()
        self.crop_damage = np.zeros_like(self.AGRI_PLOTS).tolist()
        self.conflict_location = set()
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
        self.dead_agents = set()
        self.agents_to_remove = set()
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




        #AGENT INITIALIZATION
        #-----------------------------------------------------------
        #SEQUENTIAL PROCESSING
        #-----------------------------------------------------------
        #start = timeit.default_timer()
        self.initialize_bull_elephants()
        #stop = timeit.default_timer()
        #execution_time = stop - start
        #print("Elephant agent initialization: "+str(execution_time)+" seconds") # It returns time in seconds
        
        #start = timeit.default_timer()
        self.initialize_human_agents()
        #stop = timeit.default_timer()
        #execution_time = stop - start
        #print("Human agent initialization: "+str(execution_time)+" seconds") # It returns time in seconds
        #-----------------------------------------------------------




        self.datacollector = DataCollector(model_reporters={"conflict_location":"conflict_location",
                                            "num_elephant_deaths":"num_elephant_deaths", 
                                            "num_human_deaths":"num_human_deaths"},
                                            agent_reporters={"longitude":"shape.x", 
                                            "latitude":"shape.y"})
        
        #Collect data at the start of the simulation
        self.datacollector.collect(self)
        #-------------------------------------------------------------------


    



    #-----------------------------------------------------------------------------------------------------
    def initialize_bull_elephants(self, **kwargs):
        """Initialize the elephant agents"""

        #initialization_function = "random initialization"
        initialization_function = "none"

        #elephant agents
        elephant=AgentCreator(Elephant,{"model":self})  

        if initialization_function == "random initialization":
            #random initialization
            coord_lat, coord_lon = self.elephant_distribution_random_init_forest()

        elif initialization_function == "close to fringe":
            #close to fringe
            coord_lat, coord_lon = self.elephant_distribution_close_to_fringe(distance_l=kwargs["lower"], distance_u=kwargs["upper"])

        else:
            #distribution considering landuse, elevation and proximity to fringe
            coord_lat, coord_lon = self.elephant_distribution_modified()

        # Writing the init points to a file
        # coord_lat, coord_lon = coord_lat.reset_index(drop=True), coord_lon.reset_index(drop=True)
        # dict = {}
        # for i in range(1,11):
        #     id = "id_" + str(i)
        #     dict[id]= [coord_lat[i-1], coord_lon[i-1]]
        # import json
        # json.dump(dict, open("initialisation_elephants/elev800prox2000.json", 'w'))

        for i in range(0,self.num_bull_elephants):    #initializing bull elephants

            this_x = np.array(coord_lon)[i] + self.random.randint(-10,10)
            this_y = np.array(coord_lat)[i] + self.random.randint(-10,10)

            newagent = elephant.create_agent(Point(this_x,this_y),"bull_"+str(i))
            newagent.herdID = 0   #variables for assigning the social structure of the elephants. herdID = 0 corresponds to solitary bulls.
            newagent.Leader = True   #bull elephants are always leaders
            newagent.sex = "Male"
            newagent.age=self.random.randrange(15,60) 

            #Assign body weight of the elephant agent depending on the sex and age
            newagent.body_weight = self.assign_body_weight_elephants(newagent.age, newagent.sex)

            #Assign daily dry matter intake depending on the body weight
            newagent.daily_dry_matter_intake = self.assign_daily_dietary_requiremnt(newagent.body_weight)

            self.grid.add_agents(newagent)
            self.schedule.add(newagent)
    #-----------------------------------------------------------------------------------------------------




    #-----------------------------------------------------------------------------------------------------
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




    #-----------------------------------------------------------------------------------------------------
    def assign_daily_dietary_requiremnt(self,body_weight):
        """ The function assigns the daily dietary requirement based on the body weight"""

        #The wild adult Asian elephant's daily dry matter intake: 1.5% to 1.9% of body weight
        #Source: Nutrition adivisary group handbook. Elephants: nutrition and dietary husbandry

        daily_dry_matter_intake = np.random.uniform(1.5,1.9)*body_weight/100

        return daily_dry_matter_intake
    #-----------------------------------------------------------------------------------------------------




    #-----------------------------------------------------------------------------------------------------
    def initialize_human_agents(self):
        """ The function initializes human agents"""

        coord = self.co_ordinates_residential()
        coord = np.array(coord[["0", "1", "2"]])

        coord_non_residential = self.co_ordinates_non_residential()
        coord_non_residential = np.array(coord_non_residential[["0", "1"]])

        coord_agricultural = self.co_ordinates_agricultural_plots()
        coord_agricultural = np.array(coord_agricultural[["0", "1"]])

        init_file = open(os.path.join("experiment_setup_files","init_files","population_init.json"))
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
            self.grid.add_agents(newagent)
            self.schedule.add(newagent)

        return
    #-----------------------------------------------------------------------------------------------------




    #-----------------------------------------------------------------------------------------------------
    def co_ordinates_residential(self):
        """ Function returns the locations of the buildings that are designated as residential in the study area"""

        villages=["Thannithodu","Perunad","Chittar-Seethathodu"]

        lat=[]
        lon=[]
        num_household=[]

        for village in villages:
            path = os.path.join("experiment_setup_files","environment_seethathode", "shape_files", village+"_residential.tif")

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




    #---------------------------------------------------------------------------------------------------------
    def co_ordinates_non_residential(self):
        """ Function returns the locations of the buildings that are designated as non-residential in the study area"""

        villages=["Thannithodu","Perunad","Chittar-Seethathodu"]
        lat=[]
        lon=[]
        num_household=[]

        for village in villages:

            path = os.path.join("experiment_setup_files","environment_seethathode", "shape_files", village+"_non_residential.tif")

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



    
    #---------------------------------------------------------------------------------------------------------
    def co_ordinates_agricultural_plots(self):
        """ Function returns the ID and location of the agricultural plots """

        lat=[]
        lon=[]
        ID=[]

        path = os.path.join("experiment_setup_files","environment_seethathode","Raster_Files_Seethathode_Derived","area_1100sqKm","reso_30x30","LULC.tif")

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




    #-----------------------------------------------------------------------------------------------------
    def DEM_study_area(self):
        """ Returns the digital elevation model of the study area"""

        fid = os.path.join(self.folder_root,"DEM.tif")

        DEM = gdal.Open(fid).ReadAsArray()  
        return DEM.tolist()  #Conversion to list so that the object becomes json serializable
    #-----------------------------------------------------------------------------------------------------




    #-----------------------------------------------------------------------------------------------------
    def LANDUSE_study_area(self):
        """ Returns the landuse model of the study area"""

        fid = os.path.join(self.folder_root,"LULC.tif")

        LULC = gdal.Open(fid).ReadAsArray()  
        return LULC.tolist()  #Conversion to list so that the object becomes json serializable
    #-----------------------------------------------------------------------------------------------------




    #-----------------------------------------------------------------------------------------------------
    def FOOD_MATRIX(self, prob_food_forest, prob_food_cropland):
        """ Returns the food matrix model of the study area"""

        fid = os.path.join(self.folder_root, "Food_matrix_" + str(prob_food_forest) + "_" + str(prob_food_cropland) + "_.tif")

        FOOD = gdal.Open(fid).ReadAsArray()  
        return FOOD.tolist()  #Conversion to list so that the object becomes json serializable
    #-----------------------------------------------------------------------------------------------------




    #-----------------------------------------------------------------------------------------------------
    def WATER_MATRIX(self, prob_water):
        """ Returns the water matrix model of the study area"""

        fid = os.path.join(self.folder_root, "Water_matrix_" + str(prob_water) + "_.tif")

        WATER = gdal.Open(fid).ReadAsArray()  
        return WATER.tolist() #Conversion to list so that the object becomes json serializable
    #-----------------------------------------------------------------------------------------------------





    #-----------------------------------------------------------------------------------------------------
    def SLOPE_study_area(self):
        """ Returns the slope model of the study area"""

        fid = os.path.join(self.folder_root,"slope.tif")

        slope = gdal.Open(fid).ReadAsArray()  
        return slope.tolist()  #Conversion to list so that the object becomes json serializable
    #-----------------------------------------------------------------------------------------------------




    #-----------------------------------------------------------------------------------------------------
    def initialize_road_network(self):

        #Road network in the study area: used for commute by the human agents

        north, south, east, west = self.LAT_MIN,self.LAT_MAX,self.LON_MIN,self.LON_MAX
        self.road_network=ox.graph.graph_from_bbox(north, south, east, west, network_type='all', simplify=True, retain_all=False, truncate_by_edge=False, clean_periphery=True, custom_filter=None)
        self.road_network = ox.project_graph(self.road_network,"epsg:3857")
        self.nodes_proj, self.edges_proj = ox.graph_to_gdfs(self.road_network, nodes=True, edges=True)

        return
    #-----------------------------------------------------------------------------------------------------




    #-----------------------------------------------------------------------------------------------------
    def proximity_from_plantation(self):
        """ Returns the proximity matrix from the plantations"""

        fid = os.path.join(self.folder_root, "proximity_from_plantation.tif")

        plantation_proximity = gdal.Open(fid).ReadAsArray()  
        return plantation_proximity.tolist() #Conversion to list so that the object becomes json serializable
    #-----------------------------------------------------------------------------------------------------




    #-----------------------------------------------------------------------------------------------------
    def proximity_from_forest(self):
        """ Returns the proximity matrix from the evergreen broadleaf forest"""

        fid = os.path.join(self.folder_root, "proximity_from_forest.tif")

        forest_proximity = gdal.Open(fid).ReadAsArray()  
        return forest_proximity.tolist() #Conversion to list so that the object becomes json serializable
    #-----------------------------------------------------------------------------------------------------




    #-----------------------------------------------------------------------------------------------------
    def proximity_from_water(self):
        """ Returns the proximity matrix from the plantations"""

        fid = os.path.join(self.folder_root, "proximity_from_water_body.tif")

        water_proximity = gdal.Open(fid).ReadAsArray()  
        return water_proximity.tolist() #Conversion to list so that the object becomes json serializable
    #-----------------------------------------------------------------------------------------------------




    #-----------------------------------------------------------------------------------------------------
    def PROPERTY_MATRIX(self):
        """ Returns the infrastructure and crop matrix of the study area"""

        population = gdal.Open(os.path.join(self.folder_root, "Population.tif")).ReadAsArray()

        path = os.path.join("experiment_setup_files","environment_seethathode","Raster_Files_Seethathode_Derived","area_1100sqKm","reso_30x30","LULC.tif")
        distribution = gdal.Open(path).ReadAsArray()
        m,n = distribution.shape

        plots = np.zeros_like(distribution)
        infrastructure = np.zeros_like(distribution)

        for i in range(0,m):
            for j in range(0,n):
                if distribution[i,j]==10:  
                    plots[i, j] = 1

                if population[i,j]!=0:  
                    infrastructure[i, j] = 1

        return plots.tolist(), infrastructure.tolist()
    #-----------------------------------------------------------------------------------------------------





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




    #-----------------------------------------------------------------------------------------------------
    def pixel2coord(self, row, col):
        """
        Gets the lon and lat corrsponding to the row and column indices.
        """

        lon = self.xres * 0.5  + self.xmin + col * self.xres
        lat = self.yres * 0.5  + self.ymax + row * self.yres

        return(lon, lat)
    #-----------------------------------------------------------------------------------------------------




    #-----------------------------------------------------------------------------------------------------
    def pixel2coord_raster(self, row, col, xmin, ymax, xres, yres):
        """
        Gets the lon and lat corrsponding to the row and column indices.
        """

        lon = xres * 0.5  + xmin + col * xres
        lat = yres * 0.5  + ymax + row * yres

        return(lon, lat)
    #-----------------------------------------------------------------------------------------------------




    #-----------------------------------------------------------------------------------------------------
    def update_human_disturbance_explict(self):
        #6am to 6pm: high disturbance
        #6pm to 6am: low disturbance
        if self.hour_in_day >=6 and self.hour_in_day <18:
            self.human_disturbance = self.disturbance_tolerance

        else:
            self.human_disturbance = 0
        return
    #-----------------------------------------------------------------------------------------------------




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

            print("Number of human deaths: ", self.num_human_deaths)
            print("Number of elephant deaths: ", self.num_elephant_deaths)

            data1 = self.datacollector.get_agent_vars_dataframe()
            data2 = self.datacollector.get_model_vars_dataframe()
            Step = np.arange(0, self.max_time + 1, 1)
            data1.set_index(Step, drop=True, inplace=True)
            data2.set_index(Step, drop=True, inplace=True)

            data = pd.concat([data1, data2], axis=1)
            data.to_csv(os.path.join("experiment_setup_files","simulation_results_server_run","simulation_data.csv"))

            source = os.path.join(self.folder_root, "LULC.tif")
            with rio.open(source) as src:
                ras_meta = src.profile

            loc = os.path.join("experiment_setup_files","simulation_results_server_run", "infrastructure_damage.tif")
            with rio.open(loc, 'w', **ras_meta) as dst:
                dst.write(np.array(self.infrastructure_damage).reshape(self.row_size, self.col_size).astype('int'), 1)

            loc = os.path.join("experiment_setup_files","simulation_results_server_run", "crop_damage.tif")
            with rio.open(loc, 'w', **ras_meta) as dst:
                dst.write(np.array(self.crop_damage).reshape(self.row_size, self.col_size).astype('int'), 1)
    #---------------------------------------------------------------------------------------------------------
