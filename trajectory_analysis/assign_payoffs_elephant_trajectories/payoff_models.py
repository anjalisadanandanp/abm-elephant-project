import os

import sys
sys.path.append(os.getcwd())

import pandas as pd
import movingpandas as mpd
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home2/anjali/GitHub/abm-elephant-project")

class assign_elephant_trajectory_payoff():

    def __init__(self, num_best_trajs, expt_folder):
        self.num_best_trajs = num_best_trajs
        self.expt_folder = expt_folder

        self.get_best_trajectories()

    def get_sorted_trajectories(self):

        folder = os.path.join(os.getcwd(), "trajectory_analysis/outputs", self.expt_folder)
        self.sorted_trajs_df = pd.read_csv(os.path.join(folder, "ordered_experiments.csv"))

        return

    def get_best_trajectories(self):

        self.get_sorted_trajectories()

        self.best_trajs = []
        self.max_simulation_length = 0

        for traj in range(self.num_best_trajs):
            path = self.sorted_trajs_df["file_path"].iloc[traj]
            data = pd.read_csv(os.path.join(str(path)))
            self.best_trajs.append(data)

            if len(data) > self.max_simulation_length:
                self.max_simulation_length = len(data)

        return
    
    def plot_trajectory(self, traj_df):

        traj_df["geometry"] = traj_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)

        start_date = '2010-01-01 00:00:00' 
        timestamps = pd.date_range(start=start_date, periods=len(traj_df), freq='5min')
        traj_df['timestamp'] = timestamps

        trajectory = mpd.Trajectory(traj_df, traj_id="AgentID", x="longitude", y="latitude", crs=3857, t="timestamp")
        loc = GeoDataFrame([trajectory.get_row_at(datetime(2010, 1, 1, 0, 0))])
        trajectory.add_speed(units=('km', 'h'))
        trajectory.hvplot(line_width=2.5, c="fitness", cmap="coolwarm", colorbar="True", width=500, height=500)
        img = loc.hvplot(size=100, color="red")*trajectory.hvplot(line_width=2.5, c="fitness", cmap="coolwarm", colorbar="True", width=600, height=600)

        return img
    
    def assign_payoffs_v1(self):

        for trajectory in self.best_trajs[0:1]:
            img = self.plot_trajectory(trajectory)

        return img
    


model_params = {
    "year": 2010,
    "month": "Jul",
    "num_bull_elephants": 1, 
    "area_size": 1100,              
    "spatial_resolution": 30, 
    "max_food_val_cropland": 100,
    "max_food_val_forest": 10,
    "prob_food_forest": 0.10,
    "prob_food_cropland": 0.10,
    "prob_water_sources": 0.00,
    "thermoregulation_threshold": 28,
    "num_days_agent_survives_in_deprivation": 10,     
    "knowledge_from_fringe": 1500,   
    "prob_crop_damage": 0.05,           
    "prob_infrastructure_damage": 0.01,
    "percent_memory_elephant": 0.375,   
    "radius_food_search": 750,     
    "radius_water_search": 750, 
    "radius_forest_search": 1500,
    "fitness_threshold": 0.4,   
    "terrain_radius": 750,       
    "slope_tolerance": 30,
    "max_time_steps": 288*10,
    "aggression_threshold_enter_cropland": 1.0,
    "elephant_agent_visibility_radius": 500,
    "plot_stepwise_target_selection": False,
    "threshold_days_of_food_deprivation": 0,
    "threshold_days_of_water_deprivation": 3,
    "number_of_feasible_movement_directions": 3,
    "track_in_mlflow": False,
    "elephant_starting_location": "user_input",
    "elephant_starting_latitude": 1049237,
    "elephant_starting_longitude": 8570917,
    "elephant_aggression_value": 0.8,
    "elephant_crop_habituation": False
    }

experiment_name = "exploratory-search-ID-01"

elephant_category = "solitary_bulls"
starting_location = "latitude-" + str(model_params["elephant_starting_latitude"]) + "-longitude-" + str(model_params["elephant_starting_longitude"])
landscape_food_probability = "landscape-food-probability-forest-" + str(model_params["prob_food_forest"]) + "-cropland-" + str(model_params["prob_food_cropland"])
water_holes_probability = "water-holes-within-landscape-" + str(model_params["prob_water_sources"])
memory_matrix_type = "random-memory-matrix-model"
num_days_agent_survives_in_deprivation = "num_days_agent_survives_in_deprivation-" + str(model_params["num_days_agent_survives_in_deprivation"])
maximum_food_in_a_forest_cell = "maximum-food-in-a-forest-cell-" + str(model_params["max_food_val_forest"])
elephant_thermoregulation_threshold = "thermoregulation-threshold-temperature-" + str(model_params["thermoregulation_threshold"])
threshold_food_derivation_days = "threshold_days_of_food_deprivation-" + str(model_params["threshold_days_of_food_deprivation"])
threshold_water_derivation_days = "threshold_days_of_water_deprivation-" + str(model_params["threshold_days_of_water_deprivation"])
slope_tolerance = "slope_tolerance-" + str(model_params["slope_tolerance"])
num_days_agent_survives_in_deprivation = "num_days_agent_survives_in_deprivation-" + str(model_params["num_days_agent_survives_in_deprivation"])
elephant_aggression_value = "elephant_aggression_value_" + str(model_params["elephant_aggression_value"])

output_folder = os.path.join(experiment_name, starting_location, elephant_category, landscape_food_probability, 
                                water_holes_probability, memory_matrix_type, num_days_agent_survives_in_deprivation, maximum_food_in_a_forest_cell, 
                                elephant_thermoregulation_threshold, threshold_food_derivation_days, threshold_water_derivation_days, 
                                slope_tolerance, num_days_agent_survives_in_deprivation, elephant_aggression_value,
                                str(model_params["year"]), str(model_params["month"]))

create_payoffs = assign_elephant_trajectory_payoff(num_best_trajs = 50, expt_folder = output_folder)
img = create_payoffs.assign_payoffs_v1()


img