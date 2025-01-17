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
import yaml
from osgeo import gdal
import numpy as np 
from pyproj import Proj, transform 
from mpl_toolkits.basemap import Basemap 
import matplotlib.colors as mcolors 
import matplotlib.cm as cm   
from matplotlib import colors

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home2/anjali/GitHub/abm-elephant-project")

class assign_elephant_trajectory_payoff():

    def __init__(self, num_best_trajs, ranger_visibility_radius, expt_folder):
        self.num_best_trajs = num_best_trajs
        self.ranger_visibility_radius = ranger_visibility_radius
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
    
    def plot_trajectory_interactive(self, traj_df):

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

    def read_locations(self, file_path):
        """Read latitude/longitude pairs from YAML file."""
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
                return data['locations']
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def plot_trajectories_with_ranger_location(self):

        fig, ax = plt.subplots(figsize = (10,10))
        ax.yaxis.set_inverted(True)

        ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
        data_LULC = ds.ReadAsArray()
        data_LULC = np.flip(data_LULC, axis=0)

        data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

        for i in range(1,10):
            data_LULC[data_LULC == data_value_map[i]] = i

        row_size, col_size = data_LULC.shape
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
        
        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857') 
        LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
        LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

        map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        clrs = ["greenyellow","mediumpurple","turquoise", "plum", "black", "blue", "yellow", "mediumseagreen", "forestgreen"] 
        cmap, norm = colors.from_levels_and_colors(levels, clrs)

        map.imshow(data_LULC, cmap = cmap, norm=norm, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 0.5)

        map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
        map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

        cbar = plt.colorbar(ticks=[1,2,3,4,5,6,7,8,9],fraction=0.046, pad=0.04)
        cbar.ax.set_yticks(ticks=[1,2,3,4,5,6,7,8,9]) 
        cbar.ax.set_yticklabels(["Deciduous Broadleaf Forest","Built-up Land","Mixed Forest","Shrubland","Barren Land","Water Bodies","Plantations","Grassland","Broadleaf evergreen forest"])
        
        for data in self.best_trajs:

            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
            longitude, latitude = transform(inProj, outProj, data["longitude"], data["latitude"])
            x_new, y_new = map(longitude,latitude)
            C = np.arange(len(x_new))
            nz = mcolors.Normalize()
            nz.autoscale(C)

            ax.quiver(x_new[:-1], y_new[:-1],
                        x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1], 
                        scale_units='xy', angles='xy', 
                        scale=1, zorder=1, color = cm.jet(nz(C)), 
                        width=0.0025,
                        alpha=0.5)
            
            ax.scatter(x_new[0], y_new[0], 25, marker='o', color='black', zorder=2)
            ax.scatter(x_new[-1], y_new[-1], 25, marker='^', color='black', zorder=2)

        for ranger in self.ranger_location:
            longitude, latitude = transform(inProj, outProj, ranger[0], ranger[1])
            x_new, y_new = map(longitude,latitude)

            ax.scatter(x_new, y_new, 25, marker='x', color='black', zorder=2)

            radius = self.ranger_visibility_radius/(111*1000)  
            circle = plt.Circle((x_new, y_new), radius, facecolor='purple', fill=True, alpha=0.5, edgecolor='black', linewidth=1)
            ax.add_artist(circle)
            
        plt.savefig("trajectory_analysis/assign_payoffs_elephant_trajectories/trajectories_with_ranger_locations__rangervisibility" + str(self.ranger_visibility_radius) + "m_v1.png", dpi=300, bbox_inches='tight')

    def filter_trajectories_in_ranger_radius(self):
        """Filter trajectories that intersect with ranger visibility circles."""

        intersecting_trajs = []
        non_intersecting_trajs = []
        
        for data in self.best_trajs:
            trajectory_intersects = False
            longitude, latitude = data["longitude"], data["latitude"]


            for ranger in self.ranger_location:
                ranger_lon, ranger_lat = ranger[0], ranger[1]
                
                for lon, lat in zip(longitude, latitude):
                    distance = np.sqrt((lon - ranger_lon)**2 + (lat - ranger_lat)**2)
                    
                    if distance <= self.ranger_visibility_radius:
                        trajectory_intersects = True
                        break
                        
                if trajectory_intersects:
                    break
                    
            if trajectory_intersects:
                intersecting_trajs.append(data)
            else:
                non_intersecting_trajs.append(data)
                
        return intersecting_trajs, non_intersecting_trajs

    def plot_filtered_trajectories(self, intersecting_trajs, non_intersecting_trajs):

        fig, ax = plt.subplots(figsize=(10,10))
        ax.yaxis.set_inverted(True)

        ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
        data_LULC = ds.ReadAsArray()
        data_LULC = np.flip(data_LULC, axis=0)

        data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

        for i in range(1,10):
            data_LULC[data_LULC == data_value_map[i]] = i

        row_size, col_size = data_LULC.shape
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
        
        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857') 
        LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
        LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

        map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        clrs = ["greenyellow","mediumpurple","turquoise", "plum", "black", "blue", "yellow", "mediumseagreen", "forestgreen"] 
        cmap, norm = colors.from_levels_and_colors(levels, clrs)

        map.imshow(data_LULC, cmap = cmap, norm=norm, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 0.5)

        map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
        map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

        cbar = plt.colorbar(ticks=[1,2,3,4,5,6,7,8,9],fraction=0.046, pad=0.04)
        cbar.ax.set_yticks(ticks=[1,2,3,4,5,6,7,8,9]) 
        cbar.ax.set_yticklabels(["Deciduous Broadleaf Forest","Built-up Land","Mixed Forest","Shrubland","Barren Land","Water Bodies","Plantations","Grassland","Broadleaf evergreen forest"])
        
        for data in intersecting_trajs:
            longitude, latitude =  transform(inProj, outProj, data["longitude"], data["latitude"])
            x_new, y_new = map(longitude, latitude)
            
            ax.quiver(x_new[:-1], y_new[:-1],
                    x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1],
                    scale_units='xy', angles='xy',
                    scale=1, zorder=1, color='red', width=0.0025)
            
            ax.scatter(x_new[0], y_new[0], 25, marker='o', color='black')
            ax.scatter(x_new[-1], y_new[-1], 25, marker='^', color='black')

        for data in non_intersecting_trajs:
            longitude, latitude =  transform(inProj, outProj, data["longitude"], data["latitude"])
            x_new, y_new = map(longitude, latitude)
            
            ax.quiver(x_new[:-1], y_new[:-1],
                    x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1],
                    scale_units='xy', angles='xy',
                    scale=1, zorder=1, color='green', width=0.0025)
            
            ax.scatter(x_new[0], y_new[0], 25, marker='o', color='black')
            ax.scatter(x_new[-1], y_new[-1], 25, marker='^', color='black')

        for ranger in self.ranger_location:
            longitude, latitude = transform(inProj, outProj, ranger[0], ranger[1])
            x_new, y_new = map(longitude,latitude)
            ax.scatter(x_new, y_new, 25, marker='x', color='black')
            
            radius = self.ranger_visibility_radius/(111*1000)  
            circle = plt.Circle((x_new, y_new), radius, facecolor='purple', fill=True, alpha=0.5, edgecolor='black', linewidth=1)
            ax.add_artist(circle)

        plt.savefig("trajectory_analysis/assign_payoffs_elephant_trajectories/trajectories_with_ranger_locations__rangervisibility" + str(self.ranger_visibility_radius) + "m_v2.png", dpi=300, bbox_inches='tight')

    def calculate_landuse_time(self, trajectory):
        """Calculate time steps spent in each landuse type."""

        ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
        data_LULC = ds.ReadAsArray()
        data_LULC = np.flip(data_LULC, axis=0)

        data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

        for i in range(1,10):
            data_LULC[data_LULC == data_value_map[i]] = i

        row_size, col_size = data_LULC.shape
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

        landuse_times = {
            1: 0,  # Deciduous Broadleaf Forest
            2: 0,  # Built-up Land
            3: 0,  # Mixed Forest 
            4: 0,  # Shrubland
            5: 0,  # Barren Land
            6: 0,  # Water Bodies
            7: 0,  # Plantations
            8: 0,  # Grassland
            9: 0   # Broadleaf evergreen forest
        }
        
        for lon, lat in zip(trajectory["longitude"], trajectory["latitude"]):
            # Convert coordinates to pixel indices
            px = int((lon - xmin) / xres)
            py = int((lat - ymax) / yres)
            
            # Get landuse type at location
            landuse = data_LULC[py, px]
            landuse_times[landuse] += 1

        total_time = sum(landuse_times.values())
        assert total_time == len(trajectory), "Total time steps don't match trajectory length"
   
        return landuse_times

    def calculate_daily_landuse_time(self, trajectory, steps_per_day=288):
        """Calculate landuse time for each day."""
        
        days = len(trajectory["longitude"]) // steps_per_day
        daily_landuse = []
        
        for day in range(days):
            start_idx = day * steps_per_day
            end_idx = start_idx + steps_per_day
            
            day_trajectory = trajectory.iloc[start_idx:end_idx]
            
            landuse_times = self.calculate_landuse_time(day_trajectory)
            daily_landuse.append(landuse_times)
            
        return daily_landuse
    
    def is_df_in_list(self, target_df, df_list):
        return any(target_df.equals(df) for df in df_list)

    def assign_payoffs_v1(self, cost):

        trajectory_payoffs = []

        self.ranger_location = self.read_locations("trajectory_analysis/ranger-locations/random.yaml")
        self.plot_trajectories_with_ranger_location()
        intersecting_trajs, non_intersecting_trajs = self.filter_trajectories_in_ranger_radius()
        self.plot_filtered_trajectories(intersecting_trajs, non_intersecting_trajs)

        for i, traj in enumerate(self.best_trajs):
            daily_landuse_time = self.calculate_daily_landuse_time(traj)
            daily_food_consumed = traj["food_consumed"][::288].values[1:]
            trajectory_fitness = traj["fitness"][::288].values[-1]
            plantation_use_time =[day[7] for day in daily_landuse_time]

            if self.is_df_in_list(traj, intersecting_trajs):
                for plantation_use, food in zip(plantation_use_time, daily_food_consumed):
                    if plantation_use > 0:
                        trajectory_fitness -= cost

            elif self.is_df_in_list(traj, non_intersecting_trajs):
                for plantation_use, food in zip(plantation_use_time, daily_food_consumed):
                    if plantation_use > 0 and food >= traj["daily_dry_matter_intake"].unique()[0]:
                        trajectory_fitness += cost

            trajectory_payoffs.append(trajectory_fitness)

        return trajectory_payoffs

    def sort_trajectories(self, trajectory_payoffs):

        """Sort trajectories based on payoffs."""

        pairs = list(zip(trajectory_payoffs, self.best_trajs))
        sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
        
        self.trajs_with_best_payoffs = [traj for _, traj in sorted_pairs]
        payoffs = [payoff for payoff,_ in sorted_pairs]

        fig, ax = plt.subplots(figsize=(10,10))
        ax.yaxis.set_inverted(True)
        
        ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
        data_LULC = ds.ReadAsArray()
        data_LULC = np.flip(data_LULC, axis=0)

        data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

        for i in range(1,10):
            data_LULC[data_LULC == data_value_map[i]] = i

        row_size, col_size = data_LULC.shape
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
        
        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857') 
        LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
        LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

        map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        clrs = ["greenyellow","mediumpurple","turquoise", "plum", "black", "blue", "yellow", "mediumseagreen", "forestgreen"] 
        cmap, norm = colors.from_levels_and_colors(levels, clrs)

        map.imshow(data_LULC, cmap = cmap, norm=norm, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 0.5)

        map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
        map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

        cbar = plt.colorbar(ticks=[1,2,3,4,5,6,7,8,9],fraction=0.046, pad=0.04)
        cbar.ax.set_yticks(ticks=[1,2,3,4,5,6,7,8,9]) 
        cbar.ax.set_yticklabels(["Deciduous Broadleaf Forest","Built-up Land","Mixed Forest","Shrubland","Barren Land","Water Bodies","Plantations","Grassland","Broadleaf evergreen forest"])
        
        payoff_colors = plt.cm.RdYlGn_r((trajectory_payoffs - min(trajectory_payoffs)) / 
                                        (max(trajectory_payoffs) - min(trajectory_payoffs)))

        for data, payoff, color in zip(self.trajs_with_best_payoffs, payoffs, payoff_colors):

            longitude, latitude = transform(inProj, outProj, data["longitude"], data["latitude"])
            x_new, y_new = map(longitude, latitude)
            
            ax.quiver(x_new[:-1], y_new[:-1],
                    x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1],
                    scale_units='xy', angles='xy', 
                    scale=1, zorder=int(payoff*10), color=color, width=0.0025)
            
            ax.scatter(x_new[0], y_new[0], 25, marker='o', color='black', zorder=int(payoff*10))
            ax.scatter(x_new[-1], y_new[-1], 25, marker='^', color='black', zorder=int(payoff*10))

        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, 
                                    norm=plt.Normalize(min(trajectory_payoffs), 
                                                    max(trajectory_payoffs)))
        cax2 = fig.add_axes([0.2, 0.05, 0.6, 0.02]) 
        cbar2 = plt.colorbar(sm, cax=cax2, orientation='horizontal')
        cbar2.set_label('payoffs')

        plt.savefig("trajectory_analysis/assign_payoffs_elephant_trajectories/trajectories_with_payoffs___rangervisibility" + str(self.ranger_visibility_radius) + "m_.png", dpi=300, bbox_inches='tight')
        
                    

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

create_payoffs = assign_elephant_trajectory_payoff(num_best_trajs = 100, ranger_visibility_radius=25, expt_folder = output_folder)
trajectory_payoffs = create_payoffs.assign_payoffs_v1(cost = 0.10)
create_payoffs.sort_trajectories(trajectory_payoffs)
