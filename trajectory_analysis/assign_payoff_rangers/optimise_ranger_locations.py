import os

import sys
sys.path.append(os.getcwd())

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
from scipy.optimize import minimize
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home2/anjali/GitHub/abm-elephant-project")

class optimise_ranger_locations():

    def __init__(self, num_best_trajs, num_rangers, ranger_visibility_radius, expt_folder):
        self.num_best_trajs = num_best_trajs
        self.num_rangers = num_rangers
        self.ranger_visibility_radius = ranger_visibility_radius
        self.expt_folder = expt_folder
        self.loss_history = []

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

            ax.scatter(x_new, y_new, 25, marker='x', color='white', zorder=2)

            radius = self.ranger_visibility_radius/(111*1000)  
            circle = plt.Circle((x_new, y_new), radius, facecolor='purple', fill=True, alpha=0.5, edgecolor='black', linewidth=1)
            ax.add_artist(circle)
            
        plt.savefig("trajectory_analysis/assign_payoff_rangers/trajectories_with_ranger_locations__numrangers" + str(self.num_rangers) + "_rangervisibility" + str(self.ranger_visibility_radius) + "m_v1.png", dpi=300, bbox_inches='tight')

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
            ax.scatter(x_new, y_new, 25, marker='x', color='white')
            
            radius = self.ranger_visibility_radius/(111*1000)  
            circle = plt.Circle((x_new, y_new), radius, facecolor='purple', fill=True, alpha=0.5, edgecolor='black', linewidth=1)
            ax.add_artist(circle)

        plt.savefig("trajectory_analysis/assign_payoff_rangers/trajectories_with_ranger_locations__numrangers" + str(self.num_rangers) + "_rangervisibility" + str(self.ranger_visibility_radius) + "m_v2.png", dpi=300, bbox_inches='tight')

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

    def find_first_visible_entry(self, trajectory):
         
        first_visible_entries = None
    
        for idx, row in trajectory.iterrows():
            for ranger_location in self.ranger_location:
                distance = ((row['longitude'] - ranger_location[0])**2 + 
                            (row['latitude'] - ranger_location[1])**2)**0.5
                
                if distance <= self.ranger_visibility_radius:
                    first_visible_entries = idx
                    return first_visible_entries
                else:
                    first_visible_entries = None

        return first_visible_entries

    def plot_trajectories_untill_ranger_intervention(self):

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

        for i, traj in enumerate(self.best_trajs):

            first_visible_entry = self.find_first_visible_entry(traj)

            if first_visible_entry is not None:
                data = traj.iloc[:first_visible_entry + 1]
            
            else:
                data = traj

            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
            longitude, latitude = transform(inProj, outProj, data["longitude"], data["latitude"])
            x_new, y_new = map(longitude,latitude)

            ax.plot(x_new, y_new, 
                    'b-', alpha=0.4, label='Trajectory')

            ax.scatter(x_new[0], y_new[0], 25, marker='o', color='black', zorder=2)
            ax.scatter(x_new[-1], y_new[-1], 25, marker='^', color='black', zorder=2)

        for ranger in self.ranger_location:
            longitude, latitude = transform(inProj, outProj, ranger[0], ranger[1])
            x_new, y_new = map(longitude,latitude)

            ax.scatter(x_new, y_new, 25, marker='x', color='white', zorder=2)

            radius = self.ranger_visibility_radius/(111*1000)  
            circle = plt.Circle((x_new, y_new), radius, facecolor='purple', fill=True, alpha=0.5, edgecolor='black', linewidth=1)
            ax.add_artist(circle)

        plt.savefig("trajectory_analysis/assign_payoff_rangers/trajectories_until_ranger_proximity__numrangers" + str(self.num_rangers) + "_rangervisibility" + str(self.ranger_visibility_radius) + "m_v1.png", dpi=300, bbox_inches='tight')

        return

    def cost_function_v1(self, x):

        ranger_positions = x.reshape(-1, 2)

        uncovered_trajectories = 0
        for traj in self.best_trajs:
            traj_covered = False
            for pos in ranger_positions:
                for lon, lat in zip(traj["longitude"], traj["latitude"]):
                    distance = np.sqrt((lon - pos[0])**2 + (lat - pos[1])**2)
                    if distance <= self.ranger_visibility_radius:
                        traj_covered = True
                        break
                if traj_covered:
                    break
            if not traj_covered:
                uncovered_trajectories += 1

        return uncovered_trajectories            

    def cost_function_v2(self, x):

        ranger_positions = x.reshape(-1, 2)

        uncovered_trajectories = 0
        for traj in self.best_trajs:
            traj_covered = False
            for pos in ranger_positions:
                for lon, lat in zip(traj["longitude"], traj["latitude"]):
                    distance = np.sqrt((lon - pos[0])**2 + (lat - pos[1])**2)
                    if distance <= self.ranger_visibility_radius:
                        traj_covered = True
                        break
                if traj_covered:
                    break
            if not traj_covered:
                uncovered_trajectories += 1

        forest_penalty = 0
        ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
        data_LULC = ds.ReadAsArray()
        
        data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}
        for i in range(1,10):
            data_LULC[data_LULC == data_value_map[i]] = i
            
        row_size, col_size = data_LULC.shape
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
        
        for pos in ranger_positions:
            px = int((pos[0] - xmin) / xres)
            py = int((pos[1] - ymax) / yres)
            
            landuse = data_LULC[py, px]
            if landuse != 7:  
                forest_penalty += self.num_best_trajs  

        self.loss_history.append(uncovered_trajectories + forest_penalty)

        return uncovered_trajectories + forest_penalty    
    
    def optimize(self):

        initial_positions = []

        all_lons = np.concatenate([traj["longitude"].values for traj in self.best_trajs])
        all_lats = np.concatenate([traj["latitude"].values for traj in self.best_trajs])
        # points = np.column_stack((all_lons, all_lats))
        # kmeans = KMeans(n_clusters=self.num_rangers, random_state=42)
        # kmeans.fit(points)
        # initial_positions = kmeans.cluster_centers_.flatten()

        ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
        data_LULC = ds.ReadAsArray()
        data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}
        for i in range(1,10):
            data_LULC[data_LULC == data_value_map[i]] = i
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
        plantation_indices = np.where(data_LULC == 7)
        plantation_points = []
        for py, px in zip(*plantation_indices):
            lon = xmin + px * xres
            lat = ymax + py * yres
            plantation_points.append([lon, lat])
        plantation_points = np.array(plantation_points)

        lon_min = min(all_lons) 
        lon_max = max(all_lons) 
        lat_min = min(all_lats) 
        lat_max = max(all_lats)

        plantation_points = plantation_points[(plantation_points[:,0] >= lon_min) & (plantation_points[:,0] <= lon_max) & (plantation_points[:,1] >= lat_min) & (plantation_points[:,1] <= lat_max)]
        indices = np.random.choice(len(plantation_points), self.num_rangers, replace=False)
        plantation_points = plantation_points[indices]
        initial_positions = plantation_points.flatten()

        bounds = [(lon_min, lon_max), (lat_min, lat_max)] * self.num_rangers

        result = minimize(
            self.cost_function_v2,
            initial_positions,
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        self.ranger_location = result.x.reshape(-1, 2)

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(self.loss_history, 'bo-', markersize=4, linewidth=1, alpha=0.7, 
                label='Loss per iteration')

        ax.set_xlabel('Iteration Number', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        
        min_loss_idx = np.argmin(self.loss_history)
        ax.annotate(f'Minimum Loss: {min(self.loss_history):.2f}',
                   xy=(min_loss_idx, self.loss_history[min_loss_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->'))
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')
        
        plt.savefig('trajectory_analysis/assign_payoff_rangers/optimization_loss.png', dpi=300, bbox_inches='tight')
        plt.close()

        return
    

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

optimizer = optimise_ranger_locations(num_best_trajs = 50, num_rangers=3, ranger_visibility_radius=500, expt_folder = output_folder)
optimizer.optimize()
optimizer.plot_trajectories_with_ranger_location()
optimizer.plot_trajectories_untill_ranger_intervention()
intersecting_trajs, non_intersecting_trajs = optimizer.filter_trajectories_in_ranger_radius()
optimizer.plot_filtered_trajectories(intersecting_trajs, non_intersecting_trajs)

