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
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
import pathlib

import warnings
warnings.filterwarnings('ignore')

os.chdir("/home2/anjali/GitHub/abm-elephant-project")

class optimise_ranger_locations():

    def __init__(self, num_best_trajs, num_rangers, ranger_visibility_radius, data_folder, output_folder):
        self.num_best_trajs = num_best_trajs
        self.num_rangers = num_rangers
        self.ranger_visibility_radius = ranger_visibility_radius
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.loss_history = []
        self.elephant_kde = None

        self.read_all_experiments()
        self.rank_order_trajectories(save_dataframe = True)
        self.get_best_trajectories()

    def read_all_experiments(self):
    
        path = pathlib.Path(self.data_folder)
        subfolders = [x for x in path.iterdir() if x.is_dir()]
        self.file_paths = [str(subfolder / "output_files/agent_data.csv") for subfolder in subfolders]

    def rank_order_trajectories(self, save_dataframe = False):

        data = []
        for file_path in self.file_paths:
            try:
                df = pd.read_csv(file_path)
                length_rank = len(df)
                last_fitness = df['fitness'].iloc[-1] 
                data.append({
                    'file_path': file_path,
                    'length_rank': length_rank,
                    'fitness_rank': last_fitness
                })
            except FileNotFoundError:
                print(f"Warning: File not found: {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        df_ranks = pd.DataFrame(data)
        df_ranks['length_rank'] = df_ranks['length_rank'].rank(method='dense', ascending=False)
        df_ranks['fitness_rank'] = df_ranks['fitness_rank'].rank(method='dense', ascending=False) 

        self.sorted_df = df_ranks.sort_values(by=['length_rank', 'fitness_rank'], ascending=[True, True]) 

        if save_dataframe == True:
            self.sorted_df.to_csv(os.path.join(self.output_folder, "ordered_experiments.csv"))

        return
    
    def get_sorted_trajectories(self):

        self.sorted_trajs_df = pd.read_csv(os.path.join(self.output_folder, "ordered_experiments.csv"))

        return

    def get_best_trajectories(self):

        self.get_sorted_trajectories()

        self.best_trajs = []
        self.max_simulation_length = 0

        for traj in range(self.num_best_trajs):
            path = self.sorted_trajs_df["file_path"].iloc[traj]
            data = pd.read_csv(os.path.join(str(path)))
            data = data[data["AgentID"] == "bull_0"]
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
            
        plt.savefig(os.path.join(self.output_folder, "trajectories_with_ranger_locations__numrangers" + str(self.num_rangers) + "_rangervisibility" + str(self.ranger_visibility_radius) + "m_v1.png"), dpi=300, bbox_inches='tight')

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

        plt.savefig(os.path.join(self.output_folder, "trajectories_with_ranger_locations__numrangers" + str(self.num_rangers) + "_rangervisibility" + str(self.ranger_visibility_radius) + "m_v2.png"), dpi=300, bbox_inches='tight')

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

        plt.savefig(os.path.join(self.output_folder, "trajectories_until_ranger_proximity__numrangers" + str(self.num_rangers) + "_rangervisibility" + str(self.ranger_visibility_radius) + "m_v1.png"), dpi=300, bbox_inches='tight')

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

    def cost_function_kde(self, x=None, grid_resolution=100):

        ranger_positions = x.reshape(-1, 2)

        # self.plot_kde_95(ranger_positions, save_plots=True)

        all_lons = np.concatenate([traj["longitude"].values for traj in self.best_trajs])
        all_lats = np.concatenate([traj["latitude"].values for traj in self.best_trajs])

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')
        x, y = transform(inProj, outProj, all_lons, all_lats)                                             

        X = np.vstack([
            x,
            y
            ]).T
            
        if self.elephant_kde is None:
            self.elephant_kde  = gaussian_kde(X.T)

            self.lat_grid, self.lon_grid = np.mgrid[min(y): max(y):grid_resolution*1j, min(x): max(x):grid_resolution*1j]
            positions = np.vstack([self.lon_grid.ravel(), self.lat_grid.ravel()])
            elephant_density = np.reshape(self.elephant_kde(positions).T, self.lat_grid.shape)
            self.elephant_density = elephant_density / elephant_density.max() 
            
            print("Elephant space use KDE fitted")

        ranger_coverage = np.zeros_like(self.elephant_density)
        
        ranger_x, ranger_y = transform(inProj, outProj, 
                                    ranger_positions[:, 0], 
                                    ranger_positions[:, 1])
        
        radius = self.ranger_visibility_radius /(111 * 1000)
        
        for i, (rx, ry) in enumerate(zip(ranger_x, ranger_y)):

            distances = np.sqrt((self.lon_grid - rx)**2 + (self.lat_grid - ry)**2)

            # fig, ax = plt.subplots(figsize=(6.8, 6.8))
            # map = Basemap(llcrnrlat=min(y), urcrnrlat=max(y), llcrnrlon=min(x), urcrnrlon=max(x), resolution='l')
            # map.drawparallels([min(y), max(y)], labels=[True, False, False, True])
            # map.drawmeridians([min(x), max(x)], labels=[True, False, False, True])
            # img = ax.imshow(np.flipud(distances)*111, cmap='coolwarm', alpha=0.75, extent=[min(x), max(x), min(y), max(y)], zorder=1)
            # ax.scatter(rx, ry, 50, marker='x', color='black', zorder=2)
            # plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.5)
            # plt.savefig(os.path.join(os.getcwd(), "trajectory_analysis/assign_payoff_rangers", 'ranger' + str(i) + '_distances.png'), dpi=300, bbox_inches='tight')

            coverage = np.exp(-0.5 * (distances / radius)**2)

            # fig, ax = plt.subplots(figsize=(6.8, 6.8))
            # map = Basemap(llcrnrlat=min(y), urcrnrlat=max(y), llcrnrlon=min(x), urcrnrlon=max(x), resolution='l')
            # map.drawparallels([min(y), max(y)], labels=[True, False, False, True])
            # map.drawmeridians([min(x), max(x)], labels=[True, False, False, True])
            # img = ax.imshow(np.flipud(coverage), cmap='coolwarm', alpha=0.75, extent=[min(x), max(x), min(y), max(y)], zorder=1)
            # ax.scatter(rx, ry, 50, marker='x', color='black', zorder=2)
            # plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.5)
            # plt.savefig(os.path.join(os.getcwd(), "trajectory_analysis/assign_payoff_rangers", 'ranger' + str(i) + '_coverages.png'), dpi=300, bbox_inches='tight')

            ranger_coverage = np.maximum(ranger_coverage, coverage)

        fig, ax = plt.subplots(figsize=(6.8, 6.8))
        map = Basemap(llcrnrlat=min(y), urcrnrlat=max(y), llcrnrlon=min(x), urcrnrlon=max(x), resolution='l')
        map.drawparallels([min(y), max(y)], labels=[True, False, False, True])
        map.drawmeridians([min(x), max(x)], labels=[True, False, False, True])
        img = ax.imshow(np.flipud(ranger_coverage), cmap='coolwarm', alpha=0.75, extent=[min(x), max(x), min(y), max(y)], zorder=1)
        plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.5)
        plt.savefig(os.path.join(self.output_folder, 'ranger_coverages.png'), dpi=300, bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(6.8, 6.8))
        map = Basemap(llcrnrlat=min(y), urcrnrlat=max(y), llcrnrlon=min(x), urcrnrlon=max(x), resolution='l')
        map.drawparallels([min(y), max(y)], labels=[True, False, False, True])
        map.drawmeridians([min(x), max(x)], labels=[True, False, False, True])
        img = ax.imshow(np.flipud((1 - ranger_coverage) * self.elephant_density), cmap='coolwarm', alpha=0.75, extent=[min(x), max(x), min(y), max(y)], zorder=1)
        plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.5)
        plt.savefig(os.path.join(self.output_folder, 'uncovered_elephant_density.png'), dpi=300, bbox_inches='tight')

        uncovered_score = np.sum((1 - ranger_coverage) * self.elephant_density) / np.sum(self.elephant_density)
        total_cost = uncovered_score
        
        self.loss_history.append(total_cost)

        return total_cost

    def plot_kde_95(self, ranger_locations, save_plots=False):

        all_lons = np.concatenate([traj["longitude"].values for traj in self.best_trajs])
        all_lats = np.concatenate([traj["latitude"].values for traj in self.best_trajs])

        fig, ax = plt.subplots(figsize=(6.8, 6.8))

        min_lat = min(all_lats) 
        max_lat = max(all_lats)
        min_lon = min(all_lons) 
        max_lon = max(all_lons) 

        ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
        geotransform = ds.GetGeoTransform()
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]
        x0 = geotransform[0]
        y0 = geotransform[3]

        off_x_min = int((min_lon - x0) / pixel_width)
        off_y_min = int((max_lat - y0) / pixel_height)
        off_x_max = int((max_lon - x0) / pixel_width)
        off_y_max = int((min_lat - y0) / pixel_height)
        
        x_off = min(off_x_min, off_x_max)
        y_off = min(off_y_min, off_y_max)

        x_count = abs(off_x_max - off_x_min)
        y_count = abs(off_y_max - off_y_min)
        
        data_LULC = ds.ReadAsArray(x_off, y_off, x_count, y_count)

        data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

        for i in range(1, 10):
            data_LULC[data_LULC == data_value_map[i]] = i

        data_LULC = np.flip(data_LULC, axis=0)

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857') 
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
        row_size, col_size = data_LULC.shape

        LON_MIN, LAT_MIN = transform(inProj, outProj, min_lon, min_lat)
        LON_MAX, LAT_MAX = transform(inProj, outProj, max_lon, max_lat)

        map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        clrs = ["greenyellow","mediumpurple","turquoise", "plum", "black", "blue", "yellow", "mediumseagreen", "forestgreen"] 
        cmap, norm = colors.from_levels_and_colors(levels, clrs)

        map.imshow(data_LULC, cmap = cmap, norm=norm, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 0.5)

        map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
        map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

        x, y = transform(inProj, outProj, all_lons, all_lats)

        X = np.vstack([
            x,
            y
            ]).T
        
        kde = gaussian_kde(X.T)

        lat_grid, lon_grid = np.mgrid[min(y): max(y):100j, min(x): max(x):100j]
        positions = np.vstack([lon_grid.ravel(), lat_grid.ravel()])
        
        z = np.reshape(kde(positions).T, lat_grid.shape)

        z = z/z.sum()

        sorted_z = np.sort(z.flatten())
        cumsum_z = np.cumsum(sorted_z)

        levels_list = []
        
        for val in np.arange(0.05, 1.00, 0.10):
            threshold_idx = np.searchsorted(cumsum_z, val)
            levels_list.append(sorted_z[threshold_idx])

        contour = plt.contour(lon_grid, lat_grid, z, levels=levels_list, cmap="viridis", linewidths=2.5)

        for rx, ry in zip(ranger_locations[:, 0], ranger_locations[:, 1]):
            rx, ry = transform(inProj, outProj, rx, ry)
            plt.plot(rx, ry, 'ro', markersize=8)

        plt.tight_layout()

        if save_plots:
            plt.savefig(os.path.join(self.output_folder, 'kde_plot.png'), dpi=300, bbox_inches='tight')

        plt.close()

    def optimize(self, starting_positions="kmeans", max_steps=100):

        initial_positions = []

        all_lons = np.concatenate([traj["longitude"].values for traj in self.best_trajs])
        all_lats = np.concatenate([traj["latitude"].values for traj in self.best_trajs])

        lon_min = min(all_lons) 
        lon_max = max(all_lons) 
        lat_min = min(all_lats) 
        lat_max = max(all_lats)

        if starting_positions == "kmeans":
            points = np.column_stack((all_lons, all_lats))
            kmeans = KMeans(n_clusters=self.num_rangers, random_state=42)
            kmeans.fit(points)
            initial_positions = kmeans.cluster_centers_.flatten()

        elif starting_positions == "plantations":
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

            plantation_points = plantation_points[(plantation_points[:,0] >= lon_min) & (plantation_points[:,0] <= lon_max) & (plantation_points[:,1] >= lat_min) & (plantation_points[:,1] <= lat_max)]
            indices = np.random.choice(len(plantation_points), self.num_rangers, replace=False)
            plantation_points = plantation_points[indices]
            initial_positions = plantation_points.flatten()

        bounds = [(lon_min, lon_max), (lat_min, lat_max)] * self.num_rangers

        #method='L-BFGS-B', "Nelder-Mead", "BFGS", "Powell"

        result = minimize(
            self.cost_function_kde,
            initial_positions,
            method='Powell',
            bounds=bounds,
            options={'maxiter': max_steps}
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
        
        plt.savefig(os.path.join(self.output_folder, 'optimization_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()

        return

    def assign_ranger_payoffs(self):

        """Calculate payoffs for each ranger based on intercepted trajectories."""
        ranger_payoffs = []
        
        for i, ranger_pos in enumerate(self.ranger_location):
            intercepted_trajectories = 0
            
            for traj in self.best_trajs:
                traj_intercepted = False
                first_interception_time = None
                
                for idx, (lon, lat) in enumerate(zip(traj["longitude"], traj["latitude"])):
                    distance = np.sqrt((lon - ranger_pos[0])**2 + (lat - ranger_pos[1])**2)
                    if distance <= self.ranger_visibility_radius:
                        traj_intercepted = True
                        first_interception_time = idx
                        break
                
                if traj_intercepted:
                    intercepted_trajectories += 1
                    
            ranger_payoffs.append(intercepted_trajectories)
        
        return ranger_payoffs  

    def save_strategies_to_yaml(self, strategies, filename):
        """Clean and save ranger strategies in readable YAML format."""
        clean_strategies = []
        
        for strategy in strategies:
            clean_strategy = {
                'id': int(strategy['id']),
                'ranger_locations': [
                    [float(pos[0]), float(pos[1])] 
                    for pos in strategy['ranger_locations']
                ],
                'ranger_payoffs': [float(p) for p in strategy['ranger_payoffs']],
                'total_cost': float(strategy['total_cost']),
                'convergence': bool(strategy['convergence'])
            }
            clean_strategies.append(clean_strategy)
        
        with open(filename, 'w') as f:
            yaml.dump(clean_strategies, f, default_flow_style=False, sort_keys=False)

        return

    def generate_ranger_strategies(self, starting_positions="kmeans", num_strategies=10):
        """Generate multiple ranger deployment strategies and save results."""

        print("Generating Multiple Ranger Strategies")

        strategies = []

        initial_positions = []

        all_lons = np.concatenate([traj["longitude"].values for traj in self.best_trajs])
        all_lats = np.concatenate([traj["latitude"].values for traj in self.best_trajs])

        lon_min = min(all_lons) 
        lon_max = max(all_lons) 
        lat_min = min(all_lats) 
        lat_max = max(all_lats)

        if starting_positions == "kmeans":
            points = np.column_stack((all_lons, all_lats))
            kmeans = KMeans(n_clusters=self.num_rangers, random_state=42)
            kmeans.fit(points)
            initial_positions = kmeans.cluster_centers_.flatten()

        elif starting_positions == "plantations":
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

            plantation_points = plantation_points[(plantation_points[:,0] >= lon_min) & (plantation_points[:,0] <= lon_max) & (plantation_points[:,1] >= lat_min) & (plantation_points[:,1] <= lat_max)]
            indices = np.random.choice(len(plantation_points), self.num_rangers, replace=False)
            plantation_points = plantation_points[indices]
            initial_positions = plantation_points.flatten()

        bounds = [(lon_min, lon_max), (lat_min, lat_max)] * self.num_rangers

        for i in range(num_strategies):

            result = minimize(
                self.cost_function_kde,
                initial_positions,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            self.ranger_location = result.x.reshape(-1, 2)
            ranger_payoffs = self.assign_ranger_payoffs()
            
            final_positions = []
            for pos in self.ranger_location:
                final_positions.append([pos[0], pos[1]])
            
            strategy = {
                'id': i,
                'ranger_locations': final_positions,
                'ranger_payoffs': ranger_payoffs,
                'total_cost': result.fun,
                'convergence': result.success
            }
            
            strategies.append(strategy)
            
        output_file = os.path.join(self.output_folder, 'ranger_strategies_' + str(self.num_rangers) + 'rangers.yaml')
        self.save_strategies_to_yaml(strategies, output_file)
        
        return strategies


if __name__ == "__main__":

    model_params = {
        "year": 2010,
        "month": "Mar",
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
        "elephant_crop_habituation": False,
        "num_guards": 3,
        "ranger_visibility_radius": 500
        }

    experiment_name = "ranger-deployment-v1"

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

    data_folder = os.path.join(os.getcwd(), "model_runs", "exploratory-search-ID-01", starting_location, elephant_category, landscape_food_probability, 
                                    water_holes_probability, memory_matrix_type, num_days_agent_survives_in_deprivation, maximum_food_in_a_forest_cell, 
                                    elephant_thermoregulation_threshold, threshold_food_derivation_days, threshold_water_derivation_days, 
                                    slope_tolerance, num_days_agent_survives_in_deprivation, elephant_aggression_value,
                                    str(model_params["year"]), str(model_params["month"]))

    output_folder = os.path.join(os.getcwd(), "model_runs/", experiment_name, starting_location, elephant_category, landscape_food_probability, 
                                    water_holes_probability, memory_matrix_type, num_days_agent_survives_in_deprivation, maximum_food_in_a_forest_cell, 
                                    elephant_thermoregulation_threshold, threshold_food_derivation_days, threshold_water_derivation_days, 
                                    slope_tolerance, num_days_agent_survives_in_deprivation, elephant_aggression_value,
                                    str(model_params["year"]), str(model_params["month"]), "guard_agent_placement_optimisation")

    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)

    optimizer = optimise_ranger_locations(num_best_trajs = 40, 
                                          num_rangers=model_params["num_guards"], 
                                          ranger_visibility_radius=model_params["ranger_visibility_radius"], 
                                          data_folder = data_folder, 
                                          output_folder = output_folder)
    optimizer.optimize()

    optimizer.plot_trajectories_with_ranger_location()
    optimizer.plot_trajectories_untill_ranger_intervention()
    intersecting_trajs, non_intersecting_trajs = optimizer.filter_trajectories_in_ranger_radius()
    optimizer.plot_filtered_trajectories(intersecting_trajs, non_intersecting_trajs)

    optimizer.generate_ranger_strategies(starting_positions="kmeans", num_strategies = 2)

