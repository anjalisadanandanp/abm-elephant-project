#------------importing libraries----------------#
import os

import sys
sys.path.append(os.getcwd())

import pathlib
import yaml
import itertools
import mlflow
import smtplib
from email.mime.text import MIMEText
import time
from multiprocessing import freeze_support
import pandas as pd
import numpy as np
from osgeo import gdal
from pyproj import Proj, transform
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import colors
import osmnx as ox 
from scipy.ndimage import distance_transform_edt
import rasterio as rio   
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde


from mesageo_elephant_project.elephant_project.model.abm_model_HEC_v4 import batch_run_model
from experiments.ranger_deployment.experiment_names import FancyNameGenerator
#------------importing libraries----------------#




optimisation_args = {
    "max_optimisation_steps": 12,
    "num_strategies": 6
}




model_params_all = {
    "year": 2010,
    "month": ["Mar"],
    "num_bull_elephants": 1, 
    "area_size": 1100,              
    "spatial_resolution": 30, 
    "max_food_val_cropland": 100,
    "max_food_val_forest": [10],
    "prob_food_forest": [0.10],
    "prob_food_cropland": [0.10],
    "prob_water_sources": [0.00],
    "thermoregulation_threshold": [28],
    "num_days_agent_survives_in_deprivation": [10],     
    "knowledge_from_fringe": 1500,   
    "prob_crop_damage": 0.05,           
    "prob_infrastructure_damage": 0.01,
    "percent_memory_elephant": 0.375,   
    "radius_food_search": 750,     
    "radius_water_search": 750, 
    "radius_forest_search": 1500,
    "fitness_threshold": 0.4,   
    "terrain_radius": 750,       
    "slope_tolerance": [30],
    "num_processes": 8,
    "iterations": 40,
    "max_time_steps": 288*10,
    "aggression_threshold_enter_cropland": 1.0,
    "human_habituation_tolerance": 1.0,
    "elephant_agent_visibility_radius": 500,
    "plot_stepwise_target_selection": False,
    "threshold_days_of_food_deprivation": [0],
    "threshold_days_of_water_deprivation": [3],
    "number_of_feasible_movement_directions": 3,
    "track_in_mlflow": False,
    "elephant_starting_location": "user_input",
    "elephant_starting_latitude": 1049237,
    "elephant_starting_longitude": 8570917,
    "elephant_aggression_value": [0.8],
    "elephant_crop_habituation": False,
    "num_guards": 3,
    "ranger_visibility_radius": 500
    }



def generate_parameter_combinations(model_params_all):

    month = model_params_all["month"]
    max_food_val_forest = model_params_all["max_food_val_forest"]
    prob_food_forest = model_params_all["prob_food_forest"]
    prob_food_cropland = model_params_all["prob_food_cropland"]
    thermoregulation_threshold = model_params_all["thermoregulation_threshold"]
    threshold_days_food = model_params_all["threshold_days_of_food_deprivation"]
    threshold_days_water = model_params_all["threshold_days_of_water_deprivation"]
    prob_water_sources = model_params_all["prob_water_sources"]
    num_days_agent_survives_in_deprivation = model_params_all["num_days_agent_survives_in_deprivation"]
    slope_tolerance = model_params_all["slope_tolerance"]
    elephant_aggression_value = model_params_all["elephant_aggression_value"]

    combinations = list(itertools.product(
        month,
        max_food_val_forest,
        prob_food_forest,
        prob_food_cropland,
        thermoregulation_threshold,
        threshold_days_food,
        threshold_days_water,
        prob_water_sources,
        num_days_agent_survives_in_deprivation,
        slope_tolerance,
        elephant_aggression_value
    ))

    all_param_dicts = []
    for combo in combinations:
        params_dict = model_params_all.copy()
        
        params_dict.update({
            "month": combo[0],
            "max_food_val_forest": combo[1],
            "prob_food_forest": combo[2],
            "prob_food_cropland": combo[3],
            "thermoregulation_threshold": combo[4],
            "threshold_days_of_food_deprivation": combo[5],
            "threshold_days_of_water_deprivation": combo[6],
            "prob_water_sources": combo[7],
            "num_days_agent_survives_in_deprivation": combo[8],
            "slope_tolerance": combo[9],
            "elephant_aggression_value": combo[10]
        })
        
        all_param_dicts.append(params_dict)
    
    return all_param_dicts


def run_model(experiment_name, model_params, output_folder, step, strategy):

    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_folder, 'model_parameters.yaml'), 'w') as configfile:
        yaml.dump(model_params, configfile, default_flow_style=False)

    batch_run_model(model_params, experiment_name, output_folder, step, strategy)

    return


class Experiment:

    def __init__(self, email_address, email_password):
        self.email_address = email_address
        self.email_password = email_password

        self.all_trajectories = []
        self.elephant_kde = None
        self.loss_history = []

    def create_ranger_strategies(self, data_folder, num_rangers, step, num_strategies=1, area_size=1100, make_plots=True):

        def initialize_road_network(make_plots=True):

            MAP_COORDS=[9.3245, 76.9974]   

            area = {800:"area_800sqKm", 900:"area_900sqKm", 1000:"area_1000sqKm", 1100:"area_1100sqKm"}
            delta = {800:14142, 900:15000, 1000:15811, 1100: 16583}  

            latitude_center = MAP_COORDS[0]
            longitude_center = MAP_COORDS[1]

            inProj, outProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857') 
            center_lon, center_lat  = transform(inProj, outProj, longitude_center, latitude_center)

            latextend,lonextend = [center_lat-delta[area_size],center_lat+delta[area_size]],[center_lon-delta[area_size],center_lon+delta[area_size]]
            self.LAT_MIN_epsg3857,self.LAT_MAX_epsg3857 = latextend
            self.LON_MIN_epsg3857,self.LON_MAX_epsg3857 = lonextend

            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857') 
            self.LON_MIN, self.LAT_MIN = transform(inProj, outProj, self.LON_MIN_epsg3857, self.LAT_MIN_epsg3857)
            self.LON_MAX, self.LAT_MAX = transform(inProj, outProj, self.LON_MAX_epsg3857, self.LAT_MAX_epsg3857)
            
            north, south, east, west = self.LAT_MIN, self.LAT_MAX, self.LON_MIN, self.LON_MAX
            self.road_network=ox.graph.graph_from_bbox(north, south, east, west, network_type='all', simplify=True, retain_all=False, truncate_by_edge=False, clean_periphery=True, custom_filter=None)
            self.road_network = ox.project_graph(self.road_network,"epsg:3857")
            self.nodes_proj, self.edges_proj = ox.graph_to_gdfs(self.road_network, nodes=True, edges=True)

            if make_plots:
                visualize_road_network(self.nodes_proj, self.edges_proj)

            return

        def visualize_road_network(nodes_proj, edges_proj):

            fig, ax = plt.subplots(figsize=(6.8,6.8))
            edges_proj.plot(ax=ax, linewidth=1, color='gray')
            nodes_proj.plot(ax=ax, markersize=2, color='red')
            
            plt.title('Road Network Within the Study Area')
            ax.set_axis_off()

            plt.savefig(os.path.join(pathlib.Path(data_folder), "road_network_graph.png"), 
                                     dpi=300, bbox_inches='tight')
            plt.close()
            
            return 

        def create_road_matrix(edges_proj, row_size, col_size, make_plots=True):

            min_x = self.LON_MIN_epsg3857
            max_x = self.LON_MAX_epsg3857
            min_y = self.LAT_MIN_epsg3857
            max_y = self.LAT_MAX_epsg3857

            road_matrix = np.zeros((row_size, col_size))
            
            for _, edge in edges_proj.iterrows():

                coords = edge.geometry.coords

                for i in range(len(coords)-1):

                    xs = np.linspace(coords[i][0], coords[i+1][0], 100)
                    ys = np.linspace(coords[i][1], coords[i+1][1], 100)
                    
                    matrix_x = ((xs - min_x) / (max_x - min_x) * (row_size-1)).astype(int)
                    matrix_y = ((ys - min_y) / (max_y - min_y) * (col_size-1)).astype(int)
                    
                    road_matrix[matrix_y, matrix_x] = 1

            road_matrix = np.flipud(road_matrix)

            if make_plots:
                
                fig, ax = plt.subplots(figsize=(6.8,6.8))
                ax.imshow(road_matrix, cmap='gray')
                ax.axis('off')
                plt.savefig(os.path.join(pathlib.Path(data_folder), "road_matrix.png"), 
                                        dpi=300, bbox_inches='tight')
                plt.close()

            return road_matrix 

        def calculate_proximity_map(landscape_matrix, target_class, name, make_plots = True, save=False):

            landscape_matrix = np.array(landscape_matrix)
            binary_matrix = (landscape_matrix == target_class).astype(int)
            proximity_matrix = distance_transform_edt(1 - binary_matrix)
            proximity_matrix = proximity_matrix/np.max(proximity_matrix)

            if save:
                source = os.path.join(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
                with rio.open(source) as src:
                    ras_meta = src.profile

                loc = os.path.join(pathlib.Path(data_folder), "proximity_to_" + name + ".tif")
                with rio.open(loc, 'w', **ras_meta) as dst:
                    dst.write(proximity_matrix, 1)

            if make_plots:
                fig, ax = plt.subplots(figsize=(6.8,6.8))
                img = ax.imshow(proximity_matrix, cmap='coolwarm', vmin=0, vmax=np.max(proximity_matrix))
                ax.axis('off')
                cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
                plt.savefig(os.path.join(pathlib.Path(data_folder), "proximity_to_" + name + ".png"), 
                                         dpi=300, bbox_inches='tight')
                plt.close()
            
            return proximity_matrix
        
        def read_LULC_map():

            source = os.path.join(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
            LULC = gdal.Open(source).ReadAsArray()

            return LULC

        def get_inverse_proximity_map(plantation_proximity_map, name, make_plots = True):

            plantation_proximity_inverse = np.max(plantation_proximity_map) - plantation_proximity_map

            if make_plots:
                fig, ax = plt.subplots(figsize=(6.8,6.8))
                img = ax.imshow(plantation_proximity_inverse, cmap='coolwarm', vmin=0, vmax=np.max(plantation_proximity_inverse))
                ax.axis('off')
                cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
                plt.savefig(os.path.join(pathlib.Path(data_folder), "proximity_to_" + name + "_inverse.png"),
                                                dpi=300, bbox_inches='tight')
                plt.close() 

            return plantation_proximity_inverse

        def get_plantation_forest_fringe(lulc_matrix, distance=1000, make_plots=True):

            mapping = {5: 15, 4: 15, 14: 15, 1:15}  

            remapped_matrix = lulc_matrix.copy()

            for old_value, new_value in mapping.items():
                remapped_matrix[lulc_matrix == old_value] = new_value
    
            forest_mask = (remapped_matrix == 15).astype(int)
            plantation_mask = (lulc_matrix == 10).astype(int)

            forest_mask_inverse = np.max(forest_mask) - forest_mask
            
            plantation_forest_fringe = distance_transform_edt(forest_mask_inverse, return_distances=True)
            plantation_forest_fringe_mask = np.zeros_like(plantation_forest_fringe)

            for row in range(plantation_forest_fringe.shape[0]):
                for col in range(plantation_forest_fringe.shape[1]):
                    if plantation_forest_fringe[row, col] > 0 and plantation_forest_fringe[row, col]*33.33 <= distance and plantation_mask[row, col] == 1:
                        plantation_forest_fringe_mask[row, col] = plantation_forest_fringe[row, col]
                    elif forest_mask[row, col] == 1:
                        plantation_forest_fringe_mask[row, col] = -10
                    else:
                        plantation_forest_fringe_mask[row, col] = -10

            if make_plots:
                fig, ax = plt.subplots(figsize=(6.8,6.8))
                img = ax.imshow(plantation_forest_fringe_mask, cmap='coolwarm', vmin=np.min(plantation_forest_fringe_mask), vmax=np.max(plantation_forest_fringe_mask))
                ax.axis('off')
                cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
                plt.savefig(os.path.join(pathlib.Path(data_folder), "plantation_forest_fringe.png"), 
                                        dpi=300, bbox_inches='tight')   
                plt.close()

            return plantation_forest_fringe_mask

        def calculate_landuse(trajectory):

            ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
            data_LULC = ds.ReadAsArray()
            data_LULC = np.flip(data_LULC, axis=0)

            row_size, col_size = data_LULC.shape
            xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

            landuse = []

            for lon, lat in zip(trajectory["longitude"], trajectory["latitude"]):
                px = int((lon - xmin) / xres)
                py = int((lat - ymax) / yres)
                
                landuse.append(data_LULC[py, px])

            return landuse

        def cost_function_kde_v1(x=None, grid_resolution=100):

            ranger_positions = x.reshape(-1, 2)

            for traj in self.all_trajectories:
                landuse = calculate_landuse(traj)
                traj["landuse"] = landuse

            all_lons = []
            all_lats = []

            for traj in self.all_trajectories:
                traj = traj[traj["landuse"] == 10]
                if len(traj) > 0:
                    all_lons.extend(traj["longitude"].values)
                    all_lats.extend(traj["latitude"].values)

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

            ranger_coverage = np.zeros_like(self.elephant_density)
            
            ranger_x, ranger_y = transform(inProj, outProj, 
                                        ranger_positions[:, 0], 
                                        ranger_positions[:, 1])
            
            radius = model_params_all["ranger_visibility_radius"] /(111 * 1000)
            
            for i, (rx, ry) in enumerate(zip(ranger_x, ranger_y)):

                distances = np.sqrt((self.lon_grid - rx)**2 + (self.lat_grid - ry)**2)

                coverage = np.exp(-0.5 * (distances / radius)**2)

                ranger_coverage = np.maximum(ranger_coverage, coverage)


            # fig, ax = plt.subplots(figsize=(6.8, 6.8))
            # map = Basemap(llcrnrlat=min(y), urcrnrlat=max(y), llcrnrlon=min(x), urcrnrlon=max(x), resolution='l')
            # map.drawparallels([min(y), max(y)], labels=[True, False, False, True])
            # map.drawmeridians([min(x), max(x)], labels=[True, False, False, True])
            # img = ax.imshow(np.flipud(ranger_coverage), cmap='coolwarm', alpha=0.75, extent=[min(x), max(x), min(y), max(y)], zorder=1)
            # plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.5)
            # plt.savefig(os.path.join(data_folder, 'ranger_coverages.png'), dpi=300, bbox_inches='tight')
            # plt.close()

            # fig, ax = plt.subplots(figsize=(6.8, 6.8))
            # map = Basemap(llcrnrlat=min(y), urcrnrlat=max(y), llcrnrlon=min(x), urcrnrlon=max(x), resolution='l')
            # map.drawparallels([min(y), max(y)], labels=[True, False, False, True])
            # map.drawmeridians([min(x), max(x)], labels=[True, False, False, True])
            # img = ax.imshow(np.flipud((1 - ranger_coverage) * self.elephant_density), cmap='coolwarm', alpha=0.75, extent=[min(x), max(x), min(y), max(y)], zorder=1)
            # plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.5)
            # plt.savefig(os.path.join(data_folder, 'uncovered_elephant_density.png'), dpi=300, bbox_inches='tight')
            # plt.close()


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
                    forest_penalty += 10

            uncovered_score = np.sum((1 - ranger_coverage) * self.elephant_density) / np.sum(self.elephant_density)
            total_cost = uncovered_score + forest_penalty

            self.elephant_kde = None
            self.loss_history.append(total_cost)

            print(f"Total cost: {total_cost}", "Uncovered score: ", uncovered_score, "Forest penalty: ", forest_penalty)

            return total_cost
        
        def optimize(starting_positions="kmeans_all_points", max_steps=100, strategy=None):

            initial_positions = []

            all_lons = np.concatenate([traj["longitude"].values for traj in self.all_trajectories])
            all_lats = np.concatenate([traj["latitude"].values for traj in self.all_trajectories])
            
            lon_min = min(all_lons) 
            lon_max = max(all_lons) 
            lat_min = min(all_lats) 
            lat_max = max(all_lats)

            if starting_positions == "kmeans_all_points":
                points = np.column_stack((all_lons, all_lats))
                kmeans = KMeans(n_clusters=model_params_all["num_guards"], random_state=42)
                kmeans.fit(points)
                initial_positions = kmeans.cluster_centers_.flatten()

            elif starting_positions == "kmeans_plantation_points":

                for traj in self.all_trajectories:
                    landuse = calculate_landuse(traj)
                    traj["landuse"] = landuse

                all_lons = []
                all_lats = []

                for traj in self.all_trajectories:
                    traj = traj[traj["landuse"] == 10]
                    if len(traj) > 0:
                        all_lons.extend(traj["longitude"].values)
                        all_lats.extend(traj["latitude"].values)

                points = np.column_stack((all_lons, all_lats))
                kmeans = KMeans(n_clusters=model_params_all["num_guards"], random_state=42)
                kmeans.fit(points)
                initial_positions = kmeans.cluster_centers_.flatten()

            bounds = [(lon_min, lon_max), (lat_min, lat_max)] * model_params_all["num_guards"]

            #method='L-BFGS-B', "Nelder-Mead", "BFGS", "Powell"

            result = minimize(
                cost_function_kde_v1,
                initial_positions,
                method='Powell',
                bounds=bounds,
                options={'maxiter': max_steps}
            )
            
            ranger_location = result.x.reshape(-1, 2)

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
            ax.set_title('Best Loss value', fontsize=12)
            
            plt.savefig(os.path.join(data_folder, 'optimization_loss' + str(strategy) + '.png'), dpi=300, bbox_inches='tight')
            plt.close()

            return ranger_location, result.fun, result.success

        def sample_random_ranger_locations(preferences, num_strategies=1, make_plots=True):

            ranger_locations = []
            valid_points = []

            ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
            xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
            
            for i in range(num_rangers):
                for row in range(preferences.shape[0]):
                    for col in range(preferences.shape[1]):
                        if preferences[row, col] > 0:
                            latitude = ymax + yres*row
                            longitude = xmin + xres*col
                            valid_points.append([longitude, latitude])

            ranger_strategies = []

            for strategy in range(num_strategies):

                if valid_points:

                    sampled_indices = np.random.choice(len(valid_points), num_rangers, replace=False)
                    ranger_locations = [valid_points[i] for i in sampled_indices]

                    total_cost = 0
                    for ranger_x, ranger_y in ranger_locations:
                        px = int((ranger_x - xmin) / xres)
                        py = int((ranger_y - ymax) / yres)
                        total_cost += preferences[py, px]

                    strategy_data = {
                        'id': strategy,
                        'ranger_locations': ranger_locations,
                        'ranger_payoffs': [0.0] * num_rangers,
                        'total_cost': float(total_cost),
                        'convergence': True
                    }

                    ranger_strategies.append(strategy_data)
            
                if make_plots:

                    fig, ax = plt.subplots(figsize=(6.8,6.8))
                    map = Basemap(llcrnrlon=self.LON_MIN,llcrnrlat=self.LAT_MIN,urcrnrlon=self.LON_MAX,urcrnrlat=self.LAT_MAX, epsg=4326, resolution='l')
                    map.drawparallels([self.LAT_MIN, self.LAT_MAX], labels=[1,0,0,0])
                    map.drawmeridians([self.LON_MIN, self.LON_MAX], labels=[0,0,0,1])
                    img = ax.imshow(preferences, cmap='coolwarm_r', vmin=-10, vmax=np.max(preferences), 
                                    extent=[self.LON_MIN, self.LON_MAX, self.LAT_MIN, self.LAT_MAX],
                                    alpha=0.5)
                    cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)

                    for ranger in ranger_locations:
                        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')
                        x_new, y_new = transform(inProj, outProj, ranger[0], ranger[1])
                        ax.scatter(x_new, y_new, 50, marker='x', color='black', zorder=2)

                    plt.savefig(os.path.join(pathlib.Path(data_folder), "ranger_locations__" + str(strategy) + ".png"), 
                                            dpi=300, bbox_inches='tight')   
                    plt.close()

            return ranger_strategies

        def assign_ranger_payoffs(ranger_location):

            """Calculate payoffs for each ranger based on intercepted trajectories."""
            ranger_payoffs = []
            
            for i, ranger_pos in enumerate(ranger_location):
                intercepted_trajectories = 0
                
                for traj in self.all_trajectories:
                    traj_intercepted = False
                    first_interception_time = None
                    
                    for idx, (lon, lat) in enumerate(zip(traj["longitude"], traj["latitude"])):
                        distance = np.sqrt((lon - ranger_pos[0])**2 + (lat - ranger_pos[1])**2)
                        if distance <= model_params_all["ranger_visibility_radius"]:
                            traj_intercepted = True
                            first_interception_time = idx
                            break
                    
                    if traj_intercepted:
                        intercepted_trajectories += 1
                        
                ranger_payoffs.append(intercepted_trajectories)
            
            return ranger_payoffs  
        
        def ranger_strategies_v1(num_strategies=1, make_plots=True):

            ranger_strategies = []

            for strategy in range(num_strategies):

                ranger_locations, total_cost, convergence = optimize(starting_positions="kmeans_all_points", 
                                                                     max_steps=optimisation_args["max_optimisation_steps"],
                                                                     strategy=strategy)

                strategy_data = {
                    'id': strategy,
                    'ranger_locations': ranger_locations.tolist(),
                    'ranger_payoffs': assign_ranger_payoffs(ranger_locations),
                    'total_cost': float(total_cost),
                    'convergence': convergence
                }

                ranger_strategies.append(strategy_data)
            
                if make_plots:

                    fig, ax = plt.subplots(figsize=(6.8,6.8))
                    map = Basemap(llcrnrlon=self.LON_MIN,llcrnrlat=self.LAT_MIN,urcrnrlon=self.LON_MAX,urcrnrlat=self.LAT_MAX, epsg=4326, resolution='l')
                    map.drawparallels([self.LAT_MIN, self.LAT_MAX], labels=[1,0,0,0])
                    map.drawmeridians([self.LON_MIN, self.LON_MAX], labels=[0,0,0,1])

                    all_lons = np.concatenate([traj["longitude"].values for traj in self.all_trajectories])
                    all_lats = np.concatenate([traj["latitude"].values for traj in self.all_trajectories])

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

                    plt.contour(lon_grid, lat_grid, z, levels=levels_list, cmap="viridis", linewidths=2.5)

                    for ranger in ranger_locations:
                        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')
                        x_new, y_new = transform(inProj, outProj, ranger[0], ranger[1])
                        ax.scatter(x_new, y_new, 50, marker='x', color='black', zorder=2)

                    plt.savefig(os.path.join(pathlib.Path(data_folder), "ranger_locations__" + str(strategy) + ".png"), 
                                            dpi=300, bbox_inches='tight')   
                    plt.close()

            return ranger_strategies

        def get_forest_fringe(distance=1000, make_plots=True, save=False):

            initialize_road_network()

            road_matrix = create_road_matrix(self.edges_proj, 1069, 1070, make_plots=make_plots)
            road_proximity_map = calculate_proximity_map(landscape_matrix=road_matrix, target_class=1, name="roads", make_plots=make_plots, save=save)
            road_proximity_inverse = get_inverse_proximity_map(road_proximity_map, "roads", make_plots=make_plots)

            LULC = read_LULC_map() 

            plantation_proximity_map = calculate_proximity_map(landscape_matrix=LULC, target_class=10, name="plantations", make_plots=make_plots, save=save)
            plantation_proximity_inverse = get_inverse_proximity_map(plantation_proximity_map, "plantations", make_plots=make_plots)
            
            river_proximity_map = calculate_proximity_map(landscape_matrix=LULC, target_class=5, name="rivers", make_plots=make_plots, save=save)
            river_proximity_inverse = get_inverse_proximity_map(river_proximity_map, "rivers", make_plots=make_plots)

            ebf_proximity_map = calculate_proximity_map(landscape_matrix=LULC, target_class=15, name="evergreen_broadleaf_forest", make_plots=True, save=save)
            ebf_proximity_inverse = get_inverse_proximity_map(ebf_proximity_map, "ebf", make_plots=make_plots)

            plantation_forest_distances = get_plantation_forest_fringe(LULC, distance=distance, make_plots=make_plots)

            return plantation_forest_distances
            
        if step == 1:   
            plantation_forest_distances = get_forest_fringe()
            ranger_locations = sample_random_ranger_locations(preferences=plantation_forest_distances, num_strategies=4, make_plots=True)

        else:
            ranger_locations = ranger_strategies_v1(num_strategies=1, make_plots=make_plots)

        with open(os.path.join(pathlib.Path(data_folder), "ranger_strategies_" + str(num_rangers) + "rangers.yaml"), 'w') as file:
            yaml.dump(ranger_locations, file, default_flow_style=False)

        return ranger_locations

    def evaluate_trajectories(self, data_folder, model_params, step, strategy, num_rangers, make_plots=False):

        def read_experiment_data(yaml_file_path):

            # print(f"Reading YAML file: {yaml_file_path}")

            yaml_file = os.path.join(yaml_file_path, "ranger_strategies_" + str(num_rangers) + "rangers.yaml")

            try:
                # print(f"Reading YAML file: {yaml_file}")
                with open(yaml_file, 'r') as file:
                    data = yaml.safe_load(file)
                    
                for experiment in data:
                    experiment['ranger_locations'] = np.array(experiment['ranger_locations'])
                    experiment['ranger_payoffs'] = np.array(experiment['ranger_payoffs'])
                    
                return data
            
            except Exception as e:
                print(f"Error reading YAML file: {e}")
                return None

        def filter_trajectories_in_ranger_radius(trajectories, ranger_locations):
            """Filter trajectories that intersect with ranger visibility circles."""

            intersecting_trajs = []
            non_intersecting_trajs = []
            
            for data in trajectories:
                trajectory_intersects = False
                longitude, latitude = data["longitude"], data["latitude"]


                for ranger in ranger_locations:
                    ranger_lon, ranger_lat = ranger[0], ranger[1]
                    
                    for lon, lat in zip(longitude, latitude):
                        distance = np.sqrt((lon - ranger_lon)**2 + (lat - ranger_lat)**2)
                        
                        if distance <= model_params["ranger_visibility_radius"]:
                            trajectory_intersects = True
                            break
                            
                    if trajectory_intersects:
                        break
                        
                if trajectory_intersects:
                    intersecting_trajs.append(data)
                else:
                    non_intersecting_trajs.append(data)
                    
            return intersecting_trajs, non_intersecting_trajs

        def plot_filtered_trajectories(intersecting_trajs, non_intersecting_trajs, ranger_locations):

            fig, ax = plt.subplots(figsize=(6.8,6.8))
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

            for ranger in ranger_locations:
                longitude, latitude = transform(inProj, outProj, ranger[0], ranger[1])
                x_new, y_new = map(longitude,latitude)
                ax.scatter(x_new, y_new, 25, marker='x', color='white')
                
                radius = model_params["ranger_visibility_radius"]/(111*1000)  
                circle = plt.Circle((x_new, y_new), radius, facecolor='purple', fill=True, alpha=0.5, edgecolor='black', linewidth=1)
                ax.add_artist(circle)

            plt.savefig(os.path.join(pathlib.Path(data_folder), 
                                     "trajectories_with_ranger_locations__numrangers" + str(model_params["num_guards"]) + "_rangervisibility" + str(model_params["ranger_visibility_radius"]) + "m_v2.png"), 
                                     dpi=300, bbox_inches='tight')

        def find_first_visible_entry(trajectory, ranger_locations):
            
            first_visible_entries = None
        
            for idx, row in trajectory.iterrows():
                for ranger_location in ranger_locations:
                    distance = ((row['longitude'] - ranger_location[0])**2 + 
                                (row['latitude'] - ranger_location[1])**2)**0.5
                    
                    if distance <= model_params["ranger_visibility_radius"]:
                        first_visible_entries = idx
                        return first_visible_entries
                    else:
                        first_visible_entries = None

            return first_visible_entries

        def calculate_landuse_time(trajectory):
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

        def calculate_daily_landuse_time(trajectory, steps_per_day=288):
            """Calculate landuse time for each day."""
            
            days = len(trajectory["longitude"]) // steps_per_day
            daily_landuse = []
            
            for day in range(days):
                start_idx = day * steps_per_day
                end_idx = start_idx + steps_per_day
                
                day_trajectory = trajectory.iloc[start_idx:end_idx]
                
                landuse_times = calculate_landuse_time(day_trajectory)
                daily_landuse.append(landuse_times)
                
            return daily_landuse
        
        def plot_trajectories_untill_ranger_intervention(trajectories, ranger_locations):

            fig, ax = plt.subplots(figsize = (6.8, 6.8))
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

            for i, traj in enumerate(trajectories):

                first_visible_entry = find_first_visible_entry(traj, ranger_locations)

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

            for ranger in ranger_locations:
                longitude, latitude = transform(inProj, outProj, ranger[0], ranger[1])
                x_new, y_new = map(longitude,latitude)

                ax.scatter(x_new, y_new, 25, marker='x', color='white', zorder=2)

                radius = model_params["ranger_visibility_radius"]/(111*1000)  
                circle = plt.Circle((x_new, y_new), radius, facecolor='purple', fill=True, alpha=0.5, edgecolor='black', linewidth=1)
                ax.add_artist(circle)

            plt.savefig(os.path.join(pathlib.Path(data_folder), 
                                     "trajectories_until_ranger_proximity__rangervisibility" + str(model_params["ranger_visibility_radius"]) + "m_v1.png"), 
                                     dpi=300, bbox_inches='tight')

            return

        def calculate_average_plantation_usage(trajectories):
            #cost: average number of time steps spent in the crop fields

            plantation_usage = 0
            for traj in trajectories:
                landuse_times = calculate_landuse_time(traj)[7]
                plantation_usage += landuse_times
            average_usage = plantation_usage/len(trajectories)

            print("Number of intersecting trajectories:", len(intersecting_trajs), "Number of non-intersecting trajectories:", len(non_intersecting_trajs), "Average Plantation Usage:", average_usage)

            return average_usage

        def is_df_in_list(target_df, df_list):
            return any(target_df.equals(df) for df in df_list)

        def sort_trajectories(trajectory_payoffs, trajectories, make_plots=True, name="v1"):

            """Sort trajectories based on payoffs."""

            pairs = list(zip(trajectory_payoffs, trajectories))
            sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
            
            trajs_with_best_payoffs = [traj for _, traj in sorted_pairs]
            payoffs = [payoff for payoff,_ in sorted_pairs]

            if make_plots:

                fig, ax = plt.subplots(figsize=(6.8,6.8))
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

                for data, payoff, color in zip(trajs_with_best_payoffs, payoffs, payoff_colors):

                    longitude, latitude = transform(inProj, outProj, data["longitude"], data["latitude"])
                    x_new, y_new = map(longitude, latitude)
                    
                    ax.quiver(x_new[:-1], y_new[:-1],
                            x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1],
                            scale_units='xy', angles='xy', 
                            scale=1, zorder=int(payoff*100), color=color, width=0.0025)
                    
                    ax.scatter(x_new[0], y_new[0], 25, marker='o', color='black', zorder=int(payoff*100))
                    ax.scatter(x_new[-1], y_new[-1], 25, marker='^', color='black', zorder=int(payoff*100))

                sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, 
                                            norm=plt.Normalize(min(trajectory_payoffs), 
                                                            max(trajectory_payoffs)))
                cax2 = fig.add_axes([0.2, 0.05, 0.6, 0.02]) 
                cbar2 = plt.colorbar(sm, cax=cax2, orientation='horizontal')
                cbar2.set_label('payoffs')

                plt.savefig(os.path.join(pathlib.Path(data_folder),
                                                    "trajectories_with_payoffs" + "_" + name + ".png"),
                                                    dpi=300, bbox_inches='tight'),

                plt.close()

            return trajs_with_best_payoffs, payoffs
            
        def assign_payoffs_v1(trajectories, ranger_locations, cost):

            trajectory_payoffs = []

            intersecting_trajs, non_intersecting_trajs = filter_trajectories_in_ranger_radius(trajectories, ranger_locations)

            for i, traj in enumerate(trajectories):
                daily_landuse_time = calculate_daily_landuse_time(traj)
                daily_food_consumed = traj["food_consumed"][::288].values[1:]
                trajectory_fitness = traj["fitness"][::288].values[-1]
                plantation_use_time =[day[7] for day in daily_landuse_time]

                if is_df_in_list(traj, intersecting_trajs):
                    for plantation_use, food in zip(plantation_use_time, daily_food_consumed):
                        if plantation_use > 0:
                            trajectory_fitness -= cost

                elif is_df_in_list(traj, non_intersecting_trajs):
                    for plantation_use, food in zip(plantation_use_time, daily_food_consumed):
                        if plantation_use > 0 and food >= traj["daily_dry_matter_intake"].unique()[0]:
                            trajectory_fitness += cost

                trajectory_payoffs.append(trajectory_fitness)

            return trajectory_payoffs
        
        folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]

        trajectories = []

        for folder in folders:

            agent_data = pd.read_csv(os.path.join(data_folder, folder, "output_files/agent_data.csv"))
            agent_data = agent_data[agent_data["AgentID"] == "bull_0"].reset_index(drop=True)
            trajectories.append(agent_data)
            self.all_trajectories.append(agent_data)

            ranger_strategy = read_experiment_data(os.path.join(data_folder, folder, "env/ranger_strategy/"))
            converged_experiments = [exp for exp in ranger_strategy if exp['convergence']]
            best_experiment = min(converged_experiments, key=lambda x: x['total_cost'])
            ranger_locations = best_experiment['ranger_locations']

        intersecting_trajs, non_intersecting_trajs = filter_trajectories_in_ranger_radius(trajectories, ranger_locations)
        
        if make_plots:
            plot_filtered_trajectories(intersecting_trajs, non_intersecting_trajs, ranger_locations)
            plot_trajectories_untill_ranger_intervention(trajectories, ranger_locations)

        average_usage = calculate_average_plantation_usage(trajectories)

        trajectory_payoffs = assign_payoffs_v1(trajectories, ranger_locations, cost=0.05)

        sorted_trajectories, sorted_payoffs = sort_trajectories(trajectory_payoffs, trajectories, make_plots=True, name="v1")

        return 

    def run_experiment(self, send_notification):

        print("Running experiment...!")

        freeze_support()

        start = time.time()

        generator = FancyNameGenerator()
        experiment_name = generator.generate_name()

        if model_params_all["track_in_mlflow"] == True:
            try:
                mlflow.create_experiment(experiment_name)
            except:
                print("experiment already exists")

        param_dicts = generate_parameter_combinations(model_params_all)

        for model_params in param_dicts:

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



            for step in range(1, optimisation_args["max_optimisation_steps"]+1):

                print("\nstep:", step)

                for strategy in range(1, optimisation_args["num_strategies"]+1):

                    print("\nstrategy:", strategy)

                    data_folder = os.path.join(os.getcwd(), "model_runs", experiment_name, starting_location, elephant_category, landscape_food_probability, 
                                                    water_holes_probability, memory_matrix_type, num_days_agent_survives_in_deprivation, maximum_food_in_a_forest_cell, 
                                                    elephant_thermoregulation_threshold, threshold_food_derivation_days, threshold_water_derivation_days, 
                                                    slope_tolerance, num_days_agent_survives_in_deprivation, elephant_aggression_value,
                                                    str(model_params["year"]), str(model_params["month"]), "abm-runs-with-guard-agents", 
                                                    "step-" + str(step), "strategy-" + str(strategy))

                    output_folder = os.path.join(os.getcwd(), "model_runs/", experiment_name, starting_location, elephant_category, landscape_food_probability, 
                                                    water_holes_probability, memory_matrix_type, num_days_agent_survives_in_deprivation, maximum_food_in_a_forest_cell, 
                                                    elephant_thermoregulation_threshold, threshold_food_derivation_days, threshold_water_derivation_days, 
                                                    slope_tolerance, num_days_agent_survives_in_deprivation, elephant_aggression_value,
                                                    str(model_params["year"]), str(model_params["month"]), "guard_agent_placement_optimisation",
                                                    "step-" + str(step), "strategy-" + str(strategy))

                    path = pathlib.Path(output_folder)
                    path.mkdir(parents=True, exist_ok=True)


                    #------------------create ranger strategies------------------#
                    self.create_ranger_strategies(output_folder, num_rangers=3, step=step, num_strategies=1, make_plots=True)
                    #------------------create ranger strategies------------------#



                    #------------run agent based model----------------#
                    run_model(experiment_name, model_params, data_folder, step, strategy)
                    #------------run agent based model----------------#




                    #-----evaluate the effectiveness of the trajectories and ranger placement-----#
                    self.evaluate_trajectories(data_folder, model_params, step, strategy, num_rangers=3, make_plots=True)
                    #-----evaluate the effectiveness of the trajectories and ranger placement-----#




        end = time.time()

        print("\n Total time taken:", (end-start), "seconds")

        if send_notification == True:
            self.send_notification_email()

    def send_notification_email(self):
        msg = MIMEText("elephant-abm-project: Your experiment has finished running!")
        msg['Subject'] = "Experiment Notification: PAMPA"
        msg['To'] = self.email_address

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.email_address, self.email_password)
            server.sendmail(self.email_address, self.email_address, msg.as_string())
            server.quit()
            print("Notification email sent successfully.")

        except Exception as e:
            print("Error sending email:", e)






if __name__ == "__main__":
    experiment = Experiment("anjalisadanandan96@gmail.com", "fqdceolumrwtnmxo")
    experiment.run_experiment(send_notification=False)
    
