import os

import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from osgeo import gdal
from mpl_toolkits.basemap import Basemap 
import matplotlib.cm as cm   
from matplotlib import colors
from pyproj import Proj, transform 
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')


class optimize_guard_deployment():

    """
    Algorithm for optimal guard deployment:
    """

    def __init__(self, num_guards, guard_visibility, num_best_trajs, expt_folder):
        self.num_guards = num_guards
        self.guard_visibility = guard_visibility
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
    
    def filter_daily_trajectory_data(self):

        self.elephant_locations = {}

        for day in range(0, self.max_simulation_length//288):

            day_lats = []
            day_lons = []
            for traj in self.best_trajs:
                day_traj = traj.iloc[day*288:day*288+288]
                day_lats.extend(day_traj["latitude"])
                day_lons.extend(day_traj["longitude"])

            self.elephant_locations[day] = {"latitudes": day_lats, "longitudes": day_lons}

    def plot_daily_hotspots_kde_95(self, save_plots=False):

        n_days = self.max_simulation_length//288
        n_cols = int(np.ceil(np.sqrt(n_days)))
        n_rows = int(np.ceil(n_days/n_cols))

        fig = plt.figure(figsize=(10*n_cols, 10*n_rows))

        all_lats = []
        all_lons = []

        for day in self.elephant_locations:
            all_lats.extend(self.elephant_locations[day]['latitudes'])
            all_lons.extend(self.elephant_locations[day]['longitudes'])
    
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

        for day in range(0, self.max_simulation_length//288):

            day_data = self.elephant_locations[day]
        
            ax = fig.add_subplot(n_rows, n_cols, day+1)

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

            x, y = transform(inProj, outProj, day_data['longitudes'], day_data['latitudes'])

            X = np.vstack([
                x,
                y
                ]).T
            
            kde = stats.gaussian_kde(X.T)

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

            contour = plt.contour(lon_grid, lat_grid, z, levels=levels_list, cmap="viridis")
            plt.title(f'Day: {day + 1}')

        plt.tight_layout()

        if save_plots:
            plt.savefig(os.path.join(os.getcwd(), "trajectory_analysis/outputs", self.expt_folder, 'kde_plots_daily.png'), dpi=300, bbox_inches='tight')

        plt.close()

    def identify_hotspots_DBSCAN(self):

        hotspots = {}
        
        for day in self.elephant_locations:
            coords = np.vstack([
                self.elephant_locations[day]['longitudes'],
                self.elephant_locations[day]['latitudes']
            ]).T
            
            clustering = DBSCAN(eps=5, min_samples=5).fit(coords)
            
            unique_labels = set(clustering.labels_)
            cluster_centers = []
            
            for label in unique_labels:
                if label != -1:  # Exclude noise points
                    mask = clustering.labels_ == label
                    center = coords[mask].mean(axis=0)
                    cluster_centers.append(center)
                    
            hotspots[day] = np.array(cluster_centers)
        
        self.hotspots = hotspots

        return

    def plot_hotspots_DBSCAN(self):

        n_days = self.max_simulation_length//288

        n_cols = int(np.ceil(np.sqrt(n_days)))
        n_rows = int(np.ceil(n_days/n_cols))
        
        fig = plt.figure(figsize=(10*n_cols, 10*n_rows))

        all_lats = []
        all_lons = []

        for day in self.elephant_locations:
            all_lats.extend(self.elephant_locations[day]['latitudes'])
            all_lons.extend(self.elephant_locations[day]['longitudes'])
    
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

        for day in range(n_days):

            ax = fig.add_subplot(n_rows, n_cols, day+1)
            
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

            if day in self.hotspots:

                try:
                    hotspot_lons = self.hotspots[day][:,0]
                    hotspot_lats = self.hotspots[day][:,1]
                    hotspot_lons, hotspot_lats = transform(inProj, outProj, hotspot_lons, hotspot_lats)
                    x, y = map(hotspot_lons, hotspot_lats)
                    map.scatter(x, y, c='red', marker='*', s=200, label='Hotspots')
                except:
                    print("day:", self.hotspots[day])
        
            plt.title(f'Day {day+1}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), "trajectory_analysis/outputs", self.expt_folder,'hotspots_DBSCAN.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def optimize_coverage(self):

        return
    
    def generate_schedule(self):

        return

    def optimize_single_timeframe(self):

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
    "num_processes": 8,
    "iterations": 8,
    "max_time_steps": 288*10,
    "aggression_threshold_enter_cropland": 1.0,
    "elephant_agent_visibility_radius": 500,
    "plot_stepwise_target_selection": True,
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

expt_folder = os.path.join(experiment_name, starting_location, elephant_category, landscape_food_probability, 
                                water_holes_probability, memory_matrix_type, num_days_agent_survives_in_deprivation, maximum_food_in_a_forest_cell, 
                                elephant_thermoregulation_threshold, threshold_food_derivation_days, threshold_water_derivation_days, 
                                slope_tolerance, num_days_agent_survives_in_deprivation, elephant_aggression_value,
                                str(model_params["year"]), str(model_params["month"]))

optimise_rangers = optimize_guard_deployment(num_guards=5, 
                          guard_visibility=200, 
                          num_best_trajs = 50,
                          expt_folder=expt_folder)

optimise_rangers.filter_daily_trajectory_data()
optimise_rangers.plot_daily_hotspots_kde_95(save_plots=True)
optimise_rangers.identify_hotspots_DBSCAN()
optimise_rangers.plot_hotspots_DBSCAN()
