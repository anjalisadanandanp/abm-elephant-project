import os
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import pathlib
import yaml
from osgeo import gdal
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import random
import itertools
from pyproj import Proj, transform  
from mpl_toolkits.basemap import Basemap    
import rasterio
from rasterio.features import shapes
import fiona
import geojson
import matplotlib.cm as cm
import shutil

import warnings
warnings.filterwarnings("ignore")


fontsize = 12
plt.rcParams.update(
    {
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "figure.titlesize": fontsize,
    }
)


import sys
sys.path.append(os.getcwd())

module = importlib.import_module('guarding-policies.abm-runs-no-memory.abm_model_HEC_with_landscape_deterrent_policies_with_ranger_proximity_learning_model')
batch_run_model = module.batch_run_model



def make_trajectory_summary_plots_v3(base_path, output_folder):

    def raster_to_geojson(input_raster_path, output_geojson_path):

        with rasterio.open(input_raster_path) as src:
            image = src.read(1)

            if image.dtype not in ['int16', 'int32', 'uint8', 'uint16', 'float32']:
                    image = image.astype('uint8')

            mask = image > 0
            mask = mask.astype('uint8')

            results = [
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) in enumerate(shapes(mask, transform=src.transform))
                if v == 1 
            ]

            with fiona.open(
                output_geojson_path, 
                'w', 
                driver='GeoJSON',
                schema={'geometry': 'Polygon', 'properties': {'raster_val': 'int'}}
            ) as dst:
                for feature in results:
                    dst.write(feature)
        return

    # simulation_repeats = os.listdir(base_path)
    simulation_repeats = [
    item for item in os.listdir(base_path) 
    if os.path.isdir(os.path.join(base_path, item))
    ]


    try:
        simulation_repeats.remove("attacker_strategy_matrix.png")
    except:
        pass

    try:
        simulation_repeats.remove("attacker_strategy_matrix.tif")
    except:
        pass

    try:
        simulation_repeats.remove("defender_coverage_matrix.png")
    except:
        pass

    try:
        simulation_repeats.remove("defender_coverage_matrix.tif")
    except:
        pass

    try:
        simulation_repeats.remove("model_parameters.yaml")
    except:
        pass
    
    ds = gdal.Open(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix_0.tif"))

    data = ds.ReadAsArray()
    data = np.flip(data, axis=0)
    row_size, col_size = data.shape
    xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

    fig, ax = plt.subplots(figsize = (8,8))
    ax.yaxis.set_inverted(True)

    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
    LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
    LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

    map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

    raster_to_geojson(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix_0.tif"), os.path.join(output_folder, 'guarded_patches.geojson'))

    with open('mesageo_elephant_project/elephant_project/geojson_files/landuse_10.geojson', 'r') as f:
        geojson_object = geojson.load(f)

    for feature in geojson_object['features']:
        coords = feature['geometry']['coordinates'][0]
        coords = [transform(inProj, outProj, lon, lat) for lon, lat in coords]
        coords = [(lon, lat) for lon, lat in coords]
        lon, lat = zip(*coords)

        plt.fill(lon, lat, color='yellow', alpha=0.20, zorder=1)

        map.plot(lon, lat, marker=None, color='black', linewidth=1, zorder=2)

    with open(os.path.join(output_folder, 'guarded_patches.geojson'), 'r') as f:
        geojson_object = geojson.load(f)

    for feature in geojson_object['features']:
        coords = feature['geometry']['coordinates'][0]
        coords = [transform(inProj, outProj, lon, lat) for lon, lat in coords]
        coords = [(lon, lat) for lon, lat in coords]
        lon, lat = zip(*coords)
        map.plot(lon, lat, marker=None, color='red', linewidth=2, zorder=3)

    map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
    map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

    for simulation_repeat in simulation_repeats:

        try:
            agent_data = pd.read_csv(os.path.join(base_path, simulation_repeat, "output_files", "agent_data.csv"))
            longitude = agent_data['longitude'].values
            latitude = agent_data['latitude'].values

            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
            longitude, latitude = transform(inProj, outProj, longitude, latitude)
            x_new, y_new = map(longitude,latitude)
            C = np.arange(len(x_new))
            nz = mcolors.Normalize()
            nz.autoscale(C)

            ax.quiver(x_new[:-1], y_new[:-1], 
                        x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1], 
                        scale_units='xy', angles='xy', 
                        scale=1, zorder=2, color = cm.jet(nz(C)), 
                        width=0.0010)

            ax.scatter(x_new[0], y_new[0], 5, marker='o', color='black', zorder=2) 
            ax.scatter(x_new[-1], y_new[-1], 5, marker='^', color='black', zorder=2) 

        except:
            pass

    plt.savefig(os.path.join(output_folder, "summary_of_all_simulated_trajectories.png"), dpi=750, bbox_inches='tight')

    plt.close()

    return

def calculate_crop_raid_loss(output_folder, current_game_step):

    def lat_lon_to_pixel(lats, lons, xmin, ymax, xres, yres):
        """Convert lat/lon coordinates to pixel coordinates"""
        rows = ((ymax - lats) / -yres).astype(int)
        cols = ((lons - xmin) / xres).astype(int)
        return rows, cols

    def find_cropland_use_indices(landuse_sequence):
        indices = []
        start = None
        
        for i, value in enumerate(landuse_sequence):
            if value == 10:
                if start is None:
                    start = i
            else:
                if start is not None:
                    # End subsequence if current value is not 10, 3, 9, or 6
                    if value not in [10, 3, 9, 6]:
                        indices.append((start, i))
                        start = None
        
        # Handle case where sequence ends while in a subsequence starting with 10
        # Only add if the last value is not 10, 3, 9, or 6
        if start is not None:
            last_value = landuse_sequence[-1]
            if last_value not in [10, 3, 9, 6]:
                indices.append((start, len(landuse_sequence)))
        
        return indices
        
    def find_penalties_based_on_intercepted_trajectories(output_folder):

        boundary_patch_matrix = gdal.Open("guarding-policies/abm-runs-learning-memory/defender_coverage_matrix.tif").ReadAsArray()
        agricultural_plot_matrix = gdal.Open(os.path.join("create-landholding-matrix/agricultural_plots_assignment.tif")).ReadAsArray()

        boundary_patch_unique = np.unique(boundary_patch_matrix)
        agricultural_plot_unique = np.unique(agricultural_plot_matrix)

        association_df = pd.DataFrame(
            data=0,
            index=boundary_patch_unique,
            columns=agricultural_plot_unique
        )

        association_df.to_csv(os.path.join(output_folder, "association_df_init.csv"))

        simulation_repeats = os.listdir(output_folder)

        simulation_folders = [
            os.path.join(output_folder, item)
            for item in simulation_repeats
            if os.path.isdir(os.path.join(output_folder, item))
        ]

        agricultural_plots_attacked = {}

        for simulation_repeat in simulation_folders:
            
            try:

                agent_data = pd.read_csv(os.path.join(simulation_repeat, "output_files", "agent_data.csv"))

                geotransform = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").GetGeoTransform()
                ag_xmin, ag_xres, ag_xskew, ag_ymax, ag_yskew, ag_yres = geotransform

                landuse_matrix = gdal.Open(os.path.join(simulation_repeat, "env", "LULC.tif")).ReadAsArray()
                agricultural_plts = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").ReadAsArray()

                rows, cols = lat_lon_to_pixel(
                    agent_data["latitude"].values, agent_data["longitude"].values, 
                    ag_xmin, ag_ymax, ag_xres, ag_yres
                )
                
                landuse_values = landuse_matrix[rows, cols]

                indices = find_cropland_use_indices(landuse_values)
                
                for sequence in indices:

                    for i in range(sequence[1] - sequence[0]):

                        if agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]] != 0:

                            if agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]] not in agricultural_plots_attacked:
                                agricultural_plots_attacked[agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]]] = 0
                            
                            agricultural_plots_attacked[agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]]] += 1

            except:
                pass

        # print("\n")

        covered_targets_matrix = gdal.Open(os.path.join(OUTPUT_FOLDER, "coverage_matrix_init/game_step_1/defender_coverage_matrix.tif")).ReadAsArray()
        covered_targets = np.unique(covered_targets_matrix)
        covered_targets = covered_targets[covered_targets != 0]


        # print("covered boundaries:", covered_targets)


        for plot, count in agricultural_plots_attacked.items():
            # print(f"Agricultural plot {plot} was attacked {count} times.")

            association_df.loc[covered_targets, plot] += count

        association_df.to_csv(os.path.join(output_folder, "association_df_updated.csv"))

        ds = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif")
        data = ds.ReadAsArray()
        row_size, col_size = data.shape
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

        fig, ax = plt.subplots(figsize = (8,8))

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
        LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
        LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

        map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        agricultural_plot_matrix_attacked = np.zeros_like(agricultural_plot_matrix)

        for plot, count in agricultural_plots_attacked.items():
            agricultural_plot_matrix_attacked[agricultural_plot_matrix == plot] = count

        cax = map.imshow(agricultural_plot_matrix_attacked, cmap='hot', interpolation='nearest',origin='upper', vmin=0, vmax=10)
        fig.colorbar(cax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(os.path.join(output_folder, "agricultural_plots_attacked_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()

        agricultural_plts = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").ReadAsArray()
        food_matrix = gdal.Open(os.path.join(simulation_folders[0], "env", "food_matrix_0.1_1.0_.tif")).ReadAsArray()

        total_crop_raid_loss = 0
        
        for plot, count in agricultural_plots_attacked.items():
            food_val = np.sum(food_matrix[agricultural_plts == plot])
            total_crop_raid_loss += food_val * count
            
        print(f"\nTotal crop raid loss across the landscape: {total_crop_raid_loss} (kg)\n")

        return total_crop_raid_loss
    
    total_crop_raid_loss = find_penalties_based_on_intercepted_trajectories(output_folder)
    
    return total_crop_raid_loss

def create_defender_coverage_matrix():

    potential_coverage_matrix = gdal.Open("guarding-policies/abm-runs-no-memory/defender_coverage_matrix.tif").ReadAsArray()
    
    coverage_matrix = np.zeros_like(potential_coverage_matrix)

    #-----------plot coverage matrix#-----------#
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = mcolors.ListedColormap(['white', 'red'])
    im = ax.imshow(coverage_matrix, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Protected'),
        Patch(facecolor='white', edgecolor='black', label='Unprotected')
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "coverage_matrix_init/coverage_matrix.png"), dpi=300, bbox_inches="tight")
    #-----------plot coverage matrix#-----------#

    return coverage_matrix

def plot_and_save_defender_coverage(coverage_matrix, output_folder, figsize=(8, 8), 
                          protected_color='red', unprotected_color='white'):




    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = mcolors.ListedColormap([unprotected_color, protected_color])
    
    im = ax.imshow(coverage_matrix, cmap=cmap, vmin=0, vmax=1)
    
    ax.set_xticks([])
    ax.set_yticks([])

    legend_elements = [
        Patch(facecolor=protected_color, edgecolor='black', label='Protected'),
        Patch(facecolor=unprotected_color, edgecolor='black', label='Unprotected')
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.savefig(
        os.path.join(output_folder, "defender_coverage_matrix.png"),
        bbox_inches="tight",
        dpi=300,
    )





    source_file = gdal.Open("guarding-policies/abm-runs-no-memory/defender_coverage_matrix.tif")

    cols = source_file.RasterXSize
    rows = source_file.RasterYSize
    projection = source_file.GetProjection()
    geotransform = source_file.GetGeoTransform()

    output_file = os.path.join(output_folder, "defender_coverage_matrix.tif")

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(coverage_matrix.astype(np.uint8))

    source_file = None
    output_dataset = None

    return 

def select_cells_at_distance(target_lat, target_lon, distance_cells, distance_type='euclidean'):

    dataset = gdal.Open(coverage_matrix_path)
    geotransform = dataset.GetGeoTransform()
    original_array = dataset.ReadAsArray()

    original_array[275:350, 550:650] = 0
    
    x_origin = geotransform[0]  
    y_origin = geotransform[3] 
    pixel_width = geotransform[1]
    pixel_height = geotransform[5] 
    
    target_col = int((target_lon - x_origin) / pixel_width)
    target_row = int((target_lat - y_origin) / pixel_height)

    
    rows, cols = original_array.shape
    
    if not (0 <= target_row < rows and 0 <= target_col < cols):
        raise ValueError(f"Target coordinates are outside raster bounds. "
                        f"Pixel coordinates: ({target_row}, {target_col}), "
                        f"Raster shape: ({rows}, {cols})")
    
    row_indices, col_indices = np.ogrid[:rows, :cols]
    
    if distance_type == 'euclidean':
        distances = np.sqrt((row_indices - target_row)**2 + (col_indices - target_col)**2)
    elif distance_type == 'manhattan':
        distances = np.abs(row_indices - target_row) + np.abs(col_indices - target_col)
    elif distance_type == 'chebyshev':
        distances = np.maximum(np.abs(row_indices - target_row), np.abs(col_indices - target_col))
    else:
        raise ValueError("distance_type must be 'euclidean', 'manhattan', or 'chebyshev'")

    if isinstance(distance_cells, (int, float)):
        mask = (distances <= distance_cells)
    else:
        raise ValueError("distance_cells must be a number or a tuple of (min_distance, max_distance)")
    
    result_array = np.where(mask, original_array, 0)
    
    return result_array
    
def run_abm(model_params, experiment_name, output_folder):

    with open(os.path.join(output_folder, "model_parameters.yaml"), "w") as configfile:
        yaml.dump(model_params, configfile, default_flow_style=False)

    batch_run_model(model_params, experiment_name, output_folder)
         
    return 

def PLOT_CROP_DAMAGE(STEP_DAMAGES):

    regret_values = np.array(STEP_DAMAGES)
    steps = np.arange(1, len(regret_values) + 1)
    
    plt.figure(figsize=(6, 6))
    plt.plot(steps, regret_values, 'b-', label='FPL-UE')
    
    plt.xlabel('Step')
    plt.ylabel('crops raided')
    plt.title('Crop Raiding Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_FOLDER, 'coverage_matrix_init/crop_damage_with_step.png'), dpi=300, bbox_inches='tight')
    
    plt.close()

    return

def clip_raster_by_latlon_extent(input_file, output_folder, latlon_extent):

    try:
        source_ds = gdal.Open(input_file, gdal.GA_Update)
        if source_ds is None:
            print("Error: Could not open the input file.")
            return
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    
    geo_transform = source_ds.GetGeoTransform()
    x_size = source_ds.RasterXSize
    y_size = source_ds.RasterYSize

    band = source_ds.GetRasterBand(1)
    raster_data = band.ReadAsArray()

    lon_min, lat_min, lon_max, lat_max = latlon_extent

    x_res = geo_transform[1]
    y_res = geo_transform[5] 

    x_coords = np.arange(x_size) * x_res + geo_transform[0]
    y_coords = np.arange(y_size) * y_res + geo_transform[3]

    x_out_of_bounds = (x_coords < lon_min) | (x_coords > lon_max)
    y_out_of_bounds = (y_coords < lat_min) | (y_coords > lat_max)

    x_mask, y_mask = np.meshgrid(x_out_of_bounds, y_out_of_bounds)
    out_of_bounds_mask = x_mask | y_mask

    raster_data[out_of_bounds_mask] = 0

    cmap = plt.cm.get_cmap('tab20').copy()
    cmap.set_under('white')


    source_file = gdal.Open(input_file)
    cols = source_file.RasterXSize
    rows = source_file.RasterYSize
    projection = source_file.GetProjection()
    geotransform = source_file.GetGeoTransform()
    output_file = os.path.join(output_folder, "targets_to_protect_study_area.tif")
    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)
    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)
    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(raster_data.astype(np.uint8))
    source_file = None
    output_dataset = None
    source_ds = None


    # fig, ax = plt.subplots(figsize=(8, 8))

    # cax = ax.imshow(raster_data, cmap=cmap, vmin=0.1, 
    #                 extent=(geo_transform[0], geo_transform[0] + x_size * x_res, 
    #                         geo_transform[3] + y_size * y_res, geo_transform[3]))
    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')

    # plt.savefig(os.path.join(output_folder, "targets_to_protect.png"), dpi=600, bbox_inches='tight')

    return np.unique(raster_data)

def run_single_play(model_params, experiment_name, output_folder, MAX_GAME_STEPS):

    STEP_DAMAGES = []
    
    for i in range(1, MAX_GAME_STEPS+1):

        print("\n----- GameStep", i ,"-----")

        path = pathlib.Path(os.path.join(output_folder, "game_step_" + str(i)))
        path.mkdir(parents=True, exist_ok=True)

        path = pathlib.Path(os.path.join(OUTPUT_FOLDER, "coverage_matrix_init", "game_step_" + str(i)))
        path.mkdir(parents=True, exist_ok=True)
        
        coverage_matrix = create_defender_coverage_matrix()

        plot_and_save_defender_coverage(coverage_matrix, os.path.join(output_folder, "game_step_" + str(i)))
        plot_and_save_defender_coverage(coverage_matrix, os.path.join(OUTPUT_FOLDER, "coverage_matrix_init", "game_step_" + str(i)))
        
        run_abm(model_params, experiment_name, os.path.join(output_folder, "game_step_" + str(i)))

        make_trajectory_summary_plots_v3(os.path.join(output_folder, "game_step_" + str(i)), os.path.join(OUTPUT_FOLDER, "coverage_matrix_init", "game_step_" + str(i)))

        step_penalty = calculate_crop_raid_loss(output_folder=os.path.join(output_folder, "game_step_" + str(i)), 
                                                     current_game_step=i)
        
        STEP_DAMAGES.append(step_penalty)
        


    PLOT_CROP_DAMAGE(STEP_DAMAGES)

    return  
















if __name__ == "__main__":

    boundary_raster_discretised = "boundary_raster_discretised_1200m"

    global coverage_matrix_path   
    
    coverage_matrix_path = "guarding-policies/dynamic-guarding-model-v1/create-strategy-matrix-v2/" + boundary_raster_discretised + "/boundary_raster_discretised.tif"

    potential_coverage_matrix = select_cells_at_distance(
        target_lat=1049000,
        target_lon=8570800,
        distance_cells=145,
        distance_type='euclidean'
    )

    source_file = gdal.Open(coverage_matrix_path)

    cols = source_file.RasterXSize
    rows = source_file.RasterYSize
    projection = source_file.GetProjection()
    geotransform = source_file.GetGeoTransform()

    output_file = os.path.join("guarding-policies/abm-runs-no-memory/defender_coverage_matrix.tif")

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(potential_coverage_matrix.astype(np.uint8))

    source_file = None
    output_dataset = None
    
    
    potential_coverage_matrix = gdal.Open(output_file).ReadAsArray()
    
    total_num_targets = np.unique(potential_coverage_matrix)[-1]

    TARGETS  = np.unique(potential_coverage_matrix)

    TARGETS = [x for x in TARGETS if x != 0]

    print("Total number of targets to protect:", len(TARGETS), "\n", "TARGETS:", TARGETS)

    coverage_matrix_path = os.path.join(output_file)

    num_resources_k = 0
    MAX_GAME_STEPS = 12
    

    model_params = {
            "year": 2010,
            "month": "Mar",
            "num_bull_elephants": 1,
            "area_size": 1100,
            "spatial_resolution": 30,
            "max_food_val_cropland": 100,
            "max_food_val_forest": 25,
            "prob_food_forest": 0.10,
            "prob_food_cropland": 1.0,
            "prob_water_sources": 0.05,
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
            "num_processes": 12,
            "iterations": 12,
            "max_time_steps": 288 * 10,
            "aggression_threshold_enter_cropland": 1.0,
            "human_habituation_tolerance": 1.0,
            "elephant_agent_visibility_radius": 500,
            "plot_stepwise_target_selection": False,
            "threshold_days_of_food_deprivation": 0,
            "threshold_days_of_water_deprivation": 3,
            "number_of_feasible_movement_directions": 4,
            "track_in_mlflow": False,
            "elephant_starting_location": "user_input",
            "elephant_starting_latitude": [[1049000]],
            "elephant_starting_longitude": [[8570800]],
            "elephant_aggression_value": 0.8,
            "elephant_crop_habituation": True,
            "ranger_proximity_threshold": 0.999,
            "cost_ranger_proximity_threshold": 0,
            "num_protected_targets": num_resources_k,
            "boundary_raster_discretisation": boundary_raster_discretised
        }
    
    NUM_STRATEGIC_TRAJECTORIES = 12

    experiment_name = "mitigation-measures-within-plantations"

    elephant_category = "solitary_bulls"

    starting_location = (
        "latitude-"
        + str(model_params["elephant_starting_latitude"])
        + "-longitude-"
        + str(model_params["elephant_starting_longitude"])
    )

    landscape_food_probability = (
        "landscape-food-probability-forest-"
        + str(model_params["prob_food_forest"])
        + "-cropland-"
        + str(model_params["prob_food_cropland"])
    )

    food_availability_sceanario = "random-food-distribition-within-agricultural-plots"

    water_availability_sceanario = "water-source-rivers-landscape-" + str(model_params["prob_water_sources"])

    food_memory_matrix_type = "random-memory-forest-and_plantation-fringe-model"
    
    water_memory_matrix_type = "full-memory-forest-and_plantation-model"

    num_days_agent_survives_in_deprivation = (
        "num_days_agent_survives_in_deprivation-"
        + str(model_params["num_days_agent_survives_in_deprivation"])
    )

    maximum_food_in_a_forest_cell = "maximum-food-in-a-forest-cell-" + str(
        model_params["max_food_val_forest"]
    )

    elephant_thermoregulation_threshold = (
        "thermoregulation-threshold-temperature-"
        + str(model_params["thermoregulation_threshold"])
    )

    threshold_food_derivation_days = "threshold_days_of_food_deprivation-" + str(
        model_params["threshold_days_of_food_deprivation"]
    )

    threshold_water_derivation_days = "threshold_days_of_water_deprivation-" + str(
        model_params["threshold_days_of_water_deprivation"]
    )

    slope_tolerance = "slope_tolerance-" + str(model_params["slope_tolerance"])

    elephant_aggression_value = "elephant_aggression_value_" + str(
        model_params["elephant_aggression_value"]
    )

    target_folder = f"num_protected_targets_" + str(model_params["num_protected_targets"])

    simulation_repeats = f'num_strategic_traj_{NUM_STRATEGIC_TRAJECTORIES}_num_iterations_{model_params["iterations"]}'
    
    output_folder = os.path.join(
        "guarding-policies/abm-runs-no-memory/model-runs/game_model-v3_1-run-model-with-intervention-v2-no-knowledge/",
        experiment_name,
        starting_location,
        elephant_category,
        food_availability_sceanario,
        landscape_food_probability,
        water_availability_sceanario,
        food_memory_matrix_type,
        water_memory_matrix_type,
        num_days_agent_survives_in_deprivation,
        maximum_food_in_a_forest_cell,
        elephant_thermoregulation_threshold,
        threshold_food_derivation_days,
        threshold_water_derivation_days,
        slope_tolerance,
        num_days_agent_survives_in_deprivation,
        elephant_aggression_value,
        str(model_params["year"]),
        str(model_params["month"]),
        target_folder,
        simulation_repeats,
        boundary_raster_discretised,
        "agent-based-model-runs"
    )
    
    global OUTPUT_FOLDER
    
    OUTPUT_FOLDER = os.path.join(
        "guarding-policies/abm-runs-no-memory/model-runs/game_model-v3_1-run-model-with-intervention-v2-no-knowledge/",
        experiment_name,
        starting_location,
        elephant_category,
        food_availability_sceanario,
        landscape_food_probability,
        water_availability_sceanario,
        food_memory_matrix_type,
        water_memory_matrix_type,
        num_days_agent_survives_in_deprivation,
        maximum_food_in_a_forest_cell,
        elephant_thermoregulation_threshold,
        threshold_food_derivation_days,
        threshold_water_derivation_days,
        slope_tolerance,
        num_days_agent_survives_in_deprivation,
        elephant_aggression_value,
        str(model_params["year"]),
        str(model_params["month"]),
        target_folder,
        simulation_repeats,
        boundary_raster_discretised
    )
    
    run_single_play(
        model_params=model_params,
        experiment_name=experiment_name,
        output_folder=output_folder,
        MAX_GAME_STEPS = MAX_GAME_STEPS,
    )
