import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import pathlib
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


fontsize = 8
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

def update_targets_df(output_folder, targets_df, current_game_step, MAX_STEP_CROP_RAIDING_VAL, MAX_STEP_TRAJECTORIES_ENCOUNTERED):

    def find_rewards_based_on_intercepted_trajectories(output_folder):

        dict_of_attacked_targets = {}
        
        simulation_repeats = [
        item for item in os.listdir(output_folder) 
        if os.path.isdir(os.path.join(output_folder, item))
        ]


        for simulation_folder in simulation_repeats:

            try:

                df = pd.read_csv(os.path.join(output_folder, simulation_folder, "output_files/agent_data.csv"))
                df.dropna(subset=['ROW', 'COL'], inplace=True)

                unique_targets = df["target_attacked"].dropna().unique()

                for target in unique_targets:
                    if target not in dict_of_attacked_targets:
                        dict_of_attacked_targets[target] = 0
                    dict_of_attacked_targets[target] += 1
            
            except Exception as e:
                pass

        # for target, count in dict_of_attacked_targets.items():
        #     print(f"Covered target {target} was attacked {count} times.")  

        df_dict_of_attacked_targets = pd.DataFrame.from_dict(dict_of_attacked_targets, orient='index', columns=['count'])

        df_dict_of_attacked_targets.reset_index(inplace=True)
        df_dict_of_attacked_targets.rename(columns={'index': 'target'}, inplace=True)
                                                
        df_dict_of_attacked_targets.to_csv(os.path.join(output_folder, "df_of_attacked_targets.csv"))

        return df_dict_of_attacked_targets

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

        boundary_patch_matrix = gdal.Open(os.path.join(OUTPUT_FOLDER, "coverage_matrix_init/potential_coverage_matrix.tif")).ReadAsArray()
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
            
        # print(f"\nTotal crop raid loss across the landscape: {total_crop_raid_loss} (kg)\n")

        return total_crop_raid_loss
    
    
    targets_df["reward"] = 0
    targets_df["penalty"] = 0
    
    
    df_dict_of_attacked_targets = find_rewards_based_on_intercepted_trajectories(output_folder)
    
    # print("trajectories successfully intercepted by guards (reward):", df_dict_of_attacked_targets)

    total_crop_raid_loss = find_penalties_based_on_intercepted_trajectories(output_folder)
    
    print("total crop raid loss from non-intercepted trajectories (penalty):", total_crop_raid_loss)
    
    if total_crop_raid_loss > MAX_STEP_CROP_RAIDING_VAL:
        MAX_STEP_CROP_RAIDING_VAL = total_crop_raid_loss

    for boundary_patch in df_dict_of_attacked_targets["target"].values:
        
        if df_dict_of_attacked_targets.loc[df_dict_of_attacked_targets["target"] == boundary_patch, "count"].values[0] > MAX_STEP_TRAJECTORIES_ENCOUNTERED:
            MAX_STEP_TRAJECTORIES_ENCOUNTERED = df_dict_of_attacked_targets.loc[df_dict_of_attacked_targets["target"] == boundary_patch, "count"].values[0]
    
    if MAX_STEP_TRAJECTORIES_ENCOUNTERED > 0:
        reward_normalizer = 1.0 / MAX_STEP_TRAJECTORIES_ENCOUNTERED
    else:
        reward_normalizer = 0.0

    for boundary_patch in df_dict_of_attacked_targets["target"].values:
        count = df_dict_of_attacked_targets.loc[df_dict_of_attacked_targets["target"] == boundary_patch, "count"].values[0]
        reward = (count * reward_normalizer) * 0.5
        targets_df.loc[targets_df["boundary_patch_id"] == boundary_patch, "reward"] = reward
        # print("boundary_patch:", boundary_patch, "normalized reward:", reward)

    potential_targets = np.unique(gdal.Open(os.path.join(OUTPUT_FOLDER, "coverage_matrix_init/potential_coverage_matrix.tif")).ReadAsArray())

    if MAX_STEP_CROP_RAIDING_VAL > 0:
        penalty_normalizer = 1.0 / MAX_STEP_CROP_RAIDING_VAL
    
    else:
        penalty_normalizer = 0.0

    for boundary_patch in potential_targets:
        if boundary_patch not in df_dict_of_attacked_targets["target"].values and boundary_patch != 0:

            penalty = -(total_crop_raid_loss * penalty_normalizer) * 0.5

            targets_df.loc[targets_df["boundary_patch_id"] == boundary_patch, "penalty"] = penalty
            # print("boundary_patch:", boundary_patch, "normalized penalty:", penalty)

    targets_df.to_csv(os.path.join(OUTPUT_FOLDER, "coverage_matrix_init", "game_step_" + str(int(current_game_step + 1)), "boundary_patch_reward_penalty_matrix.csv"), index=False)

    return targets_df, total_crop_raid_loss, MAX_STEP_CROP_RAIDING_VAL, MAX_STEP_TRAJECTORIES_ENCOUNTERED

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

def make_loss_df(output_folder, MAX_GAME_STEPS, NUM_LANDSCAPE_CELLS, BUDGET_K):

    coveragepath = pathlib.Path(os.path.join(OUTPUT_FOLDER, "coverage_matrix_init", "game_step_" + str(1)))
    targets_df = pd.read_csv(os.path.join(coveragepath, "boundary_patch_reward_penalty_matrix.csv"))

    STEP_DAMAGES = []
    
    MAX_STEP_CROP_RAIDING_VAL = -9999
    MAX_STEP_TRAJECTORIES_ENCOUNTERED = -9999
    
    for i in range(1, MAX_GAME_STEPS+1):
        
        print("\n----- GameStep", i ,"-----")
        
        targets_df, step_penalty, MAX_STEP_CROP_RAIDING_VAL, MAX_STEP_TRAJECTORIES_ENCOUNTERED = update_targets_df(output_folder=os.path.join(output_folder, "game_step_" + str(i)), 
                                                     targets_df=targets_df, 
                                                     current_game_step=i,
                                                     MAX_STEP_CROP_RAIDING_VAL = MAX_STEP_CROP_RAIDING_VAL,
                                                     MAX_STEP_TRAJECTORIES_ENCOUNTERED = MAX_STEP_TRAJECTORIES_ENCOUNTERED)
        
        STEP_DAMAGES.append(step_penalty)

    PLOT_CROP_DAMAGE(STEP_DAMAGES)

    game_steps = list(range(1, len(STEP_DAMAGES) + 1))

    if len(game_steps) != len(STEP_DAMAGES):
        print("Error: The number of steps does not match the number of regret values.")
    else:
        data = {
            'GameStep': game_steps,
            'croploss': STEP_DAMAGES
        }

        df = pd.DataFrame(data)

        output_filepath = os.path.join(OUTPUT_FOLDER, 'coverage_matrix_init/loss_df_FPL-UE_no-memory-model-v1_2.csv')
        df.to_csv(output_filepath, index=False)
        
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

    output_file = os.path.join("guarding-policies/FPL-UE_no-memory-model-v1_2/defender_coverage_matrix.tif")

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
    
    print("length of strategy matrix:", total_num_targets)

    TARGETS  = np.unique(potential_coverage_matrix)

    TARGETS = [x for x in TARGETS if x != 0]

    print("Total number of targets to protect:", len(TARGETS), "\n", "TARGETS:", TARGETS)

    coverage_matrix_path = os.path.join(output_file)

    proximity_filter_parameter = [0.999]
    cost_function_threshold_parameter = [0]
    
    parameter_combinations = list(itertools.product(proximity_filter_parameter, cost_function_threshold_parameter))

    num_resources_k = [7]
    
    for ranger_proximity_threshold, cost_ranger_proximity_threshold in parameter_combinations:

        for k in num_resources_k: 

            BUDGET_K = k                   # Maximum number of cells that can be protected by the defenders at every time-step
            MAX_GAME_STEPS = 50                         # Maximum number of time-steps in the game
            eta = 10                                   # reward perturbation parameter
            M = 12                                      # parameter in the GR algorithm

            FPL_UE_params = (
                "budget_k_"
                + str(BUDGET_K)
                + "-max_game_steps_"
                + str(MAX_GAME_STEPS) 
                + "-eta_"
                + str(eta)
                + "-M_"
                + str(M)
            )
    

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
                    "ranger_proximity_threshold": None,
                    "cost_ranger_proximity_threshold": None,
                    "num_protected_targets": k
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
                "guarding-policies/FPL-UE_no-memory-model-v1_2/model-runs/game_model-v3_1-run-model-with-intervention-v2-no-knowledge/",
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
                FPL_UE_params,
                "agent-based-model-runs"
            )
            
            global OUTPUT_FOLDER
            
            OUTPUT_FOLDER = os.path.join(
                "guarding-policies/FPL-UE_no-memory-model-v1_2/model-runs/game_model-v3_1-run-model-with-intervention-v2-no-knowledge/",
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
                FPL_UE_params
            )
            
            make_loss_df(
                output_folder=output_folder,
                NUM_LANDSCAPE_CELLS = total_num_targets,
                BUDGET_K = BUDGET_K,
                MAX_GAME_STEPS = MAX_GAME_STEPS
            )
