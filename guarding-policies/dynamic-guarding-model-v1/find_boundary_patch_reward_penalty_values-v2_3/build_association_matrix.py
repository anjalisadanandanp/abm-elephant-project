from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import os
import pandas as pd
from pyproj import Proj, transform    
from mpl_toolkits.basemap import Basemap    
import matplotlib.colors as mcolors   
from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore")


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
        "slope_tolerance": 35,
        "num_processes": 47,
        "iterations": 47,
        "max_time_steps": 288 * 30,
        "aggression_threshold_enter_cropland": 1.0,
        "human_habituation_tolerance": 1.0,
        "elephant_agent_visibility_radius": 500,
        "plot_stepwise_target_selection": False,
        "threshold_days_of_food_deprivation": 0,
        "threshold_days_of_water_deprivation": 3,
        "number_of_feasible_movement_directions": 4,
        "track_in_mlflow": False,
        "elephant_starting_location": "user_input",
        "elephant_starting_latitude": [[1052166]],
        "elephant_starting_longitude": [[8572829]],
        "elephant_aggression_value": 0.8,
        "elephant_crop_habituation": True,
        "ranger_proximity_threshold": None,
        "cost_ranger_proximity_threshold": None,
    }

NUM_STRATEGIC_TRAJECTORIES = 188

experiment_name = "mitigation-measures-within-plantations-FPL-UE_v3_1"

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

food_availability_sceanario = "random-food-distribition-within-agricultural-plots-and-other-plantation-cells"

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

targets_to_cover = [0]

target_folder = f"protected_targets_{'_'.join(map(str, targets_to_cover))}"

simulation_repeats = f'num_strategic_traj_{NUM_STRATEGIC_TRAJECTORIES}_num_iterations_{model_params["iterations"]}'

folder = os.path.join(
    "guarding-policies/static-guarding-model-v1/model-runs/exploratory_search_on_evading_trajectories/game_model-v3_1-run-model-without-intervention-v2/",
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
    "game_step_1"
)

print("--------processing folder--------:", folder)

landuse_matrix = gdal.Open(os.path.join("guarding-policies/static-guarding-model-v1/model-runs/exploratory_search_on_evading_trajectories/game_model-v3_1-run-model-without-intervention-v2/mitigation-measures-within-plantations-FPL-UE_v3_1/latitude-[[1052166]]-longitude-[[8572829]]/solitary_bulls/random-food-distribition-within-agricultural-plots-and-other-plantation-cells/landscape-food-probability-forest-0.1-cropland-1.0/water-source-rivers-landscape-0.05/random-memory-forest-and_plantation-fringe-model/full-memory-forest-and_plantation-model/num_days_agent_survives_in_deprivation-10/maximum-food-in-a-forest-cell-25/thermoregulation-threshold-temperature-28/threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/slope_tolerance-35/num_days_agent_survives_in_deprivation-10/elephant_aggression_value_0.8/2010/Mar/protected_targets_0/num_strategic_traj_188_num_iterations_47/game_step_1/0ae61fb2-7e21-479c-b272-88f066f7e68d/env/LULC.tif")).ReadAsArray()

food_matrix = gdal.Open(os.path.join("guarding-policies/static-guarding-model-v1/model-runs/exploratory_search_on_evading_trajectories/game_model-v3_1-run-model-without-intervention-v2/mitigation-measures-within-plantations-FPL-UE_v3_1/latitude-[[1052166]]-longitude-[[8572829]]/solitary_bulls/random-food-distribition-within-agricultural-plots-and-other-plantation-cells/landscape-food-probability-forest-0.1-cropland-1.0/water-source-rivers-landscape-0.05/random-memory-forest-and_plantation-fringe-model/full-memory-forest-and_plantation-model/num_days_agent_survives_in_deprivation-10/maximum-food-in-a-forest-cell-25/thermoregulation-threshold-temperature-28/threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/slope_tolerance-35/num_days_agent_survives_in_deprivation-10/elephant_aggression_value_0.8/2010/Mar/protected_targets_0/num_strategic_traj_188_num_iterations_47/game_step_1/0ae61fb2-7e21-479c-b272-88f066f7e68d/env/food_matrix_0.1_1.0_.tif")).ReadAsArray()

for max_cells_per_group in tqdm([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]):

    output_dir='guarding-policies/dynamic-guarding-model-v1/create-strategy-matrix-v2/boundary_raster_discretised_' + str(int(max_cells_per_group*30)) + 'm'

    save_folder = os.path.join("guarding-policies/dynamic-guarding-model-v1/find_boundary_patch_reward_penalty_values-v2_3/boundary_raster_discretised_" + str(int(max_cells_per_group*30)) + "m")

    os.makedirs(save_folder, exist_ok=True)

    agricultural_plts = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").ReadAsArray()

    boundary_patches = gdal.Open(os.path.join(output_dir, "boundary_raster_discretised.tif")).ReadAsArray()

    run_folder = os.path.join(folder)

    rainbow = plt.cm.rainbow
    colors = rainbow(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]  
    custom_cmap = matplotlib.colors.ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(boundary_patches, cmap=custom_cmap)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.colorbar(img, shrink=0.5)

    plt.savefig(os.path.join(save_folder, "boundary_patches.png"), dpi=750, bbox_inches="tight")

    plt.close()

    expts = os.listdir(run_folder)

    expts = [item for item in expts if os.path.isdir(os.path.join(run_folder, item))]

    geotransform = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").GetGeoTransform()
    ag_xmin, ag_xres, ag_xskew, ag_ymax, ag_yskew, ag_yres = geotransform
    ag_rows, ag_cols = agricultural_plts.shape

    row_size, col_size = landuse_matrix.shape
    xmin, xres, xskew, ymax, yskew, yres = gdal.Open(os.path.join("guarding-policies/static-guarding-model-v1/model-runs/exploratory_search_on_evading_trajectories/game_model-v3_1-run-model-without-intervention-v2/mitigation-measures-within-plantations-FPL-UE_v3_1/latitude-[[1052166]]-longitude-[[8572829]]/solitary_bulls/random-food-distribition-within-agricultural-plots-and-other-plantation-cells/landscape-food-probability-forest-0.1-cropland-1.0/water-source-rivers-landscape-0.05/random-memory-forest-and_plantation-fringe-model/full-memory-forest-and_plantation-model/num_days_agent_survives_in_deprivation-10/maximum-food-in-a-forest-cell-25/thermoregulation-threshold-temperature-28/threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/slope_tolerance-35/num_days_agent_survives_in_deprivation-10/elephant_aggression_value_0.8/2010/Mar/protected_targets_0/num_strategic_traj_188_num_iterations_47/game_step_1/0ae61fb2-7e21-479c-b272-88f066f7e68d/env/LULC.tif")).GetGeoTransform()
    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
    LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
    LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

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

    plot_landuse_map = True
    plot_agricultural_plot_map = True
    plot_boundary_patch_map = True
    num_cropraiding_steps = 6

    num_boundary_patchs = int(np.max(boundary_patches))
    num_agricultural_plts = int(np.max(agricultural_plts))

    association_matrix_num_visiting_trajs = np.zeros((num_boundary_patchs, 1))

    for ids, expt in enumerate(expts):

        if plot_landuse_map == True and ids < 5:

            fig, ax = plt.subplots(figsize=(8, 8))

            map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

            img = map.imshow(np.flipud(landuse_matrix), cmap = "Pastel2", extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 1)

            map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
            map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

            cb = fig.colorbar(img) 
            cb.remove()

            df = pd.read_csv(os.path.join(run_folder, expt, "output_files/agent_data.csv"))

            rows, cols = lat_lon_to_pixel(
                df["latitude"].values, df["longitude"].values, 
                ag_xmin, ag_ymax, ag_xres, ag_yres
            )
            
            landuse_values = landuse_matrix[rows, cols]

            indices = find_cropland_use_indices(landuse_values)

            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
            longitude, latitude = transform(inProj, outProj, df["longitude"], df["latitude"])
            x_new, y_new = map(longitude,latitude)

            ax.plot(x_new, y_new, linewidth=0.25, alpha=0.5, color="black", zorder=1)

            ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='red', zorder=1)
            ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='red', zorder=1)

            for sequence in indices:

                if (sequence[1] - sequence[0]) >= num_cropraiding_steps:
                    
                    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
                    longitude, latitude = transform(inProj, outProj, df["longitude"][sequence[0]:sequence[1]], df["latitude"][sequence[0]:sequence[1]])
                    x_new, y_new = map(longitude,latitude)

                    ax.plot(x_new, y_new, linewidth=0.25, zorder=2)
                    
                    ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='black', zorder=2)
                    ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='black', zorder=2)

            plt.savefig(os.path.join(save_folder, "cropland_intersection_traj_" + expt + "_.png"), dpi=500, bbox_inches="tight")
            plt.close()

        if plot_agricultural_plot_map == True and ids < 5:

            fig, ax = plt.subplots(figsize=(8, 8))

            map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

            colors = ['white', 'cyan']
            cmap_two_colors = mcolors.ListedColormap(colors)

            img = map.imshow(np.flipud(agricultural_plts), cmap = cmap_two_colors, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 1)

            map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
            map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

            cb = fig.colorbar(img) 
            cb.remove()

            df = pd.read_csv(os.path.join(run_folder, expt, "output_files/agent_data.csv"))

            rows, cols = lat_lon_to_pixel(
                df["latitude"].values, df["longitude"].values, 
                ag_xmin, ag_ymax, ag_xres, ag_yres
            )
            
            landuse_values = landuse_matrix[rows, cols]

            indices = find_cropland_use_indices(landuse_values)

            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
            longitude, latitude = transform(inProj, outProj, df["longitude"], df["latitude"])
            x_new, y_new = map(longitude,latitude)

            ax.plot(x_new, y_new, linewidth=0.25, alpha=0.5, color="black", zorder=1)

            ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='red', zorder=1)
            ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='red', zorder=1)

            for sequence in indices:

                if (sequence[1] - sequence[0]) >= num_cropraiding_steps:
                    
                    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
                    longitude, latitude = transform(inProj, outProj, df["longitude"][sequence[0]:sequence[1]], df["latitude"][sequence[0]:sequence[1]])
                    x_new, y_new = map(longitude,latitude)

                    ax.plot(x_new, y_new, linewidth=0.25, zorder=2)
                    
                    ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='black', zorder=2)
                    ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='black', zorder=2)

            plt.savefig(os.path.join(save_folder, "agri_plots_intersection_traj_" + expt + "_.png"), dpi=500, bbox_inches="tight")
            plt.close()

        if plot_boundary_patch_map == True and ids < 5:

            fig, ax = plt.subplots(figsize=(8, 8))

            map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

            rainbow = plt.cm.rainbow
            colors = rainbow(np.linspace(0, 1, 256))
            colors[0] = [1, 1, 1, 1]  
            custom_cmap = matplotlib.colors.ListedColormap(colors)

            img = map.imshow(np.flipud(boundary_patches), cmap = custom_cmap, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 1)

            map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
            map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

            cb = fig.colorbar(img) 
            cb.remove()

            df = pd.read_csv(os.path.join(run_folder, expt, "output_files/agent_data.csv"))

            rows, cols = lat_lon_to_pixel(
                df["latitude"].values, df["longitude"].values, 
                ag_xmin, ag_ymax, ag_xres, ag_yres
            )
            
            landuse_values = landuse_matrix[rows, cols]

            indices = find_cropland_use_indices(landuse_values)

            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
            longitude, latitude = transform(inProj, outProj, df["longitude"], df["latitude"])
            x_new, y_new = map(longitude,latitude)

            ax.plot(x_new, y_new, linewidth=0.25, alpha=0.5, color="black", zorder=1)

            ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='red', zorder=1)
            ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='red', zorder=1)

            for sequence in indices:

                if (sequence[1] - sequence[0]) >= num_cropraiding_steps:
                    
                    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
                    longitude, latitude = transform(inProj, outProj, df["longitude"][sequence[0]:sequence[1]], df["latitude"][sequence[0]:sequence[1]])
                    x_new, y_new = map(longitude,latitude)

                    ax.plot(x_new, y_new, linewidth=0.25, zorder=2)
                    
                    ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='black', zorder=2)
                    ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='black', zorder=2)

            plt.savefig(os.path.join(save_folder, "boundary_patch_intersection_traj_" + expt + "_.png"), dpi=500, bbox_inches="tight")
            plt.close()


        try:

            association_matrix_num_visiting_trajs_local = np.zeros((num_boundary_patchs, 1))

            df = pd.read_csv(os.path.join(run_folder, expt, "output_files/agent_data.csv"))

            rows, cols = lat_lon_to_pixel(
                df["latitude"].values, df["longitude"].values, 
                ag_xmin, ag_ymax, ag_xres, ag_yres
            )
            
            landuse_values = landuse_matrix[rows, cols]

            indices = find_cropland_use_indices(landuse_values)
            
            for sequence in indices:

                if (sequence[1] - sequence[0]) >= num_cropraiding_steps:

                    if boundary_patches[rows[sequence[0]], cols[sequence[0]]] != 0:

                        for i in range(sequence[1] - sequence[0]):
                            
                            association_matrix_num_visiting_trajs_local[int(boundary_patches[rows[sequence[0]], cols[sequence[0]]]), 0] = 1
                             
        except Exception as e:
            print(f"Error processing trajectory in {expt}: {e}")
            pass

        association_matrix_num_visiting_trajs += association_matrix_num_visiting_trajs_local

    df = pd.DataFrame(association_matrix_num_visiting_trajs, columns=["num_visiting_trajectories"], index=range(0, num_boundary_patchs))
    df.to_csv(os.path.join(save_folder, "association_matrix_num_visiting_trajs.csv"), index_label="boundary_patch_id")
