import os
import pandas as pd
from osgeo import gdal
import numpy as np
from pyproj import Proj, transform  
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap   

import warnings
warnings.filterwarnings("ignore")

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

    boundary_patch_matrix = gdal.Open(os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init/potential_coverage_matrix.tif")).ReadAsArray()
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

    simulation_repeats = [
        os.path.join(output_folder, item)
        for item in simulation_repeats
        if os.path.isdir(os.path.join(output_folder, item))
    ]

    agricultural_plots_attacked = {}

    for simulation_repeat in simulation_repeats:

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


    print("\n")

    covered_targets_matrix = gdal.Open(os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init/game_step_1/defender_coverage_matrix.tif")).ReadAsArray()
    covered_targets = np.unique(covered_targets_matrix)
    covered_targets = covered_targets[covered_targets != 0]


    print("covered boundaries:", covered_targets)


    for plot, count in agricultural_plots_attacked.items():
        print(f"Agricultural plot {plot} was attacked {count} times.")

        association_df.loc[covered_targets, plot] += count

    association_df.to_csv(os.path.join(output_folder, "association_df_updated.csv"))

    fig, ax = plt.subplots(figsize=(10, 10))

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

    cax = map.imshow(agricultural_plot_matrix_attacked, cmap='hot', interpolation='nearest',origin='upper')
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(output_folder, "agricultural_plots_attacked_heatmap.png"))
    plt.close()
                


run_folder = "/home2/anjali/GitHub/abm-elephant-project/guarding-policies/FPL-UE-v1/game_model-v3_1-run-model-with-intervention-v2-no-knowledge/mitigation-measures-within-plantations/latitude-[[1052166]]-longitude-[[8572829]]/solitary_bulls/random-food-distribition-within-agricultural-plots/landscape-food-probability-forest-0.1-cropland-1.0/water-source-rivers-landscape-0.05/random-memory-forest-and_plantation-fringe-model/full-memory-forest-and_plantation-model/num_days_agent_survives_in_deprivation-10/maximum-food-in-a-forest-cell-25/thermoregulation-threshold-temperature-28/threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/slope_tolerance-30/num_days_agent_survives_in_deprivation-10/elephant_aggression_value_0.8/2010/Mar/num_protected_targets_5/num_strategic_traj_12_num_iterations_12/boundary_raster_discretised_750m/budget_k_5-max_game_steps_10-max_gamma_1.0-min_gamma_0.85-num_steps_gamma_decay_10-eta_10.0-M_30/"

game_steps = os.listdir(run_folder)

for game_step in game_steps:
    print(game_step)
    if os.path.isdir(os.path.join(run_folder, game_step)):
        find_penalties_based_on_intercepted_trajectories(os.path.join(run_folder, game_step))
