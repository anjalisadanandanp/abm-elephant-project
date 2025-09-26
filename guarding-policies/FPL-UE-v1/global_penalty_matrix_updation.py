import os
import pandas as pd
from osgeo import gdal
import numpy as np
from pyproj import Proj, transform  
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap   

import warnings
warnings.filterwarnings("ignore")

run_folder = "guarding-policies/FPL-UE-v1/game_model-v3_1-run-model-with-intervention-v2-no-knowledge/mitigation-measures-within-plantations/latitude-[[1052166]]-longitude-[[8572829]]/solitary_bulls/random-food-distribition-within-agricultural-plots/landscape-food-probability-forest-0.1-cropland-1.0/water-source-rivers-landscape-0.05/random-memory-forest-and_plantation-fringe-model/full-memory-forest-and_plantation-model/num_days_agent_survives_in_deprivation-10/maximum-food-in-a-forest-cell-25/thermoregulation-threshold-temperature-28/threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/slope_tolerance-30/num_days_agent_survives_in_deprivation-10/elephant_aggression_value_0.8/2010/Mar/num_protected_targets_5/num_strategic_traj_12_num_iterations_12/boundary_raster_discretised_750m/budget_k_5-max_game_steps_10-max_gamma_1.0-min_gamma_0.85-num_steps_gamma_decay_10-eta_10.0-M_30/"

game_steps = os.listdir(run_folder)
game_steps = [folder for folder in game_steps if "game_step" in folder]
game_steps = sorted(game_steps, key=lambda x: int(x.split("_")[-1]))

for game_step in game_steps:
    if os.path.isdir(os.path.join(run_folder, game_step)):

        if game_step == "game_step_1":
            penalty_df_global = pd.read_csv(os.path.join(run_folder, game_step, "association_df_updated.csv"), index_col=0)
         
        else:
            penalty_df_global_step = pd.read_csv(os.path.join(run_folder, game_step, "association_df_updated.csv"), index_col=0)
            penalty_df_global += penalty_df_global_step     # type: ignore

    penalty_df_global.to_csv(os.path.join(run_folder, "association_df_global_updated.csv"))

    ds = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif")
    data = ds.ReadAsArray()
    row_size, col_size = data.shape
    xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

    fig, ax = plt.subplots(figsize = (10,10))

    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
    LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
    LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

    map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

    agricultural_plot_matrix_attacked = np.zeros_like(data)

    for row in penalty_df_global.index:

        cols = penalty_df_global.columns[penalty_df_global.loc[row] != 0]
        values = penalty_df_global.loc[row, cols].values

        print(f"Protecting Patch ID: {row}, Agricultural plots Attacked: {list(cols)} with counts {list(values)}")

        for col, value in zip(cols, values):

            print(f"Agricultural Plot {col} attacked {value} times.")

            mask = data == int(col)
            agricultural_plot_matrix_attacked[mask] = value

    cax = map.imshow(agricultural_plot_matrix_attacked, cmap='hot', origin='upper')
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(run_folder, game_step, "agricultural_plots_attacked_heatmap_accumulated.png"), dpi=300, bbox_inches='tight')
    plt.close()
                    
