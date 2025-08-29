import os                               # for file operations
import numpy as np                      # for numerical operations
from osgeo import gdal                  # for raster operations
from pyproj import Proj, transform              # for coordinate transformations
import matplotlib.pyplot as plt                 # for plotting
from mpl_toolkits.basemap import Basemap        # for plotting
from matplotlib import colors                   # for plotting
import matplotlib.cm as cm                      # for plotting
import matplotlib.colors as mcolors             # for plotting
import pandas as pd                            # for data handling
from tqdm import tqdm    

import warnings
warnings.filterwarnings("ignore")

base_folder = os.path.join("/mnt/qdata/abm-elephant-project/aryabhata-runs/model-with-deterrance-policy")

subfolder_path_first = "latitude-1049237-longitude-8570917/solitary_bulls"

food_distribution_models = [
    "random-food-distribition-within-plantation-cells"
]

subfolder_path_second = "landscape-food-probability-forest-0.1-cropland-0.1/water-source-rivers-landscape-1.0/random-memory-forest-and_plantation-fringe-model/full-memory-forest-and_plantation-model/num_days_agent_survives_in_deprivation-10/"


food_value_models = [
    "maximum-food-in-a-forest-cell-5",
    "maximum-food-in-a-forest-cell-10",
    "maximum-food-in-a-forest-cell-15",
    "maximum-food-in-a-forest-cell-20",
    "maximum-food-in-a-forest-cell-25"]


thermoregulation_models = [
    "thermoregulation-threshold-temperature-28",
    "thermoregulation-threshold-temperature-32"
]

subfolder_path_third = "threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/"

slope_tolerance_models = [
    "slope_tolerance-35",
]

subfolder_path_fourth = "num_days_agent_survives_in_deprivation-10"

aggression_models = ["elephant_aggression_value_0.2",
                     "elephant_aggression_value_0.8"]

subfolder_path_fifth = "2010"

months = ["Mar", "Aug"]

deterrant_matrix_configurations = [
    "deterrant_matrix_configuration-random-coverage-1-suitability_threshold-0.5",
    "deterrant_matrix_configuration-random-coverage-5-suitability_threshold-0.5",
    "deterrant_matrix_configuration-random-coverage-10-suitability_threshold-0.5",
    "deterrant_matrix_configuration-random-coverage-25-suitability_threshold-0.5",
    "deterrant_matrix_configuration-random-coverage-50-suitability_threshold-0.5",
    "deterrant_matrix_configuration-random-coverage-100-suitability_threshold-0.5"]
    

for food_distribution_model in food_distribution_models:
    for food_value_model in food_value_models:
        for thermoregulation_model in thermoregulation_models:
            for slope_tolerance_model in slope_tolerance_models:
                for aggression_model in aggression_models:
                    for month in months:
                        for deterrant_matrix_configuration in deterrant_matrix_configurations:
                            folder_path = os.path.join(base_folder, 
                                                       subfolder_path_first,
                                                    food_distribution_model,
                                                    subfolder_path_second,
                                                    food_value_model,
                                                    thermoregulation_model,
                                                    subfolder_path_third,
                                                    slope_tolerance_model,
                                                    subfolder_path_fourth,
                                                    aggression_model,
                                                    subfolder_path_fifth,
                                                    month,
                                                    deterrant_matrix_configuration,
                                                    "forest_fringe_buffer_for_deterrant_matrix-17",
                                                    "w_border_0.0-w_roads_0.0-w_plantation_0.0-w_dem_0.0-w_slope_[0.0]"
                                                    )
                            
                            output_folder = os.path.join("results_and_analysis/deterrance_policy_models/model_with_intervention",
                                                        subfolder_path_first,
                                                        food_distribution_model,
                                                        subfolder_path_second,
                                                        food_value_model,
                                                        thermoregulation_model,
                                                        subfolder_path_third,
                                                        slope_tolerance_model,
                                                        subfolder_path_fourth,
                                                        aggression_model,
                                                        subfolder_path_fifth,
                                                        month,
                                                        deterrant_matrix_configuration,)
                            
                            os.makedirs(output_folder, exist_ok=True)

                            runs = os.listdir(folder_path)

                            ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
                            data = ds.ReadAsArray()
                            data = np.flip(data, axis=0)
                            row_size, col_size = data.shape
                            xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

                            data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

                            for i in range(1,10):
                                data[data == data_value_map[i]] = i

                            fig, ax = plt.subplots(figsize = (6,6))
                            ax.yaxis.set_inverted(True)

                            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
                            LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
                            LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

                            map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

                            levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
                            clrs = ["greenyellow","mediumpurple","turquoise", "plum", "black", "blue", "yellow", "mediumseagreen", "forestgreen"] 
                            cmap, norm = colors.from_levels_and_colors(levels, clrs)

                            map.imshow(data, cmap = cmap, norm=norm, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 0.5, zorder=1)

                            map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
                            map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

                            cbar = plt.colorbar(ticks=[1,2,3,4,5,6,7,8,9],fraction=0.046, pad=0.04)
                            cbar.ax.set_yticks(ticks=[1,2,3,4,5,6,7,8,9]) 
                            cbar.ax.set_yticklabels(["Deciduous Broadleaf Forest","Built-up Land","Mixed Forest","Shrubland","Barren Land","Water Bodies","Plantations","Grassland","Broadleaf evergreen forest"])

                            for run in tqdm(runs):

                                try:
                                    df = pd.read_csv(os.path.join(folder_path, run, "output_files", "agent_data.csv"))
                                    longitude = df["longitude"].values
                                    latitude = df["latitude"].values
                                    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
                                    longitude, latitude = transform(inProj, outProj, longitude, latitude)
                                    x_new, y_new = map(longitude,latitude)
                                    C = np.arange(len(x_new))
                                    nz = mcolors.Normalize()
                                    nz.autoscale(C)

                                    ax.quiver(x_new[:-1], y_new[:-1], 
                                                x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1], 
                                                scale_units='xy', angles='xy', 
                                                scale=1, zorder=2, color = cm.jet(nz(C)), 
                                                width=0.0015, headwidth=3, headlength=4, headaxislength=3.5, alpha = 0.75)

                                    ax.scatter(x_new[0], y_new[0], 25, marker='o', color='blue', zorder=3) 
                                    ax.scatter(x_new[-1], y_new[-1], 25, marker='^', color='red', zorder=3) 
                                except:
                                    continue

                            plt.savefig(os.path.join(output_folder, "trajectory_on_LULC_v1.png"), dpi = 300, bbox_inches = 'tight')

                            plt.close()