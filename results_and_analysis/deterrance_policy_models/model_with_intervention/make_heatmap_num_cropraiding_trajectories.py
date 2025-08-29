import os
from osgeo import gdal
import pandas as pd
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyproj import Proj, transform              
from mpl_toolkits.basemap import Basemap      
from matplotlib import colors                 
import matplotlib.cm as cm                     
import matplotlib.colors as mcolors       
warnings.filterwarnings("ignore")

base_folder = os.path.join("/mnt/qdata/abm-elephant-project/aryabhata-runs/model-with-deterrance-policy")

subfolder_path_first = "latitude-1049237-longitude-8570917/solitary_bulls"

food_distribution_models = ["random-food-distribition-within-plantation-cells"]

subfolder_path_second = "landscape-food-probability-forest-0.1-cropland-0.1/water-source-rivers-landscape-1.0/random-memory-forest-and_plantation-fringe-model/full-memory-forest-and_plantation-model/num_days_agent_survives_in_deprivation-10/"

food_value_models = [
    "maximum-food-in-a-forest-cell-5",
    "maximum-food-in-a-forest-cell-10",
    "maximum-food-in-a-forest-cell-15",
    "maximum-food-in-a-forest-cell-20",
    "maximum-food-in-a-forest-cell-25"
]

thermoregulation_models = [
    "thermoregulation-threshold-temperature-28",
    "thermoregulation-threshold-temperature-32"
]

subfolder_path_third = "threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/"

slope_tolerance_models = ["slope_tolerance-35"]

subfolder_path_fourth = "num_days_agent_survives_in_deprivation-10"

aggression_models = [
    "elephant_aggression_value_0.2",
    "elephant_aggression_value_0.8"
]

subfolder_path_fifth = "2010"

months = ["Mar", "Aug"]

deterrant_matrix_configurations = [
    "deterrant_matrix_configuration-random-coverage-1-suitability_threshold-0.5",
    "deterrant_matrix_configuration-random-coverage-5-suitability_threshold-0.5",
    "deterrant_matrix_configuration-random-coverage-10-suitability_threshold-0.5",
    "deterrant_matrix_configuration-random-coverage-25-suitability_threshold-0.5",
    "deterrant_matrix_configuration-random-coverage-50-suitability_threshold-0.5",
    "deterrant_matrix_configuration-random-coverage-100-suitability_threshold-0.5"
]

results = {}

ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
data = ds.ReadAsArray()
data[0:300, 0:1070] = 15
data[500:1069, 0:1070] = 15
data[250:350, 500:650] = 15
data[0:1069, 0:350] = 15
data[0:1069, 600:1070] = 15

data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

from copy import deepcopy
map_to_plot = deepcopy(data)

for i in range(1,10):
    map_to_plot[data == data_value_map[i]] = i

map_to_plot = np.flipud(map_to_plot)

for food_distribution_model in food_distribution_models:
    for food_value_model in food_value_models:
        for thermoregulation_model in thermoregulation_models:
            for slope_tolerance_model in slope_tolerance_models:
                for aggression_model in aggression_models:
                    for month in months:

                        num_timesteps_in_cropraiding_list = []
                        
                        for deterrant_matrix_configuration in deterrant_matrix_configurations:
                            folder_path = os.path.join(
                                base_folder,
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
                            
                            num_timesteps_in_cropraiding = 0
                            
                            try:
                                runs = os.listdir(folder_path)
                                num_runs = 0

                                fig, ax = plt.subplots(figsize = (6,6))
                                ax.yaxis.set_inverted(True)

                                row_size, col_size = data.shape
                                xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
                                
                                outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
                                LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
                                LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

                                map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

                                levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
                                clrs = ["greenyellow","mediumpurple","turquoise", "plum", "black", "blue", "yellow", "mediumseagreen", "forestgreen"] 
                                cmap, norm = colors.from_levels_and_colors(levels, clrs)

                                map.imshow(map_to_plot, cmap = cmap, norm=norm, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 0.5, zorder=1)

                                map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
                                map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])
                                
                                for run in runs:

                                    try:
                                        df = pd.read_csv(os.path.join(folder_path, run, "output_files", "agent_data.csv"))
                                        df = df[df["mode"] == "ForagingMode"]
                                        row = df["ROW"].dropna().values.astype(int)
                                        column = df["COL"].dropna().values.astype(int)

                                        if len(set(row[data[row, column] == 10])) <= 7 and len(set(column[data[row, column] == 10])) <= 7:
                                            num_timesteps_in_cropraiding += 0
                                            
                                        else:

                                            plantation_cells = (data[row, column] == 10)
                                            num_timesteps_in_cropraiding += plantation_cells.sum()
                                            num_runs += 1

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

                                            ax.scatter(x_new[-1], y_new[-1], 25, marker='^', color='deeppink', zorder=3)   
                                        
                                    except Exception as e:
                                        continue

                                plt.savefig(os.path.join("results_and_analysis/deterrance_policy_models/model_with_intervention/crop-raiding-trajs/", f"trajectory_{food_value_model}_{thermoregulation_model}_{aggression_model}_{month}_{deterrant_matrix_configuration}.png"), dpi = 300, bbox_inches = 'tight') 
                                plt.close()
                                        
                            except FileNotFoundError:
                                print(f"Folder not found: {folder_path}")

                            if num_runs == 0:
                                num_timesteps_in_cropraiding_list.append(0)
                            else:
                                num_timesteps_in_cropraiding_list.append(num_timesteps_in_cropraiding*num_runs)
                        
                        key = (food_value_model, thermoregulation_model, aggression_model, month)
                        results[key] = num_timesteps_in_cropraiding_list



for key, values in results.items():

    food_value_model, thermoregulation_model, aggression_model, month = key
    
    coverage_labels = [c.split('-')[3] for c in deterrant_matrix_configurations]
    
    df_heatmap = pd.DataFrame(
        {'Coverage (%)': coverage_labels, 'Crop Raiding Timesteps': values}
    ).set_index('Coverage (%)').T
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_heatmap, annot=True, fmt='g', cmap="coolwarm", linewidths=.5, vmin=0)
    plt.title(f'Crop Raiding Timesteps per Deterrent Coverage\n'
              f'Food Value: {food_value_model.split("-")[-1]}, '
              f'Thermo: {thermoregulation_model.split("-")[-1]}, '
              f'Aggression: {aggression_model.split("_")[-1]}, '
              f'Month: {month}')
    plt.ylabel('Metric')
    plt.xlabel('Deterrent Coverage (%)')
    plt.tight_layout()
    
    filename = f"heatmap_{food_value_model}_{thermoregulation_model}_{aggression_model}_{month}.png"
    plt.savefig(os.path.join("results_and_analysis/deterrance_policy_models/model_with_intervention/heatmaps/", filename))
    plt.close()
