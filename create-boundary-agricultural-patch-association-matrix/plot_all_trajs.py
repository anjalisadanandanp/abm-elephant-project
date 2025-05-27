import os
import pandas as pd
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj, transform     
from mpl_toolkits.basemap import Basemap       
import matplotlib.colors as mcolors  
import matplotlib.cm as cm   
import itertools
from tqdm import tqdm



import warnings
warnings.filterwarnings("ignore")



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


def return_output_folder(experiment_name, model_params):

    elephant_category = "solitary_bulls"
    starting_location = "latitude-" + str(model_params["elephant_starting_latitude"]) + "-longitude-" + str(model_params["elephant_starting_longitude"])
    landscape_food_probability = "landscape-food-probability-forest-" + str(model_params["prob_food_forest"]) + "-cropland-" + str(model_params["prob_food_cropland"])
    food_availability_sceanario = "random-food-distribition-within-plantation"
    water_availability_sceanario = "water-source-rivers-landscape-" + str(model_params["prob_water_sources"])
    food_memory_matrix_type = "random-memory-forest-and_plantation-fringe-model"
    water_memory_matrix_type = "full-memory-forest-and_plantation-model"
    num_days_agent_survives_in_deprivation = "num_days_agent_survives_in_deprivation-" + str(model_params["num_days_agent_survives_in_deprivation"])
    maximum_food_in_a_forest_cell = "maximum-food-in-a-forest-cell-" + str(model_params["max_food_val_forest"])
    elephant_thermoregulation_threshold = "thermoregulation-threshold-temperature-" + str(model_params["thermoregulation_threshold"])
    threshold_food_derivation_days = "threshold_days_of_food_deprivation-" + str(model_params["threshold_days_of_food_deprivation"])
    threshold_water_derivation_days = "threshold_days_of_water_deprivation-" + str(model_params["threshold_days_of_water_deprivation"])
    slope_tolerance = "slope_tolerance-" + str(model_params["slope_tolerance"])
    num_days_agent_survives_in_deprivation = "num_days_agent_survives_in_deprivation-" + str(model_params["num_days_agent_survives_in_deprivation"])
    elephant_aggression_value = "elephant_aggression_value_" + str(model_params["elephant_aggression_value"])

    output_folder = os.path.join(experiment_name, starting_location, elephant_category, food_availability_sceanario, landscape_food_probability, 
                                 water_availability_sceanario, food_memory_matrix_type, water_memory_matrix_type, num_days_agent_survives_in_deprivation, maximum_food_in_a_forest_cell, 
                                 elephant_thermoregulation_threshold, threshold_food_derivation_days, threshold_water_derivation_days, 
                                 slope_tolerance, num_days_agent_survives_in_deprivation, elephant_aggression_value,
                                 str(model_params["year"]), str(model_params["month"]))
    
    return output_folder
    

model_params_all = {
    "year": 2010,
    "month": ["Mar", "Aug"],
    "num_bull_elephants": 1, 
    "area_size": 1100,              
    "spatial_resolution": 30, 
    "max_food_val_cropland": 100,
    "max_food_val_forest": [5],
    "prob_food_forest": [0.10],
    "prob_food_cropland": [0.10],
    "prob_water_sources": [1.0],
    "thermoregulation_threshold": [28, 32],
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
    "slope_tolerance": [30, 32.5, 35, 37.5, 40],
    "num_processes": 32,
    "iterations": 128,
    "max_time_steps": 288*30,
    "aggression_threshold_enter_cropland": 1.0,
    "elephant_agent_visibility_radius": 500,
    "plot_stepwise_target_selection": False,
    "threshold_days_of_food_deprivation": [0],
    "threshold_days_of_water_deprivation": [3],
    "number_of_feasible_movement_directions": 3,
    "track_in_mlflow": False,
    "elephant_starting_location": "user_input",
    "elephant_starting_latitude": 1049237,
    "elephant_starting_longitude": 8570917,
    "elephant_aggression_value": [0.2, 0.8],
    "elephant_crop_habituation": False
    }

param_dicts = generate_parameter_combinations(model_params_all)
experiment_name = "model-without-intervention"

output_folders = [return_output_folder(experiment_name, param_dict) for param_dict in param_dicts]

for folder in tqdm(output_folders):

    run_folder = os.path.join(os.getcwd(), "model_runs/", folder)
    save_folder = os.path.join(os.getcwd(), "create-boundary-agricultural-patch-association-matrix/outputs/", folder)


    base_folder = run_folder    
    folders = os.listdir(base_folder)

    ds = gdal.Open(os.path.join(base_folder, folders[0], "env/slope_matrix.tif"))
    data = ds.ReadAsArray()
    data = np.flip(data, axis=0)
    row_size, col_size = data.shape
    xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

    fig, ax = plt.subplots(figsize = (10,10))
    ax.yaxis.set_inverted(True)

    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
    LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
    LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

    map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

    map.imshow(data, cmap = "coolwarm", extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 0.75, vmin = 0, vmax = 60)

    map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
    map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

    for folder in folders:

        try:
            data = pd.read_csv(os.path.join(base_folder, folder, "output_files/agent_data.csv"))
            
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
                        width=0.0010)
            
            ax.scatter(x_new[0], y_new[0], 5, marker='o', color='red', zorder=3)
            ax.scatter(x_new[-1], y_new[-1], 5, marker='^', color='black', zorder=2)

        except:
            pass

    plt.savefig(os.path.join(save_folder, "trajectories_on_slope_map.png"), dpi = 300, bbox_inches = 'tight')
    plt.close()
