import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import importlib
import pathlib
import yaml
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


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

module = importlib.import_module('game_theory_codes.FPL-UE.play-games.abm_model_HEC_with_landscape_deterrent_policies')
batch_run_model = module.batch_run_model



model_params = {
        "year": 2010,
        "month": "Mar",
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
        "num_processes": 12,
        "iterations": 72,
        "max_time_steps": 288 * 30,
        "aggression_threshold_enter_cropland": 1.0,
        "human_habituation_tolerance": 1.0,
        "elephant_agent_visibility_radius": 1000,
        "plot_stepwise_target_selection": False,
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


experiment_name = "simulations-without-mitigation" 

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

water_holes_probability = "water-holes-within-landscape-" + str(
    model_params["prob_water_sources"]
)

memory_matrix_type = "random-memory-matrix-model"

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

num_days_agent_survives_in_deprivation = (
    "num_days_agent_survives_in_deprivation-"
    + str(model_params["num_days_agent_survives_in_deprivation"])
)

elephant_aggression_value = "elephant_aggression_value_" + str(
    model_params["elephant_aggression_value"]
)

output_folder = os.path.join(
    os.getcwd(),
    "game_theory_codes/FPL-UE/create-strategy-set",
    experiment_name,
    starting_location,
    elephant_category,
    landscape_food_probability,
    water_holes_probability,
    memory_matrix_type,
    num_days_agent_survives_in_deprivation,
    maximum_food_in_a_forest_cell,
    elephant_thermoregulation_threshold,
    threshold_food_derivation_days,
    threshold_water_derivation_days,
    slope_tolerance,
    num_days_agent_survives_in_deprivation,
    elephant_aggression_value,
    str(model_params["year"]),
    str(model_params["month"])
)


#--------------------GAME STEP AT T=0--------------------------#
output_folder_step_0 = os.path.join(output_folder, "game_step_0")


#------------------create the coverage matrix------------------#
path = pathlib.Path(output_folder_step_0)
path.mkdir(parents=True, exist_ok=True)

with open(os.path.join(output_folder_step_0, "model_parameters.yaml"), "w") as configfile:
    yaml.dump(model_params, configfile, default_flow_style=False)

coverage_matrix = np.zeros((250,250), dtype=np.int8)
#------------------create the coverage matrix------------------#



#------------------plot the coverage matrix------------------#
fig, ax = plt.subplots(figsize=(8, 8))

cmap = mcolors.ListedColormap(["white", "red"])

im = ax.imshow(coverage_matrix, cmap=cmap, vmin=0, vmax=1)

ax.set_xticks([])
ax.set_yticks([])

legend_elements = [
    Patch(facecolor="red", edgecolor='black', label='Protected'),
    Patch(facecolor="white", edgecolor='black', label='Unprotected')
]
ax.legend(handles=legend_elements, loc="upper right")

plt.savefig(
    os.path.join(output_folder_step_0, "defender_coverage_matrix.png"),
    bbox_inches="tight",
    dpi=300,
    
)

plt.close()
#------------------plot the coverage matrix------------------#


#------------------save the coverage matrix------------------#  
source_file = gdal.Open("game_theory_codes/FPL-UE/create-strategy-set/outputs/interpolated_LULC_matrix.tif")

cols = source_file.RasterXSize
rows = source_file.RasterYSize
projection = source_file.GetProjection()
geotransform = source_file.GetGeoTransform()

output_file = "game_theory_codes/FPL-UE/outputs/coverage_matrix.tif"

driver = gdal.GetDriverByName("GTiff")
output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)

output_dataset.SetProjection(projection)
output_dataset.SetGeoTransform(geotransform)

output_band = output_dataset.GetRasterBand(1)
output_band.WriteArray(coverage_matrix.astype(np.uint8))

source_file = None
output_dataset = None


output_file = os.path.join(output_folder_step_0, "defender_coverage_matrix.tif")

driver = gdal.GetDriverByName("GTiff")
output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)

output_dataset.SetProjection(projection)
output_dataset.SetGeoTransform(geotransform)

output_band = output_dataset.GetRasterBand(1)
output_band.WriteArray(coverage_matrix.astype(np.uint8))

source_file = None
output_dataset = None
#------------------save the coverage matrix------------------#  





batch_run_model(model_params, experiment_name, output_folder_step_0)




