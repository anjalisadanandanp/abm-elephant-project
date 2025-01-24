import os
import itertools

model_params_all = {
    "year": 2010,
    "month": ["Mar", "Jul"],
    "num_bull_elephants": 1, 
    "area_size": 1100,              
    "spatial_resolution": 30, 
    "max_food_val_cropland": 100,
    "max_food_val_forest": [10],
    "prob_food_forest": [0.10],
    "prob_food_cropland": [0.10],
    "prob_water_sources": [0.0, 0.001, 0.0001],
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
    "slope_tolerance": [30],
    "num_processes": 10,
    "iterations": 10,
    "max_time_steps": 288*5,
    "aggression_threshold_enter_cropland": 1.0,
    "human_habituation_tolerance": 1.0,
    "elephant_agent_visibility_radius": 500,
    "plot_stepwise_target_selection": False,
    "threshold_days_of_food_deprivation": [0, 1, 2, 3],
    "threshold_days_of_water_deprivation": [3],
    "number_of_feasible_movement_directions": 3,
    "track_in_mlflow": False,
    "elephant_starting_location": "user_input",
    "elephant_starting_latitude": 1047943,
    "elephant_starting_longitude": 8576931,
    "elephant_aggression_value": [0.8],
    "elephant_crop_habituation": False,
    "num_guards": 3,
    "ranger_visibility_radius": 500
    }

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



def find_folders_without_agent(root_dir):

    folders_to_clean = []

    param_dicts = generate_parameter_combinations(model_params_all)

    for model_params in param_dicts:

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

        data_folder = os.path.join(os.getcwd(), "model_runs", experiment_name, starting_location, elephant_category, landscape_food_probability, 
                                            water_holes_probability, memory_matrix_type, num_days_agent_survives_in_deprivation, maximum_food_in_a_forest_cell, 
                                            elephant_thermoregulation_threshold, threshold_food_derivation_days, threshold_water_derivation_days, 
                                            slope_tolerance, num_days_agent_survives_in_deprivation, elephant_aggression_value,
                                            str(model_params["year"]), str(model_params["month"]))

        if not os.path.exists(data_folder):
            # print(data_folder)
            # print("Folder does not exist")
            # print("\n")
            pass
        else:
            # print(data_folder)
            # print("Folder exists")
            # print("\n")

            #find all subfolders in the data folder
            subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
            # print(subfolders)
            # print("\n")
            
            #check if the subfolders contain the agent.csv file
            for subfolder in subfolders:
                if os.path.exists(os.path.join(subfolder, "output_files/agent_data.csv")):
                    # print(subfolder)
                    # print("Agent data file exists")
                    # print("\n")
                    pass
                else:
                    # print(subfolder)
                    # print("Agent data file does not exist")
                    folders_to_clean.append(subfolder)
                    
    return folders_to_clean

folders_to_clean = find_folders_without_agent(os.getcwd())

#write folders to clean to a text file
with open("folders_to_clean.txt", "w") as f:
    for folder in folders_to_clean:
        f.write(folder + "\n")
