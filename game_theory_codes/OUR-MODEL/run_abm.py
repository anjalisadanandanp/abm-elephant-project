import os
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import yaml
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

module = importlib.import_module('game_theory_codes.OUR-MODEL.abm_model_HEC_with_landscape_deterrent_policies')
batch_run_model = module.batch_run_model




def run_abm(model_params, experiment_name, output_folder):

    with open(os.path.join(output_folder, "model_parameters.yaml"), "w") as configfile:
        yaml.dump(model_params, configfile, default_flow_style=False)




    num_strategic_trajectories  =  0

    while num_strategic_trajectories < model_params["iterations"]:

        batch_run_model(model_params, experiment_name, output_folder)

        runs = os.listdir(output_folder)

        num_strategic_trajectories  =  0

        for run in runs:
            if os.path.isdir(os.path.join(output_folder, run)):
                agent_df = pd.read_csv(os.path.join(output_folder, run, "output_files", "agent_data.csv"))

                targets_attacked = agent_df["target_attacked"].dropna().unique()

                if len(targets_attacked) == 0:
                    num_strategic_trajectories += 1

                else:
                    flag = True
                    for target in targets_attacked:
                        agent_df_attacking = agent_df[agent_df["target_attacked"] == target]
                        if len(agent_df_attacking) > 12:
                            flag = False

                    if flag == True:
                        num_strategic_trajectories += 1

                    if flag == False:
                        shutil.rmtree(os.path.join(output_folder, run))

    return 





















if __name__ == "__main__":

    model_params = {
            "year": 2010,
            "month": "Mar",
            "num_bull_elephants": 1,
            "area_size": 1100,
            "spatial_resolution": 30,
            "max_food_val_cropland": 100,
            "max_food_val_forest": 5,
            "prob_food_forest": 0.10,
            "prob_food_cropland": 0.10,
            "prob_water_sources": 1.0,
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
            "num_processes": 8,
            "iterations": 8,
            "max_time_steps": 288 * 30,
            "aggression_threshold_enter_cropland": 1.0,
            "human_habituation_tolerance": 1.0,
            "elephant_agent_visibility_radius": 500,
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

    experiment_name = "abm_runs_strategic_agents/" 

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

    output_folder = os.path.join(
        os.getcwd(),
        "game_theory_codes/OUR-MODEL",
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
        str(model_params["month"])
    )

    os.makedirs(output_folder, exist_ok=True)

    run_abm(
        model_params=model_params,
        experiment_name=experiment_name,
        output_folder=output_folder
    )


