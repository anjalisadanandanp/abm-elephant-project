#------------importing libraries----------------#
import os

import sys
sys.path.append(os.getcwd())

import pathlib
import yaml
import itertools
import mlflow
import smtplib
from email.mime.text import MIMEText
import time
from multiprocessing import freeze_support

from mesageo_elephant_project.elephant_project.model.abm_model_HEC_v2 import batch_run_model
#------------importing libraries----------------#



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
    "prob_water_sources": [0.0001],
    "thermoregulation_threshold": [32],
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
    "slope_tolerance": [30, 35, 40],
    "num_processes": 8,
    "iterations": 8,
    "max_time_steps": 288*10,
    "aggression_threshold_enter_cropland": 1.0,
    "elephant_agent_visibility_radius": 500,
    "plot_stepwise_target_selection": True,
    "threshold_days_of_food_deprivation": [0, 1, 2, 3],
    "threshold_days_of_water_deprivation": [3],
    "number_of_feasible_movement_directions": 3,
    "track_in_mlflow": False,
    "elephant_starting_location": "user_input",
    "elephant_starting_latitude": 1049237,
    "elephant_starting_longitude": 8570917,
    "elephant_aggression_value": [0.8],
    "elephant_crop_habituation": False
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


def run_model(experiment_name, model_params):

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

    output_folder = os.path.join(os.getcwd(), "model_runs/", experiment_name, starting_location, elephant_category, landscape_food_probability, 
                                 water_holes_probability, memory_matrix_type, num_days_agent_survives_in_deprivation, maximum_food_in_a_forest_cell, 
                                 elephant_thermoregulation_threshold, threshold_food_derivation_days, threshold_water_derivation_days, 
                                 slope_tolerance, num_days_agent_survives_in_deprivation, elephant_aggression_value,
                                 str(model_params["year"]), str(model_params["month"]))
    
    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_folder, 'model_parameters.yaml'), 'w') as configfile:
        yaml.dump(model_params, configfile, default_flow_style=False)

    batch_run_model(model_params, experiment_name, output_folder)

    return


class Experiment:

    def __init__(self, email_address, email_password):
        self.email_address = email_address
        self.email_password = email_password

    def run_experiment(self, send_notification):

        print("Running experiment...!")

        freeze_support()

        start = time.time()

        experiment_name = "exploratory-search-ID-01"

        if model_params_all["track_in_mlflow"] == True:
            try:
                mlflow.create_experiment(experiment_name)
            except:
                print("experiment already exists")

        param_dicts = generate_parameter_combinations(model_params_all)

        for model_params in param_dicts:
            run_model(experiment_name, model_params)

        end = time.time()

        print("Total time taken:", (end-start), "seconds")

        if send_notification == True:
            self.send_notification_email()

    def send_notification_email(self):
        msg = MIMEText("elephant-abm-project: Your experiment has finished running!")
        msg['Subject'] = "Experiment Notification: PAMPA"
        msg['To'] = self.email_address

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.email_address, self.email_password)
            server.sendmail(self.email_address, self.email_address, msg.as_string())
            server.quit()
            print("Notification email sent successfully.")

        except Exception as e:
            print("Error sending email:", e)

if __name__ == "__main__":
    experiment = Experiment("anjalisadanandan96@gmail.com", "fqdceolumrwtnmxo")
    experiment.run_experiment(send_notification=True)
    
