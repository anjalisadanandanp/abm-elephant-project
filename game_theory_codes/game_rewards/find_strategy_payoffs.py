import os
import pandas as pd
import pathlib

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.getcwd())









def return_cost(attacker_rewards_df, defender_rewards_df, strategy_output_folder_withoutrangers, strategy_output_folder_withrangers, NUM_TARGETS, savedf_path=None):

    expected_rewards_attacker_withoutrangers = [0] * NUM_TARGETS
    expected_penalty_attacker_withoutrangers = [0] * NUM_TARGETS
    expected_rewards_defender_withoutrangers = [0] * NUM_TARGETS
    expected_penalty_defender_withoutrangers = [0] * NUM_TARGETS

    num_visits_to_target_attacker_withoutrangers = [0] * NUM_TARGETS
    num_visits_to_target_attacker_withrangers = [0] * NUM_TARGETS

    expected_rewards_attacker_withrangers = [0] * NUM_TARGETS
    expected_penalty_attacker_withrangers = [0] * NUM_TARGETS
    expected_rewards_defender_withrangers = [0] * NUM_TARGETS
    expected_penalty_defender_withrangers = [0] * NUM_TARGETS

    num_simulations_withoutrangers = 0
    num_simulations_withrangers = 0

    subfolders_without_rangers = [f.path for f in os.scandir(strategy_output_folder_withoutrangers) if f.is_dir()]
    num_subfolders_without_rangers = len(subfolders_without_rangers)
    num_simulations_withoutrangers += num_subfolders_without_rangers

    for subfolder in subfolders_without_rangers:

        subfolder_name = pathlib.Path(subfolder).parts[-1]

        print("\nsimulation without rangers:", subfolder_name)

        output_folder = os.path.join(
            strategy_output_folder_withrangers,
            subfolder_name
        )

        subfolders_with_rangers = [f.path for f in os.scandir(output_folder) if f.is_dir()]
        num_subfolders_with_rangers = len(subfolders_with_rangers)
        num_simulations_withrangers += num_subfolders_with_rangers

        strategy_df = pd.read_csv(os.path.join(output_folder, "strategy_" + subfolder_name + "_rewards_and_penalties.csv"))

        unique_targetIDs = strategy_df["targetID"].dropna().unique()
        targetID_counts = strategy_df["targetID"].value_counts()[unique_targetIDs]

        all_targetIDs = set(range(1, len(attacker_rewards_df) + 1))
        strategy_targetIDs = set(strategy_df["targetID"].dropna().unique())
        missing_targetIDs = all_targetIDs - strategy_targetIDs

        # print("targets intercepted by elephants:", strategy_targetIDs)
        # print("targets not intercepted by elephants:", missing_targetIDs)

        targetID_counts = strategy_df["targetID"].value_counts()
        for id in strategy_targetIDs:
            number_of_visits = targetID_counts[id]

            rewards = strategy_df[strategy_df["targetID"] == id].iloc[0]

            defender_reward = rewards["defender_reward"]
            defender_penalty = rewards["defender_penalty"]
            attacker_reward = rewards["attacker_reward"]
            attacker_penalty = rewards["attacker_penalty"]

            target_idx = int(id) - 1
            expected_rewards_attacker_withoutrangers[target_idx] += attacker_reward
            expected_penalty_attacker_withoutrangers[target_idx] += 0
            expected_rewards_defender_withoutrangers[target_idx] += 0
            expected_penalty_defender_withoutrangers[target_idx] += defender_penalty

            if number_of_visits > 0:
                num_visits_to_target_attacker_withoutrangers[target_idx] += 1 

        for id in missing_targetIDs:

            rewards_a = attacker_rewards_df[attacker_rewards_df["targetID"] == id].iloc[0]
            rewards_d = defender_rewards_df[defender_rewards_df["targetID"] == id].iloc[0]

            defender_reward = rewards_d["reward"]
            defender_penalty = rewards_d["penalty"]
            attacker_reward = rewards_a["reward"]
            attacker_penalty = rewards_a["penalty"]

            target_idx = int(id) - 1
            expected_rewards_attacker_withoutrangers[target_idx] += 0
            expected_penalty_attacker_withoutrangers[target_idx] += attacker_penalty
            expected_rewards_defender_withoutrangers[target_idx] += defender_reward
            expected_penalty_defender_withoutrangers[target_idx] += 0

        for subfolder in subfolders_with_rangers:

            subfolder_name = pathlib.Path(subfolder).parts[-1]

            print("simulation with rangers:", subfolder_name)

            strategy_df = pd.read_csv(os.path.join(output_folder, "strategy_" + subfolder_name + "_rewards_and_penalties.csv"))

            unique_targetIDs = strategy_df["targetID"].dropna().unique()
            targetID_counts = strategy_df["targetID"].value_counts()[unique_targetIDs]

            all_targetIDs = set(range(1, len(attacker_rewards_df) + 1))
            strategy_targetIDs = set(strategy_df["targetID"].dropna().unique())
            missing_targetIDs = all_targetIDs - strategy_targetIDs

            # print("targets intercepted by elephants:", strategy_targetIDs)
            # print("targets not intercepted by elephants:", missing_targetIDs)

            targetID_counts = strategy_df["targetID"].value_counts()
            for id in strategy_targetIDs:
                number_of_visits = targetID_counts[id]

                rewards = strategy_df[strategy_df["targetID"] == id].iloc[0]

                defender_reward = rewards["defender_reward"]
                defender_penalty = rewards["defender_penalty"]
                attacker_reward = rewards["attacker_reward"]
                attacker_penalty = rewards["attacker_penalty"]

                target_idx = int(id) - 1
                expected_rewards_attacker_withrangers[target_idx] += attacker_reward
                expected_penalty_attacker_withrangers[target_idx] += 0
                expected_rewards_defender_withrangers[target_idx] += 0
                expected_penalty_defender_withrangers[target_idx] += defender_penalty

                if number_of_visits > 0:
                    num_visits_to_target_attacker_withrangers[target_idx] += 1 

            for id in missing_targetIDs:

                rewards_a = attacker_rewards_df[attacker_rewards_df["targetID"] == id].iloc[0]
                rewards_d = defender_rewards_df[defender_rewards_df["targetID"] == id].iloc[0]

                defender_reward = rewards_d["reward"]
                defender_penalty = rewards_d["penalty"]
                attacker_reward = rewards_a["reward"]
                attacker_penalty = rewards_a["penalty"]

                target_idx = int(id) - 1
                expected_rewards_attacker_withrangers[target_idx] += 0
                expected_penalty_attacker_withrangers[target_idx] += attacker_penalty
                expected_rewards_defender_withrangers[target_idx] += defender_reward
                expected_penalty_defender_withrangers[target_idx] += 0
                

    prob_attack_target_withoutrangers = [value / num_simulations_withoutrangers if num_simulations_withoutrangers != 0 else 0 for value in num_visits_to_target_attacker_withoutrangers]
    prob_attack_target_withrangers = [value / num_simulations_withrangers if num_simulations_withrangers != 0 else 0 for value in num_visits_to_target_attacker_withrangers]


    data = {
        'expected_rewards_attacker_withrangers': expected_rewards_attacker_withrangers,
        'expected_penalty_attacker_withrangers': expected_penalty_attacker_withrangers,
        'expected_rewards_defender_withrangers': expected_rewards_defender_withrangers,
        'expected_penalty_defender_withrangers': expected_penalty_defender_withrangers,
        'expected_rewards_attacker_withoutrangers': expected_rewards_attacker_withoutrangers,
        'expected_penalty_attacker_withoutrangers': expected_penalty_attacker_withoutrangers,
        'expected_rewards_defender_withoutrangers': expected_rewards_defender_withoutrangers,
        'expected_penalty_defender_withoutrangers': expected_penalty_defender_withoutrangers,
        "num_visits_to_target_attacker_withoutrangers":num_visits_to_target_attacker_withoutrangers,
        "num_visits_to_target_attacker_withrangers":num_visits_to_target_attacker_withrangers,
        "prob_attack_target_withoutrangers": prob_attack_target_withoutrangers,
        "prob_attack_target_withrangers": prob_attack_target_withrangers
    }


    df = pd.DataFrame(data)
    df.to_csv('expected_rewards_and_penalties.csv', index=False)


    #-------------------------------------------------------------------------------#
    defender_expected_utility_withoutrangers = 0
    defender_expected_utility_withrangers = 0

    for index, row in df.iterrows():

        if row["prob_attack_target_withoutrangers"] > 0 or row["prob_attack_target_withrangers"] > 0:
            if row["prob_attack_target_withoutrangers"] > 0:  
                defender_expected_utility_withoutrangers += row["prob_attack_target_withoutrangers"]*row["expected_penalty_defender_withoutrangers"]

            if row["prob_attack_target_withrangers"] > 0:  
                defender_expected_utility_withrangers += row["prob_attack_target_withrangers"]*row["expected_penalty_defender_withrangers"]

    cost = defender_expected_utility_withoutrangers - defender_expected_utility_withrangers

    print("defender_expected_utility_withoutrangers:", defender_expected_utility_withoutrangers, defender_expected_utility_withrangers, cost)

    return cost


if __name__ == "__main__":
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
        "num_processes": 8,
        "iterations": 8,
        "max_time_steps": 288 * 10,
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
        "elephant_crop_habituation": False,
        "num_guards": 3,
        "ranger_visibility_radius": 1000,
    }

    run_name = "pulsar_01-31-25__12-31"
    experiment_name = "ranger-deployment-within-plantations/" + run_name

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

    strategy_output_folder1 = os.path.join(
        os.getcwd(),
        "model_runs",
        experiment_name,
        "without_rangers",
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

    strategy_output_folder2 = os.path.join(
        os.getcwd(),
        "model_runs",
        experiment_name,
        "with_rangers",
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



    NUM_TARGETS = 438

    attacker_rewards_df = pd.read_csv("game_theory_codes/game_rewards/outputs/attacker_rewards_penalties.csv")
    defender_rewards_df = pd.read_csv("game_theory_codes/game_rewards/outputs/defender_rewards_penalties.csv")


    return_cost(attacker_rewards_df, defender_rewards_df, strategy_output_folder1, strategy_output_folder2, NUM_TARGETS)