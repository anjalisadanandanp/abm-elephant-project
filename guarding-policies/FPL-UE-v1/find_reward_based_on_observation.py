import os
import pandas as pd
from osgeo import gdal
import numpy as np

def find_rewards_based_on_intercepted_trajectories(output_folder):

    dict_of_attacked_targets = {}

    for simulation_folder in os.listdir(output_folder):

        # print(f"Processing folder: {simulation_folder}")

        try:

            df = pd.read_csv(os.path.join(output_folder, simulation_folder, "output_files/agent_data.csv"))
            df.dropna(subset=['ROW', 'COL'], inplace=True)

            unique_targets = df["target_attacked"].dropna().unique()

            for target in unique_targets:
                if target not in dict_of_attacked_targets:
                    dict_of_attacked_targets[target] = 0
                dict_of_attacked_targets[target] += 1
        
        except Exception as e:
            pass

    for target, count in dict_of_attacked_targets.items():
        print(f"Covered target {target} was attacked {count} times.")  

    df_dict_of_attacked_targets = pd.DataFrame.from_dict(dict_of_attacked_targets, orient='index', columns=['count'])

    df_dict_of_attacked_targets.reset_index(inplace=True)
    df_dict_of_attacked_targets.rename(columns={'index': 'target'}, inplace=True)
                                            
    df_dict_of_attacked_targets.to_csv(os.path.join(output_folder, "df_of_attacked_targets.csv"))
    
    targets_matrix = gdal.Open(os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init/potential_coverage_matrix.png")).ReadAsArray()
    unique_values = np.unique(targets_matrix)
    non_zero_unique_values = unique_values[unique_values != 0]
    total_targets = len(non_zero_unique_values)
    attacker_strategy_covered_targets = [0 for i in range(total_targets - 1)]


    for target in dict_of_attacked_targets:
        attacker_strategy_covered_targets[int(target - 1)] = 1


    return attacker_strategy_covered_targets


attacker_strategy_covered_targets = find_rewards_based_on_intercepted_trajectories("/home2/anjali/GitHub/abm-elephant-project/guarding-policies/FPL-UE-v1/game_model-v3_1-run-model-with-intervention-v2-no-knowledge/mitigation-measures-within-plantations/latitude-[[1052166]]-longitude-[[8572829]]/solitary_bulls/random-food-distribition-within-agricultural-plots/landscape-food-probability-forest-0.1-cropland-1.0/water-source-rivers-landscape-0.05/random-memory-forest-and_plantation-fringe-model/full-memory-forest-and_plantation-model/num_days_agent_survives_in_deprivation-10/maximum-food-in-a-forest-cell-25/thermoregulation-threshold-temperature-28/threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/slope_tolerance-30/num_days_agent_survives_in_deprivation-10/elephant_aggression_value_0.8/2010/Mar/num_protected_targets_5/num_strategic_traj_12_num_iterations_12/boundary_raster_discretised_750m/budget_k_5-max_game_steps_10-max_gamma_1.0-min_gamma_0.85-num_steps_gamma_decay_10-eta_10.0-M_30/game_step_1")

print("\n")
print("attacker strategy vector:", attacker_strategy_covered_targets)