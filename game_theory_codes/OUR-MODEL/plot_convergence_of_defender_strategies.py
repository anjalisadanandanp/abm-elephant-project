import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def make_plots(run_folder, output_folder):

    runs = os.listdir(run_folder)

    runs.sort()

    result_dict = {}

    for run in tqdm(runs):

        game_step = int(run.split("_")[-1])

        defender_strategy_matrix = gdal.Open(os.path.join(run_folder, run, "defender_coverage_matrix.tif")).ReadAsArray()
        defender_strategy = np.unique(defender_strategy_matrix)
        defender_strategy = np.delete(defender_strategy, 0) 

        result_dict[game_step] = defender_strategy

    min_game_step = min(list(result_dict.keys()))
    max_game_step = max(list(result_dict.keys()))

    
    num_strategy_diff = []
    step_ids = []
    
    for step in range(min_game_step, max_game_step):

        strategy_step_i_1 = set(result_dict[step])
        strategy_step_i_2 = set(result_dict[step + 1])

        unique_to_list1 = list(strategy_step_i_1 - strategy_step_i_2)
        unique_to_list2 = list(strategy_step_i_2 - strategy_step_i_1)

        unique_entries = unique_to_list1 + unique_to_list2

        total_unique_entries = len(unique_entries)

        num_strategy_diff.append(total_unique_entries)
        step_ids.append(step+1)


    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    ax.plot(step_ids, num_strategy_diff, "-s", color="black", linewidth=1.0, markersize=3.5)

    plt.grid(alpha=0.25)

    ax.set_xlabel('Game Step', fontsize=10)
    ax.set_ylabel('Total Number of strategy differences \n in consecutive game steps', fontsize=10)
    ax.set_title('Strategy Convergence Over Game Steps', fontsize=10)
    plt.tight_layout()

    plt.xlim(min_game_step, max_game_step+1)

    plt.savefig(os.path.join(output_folder, "convergence_in_defender_strategy.png"), dpi=300, bbox_inches="tight")

    return

run_folder = "game_theory_codes/OUR-MODEL/mitigation-measures-within-plantations-FPL-UE_v1_1/budget_k_10-max_game_steps_35-max_gamma_1.0-min_gamma_0.2-num_steps_gamma_decay_10-eta_0.0-M_30/latitude-1049237-longitude-8570917/solitary_bulls/random-food-distribition-within-agricultural-plots-and-other-plantation-cells/landscape-food-probability-forest-0.1-cropland-0.1/water-source-rivers-landscape-1.0/random-memory-forest-and_plantation-fringe-model/full-memory-forest-and_plantation-model/num_days_agent_survives_in_deprivation-10/maximum-food-in-a-forest-cell-5/thermoregulation-threshold-temperature-28/threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/slope_tolerance-35/num_days_agent_survives_in_deprivation-10/elephant_aggression_value_0.8/2010/Mar/"
output_folder = "game_theory_codes/OUR-MODEL/mitigation-measures-within-plantations-FPL-UE_v1_1/budget_k_10-max_game_steps_35-max_gamma_1.0-min_gamma_0.2-num_steps_gamma_decay_10-eta_0.0-M_30/"


make_plots(run_folder, output_folder)