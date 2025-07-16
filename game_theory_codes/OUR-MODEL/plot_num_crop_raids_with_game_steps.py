import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def make_plots(run_folder, output_folder):

    runs = os.listdir(run_folder)

    runs.sort()

    penalties_v1 = {}
    penalties_v2 = {}
    penalties_v3 = {}

    for run in tqdm(runs):

        try:

            game_step = int(run.split("_")[-1])

            print("---step---", game_step)

            out_save_folder = os.path.join(run_folder, "game_step_" + str(int(game_step)))
            num_trajs_threatening = pd.read_csv(os.path.join(out_save_folder, "association_matrix_num_visiting_trajs.csv"))
            num_food_resources_under_attack = pd.read_csv(os.path.join(out_save_folder, "boundary_patch_association_matrix.csv"))

            game_step_penalty_v1 = 0
            game_step_penalty_v2 = 0
            game_step_penalty_v3 = 0

            for boundary_id in num_food_resources_under_attack["boundary_patch_id"].unique():

                rows = num_trajs_threatening[num_trajs_threatening["Unnamed: 0"] == f"boundary_patch_{boundary_id}"]

                num_intersecting_trajs = rows.iloc[0, 1:].sum()

                food_sources_under_attack = num_food_resources_under_attack[num_food_resources_under_attack["boundary_patch_id"] == boundary_id]["total_food_value"].values[0]

                if num_intersecting_trajs > 0:
                    print(f"Boundary Patch ID: {boundary_id}, num intersecting trajs: {num_intersecting_trajs}, food sources under attack: {food_sources_under_attack}")

                game_step_penalty_v1 += num_intersecting_trajs*food_sources_under_attack
                game_step_penalty_v2 += num_intersecting_trajs
                game_step_penalty_v3 += food_sources_under_attack

            penalties_v1[game_step] = game_step_penalty_v1
            penalties_v2[game_step] = game_step_penalty_v2
            penalties_v3[game_step] = game_step_penalty_v3

        except:
            pass

    min_game_step = min(list(penalties_v1.keys()))
    max_game_step = max(list(penalties_v1.keys()))




    sorted_penalties= []
    step_ids = []
    
    for step in range(min_game_step, max_game_step):

        strategy_step_i = penalties_v1[step]

        sorted_penalties.append(strategy_step_i)
        step_ids.append(step)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    ax.plot(step_ids, sorted_penalties, "-s", color="black", linewidth=1.0, markersize=3.5)

    plt.grid(alpha=0.25)

    ax.set_xlabel('Game Step', fontsize=10)
    ax.set_ylabel('Defender Penalties with Game Step\n (num intersecting trajs X \nfood sources under attack)', fontsize=10)
    plt.tight_layout()

    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.xlim(min_game_step, max_game_step+1)

    plt.savefig(os.path.join(output_folder, "defender_penalties_with_game_step_v3.png"), dpi=300, bbox_inches="tight")




    sorted_penalties= []
    step_ids = []
    
    for step in range(min_game_step, max_game_step):

        strategy_step_i = penalties_v2[step]

        sorted_penalties.append(strategy_step_i)
        step_ids.append(step)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    ax.plot(step_ids, sorted_penalties, "-s", color="black", linewidth=1.0, markersize=3.5)

    plt.grid(alpha=0.25)

    ax.set_xlabel('Game Step', fontsize=10)
    ax.set_ylabel('Defender Penalties with Game Step\n (total number of intersecting \ntrajectories in boundary patches)', fontsize=10)
    plt.tight_layout()

    plt.xlim(min_game_step, max_game_step+1)

    plt.savefig(os.path.join(output_folder, "defender_penalties_with_game_step_v2.png"), dpi=300, bbox_inches="tight")




    sorted_penalties= []
    step_ids = []
    
    for step in range(min_game_step, max_game_step):

        strategy_step_i = penalties_v3[step]

        sorted_penalties.append(strategy_step_i)
        step_ids.append(step)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    ax.plot(step_ids, sorted_penalties, "-s", color="black", linewidth=1.0, markersize=3.5)

    plt.grid(alpha=0.25)

    ax.set_xlabel('Game Step', fontsize=10)
    ax.set_ylabel('Defender Penalties with Game Step\n (total food in agricultural \nplots under attack)', fontsize=10)
    plt.tight_layout()

    plt.xlim(min_game_step, max_game_step+1)

    plt.savefig(os.path.join(output_folder, "defender_penalties_with_game_step_v1.png"), dpi=300, bbox_inches="tight")



    return

run_folder = "game_theory_codes/OUR-MODEL/mitigation-measures-within-plantations-FPL-UE_v1_2/budget_k_10-max_game_steps_35-max_gamma_1.0-min_gamma_0.2-num_steps_gamma_decay_10-eta_0.0-M_30/coverage_matrix_init"
output_folder = "game_theory_codes/OUR-MODEL/mitigation-measures-within-plantations-FPL-UE_v1_2/budget_k_10-max_game_steps_35-max_gamma_1.0-min_gamma_0.2-num_steps_gamma_decay_10-eta_0.0-M_30/"


make_plots(run_folder, output_folder)