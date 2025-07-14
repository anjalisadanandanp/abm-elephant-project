import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj, transform  
from mpl_toolkits.basemap import Basemap    
import rasterio
from rasterio.features import shapes
import fiona
import geojson
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def make_plots(run_folder, output_folder):

    runs = os.listdir(run_folder)

    runs.sort()

    targets = set()

    game_steps = []

    for run in runs:

        if os.path.isdir(os.path.join(run_folder, run)):

            game_step = int(run.split("_")[-1])

            df = pd.read_csv(os.path.join(run_folder, run, "reward_estimate.csv"))

            target_indices = list(df[df["reward_estimate"] > 0].index)

            targets = targets | set(target_indices)

            game_steps.append(game_step)

    targets = list(targets)
    targets.sort()

    game_steps.sort()
    
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    cmap = plt.cm.tab20 
    colors = cmap(np.linspace(0, 1, len(targets)))
    
    # np.random.shuffle(colors)

    for i, target in enumerate(targets):

        reward_estimates = []

        for game_step in game_steps:

            df = pd.read_csv(os.path.join(run_folder, "game_step_" + str(game_step), "reward_estimate.csv"))

            target_reward_estimate = df[df["Unnamed: 0"] == target]["reward_estimate"].values[0]

            reward_estimates.append(target_reward_estimate)

        ax.plot(game_steps, reward_estimates, "-s", color=colors[i], linewidth=0.8, markersize=2.5, label=f'Target {target + 1}')

    plt.grid(alpha=0.25)

    ax.set_xlabel('Game Step', fontsize=10)
    ax.set_ylabel('target reward estimates', fontsize=10)
    plt.tight_layout()

    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

    plt.savefig(os.path.join(output_folder, "target_reward_estimates.png"), dpi=300, bbox_inches="tight")

    return

run_folder = "game_theory_codes/OUR-MODEL/mitigation-measures-within-plantations-FPL-UE_v1_1/budget_k_10-max_game_steps_35-max_gamma_1.0-min_gamma_0.2-num_steps_gamma_decay_10-eta_0.0-M_30/coverage_matrix_init"
output_folder = "game_theory_codes/OUR-MODEL/mitigation-measures-within-plantations-FPL-UE_v1_1/budget_k_10-max_game_steps_35-max_gamma_1.0-min_gamma_0.2-num_steps_gamma_decay_10-eta_0.0-M_30/"

make_plots(run_folder, output_folder)