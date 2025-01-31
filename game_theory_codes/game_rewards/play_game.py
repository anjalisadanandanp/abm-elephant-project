import os
from osgeo import gdal
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import pathlib
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib import colors
from pyproj import Proj, transform
import yaml
import itertools

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

from mesageo_elephant_project.elephant_project.model.abm_model_HEC_v5 import (
    batch_run_model
)

from experiments.ranger_deployment.experiment_names import FancyNameGenerator
from game_theory_codes.game_rewards.find_ranger_locations import optimise_ranger_locations
from game_theory_codes.game_rewards.find_strategy_payoffs import return_cost
from game_theory_codes.game_rewards.GA_optimisation import RangerOptimizer

def generate_parameter_combinations(model_params_all):

    month = model_params_all["month"]
    max_food_val_forest = model_params_all["max_food_val_forest"]
    prob_food_forest = model_params_all["prob_food_forest"]
    prob_food_cropland = model_params_all["prob_food_cropland"]
    thermoregulation_threshold = model_params_all["thermoregulation_threshold"]
    threshold_days_food = model_params_all["threshold_days_of_food_deprivation"]
    threshold_days_water = model_params_all["threshold_days_of_water_deprivation"]
    prob_water_sources = model_params_all["prob_water_sources"]
    num_days_agent_survives_in_deprivation = model_params_all[
        "num_days_agent_survives_in_deprivation"
    ]
    slope_tolerance = model_params_all["slope_tolerance"]
    elephant_aggression_value = model_params_all["elephant_aggression_value"]

    combinations = list(
        itertools.product(
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
            elephant_aggression_value,
        )
    )

    all_param_dicts = []
    for combo in combinations:
        params_dict = model_params_all.copy()

        params_dict.update(
            {
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
                "elephant_aggression_value": combo[10],
            }
        )

        all_param_dicts.append(params_dict)

    return all_param_dicts


def run_abm_without_rangers(experiment_name, model_params, output_folder, step=None, strategy=None):

    print(model_params)

    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_folder, "model_parameters.yaml"), "w") as configfile:
        yaml.dump(model_params, configfile, default_flow_style=False)

    batch_run_model(model_params, experiment_name, output_folder, step, strategy)

    return


def run_abm_given_state_and_ranger_locations(experiment_name, model_params, output_folder, strategy, run_folder, **kwargs):

    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)

    model_params_restart = model_params.copy()

    for dictionary in kwargs.values():
        model_params_restart.update(dictionary)

    with open(os.path.join(output_folder, "model_parameters.yaml"), "w") as configfile:
        yaml.dump(model_params_restart, configfile, default_flow_style=False)

    batch_run_model(model_params=model_params_restart, 
                    experiment_name=experiment_name, 
                    output_folder=output_folder, 
                    strategy=strategy,
                    run_folder_name=run_folder)

    return


def save_strategies_to_yaml(strategy, filelocation):
    
    """Clean and save ranger strategies in readable YAML format."""
    
    clean_strategy = {
        'id': int(strategy['id']),
        'ranger_locations': [
            [float(pos[0]), float(pos[1])] 
            for pos in strategy['ranger_locations']
        ],
        'total_cost': float(strategy['total_cost']),
        'convergence': bool(strategy['convergence'])
    }
    
    with open(filelocation, 'w') as f:
        yaml.dump(clean_strategy, f, default_flow_style=False, sort_keys=False)

    return
    




    
class LandUseRewards:

    def __init__(self, raster_path):
        self.raster_path = raster_path
        self.lulc_data = None
        self.geotransform = None
        self.load_raster()

    def load_raster(self):
        """Read and store LULC raster data"""
        try:
            ds = gdal.Open(self.raster_path)
            if ds is None:
                raise ValueError("Could not open raster file")

            self.lulc_data = ds.ReadAsArray()
            self.geotransform = ds.GetGeoTransform()
            self.projection = ds.GetProjection()
            ds = None

        except Exception as e:
            raise Exception(f"Error reading raster: {str(e)}")

    def interpolate_matrix(self, target_shape):
        """Interpolate matrix to new dimensions"""
        if self.lulc_data is None:
            raise ValueError("Raster data not loaded")

        y, x = np.mgrid[0 : self.lulc_data.shape[0], 0 : self.lulc_data.shape[1]]
        points = np.column_stack((y.flat, x.flat))
        values = self.lulc_data.flat

        grid_y, grid_x = np.mgrid[0 : target_shape[0], 0 : target_shape[1]]
        scaling_y = self.lulc_data.shape[0] / target_shape[0]
        scaling_x = self.lulc_data.shape[1] / target_shape[1]

        grid_y = grid_y * scaling_y
        grid_x = grid_x * scaling_x

        return griddata(points, values, (grid_y, grid_x), method="nearest")

    def get_cell_value(self, row, col):
        """Get LULC code for specific cell"""
        if self.lulc_data is None:
            raise ValueError("Raster data not loaded")
        return self.lulc_data[row, col]

    def plot_matrices(self, interpolated=None):
        """Plot original and interpolated matrices side by side"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        data_value_map = {1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 9, 7: 10, 8: 14, 9: 15}

        data_LULC = self.lulc_data.copy()

        for i in range(1, 10):
            data_LULC[data_LULC == data_value_map[i]] = i

        levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        clrs = [
            "greenyellow",
            "mediumpurple",
            "turquoise",
            "plum",
            "black",
            "blue",
            "yellow",
            "mediumseagreen",
            "forestgreen",
        ]
        cmap, norm = colors.from_levels_and_colors(levels, clrs)

        im1 = ax1.imshow(data_LULC, cmap=cmap, norm=norm)
        ax1.set_title(f"Original Matrix {data_LULC.shape}")
        ax1.set_xticks([])
        ax1.set_yticks([])

        cbar = plt.colorbar(
            im1, ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9], fraction=0.046, pad=0.04, ax=ax1
        )
        cbar.ax.set_yticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        cbar.ax.set_yticklabels(
            [
                "Deciduous Broadleaf Forest",
                "Built-up Land",
                "Mixed Forest",
                "Shrubland",
                "Barren Land",
                "Water Bodies",
                "Plantations",
                "Grassland",
                "Broadleaf evergreen forest",
            ]
        )

        if interpolated is not None:

            interpolated_data = interpolated.copy()

            for i in range(1, 10):
                interpolated_data[interpolated_data == data_value_map[i]] = i

        if interpolated_data is not None:
            im2 = ax2.imshow(interpolated_data, cmap=cmap, norm=norm)
            ax2.set_title(f"Interpolated Matrix {interpolated_data.shape}")
            ax2.set_xticks([])
            ax2.set_yticks([])

            cbar = plt.colorbar(
                im2, ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9], fraction=0.046, pad=0.04, ax=ax2
            )
            cbar.ax.set_yticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9])
            cbar.ax.set_yticklabels(
                [
                    "Deciduous Broadleaf Forest",
                    "Built-up Land",
                    "Mixed Forest",
                    "Shrubland",
                    "Barren Land",
                    "Water Bodies",
                    "Plantations",
                    "Grassland",
                    "Broadleaf evergreen forest",
                ]
            )

        plt.tight_layout()
        plt.savefig(
            "game_theory_codes/game_rewards/outputs/interpolated_LULC_matrix.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def get_matrix_indices(self, lat, lon):
        """Convert lat/lon to matrix indices"""
        x = int((lon - self.geotransform[0]) / self.geotransform[1])
        y = int((lat - self.geotransform[3]) / self.geotransform[5])
        return y, x

    def get_value_at_coords(self, lat, lon, interpolated=None):
        """Get value from original or interpolated matrix at coordinates"""
        y, x = self.get_matrix_indices(lat, lon)
        if interpolated is not None:
            scale_y = interpolated.shape[0] / self.lulc_data.shape[0]
            scale_x = interpolated.shape[1] / self.lulc_data.shape[1]
            y_interp = int(y * scale_y)
            x_interp = int(x * scale_x)
            return interpolated[y_interp, x_interp]
        else:
            return self.lulc_data[y, x]

    def plot_coords(self, lat, lon, interpolated=None):
        """Plot matrices with highlighted coordinates"""
        y, x = self.get_matrix_indices(lat, lon)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        data_value_map = {1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 9, 7: 10, 8: 14, 9: 15}

        data_LULC = self.lulc_data.copy()

        for i in range(1, 10):
            data_LULC[data_LULC == data_value_map[i]] = i

        levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        clrs = [
            "greenyellow",
            "mediumpurple",
            "turquoise",
            "plum",
            "black",
            "blue",
            "yellow",
            "mediumseagreen",
            "forestgreen",
        ]
        cmap, norm = colors.from_levels_and_colors(levels, clrs)

        im1 = ax1.imshow(data_LULC, cmap=cmap, norm=norm)
        ax1.axhline(y=y, color="r", linestyle="--", alpha=0.5)
        ax1.axvline(x=x, color="r", linestyle="--", alpha=0.5)
        ax1.plot(x, y, "r*", markersize=10)
        ax1.set_title("Original Matrix")
        ax1.set_xticks([])
        ax1.set_yticks([])

        cbar = plt.colorbar(
            im1, ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9], fraction=0.046, pad=0.04, ax=ax1
        )
        cbar.ax.set_yticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        cbar.ax.set_yticklabels(
            [
                "Deciduous Broadleaf Forest",
                "Built-up Land",
                "Mixed Forest",
                "Shrubland",
                "Barren Land",
                "Water Bodies",
                "Plantations",
                "Grassland",
                "Broadleaf evergreen forest",
            ]
        )

        if interpolated is not None:

            interpolated_data = interpolated.copy()

            for i in range(1, 10):
                interpolated_data[interpolated_data == data_value_map[i]] = i

            scale_y = interpolated_data.shape[0] / self.lulc_data.shape[0]
            scale_x = interpolated_data.shape[1] / self.lulc_data.shape[1]
            y_interp = int(y * scale_y)
            x_interp = int(x * scale_x)

            im2 = ax2.imshow(interpolated_data, cmap=cmap, norm=norm)
            ax2.axhline(y=y_interp, color="r", linestyle="--", alpha=0.5)
            ax2.axvline(x=x_interp, color="r", linestyle="--", alpha=0.5)
            ax2.plot(x_interp, y_interp, "r*", markersize=10)
            ax2.set_title(f"Interpolated Matrix")
            ax2.set_xticks([])
            ax2.set_yticks([])

            cbar = plt.colorbar(
                im2, ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9], fraction=0.046, pad=0.04, ax=ax2
            )
            cbar.ax.set_yticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9])
            cbar.ax.set_yticklabels(
                [
                    "Deciduous Broadleaf Forest",
                    "Built-up Land",
                    "Mixed Forest",
                    "Shrubland",
                    "Barren Land",
                    "Water Bodies",
                    "Plantations",
                    "Grassland",
                    "Broadleaf evergreen forest",
                ]
            )

        plt.tight_layout()
        plt.savefig(
            "game_theory_codes/game_rewards/outputs/interpolated_LULC_matrix_value_at_coordinatex_x_"
            + str(x)
            + "_y_"
            + str(y)
            + ".png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def assign_defender_target_rewards_and_penalties_random(
        self, target_value=10, interpolated=None
    ):
        """Assign targetID, rewards and penalties for landuse cells"""
        if interpolated is None:
            raise ValueError("Interpolated matrix required")

        target_cells = np.where(interpolated == target_value)
        n_targets = len(target_cells[0])

        df = pd.DataFrame(
            {
                "targetID": range(1, n_targets + 1),
                "row": target_cells[0],
                "col": target_cells[1],
                "reward": np.random.uniform(1, 10, n_targets),
                "penalty": np.random.uniform(-10, 0, n_targets),
            }
        )

        df.to_csv(
            "game_theory_codes/game_rewards/outputs/defender_rewards_penalties.csv",
            index=False,
        )
        return df

    def assign_attacker_target_rewards_and_penalties_random(
        self, target_value=10, interpolated=None
    ):
        """Assign targetID, rewards and penalties for landuse cells"""
        if interpolated is None:
            raise ValueError("Interpolated matrix required")

        target_cells = np.where(interpolated == target_value)
        n_targets = len(target_cells[0])

        df = pd.DataFrame(
            {
                "targetID": range(1, n_targets + 1),
                "row": target_cells[0],
                "col": target_cells[1],
                "reward": np.random.uniform(1, 10, n_targets),
                "penalty": np.random.uniform(-10, 0, n_targets),
            }
        )

        df.to_csv(
            "game_theory_codes/game_rewards/outputs/attacker_rewards_penalties.csv",
            index=False,
        )
        return df

    def plot_with_rewards(self, targets_df, interpolated=None, name=None):
        """Plot matrices with rewards and penalties as text overlays"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        if interpolated is not None:

            data_value_map = {1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 9, 7: 10, 8: 14, 9: 15}
            interpolated_data = interpolated.copy()

            for i in range(1, 10):
                interpolated_data[interpolated_data == data_value_map[i]] = i

            levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
            clrs = [
                "greenyellow",
                "mediumpurple",
                "turquoise",
                "plum",
                "black",
                "blue",
                "yellow",
                "mediumseagreen",
                "forestgreen",
            ]
            cmap, norm = colors.from_levels_and_colors(levels, clrs)

            im1 = ax1.imshow(interpolated_data, cmap=cmap, norm=norm)
            im2 = ax2.imshow(interpolated_data, cmap=cmap, norm=norm)

            for _, row in targets_df.iterrows():
                ax1.text(
                    row["col"],
                    row["row"],
                    f'{row["reward"]:.0f}',
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=4,
                )

                ax2.text(
                    row["col"],
                    row["row"],
                    f'{row["penalty"]:.0f}',
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=4,
                )

            ax1.set_title("Rewards Distribution")
            ax2.set_title("Penalties Distribution")

            cbar = plt.colorbar(
                im1, ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9], fraction=0.046, pad=0.04, ax=ax1
            )
            cbar.ax.set_yticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9])
            cbar.ax.set_yticklabels(
                [
                    "Deciduous Broadleaf Forest",
                    "Built-up Land",
                    "Mixed Forest",
                    "Shrubland",
                    "Barren Land",
                    "Water Bodies",
                    "Plantations",
                    "Grassland",
                    "Broadleaf evergreen forest",
                ]
            )

            cbar2 = plt.colorbar(
                im2, ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9], fraction=0.046, pad=0.04, ax=ax2
            )
            cbar2.ax.set_yticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9])
            cbar2.ax.set_yticklabels(
                [
                    "Deciduous Broadleaf Forest",
                    "Built-up Land",
                    "Mixed Forest",
                    "Shrubland",
                    "Barren Land",
                    "Water Bodies",
                    "Plantations",
                    "Grassland",
                    "Broadleaf evergreen forest",
                ]
            )

            plt.tight_layout()
            plt.savefig(
                "game_theory_codes/game_rewards/outputs/"
                + name
                + "_rewards_penalties.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def check_target_interception(
        self,
        latitudes,
        longitudes,
        defender_payoff,
        attacker_payoff,
        interpolated=None,
        name=None,
        make_trajectory_plots=False,
        save_location=None
    ):
        """Check if given coordinates intercept any targets and return rewards/penalties"""
        results = []

        for lat, lon in zip(latitudes, longitudes):

            y, x = self.get_matrix_indices(lat, lon)
            if interpolated is not None:
                scale_y = interpolated.shape[0] / self.lulc_data.shape[0]
                scale_x = interpolated.shape[1] / self.lulc_data.shape[1]
                y = int(y * scale_y)
                x = int(x * scale_x)

            target_defender = defender_payoff[
                (defender_payoff["row"] == y) & (defender_payoff["col"] == x)
            ]
            target_attacker = attacker_payoff[
                (attacker_payoff["row"] == y) & (attacker_payoff["col"] == x)
            ]

            if not target_defender.empty and not target_attacker.empty:  #if the elephant trajectory has intercepted a target

                results.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "targetID": target_defender.iloc[0]["targetID"],
                        "defender_reward": 0,
                        "defender_penalty": target_defender.iloc[0]["penalty"],
                        "attacker_reward": target_attacker.iloc[0]["reward"],
                        "attacker_penalty": 0,
                    }
                )

            else:     
                results.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "targetID": None,
                        "defender_reward": None,
                        "defender_penalty": None,
                        "attacker_reward": None,
                        "attacker_penalty": None,
                    }
                )

        strategy_df = pd.DataFrame(results)

        strategy_df.to_csv(
            os.path.join(
                save_location,
                "strategy_"
                + name
                + "_rewards_and_penalties.csv",
            ),
            index=False,
        )

        if make_trajectory_plots:

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.yaxis.set_inverted(True)

            ds = gdal.Open(
                os.path.join(
                    "mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"
                )
            )
            data_LULC = interpolated.copy()
            data_LULC = np.flip(data_LULC, axis=0)

            data_value_map = {1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 9, 7: 10, 8: 14, 9: 15}

            for i in range(1, 10):
                data_LULC[data_LULC == data_value_map[i]] = i

            row_size, col_size = ds.ReadAsArray().shape
            xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

            outProj, inProj = Proj(init="epsg:4326"), Proj(init="epsg:3857")
            LON_MIN, LAT_MIN = transform(inProj, outProj, xmin, ymax + yres * col_size)
            LON_MAX, LAT_MAX = transform(inProj, outProj, xmin + xres * row_size, ymax)

            map = Basemap(
                llcrnrlon=LON_MIN,
                llcrnrlat=LAT_MIN,
                urcrnrlon=LON_MAX,
                urcrnrlat=LAT_MAX,
                epsg=4326,
                resolution="l",
            )

            levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
            clrs = [
                "greenyellow",
                "mediumpurple",
                "turquoise",
                "plum",
                "black",
                "blue",
                "yellow",
                "mediumseagreen",
                "forestgreen",
            ]
            cmap, norm = colors.from_levels_and_colors(levels, clrs)

            map.imshow(
                data_LULC,
                cmap=cmap,
                norm=norm,
                extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                alpha=0.5,
            )

            map.drawmeridians(
                [
                    LON_MIN,
                    (LON_MIN + LON_MAX) / 2 - (LON_MAX - LON_MIN) * 1 / 4,
                    (LON_MIN + LON_MAX) / 2,
                    (LON_MIN + LON_MAX) / 2 + (LON_MAX - LON_MIN) * 1 / 4,
                    LON_MAX,
                ],
                labels=[0, 1, 0, 1],
            )
            map.drawparallels(
                [
                    LAT_MIN,
                    (LAT_MIN + LAT_MAX) / 2 - (LAT_MAX - LAT_MIN) * 1 / 4,
                    (LAT_MIN + LAT_MAX) / 2,
                    (LAT_MIN + LAT_MAX) / 2 + (LAT_MAX - LAT_MIN) * 1 / 4,
                    LAT_MAX,
                ],
                labels=[1, 0, 1, 0],
            )

            cbar = plt.colorbar(
                ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9], fraction=0.046, pad=0.04
            )
            cbar.ax.set_yticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9])
            cbar.ax.set_yticklabels(
                [
                    "Deciduous Broadleaf Forest",
                    "Built-up Land",
                    "Mixed Forest",
                    "Shrubland",
                    "Barren Land",
                    "Water Bodies",
                    "Plantations",
                    "Grassland",
                    "Broadleaf evergreen forest",
                ]
            )

            outProj, inProj = Proj(init="epsg:4326"), Proj(init="epsg:3857")
            x_new, y_new = transform(inProj, outProj, longitudes, latitudes)

            C = np.arange(len(x_new))
            nz = mcolors.Normalize()
            nz.autoscale(C)

            ax.quiver(
                x_new[:-1],
                y_new[:-1],
                x_new[1:] - x_new[:-1],
                y_new[1:] - y_new[:-1],
                scale_units="xy",
                angles="xy",
                scale=1,
                zorder=1,
                color=cm.jet(nz(C)),
                width=0.0025,
                alpha=0.5,
            )

            ax.scatter(x_new[0], y_new[0], 25, marker="o", color="black", zorder=2)
            ax.scatter(x_new[-1], y_new[-1], 25, marker="^", color="black", zorder=2)

            plt.savefig(
                os.path.join(
                    save_location,
                    "strategy_" + name + "_.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )

        return strategy_df

    def read_ranger_locations(self, path=os.path.join("trajectory_analysis/ranger-locations/random_ranger_strategies_3guards.yaml")):

        def read_experiment_data(yaml_file_path):
            try:
                with open(yaml_file_path, 'r') as file:
                    data = yaml.safe_load(file)
                    
                for experiment in data:
                    experiment['ranger_locations'] = np.array(experiment['ranger_locations'])
                    experiment['ranger_payoffs'] = np.array(experiment['ranger_payoffs'])
                    
                return data
            
            except Exception as e:
                print(f"Error reading YAML file: {e}")
                return None
            
        experiments = read_experiment_data(path)
        converged_experiments = [exp for exp in experiments if exp['convergence']]
        best_experiment = min(converged_experiments, key=lambda x: x['total_cost'])
        self.ranger_locations = best_experiment['ranger_locations']
        
    def find_first_visible_entry(self, trajectory, ranger_visibility_radius):
         
        first_visible_entries = None
    
        for idx, row in trajectory.iterrows():
            for ranger_location in self.ranger_locations:
                distance = ((row['longitude'] - ranger_location[0])**2 + 
                            (row['latitude'] - ranger_location[1])**2)**0.5
                
                if distance <= ranger_visibility_radius:
                    first_visible_entries = idx
                    return first_visible_entries
                else:
                    first_visible_entries = None

        return first_visible_entries
    
    def get_elephant_agent_attributes(self, trajectory, idx):
        return trajectory.iloc[idx]






def run_abm_and_return_cost(model_params, ranger_locations, experiment_name):

    raster_path = os.path.join(
        "mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"
    )

    assign_rewards_and_penalties = LandUseRewards(raster_path)

    interpolated = assign_rewards_and_penalties.interpolate_matrix((50, 50))
    assign_rewards_and_penalties.plot_matrices(interpolated)

    lon = 8571327
    lat = 1045605
    value = assign_rewards_and_penalties.get_value_at_coords(lat, lon, interpolated)
    assign_rewards_and_penalties.plot_coords(lat, lon, interpolated)

    targets_df_defender = assign_rewards_and_penalties.assign_defender_target_rewards_and_penalties_random(
        target_value=10, interpolated=interpolated
    )
    assign_rewards_and_penalties.plot_with_rewards(
        targets_df_defender, interpolated, name="defender"
    )

    targets_df_attacker = assign_rewards_and_penalties.assign_attacker_target_rewards_and_penalties_random(
        target_value=10, interpolated=interpolated
    )
    assign_rewards_and_penalties.plot_with_rewards(
        targets_df_attacker, interpolated, name="attacker"
    )

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

    OUTPUT_FOLDER_withoutrangers = output_folder
    
    path = pathlib.Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)

    run_abm_without_rangers(
        experiment_name, model_params, output_folder
    )
    
    strategy_output_folder = os.path.join(os.getcwd(), "model_runs/", experiment_name, "guard_agent_placement_optimisation", "without_rangers")

    strategy = {
        'id': 1,
        'ranger_locations': ranger_locations,
        'total_cost': 0,
        'convergence': True
    }

    path = pathlib.Path(strategy_output_folder)
    path.mkdir(parents=True, exist_ok=True)

    save_strategies_to_yaml(strategy, os.path.join(strategy_output_folder, "ranger_strategies_" + str(model_params["num_guards"]) + "rangers.yaml"))

    subfolders = [f.path for f in os.scandir(output_folder) if f.is_dir()]

    for subfolder in subfolders:

        subfolder_name = pathlib.Path(subfolder).parts[-1]

        output_folder = os.path.join(
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
            str(model_params["month"]),
            subfolder_name
        )

        OUTPUT_FOLDER_withrangers = pathlib.Path(output_folder).parent

        output_file_path = os.path.join(subfolder, "output_files", "agent_data.csv")
        output_file = pd.read_csv(output_file_path)

        elephant_agent_data = output_file[output_file["AgentID"] == "bull_0"]
        latitudes = elephant_agent_data["latitude"]
        longitudes = elephant_agent_data["longitude"]

        path = pathlib.Path(output_folder)
        path.mkdir(parents=True, exist_ok=True)
        
        strategy_rewards_and_penalties = assign_rewards_and_penalties.check_target_interception(
            latitudes,
            longitudes,
            targets_df_defender,
            targets_df_attacker,
            interpolated=interpolated,
            name=subfolder_name,
            make_trajectory_plots=True,
            save_location=output_folder
        )

        assign_rewards_and_penalties.read_ranger_locations()

        first_entry = assign_rewards_and_penalties.find_first_visible_entry(elephant_agent_data, model_params["ranger_visibility_radius"])

        if first_entry != None and int(model_params["max_time_steps"] - first_entry) > 0:

            elephant_agent_state = assign_rewards_and_penalties.get_elephant_agent_attributes(elephant_agent_data, first_entry)
            run_state = {"elephant_fitness": float(elephant_agent_state["fitness"]),
                            "elephant_mode": str(elephant_agent_state["mode"]),
                            "daily_dry_matter_intake": float(elephant_agent_state["daily_dry_matter_intake"]),
                            "food_consumed": float(elephant_agent_state["food_consumed"]),
                            "visit_water_source": bool(elephant_agent_state["visit_water_source"]),
                            "num_days_water_source_visit": int(elephant_agent_state["num_days_water_source_visit"]),
                            "num_days_food_depreceation": int(elephant_agent_state["num_days_food_depreceation"]),
                            "elephant_starting_latitude":  float(elephant_agent_state["latitude"]),
                            "elephant_starting_longitude":  float(elephant_agent_state["longitude"]),
                            "restart": True,
                            "max_time_steps": int(model_params["max_time_steps"] - first_entry)
                            }
            
            run_abm_given_state_and_ranger_locations(experiment_name=experiment_name,
                                                        model_params=model_params, 
                                                        output_folder=output_folder, 
                                                        strategy=subfolder_name, 
                                                        run_folder=subfolder,
                                                        kwargs=run_state)

            subfolders = [f.path for f in os.scandir(output_folder) if f.is_dir()]

            for subfolder in subfolders:

                subfolder_name = pathlib.Path(subfolder).parts[-1]

                output_file_path = os.path.join(subfolder, "output_files", "agent_data.csv")
                output_file = pd.read_csv(output_file_path)

                elephant_agent_data = output_file[output_file["AgentID"] == "bull_0"]
                latitudes = elephant_agent_data["latitude"]
                longitudes = elephant_agent_data["longitude"]

                strategy_rewards_and_penalties = assign_rewards_and_penalties.check_target_interception(
                    latitudes,
                    longitudes,
                    targets_df_defender,
                    targets_df_attacker,
                    interpolated=interpolated,
                    name=subfolder_name,
                    make_trajectory_plots=True,
                    save_location=output_folder
                )

        else:
            print("Trajectory did not intercept the rangers!")
            pass

    NUM_TARGETS = 438

    attacker_rewards_df = pd.read_csv("game_theory_codes/game_rewards/outputs/attacker_rewards_penalties.csv")
    defender_rewards_df = pd.read_csv("game_theory_codes/game_rewards/outputs/defender_rewards_penalties.csv")

    cost = return_cost(attacker_rewards_df, defender_rewards_df, OUTPUT_FOLDER_withoutrangers, OUTPUT_FOLDER_withrangers, NUM_TARGETS)

    print("COST:", cost)

    return cost


def fitness(population, model_params, experiment_name, generation_ID):

    fitness_scores = np.zeros(len(population))
    
    for i in range(len(population)):
        ranger_locations = population[i]
        _experiment_name_ = experiment_name + "__" + str(generation_ID) + "__" + str(i)
        fitness_scores[i] = -run_abm_and_return_cost(model_params, ranger_locations, _experiment_name_)

    return fitness_scores


def plot_optimization_results(history):
   
   plt.figure(figsize=(6, 6))
   plt.plot(history['best'], label='Best Fitness')
   plt.plot(history['average'], label='Average Fitness')
   plt.plot(history['worst'], label='Worst Fitness')
   plt.xlabel('Generation')
   plt.ylabel('Fitness')
   plt.legend()
   plt.title('Optimization Progress')
   plt.grid(True)
   plt.show()












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
            "num_processes": 2,
            "iterations": 2,
            "max_time_steps": 288 * 5,
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
    
    generator = FancyNameGenerator()
    run_name = generator.generate_name()
    experiment_name = "ranger-deployment-within-plantations-GA-optimisation/" + run_name

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

    optimizer = RangerOptimizer(
        num_rangers=model_params["num_guards"],
        data_folder=output_folder,
        population_size=4,
        generations=4
    )

    population = optimizer.initialize_population()
    best_solution = None
    best_fitness = float('-inf')

    for generation_ID, generation in enumerate(range(optimizer.generations)):
            
        fitness_scores = fitness(population, model_params, experiment_name, generation_ID)
        
        optimizer.history['best'].append(np.max(fitness_scores))
        optimizer.history['worst'].append(np.min(fitness_scores))
        optimizer.history['average'].append(np.mean(fitness_scores))
        
        current_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > best_fitness:
            best_fitness = fitness_scores[current_best_idx]
            best_solution = population[current_best_idx].copy()
        
        parents = optimizer.select_parents(population, fitness_scores)
        offspring = optimizer.crossover(parents)
        population = optimizer.mutate(offspring)

    print(best_fitness, best_solution)

    plot_optimization_results(optimizer.history)
