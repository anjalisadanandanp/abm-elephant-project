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

            if not target_defender.empty and not target_attacker.empty:

                results.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "targetID": target_defender.iloc[0]["targetID"],
                        "defender_reward": target_defender.iloc[0]["reward"],
                        "defender_penalty": target_defender.iloc[0]["penalty"],
                        "attacker_reward": attacker_payoff.iloc[0]["reward"],
                        "attacker_penalty": attacker_payoff.iloc[0]["penalty"],
                    }
                )

            elif not target_defender.empty:
                results.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "targetID": target_defender.iloc[0]["targetID"],
                        "defender_reward": target_defender.iloc[0]["reward"],
                        "defender_penalty": target_defender.iloc[0]["penalty"],
                        "attacker_reward": None,
                        "attacker_penalty": None,
                    }
                )

            elif not target_attacker.empty:
                results.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "targetID": target_attacker.iloc[0]["targetID"],
                        "defender_reward": None,
                        "defender_penalty": None,
                        "attacker_reward": attacker_payoff.iloc[0]["reward"],
                        "attacker_penalty": attacker_payoff.iloc[0]["penalty"],
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
                os.getcwd(),
                "game_theory_codes/game_rewards/outputs/strategy_"
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

            print(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

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
                    os.getcwd(),
                    "game_theory_codes/game_rewards/outputs/strategy_" + name + "_.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )

        return strategy_df


if __name__ == "__main__":

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
        "max_time_steps": 288 * 10,
        "aggression_threshold_enter_cropland": 1.0,
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
        "elephant_crop_habituation": False,
        "num_guards": 3,
        "ranger_visibility_radius": 500,
    }

    experiment_name = "ranger-deployment-within-plantations"

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

    data_folder = os.path.join(
        os.getcwd(),
        "model_runs",
        "exploratory-search-ID-01",
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
    )

    subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]

    subfolders = np.random.choice(subfolders, 10, replace=False)

    for subfolder in subfolders:

        output_file_path = os.path.join(subfolder, "output_files", "agent_data.csv")
        output_file = pd.read_csv(output_file_path)

        elephant_agent_data = output_file[output_file["AgentID"] == "bull_0"]
        latitudes = elephant_agent_data["latitude"]
        longitudes = elephant_agent_data["longitude"]

        assign_rewards_and_penalties.check_target_interception(
            latitudes,
            longitudes,
            targets_df_defender,
            targets_df_attacker,
            interpolated=interpolated,
            name=pathlib.Path(subfolder).parts[-1],
            make_trajectory_plots=True
        )
