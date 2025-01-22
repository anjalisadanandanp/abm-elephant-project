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
import pandas as pd
import numpy as np
from osgeo import gdal
from pyproj import Proj, transform
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import colors


from mesageo_elephant_project.elephant_project.model.abm_model_HEC_v3 import batch_run_model
from trajectory_analysis.assign_payoff_rangers.optimise_ranger_locations import optimise_ranger_locations
#------------importing libraries----------------#




optimisation_args = {
    "max_optimisation_steps": 4,
    "num_best_trajs": 8,

}




model_params_all = {
    "year": 2010,
    "month": ["Mar"],
    "num_bull_elephants": 1, 
    "area_size": 1100,              
    "spatial_resolution": 30, 
    "max_food_val_cropland": 100,
    "max_food_val_forest": [10],
    "prob_food_forest": [0.10],
    "prob_food_cropland": [0.10],
    "prob_water_sources": [0.00],
    "thermoregulation_threshold": [28],
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
    "max_time_steps": 288*10,
    "aggression_threshold_enter_cropland": 1.0,
    "human_habituation_tolerance": 1.0,
    "elephant_agent_visibility_radius": 500,
    "plot_stepwise_target_selection": False,
    "threshold_days_of_food_deprivation": [0],
    "threshold_days_of_water_deprivation": [3],
    "number_of_feasible_movement_directions": 3,
    "track_in_mlflow": False,
    "elephant_starting_location": "user_input",
    "elephant_starting_latitude": 1049237,
    "elephant_starting_longitude": 8570917,
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


def run_model(experiment_name, model_params, output_folder):

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

    def evaluate_trajectories_v1(self, data_folder, model_params, step, make_plots=False):

        def read_experiment_data(yaml_file_path):

            yaml_file = next(pathlib.Path(yaml_file_path).glob("*.yaml"))

            try:
                # print(f"Reading YAML file: {yaml_file}")
                with open(yaml_file, 'r') as file:
                    data = yaml.safe_load(file)
                    
                for experiment in data:
                    experiment['ranger_locations'] = np.array(experiment['ranger_locations'])
                    experiment['ranger_payoffs'] = np.array(experiment['ranger_payoffs'])
                    
                return data
            
            except Exception as e:
                print(f"Error reading YAML file: {e}")
                return None

        def filter_trajectories_in_ranger_radius(trajectories, ranger_locations):
            """Filter trajectories that intersect with ranger visibility circles."""

            intersecting_trajs = []
            non_intersecting_trajs = []
            
            for data in trajectories:
                trajectory_intersects = False
                longitude, latitude = data["longitude"], data["latitude"]


                for ranger in ranger_locations:
                    ranger_lon, ranger_lat = ranger[0], ranger[1]
                    
                    for lon, lat in zip(longitude, latitude):
                        distance = np.sqrt((lon - ranger_lon)**2 + (lat - ranger_lat)**2)
                        
                        if distance <= model_params["ranger_visibility_radius"]:
                            trajectory_intersects = True
                            break
                            
                    if trajectory_intersects:
                        break
                        
                if trajectory_intersects:
                    intersecting_trajs.append(data)
                else:
                    non_intersecting_trajs.append(data)
                    
            return intersecting_trajs, non_intersecting_trajs

        def plot_filtered_trajectories(intersecting_trajs, non_intersecting_trajs):

            fig, ax = plt.subplots(figsize=(10,10))
            ax.yaxis.set_inverted(True)

            ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
            data_LULC = ds.ReadAsArray()
            data_LULC = np.flip(data_LULC, axis=0)

            data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

            for i in range(1,10):
                data_LULC[data_LULC == data_value_map[i]] = i

            row_size, col_size = data_LULC.shape
            xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
            
            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857') 
            LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
            LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

            map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

            levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
            clrs = ["greenyellow","mediumpurple","turquoise", "plum", "black", "blue", "yellow", "mediumseagreen", "forestgreen"] 
            cmap, norm = colors.from_levels_and_colors(levels, clrs)

            map.imshow(data_LULC, cmap = cmap, norm=norm, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 0.5)

            map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
            map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

            cbar = plt.colorbar(ticks=[1,2,3,4,5,6,7,8,9],fraction=0.046, pad=0.04)
            cbar.ax.set_yticks(ticks=[1,2,3,4,5,6,7,8,9]) 
            cbar.ax.set_yticklabels(["Deciduous Broadleaf Forest","Built-up Land","Mixed Forest","Shrubland","Barren Land","Water Bodies","Plantations","Grassland","Broadleaf evergreen forest"])
            
            for data in intersecting_trajs:
                longitude, latitude =  transform(inProj, outProj, data["longitude"], data["latitude"])
                x_new, y_new = map(longitude, latitude)
                
                ax.quiver(x_new[:-1], y_new[:-1],
                        x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1],
                        scale_units='xy', angles='xy',
                        scale=1, zorder=1, color='red', width=0.0025)
                
                ax.scatter(x_new[0], y_new[0], 25, marker='o', color='black')
                ax.scatter(x_new[-1], y_new[-1], 25, marker='^', color='black')

            for data in non_intersecting_trajs:
                longitude, latitude =  transform(inProj, outProj, data["longitude"], data["latitude"])
                x_new, y_new = map(longitude, latitude)
                
                ax.quiver(x_new[:-1], y_new[:-1],
                        x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1],
                        scale_units='xy', angles='xy',
                        scale=1, zorder=1, color='green', width=0.0025)
                
                ax.scatter(x_new[0], y_new[0], 25, marker='o', color='black')
                ax.scatter(x_new[-1], y_new[-1], 25, marker='^', color='black')

            for ranger in ranger_locations:
                longitude, latitude = transform(inProj, outProj, ranger[0], ranger[1])
                x_new, y_new = map(longitude,latitude)
                ax.scatter(x_new, y_new, 25, marker='x', color='white')
                
                radius = model_params["ranger_visibility_radius"]/(111*1000)  
                circle = plt.Circle((x_new, y_new), radius, facecolor='purple', fill=True, alpha=0.5, edgecolor='black', linewidth=1)
                ax.add_artist(circle)

            plt.savefig(os.path.join(pathlib.Path(data_folder).parent, "strategy_evaluation_files", 
                                     "step" + str(step) + "_trajectories_with_ranger_locations__numrangers" + str(model_params["num_guards"]) + "_rangervisibility" + str(model_params["ranger_visibility_radius"]) + "m_v2.png"), 
                                     dpi=300, bbox_inches='tight')

        def find_first_visible_entry(trajectory):
            
            first_visible_entries = None
        
            for idx, row in trajectory.iterrows():
                for ranger_location in ranger_locations:
                    distance = ((row['longitude'] - ranger_location[0])**2 + 
                                (row['latitude'] - ranger_location[1])**2)**0.5
                    
                    if distance <= model_params["ranger_visibility_radius"]:
                        first_visible_entries = idx
                        return first_visible_entries
                    else:
                        first_visible_entries = None

            return first_visible_entries

        def calculate_landuse_time(trajectory):
            """Calculate time steps spent in each landuse type."""

            ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
            data_LULC = ds.ReadAsArray()
            data_LULC = np.flip(data_LULC, axis=0)

            data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

            for i in range(1,10):
                data_LULC[data_LULC == data_value_map[i]] = i

            row_size, col_size = data_LULC.shape
            xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

            landuse_times = {
                1: 0,  # Deciduous Broadleaf Forest
                2: 0,  # Built-up Land
                3: 0,  # Mixed Forest 
                4: 0,  # Shrubland
                5: 0,  # Barren Land
                6: 0,  # Water Bodies
                7: 0,  # Plantations
                8: 0,  # Grassland
                9: 0   # Broadleaf evergreen forest
            }
            
            for lon, lat in zip(trajectory["longitude"], trajectory["latitude"]):
                # Convert coordinates to pixel indices
                px = int((lon - xmin) / xres)
                py = int((lat - ymax) / yres)
                
                # Get landuse type at location
                landuse = data_LULC[py, px]
                landuse_times[landuse] += 1

            total_time = sum(landuse_times.values())
            assert total_time == len(trajectory), "Total time steps don't match trajectory length"
    
            return landuse_times
    
        def plot_trajectories_untill_ranger_intervention():

            fig, ax = plt.subplots(figsize = (10,10))
            ax.yaxis.set_inverted(True)

            ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
            data_LULC = ds.ReadAsArray()
            data_LULC = np.flip(data_LULC, axis=0)

            data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

            for i in range(1,10):
                data_LULC[data_LULC == data_value_map[i]] = i

            row_size, col_size = data_LULC.shape
            xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
            
            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857') 
            LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
            LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

            map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

            levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
            clrs = ["greenyellow","mediumpurple","turquoise", "plum", "black", "blue", "yellow", "mediumseagreen", "forestgreen"] 
            cmap, norm = colors.from_levels_and_colors(levels, clrs)

            map.imshow(data_LULC, cmap = cmap, norm=norm, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 0.5)

            map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
            map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

            cbar = plt.colorbar(ticks=[1,2,3,4,5,6,7,8,9],fraction=0.046, pad=0.04)
            cbar.ax.set_yticks(ticks=[1,2,3,4,5,6,7,8,9]) 
            cbar.ax.set_yticklabels(["Deciduous Broadleaf Forest","Built-up Land","Mixed Forest","Shrubland","Barren Land","Water Bodies","Plantations","Grassland","Broadleaf evergreen forest"])

            for i, traj in enumerate(trajectories):

                first_visible_entry = find_first_visible_entry(traj)

                if first_visible_entry is not None:
                    data = traj.iloc[:first_visible_entry + 1]
                
                else:
                    data = traj

                outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
                longitude, latitude = transform(inProj, outProj, data["longitude"], data["latitude"])
                x_new, y_new = map(longitude,latitude)

                ax.plot(x_new, y_new, 
                        'b-', alpha=0.4, label='Trajectory')

                ax.scatter(x_new[0], y_new[0], 25, marker='o', color='black', zorder=2)
                ax.scatter(x_new[-1], y_new[-1], 25, marker='^', color='black', zorder=2)

            for ranger in ranger_locations:
                longitude, latitude = transform(inProj, outProj, ranger[0], ranger[1])
                x_new, y_new = map(longitude,latitude)

                ax.scatter(x_new, y_new, 25, marker='x', color='white', zorder=2)

                radius = model_params["ranger_visibility_radius"]/(111*1000)  
                circle = plt.Circle((x_new, y_new), radius, facecolor='purple', fill=True, alpha=0.5, edgecolor='black', linewidth=1)
                ax.add_artist(circle)

            plt.savefig(os.path.join(pathlib.Path(data_folder).parent, "strategy_evaluation_files", 
                                     "step" + str(step) + "_trajectories_until_ranger_proximity__rangervisibility" + str(model_params["ranger_visibility_radius"]) + "m_v1.png"), 
                                     dpi=300, bbox_inches='tight')

            return

        folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]

        trajectories = []

        for folder in folders:

            agent_data = pd.read_csv(os.path.join(data_folder, folder, "output_files/agent_data.csv"))
            agent_data = agent_data[agent_data["AgentID"] == "bull_0"].reset_index(drop=True)
            trajectories.append(agent_data)

            ranger_strategy = read_experiment_data(os.path.join(data_folder, folder, "env/ranger_strategy/"))
            converged_experiments = [exp for exp in ranger_strategy if exp['convergence']]
            best_experiment = min(converged_experiments, key=lambda x: x['total_cost'])
            ranger_locations = best_experiment['ranger_locations']

        strategy_evaluation_files = os.path.join(pathlib.Path(data_folder).parent, "strategy_evaluation_files")
        os.makedirs(strategy_evaluation_files, exist_ok=True)

        intersecting_trajs, non_intersecting_trajs = filter_trajectories_in_ranger_radius(trajectories, ranger_locations)
        
        if make_plots:
            plot_filtered_trajectories(intersecting_trajs, non_intersecting_trajs)
            plot_trajectories_untill_ranger_intervention()

        cost = 0
        for traj in trajectories:
            landuse_times = calculate_landuse_time(traj)[7]
            cost += landuse_times
        avg_cost = cost/len(trajectories)

        print("Number of intersecting trajectories:", len(intersecting_trajs), "Number of non-intersecting trajectories:", len(non_intersecting_trajs), "Average cost:", avg_cost)

        return avg_cost

    def run_experiment(self, send_notification):

        print("Running experiment...!")

        freeze_support()

        start = time.time()

        experiment_name = "ranger-deployment-v1"

        if model_params_all["track_in_mlflow"] == True:
            try:
                mlflow.create_experiment(experiment_name)
            except:
                print("experiment already exists")

        param_dicts = generate_parameter_combinations(model_params_all)

        for model_params in param_dicts:

            experiment_name = "ranger-deployment-v2"

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

            cost = []

            for step in range(1, optimisation_args["max_optimisation_steps"]+1):

                print("\nOptimising ranger locations! step:", step)

                data_folder = os.path.join(os.getcwd(), "model_runs", experiment_name, starting_location, elephant_category, landscape_food_probability, 
                                                water_holes_probability, memory_matrix_type, num_days_agent_survives_in_deprivation, maximum_food_in_a_forest_cell, 
                                                elephant_thermoregulation_threshold, threshold_food_derivation_days, threshold_water_derivation_days, 
                                                slope_tolerance, num_days_agent_survives_in_deprivation, elephant_aggression_value,
                                                str(model_params["year"]), str(model_params["month"]), "abm-runs-with-guard-agents", "step-" + str(step))




                #------------agent based model----------------#
                run_model(experiment_name, model_params, data_folder)
                #------------agent based model----------------#




                #-----evaluate the effectiveness of the ranger placement-----#
                cost.append(self.evaluate_trajectories_v1(data_folder, model_params, step, make_plots=True))
                #-----evaluate the effectiveness of the ranger placement-----#




                #------------ranger placement optimisation----------------#
                output_folder = os.path.join(os.getcwd(), "model_runs/", experiment_name, starting_location, elephant_category, landscape_food_probability, 
                                                water_holes_probability, memory_matrix_type, num_days_agent_survives_in_deprivation, maximum_food_in_a_forest_cell, 
                                                elephant_thermoregulation_threshold, threshold_food_derivation_days, threshold_water_derivation_days, 
                                                slope_tolerance, num_days_agent_survives_in_deprivation, elephant_aggression_value,
                                                str(model_params["year"]), str(model_params["month"]), "guard_agent_placement_optimisation")

                path = pathlib.Path(output_folder)
                path.mkdir(parents=True, exist_ok=True)

                optimizer = optimise_ranger_locations(num_best_trajs = optimisation_args["num_best_trajs"], 
                                                    num_rangers=model_params["num_guards"], 
                                                    ranger_visibility_radius=model_params["ranger_visibility_radius"], 
                                                    data_folder = data_folder, 
                                                    output_folder = output_folder)
                optimizer.optimize()

                optimizer.plot_trajectories_with_ranger_location()
                optimizer.plot_trajectories_untill_ranger_intervention()
                intersecting_trajs, non_intersecting_trajs = optimizer.filter_trajectories_in_ranger_radius()
                optimizer.plot_filtered_trajectories(intersecting_trajs, non_intersecting_trajs)

                optimizer.generate_ranger_strategies(starting_positions="kmeans", num_strategies = 1)
                #------------ranger placement optimisation----------------#

            #--------plotting the cost vs step------------#
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.plot(range(1, optimisation_args["max_optimisation_steps"]+1), cost, marker='o', color='blue', label='Cost')
            ax.set_xlabel('Step')
            ax.set_ylabel('Cost')
            ax.set_title('Cost vs Step')
            ax.legend()
            plt.savefig(os.path.join(pathlib.Path(data_folder).parent, "strategy_evaluation_files", "cost_vs_step.png"), dpi=300, bbox_inches='tight')
            plt.close()
            #--------plotting the cost vs step------------#

        end = time.time()

        print("\n Total time taken:", (end-start), "seconds")

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
    experiment.run_experiment(send_notification=False)
    
