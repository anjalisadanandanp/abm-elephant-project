import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12})

base_folder = "model_runs/water-availability-simulations/experiment-v1.1/nova_08-28-25__10-31/latitude-1049237-longitude-8570917/solitary_bulls/landscape-food-probability-forest-0.1-cropland-0.1/"

water_hole_scenarios = ["rivers-within-landscape-0.01",
                        "rivers-within-landscape-0.1",
                        "rivers-within-landscape-0.5",
                        "rivers-within-landscape-1.0"]

subfolder_path_first = "random-memory-matrix-model/num_days_agent_survives_in_deprivation-10/maximum-food-in-a-forest-cell-5"

thermoregulation_scenarios = ["thermoregulation-threshold-temperature-28",
                                "thermoregulation-threshold-temperature-32"]

subfolder_path_second = "threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/slope_tolerance-30/num_days_agent_survives_in_deprivation-10/"

aggression_scenarios = ["elephant_aggression_value_0.2",
                        "elephant_aggression_value_0.8"]

subfolder_path_third = "2010/Mar"

for scenario2 in thermoregulation_scenarios:
    for scenario3 in aggression_scenarios:

        plt.figure(figsize=(4.8, 3.2))

        data_to_plot = []
        labels = []

        for scenario1 in water_hole_scenarios:
            folder_path = f"{base_folder}{scenario1}/{subfolder_path_first}/{scenario2}/{subfolder_path_second}/{scenario3}/{subfolder_path_third}/"
            
            try:
                runs = os.listdir(folder_path)
            except FileNotFoundError:
                print(f"Directory not found: {folder_path}")
                continue
            
            run_dirs = [os.path.join(folder_path, run) for run in runs if os.path.isdir(os.path.join(folder_path, run))]
            
            all_scenario_data = []

            for run in run_dirs:
                try:
                    df = pd.read_csv(f"{run}/output_files/agent_data.csv")
                    num_days_water_source_visit = df['num_days_water_source_visit'].dropna().values

                    valid_data = num_days_water_source_visit[num_days_water_source_visit != 0]

                    if (scenario1 == "rivers-within-landscape-1.0" and scenario2 == "thermoregulation-threshold-temperature-28") or (scenario1 == "rivers-within-landscape-0.5" and scenario2 == "thermoregulation-threshold-temperature-28"):
                        valid_data = valid_data[valid_data <= 5]

                    elif scenario1 == "rivers-within-landscape-0.1" and scenario2 == "thermoregulation-threshold-temperature-28":
                        valid_data = valid_data[valid_data <= 8]

                    elif scenario1 == "rivers-within-landscape-0.01" and scenario2 == "thermoregulation-threshold-temperature-28":
                        valid_data = valid_data[valid_data <= 15]


                    elif (scenario1 == "rivers-within-landscape-1.0" and scenario2 == "thermoregulation-threshold-temperature-32") or (scenario1 == "rivers-within-landscape-0.5" and scenario2 == "thermoregulation-threshold-temperature-32"):
                        valid_data = valid_data[valid_data <= 8]

                    elif scenario1 == "rivers-within-landscape-0.1" and scenario2 == "thermoregulation-threshold-temperature-32":
                        valid_data = valid_data[valid_data <= 12]

                    elif scenario1 == "rivers-within-landscape-0.01" and scenario2 == "thermoregulation-threshold-temperature-32":
                        valid_data = valid_data[valid_data <= 25]


                    if valid_data.size > 0:
                        all_scenario_data.extend(valid_data)
                except FileNotFoundError:
                    print(f"agent_data.csv not found in run directory: {run}")
                
            if all_scenario_data:
                data_to_plot.append(all_scenario_data)
                labels.append(scenario1.replace("waterholes-within-landscape-", ""))

        if data_to_plot:
            plt.boxplot(data_to_plot, positions=range(1, len(data_to_plot) + 1), widths=0.625, patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor='lightblue', color='blue'),
                        medianprops=dict(color='red', linewidth=1),
                        whiskerprops=dict(color='blue', linewidth=1),
                        capprops=dict(color='blue'),
                        flierprops=dict(markerfacecolor='blue', marker='o', markersize=12, linestyle='none'))
            
            plt.xticks(range(1, len(labels) + 1), labels, rotation=0, ha='center')
            plt.xlabel("River Water Availability")
            plt.ylabel("Number of Days Between \nWater Source Visits")
            
            plt.xticks([1, 2, 3, 4], ['1%', '10%', '50%', '100%'])
            plt.grid(True)
            plt.tight_layout()
            plt.ylim(0, 26)
            
            filename = f"results_and_analysis/water-availability-simulations/num_days_water_source_visit_boxplot_{scenario2}_{scenario3}.png"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("No data found for plotting.")