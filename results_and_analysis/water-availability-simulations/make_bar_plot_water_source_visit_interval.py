
import pandas as pd
import matplotlib.pyplot as plt
import os

#set fontsize using rcParams
plt.rcParams.update({'font.size': 12})

base_folder = "model_runs/water-availability-simulations/experiment-v1.1/crimson_08-29-25__10-13/latitude-1049237-longitude-8570917/solitary_bulls/landscape-food-probability-forest-0.1-cropland-0.1/"

water_hole_scenarios = ["waterholes-within-landscape-0.0001",
                        "waterholes-within-landscape-0.001",
                        "waterholes-within-landscape-0.01",
                        "waterholes-within-landscape-0.1"]

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
            
            runs = [run for run in runs if os.path.isdir(os.path.join(folder_path, run))]

            i = 0
            found_run = False
            while found_run == False:
                run = runs[i]
                df = pd.read_csv(f"{folder_path}{run}/output_files/agent_data.csv")
                num_days_water_source_visit = df['num_days_water_source_visit'].dropna().values
                num_days_water_source_visit = num_days_water_source_visit[num_days_water_source_visit != 0]

                if num_days_water_source_visit.size == 0:
                    num_days_water_source_visit = [0]*25

                data_to_plot.append(num_days_water_source_visit)
                labels.append(scenario1.replace("waterholes-within-landscape-", ""))
                found_run = True
                i += 1

        if data_to_plot:
            plt.boxplot(data_to_plot, positions=range(1, len(data_to_plot) + 1), widths=0.625, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='blue'),
                        medianprops=dict(color='red', linewidth=1),
                        whiskerprops=dict(color='blue', linewidth=1),
                        capprops=dict(color='blue'),
                        flierprops=dict(markerfacecolor='blue', marker='o', markersize=12, linestyle='none'))
            
            plt.xticks(range(1, len(labels) + 1), labels, rotation=0, ha='center')
            plt.xlabel("Water Hole Density")
            plt.ylabel("Number of Days Between \nWater Source Visits")
            #set x-tick labels
            plt.xticks([1, 2, 3, 4], ['0.01%', '0.1%', '1%', '10%'])
            plt.grid(True)
            plt.tight_layout()
            plt.ylim(0, 30)
            filename = f"results_and_analysis/water-availability-simulations/num_days_water_source_visit_boxplot_{scenario2}_{scenario3}.png"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            pass