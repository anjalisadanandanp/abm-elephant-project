import os
import itertools

def find_folders_without_agent(root_dir):

    folders_to_clean = []
    
    data_folder = os.path.join(os.getcwd(), root_dir)
    
    if not os.path.exists(data_folder):
        # print(data_folder)
        # print("Folder does not exist")
        # print("\n")
        pass
    else:
        # print(data_folder)
        # print("Folder exists")
        # print("\n")

        #find all subfolders in the data folder
        subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
        # print(subfolders)
        # print("\n")
        
        #check if the subfolders contain the agent.csv file
        for subfolder in subfolders:
            if os.path.exists(os.path.join(subfolder, "output_files/agent_data.csv")):
                # print(subfolder)
                # print("Agent data file exists")
                # print("\n")
                pass
            else:
                # print(subfolder)
                # print("Agent data file does not exist")
                folders_to_clean.append(subfolder)
                
    return folders_to_clean



folders_to_clean = find_folders_without_agent("guarding-policies/static-guarding-model-v1/model-runs/exploratory_search_on_evading_trajectories/game_model-v3_1-run-model-without-intervention-v2/mitigation-measures-within-plantations-FPL-UE_v3_1/latitude-[[1052166]]-longitude-[[8572829]]/solitary_bulls/random-food-distribition-within-agricultural-plots-and-other-plantation-cells/landscape-food-probability-forest-0.1-cropland-1.0/water-source-rivers-landscape-0.05/random-memory-forest-and_plantation-fringe-model/full-memory-forest-and_plantation-model/num_days_agent_survives_in_deprivation-10/maximum-food-in-a-forest-cell-25/thermoregulation-threshold-temperature-28/threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/slope_tolerance-35/num_days_agent_survives_in_deprivation-10/elephant_aggression_value_0.8/2010/Mar/protected_targets_0/num_strategic_traj_188_num_iterations_47/game_step_1")

#write folders to clean to a text file
with open("folders_to_clean.txt", "w") as f:
    for folder in folders_to_clean:
        f.write(folder + "\n")
