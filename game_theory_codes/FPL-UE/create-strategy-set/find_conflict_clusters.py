import os
import pandas as pd
from osgeo import gdal
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

output_dir = "conflict_cluster_analysis"
os.makedirs(output_dir, exist_ok=True)

output_folder = "game_theory_codes/FPL-UE/create-strategy-set/simulations-without-mitigation/latitude-1049237-longitude-8570917/solitary_bulls/landscape-food-probability-forest-0.1-cropland-0.1/water-holes-within-landscape-0.0/random-memory-matrix-model/num_days_agent_survives_in_deprivation-10/maximum-food-in-a-forest-cell-10/thermoregulation-threshold-temperature-28/threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/slope_tolerance-30/num_days_agent_survives_in_deprivation-10/elephant_aggression_value_0.8/2010/Mar/game_step_0/"
potential_targets_path = "game_theory_codes/FPL-UE/create-strategy-set/outputs/potential_targets_matrix.tif"

potential_targets_ds = gdal.Open(potential_targets_path)
potential_targets = potential_targets_ds.ReadAsArray()

ny, nx = potential_targets.shape
print(f"Potential targets matrix dimensions: {ny} rows x {nx} columns")

ROWS = []
COLS = []

for simulation_folder in os.listdir(output_folder):
    try:
        df = pd.read_csv(os.path.join(output_folder, simulation_folder, "output_files/agent_data.csv"))
        df.dropna(subset=['ROW', 'COL'], inplace=True)
    
        rows = df['ROW'].astype(int).values 
        cols = df['COL'].astype(int).values

        ROWS.extend(rows)
        COLS.extend(cols)
    
    except Exception as e:
        print(f"Error processing {simulation_folder}: {e}")

target_indices = np.where(potential_targets == 1)
target_rows = target_indices[0]
target_cols = target_indices[1]

target_positions = list(zip(target_rows, target_cols))
agent_positions = list(zip(ROWS, COLS))

conflict_positions = []

for agent_position in agent_positions:
    for target_position in target_positions:
        if agent_position == target_position:
            conflict_positions.append(agent_position)

print(f"Number of conflict positions: {len(conflict_positions)}")


if conflict_positions:
    conflict_array = np.array(conflict_positions)
else:
    print("No conflict positions found!")
    exit()


plt.figure(figsize=(8, 8))
plt.imshow(potential_targets, cmap='Greys', alpha=0.5, origin='upper')
plt.colorbar(label='Potential targets', shrink=0.5)
plt.scatter(conflict_array[:, 1], conflict_array[:, 0], color="red", s=25, alpha=0.8)
plt.xlabel('Column')
plt.ylabel('Row')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, f"conflict_locations.png"), dpi=300)
plt.close()



eps_value = 5 
min_samples_value = 25

db = DBSCAN(eps=eps_value, min_samples=min_samples_value).fit(conflict_array)
labels = db.labels_


n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f'Estimated number of clusters: {n_clusters}')
print(f'Estimated number of noise points: {n_noise}')




plt.figure(figsize=(8, 8))
plt.imshow(potential_targets, cmap='Greys', alpha=0.5, origin='upper')
plt.colorbar(label='Potential targets', shrink=0.5)

plt.title('Conflict Clusters with DBSCAN')

colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
for i, color in zip(range(n_clusters), colors):
    cluster_points = conflict_array[labels == i]
    plt.scatter(cluster_points[:, 1], cluster_points[:, 0], color=color, s=50, alpha=0.8, label=f'Cluster {i+1}')

plt.legend(loc='best')
plt.xlabel('Column')
plt.ylabel('Row')
plt.grid(True, alpha=0.3)

plt.savefig(os.path.join(output_dir, f"conflict_clusters_eps{eps_value}_minsamples{min_samples_value}.png"), dpi=300)
plt.close()

if n_clusters > 0:
    print("\nCluster Statistics:")
    for i in range(n_clusters):
        cluster_size = np.sum(labels == i)
        cluster_points = conflict_array[labels == i]
        
        center_row = np.mean(cluster_points[:, 0])
        center_col = np.mean(cluster_points[:, 1])
        
        std_row = np.std(cluster_points[:, 0])
        std_col = np.std(cluster_points[:, 1])
        
        print(f"Cluster {i+1}:")
        print(f"  Size: {cluster_size} points")
        print(f"  Center: Row={center_row:.2f}, Col={center_col:.2f}")
        print(f"  Spread: σ_Row={std_row:.2f}, σ_Col={std_col:.2f}")
        print(f"  Density: {cluster_size / (np.pi * std_row * std_col):.4f} points/unit²")
        print()
