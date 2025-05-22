import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from tqdm import tqdm
from matplotlib.colors import ListedColormap

landuse = gdal.Open("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif").ReadAsArray()
geotransform = gdal.Open("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif").GetGeoTransform()

df = pd.read_excel("data/seethathode_data/seethathodu_data.xlsx")
df = df[df["Area under rubber (cent)"] > 0]
df = df[df["Area under rubber (cent)"] < 1000]

df = df.reset_index(drop=True)

areas_sqm = df["Area under rubber (cent)"] * 40.47  

mask = landuse == 10
plantation_cells = np.sum(mask)
cell_area = geotransform[1] * geotransform[1]  
plantation_area = plantation_cells * cell_area

print(f"Number of plantation cells: {plantation_cells}")
print(f"Total plantation area: {plantation_area / 1e6:.2f} sq. km")
print(f"Total area from dataframe: {sum(areas_sqm) / 1e6:.2f} sq. km")
print(f"Average plot size in dataframe: {np.mean(areas_sqm):.2f} sq. meters")

agricultural_plots = np.zeros_like(landuse, dtype=np.int32)

plantation_indices = np.where(landuse == 10)
plantation_coords = list(zip(plantation_indices[0], plantation_indices[1]))

random.shuffle(plantation_coords)

def get_valid_neighbors(y, x, agricultural_plots, landuse):
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue 
            
            ny, nx = y + dy, x + dx
            
            if 0 <= ny < landuse.shape[0] and 0 <= nx < landuse.shape[1]:
                if landuse[ny, nx] == 10 and agricultural_plots[ny, nx] == 0:
                    neighbors.append((ny, nx))
    
    return neighbors

assigned_areas = []

plot_id = 1
remaining_coords = plantation_coords.copy()

remaining_coords = random.sample(remaining_coords, 5000)

while tqdm(remaining_coords):

    target_area_sqm = random.choice(areas_sqm)
    target_cells = max(1, int(round(target_area_sqm / cell_area)))

    if target_cells < 3:
        pass

    else:

        if not remaining_coords:
            break
        
        start_y, start_x = remaining_coords.pop(0)
        
        if agricultural_plots[start_y, start_x] != 0:
            continue
        
        current_plot_cells = [(start_y, start_x)]
        agricultural_plots[start_y, start_x] = plot_id
        
        current_size = 1
        frontier = [(start_y, start_x)]
        
        while current_size < target_cells and frontier:

            current_y, current_x = frontier.pop(0)
            
            neighbors = get_valid_neighbors(current_y, current_x, agricultural_plots, landuse)
            
            random.shuffle(neighbors)
            
            for ny, nx in neighbors:
                if current_size >= target_cells:
                    break
                    
                agricultural_plots[ny, nx] = plot_id
                current_size += 1

                if (ny, nx) in remaining_coords:
                    remaining_coords.remove((ny, nx))
                    
                frontier.append((ny, nx))
                current_plot_cells.append((ny, nx))
        
        actual_area = current_size * cell_area
        assigned_areas.append(actual_area)

        if current_size < 3:
            mask = agricultural_plots == plot_id
            agricultural_plots[mask] = 0
        
        else:
            print(f"Plot {plot_id}: Target area = {target_area_sqm:.2f} sqm ({target_cells} cells), "
            f"Actual area = {actual_area:.2f} sqm ({current_size} cells)")
            plot_id += 1

assigned_cells = np.sum(agricultural_plots > 0)
assigned_percentage = (assigned_cells / plantation_cells) * 100

print(f"\nAssignment complete.")
print(f"Number of plots created: {plot_id - 1}")
print(f"Cells assigned: {assigned_cells} out of {plantation_cells} ({assigned_percentage:.2f}%)")

plt.figure(figsize=(8, 4.8))

plt.subplot(1, 2, 1)
plt.hist(areas_sqm, bins=10, color="dodgerblue", alpha=0.7)
plt.title("Original Area Distribution")
plt.xlabel("Area (sq. m)")
plt.ylabel("Count")
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(assigned_areas, bins=10, color="green", alpha=0.7)
plt.title("Assigned Area Distribution")
plt.xlabel("Area (sq. m)")
plt.ylabel("Count")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("create-landholding-matrix/area_distribution_comparison.png", dpi=300, bbox_inches="tight")

plot_sizes = defaultdict(int)
for i in range(1, plot_id):
    plot_sizes[i] = np.sum(agricultural_plots == i)

avg_plot_size = np.mean(list(plot_sizes.values())) * cell_area
median_plot_size = np.median(list(plot_sizes.values())) * cell_area
min_plot_size = np.min(list(plot_sizes.values())) * cell_area
max_plot_size = np.max(list(plot_sizes.values())) * cell_area

print(f"\nPlot size statistics (sq. meters):")
print(f"Average: {avg_plot_size:.2f}")
print(f"Median: {median_plot_size:.2f}")
print(f"Min: {min_plot_size:.2f}")
print(f"Max: {max_plot_size:.2f}")



def save_as_geotiff(array, reference_file, output_file):
    """Save a numpy array as a GeoTIFF with the same georeference as the reference file"""

    reference_ds = gdal.Open(reference_file)
    
    geotransform = reference_ds.GetGeoTransform()
    projection = reference_ds.GetProjection()
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_file,
        array.shape[1],  # width
        array.shape[0],  # height
        1,               # number of bands
        gdal.GDT_Int32   # data type
    )
    
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    
    out_ds = None
    reference_ds = None
    
    print(f"GeoTIFF saved to {output_file}")

output_tif = "create-landholding-matrix/agricultural_plots_assignment.tif"
reference_file = "mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"
save_as_geotiff(agricultural_plots, reference_file, output_tif)

