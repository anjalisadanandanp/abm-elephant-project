from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import os
import pandas as pd
from pyproj import Proj, transform    
from mpl_toolkits.basemap import Basemap    
import matplotlib.colors as mcolors   


import warnings
warnings.filterwarnings("ignore")




boundary_patches = gdal.Open("game_theory_codes/FPL-UE/create-strategy-set/outputs/high_res_indexed_forest_agricultural_fringe.tif").ReadAsArray()

rainbow = plt.cm.rainbow
colors = rainbow(np.linspace(0, 1, 256))
colors[0] = [1, 1, 1, 1]  
custom_cmap = matplotlib.colors.ListedColormap(colors)

fig, ax = plt.subplots(figsize=(8, 8))
img = ax.imshow(boundary_patches, cmap=custom_cmap)

ax.set_xticks([])
ax.set_yticks([])

plt.colorbar(img, shrink=0.5)

plt.savefig("create-boundary-agricultural-patch-association-matrix/boundary_patches.png", dpi=750, bbox_inches="tight")

plt.close()


run_folder = "model_runs/model-without-intervention/latitude-1049237-longitude-8570917/solitary_bulls/random-food-distribition-within-plantation/landscape-food-probability-forest-0.1-cropland-0.1/water-source-rivers-landscape-1.0/random-memory-forest-and_plantation-fringe-model/full-memory-forest-and_plantation-model/num_days_agent_survives_in_deprivation-10/maximum-food-in-a-forest-cell-5/thermoregulation-threshold-temperature-28/threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/slope_tolerance-30/num_days_agent_survives_in_deprivation-10/elephant_aggression_value_0.2/2010/Mar"

expts = os.listdir(run_folder)

agricultural_plts = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").ReadAsArray()

geotransform = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").GetGeoTransform()
ag_xmin, ag_xres, ag_xskew, ag_ymax, ag_yskew, ag_yres = geotransform
ag_rows, ag_cols = agricultural_plts.shape

landuse_matrix = gdal.Open(os.path.join(run_folder, expts[-1], "env", "LULC.tif")).ReadAsArray()
row_size, col_size = landuse_matrix.shape
xmin, xres, xskew, ymax, yskew, yres = gdal.Open(os.path.join(run_folder, expts[-1], "env", "LULC.tif")).GetGeoTransform()
outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

def lat_lon_to_pixel(lats, lons, xmin, ymax, xres, yres):
    """Convert lat/lon coordinates to pixel coordinates"""
    rows = ((ymax - lats) / -yres).astype(int)
    cols = ((lons - xmin) / xres).astype(int)
    return rows, cols

def find_cropland_use_indices(landuse_sequence):

    indices = []
    start = None
    
    for i, value in enumerate(landuse_sequence):
        if value == 10 or value == 9:
            if start is None:
                start = i
        else:
            if start is not None:
                indices.append((start, i-1))
                start = None
    
    if start is not None:
        indices.append((start, len(landuse_sequence)-1))
    
    return indices

plot_landuse_map = False
plot_agricultural_plot_map = False
plot_boundary_patch_map = False
num_cropraiding_steps = 12


num_boundary_patchs = int(np.max(boundary_patches))
num_agricultural_plts = int(np.max(agricultural_plts))

association_matrix = np.zeros((num_boundary_patchs, num_agricultural_plts))

print("shape of association matrix:", association_matrix.shape)

for expt in expts:

    if plot_landuse_map == True:

        fig, ax = plt.subplots(figsize=(8, 8))

        map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        img = map.imshow(np.flipud(landuse_matrix), cmap = "Pastel2", extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 1)

        map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
        map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

        cb = fig.colorbar(img) 
        cb.remove()

        df = pd.read_csv(os.path.join(run_folder, expt, "output_files/agent_data.csv"))

        rows, cols = lat_lon_to_pixel(
            df["latitude"].values, df["longitude"].values, 
            ag_xmin, ag_ymax, ag_xres, ag_yres
        )
        
        landuse_values = landuse_matrix[rows, cols]

        indices = find_cropland_use_indices(landuse_values)

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
        longitude, latitude = transform(inProj, outProj, df["longitude"], df["latitude"])
        x_new, y_new = map(longitude,latitude)

        ax.plot(x_new, y_new, linewidth=0.25, alpha=0.5, color="black", zorder=1)

        ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='red', zorder=1)
        ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='red', zorder=1)

        for sequence in indices:

            if (sequence[1] - sequence[0]) >= num_cropraiding_steps:
                
                outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
                longitude, latitude = transform(inProj, outProj, df["longitude"][sequence[0]:sequence[1]], df["latitude"][sequence[0]:sequence[1]])
                x_new, y_new = map(longitude,latitude)

                ax.plot(x_new, y_new, linewidth=0.25, zorder=2)
                
                ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='black', zorder=2)
                ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='black', zorder=2)

        plt.savefig("create-boundary-agricultural-patch-association-matrix/outputs/cropland_intersection_traj_" + expt + "_.png", dpi=500, bbox_inches="tight")
        plt.close()

    if plot_agricultural_plot_map == True:

        fig, ax = plt.subplots(figsize=(8, 8))

        map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        colors = ['white', 'cyan']
        cmap_two_colors = mcolors.ListedColormap(colors)

        img = map.imshow(np.flipud(agricultural_plts), cmap = cmap_two_colors, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 1)

        map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
        map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

        cb = fig.colorbar(img) 
        cb.remove()

        df = pd.read_csv(os.path.join(run_folder, expt, "output_files/agent_data.csv"))

        rows, cols = lat_lon_to_pixel(
            df["latitude"].values, df["longitude"].values, 
            ag_xmin, ag_ymax, ag_xres, ag_yres
        )
        
        landuse_values = landuse_matrix[rows, cols]

        indices = find_cropland_use_indices(landuse_values)

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
        longitude, latitude = transform(inProj, outProj, df["longitude"], df["latitude"])
        x_new, y_new = map(longitude,latitude)

        ax.plot(x_new, y_new, linewidth=0.25, alpha=0.5, color="black", zorder=1)

        ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='red', zorder=1)
        ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='red', zorder=1)

        for sequence in indices:

            if (sequence[1] - sequence[0]) >= num_cropraiding_steps:
                
                outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
                longitude, latitude = transform(inProj, outProj, df["longitude"][sequence[0]:sequence[1]], df["latitude"][sequence[0]:sequence[1]])
                x_new, y_new = map(longitude,latitude)

                ax.plot(x_new, y_new, linewidth=0.25, zorder=2)
                
                ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='black', zorder=2)
                ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='black', zorder=2)

        plt.savefig("create-boundary-agricultural-patch-association-matrix/outputs/agri_plots_intersection_traj_" + expt + "_.png", dpi=500, bbox_inches="tight")
        plt.close()

    if plot_boundary_patch_map == True:

        fig, ax = plt.subplots(figsize=(8, 8))

        map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        rainbow = plt.cm.rainbow
        colors = rainbow(np.linspace(0, 1, 256))
        colors[0] = [1, 1, 1, 1]  
        custom_cmap = matplotlib.colors.ListedColormap(colors)

        img = map.imshow(np.flipud(boundary_patches), cmap = custom_cmap, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 1)

        map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
        map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

        cb = fig.colorbar(img) 
        cb.remove()

        df = pd.read_csv(os.path.join(run_folder, expt, "output_files/agent_data.csv"))

        rows, cols = lat_lon_to_pixel(
            df["latitude"].values, df["longitude"].values, 
            ag_xmin, ag_ymax, ag_xres, ag_yres
        )
        
        landuse_values = landuse_matrix[rows, cols]

        indices = find_cropland_use_indices(landuse_values)

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
        longitude, latitude = transform(inProj, outProj, df["longitude"], df["latitude"])
        x_new, y_new = map(longitude,latitude)

        ax.plot(x_new, y_new, linewidth=0.25, alpha=0.5, color="black", zorder=1)

        ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='red', zorder=1)
        ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='red', zorder=1)

        for sequence in indices:

            if (sequence[1] - sequence[0]) >= num_cropraiding_steps:
                
                outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
                longitude, latitude = transform(inProj, outProj, df["longitude"][sequence[0]:sequence[1]], df["latitude"][sequence[0]:sequence[1]])
                x_new, y_new = map(longitude,latitude)

                ax.plot(x_new, y_new, linewidth=0.25, zorder=2)
                
                ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='black', zorder=2)
                ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='black', zorder=2)

        plt.savefig("create-boundary-agricultural-patch-association-matrix/outputs/boundary_patch_intersection_traj_" + expt + "_.png", dpi=500, bbox_inches="tight")
        plt.close()

    try:
        df = pd.read_csv(os.path.join(run_folder, expt, "output_files/agent_data.csv"))

        rows, cols = lat_lon_to_pixel(
            df["latitude"].values, df["longitude"].values, 
            ag_xmin, ag_ymax, ag_xres, ag_yres
        )
        
        landuse_values = landuse_matrix[rows, cols]

        indices = find_cropland_use_indices(landuse_values)
        
        for sequence in indices:

            if (sequence[1] - sequence[0]) >= num_cropraiding_steps:
                
                for i in range(sequence[1] - sequence[0]):
                    if agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]] != 0:
                        association_matrix[int(boundary_patches[rows[sequence[0]], cols[sequence[0]]]),  int(agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]])] += 1
    except:
        pass
    
print(np.min(association_matrix), np.max(association_matrix))

df = pd.DataFrame(association_matrix)
df.to_csv("association_mtarix.csv")