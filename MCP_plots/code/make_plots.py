from scipy.spatial import ConvexHull
from pyproj import Proj, transform
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import pandas as pd
from osgeo import gdal
from matplotlib import colors  

def plot_ele_traj_on_proximity_to_water_sources(input_folder, output_folder):

    ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
    data = ds.ReadAsArray()
    data = np.flip(data, axis=0)
    row_size, col_size = data.shape
    xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

    data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

    for i in range(1,10):
        data[data == data_value_map[i]] = i
        
    fig, ax = plt.subplots(figsize = (4,4))
    ax.yaxis.set_inverted(True)

    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
    LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
    LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

    map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

    levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
    clrs = ["greenyellow","mediumpurple","turquoise", "plum", "black", "blue", "yellow", "mediumseagreen", "forestgreen"] 
    cmap, norm = colors.from_levels_and_colors(levels, clrs)

    map.imshow(data, cmap = cmap, norm=norm, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 0.5)

    folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

    for folder in folders:
        agent_data = pd.read_csv(os.path.join(input_folder, folder, "output_files/agent_data.csv"))
        longitude = agent_data["longitude"]
        latitude = agent_data["latitude"]

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        longitude, latitude = transform(inProj, outProj, longitude, latitude)

        x_new, y_new = map(longitude, latitude)
        C = np.arange(len(x_new))
        nz = mcolors.Normalize()
        nz.autoscale(C)

        points = np.column_stack((x_new, y_new))
        hull = ConvexHull(points)
        
        hull_vertices = points[hull.vertices]
        
        hull_vertices = np.vstack((hull_vertices, hull_vertices[0]))
        
        ax.plot(hull_vertices[:, 0], hull_vertices[:, 1], 'k-', linewidth=1, label='Minimum Convex Polygon')
        ax.fill(hull_vertices[:, 0], hull_vertices[:, 1], alpha=0.5, color='grey')

        C = np.arange(len(x_new))
        nz = mcolors.Normalize()
        nz.autoscale(C)

        ax.plot(x_new, y_new, color="red", linewidth=0.5, alpha = 0.8)

    plt.savefig(os.path.join(output_folder, "trajectory_on_LULC_map.png"), dpi = 300, bbox_inches = 'tight')
    plt.close()

plot_ele_traj_on_proximity_to_water_sources("model_runs/water-availability-simulations/water-holes-within-simulation/dragon_03-29-25__16-17/latitude-1049237-longitude-8570917/solitary_bulls/landscape-food-probability-forest-0.1-cropland-0.1/water-holes-within-landscape-0.1/only-forest-memory-model/num_days_agent_survives_in_deprivation-10/maximum-food-in-a-forest-cell-5/thermoregulation-threshold-temperature-28/threshold_days_of_food_deprivation-0/threshold_days_of_water_deprivation-3/slope_tolerance-30/num_days_agent_survives_in_deprivation-10/elephant_aggression_value_0.2/2010/Aug/",
                                            "MCP_plots/plots/")