import os                               # for file operations
import numpy as np                      # for numerical operations
from osgeo import gdal                  # for raster operations
from pyproj import Proj, transform              # for coordinate transformations
import matplotlib.pyplot as plt                 # for plotting
from mpl_toolkits.basemap import Basemap        # for plotting
from matplotlib import colors                   # for plotting
import matplotlib.cm as cm                      # for plotting
import matplotlib.colors as mcolors             # for plotting
import pandas as pd                            # for data handling
from tqdm import tqdm    

import warnings
warnings.filterwarnings("ignore")

base_folder = os.path.join("/mnt/qdata/abm-elephant-project/aryabhata-runs/model-with-deterrance-policy")

ds = gdal.Open(os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif"))
data = ds.ReadAsArray()
data = np.flip(data, axis=0)

data[700:800, 500:650] = 15

row_size, col_size = data.shape
xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

data_value_map = {1:1, 2:3, 3:4, 4:5, 5:6, 6:9, 7:10, 8:14, 9:15}

for i in range(1,10):
    data[data == data_value_map[i]] = i

fig, ax = plt.subplots(figsize = (6,6))
ax.yaxis.set_inverted(True)

outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
clrs = ["greenyellow","mediumpurple","turquoise", "plum", "black", "blue", "yellow", "mediumseagreen", "forestgreen"] 
cmap, norm = colors.from_levels_and_colors(levels, clrs)

map.imshow(data, cmap = cmap, norm=norm, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 0.5, zorder=1)

map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

cbar = plt.colorbar(ticks=[1,2,3,4,5,6,7,8,9],fraction=0.046, pad=0.04)
cbar.ax.set_yticks(ticks=[1,2,3,4,5,6,7,8,9]) 
cbar.ax.set_yticklabels(["Deciduous Broadleaf Forest","Built-up Land","Mixed Forest","Shrubland","Barren Land","Water Bodies","Plantations","Grassland","Broadleaf evergreen forest"])


plt.savefig(os.path.join("results_and_analysis/deterrance_policy_models/model_with_intervention", "lulc_map.png"), dpi = 300, bbox_inches = 'tight')

plt.close()