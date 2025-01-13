#------------importing libraries----------------#
import os

import sys
sys.path.append(os.getcwd())

import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np 
from pyproj import Proj, transform 
from mpl_toolkits.basemap import Basemap 
import matplotlib.colors as mcolors 
import matplotlib.cm as cm   
from matplotlib import colors

import warnings 
warnings.filterwarnings('ignore') 
#------------importing libraries----------------#






class analyse_trajectories():

    def __init__(self, source_folder):
        self.source_folder = source_folder
        self.create_output_folder_structure()

    def read_all_experiments(self):
    
        path = pathlib.Path(os.path.join(os.getcwd(), "model_runs", self.source_folder))
        subfolders = [x for x in path.iterdir() if x.is_dir()]
        self.file_paths = [str(subfolder / "output_files/agent_data.csv") for subfolder in subfolders]

    def create_output_folder_structure(self):

        self.output_folder = os.path.join(os.getcwd(), "trajectory_analysis/outputs", self.source_folder)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, mode=0o777)

    def rank_order_trajectories(self, save_dataframe= False):

        data = []
        for file_path in self.file_paths:
            try:
                df = pd.read_csv(file_path)
                length_rank = len(df)
                last_fitness = df['fitness'].iloc[-1] 
                data.append({
                    'file_path': file_path,
                    'length_rank': length_rank,
                    'fitness_rank': last_fitness
                })
            except FileNotFoundError:
                print(f"Warning: File not found: {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        df_ranks = pd.DataFrame(data)
        df_ranks['length_rank'] = df_ranks['length_rank'].rank(method='dense', ascending=False)
        df_ranks['fitness_rank'] = df_ranks['fitness_rank'].rank(method='dense', ascending=False) 

        self.sorted_df = df_ranks.sort_values(by=['length_rank', 'fitness_rank'], ascending=[True, True]) 

        if save_dataframe == True:
            self.sorted_df.to_csv(os.path.join(self.output_folder, "ordered_experiments.csv"))

        return
    
    def filter_data(self, experiments, start_step, end_step, create_plots = True):

        if create_plots == True:

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
            
        for experiment in range(experiments):
            path = self.sorted_df["file_path"].iloc[experiment]
            data = pd.read_csv(os.path.join(str(path)))
            data = data.iloc[start_step:end_step]

            if create_plots == True:

                outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
                longitude, latitude = transform(inProj, outProj, data["longitude"], data["latitude"])
                x_new, y_new = map(longitude,latitude)
                C = np.arange(len(x_new))
                nz = mcolors.Normalize()
                nz.autoscale(C)

                ax.quiver(x_new[:-1], y_new[:-1],
                            x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1], 
                            scale_units='xy', angles='xy', 
                            scale=1, zorder=1, color = cm.jet(nz(C)), 
                            width=0.0025)
                
                ax.scatter(x_new[0], y_new[0], 25, marker='o', color='blue', zorder=2)
                ax.scatter(x_new[-1], y_new[-1], 25, marker='^', color='red', zorder=2)

        if create_plots == True:
            plt.title("Elephant agent trajectories")
            plt.savefig(os.path.join(self.output_folder, "best_trajectories.png"), dpi = 300, bbox_inches = 'tight')
            plt.close()

    def main(self):
        
        self.read_all_experiments()
        self.rank_order_trajectories(save_dataframe=True)
        self.filter_data(experiments=4, start_step=0, end_step = 2880, create_plots=True)




