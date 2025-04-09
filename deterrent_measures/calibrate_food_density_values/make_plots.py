import numpy as np
from osgeo import gdal
import rasterio as rio
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import product
from tqdm import tqdm



class food_density_values():

    def __init__(self, prob_food_in_forest, prob_food_in_cropland, max_food_val_forest, max_food_val_cropland, percent_memory_elephant, output_folder):

        self.prob_food_in_forest =  prob_food_in_forest
        self.prob_food_in_cropland = prob_food_in_cropland
        self.max_food_val_forest = max_food_val_forest 
        self.max_food_val_cropland = max_food_val_cropland 
        self.percent_memory_elephant = percent_memory_elephant
        self.output_folder = output_folder

    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def initialize_food_matrix(self):
        folder_path = os.path.join("mesageo_elephant_project/elephant_project/", "experiment_setup_files","environment_seethathode","Raster_Files_Seethathode_Derived", "area_1100sqKm/reso_30x30")
        fid = os.path.join(folder_path, "LULC.tif")

        Plantation = gdal.Open(fid).ReadAsArray()

        self.LANDUSE = Plantation
        self.row_size, self.col_size = Plantation.shape

        m,n=Plantation.shape

        food_matrix = np.zeros_like(Plantation)
        landscape_cell_status = np.zeros_like(Plantation)

        # for i in range(0,m):
        #     for j in range(0,n):
        #         if np.random.uniform(0,1) < self.prob_food_in_cropland and Plantation[i,j] == 10:
        #             landscape_cell_status[i,j] = 2

        #         elif np.random.uniform(0,1) < self.prob_food_in_forest and Plantation[i,j] == 15:   
        #             landscape_cell_status[i,j] = 1

        # forest_mask = (Plantation == 15) & (landscape_cell_status == 1)
        # cropland_mask = (Plantation == 10) & (landscape_cell_status == 2)


        random_vals_cropland = np.random.uniform(0, 1, size=(m, n))
        random_vals_forest = np.random.uniform(0, 1, size=(m, n))
        
        cropland_mask = (random_vals_cropland < self.prob_food_in_cropland) & (Plantation == 10)
        forest_mask = (random_vals_forest < self.prob_food_in_forest) & (Plantation == 15)
        
        landscape_cell_status[cropland_mask] = 2
        landscape_cell_status[forest_mask] = 1
    

        food_matrix[forest_mask] = np.random.uniform(0, self.max_food_val_forest + 1, size=(m,n))[forest_mask]
        food_matrix[cropland_mask] = np.random.uniform(0, self.max_food_val_cropland + 1, size=(m,n))[cropland_mask]

        self.FOOD = food_matrix

        # fid = os.path.join(folder_path, "LULC.tif")

        # with rio.open(fid) as src:
        #     ras_data = src.read()
        #     ras_meta = src.profile

        # ras_meta['dtype'] = "float64"
        # ras_meta['nodata'] = -99

        # fid = os.path.join(self.output_folder, "food_matrix_" + str(self.prob_food_in_forest) + "_" + str(self.max_food_val_forest) + "_" + str(self.prob_food_in_cropland) + "_" + str(self.max_food_val_cropland) + "_.tif")

        # with rio.open(fid, 'w', **ras_meta) as dst:
        #     dst.write(food_matrix.astype(float), 1)

        # fid = os.path.join(self.output_folder, "landscape_cell_status_" + str(self.prob_food_in_forest) + "_" + str(self.max_food_val_forest) + "_" + str(self.prob_food_in_cropland) + "_" + str(self.max_food_val_cropland) + ".tif")
        # with rio.open(fid, 'w', **ras_meta) as dst:
        #     dst.write(landscape_cell_status.astype(float), 1)

        return 
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def initialize_food_memory_matrix_random(self):
        """ Function that assigns memory matrix to elephants"""

        food_memory=np.zeros_like(self.FOOD)
        food_memory_cells=np.zeros_like(self.FOOD)

        # for i in range(0,self.row_size):
        #     for j in range(0,self.col_size):
        #         if random.uniform(0,1) < self.percent_memory_elephant:
        #             food_memory[i,j] = self.FOOD[i][j]
        #             if self.FOOD[i][j] > 0:
        #                 food_memory_cells[i,j] = 1

        memory_mask = np.random.random(self.FOOD.shape) < self.percent_memory_elephant
        food_memory[memory_mask] = self.FOOD[memory_mask]
        positive_memory_mask = memory_mask & (self.FOOD > 0)
        food_memory_cells[positive_memory_mask] = 1


        self.food_memory = food_memory.tolist() 
        self.food_memory_cells = food_memory_cells.tolist()

        # folder_path = os.path.join("mesageo_elephant_project/elephant_project/", "experiment_setup_files","environment_seethathode","Raster_Files_Seethathode_Derived", "area_1100sqKm/reso_30x30")
        # source = os.path.join(folder_path, "LULC.tif")
        # with rio.open(source) as src:
        #     ras_meta = src.profile

        # memory_loc = os.path.join(self.output_folder, "food_memory_random_" + str(self.prob_food_in_forest) + "_" + str(self.max_food_val_forest) + "_" + str(self.prob_food_in_cropland) + "_" + str(self.max_food_val_cropland) + "_.tif")
        # with rio.open(memory_loc, 'w', **ras_meta) as dst:
        #     dst.write(food_memory_cells.astype('float32'), 1)

        return 
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def plot_food_matrices(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        im1 = axes[0].imshow(self.FOOD, cmap='Greys')
        axes[0].set_title('Food Matrix')
        fig.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
        
        im2 = axes[1].imshow(self.food_memory, cmap='Greys')
        axes[1].set_title('Food Memory Matrix')
        fig.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        plt.savefig( os.path.join(self.output_folder, "food_matrix_plot_" + str(self.prob_food_in_forest) + "_" + str(self.max_food_val_forest) + "_" + str(self.prob_food_in_cropland) + "_" + str(self.max_food_val_cropland) + "_.png"), dpi=300, bbox_inches='tight')
        
        return fig, axes
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def main(self):

        self.initialize_food_matrix()
        self.initialize_food_memory_matrix_random()
        # self.plot_food_matrices()

        food_memory_array = np.array(self.food_memory)
        lulc_array = np.array(self.LANDUSE)

        plantation_mask = (lulc_array == 10)
        forest_mask = (lulc_array == 15)

        sum_food_where_lulc_10 = np.sum(food_memory_array[plantation_mask])
        num_cells_with_lulc_10 = np.count_nonzero(lulc_array == 10)
        sum_food_where_lulc_15 = np.sum(food_memory_array[forest_mask])
        num_cells_with_lulc_15 = np.count_nonzero(lulc_array == 15)

        forest_food_density = sum_food_where_lulc_15/num_cells_with_lulc_15
        cropland_food_density = sum_food_where_lulc_10/num_cells_with_lulc_10

        return forest_food_density, cropland_food_density
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------








def analyze_parameter_effects(output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    results = []
    
    prob_food_vals = np.round(np.linspace(0, 1, 11), 2)
    max_food_vals = np.round(np.linspace(0, 100, 11), 2)
    memory_vals = np.round(np.linspace(0, 1, 11), 2)


    for prob_forest, prob_crop, max_food_val_forest, max_food_val_cropland, memory_pct in tqdm(product(prob_food_vals, prob_food_vals, max_food_vals, max_food_vals, memory_vals)):

        print(prob_forest, prob_crop, max_food_val_forest, max_food_val_cropland, memory_pct)

        analyzer = food_density_values(
            prob_food_in_forest=prob_forest,
            prob_food_in_cropland=prob_crop,
            max_food_val_forest=max_food_val_forest,
            max_food_val_cropland=max_food_val_cropland,
            percent_memory_elephant=memory_pct,
            output_folder=output_folder
        )
        
        forest_density, cropland_density = analyzer.main()
        
        if forest_density > 0:
            density_ratio = cropland_density / forest_density
        else:
            density_ratio = float('inf') if cropland_density > 0 else 0
            
        results.append({
            'prob_food_in_forest': prob_forest,
            'prob_food_in_cropland': prob_crop,
            'max_food_val_forest': max_food_val_forest,
            'max_food_val_cropland': max_food_val_cropland,
            'percent_memory_elephant': memory_pct,
            'forest_density': forest_density,
            'cropland_density': cropland_density,
            'density_ratio': density_ratio
        })
        
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_folder, "parameter_analysis_results.csv")
    df.to_csv(csv_path, index=False)
    
    return df

def create_analysis_plots(output_folder):

    csv_path = os.path.join(output_folder, "parameter_analysis_results.csv")
    results_df = pd.read_csv(csv_path)

    unique_max_forest_food_values = results_df["max_food_val_forest"].unique()
    unique_max_cropland_food_values = results_df["max_food_val_cropland"].unique()

    for max_food_forest in unique_max_forest_food_values:
        for max_food_cropland in unique_max_cropland_food_values:

            plt.figure()
            
            df = results_df[
                (results_df['max_food_val_forest'] == max_food_forest) &
                    (results_df['max_food_val_cropland'] == max_food_cropland)
            ]
            
            memory_values = sorted(df['percent_memory_elephant'].unique())
            num_plots = len(memory_values)
            
            if num_plots > 0:

                try:
                    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(5, 25))
                    axes = axes.flatten()
                    
                    for i, mem_val in enumerate(memory_values):
                        mem_df = df[df['percent_memory_elephant'] == mem_val]
                        pivot = mem_df.pivot_table(
                            values='density_ratio',
                            index='prob_food_in_forest', 
                            columns='prob_food_in_cropland'
                        )

                        print(pivot)
                        
                        sns.heatmap(pivot, annot=True, cmap='coolwarm', ax=axes[i], vmin=0, vmax=1)
                        axes[i].set_title(f'Memory Percentage: {mem_val:.2f}')
                        axes[i].set_xlabel('Probability of Food in Cropland')
                        axes[i].set_ylabel('Probability of Food in Forest')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, "density_ratio_heatmap_by_memory_maxforestfood" + str(max_food_forest) + "_maxcroplandfood" + str(max_food_cropland) + "_.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                except:
                    pass

# analyse_food_matrix_initialisation = food_density_values(prob_food_in_forest=0.05, 
#                                                          prob_food_in_cropland=0.05, 
#                                                          max_food_val_forest=10, 
#                                                          max_food_val_cropland=10, 
#                                                          percent_memory_elephant=0.375,
#                                                          output_folder=output_folder)

# print(analyse_food_matrix_initialisation.main())

output_folder = "deterrent_measures/calibrate_food_density_values/outputs"
analyze_parameter_effects(output_folder)
create_analysis_plots(output_folder)