##############################################################
#importing packages
##############################################################
from osgeo import gdal   
import rasterio as rio  
import numpy as np
import os

#to supress warnings
import warnings     
warnings.filterwarnings('ignore')  

from experiment_setup_files.Initialize_Conflict_Model_environment import environment
##############################################################



#To set up the environment for the simulation
class environment(environment):


    def __init__(self, area_size, resolution, Prob_food_in_forest, Prob_food_in_cropland, Prob_water):
        super().__init__(area_size, resolution) 
        self.Prob_food_in_forest =  Prob_food_in_forest
        self.Prob_food_in_cropland = Prob_food_in_cropland
        self.Prob_water = Prob_water


    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def initialize_food_matrix(self, Prob_food_in_forest, Prob_food_in_cropland):
        """Function returns a food matrix with values 0-num, 0 being no food avavilability and num being high food availability
        """

        folder_path = os.path.join("experiment_setup_files","environment_seethathode","Raster_Files_Seethathode_Derived", self.area[self.area_size])
        fid = os.path.join(folder_path, self.reso[self.resolution], "LULC.tif")

        Plantation = gdal.Open(fid).ReadAsArray()
        m,n=Plantation.shape

        food_matrix = np.zeros_like(Plantation)

        for i in range(0,m):
            for j in range(0,n):
                if Plantation[i,j]==15 or Plantation[i,j]==4:    #Evergreen broadleaf forest and mixed forests
                    if np.random.uniform(0,1) < Prob_food_in_forest:
                        food_matrix[i,j] = np.random.randint(1,5)

                elif Plantation[i,j]==10:   #plantations/croplands
                    if np.random.uniform(0,1) < Prob_food_in_cropland:
                        food_matrix[i,j] = np.random.randint(1,10)
         
        #saving the food matrix
        fid = os.path.join(folder_path, self.reso[self.resolution], "LULC.tif")

        with rio.open(fid) as src:
            ras_data = src.read()
            ras_meta = src.profile

        # make any necessary changes to raster properties, e.g.:
        ras_meta['dtype'] = "int32"
        ras_meta['nodata'] = -99

        fid = os.path.join("experiment_setup_files","simulation_results_batch_run","food_and_water_matrix", "food_matrix_"+ str(Prob_food_in_forest) + "_" + str(Prob_food_in_cropland) + "_.tif")

        with rio.open(fid, 'w', **ras_meta) as dst:
            dst.write(food_matrix.astype(int), 1)

        return food_matrix
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------




    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def initialize_water_matrix(self,Prob_water):
        
        """The function initializes water matrix based on the simulation parameters"""
        #Prob_water: probability of water being available in a given cell 
        
        #Reading the LULC and storing the plantation area details
        folder_path = os.path.join("experiment_setup_files","environment_seethathode","Raster_Files_Seethathode_Derived", self.area[self.area_size])
        fid = os.path.join(folder_path, self.reso[self.resolution], "LULC.tif")

        LULC = gdal.Open(fid).ReadAsArray()
        rowmax, colmax = LULC.shape

        water_matrix = np.zeros_like(LULC)

        for i in range(0,rowmax):
            for j in range(0,colmax):
                if LULC[i,j]==9:
                    water_matrix[i,j]=1

                if np.random.uniform(0,1) < Prob_water:
                    water_matrix[i,j]=1

        #saving the water matrix
        fid = os.path.join(folder_path, self.reso[self.resolution], "LULC.tif")

        with rio.open(fid) as src:
            ras_data = src.read()
            ras_meta = src.profile

        # make any necessary changes to raster properties, e.g.:
        ras_meta['dtype'] = "int32"
        ras_meta['nodata'] = -99

        fid = os.path.join("experiment_setup_files","simulation_results_batch_run","food_and_water_matrix", "water_matrix_" + str(Prob_water) +"_.tif")

        with rio.open(fid, 'w', **ras_meta) as dst:
            dst.write(water_matrix.astype(int), 1)

        return water_matrix
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------   




    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def main(self):

        self.initialize_food_matrix(Prob_food_in_forest = self.Prob_food_in_forest, Prob_food_in_cropland = self.Prob_food_in_cropland)
        self.initialize_water_matrix(self.Prob_water)

        return
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------



