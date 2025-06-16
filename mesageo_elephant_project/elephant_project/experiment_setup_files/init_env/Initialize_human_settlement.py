#importing packages
################################################
#Raster manipulation tools 
from osgeo import gdal   
import rioxarray as rxr
import rasterio as rio  
import geopandas as gpd

#to supress warnings
import warnings     
warnings.filterwarnings('ignore')  

#others
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#creating polygon shape files
from shapely.geometry import Polygon
import shapely
shapely.speedups.disable()

from shapely.geometry import mapping
import os
import json
################################################


class Building_classification():

    def __init__(self):
        pass


    # Villages: "Chittar-Seethathodu", "Kolamulla", "Perunad", "Thannithodu"
    # To classify the buildings into residential and non-residential 

    #---------------------------------------------------------------------------------------------------------

    def function_classify_residential_and_commercial_buildings(self, Village, total_number_of_buildings):
        """ Function that takes the Facebook population maps, redistributes the commercial and residential building for a village as per the census data"""

        #geojson file path:

        fid = os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode", "geojson_files", Village+".geojson")
        data = json.load(open(fid))

        fid = os.path.join("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode", "shape_files")

        self.function_to_create_polygon(data["features"][0]["geometry"]["coordinates"][0][0], os.path.join(fid,Village+".shp"))   #Create the polgon shapefile of the village

        population = "mesageo_elephant_project/elephant_project/experiment_setup_files/data_downloads/population.tif"

        self.function_to_clip_raster(population, os.path.join(fid,Village+".shp"), os.path.join(fid,Village+"_population.tif"))    #Clip the population maps as per the geometry of the study area

        Building = gdal.Open(os.path.join(fid,Village+"_population.tif")).ReadAsArray()
        Building, pixel_count  = self.count_pixels_populated(Building)

        density = self.house_density_estimation(Building, 1)*total_number_of_buildings

        figure(figsize=(10,10))
        im=plt.imshow(density)
        im.set_cmap('nipy_spectral')
        plt.title(Village+": population map")
        plt.savefig(os.path.join(fid, Village+"_population_map.png"), dpi=300, bbox_inches='tight')

        buliding=density
        m,n=np.shape(buliding)
        residential = np.zeros((m,n))
        non_residential = np.zeros((m,n))

        for i in range(0,m):
            for j in range(0,n):
                if buliding[i,j]>5:
                    non_residential[i,j]=buliding[i,j]
                else:
                    residential[i,j]=buliding[i,j]

        print("No. of non residential buildings:", np.sum(non_residential))
        print("No. of residential buildings:", np.sum(residential))

        figure(figsize=(10,10))
        im=plt.imshow(non_residential)
        im.set_cmap('nipy_spectral')
        plt.title(Village+": non-residential buildings")
        plt.savefig(os.path.join(fid, Village+"_non_residential.png"), dpi=300, bbox_inches='tight')

        figure(figsize=(10,10))
        im=plt.imshow(residential)
        im.set_cmap('nipy_spectral')
        plt.title(Village+": residential buildings")
        plt.savefig(os.path.join(fid, Village+"_residential.png"), dpi=300, bbox_inches='tight')

        path = os.path.join(fid, Village+"_population.tif")
        with rio.open(path) as src:
            ras_data = src.read()
            ras_meta = src.profile

        # make any necessary changes to raster properties, e.g.:
        ras_meta['dtype'] = "int32"
        ras_meta['nodata'] = -99
        #setting the path of required files depending on the platform/operating-system

        residential_path = os.path.join(fid, Village + "_residential.tif")
        non_residential_path =   os.path.join(fid, Village+"_non_residential.tif")

        with rio.open(residential_path, 'w', **ras_meta) as dst:
            dst.write(residential.astype(int), 1)

        with rio.open(non_residential_path, 'w', **ras_meta) as dst:
            dst.write(non_residential.astype(int), 1)

        return


    #---------------------------------------------------------------------------------------------------------
    def function_to_create_polygon(self, polygon_coordinates, output_path):
        """ Creates a shape file as per the polygon geometry"""
        
        dataframe = gpd.GeoDataFrame(columns = ['name', 'geometry'],crs = {'init' :'epsg:4326'})
        polygon_geom = Polygon(polygon_coordinates)
        dataframe=dataframe.append({"name":"polygon" ,"geometry":polygon_geom},ignore_index=True)
        shapefile = dataframe
        shapefile.to_file(output_path)
        return

    #-----------------------------------------------------------------------------------------------------------
    def function_to_clip_raster(self, raster_to_clip,shapefile_with_extend, path_to_write):

        """ Clips the raster files as per the extend of the study area """
        #raster_to_clip: the raster which has to be clipped
        #shapefile_with_extend: polygon shape file used for clipping
        #path_to_write: Path to save the clipped file

        area_extend = gpd.read_file(shapefile_with_extend)
        LULC = rxr.open_rasterio(raster_to_clip,masked=True).squeeze()
        LULC_CLIPPED = LULC.rio.clip(area_extend.geometry.apply(mapping), area_extend.crs, drop=True)
        LULC_CLIPPED.rio.to_raster(path_to_write)
        return

    #----------------------------------------------------------------------------------------------------
    def count_pixels_populated(self, arr):
        #Function returns the number of pixels classified as houses

        num_cells,n = np.shape(arr)
        house_presence = np.zeros((num_cells,n))
        count=0
        for i in range(0,num_cells):
            for j in range(0,n):
                if arr[i,j]>0 and arr[i,j]<50:
                    count=count+1
                    house_presence[i,j]=1

        return house_presence, count

    #----------------------------------------------------------------------------------------------------
    def house_density_estimation(self, building_array, num_cells):
        """ Function estimates the density of human agents based on whether the surrounding cells are classified as buildings or not"""
        
        #building_array: matrix with building_present=1 and building_absent=0
        #num_cells: Number of nearby cells to consider for density estimation

        m, n = building_array.shape
        density = np.zeros((m,n))
        for i in range(0,m):
            for j in range(0,n):
                if building_array[i,j]>0:
                    density[i][j] = np.sum(building_array[i-num_cells:i+num_cells, j-num_cells:j+num_cells])
        
        return density/np.sum(density)

    #----------------------------------------------------------------------------------------------------
    def main(self):

        Total_number_of_buildings = {"Thannithodu": 4480, "Chittar-Seethathodu": 11245, "Perunad": 5324}

        self.function_classify_residential_and_commercial_buildings("Thannithodu", Total_number_of_buildings["Thannithodu"])

        self.function_classify_residential_and_commercial_buildings("Chittar-Seethathodu", Total_number_of_buildings["Chittar-Seethathodu"])

        self.function_classify_residential_and_commercial_buildings("Perunad", Total_number_of_buildings["Perunad"])

        return


settlement = Building_classification()
settlement.main()