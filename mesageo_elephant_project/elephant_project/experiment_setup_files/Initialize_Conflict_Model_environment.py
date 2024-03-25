##############################################################
#importing packages
##############################################################
#Raster manipulation tools 
from rasterio.crs import CRS
import rioxarray as rxr
import geopandas as gpd
from pyproj import Proj, transform

#to supress warnings
import warnings     
warnings.filterwarnings('ignore')  

#creating polygon shape files
from shapely.geometry import Polygon
import shapely
#shapely.speedups.disable()

#others
from shapely.geometry import mapping
import os
##############################################################





#To set up the environment for the simulation
class environment():

    def __init__(self, area_size, resolution):

        self.area_size = area_size
        self.resolution = resolution
        self.delta = {800:14142, 900:15000, 1000:15811, 1100: 16583}  #distance in meteres from the center of the polygon area
        self.area = {800:"area_800sqKm", 900:"area_900sqKm", 1000:"area_1000sqKm", 1100:"area_1100sqKm"}
        self.reso = {30: "reso_30x30", 60:"reso_60x60", 90:"reso_90x90", 120:"reso_120x120", 150:"reso_150x150"}
        
        inProj, outProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
        center_lat = 9.3245
        center_lon = 76.9974
        self.center_lon, self.center_lat  = transform(inProj, outProj, center_lon, center_lat)

        latextend,lonextend = [self.center_lat-self.delta[area_size],self.center_lat+self.delta[area_size]],[self.center_lon-self.delta[area_size],self.center_lon+self.delta[area_size]]
        minlat,maxlat = latextend
        minlon,maxlon = lonextend
        self.min_lon, self.min_lat  = minlon, minlat 
        self.max_lon, self.max_lat  = maxlon, maxlat 

        #Saving the files
        #-------------------------------------------------------------------
        self.root_folder_path = os.path.join("abm_codes", "experiment_setup_files", "environment_seethathode","Raster_Files_Seethathode_Derived",self.area[self.area_size],self.reso[self.resolution]) 
        #-------------------------------------------------------------------


        #-------------------------------------------------------------------
        #Landuse labels used in LULC
        self.landusetypes={"Deciduous Broadleaf Forest":1,"Cropland":2,"Built-up Land":3,"Mixed Forest":4,
                       "Shrubland":5,"Barren Land":6,"Fallow Land":7,"Wasteland":8,"Water Bodies":9,
                       "Plantations":10,"Aquaculture":11,"Mangrove Forest":12,"Salt Pan":13,"Grassland":14,
                       "Evergreen Broadlead Forest":15,"Deciduous Needleleaf Forest":16,
                       "Permanent Wetlands":17, "Snow and ice":18, "Evergreen Needleleaf Forest":19}
        #-------------------------------------------------------------------

        
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def function_to_create_polygon(self,minlat,minlon,maxlat,maxlon):

        """ Creates a shape file as per the extend of the study area"""
        
        dataframe = gpd.GeoDataFrame(columns = ['name', 'geometry'], crs = {'init' :'epsg:3857'})
        lat_point_list = [minlat, minlat, maxlat, maxlat]
        lon_point_list = [minlon, maxlon, maxlon, minlon]
        polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
        dataframe=dataframe.append({"name":"polygon" ,"geometry":polygon_geom},ignore_index=True)
        shapefile = dataframe

        folder_path = os.path.join("abm_codes", "experiment_setup_files", "environment_seethathode","Raster_Files_Seethathode_Derived", self.area[self.area_size]) 
        fid = os.path.join(folder_path,"polygon_shape_file","polygon_study_area.shp")

        shapefile.to_file(fid)

        return
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------




    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def reproject(self):
        #The supporting files needed:

        #1: DEM
        fid = os.path.join("abm_codes", "experiment_setup_files", "environment_seethathode","Raster_Files_Seethathode_Derived","DEM", "DEM.tif")
        self.reproject_raster_to_different_crs(fid,"DEM")

        #2: LULC
        fid = os.path.join("abm_codes", "experiment_setup_files", "environment_seethathode","Raster_Files_Seethathode_Derived","Landuse","Landuse.tif")
        self.reproject_raster_to_different_crs(fid,"LULC")

        #3: Population maps

        fid = os.path.join("abm_codes", "experiment_setup_files", "environment_seethathode","Raster_Files_Seethathode_Derived","Population","Population.tif")
        self.reproject_raster_to_different_crs(fid,"Population")
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------




    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def reproject_raster_to_different_crs(self,input_file, name):
        """ function to project a georaster from one CRS to another"""


        print("REPROJECTING RASTER FILE: " + name)

        print("Working on it......")
        input_raster = rxr.open_rasterio(input_file,masked=True).squeeze()
        print("Source CRS:", input_raster.rio.crs)
        crs_wgs84 = CRS.from_string('EPSG:3857')
        raster_wgs84 = input_raster.rio.reproject(crs_wgs84)
        raster_wgs84.rio.crs
        print("Target CRS:", raster_wgs84.rio.crs)

        folder_path = os.path.join("abm_codes", "experiment_setup_files", "environment_seethathode","Raster_Files_Seethathode_Derived", self.area[self.area_size]) 
        raster_wgs84.rio.to_raster(os.path.join(folder_path,name+"_REPROJECTED.tif"), dtype='float32', driver="GTiff")
        print("And its done......")

        return
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------




    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def function_to_clip_raster(self):

        """ Clips the raster files as per the extend of the study area"""

        #1. LULC
        #2. DEM
        #3. Population

        print("CLIPPING RASTER FILES")

        folder_path = os.path.join("abm_codes", "experiment_setup_files", "environment_seethathode","Raster_Files_Seethathode_Derived", self.area[self.area_size]) 
        fid = os.path.join(folder_path,"polygon_shape_file","polygon_study_area.shp")

        area = os.path.join(fid)
        area_extend = gpd.read_file(area)

        #LULC

        fid = os.path.join(os.path.join(folder_path,"LULC_REPROJECTED.tif"))
        LULC = rxr.open_rasterio(fid,masked=True).squeeze()
        LULC_CLIPPED = LULC.rio.clip(area_extend.geometry.apply(mapping), area_extend.crs, drop=True)
        LULC_CLIPPED.rio.to_raster(os.path.join(folder_path,"LULC_CLIPPED.tif"))


        #DEM

        fid = os.path.join(os.path.join(folder_path,"DEM_REPROJECTED.tif"))
        DEM = rxr.open_rasterio(fid,masked=True).squeeze()
        DEM_CLIPPED = DEM.rio.clip(area_extend.geometry.apply(mapping), area_extend.crs, drop=True)
        DEM_CLIPPED.rio.to_raster(os.path.join(folder_path,"DEM_CLIPPED.tif"))


        #Population

        fid = os.path.join(os.path.join(folder_path,"Population_REPROJECTED.tif"))
        population = rxr.open_rasterio(fid,masked=True).squeeze()
        population_CLIPPED = population.rio.clip(area_extend.geometry.apply(mapping), area_extend.crs, drop=True)
        population_CLIPPED.rio.to_raster(os.path.join(folder_path,"Population_CLIPPED.tif"))

        return
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------




    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def resolution_fixing(self):
        #To resize DEM raster file 
        from osgeo import gdal, gdalconst

        print("FIXING RESOLUTION")

        folder_path = os.path.join("abm_codes", "experiment_setup_files", "environment_seethathode","Raster_Files_Seethathode_Derived", self.area[self.area_size])

        #1. LULC
        #2. DEM

        #Source
        LULC = os.path.join(os.path.join(folder_path,"LULC_CLIPPED.tif"))
        DEM = os.path.join(os.path.join(folder_path,"DEM_CLIPPED.tif"))
        population = os.path.join(os.path.join(folder_path,"Population_CLIPPED.tif"))

        #destination
        LULC_dest = os.path.join(folder_path, self.reso[self.resolution], "LULC.tif")
        DEM_dest = os.path.join(folder_path,self.reso[self.resolution], "DEM.tif")
        Population_dest = os.path.join(folder_path,self.reso[self.resolution], "Population.tif")

        src1 = gdal.Open(LULC, gdalconst.GA_ReadOnly)
        src_proj = src1.GetProjection()
        src_geotrans = src1.GetGeoTransform()

        src2 = gdal.Open(DEM, gdalconst.GA_ReadOnly)
        src_proj = src2.GetProjection()
        src_geotrans = src2.GetGeoTransform()

        src3 = gdal.Open(population, gdalconst.GA_ReadOnly)
        src_proj = src3.GetProjection()
        src_geotrans = src3.GetGeoTransform()

        match_filename = os.path.join("abm_codes", "experiment_setup_files", "environment_seethathode","Raster_Files_Seethathode_Derived",self.area[self.area_size],"resolution_fixing",self.reso[self.resolution] +".tif")
        match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
        match_proj = match_ds.GetProjection()
        match_geotrans = match_ds.GetGeoTransform()

        wide = match_ds.RasterXSize
        high = match_ds.RasterYSize

        # Output / destination
        dst1 = gdal.GetDriverByName('GTiff').Create(LULC_dest, wide, high, 1, gdalconst.GDT_Int32)
        dst1.SetGeoTransform(match_geotrans)
        dst1.SetProjection(match_proj)

        dst2 = gdal.GetDriverByName('GTiff').Create(DEM_dest, wide, high, 1, gdalconst.GDT_Float32)
        dst2.SetGeoTransform(match_geotrans)
        dst2.SetProjection(match_proj)

        dst3 = gdal.GetDriverByName('GTiff').Create(Population_dest, wide, high, 1, gdalconst.GDT_Float32)
        dst3.SetGeoTransform(match_geotrans)
        dst3.SetProjection(match_proj)


        interpolation=gdalconst.GRA_NearestNeighbour
        #gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
        #gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_CubicSpline)
        gdal.ReprojectImage(src1, dst1, src_proj, match_proj, interpolation)
        del dst1 # Flush
        gdal.ReprojectImage(src2, dst2, src_proj, match_proj, interpolation)
        del dst2 # Flush
        gdal.ReprojectImage(src3, dst3, src_proj, match_proj, interpolation)
        del dst3 # Flush
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------




    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def main(self):

        #print("SETTING UP THE ENVIRONMENT FOR SIMULATION")

        self.function_to_create_polygon(self.min_lat, self.min_lon, self.max_lat, self.max_lon)
        self.reproject()
        self.function_to_clip_raster()
        self.resolution_fixing()

        #print("DONE")

        return
    #---------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------










##############################################################################################################################################
from tqdm import tqdm

if __name__ == "__main__":

    env = environment(area_size = 1100, resolution = 150)
    env.main()
##############################################################################################################################################