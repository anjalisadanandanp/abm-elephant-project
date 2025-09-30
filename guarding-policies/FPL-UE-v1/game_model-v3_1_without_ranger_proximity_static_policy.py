import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import pathlib
import yaml
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import random
import itertools
from pyproj import Proj, transform  
from mpl_toolkits.basemap import Basemap    
import rasterio
from rasterio.features import shapes
import fiona
import geojson
import matplotlib.cm as cm
import shutil

import warnings
warnings.filterwarnings("ignore")


fontsize = 8
plt.rcParams.update(
    {
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "figure.titlesize": fontsize,
    }
)


import sys
sys.path.append(os.getcwd())

module = importlib.import_module('guarding-policies.FPL-UE-v1.abm_model_HEC_with_landscape_deterrent_policies_without_ranger_proximity')
batch_run_model = module.batch_run_model



def make_trajectory_summary_plots_v1(base_path, output_folder, num_cropraiding_steps = 12):

    def raster_to_geojson(input_raster_path, output_geojson_path, target_value=1):

        with rasterio.open(input_raster_path) as src:
            image = src.read(1)

            if image.dtype not in ['int16', 'int32', 'uint8', 'uint16', 'float32']:
                    image = image.astype('uint8')

            mask = image > 0
            mask = mask.astype('uint8')

            results = [
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) in enumerate(shapes(mask, transform=src.transform))
                if v == 1 
            ]

            with fiona.open(
                output_geojson_path, 
                'w', 
                driver='GeoJSON',
                schema={'geometry': 'Polygon', 'properties': {'raster_val': 'int'}}
            ) as dst:
                for feature in results:
                    dst.write(feature)
        return

    def lat_lon_to_pixel(lats, lons, xmin, ymax, xres, yres):
        """Convert lat/lon coordinates to pixel coordinates"""
        rows = ((ymax - lats) / -yres).astype(int)
        cols = ((lons - xmin) / xres).astype(int)
        return rows, cols

    def find_cropland_use_indices(landuse_sequence):
        indices = []
        start = None
        
        for i, value in enumerate(landuse_sequence):
            if value == 10:
                if start is None:
                    start = i
            else:
                if start is not None:
                    # End subsequence if current value is not 10, 3, 9, or 6
                    if value not in [10, 3, 9, 6]:
                        indices.append((start, i))
                        start = None
        
        # Handle case where sequence ends while in a subsequence starting with 10
        # Only add if the last value is not 10, 3, 9, or 6
        if start is not None:
            last_value = landuse_sequence[-1]
            if last_value not in [10, 3, 9, 6]:
                indices.append((start, len(landuse_sequence)))
        
        return indices
    
    simulation_repeats = os.listdir(base_path)

    try:
        simulation_repeats.remove("attacker_strategy_matrix.png")
    except:
        pass

    try:
        simulation_repeats.remove("attacker_strategy_matrix.tif")
    except:
        pass

    try:
        simulation_repeats.remove("defender_coverage_matrix.png")
    except:
        pass

    try:
        simulation_repeats.remove("defender_coverage_matrix.tif")
    except:
        pass

    try:
        simulation_repeats.remove("model_parameters.yaml")
    except:
        pass

    ds = gdal.Open(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix_0.tif"))

    data = ds.ReadAsArray()
    data = np.flip(data, axis=0)
    row_size, col_size = data.shape
    xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

    fig, ax = plt.subplots(figsize = (8,8))
    ax.yaxis.set_inverted(True)

    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
    LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
    LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

    map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

    raster_to_geojson(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix_0.tif"), os.path.join(output_folder, 'guarded_patches.geojson'))

    with open('mesageo_elephant_project/elephant_project/geojson_files/landuse_10.geojson', 'r') as f:
        geojson_object = geojson.load(f)

    for feature in geojson_object['features']:
        coords = feature['geometry']['coordinates'][0]
        coords = [transform(inProj, outProj, lon, lat) for lon, lat in coords]
        coords = [(lon, lat) for lon, lat in coords]
        lon, lat = zip(*coords)

        plt.fill(lon, lat, color='yellow', alpha=0.20, zorder=1)

        map.plot(lon, lat, marker=None, color='black', linewidth=1, zorder=2)

    with open(os.path.join(output_folder, 'guarded_patches.geojson'), 'r') as f:
        geojson_object = geojson.load(f)

    for feature in geojson_object['features']:
        coords = feature['geometry']['coordinates'][0]
        coords = [transform(inProj, outProj, lon, lat) for lon, lat in coords]
        coords = [(lon, lat) for lon, lat in coords]
        lon, lat = zip(*coords)
        map.plot(lon, lat, marker=None, color='red', linewidth=2, zorder=3)

    map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
    map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

    for simulation_repeat in simulation_repeats:

        try:
            agent_data = pd.read_csv(os.path.join(base_path, simulation_repeat, "output_files", "agent_data.csv"))
            longitude = agent_data['longitude'].values
            latitude = agent_data['latitude'].values

            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
            longitude, latitude = transform(inProj, outProj, longitude, latitude)
            x_new, y_new = map(longitude,latitude)
            C = np.arange(len(x_new))
            nz = mcolors.Normalize()
            nz.autoscale(C)

            geotransform = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").GetGeoTransform()
            ag_xmin, ag_xres, ag_xskew, ag_ymax, ag_yskew, ag_yres = geotransform

            landuse_matrix = gdal.Open(os.path.join(base_path, simulation_repeat, "env", "LULC.tif")).ReadAsArray()
            boundary_patches_guarded = gdal.Open(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix_0.tif")).ReadAsArray()
            agricultural_plts = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").ReadAsArray()

            boundary_patches = gdal.Open("create-strategy-matrix/boundary_raster_discretised.tif").ReadAsArray()

            boundary_patches_unguarded = boundary_patches - boundary_patches_guarded

            rows, cols = lat_lon_to_pixel(
                agent_data["latitude"].values, agent_data["longitude"].values, 
                ag_xmin, ag_ymax, ag_xres, ag_yres
            )
            
            landuse_values = landuse_matrix[rows, cols]

            indices = find_cropland_use_indices(landuse_values)
            
            for sequence in indices:

                if (sequence[1] - sequence[0]) >= num_cropraiding_steps:

                    # if boundary_patches_guarded[rows[sequence[0]], cols[sequence[0]]] != 0:

                    #     for i in range(sequence[1] - sequence[0]):

                    #         if agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]] != 0:

                    #             ax.plot(x_new[sequence[0] + i], y_new[sequence[0] + i], marker='s', markersize=2, color='forestgreen', linewidth=1, zorder=2)

                    #             ax.quiver(x_new[:-1], y_new[:-1], 
                    #                         x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1], 
                    #                         scale_units='xy', angles='xy', 
                    #                         scale=1, zorder=2, color = cm.jet(nz(C)), 
                    #                         width=0.0010)

                    #             ax.scatter(x_new[0], y_new[0], 5, marker='o', color='black', zorder=2) 
                    #             ax.scatter(x_new[-1], y_new[-1], 5, marker='^', color='black', zorder=2) 


                    if boundary_patches_unguarded[rows[sequence[0]], cols[sequence[0]]] != 0:

                        for i in range(sequence[1] - sequence[0]):

                            if agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]] != 0:

                                ax.plot(x_new[sequence[0] + i], y_new[sequence[0] + i], marker='s', markersize=2, color='forestgreen', linewidth=1, zorder=2)

                                ax.quiver(x_new[:-1], y_new[:-1], 
                                            x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1], 
                                            scale_units='xy', angles='xy', 
                                            scale=1, zorder=2, color = "black", 
                                            width=0.0010)

                                ax.scatter(x_new[0], y_new[0], 5, marker='o', color='black', zorder=2) 
                                ax.scatter(x_new[-1], y_new[-1], 5, marker='^', color='black', zorder=2)   

        except:
            pass

    plt.savefig(os.path.join(output_folder, "crop_raiding_trajs_through_unguarded_boundary_patches.png"), dpi=750, bbox_inches='tight')

    plt.close()

    return

def make_trajectory_summary_plots_v2(base_path, output_folder, num_cropraiding_steps = 12):

    def raster_to_geojson(input_raster_path, output_geojson_path):

        with rasterio.open(input_raster_path) as src:
            image = src.read(1)

            if image.dtype not in ['int16', 'int32', 'uint8', 'uint16', 'float32']:
                    image = image.astype('uint8')

            mask = image > 0
            mask = mask.astype('uint8')

            results = [
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) in enumerate(shapes(mask, transform=src.transform))
                if v == 1 
            ]

            with fiona.open(
                output_geojson_path, 
                'w', 
                driver='GeoJSON',
                schema={'geometry': 'Polygon', 'properties': {'raster_val': 'int'}}
            ) as dst:
                for feature in results:
                    dst.write(feature)
        return

    def lat_lon_to_pixel(lats, lons, xmin, ymax, xres, yres):
        """Convert lat/lon coordinates to pixel coordinates"""
        rows = ((ymax - lats) / -yres).astype(int)
        cols = ((lons - xmin) / xres).astype(int)
        return rows, cols

    def find_cropland_use_indices(landuse_sequence):
        indices = []
        start = None
        
        for i, value in enumerate(landuse_sequence):
            if value == 10:
                if start is None:
                    start = i
            else:
                if start is not None:
                    # End subsequence if current value is not 10, 3, 9, or 6
                    if value not in [10, 3, 9, 6]:
                        indices.append((start, i))
                        start = None
        
        # Handle case where sequence ends while in a subsequence starting with 10
        # Only add if the last value is not 10, 3, 9, or 6
        if start is not None:
            last_value = landuse_sequence[-1]
            if last_value not in [10, 3, 9, 6]:
                indices.append((start, len(landuse_sequence)))
        
        return indices
    
    simulation_repeats = os.listdir(base_path)

    try:
        simulation_repeats.remove("attacker_strategy_matrix.png")
    except:
        pass

    try:
        simulation_repeats.remove("attacker_strategy_matrix.tif")
    except:
        pass

    try:
        simulation_repeats.remove("defender_coverage_matrix.png")
    except:
        pass

    try:
        simulation_repeats.remove("defender_coverage_matrix.tif")
    except:
        pass

    try:
        simulation_repeats.remove("model_parameters.yaml")
    except:
        pass

    ds = gdal.Open(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix_0.tif"))

    data = ds.ReadAsArray()
    data = np.flip(data, axis=0)
    row_size, col_size = data.shape
    xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

    fig, ax = plt.subplots(figsize = (8,8))
    ax.yaxis.set_inverted(True)

    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
    LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
    LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

    map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

    raster_to_geojson(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix_0.tif"), os.path.join(output_folder, 'guarded_patches.geojson'))

    with open('mesageo_elephant_project/elephant_project/geojson_files/landuse_10.geojson', 'r') as f:
        geojson_object = geojson.load(f)

    for feature in geojson_object['features']:
        coords = feature['geometry']['coordinates'][0]
        coords = [transform(inProj, outProj, lon, lat) for lon, lat in coords]
        coords = [(lon, lat) for lon, lat in coords]
        lon, lat = zip(*coords)

        plt.fill(lon, lat, color='yellow', alpha=0.20, zorder=1)

        map.plot(lon, lat, marker=None, color='black', linewidth=1, zorder=2)

    with open(os.path.join(output_folder, 'guarded_patches.geojson'), 'r') as f:
        geojson_object = geojson.load(f)

    for feature in geojson_object['features']:
        coords = feature['geometry']['coordinates'][0]
        coords = [transform(inProj, outProj, lon, lat) for lon, lat in coords]
        coords = [(lon, lat) for lon, lat in coords]
        lon, lat = zip(*coords)
        map.plot(lon, lat, marker=None, color='red', linewidth=2, zorder=3)

    map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
    map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

    for simulation_repeat in simulation_repeats:

        try:
            agent_data = pd.read_csv(os.path.join(base_path, simulation_repeat, "output_files", "agent_data.csv"))
            longitude = agent_data['longitude'].values
            latitude = agent_data['latitude'].values

            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
            longitude, latitude = transform(inProj, outProj, longitude, latitude)
            x_new, y_new = map(longitude,latitude)
            C = np.arange(len(x_new))
            nz = mcolors.Normalize()
            nz.autoscale(C)

            geotransform = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").GetGeoTransform()
            ag_xmin, ag_xres, ag_xskew, ag_ymax, ag_yskew, ag_yres = geotransform

            landuse_matrix = gdal.Open(os.path.join(base_path, simulation_repeat, "env", "LULC.tif")).ReadAsArray()
            boundary_patches_guarded = gdal.Open(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix_0.tif")).ReadAsArray()
            agricultural_plts = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").ReadAsArray()

            boundary_patches = gdal.Open("create-strategy-matrix/boundary_raster_discretised.tif").ReadAsArray()

            boundary_patches_unguarded = boundary_patches - boundary_patches_guarded

            rows, cols = lat_lon_to_pixel(
                agent_data["latitude"].values, agent_data["longitude"].values, 
                ag_xmin, ag_ymax, ag_xres, ag_yres
            )
            
            landuse_values = landuse_matrix[rows, cols]

            indices = find_cropland_use_indices(landuse_values)
            
            for sequence in indices:

                if (sequence[1] - sequence[0]) >= num_cropraiding_steps:

                    if boundary_patches_guarded[rows[sequence[0]], cols[sequence[0]]] != 0:

                        for i in range(sequence[1] - sequence[0]):

                            if agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]] != 0:

                                ax.plot(x_new[sequence[0] + i], y_new[sequence[0] + i], marker='s', markersize=2, color='forestgreen', linewidth=1, zorder=2)

                                ax.quiver(x_new[:-1], y_new[:-1], 
                                            x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1], 
                                            scale_units='xy', angles='xy', 
                                            scale=1, zorder=2, color = cm.jet(nz(C)), 
                                            width=0.0010)

                                ax.scatter(x_new[0], y_new[0], 5, marker='o', color='black', zorder=2) 
                                ax.scatter(x_new[-1], y_new[-1], 5, marker='^', color='black', zorder=2) 


                    # if boundary_patches_unguarded[rows[sequence[0]], cols[sequence[0]]] != 0:

                    #     for i in range(sequence[1] - sequence[0]):

                    #         if agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]] != 0:

                    #             ax.plot(x_new[sequence[0] + i], y_new[sequence[0] + i], marker='s', markersize=2, color='forestgreen', linewidth=1, zorder=2)

                    #             ax.quiver(x_new[:-1], y_new[:-1], 
                    #                         x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1], 
                    #                         scale_units='xy', angles='xy', 
                    #                         scale=1, zorder=2, color = "black", 
                    #                         width=0.0010)

                    #             ax.scatter(x_new[0], y_new[0], 5, marker='o', color='black', zorder=2) 
                    #             ax.scatter(x_new[-1], y_new[-1], 5, marker='^', color='black', zorder=2)   

        except:
            pass

    plt.savefig(os.path.join(output_folder, "crop_raiding_trajs_through_guarded_boundary_patches.png"), dpi=750, bbox_inches='tight')

    plt.close()

    return

def make_trajectory_summary_plots_v3(base_path, output_folder):

    def raster_to_geojson(input_raster_path, output_geojson_path):

        with rasterio.open(input_raster_path) as src:
            image = src.read(1)

            if image.dtype not in ['int16', 'int32', 'uint8', 'uint16', 'float32']:
                    image = image.astype('uint8')

            mask = image > 0
            mask = mask.astype('uint8')

            results = [
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) in enumerate(shapes(mask, transform=src.transform))
                if v == 1 
            ]

            with fiona.open(
                output_geojson_path, 
                'w', 
                driver='GeoJSON',
                schema={'geometry': 'Polygon', 'properties': {'raster_val': 'int'}}
            ) as dst:
                for feature in results:
                    dst.write(feature)
        return

    simulation_repeats = os.listdir(base_path)

    try:
        simulation_repeats.remove("attacker_strategy_matrix.png")
    except:
        pass

    try:
        simulation_repeats.remove("attacker_strategy_matrix.tif")
    except:
        pass

    try:
        simulation_repeats.remove("defender_coverage_matrix.png")
    except:
        pass

    try:
        simulation_repeats.remove("defender_coverage_matrix.tif")
    except:
        pass

    try:
        simulation_repeats.remove("model_parameters.yaml")
    except:
        pass

    ds = gdal.Open(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix_0.tif"))

    data = ds.ReadAsArray()
    data = np.flip(data, axis=0)
    row_size, col_size = data.shape
    xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

    fig, ax = plt.subplots(figsize = (8,8))
    ax.yaxis.set_inverted(True)

    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
    LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
    LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

    map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

    raster_to_geojson(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix_0.tif"), os.path.join(output_folder, 'guarded_patches.geojson'))

    with open('mesageo_elephant_project/elephant_project/geojson_files/landuse_10.geojson', 'r') as f:
        geojson_object = geojson.load(f)

    for feature in geojson_object['features']:
        coords = feature['geometry']['coordinates'][0]
        coords = [transform(inProj, outProj, lon, lat) for lon, lat in coords]
        coords = [(lon, lat) for lon, lat in coords]
        lon, lat = zip(*coords)

        plt.fill(lon, lat, color='yellow', alpha=0.20, zorder=1)

        map.plot(lon, lat, marker=None, color='black', linewidth=1, zorder=2)

    with open(os.path.join(output_folder, 'guarded_patches.geojson'), 'r') as f:
        geojson_object = geojson.load(f)

    for feature in geojson_object['features']:
        coords = feature['geometry']['coordinates'][0]
        coords = [transform(inProj, outProj, lon, lat) for lon, lat in coords]
        coords = [(lon, lat) for lon, lat in coords]
        lon, lat = zip(*coords)
        map.plot(lon, lat, marker=None, color='red', linewidth=2, zorder=3)

    map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
    map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

    for simulation_repeat in simulation_repeats:

        try:
            agent_data = pd.read_csv(os.path.join(base_path, simulation_repeat, "output_files", "agent_data.csv"))
            longitude = agent_data['longitude'].values
            latitude = agent_data['latitude'].values

            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
            longitude, latitude = transform(inProj, outProj, longitude, latitude)
            x_new, y_new = map(longitude,latitude)
            C = np.arange(len(x_new))
            nz = mcolors.Normalize()
            nz.autoscale(C)

            ax.quiver(x_new[:-1], y_new[:-1], 
                        x_new[1:]-x_new[:-1], y_new[1:]-y_new[:-1], 
                        scale_units='xy', angles='xy', 
                        scale=1, zorder=2, color = cm.jet(nz(C)), 
                        width=0.0010)

            ax.scatter(x_new[0], y_new[0], 5, marker='o', color='black', zorder=2) 
            ax.scatter(x_new[-1], y_new[-1], 5, marker='^', color='black', zorder=2) 

        except:
            pass

    plt.savefig(os.path.join(output_folder, "summary_of_all_simulated_trajectories.png"), dpi=750, bbox_inches='tight')

    plt.close()

    return

def update_targets_df(output_folder, targets_df, current_game_step, num_cropraiding_steps=12):

    # print("\n#----------------UPDATING TARGETS DATAFRAME BASED ON OBSERVATIONS----------------#")

    def find_rewards_based_on_intercepted_trajectories(output_folder):

        dict_of_attacked_targets = {}

        for simulation_folder in os.listdir(output_folder):

            try:

                df = pd.read_csv(os.path.join(output_folder, simulation_folder, "output_files/agent_data.csv"))
                df.dropna(subset=['ROW', 'COL'], inplace=True)

                unique_targets = df["target_attacked"].dropna().unique()

                for target in unique_targets:
                    if target not in dict_of_attacked_targets:
                        dict_of_attacked_targets[target] = 0
                    dict_of_attacked_targets[target] += 1
            
            except Exception as e:
                pass

        # for target, count in dict_of_attacked_targets.items():
        #     print(f"Covered target {target} was attacked {count} times.")  

        df_dict_of_attacked_targets = pd.DataFrame.from_dict(dict_of_attacked_targets, orient='index', columns=['count'])

        df_dict_of_attacked_targets.reset_index(inplace=True)
        df_dict_of_attacked_targets.rename(columns={'index': 'target'}, inplace=True)
                                                
        df_dict_of_attacked_targets.to_csv(os.path.join(output_folder, "df_of_attacked_targets.csv"))

        return df_dict_of_attacked_targets

    def lat_lon_to_pixel(lats, lons, xmin, ymax, xres, yres):
        """Convert lat/lon coordinates to pixel coordinates"""
        rows = ((ymax - lats) / -yres).astype(int)
        cols = ((lons - xmin) / xres).astype(int)
        return rows, cols

    def find_cropland_use_indices(landuse_sequence):
        indices = []
        start = None
        
        for i, value in enumerate(landuse_sequence):
            if value == 10:
                if start is None:
                    start = i
            else:
                if start is not None:
                    # End subsequence if current value is not 10, 3, 9, or 6
                    if value not in [10, 3, 9, 6]:
                        indices.append((start, i))
                        start = None
        
        # Handle case where sequence ends while in a subsequence starting with 10
        # Only add if the last value is not 10, 3, 9, or 6
        if start is not None:
            last_value = landuse_sequence[-1]
            if last_value not in [10, 3, 9, 6]:
                indices.append((start, len(landuse_sequence)))
        
        return indices
        
    def find_penalties_based_on_intercepted_trajectories(output_folder):

        boundary_patch_matrix = gdal.Open(os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init/potential_coverage_matrix.tif")).ReadAsArray()
        agricultural_plot_matrix = gdal.Open(os.path.join("create-landholding-matrix/agricultural_plots_assignment.tif")).ReadAsArray()

        boundary_patch_unique = np.unique(boundary_patch_matrix)
        agricultural_plot_unique = np.unique(agricultural_plot_matrix)

        association_df = pd.DataFrame(
            data=0,
            index=boundary_patch_unique,
            columns=agricultural_plot_unique
        )

        association_df.to_csv(os.path.join(output_folder, "association_df_init.csv"))

        simulation_repeats = os.listdir(output_folder)

        simulation_repeats = [
            os.path.join(output_folder, item)
            for item in simulation_repeats
            if os.path.isdir(os.path.join(output_folder, item))
        ]

        agricultural_plots_attacked = {}

        for simulation_repeat in simulation_repeats:

            agent_data = pd.read_csv(os.path.join(simulation_repeat, "output_files", "agent_data.csv"))

            geotransform = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").GetGeoTransform()
            ag_xmin, ag_xres, ag_xskew, ag_ymax, ag_yskew, ag_yres = geotransform

            landuse_matrix = gdal.Open(os.path.join(simulation_repeat, "env", "LULC.tif")).ReadAsArray()
            agricultural_plts = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").ReadAsArray()

            rows, cols = lat_lon_to_pixel(
                agent_data["latitude"].values, agent_data["longitude"].values, 
                ag_xmin, ag_ymax, ag_xres, ag_yres
            )
            
            landuse_values = landuse_matrix[rows, cols]

            indices = find_cropland_use_indices(landuse_values)
            
            for sequence in indices:

                for i in range(sequence[1] - sequence[0]):

                    if agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]] != 0:

                        if agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]] not in agricultural_plots_attacked:
                            agricultural_plots_attacked[agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]]] = 0
                        
                        agricultural_plots_attacked[agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]]] += 1


        # print("\n")

        covered_targets_matrix = gdal.Open(os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init/game_step_1/defender_coverage_matrix.tif")).ReadAsArray()
        covered_targets = np.unique(covered_targets_matrix)
        covered_targets = covered_targets[covered_targets != 0]


        # print("covered boundaries:", covered_targets)


        for plot, count in agricultural_plots_attacked.items():
            # print(f"Agricultural plot {plot} was attacked {count} times.")

            association_df.loc[covered_targets, plot] += count

        association_df.to_csv(os.path.join(output_folder, "association_df_updated.csv"))

        ds = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif")
        data = ds.ReadAsArray()
        row_size, col_size = data.shape
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

        fig, ax = plt.subplots(figsize = (8,8))

        outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
        LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
        LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

        map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        agricultural_plot_matrix_attacked = np.zeros_like(agricultural_plot_matrix)

        for plot, count in agricultural_plots_attacked.items():
            agricultural_plot_matrix_attacked[agricultural_plot_matrix == plot] = count

        cax = map.imshow(agricultural_plot_matrix_attacked, cmap='hot', interpolation='nearest',origin='upper', vmin=0, vmax=10)
        fig.colorbar(cax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(os.path.join(output_folder, "agricultural_plots_attacked_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()

        agricultural_plts = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").ReadAsArray()
        food_matrix = gdal.Open(os.path.join(simulation_repeats[0], "env", "food_matrix_0.1_1.0_.tif")).ReadAsArray()

        total_crop_raid_loss = 0
        for plot, count in agricultural_plots_attacked.items():
            food_val = np.sum(food_matrix[agricultural_plts == plot])
            total_crop_raid_loss += food_val * count

        total_food_value = np.sum(food_matrix[agricultural_plts != 0])

        print(f"\nTotal crop raid loss across the landscape: {total_crop_raid_loss} (kg)\n")

        return total_crop_raid_loss/total_food_value
        
    df_dict_of_attacked_targets = find_rewards_based_on_intercepted_trajectories(output_folder)

    step_penalty = find_penalties_based_on_intercepted_trajectories(output_folder)

    for boundary_patch in df_dict_of_attacked_targets["target"].values:

        reward = df_dict_of_attacked_targets.loc[df_dict_of_attacked_targets["target"] == boundary_patch, "count"].values[0]/sum(df_dict_of_attacked_targets["count"].values)
        targets_df.loc[targets_df["boundary_patch_id"] == boundary_patch, "reward"] += reward

        print("boundary_patch:", boundary_patch, "reward:", reward)

    potential_targets = np.unique(gdal.Open(os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init/potential_coverage_matrix.tif")).ReadAsArray())

    for boundary_patch in potential_targets:
        if boundary_patch not in df_dict_of_attacked_targets["target"].values and boundary_patch != 0:
            penalty = step_penalty/len(potential_targets)
            targets_df.loc[targets_df["boundary_patch_id"] == boundary_patch, "penalty"] -= penalty
            print("boundary_patch:", boundary_patch, "penalty:", penalty)

    targets_df.to_csv(os.path.join(os.getcwd(), "guarding-policies/FPL-UE-v1/coverage_matrix_init", "game_step_" + str(int(current_game_step + 1)), "boundary_patch_reward_penalty_matrix.csv"), index=False)

    return targets_df, step_penalty

def create_defender_coverage_matrix(defender_strategy):

    potential_coverage_matrix = gdal.Open(os.path.join("/home2/anjali/GitHub/abm-elephant-project/guarding-policies/FPL-UE-v1/coverage_matrix_init/potential_coverage_matrix.tif")).ReadAsArray()
    
    coverage_matrix = np.zeros_like(potential_coverage_matrix)

    target_ids = [index + 1 for index, value in enumerate(defender_strategy) if value != 0]

    for target_id in target_ids:
        mask = potential_coverage_matrix == target_id
        coverage_matrix[mask] = target_id

    #-----------plot coverage matrix#-----------#
    # fig, ax = plt.subplots(figsize=(8, 8))
    # cmap = mcolors.ListedColormap(['white', 'red'])
    # im = ax.imshow(coverage_matrix, cmap=cmap, vmin=0, vmax=1)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # legend_elements = [
    #     Patch(facecolor='red', edgecolor='black', label='Protected'),
    #     Patch(facecolor='white', edgecolor='black', label='Unprotected')
    # ]
    # ax.legend(handles=legend_elements, loc="upper right")
    # plt.show()
    #-----------plot coverage matrix#-----------#

    return coverage_matrix

def plot_and_save_defender_coverage(coverage_matrix, output_folder, figsize=(8, 8), 
                          protected_color='red', unprotected_color='white'):





    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = mcolors.ListedColormap([unprotected_color, protected_color])
    
    im = ax.imshow(coverage_matrix, cmap=cmap, vmin=0, vmax=1)
    
    ax.set_xticks([])
    ax.set_yticks([])

    legend_elements = [
        Patch(facecolor=protected_color, edgecolor='black', label='Protected'),
        Patch(facecolor=unprotected_color, edgecolor='black', label='Unprotected')
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.savefig(
        os.path.join(output_folder, "defender_coverage_matrix.png"),
        bbox_inches="tight",
        dpi=300,
    )





    source_file = gdal.Open("guarding-policies/FPL-UE-v1/coverage_matrix_init/potential_coverage_matrix.tif")

    cols = source_file.RasterXSize
    rows = source_file.RasterYSize
    projection = source_file.GetProjection()
    geotransform = source_file.GetGeoTransform()

    output_file = os.path.join(output_folder, "defender_coverage_matrix.tif")

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(coverage_matrix.astype(np.uint8))

    source_file = None
    output_dataset = None

    return 

def combination_to_binary_vector(combination, NUM_LANDSCAPE_CELLS):

    binary_vector = np.zeros(NUM_LANDSCAPE_CELLS, dtype=int)
    indices = [int(index - 1) for index in combination]
    binary_vector[indices] = 1
    return binary_vector

def generate_defender_strategies(BUDGET_K, NUM_LANDSCAPE_CELLS, targets_df):

    def select_cells_at_distance(target_lat, target_lon, distance_cells, distance_type='euclidean'):

        dataset = gdal.Open(coverage_matrix_path)
        geotransform = dataset.GetGeoTransform()
        original_array = dataset.ReadAsArray()

        original_array[275:350, 550:650] = 0
        
        x_origin = geotransform[0]  
        y_origin = geotransform[3] 
        pixel_width = geotransform[1]
        pixel_height = geotransform[5] 
        
        target_col = int((target_lon - x_origin) / pixel_width)
        target_row = int((target_lat - y_origin) / pixel_height)
    
        
        rows, cols = original_array.shape
        
        if not (0 <= target_row < rows and 0 <= target_col < cols):
            raise ValueError(f"Target coordinates are outside raster bounds. "
                            f"Pixel coordinates: ({target_row}, {target_col}), "
                            f"Raster shape: ({rows}, {cols})")
        
        row_indices, col_indices = np.ogrid[:rows, :cols]
        
        if distance_type == 'euclidean':
            distances = np.sqrt((row_indices - target_row)**2 + (col_indices - target_col)**2)
        elif distance_type == 'manhattan':
            distances = np.abs(row_indices - target_row) + np.abs(col_indices - target_col)
        elif distance_type == 'chebyshev':
            distances = np.maximum(np.abs(row_indices - target_row), np.abs(col_indices - target_col))
        else:
            raise ValueError("distance_type must be 'euclidean', 'manhattan', or 'chebyshev'")

        if isinstance(distance_cells, (int, float)):
            mask = (distances <= distance_cells)
        else:
            raise ValueError("distance_cells must be a number or a tuple of (min_distance, max_distance)")
        
        result_array = np.where(mask, original_array, 0)
        
        return result_array
    
    potential_coverage_matrix = gdal.Open(os.path.join(coverage_matrix_path)).ReadAsArray()

    potential_targets = targets_df["boundary_patch_id"].tolist()

    potential_coverage_matrix = select_cells_at_distance(
        target_lat=1052166,
        target_lon=8572829,
        distance_cells=125,
        distance_type='euclidean'
    )

    potential_coverage_matrix = potential_coverage_matrix.astype(int)

    mask = np.isin(potential_coverage_matrix, potential_targets)
    potential_coverage_matrix[~mask] = 0
    potential_coverage_matrix = potential_coverage_matrix.astype(int)

    unique_values = np.unique(potential_coverage_matrix)
    non_zero_unique_values = unique_values[unique_values != 0]

    print("boundary patches to monitor:", non_zero_unique_values, "total numbers:", len(non_zero_unique_values))

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = mcolors.ListedColormap(['white', 'black'])
    im = ax.imshow(potential_coverage_matrix, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    legend_elements = [
        Patch(facecolor='black', edgecolor='black', label='Potential Coverage'),
        Patch(facecolor='white', edgecolor='black', label='No Coverage')
    ]

    ax.legend(handles=legend_elements, loc="upper right")
    plt.savefig(
        os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init", "potential_coverage_matrix.png"),
        bbox_inches="tight",
        dpi=500,
    )
    plt.close(fig)

    combinations_of_size_k = itertools.combinations(non_zero_unique_values, BUDGET_K)

    defender_strategies = []

    for combination in tqdm(combinations_of_size_k):
        strategy_vector = combination_to_binary_vector(combination, NUM_LANDSCAPE_CELLS)
        defender_strategies.append(strategy_vector)

    source_file = gdal.Open("mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif")

    cols = source_file.RasterXSize
    rows = source_file.RasterYSize
    projection = source_file.GetProjection()
    geotransform = source_file.GetGeoTransform()

    output_file = "guarding-policies/FPL-UE-v1/coverage_matrix_init/potential_coverage_matrix.tif"

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(potential_coverage_matrix.astype(np.uint8))

    source_file = None
    output_dataset = None

    return defender_strategies

def calculate_reward_for_strategy(defender_strategy, perturbed_reward):
    v = np.array(defender_strategy)
    total_reward = np.dot(v, perturbed_reward)
    return total_reward, v

def find_best_strategy_parallel(defender_strategies, perturbed_reward, n_processes=16):
    
    process_func = partial(
        calculate_reward_for_strategy,
        perturbed_reward=perturbed_reward
    )
    
    max_reward = float('-inf')
    best_strategy = None
    
    with mp.Pool(processes=n_processes) as pool:

        for total_reward, v in tqdm(pool.imap(process_func, defender_strategies, chunksize=4096)):
            if total_reward > max_reward:
                max_reward = total_reward
                best_strategy = v
    
    return best_strategy

def select_defender_strategy(
    defender_strategies,
    estimated_reward: np.ndarray,
    eta: float,
    gamma,
    NUM_LANDSCAPE_CELLS,
    budget_k
    ) -> np.ndarray:


    flag = np.random.random() < gamma 

    if flag: 

        # print("Random Strategy Selected")

        potential_coverage_matrix = gdal.Open(os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init/potential_coverage_matrix.tif")).ReadAsArray()
        unique_values = np.unique(potential_coverage_matrix)
        non_zero_unique_values = list(unique_values[unique_values != 0])
        random_sample = random.sample(non_zero_unique_values, budget_k)

        v_t = combination_to_binary_vector(random_sample, NUM_LANDSCAPE_CELLS)

    else:  

        # print("Optimal Strategy Selected")

        # n = len(estimated_reward)
        # z = np.random.exponential(scale=1/eta, size=n)
        # perturbed_reward = estimated_reward + z

        perturbed_reward = estimated_reward

        v_t = find_best_strategy_parallel(defender_strategies, perturbed_reward)

    return v_t

def run_abm(model_params, experiment_name, output_folder, NUM_STRATEGIC_TRAJECTORIES):




    with open(os.path.join(output_folder, "model_parameters.yaml"), "w") as configfile:
        yaml.dump(model_params, configfile, default_flow_style=False)




    num_strategic_trajectories  =  0

    while num_strategic_trajectories < NUM_STRATEGIC_TRAJECTORIES:

        batch_run_model(model_params, experiment_name, output_folder)

        runs = os.listdir(output_folder)

        num_strategic_trajectories  =  0

        for run in runs:
            flag = True

            if flag == True:
                num_strategic_trajectories += 1

            if flag == False:
                shutil.rmtree(os.path.join(output_folder, run))




    dict_of_attacked_targets = {}

    for simulation_folder in os.listdir(output_folder):

        try:

            df = pd.read_csv(os.path.join(output_folder, simulation_folder, "output_files/agent_data.csv"))
            df.dropna(subset=['ROW', 'COL'], inplace=True)

            unique_targets = df["target_attacked"].dropna().unique()

            for target in unique_targets:
                if target not in dict_of_attacked_targets:
                    dict_of_attacked_targets[target] = 0
                dict_of_attacked_targets[target] += 1
        
        except Exception as e:
            pass

    for target, count in dict_of_attacked_targets.items():
        print(f"Covered target {target} was attacked {count} times.")  

    df_dict_of_attacked_targets = pd.DataFrame.from_dict(dict_of_attacked_targets, orient='index', columns=['count'])

    df_dict_of_attacked_targets.reset_index(inplace=True)
    df_dict_of_attacked_targets.rename(columns={'index': 'target'}, inplace=True)
                                            
    df_dict_of_attacked_targets.to_csv(os.path.join(output_folder, "df_of_attacked_targets.csv"))
    
    targets_matrix = gdal.Open(os.path.join(coverage_matrix_path)).ReadAsArray()
    unique_values = np.unique(targets_matrix)
    #unique_values = unique_values[unique_values != 0]
    total_targets = len(unique_values)
    attacker_strategy_covered_targets = [0 for i in range(total_targets - 1)]


    for target in dict_of_attacked_targets:
        attacker_strategy_covered_targets[int(target - 1)] = 1


    return attacker_strategy_covered_targets

def step_utility_defender(attacker_strategy_i, defender_strategy_i, targets_df):

    attacker_strategy = np.array(attacker_strategy_i)
    defender_strategy = np.array(defender_strategy_i)

    r = (targets_df['reward'] - targets_df['penalty']).values
    r_t = [a * b for a, b in zip(attacker_strategy, r)]

    reward_01 = np.dot(defender_strategy, r_t)
    reward_02 = np.dot(attacker_strategy, targets_df['penalty'].values)

    return reward_01 + reward_02

def calculate_reward_for_strategy_best(defender_strategy, attacker_strategy_history, targets_df):

    v = np.array(defender_strategy)
    
    total_strategy_utility = 0
    for attacker_strategy in attacker_strategy_history:
        step_utility = step_utility_defender(attacker_strategy, v, targets_df)
        total_strategy_utility += step_utility
    
    return total_strategy_utility, v
    
def calculate_best_strategy(defender_strategies, attacker_strategy_history, targets_df, n_processes=16):



    process_func = partial(
        calculate_reward_for_strategy_best,
        attacker_strategy_history=attacker_strategy_history,
        targets_df=targets_df
    )
    
    max_reward = float('-inf')
    best_strategy = None
    
    with mp.Pool(processes=n_processes) as pool:
        
        for total_reward, v in tqdm(pool.imap(process_func, defender_strategies, chunksize=512)):
            if total_reward > max_reward:
                max_reward = total_reward
                best_strategy = v



    # def calculate_reward_wrapper(args):
    #     attacker_strategy, target_df = args
    #     return calculate_reward_for_strategy_best(
    #         defender_strategies,  
    #         attacker_strategy_history=attacker_strategy,
    #         targets_df=target_df
    #     )

    # process_args = list(zip(attacker_strategy_history, targets_df_history))

    # max_reward = float('-inf')
    # best_strategy = None

    # with mp.Pool(processes=n_processes) as pool:
        
    #     for total_reward, v in tqdm(pool.imap(calculate_reward_wrapper, process_args, chunksize=512)):
    #         if total_reward > max_reward:
    #             max_reward = total_reward
    #             best_strategy = v




    return best_strategy

def GR_algorithm(defender_strategy,
                 eta: float, 
                 gamma,
                 M: int, 
                 estimated_reward: np.ndarray, 
                 NUM_LANDSCAPE_CELLS, 
                 budget_k) -> np.ndarray:
    """
    Implements the GR (Geometric Resampling) Algorithm.
    """
    n = len(estimated_reward)
    K = np.zeros(n, dtype=int)
    k = 1
    
    while k <= M:

        v_tilde = select_defender_strategy(defender_strategy, estimated_reward, eta, gamma, NUM_LANDSCAPE_CELLS, budget_k)
        
        for i in range(n):
            if k < M and v_tilde[i] == 1 and K[i] == 0:
                K[i] = k
            elif k == M and K[i] == 0:
                K[i] = M
        
        if np.all(K > 0):
            break
            
        k += 1
    
    return K

def update_estimated_reward(
    estimated_reward: np.ndarray,
    K: np.ndarray,
    attacker_strategy: np.ndarray,
    defender_strategy: np.ndarray,
    targets_df: pd.DataFrame,
    w1: float,
    w2: float) -> np.ndarray:

    attacker_strategy = np.array(attacker_strategy)
    defender_strategy = np.array(defender_strategy)

    r = (targets_df['reward']*w1 - targets_df['penalty']*w2).values
    r_t = [a * b for a, b in zip(attacker_strategy, r)]

    updated_reward = estimated_reward.copy()

    protected_cells = np.where((defender_strategy == 1))[0]
    
    for idx in protected_cells:
        updated_reward[idx] += K[idx] * r_t[idx]
    
    return updated_reward

def plot_defender_regret(defender_regret_values):

    regret_values = np.array(defender_regret_values)
    steps = np.arange(1, len(regret_values) + 1)
    
    plt.figure(figsize=(6, 6))
    plt.plot(steps, regret_values, 'b-', label='FPL-UE')
    
    plt.xlabel('Step')
    plt.ylabel('Regret Value')
    plt.title('Defender Regret Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()

    plt.savefig('guarding-policies/FPL-UE-v1/coverage_matrix_init/defender_regret_plot.png', dpi=300, bbox_inches='tight')
    
    plt.close()

    return

def PLOT_CROP_DAMAGE(STEP_DAMAGES):

    regret_values = np.array(STEP_DAMAGES)
    steps = np.arange(1, len(regret_values) + 1)
    
    plt.figure(figsize=(6, 6))
    plt.plot(steps, regret_values, 'b-', label='FPL-UE')
    
    plt.xlabel('Step')
    plt.ylabel('crops raided')
    plt.title('Crop Raiding Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()

    plt.savefig('guarding-policies/FPL-UE-v1/coverage_matrix_init/crop_damage_with_step.png', dpi=300, bbox_inches='tight')
    
    plt.close()

    return

def calculate_defender_regret(defender_strategy_history, attacker_strategy_history, best_hindsight_strategy, targets_df):
    
    assert len(defender_strategy_history) == len(attacker_strategy_history)
    max_steps = len(defender_strategy_history)

    regret_i_hindsight = 0
    regret_i = 0

    for step in range(max_steps):
        attacker_strategy_i = attacker_strategy_history[step]
        defender_strategy_i = defender_strategy_history[step]

        r = (targets_df['reward'] - targets_df['penalty']).values
        r_t = [a * b for a, b in zip(attacker_strategy_i, r)]

        regret_i_hindsight += np.dot(best_hindsight_strategy, r_t)

    for step in range(max_steps):
        attacker_strategy_i = attacker_strategy_history[step]
        defender_strategy_i = defender_strategy_history[step]

        r = (targets_df['reward'] - targets_df['penalty']).values
        r_t = [a * b for a, b in zip(attacker_strategy_i, r)]

        regret_i += np.dot(defender_strategy_i, r_t)

    REGRET = (regret_i_hindsight - regret_i)/max_steps

    return REGRET

def clip_raster_by_latlon_extent(input_file, output_folder, latlon_extent):

    try:
        source_ds = gdal.Open(input_file, gdal.GA_Update)
        if source_ds is None:
            print("Error: Could not open the input file.")
            return
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    
    geo_transform = source_ds.GetGeoTransform()
    x_size = source_ds.RasterXSize
    y_size = source_ds.RasterYSize

    band = source_ds.GetRasterBand(1)
    raster_data = band.ReadAsArray()

    lon_min, lat_min, lon_max, lat_max = latlon_extent

    x_res = geo_transform[1]
    y_res = geo_transform[5] 

    x_coords = np.arange(x_size) * x_res + geo_transform[0]
    y_coords = np.arange(y_size) * y_res + geo_transform[3]

    x_out_of_bounds = (x_coords < lon_min) | (x_coords > lon_max)
    y_out_of_bounds = (y_coords < lat_min) | (y_coords > lat_max)

    x_mask, y_mask = np.meshgrid(x_out_of_bounds, y_out_of_bounds)
    out_of_bounds_mask = x_mask | y_mask

    raster_data[out_of_bounds_mask] = 0

    cmap = plt.cm.get_cmap('tab20').copy()
    cmap.set_under('white')


    source_file = gdal.Open(input_file)
    cols = source_file.RasterXSize
    rows = source_file.RasterYSize
    projection = source_file.GetProjection()
    geotransform = source_file.GetGeoTransform()
    output_file = os.path.join(output_folder, "targets_to_protect_study_area.tif")
    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Byte)
    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)
    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(raster_data.astype(np.uint8))
    source_file = None
    output_dataset = None
    source_ds = None


    # fig, ax = plt.subplots(figsize=(8, 8))

    # cax = ax.imshow(raster_data, cmap=cmap, vmin=0.1, 
    #                 extent=(geo_transform[0], geo_transform[0] + x_size * x_res, 
    #                         geo_transform[3] + y_size * y_res, geo_transform[3]))
    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')

    # plt.savefig(os.path.join(output_folder, "targets_to_protect.png"), dpi=600, bbox_inches='tight')

    return np.unique(raster_data)

def run_single_play(model_params, experiment_name, output_folder, MAX_GAME_STEPS, NUM_LANDSCAPE_CELLS, TARGETS, BUDGET_K, M,  max_gamma, min_gamma, num_steps_gamma_decay, eta, targets_df, NUM_STRATEGIC_TRAJECTORIES, w1, w2):



    #---------start with an initial estimate of target rewards: no prior information---------#
    # estimated_reward = np.zeros(NUM_LANDSCAPE_CELLS)
    #---------start with an initial estimate of target rewards: no prior information---------#

    #---------start with an initial estimate of target rewards: with prior information---------#
    estimated_reward = targets_df["reward"].values - targets_df["penalty"].values
    #---------start with an initial estimate of target rewards: with prior information---------#

    defender_strategy_history = []
    attacker_strategy_history = []

    defender_regret_values = []
    STEP_DAMAGES = []

    defender_strategies = generate_defender_strategies(BUDGET_K, NUM_LANDSCAPE_CELLS, targets_df)

    print("generated defender strategies!")
    
    for i in range(1, MAX_GAME_STEPS+1):

        print("\n----- GameStep", i ,"-----")

        gamma = max_gamma - (max_gamma - min_gamma) * (i / num_steps_gamma_decay)

        if gamma < min_gamma:
            gamma = min_gamma
            
        print(f"gamma: {gamma}")

        defender_strategy_i = select_defender_strategy(defender_strategies, estimated_reward, eta, gamma, NUM_LANDSCAPE_CELLS, BUDGET_K)

        print("Defender strategy:", defender_strategy_i)

        coverage_matrix = create_defender_coverage_matrix(defender_strategy_i)

        path = pathlib.Path(os.path.join(output_folder, "game_step_" + str(i)))
        path.mkdir(parents=True, exist_ok=True)

        path = pathlib.Path(os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init", "game_step_" + str(i)))
        path.mkdir(parents=True, exist_ok=True)
        
        plot_and_save_defender_coverage(coverage_matrix, os.path.join(output_folder, "game_step_" + str(i)))
        plot_and_save_defender_coverage(coverage_matrix, os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init", "game_step_" + str(i)))

        attacker_strategy_i = run_abm(model_params, experiment_name, os.path.join(output_folder, "game_step_" + str(i)), NUM_STRATEGIC_TRAJECTORIES)

        print("Attacker strategy:", attacker_strategy_i)

        print(f"Protected target IDs: {targets_df['boundary_patch_id'].loc[np.where(np.array(defender_strategy_i) == 1)[0]].tolist()}")
        print(f"Attacked target IDs: {targets_df['boundary_patch_id'].loc[np.where(np.array(attacker_strategy_i) == 1)[0]].tolist()}")

        make_trajectory_summary_plots_v1(os.path.join(output_folder, "game_step_" + str(i)), os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init", "game_step_" + str(i)), num_cropraiding_steps = 12)
        make_trajectory_summary_plots_v2(os.path.join(output_folder, "game_step_" + str(i)), os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init", "game_step_" + str(i)), num_cropraiding_steps = 12)
        make_trajectory_summary_plots_v3(os.path.join(output_folder, "game_step_" + str(i)), os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init", "game_step_" + str(i)))

        defender_strategy_history.append(defender_strategy_i)
        attacker_strategy_history.append(attacker_strategy_i)



        path = pathlib.Path(os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init", "game_step_" + str(i + 1)))
        path.mkdir(parents=True, exist_ok=True)


        targets_df, step_penalty = update_targets_df(output_folder=os.path.join(output_folder, "game_step_" + str(i)), targets_df=targets_df, current_game_step=i)


        K = GR_algorithm(defender_strategies, eta, gamma, M, estimated_reward, NUM_LANDSCAPE_CELLS, BUDGET_K)

        estimated_reward = update_estimated_reward(estimated_reward, K, attacker_strategy_i, defender_strategy_i, targets_df, w1, w2)

        df = pd.DataFrame(estimated_reward, columns=['reward_estimate'])
        df.to_csv(os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init", "game_step_" + str(i), "reward_estimate.csv"))





        best_defender_strategy_t = calculate_best_strategy(defender_strategies, attacker_strategy_history, targets_df)

        print("Best defender strategy:", best_defender_strategy_t)

        print("step utility for defender:", step_utility_defender(attacker_strategy_i, defender_strategy_i, targets_df))

        regret_i = calculate_defender_regret(defender_strategy_history, attacker_strategy_history, best_defender_strategy_t, targets_df)

        print("Defender regret:", regret_i)

        defender_regret_values.append(regret_i)
        STEP_DAMAGES.append(step_penalty)




    plot_defender_regret(defender_regret_values)
    PLOT_CROP_DAMAGE(STEP_DAMAGES)

    return  







def optimise_strategy(model_params, experiment_name, output_folder, NUM_LANDSCAPE_CELLS, TARGETS, BUDGET_K, MAX_GAME_STEPS, max_gamma, min_gamma, num_steps_gamma_decay, eta, M, NUM_STRATEGIC_TRAJECTORIES, w1, w2):




    #------------create a vector of random numbers for reward and penalty------------#
    # np.random.seed(42)  
    reward = np.random.uniform(0.0, 0.05, size=len(TARGETS))
    penalty = np.random.uniform(-0.05, 0.0, size=len(TARGETS))

    #create a dataframe with boundary_patch_id, reward and penalty columns
    targets_df = pd.DataFrame({
        "boundary_patch_id": [i for i in TARGETS],
        "reward": reward,
        "penalty": penalty
    })
    #------------create a vector of random numbers for reward and penalty------------#






    path = pathlib.Path(os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init", "game_step_1"))
    path.mkdir(parents=True, exist_ok=True)

    
    targets_df.to_csv(os.path.join("guarding-policies/FPL-UE-v1/coverage_matrix_init/game_step_1/boundary_patch_reward_penalty_matrix.csv"), index=False)



    run_single_play(model_params, experiment_name, output_folder, MAX_GAME_STEPS, NUM_LANDSCAPE_CELLS, TARGETS, BUDGET_K, M, max_gamma, min_gamma, num_steps_gamma_decay, eta, targets_df, NUM_STRATEGIC_TRAJECTORIES, w1, w2)












if __name__ == "__main__":

    boundary_raster_discretised = "boundary_raster_discretised_750m"

    global coverage_matrix_path   
    
    coverage_matrix_path = "guarding-policies/dynamic-guarding-model-v1/create-strategy-matrix-v2/" + boundary_raster_discretised + "/boundary_raster_discretised.tif"

    total_num_targets = len(np.unique(gdal.Open(coverage_matrix_path).ReadAsArray())) - 1

    TARGETS  = np.unique(gdal.Open(coverage_matrix_path).ReadAsArray())

    TARGETS = [x for x in TARGETS if x != 0]

    print("Total number of targets to protect:", total_num_targets)

    proximity_filter_parameter = [0.999]
    cost_function_threshold_parameter = [0]
    
    parameter_combinations = list(itertools.product(proximity_filter_parameter, cost_function_threshold_parameter))

    num_resources_k = [5]
    
    for ranger_proximity_threshold, cost_ranger_proximity_threshold in parameter_combinations:

        for k in num_resources_k: 

            BUDGET_K = k                   # Maximum number of cells that can be protected by the defenders at every time-step
            MAX_GAME_STEPS = 25                         # Maximum number of time-steps in the game
            max_gamma = 0.05                             # Exploration/Exploitation Trade-off parameter
            min_gamma = 0.0                              # Exploration/Exploitation Trade-off parameter
            num_steps_gamma_decay = 5                  # Exploration/Exploitation Trade-off parameter
            eta = 10.0                                   # reward perturbation parameter
            M = 30                                      # parameter in the GR algorithm
            w1 = 0.25                                    #weight of reward in reward-estimation
            w2 = 0.75                                    #weight of penalty in reward-estimation

            FPL_UE_params = (
                "budget_k_"
                + str(BUDGET_K)
                + "-max_game_steps_"
                + str(MAX_GAME_STEPS) 
                + "-max_gamma_"
                + str(max_gamma)
                + "-min_gamma_"
                + str(min_gamma)
                + "-num_steps_gamma_decay_"
                + str(num_steps_gamma_decay)
                + "-eta_"
                + str(eta)
                + "-M_"
                + str(M)
            )

            model_params = {
                    "year": 2010,
                    "month": "Mar",
                    "num_bull_elephants": 1,
                    "area_size": 1100,
                    "spatial_resolution": 30,
                    "max_food_val_cropland": 100,
                    "max_food_val_forest": 25,
                    "prob_food_forest": 0.10,
                    "prob_food_cropland": 1.0,
                    "prob_water_sources": 0.05,
                    "thermoregulation_threshold": 28,
                    "num_days_agent_survives_in_deprivation": 10,
                    "knowledge_from_fringe": 1500,
                    "prob_crop_damage": 0.05,
                    "prob_infrastructure_damage": 0.01,
                    "percent_memory_elephant": 0.375,
                    "radius_food_search": 750,
                    "radius_water_search": 750,
                    "radius_forest_search": 1500,
                    "fitness_threshold": 0.4,
                    "terrain_radius": 750,
                    "slope_tolerance": 30,
                    "num_processes": 12,
                    "iterations": 12,
                    "max_time_steps": 288 * 21,
                    "aggression_threshold_enter_cropland": 1.0,
                    "human_habituation_tolerance": 1.0,
                    "elephant_agent_visibility_radius": 500,
                    "plot_stepwise_target_selection": False,
                    "threshold_days_of_food_deprivation": 0,
                    "threshold_days_of_water_deprivation": 3,
                    "number_of_feasible_movement_directions": 4,
                    "track_in_mlflow": False,
                    "elephant_starting_location": "user_input",
                    "elephant_starting_latitude": [[1052166]],
                    "elephant_starting_longitude": [[8572829]],
                    "elephant_aggression_value": 0.8,
                    "elephant_crop_habituation": True,
                    "ranger_proximity_threshold": None,
                    "cost_ranger_proximity_threshold": None,
                    "num_protected_targets": k
                }
            
            NUM_STRATEGIC_TRAJECTORIES = 12

            experiment_name = "mitigation-measures-within-plantations"

            elephant_category = "solitary_bulls"

            starting_location = (
                "latitude-"
                + str(model_params["elephant_starting_latitude"])
                + "-longitude-"
                + str(model_params["elephant_starting_longitude"])
            )

            landscape_food_probability = (
                "landscape-food-probability-forest-"
                + str(model_params["prob_food_forest"])
                + "-cropland-"
                + str(model_params["prob_food_cropland"])
            )

            food_availability_sceanario = "random-food-distribition-within-agricultural-plots"

            water_availability_sceanario = "water-source-rivers-landscape-" + str(model_params["prob_water_sources"])

            food_memory_matrix_type = "random-memory-forest-and_plantation-fringe-model"
            
            water_memory_matrix_type = "full-memory-forest-and_plantation-model"

            num_days_agent_survives_in_deprivation = (
                "num_days_agent_survives_in_deprivation-"
                + str(model_params["num_days_agent_survives_in_deprivation"])
            )

            maximum_food_in_a_forest_cell = "maximum-food-in-a-forest-cell-" + str(
                model_params["max_food_val_forest"]
            )

            elephant_thermoregulation_threshold = (
                "thermoregulation-threshold-temperature-"
                + str(model_params["thermoregulation_threshold"])
            )

            threshold_food_derivation_days = "threshold_days_of_food_deprivation-" + str(
                model_params["threshold_days_of_food_deprivation"]
            )

            threshold_water_derivation_days = "threshold_days_of_water_deprivation-" + str(
                model_params["threshold_days_of_water_deprivation"]
            )

            slope_tolerance = "slope_tolerance-" + str(model_params["slope_tolerance"])

            elephant_aggression_value = "elephant_aggression_value_" + str(
                model_params["elephant_aggression_value"]
            )

            target_folder = f"num_protected_targets_" + str(model_params["num_protected_targets"])

            simulation_repeats = f'num_strategic_traj_{NUM_STRATEGIC_TRAJECTORIES}_num_iterations_{model_params["iterations"]}'



            output_folder = os.path.join(
                "guarding-policies/FPL-UE-v1/model-runs/game_model-v3_1-run-model-with-intervention-v2-no-knowledge-static-policy/",
                experiment_name,
                starting_location,
                elephant_category,
                food_availability_sceanario,
                landscape_food_probability,
                water_availability_sceanario,
                food_memory_matrix_type,
                water_memory_matrix_type,
                num_days_agent_survives_in_deprivation,
                maximum_food_in_a_forest_cell,
                elephant_thermoregulation_threshold,
                threshold_food_derivation_days,
                threshold_water_derivation_days,
                slope_tolerance,
                num_days_agent_survives_in_deprivation,
                elephant_aggression_value,
                str(model_params["year"]),
                str(model_params["month"]),
                target_folder,
                simulation_repeats,
                boundary_raster_discretised,
                FPL_UE_params
            )


            optimise_strategy(
                model_params=model_params,
                experiment_name=experiment_name,
                output_folder=output_folder,
                NUM_LANDSCAPE_CELLS = total_num_targets,
                TARGETS = TARGETS,
                BUDGET_K = BUDGET_K,
                MAX_GAME_STEPS = MAX_GAME_STEPS,
                max_gamma = max_gamma, 
                min_gamma = min_gamma,
                num_steps_gamma_decay = num_steps_gamma_decay,
                eta = eta,
                M = M,
                NUM_STRATEGIC_TRAJECTORIES = NUM_STRATEGIC_TRAJECTORIES,
                w1 = w1,
                w2 = w2
            )


