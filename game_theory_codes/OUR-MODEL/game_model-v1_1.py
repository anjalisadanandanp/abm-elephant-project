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

module = importlib.import_module('game_theory_codes.OUR-MODEL.abm_model_HEC_with_landscape_deterrent_policies')
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

    ds = gdal.Open(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix.tif"))

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

    raster_to_geojson(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix.tif"), os.path.join(output_folder, 'guarded_patches.geojson'))

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
            boundary_patches_guarded = gdal.Open(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix.tif")).ReadAsArray()
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

    ds = gdal.Open(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix.tif"))

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

    raster_to_geojson(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix.tif"), os.path.join(output_folder, 'guarded_patches.geojson'))

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
            boundary_patches_guarded = gdal.Open(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix.tif")).ReadAsArray()
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

    ds = gdal.Open(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix.tif"))

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

    raster_to_geojson(os.path.join(base_path, simulation_repeats[0], "env", "defender_coverage_matrix.tif"), os.path.join(output_folder, 'guarded_patches.geojson'))

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

    run_folder = os.path.join(output_folder, "game_step_" + str(int(current_game_step)))
    expts = os.listdir(run_folder)

    save_folder = os.path.join(os.getcwd(), "game_theory_codes/OUR-MODEL/coverage_matrix_init", "game_step_" + str(int(current_game_step)))
    os.makedirs(save_folder, exist_ok=True)

    landuse_matrix = gdal.Open(os.path.join(run_folder, expts[-1], "env", "LULC.tif")).ReadAsArray()

    row_size, col_size = landuse_matrix.shape
    xmin, xres, xskew, ymax, yskew, yres = gdal.Open(os.path.join(run_folder, expts[-1], "env", "LULC.tif")).GetGeoTransform()
    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
    LON_MIN,LAT_MIN = transform(inProj, outProj, xmin, ymax + yres*col_size)
    LON_MAX,LAT_MAX = transform(inProj, outProj, xmin + xres*row_size, ymax)

    boundary_patches = gdal.Open("create-strategy-matrix/boundary_raster_discretised.tif").ReadAsArray()
    agricultural_plts = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").ReadAsArray()

    geotransform = gdal.Open("create-landholding-matrix/agricultural_plots_assignment.tif").GetGeoTransform()
    ag_xmin, ag_xres, ag_xskew, ag_ymax, ag_yskew, ag_yres = geotransform
    ag_rows, ag_cols = agricultural_plts.shape

    food_matrix = gdal.Open(os.path.join(run_folder, expts[-1], "env", "food_matrix_0.1_0.1_.tif")).ReadAsArray()


    #-----------------build assciaiation matrix between agricultural and boundary patches-----------------#
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


    num_boundary_patchs = int(np.max(boundary_patches))
    num_agricultural_plts = int(np.max(agricultural_plts))

    association_matrix_num_visiting_trajs = np.zeros((num_boundary_patchs, num_agricultural_plts))

    for ids, expt in enumerate(expts):


        # -----------------------plot trajectories on cropland with crop raid episodes-----------------------#
        # try:

        #     fig, ax = plt.subplots(figsize=(8, 8))

        #     map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

        #     img = map.imshow(np.flipud(landuse_matrix), cmap = "Pastel2", extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 1)

        #     map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
        #     map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

        #     cb = fig.colorbar(img) 
        #     cb.remove()

        #     df = pd.read_csv(os.path.join(run_folder, expt, "output_files/agent_data.csv"))

        #     rows, cols = lat_lon_to_pixel(
        #         df["latitude"].values, df["longitude"].values, 
        #         ag_xmin, ag_ymax, ag_xres, ag_yres
        #     )
            
        #     landuse_values = landuse_matrix[rows, cols]

        #     indices = find_cropland_use_indices(landuse_values)

        #     outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
        #     longitude, latitude = transform(inProj, outProj, df["longitude"], df["latitude"])
        #     x_new, y_new = map(longitude,latitude)

        #     ax.plot(x_new, y_new, linewidth=0.25, alpha=0.5, color="black", zorder=1)

        #     ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='red', zorder=1)
        #     ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='red', zorder=1)

        #     for sequence in indices:

        #         if (sequence[1] - sequence[0]) >= num_cropraiding_steps:
                    
        #             outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
        #             longitude, latitude = transform(inProj, outProj, df["longitude"][sequence[0]:sequence[1]], df["latitude"][sequence[0]:sequence[1]])
        #             x_new, y_new = map(longitude,latitude)

        #             ax.plot(x_new, y_new, linewidth=0.25, zorder=2)
                    
        #             ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='black', zorder=2)
        #             ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='black', zorder=2)

        #     plt.savefig(os.path.join(save_folder, "cropland_intersection_traj_" + expt + "_.png"), dpi=500, bbox_inches="tight")
        #     plt.close()

        # except Exception as e:
        #     pass
        # -----------------------plot trajectories on cropland with crop raid episodes-----------------------#



        try:

            association_matrix_num_visiting_trajs_local = np.zeros((num_boundary_patchs, num_agricultural_plts))

            df = pd.read_csv(os.path.join(run_folder, expt, "output_files/agent_data.csv"))

            rows, cols = lat_lon_to_pixel(
                df["latitude"].values, df["longitude"].values, 
                ag_xmin, ag_ymax, ag_xres, ag_yres
            )
            
            landuse_values = landuse_matrix[rows, cols]

            indices = find_cropland_use_indices(landuse_values)
            
            for sequence in indices:

                if (sequence[1] - sequence[0]) >= num_cropraiding_steps:

                    if boundary_patches[rows[sequence[0]], cols[sequence[0]]] != 0:

                        for i in range(sequence[1] - sequence[0]):
    
                            if agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]] != 0:

                                association_matrix_num_visiting_trajs_local[int(boundary_patches[rows[sequence[0]], cols[sequence[0]]]),  int(agricultural_plts[rows[sequence[0] + i], cols[sequence[0] + i]])] = 1
            
        except Exception as e:
            # print(f"Error processing trajectory in {expt}: {e}")
            pass

        association_matrix_num_visiting_trajs += association_matrix_num_visiting_trajs_local

    print("max number of visiting trajectories in any agricultural plot:", np.max(association_matrix_num_visiting_trajs))
    print("min number of visiting trajectories in any agricultural plot:", np.min(association_matrix_num_visiting_trajs))

    df = pd.DataFrame(association_matrix_num_visiting_trajs, columns=[f"agricultural_plot_{i}" for i in range(num_agricultural_plts)],
                    index=[f"boundary_patch_{i}" for i in range(num_boundary_patchs)])
    
    df.to_csv(os.path.join(save_folder, "association_matrix_num_visiting_trajs.csv"))
    #-----------------build assciaiation matrix between agricultural and boundary patches-----------------#




    #-----------------------plot trajectories with boundary patch intersection-----------------------#
    fig, ax = plt.subplots(figsize=(8, 8))

    map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

    rainbow = plt.cm.rainbow
    colors = rainbow(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]  
    custom_cmap = mcolors.ListedColormap(colors)

    img = map.imshow(np.flipud(boundary_patches), cmap = custom_cmap, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], alpha = 1)

    map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
    map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

    cb = fig.colorbar(img) 
    cb.remove()

    for ids, expt in enumerate(expts):

        try:

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

            num_boundary_patches = np.unique(boundary_patches)

            # ax.plot(x_new, y_new, linewidth=0.25, alpha=0.5, color="black", zorder=1)
            # ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='red', zorder=1)
            # ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='red', zorder=1)

            for sequence in indices:

                if (sequence[1] - sequence[0]) >= num_cropraiding_steps:
                    
                    outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   
                    longitude, latitude = transform(inProj, outProj, df["longitude"][sequence[0]:sequence[1]], df["latitude"][sequence[0]:sequence[1]])
                    x_new, y_new = map(longitude,latitude)

                    ax.plot(x_new, y_new, linewidth=0.25, zorder=2)
                    
                    ax.scatter(x_new[0], y_new[0], 0.5, marker='o', color='black', zorder=2)
                    ax.scatter(x_new[-1], y_new[-1], 0.5, marker='^', color='black', zorder=2)

        except Exception as e:
            # print(f"Error processing {expt}: {e}")
            continue

    plt.savefig(os.path.join(save_folder, "boundary_patch_intersection_trajs_.png"), dpi=500, bbox_inches="tight")
    plt.close()
    #-----------------------plot trajectories with boundary patch intersection-----------------------#


                

    #-----------------------plot boundary patch association-----------------------#
    df = pd.read_csv(os.path.join(save_folder, "association_matrix_num_visiting_trajs.csv"))

    for index, row in df.iterrows():

        if index == 0:
            pass

        else:

            non_zero_items = [(col, val) for col, val in row.items() if val != 0]

            row_name = row['Unnamed: 0']
            col_names = [col for col in df.columns if col != 'Unnamed: 0']

            boundary_patch_id = int(row_name.split("_")[-1])

            # print(f"Boundary Patch ID: {boundary_patch_id}")

            matrix_to_plot = np.zeros((ag_rows, ag_cols))

            mask = boundary_patches == boundary_patch_id

            matrix_to_plot[mask] = 1

            flag = False
            
            for col, val in non_zero_items:

                try:
                    agricultural_plot_id = int(col.split("_")[-1])
                    # print(f"  {col}: {val}")
                    ag_mask = agricultural_plts == agricultural_plot_id
                    matrix_to_plot[ag_mask] = 2
                    if np.any(ag_mask):
                        flag = True
                except:
                    pass

            if os.path.exists(os.path.join(save_folder, "boundary_patch_association_" + str(boundary_patch_id) + "_.png")):
                os.remove(os.path.join(save_folder, "boundary_patch_association_" + str(boundary_patch_id) + "_.png"))

            if flag == True:

                fig, ax = plt.subplots(figsize=(8, 8))

                map = Basemap(llcrnrlon=LON_MIN,llcrnrlat=LAT_MIN,urcrnrlon=LON_MAX,urcrnrlat=LAT_MAX, epsg=4326, resolution='l')

                colors = ["white", "red", "forestgreen"]
                custom_cmap = mcolors.ListedColormap(colors)
                
                img = map.imshow(np.flipud(matrix_to_plot), cmap=custom_cmap, interpolation='nearest', zorder=1)

                map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
                map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

                with open('mesageo_elephant_project/elephant_project/geojson_files/landuse_10.geojson', 'r') as f:
                    geojson_object = geojson.load(f)

                for feature in geojson_object['features']:
                    coords = feature['geometry']['coordinates'][0]
                    coords = [transform(inProj, outProj, lon, lat) for lon, lat in coords]
                    coords = [(lon, lat) for lon, lat in coords]
                    lon, lat = zip(*coords)

                    # plt.fill(lon, lat, color='yellow', alpha=0.10, zorder=2)

                    map.plot(lon, lat, marker=None, color='black', linewidth=0.25, zorder=1) 

                plt.savefig(os.path.join(save_folder, "boundary_patch_association_" + str(boundary_patch_id) + "_.png"), dpi=300, bbox_inches="tight")
                plt.close()
    #-----------------------plot boundary patch association-----------------------#




    #-----------------make boundary patch association dataframe-----------------#
    df = pd.read_csv(os.path.join(save_folder, "association_matrix_num_visiting_trajs.csv"))

    column_boundary_ids = []
    column_food_value = []

    for index, row in df.iterrows():

        if index == 0:
            pass

        else:

            non_zero_items = [(col, val) for col, val in row.items() if val != 0]

            row_name = row['Unnamed: 0']
            col_names = [col for col in df.columns if col != 'Unnamed: 0']

            boundary_patch_id = int(row_name.split("_")[-1])

            boundarymask = boundary_patches == boundary_patch_id
            associated_plots = []
            food_within_plots = []

            flag = False
            
            for col, val in non_zero_items:

                try:
                    agricultural_plot_id = int(col.split("_")[-1])
                    ag_mask = agricultural_plts == agricultural_plot_id
                    if np.any(ag_mask):
                        flag = True
                        associated_plots.append(agricultural_plot_id)

                        foodmask = food_matrix[ag_mask]
                        food_within_plots.append(np.sum(foodmask))
                except:
                    pass

            if flag:
                print(f"Boundary Patch ID: {boundary_patch_id}, Associated Agricultural Plots: {associated_plots}, Food within Plots: {food_within_plots}")
                total_food = np.sum(food_within_plots)

            else:
                total_food = 0

            column_boundary_ids.append(boundary_patch_id)
            column_food_value.append(total_food)

    df_new = pd.DataFrame({
        "boundary_patch_id": column_boundary_ids,
        "total_food_value": column_food_value
    })
    df_new.to_csv(os.path.join(save_folder, "boundary_patch_association_matrix.csv"), index=False)
    #-----------------make boundary patch associtaion dataframe-----------------#




    #-----------------make total trajectory visiting matrix-----------------#
    def lat_lon_to_pixel(lat, lon, xmin, ymax, xres, yres):
        """Convert lat/lon coordinates to pixel coordinates"""
        col = int((lon - xmin) / xres)
        row = int((ymax - lat) / abs(yres))  
        return row, col

    def get_agricultural_plot_values(lat_array, lon_array, ag_plots, ag_geotransform):
        """Get agricultural plot values for trajectory points"""
        ag_xmin, ag_xres, ag_xskew, ag_ymax, ag_yskew, ag_yres = ag_geotransform
        ag_rows, ag_cols = ag_plots.shape
        
        plot_values = []
        
        for lat, lon in zip(lat_array, lon_array):
            row, col = lat_lon_to_pixel(lat, lon, ag_xmin, ag_ymax, ag_xres, ag_yres)
            
            if 0 <= row < ag_rows and 0 <= col < ag_cols:
                plot_value = ag_plots[row, col]
                plot_values.append(plot_value)
            else:
                plot_values.append(np.nan)
        
        return np.array(plot_values)
    
    base_folder = run_folder
    
    folders = os.listdir(base_folder)

    attack_numbers = {}
    plot_ids = np.unique(agricultural_plts)
    
    for plot_id in plot_ids:
        attack_numbers[plot_id] = 0

    ds = gdal.Open(os.path.join(base_folder, folders[-1], "env/slope_matrix.tif"))
    data = ds.ReadAsArray()
    data = np.flip(data, axis=0)
    row_size, col_size = data.shape
    xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()

    for folder in folders:

        try:
            data = pd.read_csv(os.path.join(base_folder, folder, "output_files/agent_data.csv"))

            lat = data["latitude"]
            lon = data["longitude"]

            ag_plot_values = get_agricultural_plot_values(lat, lon, agricultural_plts, geotransform)
            unique_plots = np.unique(ag_plot_values[~np.isnan(ag_plot_values)])

            for plot in unique_plots:
                attack_numbers[plot] += 1
            
        except Exception as e:
            # print(f"Error processing folder {folder}: {e}")
            pass
    
    attack_matrix = np.zeros_like(agricultural_plts)

    for plot_id, attack_count in attack_numbers.items():

        if plot_id == 0:
            pass

        else:
            mask = agricultural_plts == plot_id
            attack_matrix[mask] = attack_count
    #-----------------make total trajectory visiting matrix-----------------#




    #-----------------make total trajectory visiting matrix plot-----------------#
    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(attack_matrix, vmin=0, vmax=10, cmap = plt.cm.coolwarm, interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label('Number of Attacks', rotation=270, labelpad=15)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(os.path.join(save_folder, "total_trajectory_visiting.png"), dpi=300, bbox_inches="tight")

    plt.close()
    #-----------------make total trajectory visiting matrix plot-----------------#


            


    # all_rewards = []

    # i = 1

    # while i <= current_game_step:

    #     print("---step---", i)

    #     rewards = []

    #     out_save_folder = os.path.join(os.getcwd(), "game_theory_codes/OUR-MODEL/coverage_matrix_init", "game_step_" + str(int(i)))
    #     num_trajs_threatening = pd.read_csv(os.path.join(out_save_folder, "association_matrix_num_visiting_trajs.csv"))
    #     num_food_resources_under_attack = pd.read_csv(os.path.join(out_save_folder, "boundary_patch_association_matrix.csv"))

    #     for boundary_id in num_food_resources_under_attack["boundary_patch_id"].unique():

    #         reward = 0.0

    #         rows = num_trajs_threatening[num_trajs_threatening["Unnamed: 0"] == f"boundary_patch_{boundary_id}"]

    #         num_intersecting_trajs = rows.iloc[0, 1:].sum()

    #         if num_intersecting_trajs == 0:
    #             reward += 0.0

    #         else:
    #             reward += num_food_resources_under_attack[num_food_resources_under_attack["boundary_patch_id"] == boundary_id]["total_food_value"].values[0]*num_intersecting_trajs

    #             print(f"Boundary Patch ID: {boundary_id}, Reward: {reward}")

    #         rewards.append(reward)

    #     all_rewards.append(rewards)

    #     i += 1



    # all_rewards = np.array(all_rewards)

    # # print("reward update before normalisation:", all_rewards)

    # print("shape of reward update matrix:", all_rewards.shape)

    # global_min = np.min(all_rewards)
    # global_max = np.max(all_rewards)

    # normalized_all = (all_rewards - global_min) / (global_max - global_min) / 2

    # all_rewards = np.mean(normalized_all, axis=0).flatten().tolist()

    # # k = min(current_game_step, 3)
    # # all_rewards = np.sort(normalized_all, axis=0)[-k:].mean(axis=0).flatten().tolist()

    # print("reward update:", all_rewards)

    # all_penalties = [-r for r in all_rewards]

    # targets_df["reward"] = np.array(all_rewards)
    # targets_df["penalty"] = np.array(all_penalties)

    # targets_df.to_csv(os.path.join(save_folder, "boundary_patch_reward_penalty_matrix.csv"), index=False)

    return

def create_defender_coverage_matrix(defender_strategy):

    potential_coverage_matrix = gdal.Open(os.path.join("game_theory_codes/OUR-MODEL/coverage_matrix_init/potential_coverage_matrix.tif")).ReadAsArray()
    
    coverage_matrix = np.zeros_like(potential_coverage_matrix)

    target_ids = [index + 1 for index, value in enumerate(defender_strategy) if value != 0]

    for target_id in target_ids:
        mask = potential_coverage_matrix == target_id
        coverage_matrix[mask] = target_id

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





    source_file = gdal.Open("game_theory_codes/OUR-MODEL/coverage_matrix_init/potential_coverage_matrix.tif")

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

        dataset = gdal.Open("create-strategy-matrix/boundary_raster_discretised.tif")
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
    
    potential_coverage_matrix = gdal.Open(os.path.join("create-strategy-matrix/boundary_raster_discretised.tif")).ReadAsArray()

    potential_targets = targets_df["boundary_patch_id"].tolist()

    potential_coverage_matrix = select_cells_at_distance(
        target_lat=1049237,
        target_lon=8570917,
        distance_cells=175,
        distance_type='euclidean'
    )

    potential_coverage_matrix = potential_coverage_matrix.astype(int)

    mask = np.isin(potential_coverage_matrix, potential_targets)
    potential_coverage_matrix[~mask] = 0
    potential_coverage_matrix = potential_coverage_matrix.astype(int)

    unique_values = np.unique(potential_coverage_matrix)
    non_zero_unique_values = unique_values[unique_values != 0]

    print("boundary patches:", non_zero_unique_values, "total numbers:", len(non_zero_unique_values))

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
        os.path.join("game_theory_codes/OUR-MODEL/coverage_matrix_init", "potential_coverage_matrix.png"),
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

    output_file = "game_theory_codes/OUR-MODEL/coverage_matrix_init/potential_coverage_matrix.tif"

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

        potential_coverage_matrix = gdal.Open(os.path.join("game_theory_codes/OUR-MODEL/coverage_matrix_init/potential_coverage_matrix.tif")).ReadAsArray()
        unique_values = np.unique(potential_coverage_matrix)
        non_zero_unique_values = list(unique_values[unique_values != 0])
        random_sample = random.sample(non_zero_unique_values, budget_k)

        v_t = combination_to_binary_vector(random_sample, NUM_LANDSCAPE_CELLS)

    else:  

        # n = len(estimated_reward)
        # z = np.random.exponential(scale=1/eta, size=n)
        # perturbed_reward = estimated_reward + z

        perturbed_reward = estimated_reward

        v_t = find_best_strategy_parallel(defender_strategies, perturbed_reward)

    return v_t

def run_abm(model_params, experiment_name, output_folder):

    with open(os.path.join(output_folder, "model_parameters.yaml"), "w") as configfile:
        yaml.dump(model_params, configfile, default_flow_style=False)




    batch_run_model(model_params, experiment_name, output_folder)




    potential_coverage_matrix = gdal.Open(os.path.join("game_theory_codes/OUR-MODEL/coverage_matrix_init/potential_coverage_matrix.tif")).ReadAsArray()
    
    trajectory_matrix = np.zeros_like(potential_coverage_matrix, dtype=np.uint8)

    num_simulations = 0

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
        
            rows = df['ROW'].astype(int).values
            cols = df['COL'].astype(int).values
        
            mask = (0 <= rows) & (0 <= cols)
            valid_rows = rows[mask]
            valid_cols = cols[mask]
            
            trajectory_matrix[valid_rows, valid_cols] = 1

            num_simulations += 1
        
        except Exception as e:
            pass

    mask = potential_coverage_matrix == 0
    trajectory_matrix[mask] = 0



    trajectory_matrix = trajectory_matrix/num_simulations



    for target, count in dict_of_attacked_targets.items():
        print(f"Covered target {target} was attacked {count} times.")   \


    

    fig, ax = plt.subplots(figsize=(8, 8))
    
    cmap = 'coolwarm'
    
    im = ax.imshow(trajectory_matrix, cmap=cmap, vmin=0, vmax=np.max(trajectory_matrix))
    
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = plt.colorbar(im, shrink=0.5)

    ticks = np.linspace(0, np.max(trajectory_matrix), num=5) 
    cbar.set_label("Attack Probability", rotation=90)
    cbar.set_ticks(ticks)

    plt.savefig(
        os.path.join(output_folder, "attacker_strategy_matrix.png"),
        bbox_inches="tight",
        dpi=300,
    )




    source_file = gdal.Open("create-strategy-matrix/boundary_raster_discretised.tif")

    cols = source_file.RasterXSize
    rows = source_file.RasterYSize
    projection = source_file.GetProjection()
    geotransform = source_file.GetGeoTransform()

    output_file = os.path.join(output_folder, "attacker_strategy_matrix.tif")

    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_file, cols, rows, 1, gdal.GDT_Float32)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(trajectory_matrix.astype(np.float32))

    source_file = None
    output_dataset = None



    targets_matrix = gdal.Open("create-strategy-matrix/boundary_raster_discretised.tif").ReadAsArray()
    unique_values = np.unique(targets_matrix)
    non_zero_unique_values = unique_values[unique_values != 0]
    total_targets = len(non_zero_unique_values)
    attacker_strategy_covered_targets = [0 for i in range(total_targets - 1)]


    for target in dict_of_attacked_targets:
        attacker_strategy_covered_targets[int(target - 1)] = 1


    targets_matrix = gdal.Open("create-strategy-matrix/boundary_raster_discretised.tif").ReadAsArray()
    unique_values = np.unique(targets_matrix)
    non_zero_unique_values = unique_values[unique_values != 0]
    total_targets = len(non_zero_unique_values)
    attacker_strategy_uncovered_targets = [0 for i in range(total_targets - 1)]

    coverage_matrix = gdal.Open("create-strategy-matrix/boundary_raster_discretised.tif").ReadAsArray()
    plantation_rows, plantation_cols = np.where(coverage_matrix != 0)
    unique_values, counts = np.unique(coverage_matrix, return_counts=True)
    value_counts = dict(zip(unique_values, counts))

    for row, col in zip(plantation_rows, plantation_cols):
        if trajectory_matrix[row, col] > 0:
            attacker_strategy_uncovered_targets[int(coverage_matrix[row,col]-1)] += trajectory_matrix[row, col]

    def probabilistic_binary_conversion(original_list):

        filtered_list = [value for value in original_list if value != 0]
        percentile_value = np.percentile(filtered_list, 50)

        binary_list = []
        
        for value in original_list:

            probability = min(1, value / percentile_value)
            binary_value = np.random.binomial(1, probability)
            binary_list.append(binary_value)
        
        return binary_list
    
    attacker_strategy_uncovered_targets = probabilistic_binary_conversion(attacker_strategy_uncovered_targets)



    attacker_strategy = [min(1, v1+v2) for v1, v2 in zip(attacker_strategy_covered_targets, attacker_strategy_uncovered_targets)]
 


    return attacker_strategy

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
    targets_df: pd.DataFrame) -> np.ndarray:

    attacker_strategy = np.array(attacker_strategy)
    defender_strategy = np.array(defender_strategy)

    r = (targets_df['reward'] - targets_df['penalty']).values
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

    plt.savefig('game_theory_codes/OUR-MODEL/defender_regret_plot.png', dpi=300, bbox_inches='tight')
    
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
     
def run_single_play(model_params, experiment_name, output_folder, MAX_GAME_STEPS, NUM_LANDSCAPE_CELLS, BUDGET_K, M,  max_gamma, min_gamma, num_steps_gamma_decay, eta, targets_df):

    #---------start with an initial estimate of target rewards: no prior information---------#
    estimated_reward = np.zeros(NUM_LANDSCAPE_CELLS)
    #---------start with an initial estimate of target rewards: no prior information---------#

    #---------start with an initial estimate of target rewards: with prior information---------#
    # estimated_reward = targets_df["reward"].values - targets_df["penalty"].values
    #---------start with an initial estimate of target rewards: with prior information---------#

    defender_strategy_history = []
    attacker_strategy_history = []

    defender_regret_values = []

    defender_strategies = generate_defender_strategies(BUDGET_K, NUM_LANDSCAPE_CELLS, targets_df)
    
    for i in range(1, MAX_GAME_STEPS+1):

        print("\n----- GameStep", i ,"-----")

        gamma = max_gamma - (max_gamma - min_gamma) * (i / num_steps_gamma_decay)

        print(f"gamma: {gamma}")

        defender_strategy_i = select_defender_strategy(defender_strategies, estimated_reward, eta, gamma, NUM_LANDSCAPE_CELLS, BUDGET_K)

        print("Defender strategy:", defender_strategy_i)

        coverage_matrix = create_defender_coverage_matrix(defender_strategy_i)

        path = pathlib.Path(os.path.join(output_folder, "game_step_" + str(i)))
        path.mkdir(parents=True, exist_ok=True)

        path = pathlib.Path(os.path.join("game_theory_codes/OUR-MODEL/coverage_matrix_init", "game_step_" + str(i + 1)))
        path.mkdir(parents=True, exist_ok=True)

        plot_and_save_defender_coverage(coverage_matrix, os.path.join(output_folder, "game_step_" + str(i)))
        plot_and_save_defender_coverage(coverage_matrix, os.path.join("game_theory_codes/OUR-MODEL/coverage_matrix_init"))

        attacker_strategy_i = run_abm(model_params, experiment_name, os.path.join(output_folder, "game_step_" + str(i)))

        print("Attacker strategy:", attacker_strategy_i)

        print(f"Protected target IDs: {targets_df['boundary_patch_id'].loc[np.where(np.array(defender_strategy_i) == 1)[0]].tolist()}")
        print(f"Attacked target IDs: {targets_df['boundary_patch_id'].loc[np.where(np.array(attacker_strategy_i) == 1)[0]].tolist()}")

        make_trajectory_summary_plots_v1(os.path.join(output_folder, "game_step_" + str(i)), os.path.join("game_theory_codes/OUR-MODEL/coverage_matrix_init", "game_step_" + str(i)), num_cropraiding_steps = 12)
        make_trajectory_summary_plots_v2(os.path.join(output_folder, "game_step_" + str(i)), os.path.join("game_theory_codes/OUR-MODEL/coverage_matrix_init", "game_step_" + str(i)), num_cropraiding_steps = 12)
        make_trajectory_summary_plots_v3(os.path.join(output_folder, "game_step_" + str(i)), os.path.join("game_theory_codes/OUR-MODEL/coverage_matrix_init", "game_step_" + str(i)))

        defender_strategy_history.append(defender_strategy_i)
        attacker_strategy_history.append(attacker_strategy_i)




        update_targets_df(output_folder=output_folder, targets_df=targets_df, current_game_step=i)




        K = GR_algorithm(defender_strategies, eta, gamma, M, estimated_reward, NUM_LANDSCAPE_CELLS, BUDGET_K)

        estimated_reward = update_estimated_reward(estimated_reward, K, attacker_strategy_i, defender_strategy_i, targets_df)

        df = pd.DataFrame(estimated_reward, columns=['reward_estimate'])
        df.to_csv(os.path.join("game_theory_codes/OUR-MODEL/coverage_matrix_init", "game_step_" + str(i), "reward_estimate.csv"))





        best_defender_strategy_t = calculate_best_strategy(defender_strategies, attacker_strategy_history, targets_df)

        print("Best defender strategy:", best_defender_strategy_t)

        print("step utility for defender:", step_utility_defender(attacker_strategy_i, defender_strategy_i, targets_df))

        regret_i = calculate_defender_regret(defender_strategy_history, attacker_strategy_history, best_defender_strategy_t, targets_df)

        print("Defender regret:", regret_i)

        defender_regret_values.append(regret_i)




    plot_defender_regret(defender_regret_values)

    return  







def optimise_strategy(model_params, experiment_name, output_folder, BUDGET_K, MAX_GAME_STEPS, max_gamma, min_gamma, num_steps_gamma_decay, eta, M):

    NUM_LANDSCAPE_CELLS = 238      # Total number of landscape cells within the simulation extent




    #------------create a vector of random numbers for reward and penalty------------#
    # np.random.seed(42)  
    # reward = np.random.uniform(0.0, 0.05, size=NUM_LANDSCAPE_CELLS)
    # penalty = np.random.uniform(-0.05, 0.0, size=NUM_LANDSCAPE_CELLS)

    # #create a dataframe with boundary_patch_id, reward and penalty columns
    # targets_df = pd.DataFrame({
    #     "boundary_patch_id": [i + 1 for i in range(NUM_LANDSCAPE_CELLS)],
    #     "reward": reward,
    #     "penalty": penalty
    # })
    #------------create a vector of random numbers for reward and penalty------------#




    targets_df = pd.read_csv("assign_rewards_and_penalties/boundary_patch_reward_penalty_matrix.csv")




    path = pathlib.Path(os.path.join("game_theory_codes/OUR-MODEL/coverage_matrix_init", "game_step_1"))
    path.mkdir(parents=True, exist_ok=True)
    
    targets_df.to_csv(os.path.join("game_theory_codes/OUR-MODEL/coverage_matrix_init/game_step_1/boundary_patch_reward_penalty_matrix.csv"), index=False)



    run_single_play(model_params, experiment_name, output_folder, MAX_GAME_STEPS, NUM_LANDSCAPE_CELLS, BUDGET_K, M, max_gamma, min_gamma, num_steps_gamma_decay, eta, targets_df)












if __name__ == "__main__":

    model_params = {
            "year": 2010,
            "month": "Mar",
            "num_bull_elephants": 1,
            "area_size": 1100,
            "spatial_resolution": 30,
            "max_food_val_cropland": 100,
            "max_food_val_forest": 5,
            "prob_food_forest": 0.10,
            "prob_food_cropland": 0.10,
            "prob_water_sources": 1.0,
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
            "slope_tolerance": 35,
            "num_processes": 42,
            "iterations": 42,
            "max_time_steps": 288 * 30,
            "aggression_threshold_enter_cropland": 1.0,
            "human_habituation_tolerance": 1.0,
            "elephant_agent_visibility_radius": 500,
            "plot_stepwise_target_selection": False,
            "threshold_days_of_food_deprivation": 0,
            "threshold_days_of_water_deprivation": 3,
            "number_of_feasible_movement_directions": 3,
            "track_in_mlflow": False,
            "elephant_starting_location": "user_input",
            "elephant_starting_latitude": 1049237,
            "elephant_starting_longitude": 8570917,
            "elephant_aggression_value": 0.8,
            "elephant_crop_habituation": False
        }

    BUDGET_K = 10                               # Maximum number of cells that can be protected by the defenders at every time-step
    MAX_GAME_STEPS = 35                         # Maximum number of time-steps in the game
    max_gamma = 1.0                             # Exploration/Exploitation Trade-off parameter
    min_gamma = 0.20                            # Exploration/Exploitation Trade-off parameter
    num_steps_gamma_decay = 10                  # Exploration/Exploitation Trade-off parameter
    eta = 1.0                                   # reward perturbation parameter
    M = 30                                      # parameter in the GR algorithm

    experiment_name = "mitigation-measures-within-plantations-FPL-UE_v1_1/" 

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

    food_availability_sceanario = "random-food-distribition-within-agricultural-plots-and-other-plantation-cells"

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

    game_params = "budget_k_" + str(BUDGET_K) + "_MAX_GAME_STEPS_" + str(MAX_GAME_STEPS)

    output_folder = os.path.join(
        os.getcwd(),
        "game_theory_codes/OUR-MODEL",
        experiment_name,
        FPL_UE_params,
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
        str(model_params["month"])
    )

    optimise_strategy(
        model_params=model_params,
        experiment_name=experiment_name,
        output_folder=output_folder,
        BUDGET_K = BUDGET_K,
        MAX_GAME_STEPS = MAX_GAME_STEPS,
        max_gamma = max_gamma, 
        min_gamma = min_gamma,
        num_steps_gamma_decay = num_steps_gamma_decay,
        eta = eta,
        M = M
    )


