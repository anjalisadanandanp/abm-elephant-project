import os
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import pathlib
import yaml
import shutil
from osgeo import gdal
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pyproj import Proj, transform  
from mpl_toolkits.basemap import Basemap    
import rasterio
import matplotlib.patches as mpatches
from rasterio.features import shapes
import fiona
import geojson
import matplotlib.cm as cm
import shutil

import warnings
warnings.filterwarnings("ignore")


fontsize = 12
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
    
    simulation_repeats = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

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
    
    simulation_repeats = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

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

    simulation_repeats = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

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

def create_defender_coverage_matrix(targets_to_cover):

    potential_coverage_matrix = gdal.Open(os.path.join("game_theory_codes/OUR-MODEL/coverage_matrix_init/potential_coverage_matrix.tif")).ReadAsArray()
    
    coverage_matrix = np.zeros_like(potential_coverage_matrix)

    target_ids = [value for index, value in enumerate(targets_to_cover) if value != 0]

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

def run_abm(model_params, experiment_name, output_folder, NUM_STRATEGIC_TRAJECTORIES):




    with open(os.path.join(output_folder, "model_parameters.yaml"), "w") as configfile:
        yaml.dump(model_params, configfile, default_flow_style=False)




    num_strategic_trajectories  =  0

    while num_strategic_trajectories < NUM_STRATEGIC_TRAJECTORIES:

        batch_run_model(model_params, experiment_name, output_folder)

        runs = os.listdir(output_folder)

        num_strategic_trajectories  =  0

        for run in runs:
            if os.path.isdir(os.path.join(output_folder, run)):
                agent_df = pd.read_csv(os.path.join(output_folder, run, "output_files", "agent_data.csv"))

                targets_attacked = agent_df["target_attacked"].dropna().unique()

                COUNTS = {}
                for target in targets_attacked:
                    agent_df_attacking = agent_df[agent_df["target_attacked"] == target]
                    COUNTS[target] = len(agent_df_attacking)
                COUNTS = dict(sorted(COUNTS.items(), key=lambda item: item[1], reverse=True))

                if len(targets_attacked) == 0:
                    num_strategic_trajectories += 1

                else:
                    flag = True
                    for target in targets_attacked:
                        agent_df_attacking = agent_df[agent_df["target_attacked"] == target]
                        if len(agent_df_attacking) > 12:
                            flag = False

                    if flag == True:
                        num_strategic_trajectories += 1

                    if flag == False:
                        shutil.rmtree(os.path.join(output_folder, run))






    TOTAL_DAMAGE_VALUE = 0
    num_simulation_repeats = 0

    for simulation_folder in os.listdir(output_folder):

        try:

            df = pd.read_csv(os.path.join(output_folder, simulation_folder, "output_files/agent_data.csv"))
            landscape_cell_status = gdal.Open(os.path.join(output_folder, simulation_folder, "env/landscape_cell_status.tif"))
            landscape_cell_status_matrix = landscape_cell_status.ReadAsArray()

            food_matrix = gdal.Open(os.path.join(output_folder, simulation_folder, "env/food_matrix_0.1_0.1_.tif")).ReadAsArray()
            
            rows = df['ROW'].values.astype(int)
            cols = df['COL'].values.astype(int)
        
            matrix_rows, matrix_cols = landscape_cell_status_matrix.shape
            
            agent_locations_matrix = np.zeros((matrix_rows, matrix_cols))

            agent_values = []
            target_values_under_attack = []
            for r, c in zip(rows, cols):
                if 0 <= r < matrix_rows and 0 <= c < matrix_cols:
                    value = landscape_cell_status_matrix[r, c]
                    agent_values.append(value)
                    agent_locations_matrix[r, c] = 1 

                    if value == 2:
                        target_values_under_attack.append(food_matrix[r,c])

            fig, ax = plt.subplots(figsize=(6,6))
            im = ax.imshow(agent_locations_matrix, cmap='viridis')
            fig.colorbar(im, ax=ax, label='Agent Locations (1 = agent present)', shrink=0.5)
            ax.set_title(f'{simulation_folder}')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(os.path.join(output_folder, f"{simulation_folder}.png"), bbox_inches="tight", dpi=300)
            plt.close()

            num_simulation_repeats += 1

            TOTAL_DAMAGE_VALUE += sum(target_values_under_attack)

        except Exception as e:
            pass

    return TOTAL_DAMAGE_VALUE

def run_single_play(model_params, experiment_name, output_folder, NUM_STRATEGIC_TRAJECTORIES, NUM_GAME_STEPS):

    damage_values = []

    for step in range(1, NUM_GAME_STEPS+1):

        path = pathlib.Path(os.path.join(output_folder, "game_step_" + str(step)))
        path.mkdir(parents=True, exist_ok=True)

        STEP_DAMAGE_VALUE = run_abm(model_params, experiment_name, os.path.join(output_folder, "game_step_" + str(step)), NUM_STRATEGIC_TRAJECTORIES)

        damage_values.append(STEP_DAMAGE_VALUE)

        make_trajectory_summary_plots_v1(os.path.join(output_folder, "game_step_" + str(step)), os.path.join(output_folder, "game_step_" + str(step)), num_cropraiding_steps = 12)
        make_trajectory_summary_plots_v2(os.path.join(output_folder, "game_step_" + str(step)), os.path.join(output_folder, "game_step_" + str(step)), num_cropraiding_steps = 12)
        make_trajectory_summary_plots_v3(os.path.join(output_folder, "game_step_" + str(step)), os.path.join(output_folder, "game_step_" + str(step)))

    plt.figure(figsize=(6, 4.8))
    plt.plot(range(1, NUM_GAME_STEPS+1), damage_values, marker='o', linestyle='-', color='b', label='Damage Value')
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel('Damage Value (kg)', fontsize=12)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, "damage_with_game_step.png"), bbox_inches="tight", dpi=300)

    data = {'step': range(1, NUM_GAME_STEPS+1), 'damage_value': damage_values}
    df = pd.DataFrame(data)
    csv_file_path = "step_damage_data.csv"
    df.to_csv(os.path.join(output_folder, csv_file_path), index=False)

    return  














if __name__ == "__main__":

    model_params = {
            "year": 2010,
            "month": "Mar",
            "num_bull_elephants": 1,
            "area_size": 1100,
            "spatial_resolution": 30,
            "max_food_val_cropland": 100,
            "max_food_val_forest": 25,
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
            "slope_tolerance": 32.5,
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
            "elephant_starting_latitude": 1047232,
            "elephant_starting_longitude": 8568225,
            "elephant_aggression_value": 0.8,
            "elephant_crop_habituation": False
        }
    
    NUM_STRATEGIC_TRAJECTORIES = 42

    experiment_name = "mitigation-measures-within-plantations-FPL-UE_v3_1"

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

    targets_to_cover = [0]

    target_folder = f"protected_targets_{'_'.join(map(str, targets_to_cover))}"

    simulation_repeats = f'num_strategic_traj_{NUM_STRATEGIC_TRAJECTORIES}_num_iterations_{model_params["iterations"]}'

    output_folder = os.path.join(
        "/home/anjali/mnt/abm-elephant-project/aryabhata-runs/convergence-of-attacker-trajectories/",
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
        target_folder
    )

    coverage_matrix = create_defender_coverage_matrix(targets_to_cover)
    
    plot_and_save_defender_coverage(coverage_matrix, "game_theory_codes/OUR-MODEL/coverage_matrix_init/")

    run_single_play(
        model_params=model_params,
        experiment_name=experiment_name,
        output_folder=output_folder,
        NUM_STRATEGIC_TRAJECTORIES=NUM_STRATEGIC_TRAJECTORIES,
        NUM_GAME_STEPS = 10
    )
