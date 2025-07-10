import rasterio
from rasterio import features
import numpy as np
from shapely.geometry import Point, LineString, Polygon
import geojson
import os

def create_boundary_raster_with_ids(geojson_path, reference_tif_path, output_tif_path):
    """
    Create a .tif file from GeoJSON boundaries with unique boundary IDs
    
    Parameters:
    - geojson_path: path to input GeoJSON file
    - reference_tif_path: path to reference .tif for spatial properties
    - output_tif_path: path for output .tif file
    """

    
    with open(geojson_path) as f:
        gj = geojson.load(f)

    with rasterio.open(reference_tif_path) as ref:
        transform = ref.transform
        width = ref.width
        height = ref.height
        crs = ref.crs
        bounds = ref.bounds
    
    boundary_features = []
    boundary_id = 1
    boundary_info = []

    for poly_idx, row in enumerate(gj['features']):
        polygon = Polygon(row.geometry.coordinates[0])
        
        exterior_line = LineString(polygon.exterior.coords)
        boundary_features.append((exterior_line, boundary_id))
        boundary_info.append({
            'boundary_id': boundary_id,
            'polygon_id': poly_idx,
            'boundary_type': 'exterior',
            'length': exterior_line.length
        })
        boundary_id += 1
    
    raster = np.zeros((height, width), dtype='uint32')
    
    if boundary_features:
        boundary_raster = features.rasterize(
            boundary_features,
            out_shape=raster.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=None
        )
    
    with rasterio.open(output_tif_path, 'w',
                       driver='GTiff',
                       height=height,
                       width=width,
                       count=1,
                       dtype=raster.dtype,
                       crs=crs,
                       transform=transform,
                       nodata=0) as dst:
        dst.write(boundary_raster, 1)
    
    return

def discretize_boundaries_with_ids(geojson_path, reference_tif_path, discretised_raster_path, segment_length):

    with open(geojson_path) as f:
        gj = geojson.load(f)

    def discretize_linestring(line, segment_length):
        coords = list(line.coords)

        if len(coords) < 2:
            return [LineString(coords[0][0], coords[0][1], 0)] if coords else []
        
        linestrings = []

        i = 0

        while i <= len(coords):
            
            start = i

            if i+segment_length < len(coords):
                end = i + segment_length + 1
            else:
                end = len(coords) 

            linestrings.append(LineString(coords[start:end]))

            i = i + segment_length
            
        return linestrings

    boundary_features = []
    segment_id = 1

    with rasterio.open(reference_tif_path) as ref:
        transform = ref.transform
        width = ref.width
        height = ref.height
        crs = ref.crs
        bounds = ref.bounds

    for poly_idx, row in enumerate(gj['features']):
        polygon = Polygon(row.geometry.coordinates[0])
        
        linestrings = discretize_linestring(polygon.exterior, segment_length)

        for i in range(len(linestrings) - 1):

            segment_line = linestrings[i]

            boundary_features.append((segment_line, segment_id))
            segment_id += 1

    raster = np.zeros((height, width), dtype='uint32')

    if boundary_features:
        boundary_raster = features.rasterize(
            boundary_features,
            out_shape=raster.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=None
        )
    
    with rasterio.open(discretised_raster_path, 'w',
                       driver='GTiff',
                       height=height,
                       width=width,
                       count=1,
                       dtype=raster.dtype,
                       crs=crs,
                       transform=transform,
                       nodata=0) as dst:
        
        dst.write(boundary_raster, 1)

    return 

def create_combined_raster(geojson_path, reference_tif_path, output_dir, segment_length):
    
    os.makedirs(output_dir, exist_ok=True)
    
    boundary_raster_path = os.path.join(output_dir, 'boundary_raster.tif')
    discretised_raster_path = os.path.join(output_dir, 'boundary_raster_discretised.tif')
    
    create_boundary_raster_with_ids(
        geojson_path, 
        reference_tif_path, 
        boundary_raster_path
    )
    
    discretize_boundaries_with_ids(
        geojson_path, 
        reference_tif_path,
        discretised_raster_path,
        segment_length
    )
    
    return 

if __name__ == "__main__":

    create_combined_raster(
        geojson_path='mesageo_elephant_project/elephant_project/geojson_files/landuse_10.geojson',
        reference_tif_path='mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif',
        output_dir='create-strategy-matrix',
        segment_length=15
    )