import rasterio
from rasterio import features
import numpy as np
from shapely.geometry import LineString, Polygon
import geojson
import os
import matplotlib.pyplot as plt
from osgeo import gdal
from matplotlib.colors import ListedColormap
from rasterio import features
from collections import deque


def create_boundary_raster_with_ids(geojson_path, reference_tif_path, output_tif_path):

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

def discretize_raster_by_cells_from_raster(input_raster_path, output_raster_path, max_cells_per_group, target_values=[7, 17]):

    with rasterio.open(input_raster_path) as src:
        boundary_mask = src.read(1)
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs

    output_raster = np.zeros((height, width), dtype=np.uint32)
    current_group_id = 1
    
    for r in range(height):
        for c in range(width):
            for target_value in target_values:
                if boundary_mask[r, c] == target_value and output_raster[r, c] == 0:
                    q = deque([(r, c)])
                    cells_in_group = 0

                    while q and cells_in_group < max_cells_per_group:
                        row, col = q.popleft()

                        if output_raster[row, col] == 0 and boundary_mask[row, col] == target_value:
                            output_raster[row, col] = current_group_id
                            cells_in_group += 1

                            neighbors = [
                                (row - 1, col), (row + 1, col),
                                (row, col - 1), (row, col + 1)
                            ]
                            
                            for next_row, next_col in neighbors:
                                if 0 <= next_row < height and 0 <= next_col < width and \
                                output_raster[next_row, next_col] == 0 and \
                                boundary_mask[next_row, next_col] == target_value:
                                    q.append((next_row, next_col))
                    
                    if cells_in_group > 0:
                        current_group_id += 1

    with rasterio.open(output_raster_path, 'w',
                       driver='GTiff',
                       height=height,
                       width=width,
                       count=1,
                       dtype=output_raster.dtype,
                       crs=crs,
                       transform=transform,
                       nodata=0) as dst:
        
        dst.write(output_raster, 1)

def create_combined_raster(output_dir, max_cells_per_group):
    
    os.makedirs(output_dir, exist_ok=True)
    
    discretised_raster_path = os.path.join(output_dir, 'boundary_raster_files/boundary_raster.tif')

    save_dir = os.path.join(output_dir, 'boundary_raster_discretised_' + str(int(max_cells_per_group*30)) + 'm')
    
    os.makedirs(save_dir, exist_ok=True)

    discretize_raster_by_cells_from_raster(discretised_raster_path, os.path.join(save_dir, 'boundary_raster_discretised.tif'), max_cells_per_group=max_cells_per_group)
    
    return 

def plot_discretised_raster(output_dir):

    with rasterio.open(os.path.join(output_dir, 'boundary_raster_discretised.tif')) as src:
        raster_data = gdal.Open(os.path.join(output_dir, 'boundary_raster_discretised.tif')).ReadAsArray()
        transform = src.transform
        crs = src.crs

    unique_values = np.unique(raster_data[raster_data != src.nodata]).tolist()

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1, 1, 1)

    num_unique_values = len(unique_values) + 1
    random_colors = np.random.rand(num_unique_values, 3)  
    random_colors[0] = [1.0, 1.0, 1.0]
    random_colormap = ListedColormap(random_colors)
    
    im = ax.imshow(raster_data, cmap=random_colormap)

    for value in unique_values:

        y_coords, x_coords = np.where(raster_data == value)

        if len(x_coords) > 0 and len(y_coords) > 0:
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)

            ax.text(center_x, center_y, str(value),
                    ha='center', va='center', fontsize=5, color='black')
            
    ax.set_xticks([])
    ax.set_yticks([])

    output_path = os.path.join(output_dir, 'discretised_raster_plot.png')
    plt.savefig(output_path, dpi=750, bbox_inches='tight')

if __name__ == "__main__":

    for max_cells_per_group in [35, 40, 45, 50]:

        create_combined_raster(
            output_dir='guarding-policies/static-guarding-model-v1/create-strategy-matrix-v2', max_cells_per_group=max_cells_per_group
        )

        plot_discretised_raster(output_dir='guarding-policies/static-guarding-model-v1/create-strategy-matrix-v2/boundary_raster_discretised_' + str(int(max_cells_per_group*30)) + 'm')