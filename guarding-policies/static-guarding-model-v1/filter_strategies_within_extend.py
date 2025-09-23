import pandas as pd
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

def clip_raster_by_latlon_extent(input_file, latlon_extent):

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

    print(latlon_extent)

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

    fig, ax = plt.subplots(figsize=(8, 8))

    cax = ax.imshow(raster_data, cmap=cmap, vmin=0.1, 
                    extent=(geo_transform[0], geo_transform[0] + x_size * x_res, 
                            geo_transform[3] + y_size * y_res, geo_transform[3]))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.savefig("clipped_raster.png")

    return np.unique(raster_data)

input_file = "guarding-policies/static-guarding-model-v1/create-strategy-matrix-v2/boundary_raster_discretised_600m/boundary_raster_discretised.tif"

latlon_extent = (8563700, 1043400, 8574155, 1056000) 

targets = clip_raster_by_latlon_extent(input_file, latlon_extent)

reward_df = pd.read_csv("guarding-policies/static-guarding-model-v1/find_boundary_patch_reward_penalty_values-v2_4/boundary_raster_discretised_600m/boundary_patch_reward_penalty_matrix.csv")
sorted_df = reward_df.sort_values(by='reward', ascending=False)

boundary_patches = gdal.Open("guarding-policies/static-guarding-model-v1/create-strategy-matrix-v2/boundary_raster_discretised_600m/boundary_raster_discretised.tif")

filtered_df = sorted_df[sorted_df['boundary_patch_id'].isin(targets)]

