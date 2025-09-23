import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj, transform  
from mpl_toolkits.basemap import Basemap    
import rasterio
from rasterio.features import shapes
import fiona
import geojson
import matplotlib.cm as cm
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def make_plots(output_folder, output_folder_raster):

    def raster_to_geojson(input_raster_path, output_geojson_path):

        with rasterio.open(input_raster_path) as src:
            image = src.read(1)

            if image.dtype not in ['int16', 'int32', 'uint8', 'uint16', 'float32']:
                    image = image.astype('int32')

            mask = image > 0
            mask = mask.astype('uint8')

            results = [
                {'properties': {'raster_val': int(v)}, 'geometry': s}
                for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.transform))
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

    ds = gdal.Open(os.path.join(output_folder_raster, "boundary_raster_discretised.tif"))

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

    raster_to_geojson(os.path.join(output_folder_raster, "boundary_raster_discretised.tif"), os.path.join(output_folder, 'boundary_patches.geojson'))

    with open('mesageo_elephant_project/elephant_project/geojson_files/landuse_10.geojson', 'r') as f:
        geojson_object = geojson.load(f)

    reward_df = pd.read_csv(os.path.join(output_folder, "boundary_patch_reward_penalty_matrix.csv"))

    for feature in geojson_object['features']:
        coords = feature['geometry']['coordinates'][0]
        coords = [transform(inProj, outProj, lon, lat) for lon, lat in coords]
        coords = [(lon, lat) for lon, lat in coords]
        lon, lat = zip(*coords)

        plt.fill(lon, lat, color='yellow', alpha=0.20, zorder=1)

        map.plot(lon, lat, marker=None, color='black', linewidth=1, zorder=2)

    with open(os.path.join(output_folder, 'boundary_patches.geojson'), 'r') as f:
        geojson_object = geojson.load(f)

    raster_values = [feature['properties']['raster_val'] for feature in geojson_object['features']]

    reward_values = {}

    for raster_value in raster_values:
        
        try:
            # print(reward_df[reward_df["boundary_patch_id"] == raster_value])
            reward_values[raster_value] = reward_df[reward_df["boundary_patch_id"] == raster_value]["reward"].values[0]
        except:
            reward_values[raster_value] = 0

    min_val = min(reward_values.values())
    max_val = max(reward_values.values())

    cmap = cm.get_cmap('rainbow') 
    norm = plt.Normalize(vmin=min_val, vmax=max_val)

    for feature in geojson_object['features']:
        coords = feature['geometry']['coordinates'][0]
        coords = [transform(inProj, outProj, lon, lat) for lon, lat in coords]
        coords = [(lon, lat) for lon, lat in coords]
        lon, lat = zip(*coords)

        raster_val = feature['properties']['raster_val']

        if reward_values[raster_val] > 0:

            color = cmap(norm(reward_values[raster_val]))

            map.plot(lon, lat, marker=None, color=color, linewidth=1, zorder=3)

            centroid_lon = sum(lon) / len(lon)
            centroid_lat = sum(lat) / len(lat)

            raster_val = feature['properties']['raster_val']
                
            plt.text(centroid_lon, centroid_lat+0.005, str(raster_val), 
                    fontsize=1.75, ha='center', va='center', 
                    bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', 
                            edgecolor='blue', 
                            linewidth=0.5,
                            alpha=1.0),
                    zorder=4)
        
    map.drawmeridians([LON_MIN,(LON_MIN+LON_MAX)/2-(LON_MAX-LON_MIN)*1/4,(LON_MIN+LON_MAX)/2,(LON_MIN+LON_MAX)/2+(LON_MAX-LON_MIN)*1/4,LON_MAX], labels=[0,1,0,1],)
    map.drawparallels([LAT_MIN,(LAT_MIN+LAT_MAX)/2-(LAT_MAX-LAT_MIN)*1/4,(LAT_MIN+LAT_MAX)/2,(LAT_MIN+LAT_MAX)/2+(LAT_MAX-LAT_MIN)*1/4,LAT_MAX], labels=[1,0,1,0])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, label='target rewards', shrink=0.5)
    cbar.ax.tick_params(labelsize=8)  

    plt.savefig(os.path.join(output_folder, "boundary_patches_with_rewards.png"), dpi=750, bbox_inches='tight')

    plt.close()

    return


from tqdm import tqdm

for max_cells_per_group in tqdm([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]):

    make_plots(output_folder="guarding-policies/dynamic-guarding-model-v1/find_boundary_patch_reward_penalty_values-v2_4/boundary_raster_discretised_" + str(int(max_cells_per_group*30)) + "m", output_folder_raster="guarding-policies/dynamic-guarding-model-v1/create-strategy-matrix-v2/boundary_raster_discretised_" + str(int(max_cells_per_group*30)) + "m", )