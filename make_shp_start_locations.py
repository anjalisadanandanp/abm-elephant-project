import geopandas
import pandas as pd
from shapely.geometry import Point

def create_shapefile_from_points(points_data, crs, output_filename):

    try:
        df = pd.DataFrame(points_data)

        df['geometry'] = df.apply(
            lambda row: Point(row['longitude'], row['latitude']), axis=1
        )

        gdf = geopandas.GeoDataFrame(df, geometry='geometry', crs=crs)

        gdf.to_file(output_filename, driver='ESRI Shapefile')

        print(f"Successfully created shapefile: {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have geopandas installed and the input data is correct.")

if __name__ == "__main__":

    sample_points = [
        {'name': 'Point A', 'latitude': 1051169, 'longitude': 8572676},
        {'name': 'Point B', 'latitude': 1044999, 'longitude': 8572569},
        {'name': 'Point C', 'latitude': 1046859, 'longitude': 8567834},
    ]

    point_crs = 'EPSG:3857'  

    output_file = 'shp_start_locations.shp'

    create_shapefile_from_points(sample_points, point_crs, output_file)
