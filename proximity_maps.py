import numpy as np
from scipy.ndimage import distance_transform_edt

def calculate_proximity_matrix(landscape_matrix, target_class):
    """
    Calculate proximity matrix for a given land use class
    
    Args:
        landscape_matrix: 2D numpy array with land use classes
        target_class: integer representing the land use class to calculate proximity for
    
    Returns:
        2D numpy array with distances to nearest target class cell
    """
    # Create binary matrix where 1 represents target class
    binary_matrix = (landscape_matrix == target_class).astype(int)
    
    # Calculate euclidean distance transform
    # This gives distance from each cell to nearest target class
    proximity_matrix = distance_transform_edt(1 - binary_matrix)
    
    return proximity_matrix

#open the .tif file using gdal
#path: /home2/anjali/GitHub/abm-elephant-project/mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif
from osgeo import gdal
import numpy as np

# Open the file
ds = gdal.Open('/home2/anjali/GitHub/abm-elephant-project/mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif')
landscape = np.array(ds.GetRasterBand(1).ReadAsArray())

# print(np.unique(landscape))


# Example usage
# landscape = np.random.randint(1, 5, size=(100, 100))  # Random 100x100 landscape
forest_proximity = calculate_proximity_matrix(landscape, target_class=15)  # Proximity to forest

#plot the original landscape and the proximity map
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
img1 = plt.imshow(landscape, cmap='tab20b')
cbar = plt.colorbar(img1, fraction=0.046, pad=0.04)
plt.title('Landscape')
plt.subplot(1, 2, 2)
plt.imshow(forest_proximity, cmap='viridis')
plt.title('Proximity to Forest')
plt.show()
