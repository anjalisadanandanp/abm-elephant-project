from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

def find_first_reachable_targets(matrix, start_row, start_col, target_value, max_distance):
    rows, cols = len(matrix), len(matrix[0])
    
    visited = set([(start_row, start_col)])
    
    first_targets_found = set()
    
    directions = [
        (-1, 0),  # up
        (-1, 1),  # up-right
        (0, 1),   # right
        (1, 1),   # down-right
        (1, 0),   # down
        (1, -1),  # down-left
        (0, -1),  # left
        (-1, -1)  # up-left
    ]
    
    queue = deque([(start_row, start_col, False)])
    
    while queue:
        row, col, path_has_target = queue.popleft()

        if matrix[row][col] == target_value and (row, col) != (start_row, start_col):
            if not path_has_target:
                first_targets_found.add((row, col))
            continue
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if (new_row, new_col) not in visited:
                    distance = abs(new_row - start_row) + abs(new_col - start_col)
                    if distance <= max_distance:
                        new_path_has_target = path_has_target or (
                            matrix[row][col] == target_value and (row, col) != (start_row, start_col)
                        )
                        
                        queue.append((new_row, new_col, new_path_has_target))
                        visited.add((new_row, new_col))
    
    return list(first_targets_found)

def plot_targets_on_matrix(matrix, start_row, start_col, targets, target_value, max_distance):
    """
    Plot the matrix with the start cell and reachable targets highlighted
    """
    plt.figure(figsize=(8, 8))
    
    unique_values = set()
    for row in matrix:
        for val in row:
            unique_values.add(val)
    
    plt.imshow(matrix, cmap='gray', interpolation='nearest')
    
    plt.plot(start_col, start_row, 'rx', markersize=10, markeredgewidth=2.5)
    
    for row, col in targets:
        plt.plot(col, row, 'gs', markersize=1)
        
    plt.title(f"First Reachable Cells with Value {target_value} from ({start_row}, {start_col}) within Distance {max_distance}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    
    plt.plot([], [], 'rx', markersize=10, label='Start Cell')
    plt.plot([], [], 'gs', markersize=10, label='Reachable Targets')
    plt.legend(loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("game_theory_codes/FPL-UE/create-strategy-set/outputs/first_reachable_targets.png", dpi=300)

    return 

def save_targets_as_tiff(output_path, original_tiff_path, reachable_targets, matrix_shape):

    src_ds = gdal.Open(original_tiff_path)
    if src_ds is None:
        raise ValueError(f"Could not open the original GeoTIFF file: {original_tiff_path}")
    
    geotransform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    
    rows, cols = matrix_shape
    binary_matrix = np.zeros((rows, cols), dtype=np.uint8)
    
    for row, col in reachable_targets:
        if 0 <= row < rows and 0 <= col < cols:  
            binary_matrix[row, col] = 1

    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte)
    
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(projection)
    
    dst_ds.GetRasterBand(1).WriteArray(binary_matrix)
    
    src_ds = None
    dst_ds = None
    
    return output_path

if __name__ == "__main__":

    dataset = gdal.Open("game_theory_codes/FPL-UE/create-strategy-set/outputs/interpolated_LULC_matrix.tif")

    matrix = dataset.ReadAsArray()
    if isinstance(matrix, np.ndarray):
        matrix = matrix.tolist()
    
    dataset = None

    start_row, start_col = 375, 600
    target_value = 10
    max_distance = 600
    
    reachable_targets = find_first_reachable_targets(matrix, start_row, start_col, target_value, max_distance)

    plot_targets_on_matrix(matrix, start_row, start_col, reachable_targets, target_value, max_distance)

    save_targets_as_tiff("game_theory_codes/FPL-UE/create-strategy-set/outputs/first_reachable_targets.tif", "game_theory_codes/FPL-UE/create-strategy-set/outputs/interpolated_LULC_matrix.tif", reachable_targets, (len(matrix), len(matrix[0])))