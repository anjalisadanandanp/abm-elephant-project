import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from rasterio import open as rasterio_open
import rasterio
import os

def calculate_proximity(tif_path, target_value):
    """
    Calculate the proximity (distance) to cells with the target value in a TIFF file.
    
    Parameters:
    -----------
    tif_path : str
        Path to the TIFF file
    target_value : int or float
        The cell value to calculate proximity to
        
    Returns:
    --------
    tuple
        (proximity_matrix, original_data, transform, crs)
    """
    # Open the TIFF file
    with rasterio_open(tif_path) as src:
        # Read the data
        data = src.read(1)  # Read the first band
        transform = src.transform
        crs = src.crs
        
        # Create a binary mask where target cells are 1, others are 0
        mask = (data == target_value).astype(np.uint8)
        
        # Calculate the Euclidean distance to the nearest target cell
        # Note: distance_transform_edt calculates distance from 0s to 1s,
        # so we invert the mask to get distance from target cells
        proximity = distance_transform_edt(1 - mask)
        
        # Convert distance from pixels to coordinate units if needed
        # Assuming square pixels where resolution is transform.a
        proximity = proximity * abs(transform.a)
        
    return proximity, data, transform, crs

def plot_proximity(proximity, original_data, output_path=None, target_value=None):
    """
    Plot the proximity matrix with the original data outline.
    
    Parameters:
    -----------
    proximity : numpy.ndarray
        The proximity matrix
    original_data : numpy.ndarray
        The original data from the TIFF file
    output_path : str, optional
        Path to save the plot
    target_value : int or float, optional
        The target value that was used for proximity calculation
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot proximity as a heatmap
    im = ax.imshow(proximity, cmap='Greys_r', vmin=0, vmax=10000)
    plt.colorbar(im, ax=ax, label='Distance')
    
    # Overlay the outline of the target areas
    if target_value is not None:
        mask = (original_data == target_value)
        ax.contour(mask, colors='red', linewidths=0.5, levels=[0.5])
    
    ax.set_title(f'Proximity to Cells with Value {target_value}')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Save the plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
def main():
    # Parameters to adjust
    tif_path = "deterrent-measures/outputs/road_raster.tif"  # Replace with your TIFF file path
    target_value = 1  # Replace with the cell value you're interested in
    output_path = "deterrent-measures/outputs/road_proximity_map.png"  # Output plot file path
    
    # Calculate proximity
    proximity, original_data, transform, crs = calculate_proximity(tif_path, target_value)
    
    # Plot results
    plot_proximity(proximity, original_data, output_path, target_value)
    
    # Optionally save the proximity matrix as a new TIFF file
    output_tif = "deterrent-measures/outputs/road_proximity_matrix.tif"
    with rasterio.open(
        output_tif, 
        'w',
        driver='GTiff',
        height=proximity.shape[0],
        width=proximity.shape[1],
        count=1,
        dtype=proximity.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(proximity, 1)
    
    print(f"Proximity matrix saved to {output_tif}")
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()