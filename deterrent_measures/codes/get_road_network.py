import osmnx as ox
import geopandas as gpd
from pyproj import Transformer
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
import os
import rasterio
from rasterio import features

def get_road_network_from_extent(xmin, ymin, xmax, ymax, epsg=3857, road_types=None):
    """
    Extract road network within a bounding box defined in EPSG:3857 (Web Mercator).
    
    Parameters:
    -----------
    xmin, ymin, xmax, ymax : float
        Bounding box coordinates in EPSG:3857
    epsg : int
        The EPSG code of the input coordinates (default: 3857 Web Mercator)
    road_types : list
        List of OSM road types to include. If None, includes all roads.
        Example: ['motorway', 'primary', 'secondary', 'tertiary', 'residential']
    
    Returns:
    --------
    G : networkx.MultiDiGraph
        Road network graph
    gdf_nodes : GeoDataFrame
        GeoDataFrame of nodes
    gdf_edges : GeoDataFrame
        GeoDataFrame of edges
    """

    transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)
    
    bbox = (lat_min, lat_max, lon_min, lon_max)
    
    print(f"Querying OSM with bbox (lat_min, lat_max, lon_min, lon_max): {bbox}")
    
    if road_types is None:
        G = ox.graph_from_bbox(lat_min, lat_max, lon_min, lon_max, network_type='drive', retain_all=True, truncate_by_edge=False)
    else:
        custom_filter = f'["highway"~"{"|".join(road_types)}"]'
        G = ox.graph_from_bbox(lat_min, lat_max, lon_min, lon_max, custom_filter=custom_filter)
    
    G_proj = ox.project_graph(G, to_crs=epsg)

    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G_proj)
    
    return G_proj, gdf_nodes, gdf_edges

def visualize_network(gdf_nodes, gdf_edges, figsize=(12, 10), edge_width=1.5, node_size=15, save_path=None):
    """
    Visualize the road network with basemap.
    
    Parameters:
    -----------
    gdf_nodes : GeoDataFrame
        GeoDataFrame of nodes
    gdf_edges : GeoDataFrame
        GeoDataFrame of edges
    figsize : tuple
        Figure size
    edge_width : float
        Width of the road edges in the plot
    node_size : float
        Size of the intersection nodes in the plot
    save_path : str
        Path to save the figure (if None, display only)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    gdf_edges.plot(ax=ax, linewidth=edge_width, color='blue', alpha=0.7)
    
    gdf_nodes.plot(ax=ax, markersize=node_size, color='red', alpha=0.7)
    
    try:
        ctx.add_basemap(ax, crs=gdf_edges.crs.to_string())
    except Exception as e:
        print(f"Unable to add basemap: {e}")
    
    ax.set_axis_off()
    
    plt.title('OpenStreetMap Road Network', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig, ax

def extract_road_network_stats(G, gdf_edges):
    """
    Extract basic statistics about the road network.
    
    Parameters:
    -----------
    G : networkx.MultiDiGraph
        Road network graph
    gdf_edges : GeoDataFrame
        GeoDataFrame of edges
    
    Returns:
    --------
    stats : dict
        Dictionary with network statistics
    """
    stats = {}
    stats['node_count'] = len(G.nodes)
    stats['edge_count'] = len(G.edges)
    
    if 'length' in gdf_edges.columns:
        stats['total_length_km'] = gdf_edges['length'].sum() / 1000
    
    if 'length' in gdf_edges.columns:

        bbox = gdf_edges.total_bounds  
        width = (bbox[2] - bbox[0]) / 1000  # Convert to km
        height = (bbox[3] - bbox[1]) / 1000  # Convert to km
        area = width * height
        stats['area_km2'] = area
        stats['road_density'] = stats['total_length_km'] / area if area > 0 else 0
    
    return stats

def rasterize_roads(road_gdf, reference_raster_path, output_raster_path):
    """
    Create a raster with 1s where roads exist and 0s elsewhere
    
    Parameters:
    -----------
    road_gdf : GeoDataFrame
        GeoDataFrame containing road geometries
    reference_raster_path : str
        Path to the reference raster file
    output_raster_path : str
        Path where the output raster will be saved
    """

    with rasterio.open(reference_raster_path) as src:
        rasterized = np.zeros((src.height, src.width), dtype=np.uint8)
        
        shapes = [(geom, 1) for geom in road_gdf.geometry]
        burned = features.rasterize(shapes=shapes,
                                   out=rasterized,
                                   transform=src.transform,
                                   fill=0,
                                   all_touched=True)  
        
        meta = src.meta.copy()
        meta.update({
            'dtype': rasterio.uint8,
            'count': 1,
            'nodata': 0
        })
        
        with rasterio.open(output_raster_path, 'w', **meta) as dst:
            dst.write(burned, 1)
        
        print(f"Created road raster at: {output_raster_path}")
        
        return burned

def plot_results(reference_raster_path, road_raster, output_path="deterrent-measures/outputs"):
    """
    Plot the reference raster and the road raster for visual comparison
    
    Parameters:
    -----------
    reference_raster_path : str
        Path to the reference raster file
    road_raster : numpy.ndarray
        The rasterized road network
    output_path : str
        Path where the plot will be saved
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with rasterio.open(reference_raster_path) as src:
        ref_data = src.read(1)  
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    im1 = ax1.imshow(ref_data, cmap='coolwarm')
    ax1.set_title("Reference Raster")
    fig.colorbar(im1, ax=ax1, shrink=0.5)
    
    im2 = ax2.imshow(road_raster, cmap='binary', vmin=0, vmax=1)
    ax2.set_title("Road Raster (0=No Road, 1=Road)")
    fig.colorbar(im2, ax=ax2, shrink=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "road_raster.png"), dpi=300)
    plt.close()
    
    print(f"Comparison plot saved to: {os.path.join(output_path, 'road_raster.png')}")

def main():

    reference_raster_path = "deterrent-measures/data/DEM.tif"
    
    output_raster_path = "deterrent-measures/outputs/road_raster.tif"
    
    print("\nFetching road network from OpenStreetMap...")
    G, nodes, road_gdf = get_road_network_from_extent(xmin, ymin, xmax, ymax, epsg=3857, road_types=road_types)
    
    print(f"Retrieved {len(road_gdf)} road segments")
    
    print("\nRasterizing road network...")
    road_raster = rasterize_roads(road_gdf, reference_raster_path, output_raster_path)
    
    print("\nCreating comparison plot...")
    plot_results(reference_raster_path, road_raster)
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":

    xmin, ymin = 8554718.2081622108817101, 1026034.7470012035919353  # Lower left corner
    xmax, ymax = 8587897.4710439480841160, 1059183.0012260428629816  # Upper right corner
    
    road_types = None
    
    G, nodes, edges = get_road_network_from_extent(
        xmin, ymin, xmax, ymax, epsg=3857, road_types=road_types
    )
    
    stats = extract_road_network_stats(G, edges)
    print("\nRoad Network Statistics:")
    for key, value in stats.items():
        if key == 'road_types':
            print(f"Road types distribution:")
            for road_type, count in value.items():
                print(f"  - {road_type}: {count} segments")
        else:
            print(f"- {key}: {value}")
    

    fig, ax = visualize_network(nodes, edges, save_path="deterrent-measures/outputs/road_network.png")

    main()