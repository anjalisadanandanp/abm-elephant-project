import numpy as np
import rasterio
import matplotlib.pyplot as plt
import random
import os
import matplotlib.patches as mpatches
from scipy.ndimage import distance_transform_edt

class DeterrentPolicyPlanner:

    def __init__(self, landuse_path, dem_path, slope_path, road_path, forest_agricultural_boundary_buffer):

        self.landuse, self.landuse_meta = self._load_raster(landuse_path)
        self.dem, _ = self._load_raster(dem_path)
        self.slope, _ = self._load_raster(slope_path)
        self.roads, _ = self._load_raster(road_path)
        self.forest_agricultural_boundary_buffer = forest_agricultural_boundary_buffer
        
        self.shape = self.landuse.shape
        self.transform = self.landuse_meta['transform']
        self.crs = self.landuse_meta['crs']
        
        self.current_policy = None
    
    def _load_raster(self, path):

        with rasterio.open(path) as src:
            data = src.read(1)
            meta = src.meta.copy()
        return data, meta

    def _detect_forest_plantation_border(self, buffer_size):

        forest_mask = (self.landuse == 15)  | (self.landuse == 5) | (self.landuse == 4) 
        plantation_mask = self.landuse == 10
        
        
        distance_from_forest = distance_transform_edt(~forest_mask)
        
        border = plantation_mask & (distance_from_forest <= self.forest_agricultural_boundary_buffer)
        
        return border

    def _calculate_suitability_plantations(self):

        suitability = np.zeros(self.landuse.shape, dtype=np.float32)

        plantation_mask = self.landuse == 10
        suitability += plantation_mask

        if suitability.max() > 0:
            suitability = suitability / suitability.max()
        
        return suitability

    def _calculate_suitability_forest_plantation_border(self):

        suitability = np.zeros(self.landuse.shape, dtype=np.float32)
        
        border_mask = self._detect_forest_plantation_border(buffer_size=self.forest_agricultural_boundary_buffer)
        suitability += border_mask 
        
        if suitability.max() > 0:
            suitability = suitability / suitability.max()
        
        return suitability

    def _calculate_suitability_v1(self, w_border, w_roads, w_plantation, w_dem, w_slope):

        suitability = np.zeros(self.landuse.shape, dtype=np.float32)
        
        border_mask = self._detect_forest_plantation_border(buffer_size=self.forest_agricultural_boundary_buffer)
        suitability += border_mask * w_border
        
        road_mask = self.roads > 0
        if np.any(road_mask):
            road_dist = distance_transform_edt(~road_mask) * self.landuse_meta['transform'][0]
            road_proximity = np.clip(1 - (road_dist / 500), 0, 1)
            suitability += road_proximity * w_roads

        plantation_mask = self.landuse == 10
        suitability += plantation_mask * w_plantation
        
        if self.dem.min() != self.dem.max():
            elevation_norm = (self.dem - self.dem.min()) / (self.dem.max() - self.dem.min())
            elevation_suitability = 1 - elevation_norm
            suitability += elevation_suitability * w_dem

        slope_suitability = (self.slope < 30).astype(np.float32)
        suitability += slope_suitability *w_slope
        
        if suitability.max() > 0:
            suitability = suitability / suitability.max()
        
        return suitability
    
    def generate_random_policy(self, coverage_percentage, threshold=0.5):

        self.suitability = self._calculate_suitability_plantations()

        suitable_areas = self.suitability >= threshold
        
        total_suitable_cells = np.sum(suitable_areas)
        num_policy_cells = int(total_suitable_cells * coverage_percentage / 100)
        
        suitable_indices = np.where(suitable_areas)
        suitable_indices = list(zip(suitable_indices[0], suitable_indices[1]))
        
        suitability_values = [self.suitability[i, j] for i, j in suitable_indices]
        
        sorted_indices = [x for _, x in sorted(zip(suitability_values, suitable_indices), reverse=True)]

        selected_indices = random.choices(sorted_indices, k=num_policy_cells)

        policy = np.zeros(self.shape, dtype=np.uint8)
        for i, j in selected_indices:
            policy[i, j] = 1
        
        self.current_policy = policy

        return policy
    
    def generate_perimeter_policy(self, coverage_percentage, threshold=0.5):

        self.suitability = self._calculate_suitability_forest_plantation_border()

        suitable_areas = self.suitability >= threshold
        
        total_suitable_cells = np.sum(suitable_areas)
        num_policy_cells = int(total_suitable_cells * coverage_percentage / 100)
        
        suitable_indices = np.where(suitable_areas)
        suitable_indices = list(zip(suitable_indices[0], suitable_indices[1]))
        
        suitability_values = [self.suitability[i, j] for i, j in suitable_indices]
        
        sorted_indices = [x for _, x in sorted(zip(suitability_values, suitable_indices), reverse=True)]

        selected_indices = random.choices(sorted_indices, k=num_policy_cells)

        policy = np.zeros(self.shape, dtype=np.uint8)
        for i, j in selected_indices:
            policy[i, j] = 1
        
        self.current_policy = policy

        return policy

    def generate_clustered_policy(self, coverage_percentage, threshold, w_border, w_roads, w_plantation, w_dem, w_slope):

        self.suitability = self._calculate_suitability_v1(w_border, w_roads, w_plantation, w_dem, w_slope)

        suitable_areas = self.suitability >= threshold
        
        total_suitable_cells = np.sum(suitable_areas)
        num_policy_cells = int(total_suitable_cells * coverage_percentage / 100)
        
        suitable_indices = np.where(suitable_areas)
        suitable_indices = list(zip(suitable_indices[0], suitable_indices[1]))
        
        suitability_values = [self.suitability[i, j] for i, j in suitable_indices]
        
        sorted_indices = [x for _, x in sorted(zip(suitability_values, suitable_indices), reverse=True)]

        selected_indices = random.choices(sorted_indices, k=num_policy_cells)

        policy = np.zeros(self.shape, dtype=np.uint8)
        for i, j in selected_indices:
            policy[i, j] = 1
        
        self.current_policy = policy
        
        return policy

    def save_policy_raster(self, output_path):
        """Save the current policy as a raster file."""
        if self.current_policy is None:
            raise ValueError("No policy has been generated yet")
        
        meta = self.landuse_meta.copy()
        meta.update({
            'dtype': 'uint8',
            'count': 1,
            'nodata': 0
        })
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(self.current_policy, 1)
    
    def plot_suitability(self, ax=None, title="Suitability Map", show_roads=True, show_borders=True):
        """Visualize the suitability map with forest-plantation borders and roads."""
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        suit_plot = ax.imshow(self.suitability, cmap='coolwarm', alpha=0.5)
        
        if show_borders:
            forest_plantation_border = self._detect_forest_plantation_border(buffer_size=self.forest_agricultural_boundary_buffer)
            if np.any(forest_plantation_border):
                border_line = ax.contour(forest_plantation_border, colors='blue', linewidths=1.0, alpha=0.8)

        if show_roads and np.any(self.roads):
            roads_line = ax.contour(self.roads > 0, colors='black', linewidths=0.5, alpha=1)
        
        ax.set_title(title)
        plt.colorbar(suit_plot, ax=ax, label='Suitability', shrink=0.5)
        
        ax.set_axis_off()
        
        border_patch = mpatches.Patch(color='blue', alpha=0.5, label='Forest-Plantation Border')
        road_patch = mpatches.Patch(color='black', alpha=0.7, label='Roads')
        
        handles = []
        if show_borders:
            handles.append(border_patch)
        if show_roads and np.any(self.roads):
            handles.append(road_patch)
        
        if handles:
            ax.legend(handles=handles, loc='upper right', fontsize=8)
        
        return ax

    def plot_policy(self, ax=None, title="Current Policy", show_roads=True, show_borders=True):
        """Visualize the current policy implementation with forest-plantation borders and roads."""
        
        if self.current_policy is None:
            raise ValueError("No policy has been generated yet")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        policy_plot = ax.imshow(self.current_policy, cmap='Greys_r', alpha=0.5)
        
        if show_borders:
            forest_plantation_border = self._detect_forest_plantation_border(buffer_size=self.forest_agricultural_boundary_buffer)
            if np.any(forest_plantation_border):
                border_line = ax.contour(forest_plantation_border, colors='blue', linewidths=1.0, alpha=0.8)
        
        if show_roads and np.any(self.roads):
            roads_line = ax.contour(self.roads > 0, colors='black', linewidths=0.5, alpha=1)
        
        ax.set_title(title)
        plt.colorbar(policy_plot, ax=ax, label='Policy Intensity', shrink=0.5)
        
        ax.set_axis_off()
        
        border_patch = mpatches.Patch(color='blue', alpha=0.5, label='Forest-Plantation Border')
        road_patch = mpatches.Patch(color='black', alpha=0.7, label='Roads')
        
        handles = []
        if show_borders:
            handles.append(border_patch)
        if show_roads and np.any(self.roads):
            handles.append(road_patch)
        
        ax.legend(handles=handles, loc='upper right', fontsize=8)
        
        return ax

    def make_plots(self, output_dir, name):
        """Generate a single image with both suitability and policy plots."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        self.plot_suitability(ax=ax1, title="Suitability Map")
        
        self.plot_policy(ax=ax2, title="Current Deterrent Policy")
        
        plt.tight_layout()

        fig.savefig(os.path.join(output_dir, name + ".png"), dpi=300, bbox_inches='tight')

        plt.close(fig)

        return 

    def save_raster(self, output_path, reference_raster_path, nodata_value=-1, dtype=None):

        data = self.current_policy
        
        with rasterio.open(reference_raster_path) as src:
            transform = src.transform
            crs = src.crs
            
            if dtype is None:
                dtype = data.dtype
            
            if len(data.shape) == 2:
                height, width = data.shape
                count = 1
            elif len(data.shape) == 3:
                count, height, width = data.shape
            else:
                raise ValueError("Input data must be 2D (single band) or 3D (multi-band)")
            
            new_dataset = rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=count,
                dtype=dtype,
                crs=crs,
                transform=transform,
                nodata=nodata_value
            )
            
            if count == 1:
                new_dataset.write(data, 1)
            else:
                for i in range(count):
                    new_dataset.write(data[i], i+1)
            
            new_dataset.close()
        
    def make_policies(self, config, output_dir=None):
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        config_type = config.get('type', 'random')
        coverage = config.get('coverage')
        threshold = config.get('threshold')

        if config_type == 'random':
            policy = self.generate_random_policy(coverage, threshold)
            title = f"Random Policy: ({coverage}% coverage at {threshold} suitability threshold)"
            config_id = f"{config_type}__coverage_{coverage}__threshold_{threshold}"
        
        elif config_type == 'perimeter':
            buffer_dist = config.get('buffer_distance')
            policy = self.generate_perimeter_policy(coverage, threshold)
            title = f"Perimeter Policy ({coverage}% coverage, {buffer_dist}m buffer at {threshold} suitability threshold)"
            config_id = f"{config_type}__coverage_{coverage}__threshold_{threshold}"
        
        elif config_type == 'clustered':
            w_border = config.get('w_border')
            w_roads = config.get('w_roads')
            w_plantation = config.get('w_plantation')
            w_dem = config.get('w_dem')
            w_slope = config.get('w_slope')
            policy = self.generate_clustered_policy(coverage, threshold, w_border, w_roads, w_plantation, w_dem, w_slope)
            title = f"Clustered Policy ({coverage}% coverage)"
            config_id = f"{config_type}__coverage_{coverage}__threshold_{threshold}__w_border_{w_border}__w_roads_{w_roads}__w_plantation_{w_plantation}__w_dem_{w_dem}__w_slope_{w_slope}"
        
        else:
            raise ValueError(f"Unknown policy type!")
        
        self.make_plots(output_dir, name=config_id)
            
        return
      
if __name__ == "__main__":

    landuse_path = "deterrent-measures/data/LULC.tif"
    dem_path = "deterrent-measures/data/DEM.tif"
    slope_path = "deterrent-measures/data/slope_matrix.tif"
    road_path = "deterrent-measures/outputs/road_raster.tif"

    forest_agricultural_boundary_buffer = 10
    
    planner = DeterrentPolicyPlanner(landuse_path, dem_path, slope_path, road_path, forest_agricultural_boundary_buffer)
    
    output_dir = "deterrent-measures/outputs/deterrent_policy_results"
    
    configurations = [

        {'type': 'random', 'coverage': 10, 'threshold': 0.4},       
        {'type': 'random', 'coverage': 10, 'threshold': 0.6},
        
        {'type': 'perimeter', 'coverage': 10, 'threshold': 0.4, "buffer_distance":forest_agricultural_boundary_buffer},
        
        {'type': 'clustered', 'coverage': 100, 'threshold': 0.4, "w_border":0.5, "w_roads":0.25, "w_plantation":0.25, "w_dem":0, "w_slope":0},
        {'type': 'clustered', 'coverage': 100, 'threshold': 0.6, "w_border":0, "w_roads":1, "w_plantation":0, "w_dem":0, "w_slope":0},
        {'type': 'clustered', 'coverage': 100, 'threshold': 0.8, "w_border":0.25, "w_roads":0.25, "w_plantation":0.5, "w_dem":0, "w_slope":0},
    ]

    for configuration in configurations:
        planner.make_policies(configuration, output_dir)

        planner.save_raster(
                output_path=os.path.join(output_dir, "deterrent_policy.tif"),
                reference_raster_path=landuse_path, 
                dtype=rasterio.int8
            )