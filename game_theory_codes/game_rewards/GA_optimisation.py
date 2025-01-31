import numpy as np
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import yaml
import rasterio

class RangerOptimizer:
    def __init__(
        self,
        num_rangers: int,
        population_size: int,
        generations: int,
        data_folder: str
    ):
        self.num_rangers = num_rangers
        self.data_folder = data_folder

        self.population_size = population_size
        self.generations = generations

        self.history = {'best': [],
                        'worst': [],
                        'average':[]}
        

        with rasterio.open('mesageo_elephant_project/elephant_project/experiment_setup_files/environment_seethathode/Raster_Files_Seethathode_Derived/area_1100sqKm/reso_30x30/LULC.tif') as src:
            bounds = src.bounds
            self.bounds = [[bounds.left, bounds.right], [bounds.bottom, bounds.top]]

        # print(self.bounds)

    def read_all_experiments(self):
    
        path = pathlib.Path(self.data_folder)
        subfolders = [x for x in path.iterdir() if x.is_dir()]
        self.file_paths = [str(subfolder / "output_files/agent_data.csv") for subfolder in subfolders]

        self.trajectories = []

        for file_path in self.file_paths:
            try:
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    self.trajectories.append(df)
            except:
                pass
        
    def set_bounds_data(self):

        all_lons = np.concatenate([traj["longitude"].values for traj in self.trajectories])
        all_lats = np.concatenate([traj["latitude"].values for traj in self.trajectories])
        
        lon_min = min(all_lons) 
        lon_max = max(all_lons) 
        lat_min = min(all_lats) 
        lat_max = max(all_lats)

        self.bounds = np.array([(lon_min, lon_max), (lat_min, lat_max)])

    def initialize_population(self):

        population_lon = np.random.uniform(self.bounds[0][0], self.bounds[0][1], size=self.population_size*self.num_rangers)
        population_lat = np.random.uniform(self.bounds[1][0], self.bounds[1][1], size=self.population_size*self.num_rangers)
        population = np.column_stack((population_lon, population_lat))
        population = population.reshape(self.population_size, self.num_rangers, 2)

        return population

    def select_parents(self, population, fitness_scores):
        tournament_size = 3
        parents = np.zeros((self.population_size, self.num_rangers, 2))
        
        for i in range(self.population_size):
            tournament_idx = np.random.choice(self.population_size, tournament_size)
            winner_idx = tournament_idx[np.argmax(fitness_scores[tournament_idx])]
            parents[i] = population[winner_idx]
            
        return parents

    def crossover(self, parents):
        offspring = np.zeros_like(parents)
        
        for i in range(0, self.population_size, 2):
            if np.random.random() < 0.8:  # Crossover probability
                crossover_point = np.random.randint(1, self.num_rangers)
                offspring[i, :crossover_point] = parents[i, :crossover_point]
                offspring[i, crossover_point:] = parents[i+1, crossover_point:]
                offspring[i+1, :crossover_point] = parents[i+1, :crossover_point]
                offspring[i+1, crossover_point:] = parents[i, crossover_point:]
            else:
                offspring[i] = parents[i]
                offspring[i+1] = parents[i+1]
                
        return offspring

    def mutate(self, offspring):
        mutation_rate = 0.1
        
        mask = np.random.random(offspring.shape) < mutation_rate
        mutations = np.random.uniform(-0.1, 0.1, offspring.shape)
        offspring[mask] += mutations[mask]
        
        offspring[:,:,0] = np.clip(offspring[:,:,0], self.bounds[0][0], self.bounds[0][1])
        offspring[:,:,1] = np.clip(offspring[:,:,1], self.bounds[1][0], self.bounds[1][1])
        
        return offspring
