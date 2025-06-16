import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from game_theory_codes.game_rewards.GA_optimisation import RangerOptimizer



def fitness_function(positions):
    center = np.array([np.mean(positions[:,:,0]), np.mean(positions[:,:,1])])
    distances = np.sqrt(np.sum((positions - center)**2, axis=2))
    return np.sum(distances, axis=1)

def test_ranger_optimizer():
    optimizer = RangerOptimizer(
        num_rangers=4,
        population_size=10,
        generations=1,
        data_folder="path/to/data",
        tournament_size=2
    )
    
    population = optimizer.initialize_population_v2()
    assert population.shape == (10, 4, 2)
    # assert np.all(population[:,:,0] >= 0) and np.all(population[:,:,0] <= 100)
    
    fitness_scores = fitness_function(population)

    for i, (pop, fit) in enumerate(zip(population, fitness_scores)):
        print(f"Individual {i}: Position={pop.round(3)}, Fitness={fit:.3f}")
    
    print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
    parents = optimizer.select_parents(population, fitness_scores)
    assert parents.shape == population.shape

    print("parents selected:", parents)

    offspring = optimizer.crossover(parents)
    assert offspring.shape == population.shape

    print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")

    print("offsprings:", offspring)
    
    mutated = optimizer.mutate_v2(offspring)
    assert mutated.shape == population.shape
    # assert np.all(mutated[:,:,0] >= 0) and np.all(mutated[:,:,0] <= 100)

    print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")

    print("mutated:", mutated)
    
    print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_ranger_optimizer()