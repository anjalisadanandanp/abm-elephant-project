import numpy as np
import pandas as pd
import itertools
from scipy.optimize import linprog

class SecurityGame:
    
    def __init__(self, targets_file, k, m):

        self.targets = pd.read_csv(targets_file)
        self.num_targets = len(self.targets)
        self.k = k  
        self.m = m 
        self.target_ids = self.targets['targetID'].values
        
        self.rewards = self.targets['reward'].values
        self.penalties = self.targets['penalty'].values
        
        self.attacker_pure_strategies = self._generate_attacker_strategies()
        self.num_attacker_strategies = len(self.attacker_pure_strategies)
        self.defender_pure_strategies = self._generate_defender_strategies()

        self.num_defender_strategies = len(self.defender_pure_strategies)

        print(f"Number of targets: {self.num_targets}", 
              f"\nNumber of attacker pure strategies: {self.num_attacker_strategies}",
              f"\nNumber of defender pure strategies: {self.num_defender_strategies}")
        
    def _generate_attacker_strategies(self):
        strategies = []
        
        for i in range(1, self.m + 1):
            for targets_to_attack in itertools.combinations(range(self.num_targets), i):
                strategy = np.zeros(self.num_targets, dtype=int)
                strategy[list(targets_to_attack)] = 1
                strategies.append(strategy)
        
        return strategies

    def _generate_defender_strategies(self):
        strategies = []
        
        for i in range(1, self.k + 1):
            for targets_to_protect in itertools.combinations(range(self.num_targets), i):
                strategy = np.zeros(self.num_targets, dtype=int)
                strategy[list(targets_to_protect)] = 1
                strategies.append(strategy)
        
        return strategies
    
    def compute_expected_defender_utility(self, defender_strategy, attacker_strategy):
        
        utility = 0
        for i in range(self.num_targets):
            if attacker_strategy[i] == 1: 
                utility += defender_strategy[i] * self.rewards[i] + (1 - defender_strategy[i]) * self.penalties[i]
        
        return utility
    
    def compute_payoff_matrix(self):

        payoff_matrix = np.zeros((self.num_defender_strategies, self.num_attacker_strategies))
        
        for i, defender_strategy in enumerate(self.defender_pure_strategies):
            for j, attacker_strategy in enumerate(self.attacker_pure_strategies):
                payoff_matrix[i, j] = self.compute_expected_defender_utility(
                    defender_strategy, attacker_strategy)
        
        return payoff_matrix
    
    def choose_random_defender_strategy(self, num_game_rounds):
        defender_strategies_history = []
        
        for _ in range(num_game_rounds):
            defender_strategy = np.random.choice(self.num_defender_strategies)
            defender_strategies_history.append(self.defender_pure_strategies[defender_strategy])
        
        return defender_strategies_history
    
    def maximin_mixed_strategy(self):

        payoff_matrix = self.compute_payoff_matrix()

        print("payoff matrix: \n", payoff_matrix)

        c = np.zeros(self.num_defender_strategies + 1)
        c[-1] = 1

        print("cost:", c)
        
        A_ub = np.zeros((self.num_attacker_strategies, self.num_defender_strategies + 1))

        for i in range(self.num_attacker_strategies):
            A_ub[i, :-1] = payoff_matrix[:, i]
            A_ub[i, -1] = -1

        print("A_ub:", A_ub)
        
        b_ub = np.zeros(self.num_attacker_strategies)

        print("b_ub:", b_ub)
        
        A_eq = np.zeros((1, self.num_defender_strategies + 1))
        A_eq[0, :-1] = 1
        b_eq = np.ones(1)

        print("A_eq:", A_eq, "b_eq:", b_eq)

        bounds = [(0,1) for val in range(self.num_defender_strategies)]
        bounds.append((None,None))

        print("bounds:", bounds)
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        
        if result.success:
            attacker_mixed_strategy = result.x[:-1] 
            return attacker_mixed_strategy
        
        else:
            print("Linear programming solution failed!")
            return None
    
    def get_optimal_strategy_for_defender(self):

        mixed_strategy = self.maximin_mixed_strategy()
        
        return {
            'mixed_strategy': mixed_strategy
        }
    




if __name__ == "__main__":


    k = 2  # defender resources to protect the targets
    m = 2  # number of targets attacker can attack in one game round
    game = SecurityGame("game_theory_codes/attacker-strategies/adversarial/code/target_rewards_penalties.csv", k, m)
    
    optimal_strategy = game.get_optimal_strategy_for_defender()

    print(optimal_strategy)

    for probability,strategy in zip(optimal_strategy["mixed_strategy"], game.defender_pure_strategies):
        if probability > 0:
            print(strategy, probability)


