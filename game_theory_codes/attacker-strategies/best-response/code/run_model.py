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
    
    def calculate_best_response_attacker_strategy(self, defender_strategy):
        
        expected_utilities = np.zeros(self.num_attacker_strategies)
        
        defender_coverage = np.zeros(self.num_targets)
        for i, pure_strategy in enumerate(self.defender_pure_strategies):
            defender_coverage += pure_strategy * defender_strategy[i]
        
        for j in range(self.num_attacker_strategies):
            attacker_strategy = self.attacker_pure_strategies[j]
            utility = 0
            for t in range(self.num_targets):
                if attacker_strategy[t] == 1:
                    utility += defender_coverage[t] * self.rewards[t] + (1 - defender_coverage[t]) * self.penalties[t]
            expected_utilities[j] = -utility
        
        best_response_idx = np.argmax(expected_utilities)
        best_response = self.attacker_pure_strategies[best_response_idx]
        
        return {
            'best_response_strategy': best_response,
            'strategy_index': best_response_idx,
            'expected_utility': expected_utilities[best_response_idx]
        }   



if __name__ == "__main__":


    k = 1  # defender resources to protect the targets
    m = 1  # number of targets attacker can attack in one game round
    game = SecurityGame("game_theory_codes/attacker-strategies/adversarial/code/target_rewards_penalties.csv", k, m)

    print(game.calculate_best_response_attacker_strategy([0.5, 0.25, 0.25, 0.25]))




