import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.optimize import linprog

class RepeatedStackelbergGame:

    def __init__(
        self,
        num_targets,
        num_defender_resources,
        defender_payoffs,
        defender_penalties,
        learning_rate=0.75,
        exploration_rate=0.5,
        decay_rate=0.80,
        eta = 10
    ):
        
        self.num_targets = num_targets
        self.num_defender_resources = num_defender_resources

        self.defender_payoffs = np.array(defender_payoffs)
        self.defender_penalties = np.array(defender_penalties)

        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.decay_rate = decay_rate
        self.eta = eta
        
        self.defender_strategy_history = []
        self.attacker_strategy_history = []

        self.estimated_adversary_payoffs = np.random.exponential(scale=1/self.eta, size=self.num_targets)

    def adversary_quantal_response(self):
        mask = self.estimated_adversary_payoffs > 0.5
        attack_probs = np.zeros_like(self.estimated_adversary_payoffs)
        attack_probs[mask] = 1.0 if np.any(mask) else 0
        return attack_probs
        
    def expected_step_utility_defender(self, attacker_strategy, defender_strategy):

        r = (self.defender_payoffs + self.defender_penalties).values
        r_t = [a * b for a, b in zip(attacker_strategy, r)]

        reward_01 = np.dot(defender_strategy, r_t)
        reward_02 = np.dot(attacker_strategy, self.defender_penalties)

        U = reward_01 + reward_02

        return U
    
    def update_model_from_observation(self, attacker_strategy):

        for i in range(self.num_targets):
            
            if attacker_strategy[i] == 1:
                self.estimated_adversary_payoffs[i] -= ((self.defender_penalties[i]) * self.learning_rate )

            else:
                self.estimated_adversary_payoffs[i] += ((self.defender_penalties[i]) * self.learning_rate)

        self.estimated_adversary_payoffs = np.clip(self.estimated_adversary_payoffs, 0, 1)
    
    def take_action(self):
        """
        Compute the optimal defender strategy using linear programming.
        """

        import numpy as np
        from itertools import combinations
        
        best_utility = float('-inf')
        best_strategy = np.zeros(self.num_targets)
        
        for targets_to_cover in combinations(range(self.num_targets), self.num_defender_resources):

            strategy = np.zeros(self.num_targets)
            strategy[list(targets_to_cover)] = 1

            attack_probability = self.adversary_quantal_response()

            utility = 0
            for i in range(self.num_targets):
                if strategy[i] == 1: 
                    utility += attack_probability[i] * self.defender_payoffs[i]
                else: 
                    utility += attack_probability[i] * self.defender_penalties[i]
            
            if utility > best_utility:
                best_utility = utility
                best_strategy = strategy.copy()
        
        if np.random.random() < self.exploration_rate:
            random_indices = np.random.choice(
                self.num_targets, 
                size=self.num_defender_resources, 
                replace=False
            )
            
            random_strategy = np.zeros(self.num_targets)
            random_strategy[random_indices] = 1
            
            best_strategy = random_strategy
        
        self.exploration_rate *= self.decay_rate
        
        return best_strategy

    
    def run_simulation(self, num_steps, elephant_simulator=None):

        for step in tqdm(range(num_steps)):

            defender_strategy_i = self.take_action()
            self.defender_strategy_history.append(defender_strategy_i)

            attacker_strategy_i = elephant_simulator()
            self.attacker_strategy_history.append(attacker_strategy_i)

            self.update_model_from_observation(attacker_strategy_i)

            print(defender_strategy_i, attacker_strategy_i)
        
        return






def elephant_strategy_simulator():

    attacks = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1]
        
    return attacks





if __name__ == "__main__":

    num_targets = 10
    num_defender_resources = 3
    
    defender_payoffs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.5]
    defender_penalties = [-0.25, -0.1, -0.1, -0.1, -0.3, -0.1, -0.4, -0.1, -0.1, -0.1]
    
    game = RepeatedStackelbergGame(
        num_targets=num_targets,
        num_defender_resources=num_defender_resources,
        defender_payoffs=defender_payoffs,
        defender_penalties = defender_penalties
    )
    
    game.run_simulation(num_steps=100, elephant_simulator=elephant_strategy_simulator)
    