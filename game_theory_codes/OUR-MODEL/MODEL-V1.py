import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class RepeatedStackelbergGame:

    def __init__(
        self,
        num_targets,
        num_defender_resources,
        adversary_payoffs,
        adversary_penalties,
        defender_payoffs,
        defender_penalties,
        lamda=1.0,
        learning_rate=0.5,
        exploration_rate=0.25,
        decay_rate=0.95
    ):
        
        self.num_targets = num_targets
        self.num_defender_resources = num_defender_resources

        self.adversary_payoffs = np.array(adversary_payoffs)
        self.adversary_penalties = np.array(adversary_penalties)
        self.defender_payoffs = np.array(defender_payoffs)
        self.defender_penalties = np.array(defender_penalties)

        self.lamda = lamda
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.decay_rate = decay_rate
        

        self.defender_strategy_history = []
        self.attack_history = []
        self.defender_utility_history = []
        self.adversary_utility_history = []
        
        self.estimated_adversary_payoffs = np.zeros_like(adversary_payoffs)
        self.estimated_adversary_penalties = np.zeros_like(adversary_penalties)
        
        self.current_defender_strategy = np.ones(num_targets) * (num_defender_resources / num_targets)
        self.current_defender_strategy = self.project_to_simplex(self.current_defender_strategy)

    def project_to_simplex(self, v):
        """Project vector to simplex with sum = num_defender_resources"""
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - self.num_defender_resources
        rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
        theta = cssv[rho] / (rho + 1)
        w = np.maximum(v - theta, 0)
        return np.clip(w, 0, 1)
    
    def expected_utility_adversary(self, defender_coverage):
        """Calculate expected utility for adversary for each target"""
        U = defender_coverage * self.estimated_adversary_penalties + \
            (1 - defender_coverage) * self.estimated_adversary_payoffs
        return U
    
    def adversary_quantal_response(self, expected_utilities):
        """Calculate quantal response (attack probabilities) given expected utilities"""
        exp_terms = np.exp(self.lamda * expected_utilities)
        return exp_terms / np.sum(exp_terms)
    
    def expected_utility_defender(self, attack_probabilities, defender_coverage):
        """Calculate expected utility for defender"""
        U = attack_probabilities * (
            defender_coverage * self.defender_payoffs + 
            (1 - defender_coverage) * self.defender_penalties
        )
        return U
    
    def compute_gradient(self, defender_coverage):
        """Compute gradient of defender's expected utility"""

        adv_utility = self.expected_utility_adversary(defender_coverage)
        
        attack_probs = self.adversary_quantal_response(adv_utility)

        print("attack probs:", attack_probs)
        
        gradient = np.zeros(self.num_targets)
        
        exp_terms = np.exp(self.lamda * adv_utility)
        sum_exp = np.sum(exp_terms)
        
        for i in range(self.num_targets):

            direct_effect = attack_probs[i] * (self.adversary_payoffs[i] - self.adversary_penalties[i])

            print("direct_effect:", direct_effect)
            

            # for j in range(self.num_targets):
            #     dU_adv_j = self.estimated_adversary_penalties[j] - self.estimated_adversary_payoffs[j] if j == i else 0
                
            #     dQ_j = (
            #         self.lamda * exp_terms[j] * 
            #         (dU_adv_j * sum_exp - exp_terms[j] * dU_adv_j) / 
            #         (sum_exp * sum_exp)
            #     )
                
            #     indirect_effect += dQ_j * (
            #         defender_coverage[j] * self.defender_payoffs[j] +
            #         (1 - defender_coverage[j]) * self.defender_penalties[j]
            #     )

            indirect_effect = defender_coverage[i] * (self.defender_payoffs[i] - self.defender_penalties[i])

            print("indirect_effect:", indirect_effect)
            
            gradient[i] = direct_effect + indirect_effect

        print("gradient:", gradient)
        
        return gradient
    
    def generate_random_strategy(self):
        """Generate a random strategy for exploration"""
        return self.project_to_simplex(np.random.random(self.num_targets))
    
    def update_model_from_observation(self, defender_strategy, actual_attacks):
        """
        Update our model of elephant behavior based on actual observation
        actual_attacks: Binary vector indicating which targets were attacked
        """

        defender_strategy = np.array(defender_strategy)
        actual_attacks = np.array(actual_attacks)
        
        model_lr = 1
        
        # For each target that was attacked, update our understanding of elephant preferences
        for i in range(self.num_targets):

            if actual_attacks[i] == 1:

                # If target was attacked 
                # print("updating reward estimate!", self.estimated_adversary_payoffs[i])
                self.estimated_adversary_payoffs[i] += model_lr
                # print("updated reward estimate!", self.estimated_adversary_payoffs[i])

            # else:
            #     # If target was not attacked despite being unprotected, it may be less valuable
            #     if defender_strategy[i] > 0:
            #         self.estimated_adversary_payoffs[i] -= model_lr 
            #         # self.estimated_adversary_penalties[i] -= model_lr 
        
        # Ensure estimated values stay within reasonable bounds
        self.estimated_adversary_payoffs = np.clip(self.estimated_adversary_payoffs, 0.1, 10)
        # self.estimated_adversary_penalties = np.clip(self.estimated_adversary_penalties, -10, -0.1)
    
    def optimize_br_qr(self, num_iterations=100):
        """Find the best BR-QR strategy using gradient descent"""
        coverage = self.current_defender_strategy.copy()
        
        for _ in range(num_iterations):
            gradient = self.compute_gradient(coverage)
            coverage = coverage + self.learning_rate * gradient

            print("coverage", coverage)

            coverage = self.project_to_simplex(coverage)
            
            if np.sum(np.abs(coverage - self.current_defender_strategy)) < 1e-5:
                break

        # print(coverage)
        
        return coverage
    
    def take_action(self, actual_attacks=None):

        self.defender_strategy_history.append(self.current_defender_strategy.copy())
        
        if np.random.random() < self.exploration_rate:
            strategy = self.generate_random_strategy()
        else:
            strategy = self.optimize_br_qr()
        
        # If no actual attacks provided, simulate them using our model
        if actual_attacks is None:
            adv_utility = self.expected_utility_adversary(strategy)
            attack_probs = self.adversary_quantal_response(adv_utility)
            actual_attacks = np.random.binomial(1, attack_probs)
        
        # Record observed attacks
        self.attack_history.append(actual_attacks)
        
        # Calculate utilities
        def_utility = np.sum(self.expected_utility_defender(actual_attacks, strategy))
        adv_utility = np.sum(self.expected_utility_adversary(strategy) * actual_attacks)
        
        self.defender_utility_history.append(def_utility)
        self.adversary_utility_history.append(adv_utility)
        
        self.update_model_from_observation(strategy, actual_attacks)
        
        self.exploration_rate *= self.decay_rate
        
        self.current_defender_strategy = strategy
        
        return strategy
    
    def run_simulation(self, num_steps, elephant_simulator=None):
        """
        Run the repeated game for multiple steps
        elephant_simulator: Function that simulates elephant behavior given defender strategy
        """
        all_strategies = []
        
        for step in tqdm(range(num_steps)):
            if elephant_simulator:
                strategy = self.current_defender_strategy.copy()
                actual_attacks = elephant_simulator(strategy, step)
                strategy = self.take_action(actual_attacks)
            else:
                strategy = self.take_action()
            
            all_strategies.append(strategy)
        
        return all_strategies
    
    def plot_results(self):
        """Plot the results of the simulation"""
        steps = len(self.defender_strategy_history)
        
        plt.figure(figsize=(12, 12))
        
        plt.subplot(2, 2, 1)
        strategies = np.array(self.defender_strategy_history)
        for i in range(self.num_targets):
            plt.plot(range(steps), strategies[:, i], label=f'Target {i+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Coverage Probability')
        plt.title('Defender Strategy Evolution')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 2)
        attack_freq = np.zeros((steps, self.num_targets))
        window = min(10, steps)
        for t in range(steps):
            if t < window:
                attack_freq[t] = np.mean([self.attack_history[max(0, i)] for i in range(t+1)], axis=0)
            else:
                attack_freq[t] = np.mean([self.attack_history[i] for i in range(t-window, t+1)], axis=0)
        
        for i in range(self.num_targets):
            plt.plot(range(steps), attack_freq[:, i], label=f'Target {i+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Attack Frequency')
        plt.title('Attack Pattern Evolution (Moving Average)')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # plt.subplot(2, 2, 3)
        # plt.plot(range(steps), self.defender_utility_history, 'b-', label='Defender')
        # plt.plot(range(steps), self.adversary_utility_history, 'r-', label='Adversary')
        # plt.xlabel('Time Step')
        # plt.ylabel('Expected Utility')
        # plt.title('Utility Evolution')
        # plt.legend()
        # plt.grid(alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.bar(np.arange(self.num_targets) - 0.2, self.adversary_penalties, width=0.4, label='Actual Penalty')
        plt.bar(np.arange(self.num_targets) + 0.2, self.estimated_adversary_penalties, width=0.4, label='Estimated Penalty')
        plt.xlabel('Target')
        plt.ylabel('Payoff Value')
        plt.title('Actual vs. Estimated Adversary Penalty')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.bar(np.arange(self.num_targets) - 0.2, self.adversary_payoffs, width=0.4, label='Actual Payoff')
        plt.bar(np.arange(self.num_targets) + 0.2, self.estimated_adversary_payoffs, width=0.4, label='Estimated Payoff')
        plt.xlabel('Target')
        plt.ylabel('Payoff Value')
        plt.title('Actual vs. Estimated Adversary Payoffs')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()



def elephant_strategy_simulator(defender_strategy, time_step):

    num_targets = len(defender_strategy)
    
    attack_probs = np.zeros(num_targets)

    for i in range(num_targets):

        base_attraction = 0.3 + 0.5 * np.sin(time_step/10 + i)
        
        deterrence = 0.8 * defender_strategy[i]
        
        attack_probs[i] = max(0, min(1, base_attraction - deterrence))
    
    attacks = np.random.binomial(1, attack_probs)
    
    if np.sum(attacks) == 0:
        attacks[np.argmax(attack_probs)] = 1

    attacks = [1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
        
    return attacks


if __name__ == "__main__":

    num_targets = 10
    num_defender_resources = 3
    
    adversary_payoffs = [3, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    adversary_penalties = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    defender_payoffs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    defender_penalties = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    game = RepeatedStackelbergGame(
        num_targets=num_targets,
        num_defender_resources=num_defender_resources,
        adversary_payoffs=adversary_payoffs,
        adversary_penalties=adversary_penalties,
        defender_payoffs=defender_payoffs,
        defender_penalties=defender_penalties,
        lamda=0.5, 
        exploration_rate=0.10,
        decay_rate=0.75
    )
    
    game.run_simulation(500, elephant_simulator=elephant_strategy_simulator)
    
    game.plot_results()