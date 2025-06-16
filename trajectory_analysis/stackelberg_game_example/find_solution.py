import numpy as np
from typing import Callable, Tuple, List

class BackwardInduction:

    def __init__(
        self,
        player1_payoff: Callable[[float, float], float],
        player2_payoff: Callable[[float, float], float],
        strategy_space: Tuple[List[float], List[float]]
    ):
        """
        Initialize the backward induction solver.
        
        Args:
            player1_payoff: Function that takes (s1, s2) and returns player 1's payoff
            player2_payoff: Function that takes (s1, s2) and returns player 2's payoff
            strategy_space: Tuple of (player1_strategies, player2_strategies)
        """
        self.player1_payoff = player1_payoff
        self.player2_payoff = player2_payoff
        self.player1_strategies = strategy_space[0]
        self.player2_strategies = strategy_space[1]
        
    def find_best_response(
        self,
        player1_strategy: float,
        player: int
    ) -> Tuple[float, float]:
        """
        Find the best response for a player given the other player's strategy.
        
        Args:
            player1_strategy: The strategy chosen by player 1
            player: Which player's best response to find (1 or 2)
            
        Returns:
            Tuple of (best_strategy, best_payoff)
        """
        if player == 1:
            strategies = self.player1_strategies
            payoff_func = lambda s: self.player1_payoff(s, player1_strategy)
        else:
            strategies = self.player2_strategies
            payoff_func = lambda s: self.player2_payoff(player1_strategy, s)
            
        payoffs = [payoff_func(s) for s in strategies]
        best_idx = np.argmax(payoffs)
        return strategies[best_idx], payoffs[best_idx]
    
    def solve(self) -> Tuple[float, float, float, float]:
        """
        Solve the game using backward induction.
        
        Returns:
            Tuple of (player1_strategy, player2_strategy, player1_payoff, player2_payoff)
        """
        # Step 1: For each possible player 1 strategy, find player 2's best response
        player2_responses = {}
        for s1 in self.player1_strategies:
            best_s2, _ = self.find_best_response(s1, 2)
            player2_responses[s1] = best_s2
            
        # Step 2: Given player 2's best responses, find player 1's best strategy
        best_payoff = float('-inf')
        best_s1 = None
        best_s2 = None
        
        for s1 in self.player1_strategies:
            s2 = player2_responses[s1]
            payoff = self.player1_payoff(s1, s2)
            
            if payoff > best_payoff:
                best_payoff = payoff
                best_s1 = s1
                best_s2 = s2
                
        # Calculate final payoffs
        final_p1_payoff = self.player1_payoff(best_s1, best_s2)
        final_p2_payoff = self.player2_payoff(best_s1, best_s2)
        
        return best_s1, best_s2, final_p1_payoff, final_p2_payoff

# Example usage for Stackelberg competition
def stackelberg_example():
    # Parameters
    a = 100  # Maximum price
    b = 1    # Demand slope
    c1 = c2 = 20  # Marginal costs
    
    # Payoff functions
    def price(s1, s2):
        return a - b * (s1 + s2)
    
    def player1_payoff(s1, s2):
        p = price(s1, s2)
        return p * s1 - c1 * s1
    
    def player2_payoff(s1, s2):
        p = price(s1, s2)
        return p * s2 - c2 * s2
    
    # Create strategy spaces (discrete approximation)
    strategy_space = (
        np.linspace(0, 80, 81).tolist(),  # Player 1 strategies
        np.linspace(0, 80, 81).tolist()   # Player 2 strategies
    )
    
    # Solve the game
    solver = BackwardInduction(player1_payoff, player2_payoff, strategy_space)
    s1, s2, p1_payoff, p2_payoff = solver.solve()
    
    return s1, s2, p1_payoff, p2_payoff

# Run the example
if __name__ == "__main__":
    s1, s2, p1_payoff, p2_payoff = stackelberg_example()
    print(f"Player 1 strategy: {s1:.2f}")
    print(f"Player 2 strategy: {s2:.2f}")
    print(f"Player 1 payoff: {p1_payoff:.2f}")
    print(f"Player 2 payoff: {p2_payoff:.2f}")