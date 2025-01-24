import numpy as np
import matplotlib.pyplot as plt

class GameTheoryPlotter:
    def __init__(self, a=100, b=1, c1=20, c2=20):
        self.a = a  # Maximum price
        self.b = b  # Demand slope
        self.c1 = c1  # Marginal cost for firm 1
        self.c2 = c2  # Marginal cost for firm 2
        
        # Calculate equilibrium points
        self.cournot_q = (self.a - self.c1) / (3 * self.b)
        self.stackelberg_leader = (self.a - self.c1) / (2 * self.b)
        self.stackelberg_follower = (self.a - self.b * self.stackelberg_leader - self.c2) / (2 * self.b)
    
    def price(self, q1, q2):
        """Calculate market price given quantities"""
        return self.a - self.b * (q1 + q2)
    
    def profit1(self, q1, q2):
        """Calculate firm 1's profit"""
        p = self.price(q1, q2)
        return q1 * (p - self.c1)
    
    def profit2(self, q1, q2):
        """Calculate firm 2's profit"""
        p = self.price(q1, q2)
        return q2 * (p - self.c2)
    
    def plot_payoffs(self, q1_fixed):
        """Plot payoffs for both players given a fixed strategy for player 1"""
        q2_range = np.linspace(0, 80, 200)
        profit1_values = [self.profit1(q1_fixed, q2) for q2 in q2_range]
        profit2_values = [self.profit2(q1_fixed, q2) for q2 in q2_range]
        
        plt.figure(figsize=(12, 8))
        
        # Plot profit curves
        plt.plot(q2_range, profit1_values, 'b-', label=f'Player 1 Profit (q1={q1_fixed:.1f})')
        plt.plot(q2_range, profit2_values, 'g-', label='Player 2 Profit')
        
        # Add vertical lines for equilibrium points
        plt.axvline(x=self.cournot_q, color='r', linestyle='--', alpha=0.5, 
                   label=f'Cournot Quantity ({self.cournot_q:.1f})')
        plt.axvline(x=self.stackelberg_follower, color='orange', linestyle='--', alpha=0.5,
                   label=f'Stackelberg Follower ({self.stackelberg_follower:.1f})')
        
        # Add horizontal line at zero profit
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Formatting
        plt.title(f'Payoffs vs Player 2 Strategy (Player 1 Strategy = {q1_fixed:.1f})')
        plt.xlabel('Player 2 Quantity (q2)')
        plt.ylabel('Profit')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return plt
    
    def plot_best_response(self):
        """Plot best response functions for both players"""
        q_range = np.linspace(0, 80, 200)
        
        # Best response functions
        br1 = [(self.a - self.c1 - self.b * q2) / (2 * self.b) for q2 in q_range]
        br2 = [(self.a - self.c2 - self.b * q1) / (2 * self.b) for q1 in q_range]
        
        plt.figure(figsize=(10, 10))
        
        # Plot best response curves
        plt.plot(q_range, br1, 'b-', label='Player 1 Best Response')
        plt.plot(br2, q_range, 'g-', label='Player 2 Best Response')
        
        # Add equilibrium points
        plt.plot(self.cournot_q, self.cournot_q, 'ro', 
                label=f'Cournot Equilibrium ({self.cournot_q:.1f}, {self.cournot_q:.1f})')
        plt.plot(self.stackelberg_leader, self.stackelberg_follower, 'ko',
                label=f'Stackelberg Equilibrium ({self.stackelberg_leader:.1f}, {self.stackelberg_follower:.1f})')
        
        # Formatting
        plt.title('Best Response Functions')
        plt.xlabel('Player 1 Quantity (q1)')
        plt.ylabel('Player 2 Quantity (q2)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return plt

# Example usage
if __name__ == "__main__":
    plotter = GameTheoryPlotter()
    
    # Plot payoffs for different fixed strategies of player 1
    strategies = [plotter.cournot_q, plotter.stackelberg_leader]
    
    for q1 in strategies:
        plt = plotter.plot_payoffs(q1)
        plt.show()
    
    # Plot best response functions
    plt = plotter.plot_best_response()
    plt.show()