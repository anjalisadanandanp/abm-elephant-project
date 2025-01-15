import numpy as np
from typing import Callable, Tuple, List

class GameEquilibrium:

    def __init__(
        self,
        a: float,  # Maximum price
        b: float,  # Demand slope
        c1: float, # Marginal cost for firm 1
        c2: float  # Marginal cost for firm 2
    ):
        self.a = a
        self.b = b
        self.c1 = c1
        self.c2 = c2

    def price(self, q1: float, q2: float) -> float:
        """Calculate market price given quantities"""
        return self.a - self.b * (q1 + q2)

    def profit1(self, q1: float, q2: float) -> float:
        """Calculate firm 1's profit"""
        p = self.price(q1, q2)
        return q1 * (p - self.c1)

    def profit2(self, q1: float, q2: float) -> float:
        """Calculate firm 2's profit"""
        p = self.price(q1, q2)
        return q2 * (p - self.c2)

    def cournot_equilibrium(self) -> Tuple[float, float, float, float]:
        """
        Calculate Cournot equilibrium quantities and profits.
        In Cournot, firms choose quantities simultaneously.
        """
        # Best response functions derived from first-order conditions
        # For symmetric costs, quantities are equal
        q1 = (self.a - 2*self.c1 + self.c2) / (3*self.b)
        q2 = (self.a - 2*self.c2 + self.c1) / (3*self.b)
        
        # Calculate profits at equilibrium
        profit1 = self.profit1(q1, q2)
        profit2 = self.profit2(q1, q2)
        
        return q1, q2, profit1, profit2

    def stackelberg_equilibrium(self) -> Tuple[float, float, float, float]:
        """
        Calculate Stackelberg equilibrium quantities and profits.
        Firm 1 is the leader, Firm 2 is the follower.
        """
        # Firm 2's best response function
        # q2(q1) = (a - b*q1 - c2)/(2*b)
        
        # Firm 1's optimal quantity considering Firm 2's response
        q1 = (self.a - 2*self.c1 + self.c2) / (2*self.b)
        
        # Firm 2's best response to q1
        q2 = (self.a - self.c2) / (2*self.b) - q1/2
        
        # Calculate profits at equilibrium
        profit1 = self.profit1(q1, q2)
        profit2 = self.profit2(q1, q2)
        
        return q1, q2, profit1, profit2

def compare_equilibria():
    # Example parameters
    a = 100  # Maximum price
    b = 1    # Demand slope
    c1 = c2 = 20  # Marginal costs (symmetric case)
    
    game = GameEquilibrium(a, b, c1, c2)
    
    # Calculate Cournot equilibrium
    cq1, cq2, cp1, cp2 = game.cournot_equilibrium()
    
    # Calculate Stackelberg equilibrium
    sq1, sq2, sp1, sp2 = game.stackelberg_equilibrium()
    
    print("Cournot Equilibrium:")
    print(f"Firm 1 quantity: {cq1:.2f}")
    print(f"Firm 2 quantity: {cq2:.2f}")
    print(f"Firm 1 profit: {cp1:.2f}")
    print(f"Firm 2 profit: {cp2:.2f}")
    print(f"Market price: {game.price(cq1, cq2):.2f}")
    print(f"Total quantity: {cq1 + cq2:.2f}")
    print("\nStackelberg Equilibrium:")
    print(f"Firm 1 quantity: {sq1:.2f}")
    print(f"Firm 2 quantity: {sq2:.2f}")
    print(f"Firm 1 profit: {sp1:.2f}")
    print(f"Firm 2 profit: {sp2:.2f}")
    print(f"Market price: {game.price(sq1, sq2):.2f}")
    print(f"Total quantity: {sq1 + sq2:.2f}")

if __name__ == "__main__":
    compare_equilibria()