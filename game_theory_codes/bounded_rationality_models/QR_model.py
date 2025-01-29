import numpy as np
import matplotlib.pyplot as plt

def calculate_qr_probabilities(utilities, lambda_val):
    """
    Calculate probabilities using the Quantal Response model.
    
    Args:
        utilities (np.array): Array of utilities for each action
        lambda_val (float): Rationality parameter
        
    Returns:
        np.array: Probabilities for each action
    """
    # Calculate exponential terms
    exp_terms = np.exp(lambda_val * utilities)
    
    # Normalize to get probabilities
    probabilities = exp_terms / np.sum(exp_terms)
    
    return probabilities

# Example scenario: Protecting 3 targets
# Utilities represent the attacker's expected payoff for attacking each target
utilities = np.array([8, 5, 2])  # Target 1 has highest utility, Target 3 lowest
n_targets = len(utilities)

# Test different lambda values
lambda_values = [0, 0.1, 0.5, 1, 2, 5]
plt.figure(figsize=(12, 6))

for idx, lambda_val in enumerate(lambda_values):
    probs = calculate_qr_probabilities(utilities, lambda_val)
    
    # Plot probabilities for each target
    plt.plot(range(n_targets), probs, 'o-', label=f'λ={lambda_val}')

plt.xlabel('Target')
plt.ylabel('Attack Probability')
plt.title('QR Model: Attack Probabilities vs Target')
plt.xticks(range(n_targets), [f'Target {i+1}\n(Utility={u})' for i, u in enumerate(utilities)])
plt.grid(True)
plt.legend()
plt.tight_layout()

# Now let's show how probabilities change with lambda for the highest utility target
lambda_range = np.linspace(0, 5, 100)
probs_target1 = []

for lambda_val in lambda_range:
    probs = calculate_qr_probabilities(utilities, lambda_val)
    probs_target1.append(probs[0])  # Probability for Target 1

plt.figure(figsize=(10, 6))
plt.plot(lambda_range, probs_target1)
plt.xlabel('λ (Rationality Parameter)')
plt.ylabel('Probability of Choosing Target 1')
plt.title('Probability of Choosing Highest-Utility Target vs λ')
plt.grid(True)
plt.tight_layout()
plt.show()