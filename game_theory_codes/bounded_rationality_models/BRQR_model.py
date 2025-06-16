import numpy as np
import matplotlib.pyplot as plt

def qr_probabilities(utilities, lambda_val):
    """Standard QR model probabilities."""
    exp_terms = np.exp(lambda_val * utilities)
    return exp_terms / np.sum(exp_terms)

def brqr_probabilities(utilities, lambda_val, sigma, n_samples=1000):
    """
    BRQR model probabilities with perception noise.
    
    Args:
        utilities: True utilities of targets
        lambda_val: Rationality parameter
        sigma: Standard deviation of perception noise
        n_samples: Number of Monte Carlo samples
    """
    n_targets = len(utilities)
    accumulated_probs = np.zeros(n_targets)
    
    # Monte Carlo simulation of perception noise
    for _ in range(n_samples):
        # Generate noisy perceived utilities
        perception_noise = np.random.normal(0, sigma, n_targets)
        perceived_utilities = utilities + perception_noise
        
        # Calculate QR probabilities for these perceived utilities
        probs = qr_probabilities(perceived_utilities, lambda_val)
        accumulated_probs += probs
    
    # Average probabilities over all samples
    return accumulated_probs / n_samples

# Example scenario
utilities = np.array([8, 5, 2])  # Three targets with different utilities
lambda_values = [0.5, 1, 2]
sigmas = [0, 1, 3]  # Different levels of perception noise

# Create comparison plots
fig, axes = plt.subplots(len(lambda_values), 1, figsize=(10, 12))
fig.suptitle('Comparison of QR vs BRQR Models')

for i, lambda_val in enumerate(lambda_values):
    ax = axes[i]
    
    # Calculate probabilities for different noise levels
    for j, sigma in enumerate(sigmas):
        if sigma == 0:
            # Standard QR
            probs = qr_probabilities(utilities, lambda_val)
            label = 'QR (no noise)'
        else:
            # BRQR with perception noise
            probs = brqr_probabilities(utilities, lambda_val, sigma)
            label = f'BRQR (σ={sigma})'
            
        ax.bar(np.arange(len(utilities))+j*0.25, probs, 
               width=0.25, label=label, alpha=0.7)
    
    ax.set_xlabel('Target')
    ax.set_ylabel('Attack Probability')
    ax.set_title(f'λ={lambda_val}')
    ax.set_xticks(range(len(utilities)))
    ax.set_xticklabels([f'Target {i+1}\n(U={u})' for i, u in enumerate(utilities)])
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()  