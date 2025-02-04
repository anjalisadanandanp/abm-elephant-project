import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from game_theory_codes.bounded_rationality_models.solve_BRQR import brqr_multiple_starts, expected_utility_adversary, adversary_quantal_response

adversary_df = pd.read_csv("game_theory_codes/game_rewards/outputs/attacker_rewards_penalties.csv")
defender_df = pd.read_csv("game_theory_codes/game_rewards/outputs/defender_rewards_penalties.csv")

global num_targets_to_protect
num_targets_to_protect = 438
num_defender_resources = 10
num_starts = 25
lamda = 1

adversary_payoffs = adversary_df["reward"].values
adversary_penaltys = adversary_df["penalty"].values

defender_payoffs = defender_df["reward"].values
defender_penaltys = defender_df["penalty"].values

global_opt, global_x_opt, all_starts, all_results, all_values, adversary_utility = (
    brqr_multiple_starts(
        num_targets_to_protect,
        num_defender_resources,
        adversary_payoffs,
        adversary_penaltys,
        defender_payoffs,
        defender_penaltys,
        lamda,
        num_starts,
    )
)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for i in range(num_starts):
    plt.scatter(
        range(num_targets_to_protect),
        all_starts[i],
        alpha=0.3,
        marker="o",
        label=f"Starting Points" if i == 0 else None,
        s=1,
        color="b",
    )
    plt.plot(
        range(num_targets_to_protect), all_starts[i], "b--", linewidth=0.1, zorder=1
    )
    plt.scatter(
        range(num_targets_to_protect),
        all_results[i],
        alpha=0.3,
        marker="s",
        label=f"Converged Results" if i == 0 else None,
        s=1,
        color="g",
    )
    plt.plot(
        range(num_targets_to_protect),
        all_results[i],
        "g--",
        linewidth=0.4,
        zorder=1,
    )

plt.plot(
    range(num_targets_to_protect),
    global_x_opt,
    "r-",
    linewidth=2,
    label="Global Best",
    zorder=2,
)
plt.xlabel("Target")
plt.ylabel("Coverage Probability")
plt.title("Starting Points and Results")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(range(num_starts), -np.array(all_values), "bo-")
plt.axhline(y=global_opt, color="r", linestyle="--", label="Global Best")
plt.xlabel("Start Number")
plt.ylabel("Defender Expected Utility")
plt.title("Results from Different Starting Points")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(range(num_starts), np.array(adversary_utility), "bo-")
plt.xlabel("Start Number")
plt.ylabel("Adversary Expected Utility")
plt.title("Results from Different Starting Points")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nGlobal best defender utility: {global_opt:.4f}")
print("Global best coverage strategy:", global_x_opt)
print("total number of defender resources:", sum(global_x_opt))

U_final = expected_utility_adversary(
    global_x_opt, adversary_payoffs, adversary_penaltys, num_targets_to_protect
)
Q_final = adversary_quantal_response(lamda, U_final, num_targets_to_protect)
print("\nAdversary's expected utilities under best strategy:", sum(U_final))
print("\nAdversary's attack probabilities under best strategy:", Q_final)
