import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm





def expected_utility_adversary(
    defender_coverage_probability, adversary_payoffs, adversary_penaltys, num_targets_to_protect
):
    # returns the expected utility of the adversary for attacking each target

    U = np.zeros(num_targets_to_protect)

    for i in range(num_targets_to_protect):
        U[i] = (
            defender_coverage_probability[i] * adversary_penaltys[i]
            + (1 - defender_coverage_probability[i]) * adversary_payoffs[i]
        )

    return U


def adversary_quantal_response(lamda, U, num_targets_to_protect):
    # returns the quantal response of the adversary for attacking each target

    Q = np.zeros(num_targets_to_protect)

    for i in range(num_targets_to_protect):
        Q[i] = np.exp(lamda * U[i]) / np.sum(np.exp(lamda * U))

    return Q


def expected_utility_defender(
    Q, defender_payoffs, defender_penaltys, defender_coverage_probability, num_targets_to_protect
):
    # returns the expected utility of the defender for defending each target

    U = np.zeros(num_targets_to_protect)

    for i in range(num_targets_to_protect):
        U[i] = Q[i] * (
            defender_coverage_probability[i] * defender_payoffs[i]
            + (1 - defender_coverage_probability[i]) * defender_penaltys[i]
        )

    return U


def project_to_simplex(v, s=4.0):
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - s
    rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1)
    w = np.maximum(v - theta, 0)
    return np.clip(w, 0, 1)


def compute_gradient(
    defender_coverage,
    adversary_payoffs,
    adversary_penaltys,
    defender_payoffs,
    defender_penaltys,
    lamda,
    num_targets_to_protect
):
    """
    Compute gradient of defender's expected utility with respect to coverage probabilities
    """

    U_adv = expected_utility_adversary(
        defender_coverage, adversary_payoffs, adversary_penaltys, num_targets_to_protect
    )

    Q = adversary_quantal_response(lamda, U_adv, num_targets_to_protect)

    gradient = np.zeros(len(defender_coverage))

    exp_terms = np.exp(lamda * U_adv)
    sum_exp = np.sum(exp_terms)

    for i in range(len(defender_coverage)):

        direct_effect = Q[i] * (defender_payoffs[i] - defender_penaltys[i])

        indirect_effect = 0
        for j in range(len(defender_coverage)):
            dU_adv_j = adversary_penaltys[j] - adversary_payoffs[j] if j == i else 0
            dQ_j = (
                lamda
                * exp_terms[j]
                * (dU_adv_j * sum_exp - exp_terms[j] * dU_adv_j)
                / (sum_exp * sum_exp)
            )
            indirect_effect += dQ_j * (
                defender_coverage[j] * defender_payoffs[j]
                + (1 - defender_coverage[j]) * defender_penaltys[j]
            )

        gradient[i] = direct_effect + indirect_effect

    return gradient


def optimize_coverage(
    initial_defender_coverage,
    adversary_payoffs,
    adversary_penaltys,
    defender_payoffs,
    defender_penaltys,
    lamda,
    num_targets_to_protect,
    num_defender_resources,
    learning_rate=0.01,
    num_iterations=10000,

):
    """
    Optimize defender coverage using projected gradient descent
    """
    coverage = initial_defender_coverage.copy()
    history = []
    utility_history = []

    for i in range(num_iterations):

        gradient = compute_gradient(
            coverage,
            adversary_payoffs,
            adversary_penaltys,
            defender_payoffs,
            defender_penaltys,
            lamda,
            num_targets_to_protect
        )

        coverage = coverage + learning_rate * gradient

        coverage = project_to_simplex(coverage, s=num_defender_resources)

        history.append(coverage.copy())

        U_adv = expected_utility_adversary(
            coverage, adversary_payoffs, adversary_penaltys, num_targets_to_protect
        )
        Q = adversary_quantal_response(lamda, U_adv, num_targets_to_protect)
        U_def = expected_utility_defender(
            Q, defender_payoffs, defender_penaltys, coverage, num_targets_to_protect
        )
        utility_history.append(np.sum(U_def))

        if i > 0 and abs(utility_history[-1] - utility_history[-2]) < 1e-6:
            break

    return coverage, history, utility_history


def generate_random_feasible_point(num_targets_to_protect, num_resources):
    """
    Generate a random feasible starting point where:
    - Each element is between 0 and 1
    - Sum equals num_resources
    """
    x = np.random.random(num_targets_to_protect)
    return project_to_simplex(x, s=num_resources)


def find_local_minimum(
    x0,
    adversary_payoffs,
    adversary_penaltys,
    defender_payoffs,
    defender_penaltys,
    lamda,
    num_targets_to_protect,
    num_defender_resources,
    learning_rate=0.01,
    max_iterations=1000,
):
    """
    Find local minimum starting from x0 using gradient descent
    Returns:
    - opt_value: optimal value found
    - opt_point: optimal point found
    """

    coverage, history, utility_history = optimize_coverage(
        x0,
        adversary_payoffs,
        adversary_penaltys,
        defender_payoffs,
        defender_penaltys,
        lamda,
        num_targets_to_protect,
        num_defender_resources,
        learning_rate,
        max_iterations,
    )

    U_adv = expected_utility_adversary(coverage, adversary_payoffs, adversary_penaltys, num_targets_to_protect)
    Q = adversary_quantal_response(lamda, U_adv, num_targets_to_protect)
    U_def = expected_utility_defender(Q, defender_payoffs, defender_penaltys, coverage, num_targets_to_protect)
    final_utility_defender = np.sum(U_def)
    final_utility_adversary = np.sum(U_adv)

    return -final_utility_defender, final_utility_adversary, coverage


def brqr_multiple_starts(
    num_targets_to_protect,
    num_defender_resources,
    adversary_payoffs,
    adversary_penaltys,
    defender_payoffs,
    defender_penaltys,
    lamda,
    num_starts=10,
):
    """
    Implement BRQR algorithm with multiple starting points
    """
    opt_g = float("inf")
    x_opt = None

    all_starts = []
    all_results = []
    all_values = []
    adversary_utility = []

    for i in tqdm(range(num_starts)):

        x0 = generate_random_feasible_point(num_targets_to_protect, num_defender_resources)
        all_starts.append(x0)

        opt_i, final_utility_adversary, x_star = find_local_minimum(
            x0,
            adversary_payoffs,
            adversary_penaltys,
            defender_payoffs,
            defender_penaltys,
            lamda,
            num_targets_to_protect,
            num_defender_resources
        )

        all_results.append(x_star)
        all_values.append(opt_i)
        adversary_utility.append(final_utility_adversary)

        if opt_i < opt_g:
            opt_g = opt_i
            x_opt = x_star

    return -opt_g, x_opt, all_starts, all_results, all_values, adversary_utility


if __name__ == "__main__":

    num_targets_to_protect = 10
    num_defender_resources = 4
    num_starts = 25
    lamda = 1

    adversary_payoffs = [1, 4, 6, 1, 5, 7, 1, 4, 6, 1]  # payoff for each target
    adversary_penaltys = [-1, 0, -6, 0, -5, -8, 0, 0, -3, 0]  # penalty for each target

    defender_payoffs = [1, 4, 6, 1, 5, 7, 1, 4, 6, 1]  # payoff for each target
    defender_penaltys = [-1, 0, -6, 0, -5, -8, 0, 0, -3, 0]  # penalty for each target

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
