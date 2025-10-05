import pandas as pd

df = pd.read_csv("guarding-policies/FPL-UE-v1/coverage_matrix_init/game_step_29/boundary_patch_reward_penalty_matrix.csv")

print("minimim penalty:", df["penalty"].min())
print("maximum penalty:", df["penalty"].max())

print("minimim reward:", df["reward"].min())
print("minimim reward:", df["reward"].max())