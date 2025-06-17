import numpy as np
from game.core_ai import ShootingGameEnv
import os


def evaluate_solution(solution):
    total_reward = 0
    env = ShootingGameEnv(seed=13, render_mode=True)
    for action in solution:
        _, reward, done = env.step(action)
        total_reward += reward
        if done:
            break
    env.close()
    return total_reward


for i in range(3):
    solution = np.load(os.path.join("training", "pygad_sols", f"sol_{i}.npy"))
    # solution = np.load(os.path.join("training", "pygad_sols", f"sol_{i}_old.npy"))
    reward = evaluate_solution(solution)
    print(f"Reward: {reward}")
