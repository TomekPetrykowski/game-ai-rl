import numpy as np
from game.core_ai import ShootingGameEnv
import pygad
import matplotlib.pyplot as plt
from os import path


def fitness_func(instance, solution, solution_idx):
    env.reset()
    total_reward = 0
    for i in range(len(solution)):
        action = solution[i]
        _, reward, done = env.step(int(action))
        total_reward += reward

        if done:
            break

    return total_reward


def fitness_func_old(instance, solution, solution_idx):
    env.reset()
    total_reward = 0

    grid_size = 20

    def get_neighbors(x, y, size):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    neighbors.append((nx, ny))
        return neighbors

    for i in range(len(solution)):
        action = solution[i]
        state, reward, done = env.step(int(action))
        mid_reward = 0

        player_cells = np.argwhere(state[..., 0] > 0.5)
        bullet_cells = np.argwhere(state[..., 1] > 0.5)

        for px, py in player_cells:  # type: ignore
            for nx, ny in get_neighbors(px, py, grid_size):
                target_val = state[nx, ny, 2]
                if target_val > 0.5:
                    mid_reward -= 2
                elif target_val < -0.5:
                    mid_reward += 2

        for bx, by in bullet_cells:  # type: ignore
            for nx, ny in get_neighbors(bx, by, grid_size):
                target_val = state[nx, ny, 2]
                if target_val > 0.5:
                    mid_reward += 1
                elif target_val < -0.5:
                    mid_reward -= 1

        mid_reward += reward
        total_reward += mid_reward

        if done:
            break

    return total_reward


def on_generation(ga):
    _, best_fitness, _ = ga.best_solution()
    print(f"Generation {ga.generations_completed} - Best Fitness: {best_fitness:.2f}")


action_space_size = 4
sequence_length = 1000
generations = 150
success = 100_000
num_trials = 5
generations_needed = []


for trial in range(num_trials):
    env = ShootingGameEnv()

    ga_instance = pygad.GA(
        num_generations=generations,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=30,
        num_genes=sequence_length,
        gene_space=list(range(action_space_size)),
        keep_parents=4,
        parent_selection_type="tournament",
        crossover_type="single_point",
        mutation_type="random",
        stop_criteria=[f"reach_{success}", "saturate_60"],
        on_generation=on_generation,
    )

    ga_instance.run()
    solution, fitness, _ = ga_instance.best_solution()
    np.save(path.join("training", "pygad_sols", f"sol_{trial}.npy"), solution)
    print(f"Saved best solution with fitness: {fitness:.2f}")

    env.close()

    gens = ga_instance.generations_completed
    if fitness >= success:
        generations_needed.append(gens)
    else:
        generations_needed.append(generations + 1)

generations_needed = np.array(generations_needed)
successful = generations_needed <= generations

print("\nStatystyka (dla udanych prób):")
print(f"Średnia liczba pokoleń: {np.mean(generations_needed[successful]):.2f}")

plt.figure(figsize=(10, 5))
plt.bar(
    range(1, num_trials + 1),
    generations_needed,
    color=["green" if g < generations else "red" for g in generations_needed],
)
plt.axhline(
    float(np.mean(generations_needed[generations_needed < generations])),
    color="blue",
    linestyle="--",
    label="Średnia (sukcesy)",
)

plt.title("Liczba generacji do sukcesu w każdej próbie")
plt.xlabel("Numer próby")
plt.ylabel("Generacje do sukcesu")
plt.xticks(range(1, num_trials + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
