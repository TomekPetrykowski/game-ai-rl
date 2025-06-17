import numpy as np
from game.core_ai import ShootingGameEnv
import pygad
import matplotlib.pyplot as plt
from os import path


def fitness_func_old(instance, solution, solution_idx):
    env.reset()
    total_reward = 0
    for i in range(len(solution)):
        action = solution[i]
        _, reward, done = env.step(int(action))
        total_reward += reward

        if done:
            break

    return total_reward


def fitness_func(instance, solution, solution_idx):
    env.reset()
    total_reward = 0
    prev_state = env.get_state()
    for i in range(len(solution)):
        action = int(solution[i])
        state, reward, done = env.step(action)
        total_reward += reward

        # state = [player_x, move_dir, ally_dist, ally_x_rel]
        prev_ally_x_rel = prev_state[3] if len(prev_state) > 3 else 0
        curr_ally_x_rel = state[3] if len(state) > 3 else 0

        if abs(curr_ally_x_rel) < abs(prev_ally_x_rel):
            total_reward += 2

        prev_state = state

        if done:
            break

    return total_reward


def on_generation(ga):
    _, best_fitness, _ = ga.best_solution()
    print(f"Generation {ga.generations_completed} - Best Fitness: {best_fitness:.2f}")


sequence_length = 800
generations = 300
success = 100_000
num_trials = 3
generations_needed = []


for trial in range(num_trials):
    env = ShootingGameEnv(seed=13)

    ga_instance = pygad.GA(
        num_generations=generations,
        num_parents_mating=20,
        fitness_func=fitness_func,
        sol_per_pop=50,
        num_genes=sequence_length,
        gene_space=[1, 2],  # Actions: LEFT or RIGHT
        keep_parents=5,
        parent_selection_type="tournament",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=18,
        stop_criteria=[f"reach_{success}", "saturate_150"],
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
