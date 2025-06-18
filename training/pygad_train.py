import numpy as np
from game.core_ai import ShootingGameEnv
import pygad
import matplotlib.pyplot as plt
from os import path
import json


def fitness_func_detailed(instance, solution, solution_idx):
    env.reset()
    total_positioning_reward = 0
    final_score = 0
    allies_catches = 0

    for i in range(len(solution)):
        action = int(solution[i])
        state, reward, score, done = env.step(action)

        total_positioning_reward += reward
        if reward > 0.4:
            allies_catches += 1

        final_score = score
        if done:
            break

    pos_fitness = total_positioning_reward
    eff_fitness = allies_catches

    total_fitness = pos_fitness + final_score * 3 + eff_fitness

    if not hasattr(instance, "detailed_stats"):
        instance.detailed_stats = []

    instance.detailed_stats.append(
        {
            "generation": getattr(instance, "generations_completed", 0),
            "total_fitness": total_fitness,
            "final_score": final_score,
            "allies_catches": allies_catches,
            "positioning_fitness": pos_fitness,
        }
    )

    return total_fitness


def on_generation_detailed(ga):
    _, best_fitness, __ = ga.best_solution()
    current_gen = ga.generations_completed

    if hasattr(ga, "detailed_stats"):
        current_gen_stats = [
            s for s in ga.detailed_stats if s["generation"] == current_gen
        ]

        if current_gen_stats:
            avg_fitness = np.mean([s["total_fitness"] for s in current_gen_stats])
            avg_positioning = np.mean(
                [s["positioning_fitness"] for s in current_gen_stats]
            )
            avg_score = np.mean([s["final_score"] for s in current_gen_stats])
            max_allies = max([s["allies_catches"] for s in current_gen_stats])

            print(
                f"Gen {current_gen:3d} | Best: {best_fitness:8.2f} | Avg: {avg_fitness:6.2f} | "
                f"Pos: {avg_positioning:6.2f} | AvgScore: {avg_score:6.1f} | "
                f"Catches: {max_allies}"
            )


def evaluate_solution(solution, env, num_evaluations=10):
    """Evaluate a solution multiple times for better statistics"""
    results = []

    for _ in range(num_evaluations):
        env.reset()
        total_positioning_reward = 0
        allies_catches = 0
        f_score = 0

        for action in solution:
            state, reward, score, done = env.step(int(action))
            total_positioning_reward += reward
            if reward > 0.4:
                allies_catches += 1

            f_score = score
            if done:
                break

        results.append(
            {
                "final_score": f_score,
                "positioning_reward": total_positioning_reward,
                "allies_catches": allies_catches,
            }
        )

    return results


# Main training parameters
sequence_length = 1000
generations = 70
success_threshold = 820.0  # Adjusted for new fitness function
num_trials = 5

all_trial_stats = []
generations_needed = []

for trial in range(num_trials):
    print(f"\n=== TRIAL {trial + 1}/{num_trials} ===")
    env = ShootingGameEnv(seed=7, true_seed=True)

    ga_instance = pygad.GA(
        num_generations=generations,
        num_parents_mating=100,
        fitness_func=fitness_func_detailed,
        sol_per_pop=200,
        mutation_by_replacement=True,
        mutation_percent_genes=20,  # type: ignore
        parent_selection_type="tournament",
        num_genes=sequence_length,
        keep_elitism=10,
        K_tournament=50,
        gene_space=[1, 2],
        stop_criteria=[f"reach_{success_threshold}"],
        on_generation=on_generation_detailed,
    )

    ga_instance.run()
    solution, fitness, _ = ga_instance.best_solution()

    evaluation_results = evaluate_solution(solution, env)

    np.save(path.join("training", "pygad_sols", f"sol_{trial}.npy"), solution)

    trial_stats = {
        "trial": trial,
        "best_fitness": fitness,
        "generations_completed": ga_instance.generations_completed,
        "evaluation_results": evaluation_results,
        "detailed_stats": getattr(ga_instance, "detailed_stats", []),
    }

    all_trial_stats.append(trial_stats)

    with open(
        path.join("training", "pygad_sols", f"trial_{trial}_stats.json"), "w"
    ) as f:
        json.dump(trial_stats, f, indent=2)

    print(f"\nTrial {trial + 1} Results:")
    print(f"Best Fitness: {fitness:.2f}")
    print(
        f"Avg Final Score: {np.mean([r['final_score'] for r in evaluation_results]):.1f}"
    )
    print(
        f"Avg Pos Reward: {np.mean([r['positioning_reward'] for r in evaluation_results]):.2f}"
    )

    env.close()

    gens = ga_instance.generations_completed
    generations_needed.append(gens if fitness >= success_threshold else generations + 1)

# Final Analysis and Plotting
generations_needed = np.array(generations_needed)
successful = generations_needed <= generations

print(f"\n=== FINAL STATISTICS ===")
print(f"Successful trials: {np.sum(successful)}/{num_trials}")
if np.any(successful):
    print(
        f"Average generations (successful): {np.mean(generations_needed[successful]):.2f}"
    )

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Generations needed
ax1.bar(
    range(1, num_trials + 1),
    generations_needed,
    color=["green" if g <= generations else "red" for g in generations_needed],
)
ax1.axhline(
    np.mean(generations_needed[successful]) if np.any(successful) else 0,
    color="blue",
    linestyle="--",
    label="Average (successful)",
)
ax1.set_title("Generations to Success")
ax1.set_xlabel("Trial")
ax1.set_ylabel("Generations")
ax1.legend()

# Plot 2: Best fitness evolution
for trial, stats in enumerate(all_trial_stats):
    if stats["detailed_stats"]:
        gen_data = {}
        for stat in stats["detailed_stats"]:
            gen = stat["generation"]
            if gen not in gen_data:
                gen_data[gen] = []
            gen_data[gen].append(stat["total_fitness"])

        generations = sorted(gen_data.keys())
        best_fitness_per_gen = [max(gen_data[gen]) for gen in generations]
        ax2.plot(generations, best_fitness_per_gen, label=f"Trial {trial+1}", alpha=0.7)

ax2.set_title("Best Fitness Evolution")
ax2.set_xlabel("Generation")
ax2.set_ylabel("Best Fitness")
ax2.legend()

# Plot 3: Final evaluation scores
final_scores = []
for stats in all_trial_stats:
    avg_score = np.mean([r["final_score"] for r in stats["evaluation_results"]])
    final_scores.append(avg_score)

ax3.bar(range(1, num_trials + 1), final_scores)
ax3.set_title("Average Final Game Scores")
ax3.set_xlabel("Trial")
ax3.set_ylabel("Average Score")

# Plot 4: Positioning rewards
positioning_rewards = []
for stats in all_trial_stats:
    avg_positioning = np.mean(
        [r["positioning_reward"] for r in stats["evaluation_results"]]
    )
    positioning_rewards.append(avg_positioning)

ax4.bar(range(1, num_trials + 1), positioning_rewards)
ax4.set_title("Average Positioning Rewards")
ax4.set_xlabel("Trial")
ax4.set_ylabel("Average Positioning Reward")

plt.tight_layout()
plt.savefig(path.join("training", "pygad_sols", "training_analysis.png"), dpi=300)
plt.show()

# Save comprehensive results
summary = {
    "total_trials": num_trials,
    "successful_trials": int(np.sum(successful)),
    "success_rate": float(np.sum(successful) / num_trials),
    "avg_generations_successful": (
        float(np.mean(generations_needed[successful])) if np.any(successful) else None
    ),
    "best_trial": int(np.argmax([s["best_fitness"] for s in all_trial_stats])),
    "best_fitness": float(max([s["best_fitness"] for s in all_trial_stats])),
}

with open(path.join("training", "pygad_sols", "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nTraining complete")
