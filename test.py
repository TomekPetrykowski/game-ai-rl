from game.core_ai import ShootingGameEnv
import random

if __name__ == "__main__":
    env = ShootingGameEnv(seed=42, render_mode=False, max_steps=1000, true_seed=True)
    rewards = []
    env.speed = 5

    for _ in range(1000):
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = random.choice([1, 2])
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

        env.reset()
        rewards.append(total_reward)
    env.close()

    print("Average Reward:", sum(rewards) / len(rewards))
    print("Max Reward:", max(rewards))
    print("Min Reward:", min(rewards))

# When actions are just LEFT or RIGHT and there is no OPPONENTS. Env seeded
# Average Reward: 8751.73
# Max Reward: 88240
# Min Reward: -54880

# When actions are just LEFT or RIGHT and there is no OPPONENTS. Env NOT seeded
# Average Reward: 15277.26
# Max Reward: 104080
# Min Reward: -54880
