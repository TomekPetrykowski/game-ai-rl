from game.core_ai import ShootingGameEnv
import random

if __name__ == "__main__":
    env = ShootingGameEnv(render_mode=False, max_steps=1000)
    rewards = []

    for _ in range(1000):
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = random.choice([0, 1, 2, 3])
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

        env.reset()
        rewards.append(total_reward)
    env.close()

    print("Average Reward:", sum(rewards) / len(rewards))
    print("Max Reward:", max(rewards))
    print("Min Reward:", min(rewards))
