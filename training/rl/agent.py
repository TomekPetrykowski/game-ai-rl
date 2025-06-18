import torch
import random
import numpy as np
from collections import deque
from game.core_ai import ShootingGameEnv
from training.rl.model import Linear_QNet, QTrainer, DEVICE
import matplotlib.pyplot as plt

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.005
EPISODES = 500  # (recommended or even higher)


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.8
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(9, 512, 2).to(DEVICE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        action_idx = 0 if action == 1 else 1  # 1=LEFT->0, 2=RIGHT->1

        action_np = np.array([action_idx], dtype=np.int64)
        reward_np = np.array([reward], dtype=np.float32)

        self.trainer.train_step(state, action_np, reward_np, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        action_indices = [0 if action == 1 else 1 for action in actions]

        states_np = np.array(states, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        actions_np = np.array(action_indices, dtype=np.int64)
        rewards_np = np.array(rewards, dtype=np.float32)

        self.trainer.train_step(
            states_np, actions_np, rewards_np, next_states_np, dones
        )

    def get_action(self, state):
        if random.random() < self.epsilon:
            move_idx = random.randint(0, 1)
        else:
            state_tensor = torch.from_numpy(state).to(DEVICE)
            with torch.no_grad():
                prediction = self.model(state_tensor)
                move_idx = torch.argmax(prediction).item()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return 1 if move_idx == 0 else 2


def calculate_win_reward(final_score, positioning_reward, steps_taken, max_steps):

    reward = positioning_reward

    # Large bonus for winning
    if final_score >= 300:
        reward += 50.0
        efficiency_bonus = max(0, (max_steps - steps_taken) / max_steps * 20)
        reward += efficiency_bonus
    elif final_score <= -500:  # loosing game
        reward -= 25.0
    else:
        # small bonus for positive scores
        reward += max(0, final_score * 0.1)

    return reward


def train():
    plot_scores = []
    plot_mean_scores = []
    plot_positioning_rewards = []
    plot_win_rate = []

    total_score = 0
    max_steps = 1500
    total_positioning = 0
    wins = 0
    recent_scores = deque(maxlen=100)

    agent = Agent()
    env = ShootingGameEnv(render_mode=False, max_steps=max_steps)

    episode = 0
    while episode < EPISODES:
        env.reset()
        state_old = env.get_state()
        episode_positioning_reward = 0
        steps_taken = 0

        while not env.done:
            final_move = agent.get_action(state_old)
            state_new, positioning_reward, game_score, done = env.step(final_move)

            episode_positioning_reward += positioning_reward
            steps_taken += 1

            if done:
                total_reward = calculate_win_reward(
                    game_score, positioning_reward, steps_taken, max_steps
                )
            else:
                total_reward = positioning_reward

            agent.train_short_memory(
                state_old, final_move, total_reward, state_new, done
            )
            agent.remember(state_old, final_move, total_reward, state_new, done)

            state_old = state_new

            if done:
                break

        # finishing episode
        agent.n_games += 1
        agent.train_long_memory()
        final_score = env.score
        recent_scores.append(final_score)

        if final_score >= 300:
            wins += 1

        recent_wins = sum(1 for score in recent_scores if score >= 300)
        win_rate = recent_wins / len(recent_scores) if recent_scores else 0

        if episode % (EPISODES // 20) == 0:
            avg_recent_score = np.mean(recent_scores) if recent_scores else 0
            print(f"Episode {episode + 1}/{EPISODES}")
            print(f"  Score: {final_score}")
            print(f"  Positioning Reward: {episode_positioning_reward:.2f}")
            print(f"  Win Rate (last 100): {win_rate:.2%}")
            print(f"  Avg Score (last 100): {avg_recent_score:.1f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Total Wins: {wins}")

        plot_scores.append(final_score)
        plot_positioning_rewards.append(episode_positioning_reward)
        plot_win_rate.append(win_rate)

        total_score += final_score
        total_positioning += episode_positioning_reward

        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)

        episode += 1

        if episode % (EPISODES // 10) == 0:
            agent.model.save(file_name=f"model_checkpoint_{episode}.pth")

    agent.model.save(file_name="model_final.pth")

    print(f"\nTraining Complete!")
    print(f"Total Wins: {wins}/{EPISODES} ({wins/EPISODES:.2%})")

    return plot_scores, plot_mean_scores, plot_positioning_rewards, plot_win_rate


if __name__ == "__main__":
    plot_scores, plot_mean_scores, plot_positioning_rewards, plot_win_rate = train()
    print("Training finished.")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Scores
    ax1.plot(plot_scores, alpha=0.6, label="Episode Score")
    ax1.plot(plot_mean_scores, label="Mean Score")
    ax1.axhline(y=300, color="g", linestyle="--", label="Win Threshold")
    ax1.set_title("Game Scores")
    ax1.legend()

    # Win rate
    ax2.plot(plot_win_rate)
    ax2.set_title("Win Rate (last 100 games)")
    ax2.set_ylabel("Win Rate")

    # Positioning rewards
    ax3.plot(plot_positioning_rewards, alpha=0.6)
    ax3.set_title("Positioning Rewards")

    # Score distribution (last 500 episodes)
    recent_scores = plot_scores[-500:] if len(plot_scores) >= 500 else plot_scores
    ax4.hist(recent_scores, bins=30, alpha=0.7)
    ax4.axvline(x=300, color="g", linestyle="--", label="Win Threshold")
    ax4.set_title("Score Distribution (Recent)")
    ax4.legend()

    plt.tight_layout()
    plt.show()
