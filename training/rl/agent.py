import torch
import random
import numpy as np
from collections import deque
from game.core_ai import ShootingGameEnv
from training.rl.model import Linear_QNet, QTrainer, DEVICE
from helper import plot
import time

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
EPISODES = 10


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(4, 256, 2).to(DEVICE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        action = torch.tensor([action], dtype=torch.long).to(DEVICE)
        state = torch.tensor(np.array(state), dtype=torch.float).to(DEVICE)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(DEVICE)
        reward = torch.tensor([reward], dtype=torch.float).to(DEVICE)
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = (EPISODES // 2) - self.n_games
        final_move = None
        if random.randint(0, EPISODES // 2 + (EPISODES // 10)) < self.epsilon:
            move = random.randint(1, 2)
            final_move = move
        else:
            state0 = torch.tensor(np.array(state), dtype=torch.float).to(DEVICE)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move = move

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    env = ShootingGameEnv(render_mode=False, max_steps=800)
    env.speed = 1
    episode = 0
    while episode < EPISODES:
        # get old state
        state_old = env.get_state()

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        state, reward, done = env.step(final_move)
        state_new = state

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            env.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if reward > record:
                record = reward
                agent.model.save(file_name=f"model_{record}.pth")

            print("Game", agent.n_games, "Score", reward, "Record:", record)

            plot_scores.append(reward)
            total_score += reward
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            episode += 1
    return plot_scores, plot_mean_scores


if __name__ == "__main__":
    plot_scores, plot_mean_scores = train()
    print("Training finished.")
    plot(plot_scores, plot_mean_scores)
    # time.sleep(10)
