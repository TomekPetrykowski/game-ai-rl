from game.core import Game
from game.core_ai import ShootingGameEnv
import random
import numpy as np


if __name__ == "__main__":
    # For human playing uncomment the following 2 lines:
    # game = Game()
    # game.run()
    env = ShootingGameEnv(max_steps=10000, render_mode=True)
    done = False
    while not done:
        action = random.choice([0, 1, 2, 3])
        state, reward, done = env.step(action)

        print(reward, done)
        # np.set_printoptions(threshold=np.inf)
        print(state)

    env.close()
