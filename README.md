# Shooting Game with Reinforcement Learning

This project is a classic shooting game implemented in Python using Pygame, enhanced with AI agents trained via a Genetic Algorithm (GA) and a Q-network (deep reinforcement learning). The goal is to control a player, shoot targets, and maximize the score while avoiding penalties. The project supports both manual play and AI research.

---

## Features

- **Playable Game:** Control a player, shoot bullets, and interact with two types of targets (opponents and allies).
- **AI Training Environment:** Custom RL environment for training agents.
- **Genetic Algorithm Training:** Uses [PyGAD](https://pygad.readthedocs.io/) to evolve action sequences.
- **Q-Network Agent:** Implemented with PyTorch.
- **Visualization:** Training progress and evaluation results visualized with Matplotlib.
- **Customizable:** Easily modify game rules, rewards, and environment.

---

## Project Structure

```
project/
│
├── game/
│   ├── core.py            # Main game loop (manual play)
│   ├── core_ai.py         # AI environment (ShootingGameEnv)
│   ├── entities/          # Player, Bullet, Target classes
│   ├── settings.py        # Game settings and constants
│   ├── types.py           # Enums for actions and target types
│   └── ...
│
├── training/
│   ├── pygad_train.py     # Genetic Algorithm training script
│   ├── pygad_test.py      # Test/evaluate GA solutions
│   ├── pygad_sols/        # Saved GA solutions (.npy, .png)
│   └── rl/
│       ├── agent.py       # Q-network agent training script
│       ├── model.py       # Q-network model and trainer
│       ├── eval.py        # Q-network evaluation script
│
├── main.py                # Launches the manual game
├── test.py                # Random agent test script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

- Requires Python 3.8+ - 3.11+.
- For Q-network training, also install: `torch`, `matplotlib`, `pygad` (if not already installed).

### 2. Play the Game Manually

```bash
python main.py
```

- Use arrow keys or A/D to move, SPACE to shoot.

### 3. Train with Genetic Algorithm

```bash
python training/pygad_train.py
```

- Trains action sequences using PyGAD and saves the best solutions in `training/pygad_sols/`.
- Progress and statistics are visualized and saved as `.png` files in the same folder.

### 4. Evaluate GA Solutions

```bash
python training/pygad_test.py
```

- Loads and evaluates saved solutions, printing their rewards.

### 5. Train Q-Network Agent

```bash
python training/rl/agent.py
```

- Trains a Q-network agent using deep reinforcement learning.
- Training progress (scores and mean scores) is visualized using Matplotlib.
- Trained models are saved in the `models/` directory (e.g., `model_XXX.pth`).

### 6. Evaluate Q-Network Agent

```bash
python training/rl/eval.py
```

- Loads a trained Q-network model and runs it in the environment with rendering enabled.
- Prints the total reward achieved by the agent.

---

## Game Rules & Rewards

- **Player:** Moves left/right, shoots bullets (not used in traninig).
- **Targets:** Opponents (red, not used in training) and allies (blue) spawn at the top and move down.
- **Rewards:**
  - Colliding with ally: +30
  - Letting ally pass: -20

---

## AI Environment

- **Observation:** [player_x, move_direction, ally_dist, ally_x_relative_distance].
- **Actions:** 1 = LEFT, 2 = RIGHT.
- **Episode ends:** On win/loss or after max steps.

---

## Customization

- Change game parameters in [`game/settings.py`](game/settings.py).
- Modify reward structure or environment in [`game/core_ai.py`](game/core_ai.py).

---

### Genetic Algorithm (GA)

- The GA evolves fixed-length action sequences (e.g., 800 steps).
- Each solution is evaluated by running the sequence in the environment and summing the rewards.
- The best solutions are saved as `.npy` files in `training/pygad_sols/`.
- You can test these solutions using `pygad_test.py`, which will print the total reward for each.

#### Understanding the GA Plots

- The `.png` files in `training/pygad_sols/` (e.g., `res.png`) visualize the number of generations required to reach a successful solution in each trial.
- **Green bars** indicate trials where the GA found a solution within the allowed number of generations.
- **Red bars** indicate trials where the GA did not reach the success threshold.
- The dashed blue line shows the average number of generations needed for successful trials.
- These plots help you assess the efficiency and reliability of the GA for your environment.

### Q-Network Agent

- The Q-network agent is implemented in [`training/rl/agent.py`](training/rl/agent.py) and [`training/rl/model.py`](training/rl/model.py).
- Training progress is plotted and saved (if it would work).
- Trained models are saved in the `models/` directory.
- You can evaluate a trained model using [`training/rl/eval.py`](training/rl/eval.py), which will display the agent's performance in the environment. (does not work either for now)
