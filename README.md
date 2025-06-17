# Shooting Game with Reinforcement Learning

This project is a simple shooting game implemented in Python using Pygame, with AI agents trained using machine learning techniques (Genetic Algorithm and Deep Q-Network). The goal is to control a player, shoot targets, and maximize the score while avoiding penalties.

---

## Features

- **Playable Game:** Classic shooting game with player, bullets, and two types of targets (opponents and allies).
- **AI Environment:** Custom environment for training AI agents using RL algorithms.
- **Genetic Algorithm Training:** Uses [PyGAD](https://pygad.readthedocs.io/) to evolve action sequences.
- **DQN Training:** Deep Q-Network agent implemented with PyTorch.
- **Visualization:** Training progress and evaluation results are visualized with Matplotlib.

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
│   ├── dqn_train.py       # DQN agent training script
│   ├── pygad_train.py     # Genetic Algorithm training script
│   ├── pygad_test.py      # Test/evaluate GA solutions
│   └── pygad_sols/        # Saved GA solutions (.npy)
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

- Requires Python 3.8+.
- For DQN training, also install: `torch`, `matplotlib`, `pygad` (if not already installed).

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

### 4. Evaluate GA Solutions

```bash
python training/pygad_test.py
```

- Loads and evaluates saved solutions, printing their rewards.

### 5. Train with DQN

```bash
python training/dqn_train.py
```

- Trains a DQN agent and saves the model as `dqn_model.pth`.

---

## Game Rules & Rewards

- **Player:** Moves left/right, shoots bullets.
- **Targets:** Opponents (red) and allies (blue) spawn at the top and move down.
- **Rewards:**
  - Shooting opponent: +30
  - Shooting ally: -10
  - Colliding with opponent: -5
  - Colliding with ally: +30
  - Letting opponent pass: -5
  - Letting ally pass: -10

---

## AI Environment

- **Observation:** 20x20x3 grid (player, bullets, targets).
- **Actions:** 0 = NONE, 1 = LEFT, 2 = RIGHT, 3 = SHOOT.
- **Episode ends:** On win/loss or after max steps.

---

## Customization

- Change game parameters in [`game/settings.py`](game/settings.py).
- Modify reward structure or environment in [`game/core_ai.py`](game/core_ai.py).

---

## Credits

- Developed for an AI/ML course project.
- Uses [PyGAD](https://pygad.readthedocs.io/) and [PyTorch](https://pytorch.org/).

---
