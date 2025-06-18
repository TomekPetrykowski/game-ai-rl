from game.core_ai import ShootingGameEnv
from training.rl.model import Linear_QNet, DEVICE
import torch
import sys

filename = sys.argv[1] if sys.argv else "models/model_final.pth"

model = Linear_QNet(9, 512, 2).to(DEVICE)
model.load_state_dict(torch.load(filename, map_location=DEVICE))
model.eval()

env = ShootingGameEnv(render_mode=True)

state = env.get_state()
done = False
total_reward = 0
score = 0

print("Starting evaluation...")

while not done:
    state_tensor = torch.from_numpy(state).to(DEVICE)

    with torch.no_grad():
        prediction = model(state_tensor)
        action_idx = torch.argmax(prediction).item()

    # 0 -> 1 (LEFT), 1 -> 2 (RIGHT)
    game_action = 1 if action_idx == 0 else 2

    state, positioning_reward, game_score, done = env.step(game_action)
    total_reward += positioning_reward
    score = game_score

print(f"Evaluation finished!")
print(f"Final game score: {score}")
print(f"Total positioning reward: {total_reward:.2f}")

env.close()
