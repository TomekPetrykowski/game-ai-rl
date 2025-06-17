import numpy as np
from game.core_ai import ShootingGameEnv
from training.rl.model import Linear_QNet, DEVICE
import torch

# Load model
model = Linear_QNet(4, 256, 2).to(DEVICE)
model.load_state_dict(torch.load("models/model_180.pth", map_location=DEVICE))
model.eval()

env = ShootingGameEnv(render_mode=True)
state = env.get_state()
done = False
total_reward = 0

while not done:
    state_tensor = torch.tensor(np.array(state), dtype=torch.float).to(DEVICE)
    with torch.no_grad():
        action = torch.argmax(model(state_tensor)).item()
    state, reward, done = env.step(action)
    total_reward += reward

print("Total reward:", total_reward)
env.close()
