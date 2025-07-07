# Starter file
# agents/defender.py

import os
import sys

# ✅ Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ✅ Import CyberEnv
from env.network_env import CyberEnv

import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# ✅ Custom wrapper (optional): simulate attack risk
class DefenderCyberEnv(CyberEnv):
    def __init__(self, num_nodes=6):
        super().__init__(num_nodes)
        self.defended_nodes = set()

    def step(self, action):
        # Defender selects a node to defend
        reward = 0
        terminated = False
        truncated = False

        if action in self.attacked_nodes:
            # Failed to defend before attack
            reward = -1
        elif action not in self.defended_nodes:
            self.defended_nodes.add(action)
            reward = 1  # rewarded for successful proactive defense
        else:
            reward = -0.2  # discouraged from defending same node repeatedly

        if len(self.defended_nodes) == self.num_nodes:
            terminated = True

        return self.state, reward, terminated, truncated, {}

# ✅ Initialize environment and validate
env = DefenderCyberEnv(num_nodes=6)
check_env(env)

# ✅ Set up logs
log_dir = "./logs/defender/"
os.makedirs(log_dir, exist_ok=True)

# ✅ Build model
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=100,
    batch_size=32,
    gamma=0.95,
    verbose=1,
    tensorboard_log=log_dir,
    train_freq=4,
    target_update_interval=100
)

# ✅ Train and save
print("🛡️ Starting defender training...")
model.learn(total_timesteps=10_000)
model.save("defender_dqn_model")
print("✅ Defender model trained and saved successfully!")
