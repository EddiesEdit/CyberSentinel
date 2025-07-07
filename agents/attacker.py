# agents/attacker.py

import os
import sys

# ✅ Add parent directory (project root) to sys.path so we can import `env.network_env`
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ✅ Now it's safe to import
from env.network_env import CyberEnv

import gym
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# ✅ Create and check the environment
env = CyberEnv(num_nodes=6)
check_env(env)

# ✅ Create log directory
log_dir = "./logs/attacker/"
os.makedirs(log_dir, exist_ok=True)

# ✅ Initialize DQN model
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

# ✅ Train the model
print("🚀 Starting attacker training...")
model.learn(total_timesteps=10_000)

# ✅ Save the model
model.save("attacker_dqn_model")

print("✅ Attacker model trained and saved successfully!")
