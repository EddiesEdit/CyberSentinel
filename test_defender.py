
# test_defender.py

import os
import sys
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env.network_env import CyberEnv

# Create environment
env = CyberEnv(num_nodes=6)
obs, _ = env.reset()

print("🔄 Starting environment test...")

for step in range(12):  # Simulate 12 turns (6 per agent)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"🔹 Step {step + 1} - Agent: {'Attacker' if env.current_agent == 1 else 'Defender'}")
    print(f"   ➤ Action Taken: {action}")
    print(f"   ➤ Reward: {reward}")
    print(f"   ➤ State: {obs}")
    print(f"   ➤ Attacked: {env.attacked_nodes}")
    print(f"   ➤ Defended: {env.defended_nodes}")
    print("-" * 50)

    if terminated:
        print("✅ Episode terminated.")
        break
