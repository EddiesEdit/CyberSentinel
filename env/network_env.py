# Starter file
# env/network_env.py


import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CyberEnv(gym.Env):
    def __init__(self, num_nodes=6):
        super(CyberEnv, self).__init__()
        self.num_nodes = num_nodes
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_nodes)
        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        self.attacked_nodes = set()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        self.attacked_nodes = set()
        return self.state, {}


    def step(self, action):
        action = int(action)
        reward = 0
        terminated = False
        truncated = False

        if action not in self.attacked_nodes:
            self.state[action] = 1  # mark node as attacked
            self.attacked_nodes.add(action)
            reward = 1
        else:
            reward = -0.5  # discourage re-attacking

        if len(self.attacked_nodes) == self.num_nodes:
            terminated = True  # all nodes attacked

        return self.state, reward, terminated, truncated, {}
    def render(self, mode="human"):
        print(f"Current network state: {self.state}")
