# Starter file
# env/network_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CyberEnv(gym.Env):
    """
    Multi-agent Cyber Defense Environment
    Agent 0: Attacker
    Agent 1: Defender
    """

    def __init__(self, num_nodes=6):
        super(CyberEnv, self).__init__()
        self.num_nodes = num_nodes

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_nodes)

        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        self.attacked_nodes = set()
        self.defended_nodes = set()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        self.attacked_nodes.clear()
        self.defended_nodes.clear()
        return self.state, {}

    def step(self, action):
        # ✅ Accept (agent_type, action) tuple
        if isinstance(action, tuple):
            agent_type, node = action
        else:
            raise ValueError("Action must be a tuple (agent_type, action)")

        reward = 0
        terminated = False
        truncated = False

        if agent_type == 0:
            # Attacker
            if node not in self.attacked_nodes:
                if node in self.defended_nodes:
                    reward = -1
                else:
                    self.state[node] = 1
                    self.attacked_nodes.add(node)
                    reward = 1
            else:
                reward = -0.5

        elif agent_type == 1:
            # Defender
            if node not in self.defended_nodes:
                self.defended_nodes.add(node)
                reward = 1
            else:
                reward = -0.2

        # Terminate if all nodes are covered
        if len(self.attacked_nodes.union(self.defended_nodes)) >= self.num_nodes:
            terminated = True

        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(f"State: {self.state}")
        print(f"Attacked Nodes: {self.attacked_nodes}")
        print(f"Defended Nodes: {self.defended_nodes}")
