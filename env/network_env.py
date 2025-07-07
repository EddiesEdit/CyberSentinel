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

        # Observation and action space (same for both agents)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_nodes)

        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        self.attacked_nodes = set()
        self.defended_nodes = set()

        self.current_agent = 0  # 0 = attacker, 1 = defender

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        self.attacked_nodes.clear()
        self.defended_nodes.clear()
        self.current_agent = 0  # Attacker starts
        return self.state, {}

    def step(self, action):
        action = int(action)
        reward = 0
        terminated = False
        truncated = False

        if self.current_agent == 0:
            # Attacker's turn
            if action not in self.attacked_nodes:
                if action in self.defended_nodes:
                    reward = -1  # defender blocked it
                else:
                    self.state[action] = 1  # attack success
                    self.attacked_nodes.add(action)
                    reward = 1
            else:
                reward = -0.5  # repeat attack

        else:
            # Defender's turn
            if action not in self.defended_nodes:
                self.defended_nodes.add(action)
                reward = 1  # block a new node
            else:
                reward = -0.2  # already defended

        # End episode if all nodes are attacked or defended
        if len(self.attacked_nodes.union(self.defended_nodes)) >= self.num_nodes:
            terminated = True

        # Alternate turn
        self.current_agent = 1 - self.current_agent

        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(f"State: {self.state}")
        print(f"Attacked Nodes: {self.attacked_nodes}")
        print(f"Defended Nodes: {self.defended_nodes}")
        print(f"Current Agent: {'Attacker' if self.current_agent == 0 else 'Defender'}")
