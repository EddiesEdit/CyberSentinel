# env/network_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CyberEnv(gym.Env):
    """
    A Multi-Agent Cyber Defense Simulation Environment.

    Agents:
        - Agent 0: Attacker
        - Agent 1: Defender

    State:
        A vector of node statuses (0 = safe, 1 = attacked).

    Action:
        Each agent selects a node to attack or defend.

    Termination:
        - All nodes are either attacked or defended
        - OR max_steps reached (if set)
    """

    def __init__(self, num_nodes=6, max_steps=100):
        super(CyberEnv, self).__init__()
        self.num_nodes = num_nodes
        self.max_steps = max_steps

        # Define observation and action spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_nodes)

        # Initialize environment state
        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        self.attacked_nodes = set()
        self.defended_nodes = set()
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        self.attacked_nodes.clear()
        self.defended_nodes.clear()
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        if not isinstance(action, tuple):
            raise ValueError("Action must be a tuple: (agent_type, node_index)")

        agent_type, node = action
        reward = 0
        terminated = False
        truncated = False

        if agent_type == 0:
            # Attacker logic
            if node not in self.attacked_nodes:
                if node in self.defended_nodes:
                    reward = -1  # Attacked a defended node
                else:
                    self.state[node] = 1  # Mark node as attacked
                    self.attacked_nodes.add(node)
                    reward = 1
            else:
                reward = -0.5  # Redundant attack

        elif agent_type == 1:
            # Defender logic
            if node not in self.defended_nodes:
                self.defended_nodes.add(node)
                reward = 1
            else:
                reward = -0.2  # Redundant defense

        else:
            raise ValueError("Invalid agent_type. Must be 0 (Attacker) or 1 (Defender).")

        # Update step counter
        self.step_count += 1

        # Termination logic
        if len(self.attacked_nodes.union(self.defended_nodes)) >= self.num_nodes:
            terminated = True
        elif self.step_count >= self.max_steps:
            truncated = True

        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(f"State: {self.state}")
        print(f"Step: {self.step_count}")
        print(f"Attacked Nodes: {self.attacked_nodes}")
        print(f"Defended Nodes: {self.defended_nodes}")
