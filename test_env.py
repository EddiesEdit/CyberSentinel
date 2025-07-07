from env.network_env import CyberEnv

env = CyberEnv(num_nodes=6)
obs, _ = env.reset()

for _ in range(12):  # 6 attackers + 6 defenders
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        print("✅ Episode ended.")
        break
