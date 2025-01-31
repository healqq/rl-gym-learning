import gymnasium as gym
from agent import Agent


env = gym.make("MountainCarContinuous-v0", render_mode="human")

n_episodes = 100
agent = Agent(
    env=env,
    learning_rate=0,
    load=True,
    initial_epsilon=0,
    epsilon_decay=0,
    final_epsilon=0,
)

for episode in range(n_episodes):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        is_exploration, action, log_probs = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step([action])

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    if episode % 100 == 0:
        print(episode)
