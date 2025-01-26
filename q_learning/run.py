import pickle
import numpy as np
from collections import defaultdict
import gymnasium as gym
from q_learning.agent import Agent


env = gym.make("MountainCar-v0", render_mode="human")

n_episodes = 100
q_values = defaultdict(lambda: np.zeros(env.action_space.n))
f = open("./q_values_ft", "b+r")
q_values = pickle.load(f)

agent = Agent(
    env=env,
    learning_rate=0,
    initial_epsilon=0,
    epsilon_decay=0,
    final_epsilon=0,
    initial_q_values=q_values,
)

for episode in range(n_episodes):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    if episode % 100 == 0:
        print(episode)
