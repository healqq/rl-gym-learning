import gymnasium as gym
from agent import Agent
from matplotlib import pyplot as plt
import numpy as np

env = gym.make("MountainCarContinuous-v0")

n_episodes = 100
agent = Agent(
    env=env,
    load_filename="model",
    initial_epsilon=0,
    epsilon_decay=0,
    final_epsilon=0,
)

episode_rewards = []
for episode in range(n_episodes):
    obs, info = env.reset()
    done = False

    rewards = []
    # play one episode
    while not done:
        is_exploration, action, log_probs = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step([action])

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

        rewards.append(reward)

    episode_rewards.append(sum(rewards))
    if episode % 100 == 0:
        print(episode)

fig = plt.figure()

# np.convolve will compute the rolling mean for 100 episodes
bucket_size = int(n_episodes / 20)
ax3 = fig.add_subplot(1, 1, 1)
ax3.plot(np.convolve(episode_rewards, np.ones(bucket_size)))
ax3.set_title("Episode Rewards")
ax3.set_xlabel("Episode")
ax3.set_ylabel("Reward")

plt.show()
