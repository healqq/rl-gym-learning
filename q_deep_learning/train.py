import gymnasium as gym
from agent import Agent
from collections import deque
import numpy as np
import torch
import torch.optim as optim

learning_rate = 0.01
n_episodes = 1_000
n_epochs = 10
start_epsilon = 0.05
epsilon_decay = start_epsilon / (
    n_episodes * n_epochs / 2
)  # reduce the exploration over time
final_epsilon = 0.01
gamma = 0.99

env = gym.make("MountainCarContinuous-v0")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
observation, info = env.reset()

agent = Agent(
    env=env,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

optimizer = optim.Adam(agent.policy.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

scores_deque = deque(maxlen=10)
steps_deque = deque(maxlen=10)
scores = []
steps = []

max_x = -2
min_x = 2

for epoch in range(n_epochs):
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        rewards = []
        full_rewards = []
        saved_log_probs = []
        step = 0

        # play one episode
        while not done:
            is_exploration, action, log_probs = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step([action])
            # reward += abs(next_obs[0])
            max_x = max(max_x, next_obs[0])
            min_x = min(min_x, next_obs[0])
            if terminated:
                print(action, reward)
                reward = 1000
            else:
                if truncated:
                    reward = -1000
                else:
                    if next_obs[0] > 0.2:
                        reward = -0.5
                    else:
                        if next_obs[0] > 0.3:
                            reward = -0.3
                        else:
                            reward = -1 + max(next_obs[0], 0) * 5
            # penalty for big actions
            reward -= action**2
            full_rewards.append(reward)
            if not is_exploration:
                rewards.append(reward)
                saved_log_probs.append(log_probs)
            # # update the agent
            # agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs
            step += 1

        scores_deque.append(sum(full_rewards))
        scores.append(sum(full_rewards))

        steps_deque.append(step)
        steps.append(step)

        n_steps = len(rewards)
        returns = deque(maxlen=1000)

        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])

        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        agent.decay_epsilon()

        if episode % 10 == 0:
            print(
                "Episode {}\tAverage Score: {:.2f}\tCurrent epsilon {:.2f}\tCurrent avg steps {:.1f}\tMax.obs {:.1f}\tMin.obs {:.1f}".format(
                    episode,
                    np.mean(scores_deque),
                    agent.epsilon,
                    np.mean(steps_deque),
                    max_x,
                    min_x,
                )
            )

            max_x = -2
            min_x = 2

    scheduler.step()
    print("New lerning rate {}".format(scheduler.get_last_lr()))
    agent.save()
