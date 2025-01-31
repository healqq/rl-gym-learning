import gymnasium as gym
from agent import Agent
from collections import deque
import pickle
import numpy as np
import torch


def write_model(values):
    f = open("./q_values_ft", "b+w")
    # pickle can't save defaultdict
    pickle.dump(dict(values), f)


# def create_grids(agent):
#     """Create value and policy grid given an agent."""
#     # convert our state-action values to state values
#     # and build a policy dictionary that maps observations to actions
#     state_value = defaultdict(float)
#     policy = defaultdict(int)
#     for obs, action_values in agent.q_values.items():
#         state_value[obs] = float(np.max(action_values))
#         policy[obs] = int(np.argmax(action_values))

#     distance, accel = np.meshgrid(
#         # players count, dealers face-up card
#         agent.distance_lin_space,
#         agent.accel_lin_space,
#     )

#     # create the value grid for plotting
#     value = np.apply_along_axis(
#         lambda obs: state_value[get_index_in_linspace(agent, obs)],
#         axis=2,
#         arr=np.dstack([distance, accel]),
#     )
#     value_grid = distance, accel, value

#     # create the policy grid for plotting
#     policy_grid = np.apply_along_axis(
#         lambda obs: policy[get_index_in_linspace(agent, obs)],
#         axis=2,
#         arr=np.dstack([distance, accel]),
#     )
#     return value_grid, policy_grid


# def create_plots(value_grid, policy_grid):
#     """Creates a plot using a value and policy grid."""
#     # create a new figure with 2 subplots (left: state values, right: policy)
#     distance, accel, value = value_grid
#     fig = plt.figure()

#     # plot the state values
#     ax1 = fig.add_subplot(2, 3, 1, projection="3d")
#     ax1.plot_surface(
#         distance,
#         accel,
#         value,
#         rstride=1,
#         cstride=1,
#         cmap="viridis",
#         edgecolor="none",
#     )

#     ax1.view_init(20, 220)

#     # plot the policy
#     fig.add_subplot(2, 3, 2)
#     ax2 = sns.heatmap(policy_grid, linewidth=0, cmap="Accent_r", cbar=False)
#     ax2.set_title("Policy")
#     ax2.set_xlabel("distance bucket")
#     ax2.set_ylabel("accel bucket")

#     # np.convolve will compute the rolling mean for 100 episodes
#     bucket_size = int(n_episodes / 20)
#     ax3 = fig.add_subplot(2, 3, 4)
#     ax3.plot(np.convolve(env.return_queue, np.ones(bucket_size)))
#     ax3.set_title("Episode Rewards")
#     ax3.set_xlabel("Episode")
#     ax3.set_ylabel("Reward")

#     ax4 = fig.add_subplot(2, 3, 5)
#     ax4.plot(np.convolve(env.length_queue, np.ones(bucket_size)))
#     ax4.set_title("Episode Lengths")
#     ax4.set_xlabel("Episode")
#     ax4.set_ylabel("Length")

#     ax5 = fig.add_subplot(2, 3, 6)
#     ax5.plot(np.convolve(agent.training_error, np.ones(bucket_size)))
#     ax5.set_title("Training Error")
#     ax5.set_xlabel("Episode")
#     ax5.set_ylabel("Temporal Difference")
#     return fig


learning_rate = 0.01
# n_episodes = 1_000
n_episodes = 10_000
start_epsilon = 0.05
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.01
gamma = 0.99

# env = gym.make("MountainCar-v0", render_mode="human")
env = gym.make("MountainCarContinuous-v0")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
observation, info = env.reset()

# f = open("./q_values", "b+r")
# q_values = pickle.load(f)

agent = Agent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

scores_deque = deque(maxlen=10)
steps_deque = deque(maxlen=10)
scores = []
steps = []

max_x = -2
min_x = 2

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
                if abs(action) > 1:
                    reward = -10
                else:
                    if next_obs[0] > 0.2:
                        reward = -0.5
                    else:
                        if next_obs[0] > 0.3:
                            reward = -0.3
                        else:
                            reward = -1 + max(next_obs[0], 0) * 5
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

    # if n_steps < 999:
    #     print(returns)
    #     print(policy_loss)
    agent.update(policy_loss)
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

agent.save()
# evaluate
# for episode in range(n_eval_episodes):
#         state = env.reset()
#         step = 0
#         done = False
#         total_rewards_ep = 0

#         for step in range(max_steps):
#             action, _ = policy.act(state)
#             new_state, reward, done, info = env.step(action)
#             total_rewards_ep += reward

#             if done:
#                 break
#             state = new_state
# write_model(agent.q_values)


# # state values & policy with usable ace (ace counts as 11)
# value_grid, policy_grid = create_grids(agent)
# fig1 = create_plots(value_grid, policy_grid)
# plt.tight_layout()
# plt.show()
