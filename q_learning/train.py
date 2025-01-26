import gymnasium as gym
from q_learning.agent import Agent
from matplotlib import pyplot as plt
from collections import defaultdict
import seaborn as sns
import numpy as np
import pickle


def write_model(values):
    f = open("./q_values_ft", "b+w")
    # pickle can't save defaultdict
    pickle.dump(dict(values), f)


def get_index_in_linspace(self, obs: tuple[int, int, bool]):
    return (
        np.argmin(np.abs(self.distance_lin_space - obs[0])),
        np.argmin(np.abs(self.accel_lin_space - obs[1])),
    )


def create_grids(agent):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    distance, accel = np.meshgrid(
        # players count, dealers face-up card
        agent.distance_lin_space,
        agent.accel_lin_space,
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[get_index_in_linspace(agent, obs)],
        axis=2,
        arr=np.dstack([distance, accel]),
    )
    value_grid = distance, accel, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[get_index_in_linspace(agent, obs)],
        axis=2,
        arr=np.dstack([distance, accel]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    distance, accel, value = value_grid
    fig = plt.figure()

    # plot the state values
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.plot_surface(
        distance,
        accel,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )

    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(2, 3, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, cmap="Accent_r", cbar=False)
    ax2.set_title("Policy")
    ax2.set_xlabel("distance bucket")
    ax2.set_ylabel("accel bucket")

    # np.convolve will compute the rolling mean for 100 episodes
    bucket_size = int(n_episodes / 20)
    ax3 = fig.add_subplot(2, 3, 4)
    ax3.plot(np.convolve(env.return_queue, np.ones(bucket_size)))
    ax3.set_title("Episode Rewards")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Reward")

    ax4 = fig.add_subplot(2, 3, 5)
    ax4.plot(np.convolve(env.length_queue, np.ones(bucket_size)))
    ax4.set_title("Episode Lengths")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Length")

    ax5 = fig.add_subplot(2, 3, 6)
    ax5.plot(np.convolve(agent.training_error, np.ones(bucket_size)))
    ax5.set_title("Training Error")
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Temporal Difference")
    return fig


learning_rate = 0.01
n_episodes = 100_000
# n_episodes = 100
start_epsilon = 0.1
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.001

# env = gym.make("MountainCar-v0", render_mode="human")
env = gym.make("MountainCar-v0", render_mode="rgb_array")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
observation, info = env.reset()

f = open("./q_values", "b+r")
q_values = pickle.load(f)

agent = Agent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    initial_q_values=q_values,
)


for episode in range(n_episodes):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

    if episode % 100 == 0:
        print(episode)

write_model(agent.q_values)


# state values & policy with usable ace (ace counts as 11)
value_grid, policy_grid = create_grids(agent)
fig1 = create_plots(value_grid, policy_grid)
plt.tight_layout()
plt.show()
