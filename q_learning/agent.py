import numpy as np
from collections import defaultdict
import gymnasium as gym


class Agent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        initial_q_values: dict = dict(),
    ):
        print(env.action_space)
        print(env.observation_space)

        self.env = env
        self.q_values = defaultdict(
            lambda: np.zeros(env.action_space.n), initial_q_values
        )

        #
        self.distance_lin_space = np.linspace(
            env.observation_space.low[0], env.observation_space.high[0], 25
        )
        self.accel_lin_space = np.linspace(
            env.observation_space.low[1], env.observation_space.high[1], 25
        )

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, continious_obs: tuple[int, int, bool]):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            obs = self.get_index_in_linspace(continious_obs)
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        continious_obs: tuple[int, int],
        action: int,
        reward: float,
        terminated: bool,
        continious_next_obs: tuple[int, int],
    ):
        obs = self.get_index_in_linspace(continious_obs)
        next_obs = self.get_index_in_linspace(continious_next_obs)
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def get_index_in_linspace(self, obs: tuple[int, int, bool]):
        return (
            np.argmin(np.abs(self.distance_lin_space - obs[0])),
            np.argmin(np.abs(self.accel_lin_space - obs[1])),
        )
