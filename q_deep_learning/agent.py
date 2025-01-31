import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.relu(x)


class Agent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        load: bool = False,
    ):
        self.env = env

        self.lr = learning_rate
        self.discount_factor = discount_factor

        s_size = env.observation_space.shape[0]
        self.training_error = []
        self.policy = Policy(s_size, 2, 32).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        if load:
            self.policy.load_state_dict(torch.load("./model", weights_only=True))

    def get_action(self, continious_obs: tuple[bool, int, int]):
        if np.random.random() < self.epsilon:
            return True, np.random.choice([-1, 1]), 0
        else:
            state = torch.from_numpy(continious_obs).float().unsqueeze(0).to(device)
            action_parameters = self.policy.forward(state).cpu()
            mu, sigma = action_parameters[:, :1], torch.exp(action_parameters[:, 1:])
            m = Normal(mu[:, 0], sigma[:, 0])

            action = m.sample()
            log_action = m.log_prob(action)

            return False, action.item(), log_action

    def update(self, policy_loss):
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.policy.state_dict(), "./model")

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
