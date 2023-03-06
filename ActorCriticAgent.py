import numpy as np
import gym
from tqdm import trange
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt
from torch.distributions import Categorical
from sklearn.preprocessing import scale
import statistics
import collections


class ActorCritic(nn.Module):
    def __init__(self, env, hidden_units, gamma, alpha, num_episode):
        super(ActorCritic, self).__init__()
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.num_episode = num_episode
        self.num_action = self.env.action_space.n
        self.hidden_units = hidden_units
        self.layer1 = nn.Linear(4, hidden_units)
        self.actor = nn.Linear(hidden_units, self.num_action)
        self.critic = nn.Linear(hidden_units, 1)
        self.saved_actions = []
        self.reward = []
        self.optimizer = opt.Adam(self.parameters(), lr=alpha)

    def forward(self, x):
        x = f.relu(self.layer1(x))
        probs = f.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return probs, state_values

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, value = self(state)
        actions = Categorical(probs)
        action = actions.sample()  # returns tensor object
        self.saved_actions.append((actions.log_prob(action), value))
        return action.item()  # get item from tensor

    def backprop(self):
        total_reward = 0
        policy_loss, value_loss = [], []
        returns = []

        for reward in self.reward[::-1]:
            total_reward = reward + self.gamma * total_reward
            returns.insert(0, total_reward)

        returns = scale(returns)
        returns = torch.tensor(returns)

        for (log_prob, state_value), total_reward in zip(self.saved_actions, returns):
            # print("in for loop")
            adv = total_reward - state_value.item()
            policy_loss.append(-log_prob * adv)
            value_loss.append(f.smooth_l1_loss(
                state_value, torch.tensor([total_reward])))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        self.optimizer.step()

        del self.reward[:]
        del self.saved_actions[:]

    def simulate_episodes(self, verbose=False):
        # won = False
        running_reward = 0
        episodes_reward = collections.deque(maxlen=100)
        for episode in range(1, self.num_episode + 1):
            (state, _) = self.env.reset()
            episode_reward = 0
            terminal = False

            while not terminal:

                action = self.select_action(state)
                (next_state, reward, terminal, _, _) = self.env.step(action)
                self.reward.append(reward)
                episode_reward += reward
                state = next_state

                # if episode_reward >= 500:  # agent wins
                #     won = True
                #     break

            self.backprop()

            if verbose and episode % 10 == 0:
                print(f"Episode {episode}, reward {int(episode_reward)}")

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            if running_reward > self.env.spec.reward_threshold and episode > 150:
                print(
                    f"Solved at episode {episode}, average reward: {running_reward:.2f}")
                break
