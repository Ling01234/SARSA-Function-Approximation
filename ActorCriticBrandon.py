import numpy as np
import gymnasium as gym
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
import torch.nn.functional as f
import torch.optim as opt
from torch.distributions import Categorical
from sklearn.preprocessing import scale, OneHotEncoder
import collections
from scipy.stats import sem
import matplotlib.colors as mcolors
import random

GAMMA = 0.99
ALPHAS = [1/4, 1/8, 1/16]
EPSILONS = [0.05, 0.15, 0.25]
EPISODES = 1000
MAX_STEPS = 10000
RUNS = 10

class Agent:
    def __init__(self, env, alpha, gamma, epsilon, num_episodes, num_bins, seed) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.num_bins = num_bins
        self.lowerbound = env.observation_space.low
        self.lowerbound[1] = -3.5
        self.lowerbound[3] = -10
        self.upperbound = env.observation_space.high
        self.upperbound[1] = 3.5
        self.upperbound[3] = 10
        #self.env.seed(seed)
        self.seed = seed
        random.seed(self.seed)
        self.num_action = env.action_space.n
        self.reward = []
        self.state_values = np.random.uniform(low=-0.001, high=0.001,
                                         size=(num_bins, num_bins, num_bins, num_bins, self.num_action))
        self.policy = np.zeros(size=(num_bins, num_bins, num_bins, num_bins, self.num_action), dtype = int) # Initialize policy for each state to 0
        self.weights = np.random.uniform(low=-.001, high=0.001, size=(5))   # Weights (need to fix the size)
        self.bins = []
        for i in range(4):
            self.bins.append(np.linspace(
                self.lowerbound[i], self.upperbound[i], self.num_bins))

    def one_step_AC(self, max_steps):
        state = 0
        for t in range(max_steps):
            # Sample an action
            policy = self.policy[state]
            
            # Execute the action


env = gym.make("CartPole-v1")
(state, _) = env.reset()
agent = Agent(env, 1/4, GAMMA, 0.15, 1000, 2, seed=1)

print(agent.Qvalues)



