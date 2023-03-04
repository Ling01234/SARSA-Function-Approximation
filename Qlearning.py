import numpy as np
import random


class Qlearning:
    def __init__(self, env, alpha, gamma, epsilon, num_episodes, num_bins,
                 lowerbound, upperbound, seed) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.num_bins = num_bins
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.seed = seed
        random.seed(seed)

        self.num_action = env.action_space.n
        self.reward = []
        self.Qvalues = np.random.uniform(low=-0.001, high=0.001,
                                         shape=(self.num_bins, self.num_bins, self.num_bins, self.num_bins, self.num_action))
