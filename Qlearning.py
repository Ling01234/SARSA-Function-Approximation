import numpy as np
import random

# Actions:
# 0: left
# 1: right

# params initialization:
ALPHAS = [1/4, 1/8, 1/16]
EPSILONS = [0.05, 0.15, 0.25]
BINS = 10
EPISODES = 1000
RUNS = 10


class Qlearning:
    def __init__(self, env, alpha, gamma, epsilon, num_episodes, num_bins, seed) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.num_bins = num_bins
        self.lowerbound = [-4.8, -4, -24, -10]
        self.upperbound = [4.8, 4, 24, 10]
        self.seed = seed
        random.seed(seed)

        self.num_action = env.action_space.n
        self.reward = []
        self.Qvalues = np.zeros(
            (self.num_bins, self.num_bins, self.num_bins, self.num_bins, self.num_action))

    def init_start_state(self):
        """
        Generate starting observation to be between -0.001 and 0.001

        Returns:
            np array: starting state for a new episode with shape (4,)
        """
        (state, _) = self.env.reset()
        state /= 50
        return state
