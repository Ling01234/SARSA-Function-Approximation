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
        self.Qvalues = np.random.uniform(low=-0.001, high=0.001,
                                         shape=(self.num_bins, self.num_bins, self.num_bins, self.num_bins, self.num_action))
        self.bins = []
        for i in range(4):
            self.bins.append(np.linspace(
                self.lowerbound[i], self.upperbound[i], self.num_bins))

    # def init_start_state(self):
    #     """
    #     Generate starting observation to be between -0.001 and 0.001

    #     Returns:
    #         np array: starting state for a new episode with shape (4,)
    #     """
    #     (state, _) = self.env.reset()
    #     state /= 50
    #     return state

    def discritize_state(self, state):
        """
        Discritize continuous state into a discrete state

        Args:
            state (np array (4,)): current continuous state of agent

        Returns:
            state (4-tuple): current discritized state of agent
        """
        new_state = []

        for i in range(4):
            bin = np.digitize(state[i], self.bins[i]-1)
            new_state.append(bin)

        return tuple(new_state)

    def select_action(self, state, episode):
        if episode < 100:  # randomly explore in the first 100 episodes
            return np.choice(self.num_action)

        if episode > 850:  # lower epsilon after many episodes
            self.epsilon *= 0.95

        # epsilon greedy
        number = np.random.random()
        if number < self.epsilon:  # uniformly choose action
            return np.choice(self.num_action)

        # greedy selection
        discritized_state = self.discritize_state(state)
        best_states = np.where(self.Qvalues[discritized_state] == np.max(
            self.Qvalues[discritized_state]))[0]
        return np.random.choice(best_states)

    def simulate_episodes(self):
        for episode in range(1, self.num_episodes+1):
            # reset env
            (state, _) = self.env.reset()
            state = list(state)

            episode_reward = 0
            terminal = False
            while not terminal:
                discritized_state = self.discritize_state(state)
                action = self.select_action(state, episode)
                (next_state, reward, terminal, _, _) = self.env.step(action)
                episode_reward.append(reward)

                next_discritized_state = self.discritize_state(
                    list(next_state))
                q_max = np.max(self.Qvalues[next_discritized_state])

                self.qlearning_update(terminal, reward, action, state, q_max)
                state = next_state

            self.reward.append(episode_reward)

    def qlearning_update(self, terminal, reward, action, state, q_max):
        if not terminal:
            loss = reward + self.gamma * q_max - \
                self.Qvalues[state + (action,)]
            self.Qvalues[state + (action,)] += self.alpha * loss
        else:
            loss = reward - self.Qvalues[state + (action,)]
            self.Qvalues += self.alpha * loss
