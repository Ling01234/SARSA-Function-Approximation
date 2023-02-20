import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from scipy.special import softmax
# from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class SARSA:
    def __init__(self, env, alpha, temp, gamma, num_episodes, expected):
        self.env = env
        self.alpha = alpha
        self.temp = temp
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.expected = expected
        self.state_num = env.observation_space.n
        self.action_num = env.action_space.n
        self.learned_policy = np.zeros(self.state_num)  # learned policy
        # action value matrix
        self.Qvalues = np.zeros((self.state_num, self.action_num))

    def select_action(self, state, episode):  # NEED TO IMPLEMENT
        """
        This function selects an action given a state in the game.
        The exploration is done using softmax (Boltmann).

        Args:
            state (int): current state in the game
            episode (int): current episode in the run
        """
        if episode % 11 == 0:
            pass  # SELECT GREEDILY - TO IMPLEMENT
        action_values = self.Qvalues[state, :]
        preferences = action_values/self.temp
        preferences = softmax(preferences)

    def simulate_episodes(self):
        """
        This function simulates episodes in the frozen lake environment
        """
        for episode in range(self.num_episodes):
            # reset env
            (state, prob) = self.env.reset()
            action = self.select_action(state, episode)

            print(f"Simulating episode {episode}.")

            # while loop until terminal
            terminal = False
            while not terminal:
                (next_state, reward, terminal_state, _, _) = self.env.step(action)

                # next action
                next_action = self.select_action(next_state, episode)

                if not terminal_state:
                    loss = reward + self.gamma * \
                        self.Qvalues[next_state, next_action] - \
                        self.Qvalues[state, action]
                    self.Qvalues[state, action] = self.Qvalues[state,
                                                               action] + self.alpha * loss
                else:  # terminal state
                    loss = reward - self.Qvalues[state, action]
                    self.Qvalues[state, action] = self.Qvalues[state,
                                                               action] + self.alpha * loss

                state = next_state
                action = next_action

                if episode % 10 == 0:  # update policy for each segment
                    self.final_policy()

    def final_policy(self):
        """
        Obtain the final policy based on the episodes.
        """
        for i in range(self.state_num):
            self.learned_policy[i] = np.random.choice(np.where(
                self.Qvalues[i] == np.max(self.Qvalues[i]))[0])  # may have more than 1 max
