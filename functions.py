import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from scipy.special import softmax
import random
from tqdm import tqdm
import time
# from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Actions:
# 0: left
# 1: down
# 2: right
# 3: up


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
        if episode != 0 and episode % 11 == 0:
            action = int(self.learned_policy[state])
            return action

        action_values = self.Qvalues[state, :]
        preferences = action_values/self.temp
        preferences = softmax(preferences)
        action = np.random.choice(a=np.arange(
            self.action_num), p=preferences)

        return action

    def simulate_episodes(self, verbose=False):
        """
        This function simulates episodes in the frozen lake environment

        Args:
            verbose (bool, optional): set to True for some print statements. Defaults to False.

        Returns:
            np array: an array that contains the reward for each episode
        """
        total_reward = []
        for episode in tqdm(range(self.num_episodes)):
            # reset env
            (state, prob) = self.env.reset()
            action = self.select_action(state, episode)

            if verbose:
                print(f"Simulating episode {episode}.")

            averaged_reward = 0
            counter = 0
            terminal = False
            # move at most 100 times in a single episode
            # avoid to be stuck in infinite loop
            # if game can't terminate in 100 moves -> reward 0
            for _ in range(100):
                if terminal:
                    break

                (next_state, reward, terminal, _, _) = self.env.step(action)
                averaged_reward += reward

                # next action
                next_action = self.select_action(next_state, episode)

                if not terminal:
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
                counter += 1

            if episode % 10 == 0:  # update policy for each segment
                self.final_policy()

            averaged_reward /= counter
            total_reward.append(averaged_reward)

        self.final_policy()
        return total_reward

    def final_policy(self):
        """
        Obtain the best policy based on the episodes played.
        """
        for i in range(self.state_num):
            self.learned_policy[i] = np.random.choice(np.where(
                self.Qvalues[i] == np.max(self.Qvalues[i]))[0])  # may have more than 1 max


# TO DO
# PUT FUNCTION INSIDE CLASS
def visualize(learned_policy, num_games):
    """
    Visualize the game being played on pygame

    Args:
        learned_policy (np array): current best policy
        num_games (int): number of games to be played
    """
    for game in range(num_games):
        env = gym.make("FrozenLake-v1", desc=None,
                       map_name="4x4", is_slippery=False, render_mode="human")
        (state, prob) = env.reset()
        env.render()
        time.sleep(1)

        terminal = False
        while not terminal:
            if not terminal:
                (state, reward, terminal, _, _) = env.step(
                    int(learned_policy[state]))
                time.sleep(1)
            else:  # reached terminal state
                break
        time.sleep(0.5)
    env.close()
