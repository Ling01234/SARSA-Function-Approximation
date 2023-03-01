import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from scipy.special import softmax
import random
from tqdm import tqdm
from scipy.stats import sem
import time
# from gymnasium.envs.toy_text.frozen_lake import generate_random_map


# params initialization
ALPHAS = [0.01, 0.1, 0.5]
GAMMA = 0.95
TEMPERATURE = [0.05, 0.1, 50]
EPISODES = 5500
SEEDS = np.arange(10)
EPSILON = 0.2

# Actions:
# 0: left
# 1: down
# 2: right
# 3: up


class SARSA:
    def __init__(self, env, alpha, temp, gamma, num_episodes, expected, epsilon):
        self.env = env
        self.alpha = alpha
        self.temp = temp
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.expected = expected
        self.epsilon = epsilon
        self.state_num = env.observation_space.n
        self.action_num = env.action_space.n
        self.learned_policy = np.zeros(self.state_num)  # learned policy
        # action value matrix
        self.Qvalues = np.zeros((self.state_num, self.action_num))
        self.reward = []

    def select_action(self, state, episode):
        """
        This function selects an action given a state in the game.
        The exploration is done using softmax (Boltmann).

        Args:
            state (int): current state in the game
            episode (int): current episode in the run
        """
        action_values = self.Qvalues[state, :]

        if episode % 11 == 0:
            # randomized best action
            action = np.random.choice(
                np.where(action_values == np.max(action_values))[0])
            return action

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
        for episode in range(1, self.num_episodes+1):
            # reset env
            (state, _) = self.env.reset()
            action = self.select_action(state, episode)

            if verbose:
                print(f"Simulating episode {episode}.")

            episode_reward = 0
            terminal = False

            # move at most 100 times in a single episode
            # avoid to be stuck in infinite loop
            # if game can't terminate in 100 moves -> reward 0
            for s in range(100):
                if terminal:
                    break

                (next_state, reward, terminal, _, _) = self.env.step(action)
                episode_reward += reward * (self.gamma ** s)

                # next action
                next_action = self.select_action(next_state, episode)

                if episode % 11 != 0:  # does not update Qvalues on testing episode
                    if not self.expected:  # sarsa
                        self.sarsa_update(
                            state, next_state, action, next_action, reward, terminal)

                    else:  # expected sarsa
                        self.expected_update(state, next_state, action, reward)

                state = next_state
                action = next_action

            if episode % 10 == 0:  # update policy for each segment
                self.final_policy()
            self.reward.append(episode_reward)

        self.final_policy()

    def sarsa_update(self, state, next_state, action, next_action, reward, terminal):
        """
        Update rule for SARSA.

        Args:
            state (int): current state
            next_state (int): next state
            action (int): current action
            next_action (int): next action
            reward (int): reward
            terminal (bool): True if agent is at terminal state, False otherwise
        """
        if not terminal:
            loss = reward + self.gamma * \
                self.Qvalues[next_state, next_action] - \
                self.Qvalues[state, action]
            self.Qvalues[state, action] += self.alpha * loss
        else:  # terminal state
            loss = reward - self.Qvalues[state, action]
            self.Qvalues[state, action] += self.alpha * loss

    def expected_update(self, state, next_state, action, reward):
        """
        Update rule for Expected SARSA.

        Args:
            state (int): current state
            next_state (int): next state
            action (int): current action
            reward (int): reward
        """
        expected = 0
        action_values = self.Qvalues[next_state, :]
        preferences = action_values/self.temp
        preferences = softmax(preferences)

        for a in range(self.action_num):
            expected += preferences[a] * self.Qvalues[next_state][a]

        loss = reward + self.gamma * expected - \
            self.Qvalues[state, action]
        self.Qvalues[state, action] += self.alpha * loss

    def final_policy(self):
        """
        Calculate the best policy based on the episodes played.
        """
        for i in range(self.state_num):
            self.learned_policy[i] = np.random.choice(np.where(
                self.Qvalues[i] == np.max(self.Qvalues[i]))[0])  # may have more than 1 max

    def visualize(self, num_games, slippery=True):
        """
        Visualize the game being played on pygame

        Args:
            num_games (int): number of games to be played
            slippery (bool): True if env is slippery, False otherwise
        """
        for _ in range(num_games):
            env = gym.make("FrozenLake-v1", desc=None,
                           map_name="4x4", is_slippery=slippery, render_mode="human")
            (state, prob) = env.reset()
            env.render()
            time.sleep(1)

            terminal = False
        for i in range(100):
            if not terminal:
                (state, reward, terminal, _, _) = env.step(
                    int(self.learned_policy[state]))
                time.sleep(1)
            else:
                break
            time.sleep(0.5)
        env.close()

    def train_reward(self):
        """
        Obtain the last 10 training episode rewards

        Returns:
            np array: reward of last 10 training episodes
        """
        return sum(self.reward[-11:-1])/10

    def test_reward(self):
        """
        Obtain the last testing episode reward

        Returns:
            int: last testing episode reward
        """
        return self.reward[-1]


def training_sarsa():
    """
    Function to simulate sarsa in training.
    The function plots the alpha values against return by agent.
    The function illustrates the effect of 3 different temperatures.
    """
    for temp in tqdm(TEMPERATURE):
        rewards_train = []
        for alpha in ALPHAS:
            average_reward_train = 0
            for seed in SEEDS:
                random.seed(seed)

                env = gym.make("FrozenLake-v1", desc=None,
                               map_name="4x4", is_slippery=True)
                env.reset()
                sarsa = SARSA(env, alpha, temp, GAMMA,
                              EPISODES, False, EPSILON)
                sarsa.simulate_episodes()
                # final_policy = sarsa.learned_policy
                train_reward = sarsa.train_reward()
                average_reward_train += train_reward

            average_reward_train /= 10
            rewards_train.append(average_reward_train)

        plt.plot(ALPHAS, rewards_train, label=f"temperature = {temp}")

    plt.legend(bbox_to_anchor=(1, 0.5), loc="best")
    plt.title("Training on SARSA")
    plt.xlabel("alpha")
    plt.ylabel("Return")
    plt.show()


def testing_sarsa():
    """
    Function to simulate sarsa in testing.
    The function plots the alpha values against return by agent.
    The function illustrates the effect of 3 different temperatures.
    """
    for temp in tqdm(TEMPERATURE):
        rewards_test = []
        for alpha in ALPHAS:
            average_reward_test = 0
            for seed in SEEDS:
                random.seed(seed)

                env = gym.make("FrozenLake-v1", desc=None,
                               map_name="4x4", is_slippery=True)
                env.reset()
                sarsa = SARSA(env, alpha, temp, GAMMA,
                              EPISODES, False, EPSILON)
                sarsa.simulate_episodes()
                # final_policy = sarsa.learned_policy
                test_reward = sarsa.test_reward()
                average_reward_test += test_reward

            average_reward_test = average_reward_test/10
            rewards_test.append(average_reward_test)

        plt.plot(ALPHAS, rewards_test, label=f"temperature = {temp}")

    plt.legend(bbox_to_anchor=(1, 0.5), loc="best")
    plt.title("Testing on SARSA")
    plt.xlabel("alpha")
    plt.ylabel("Return")
    plt.show()


def best_params_sarsa(alpha, temp):
    """
    Plot the learning curve based on the best params on sarsa.

    Args:
        alpha (float): alpha value
        temp (float): temperature value
    """
    train_reward = []
    for seed in tqdm(SEEDS):
        random.seed(seed)
        env = gym.make("FrozenLake-v1", desc=None,
                       map_name="4x4", is_slippery=True)
        env.reset()
        sarsa = SARSA(env, alpha, temp, GAMMA, EPISODES,
                      False, EPSILON)  # best params chosen
        sarsa.simulate_episodes()
        reward = sarsa.reward
        train_reward.append(reward)

    train_reward = np.array(train_reward)
    train_reward_mean = np.mean(train_reward, axis=0)
    sd = np.std(train_reward_mean)
    x = np.arange(5500)

    plt.plot(x, train_reward_mean)
    plt.fill_between(x, 0, train_reward_mean + sd, alpha=0.35, color="g")
    plt.title("Return of Agent over the Course of Training for SARSA")
    plt.xlabel("Episode averaged over 10 runs")
    plt.ylabel("Return")
    plt.xlim(0, 6000)
    plt.show()


def training_esarsa():
    """
    Function to simulate expected sarsa in training.
    The function plots the alpha values against return by agent.
    The function illustrates the effect of 3 different temperatures.
    """
    for temp in tqdm(TEMPERATURE):
        rewards_train = []
        for alpha in ALPHAS:
            average_reward_train = 0
            for seed in SEEDS:
                random.seed(seed)

                env = gym.make("FrozenLake-v1", desc=None,
                               map_name="4x4", is_slippery=True)
                env.reset()
                sarsa = SARSA(env, alpha, temp, GAMMA, EPISODES, True, EPSILON)
                sarsa.simulate_episodes()
                # final_policy = sarsa.learned_policy
                train_reward = sarsa.train_reward()
                average_reward_train += train_reward

            average_reward_train = average_reward_train/10
            rewards_train.append(average_reward_train)

        plt.plot(ALPHAS, rewards_train, label=f"temperature = {temp}")

    plt.legend(bbox_to_anchor=(1, 0.5), loc="best")
    plt.title("Training on Expected SARSA")
    plt.xlabel("alpha")
    plt.ylabel("Return")
    plt.show()


def testing_esarsa():
    """
    Function to simulate expected sarsa in testing.
    The function plots the alpha values against return by agent.
    The function illustrates the effect of 3 different temperatures.
    """
    for temp in tqdm(TEMPERATURE):
        rewards_test = []
        for alpha in ALPHAS:
            average_reward_test = 0
            for seed in SEEDS:
                random.seed(seed)

                env = gym.make("FrozenLake-v1", desc=None,
                               map_name="4x4", is_slippery=True)
                env.reset()
                sarsa = SARSA(env, alpha, temp, GAMMA, EPISODES, True, EPSILON)
                sarsa.simulate_episodes()
                # final_policy = sarsa.learned_policy
                test_reward = sarsa.test_reward()
                average_reward_test += test_reward

            average_reward_test = average_reward_test/10
            rewards_test.append(average_reward_test)

        plt.plot(ALPHAS, rewards_test, label=f"temperature = {temp}")

    plt.legend(bbox_to_anchor=(1, 0.5), loc="best")
    plt.title("Testing on Expected SARSA")
    plt.xlabel("alpha")
    plt.ylabel("Return")
    plt.show()


def best_params_esarsa(alpha, temp):
    """
    Plot the learning curve based on the best params on expected sarsa.

    Args:
        alpha (float): alpha value
        temp (float): temperature value
    """
    train_reward = []
    for seed in tqdm(SEEDS):
        random.seed(seed)
        env = gym.make("FrozenLake-v1", desc=None,
                       map_name="4x4", is_slippery=True)
        env.reset()
        sarsa = SARSA(env, alpha, temp, GAMMA, EPISODES,
                      True, EPSILON)  # best params chosen
        sarsa.simulate_episodes()
        reward = sarsa.reward
        train_reward.append(reward)

    train_reward = np.array(train_reward)
    train_reward_mean = np.mean(train_reward, axis=0)
    sd = np.std(train_reward_mean)
    x = np.arange(5500)

    plt.plot(x, train_reward_mean)
    plt.fill_between(x, 0, train_reward_mean + sd, alpha=0.35, color="g")
    plt.title("Return of Agent over the Course of Training for Expected SARSA")
    plt.xlabel("Episode averaged over 10 runs")
    plt.ylabel("Return")
    plt.xlim(0, 6000)
    plt.show()
