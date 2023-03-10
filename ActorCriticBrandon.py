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
BETA = 0.99
ALPHAS = [1/4, 1/8, 1/16]
EPSILONS = [0.05, 0.15, 0.25]
EPISODES = 1000
MAX_STEPS = 10000
RUNS = 10

class Agent:
    def __init__(self, env, alpha, beta, gamma, num_episodes, num_bins, seed) -> None:
        self.env = env
        self.alpha = alpha  # Learning rate for value function (Critic)
        self.beta = beta    # Lerning rate for policy (Actor)
        self.gamma = gamma
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

        # Initialize value of each state to 0
        self.V = np.zeros(shape=(num_bins, num_bins, num_bins, num_bins, self.num_action))

        # Initialize policy for each state to 0 (go left)
        self.policy = np.zeros(shape=(num_bins, num_bins, num_bins, num_bins, self.num_action), dtype = int) 

        # Weights (need to fix the size)
        self.w = np.random.uniform(low=-0.001, high=0.001,size=(10 * 4)) 
        
        self.bins = []
        for i in range(4):
            self.bins.append(np.linspace(
                self.lowerbound[i], self.upperbound[i], self.num_bins))

    def feature(self,state):
        state = list(state)
        one_hot = np.zeros(10 * 4)
        for i in range(4):
            index = np.maximum(np.digitize(state[i], self.bins[i]) - 1, 0)
            one_hot[10 * i + index] = 1  # one hot encoding

        return one_hot

    def discritize_state(self, state):
        """
        Discritize continuous state into a discrete state

        Args:
            state (list of length 4): Current continuous state of agent

        Returns:
            state (4-tuple): Current discritized state of agent
        """
        new_state = []
        for i in range(4):
            index = np.maximum(np.digitize(state[i], self.bins[i]) - 1, 0)
            new_state.append(index)

        return tuple(new_state)
    
    def run_one_episode(self, env, policy, render=False):

        (state,_) = env.reset()
        total_reward = 0
        d_states = []
        actions = []
        rewards = []
        probs = []
        done = False
        t = 0

        while not done:
            if render:
                env.render()
            d_state = self.feature(state)    # discretize the state
            d_states.append(d_state)                        # Check surroundings
            action, prob = policy.action(self.feature(state))         # Choose action based on surroundings
            actions.append(action)
            probs.append(prob)

            state, reward, done, _, _ = env.step(action)  # Execute action
            total_reward += reward
            rewards.append(reward)

            # Calculate delta
            #delta = reward + self.gamma * self.V[d_state + (action,)] - self.V[d_states[t] + (action,)]
            
            delta = reward + self.gamma * ((d_state)@self.w) - ((d_states[t])@self.w)


            # Update Value function
            #self.V[d_state] += self.alpha * delta * (self.V[self.discritize_state(state)] - self.V[d_state])
            self.w += self.alpha * delta * np.array((d_states[t]))
            #print(f"TEST: {self.w}")

            # Update policy
            policy.update(d_states[t], actions[t], delta)

            t += 1

        return total_reward, np.array(rewards), np.array(d_states), np.array(actions), np.array(probs)

    def train(self, theta, Policy, MAX_EPISODES=1000, seed=None, evaluate=False):
        #env = gym.make("CartPole-v1")
        # if seed is not None:
        #     env.seed(seed)
                    
        episode_rewards = []
        policy = Policy(theta, self.beta, self.gamma)

        for i in range(MAX_EPISODES):
                        
            # Run one episode
            total_reward, rewards, observations, actions, probs = self.run_one_episode(env, policy)

            # Keep track of the total reward for the episode
            episode_rewards.append(total_reward)

            print("EP: " + str(i) + " Score: " + str(total_reward) + " ")

        return episode_rewards, policy


class Critic:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.w = np.zeros(4)
    
    def value_fn(self, x):
        return x @ self.w
    
    def logistic(self, y):
        return 1/(1+np.exp(-y))
    
    def update(self,x , delta):
        self.w += self.alpha * delta * x
    


class LogisticPolicy:
    def __init__(self, theta, beta, gamma):
        self.theta = theta
        self.beta = beta
        self.gamma = gamma

    def logistic(self, y):
        return 1/(1+np.exp(-y))
    
    def probs(self, x):
        y = x @ self.theta

        prob_left = self.logistic(y)

        return np.array([prob_left, 1-prob_left])
    
    def action(self, x):
        probs = self.probs(x)
        action = np.random.choice([0,1], p = probs)

        return action, probs[action]
    
    def grad_log_p(self, x):
        y = x @ self.theta
        grad_log_pleft = np.array(x) - np.array(x)*self.logistic(y)
        grad_log_pright = - np.array(x)*self.logistic(y)

        return grad_log_pleft, grad_log_pright
    
    def update(self, observation, action, delta):

        # Calculate gradient for action at time t
        grad_log_p = self.grad_log_p(observation)[action]

        # Update policy
        self.theta += self.beta * delta * grad_log_p



GLOBAL_SEED = 0
np.random.seed(GLOBAL_SEED)
env = gym.make("CartPole-v1")

AC_agent = Agent(env, 0.25,0.25,0.95, EPISODES, 10, GLOBAL_SEED)

episode_rewards, policy = AC_agent.train(np.random.rand(40), LogisticPolicy,seed=GLOBAL_SEED)



plt.plot(episode_rewards)
plt.show()


