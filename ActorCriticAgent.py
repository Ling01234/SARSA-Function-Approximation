import numpy as np
import gym
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
import torch.nn.functional as f
import torch.optim as opt
from torch.distributions import Categorical
from sklearn.preprocessing import scale
import collections
from scipy.stats import sem
import matplotlib.colors as mcolors
import keras_gym as kg
from keras import layers


GAMMA = 0.99
ALPHAS = [1/4, 1/8, 1/16]
EPSILONS = [0.05, 0.15, 0.25]
EPISODES = 1000
MAX_STEPS = 10000
RUNS = 10


class LinearApproximator(kg.FunctionApproximator):
    def body(self, x):
        return layers.Flatten()(x)


env = gym.make("CartPole-v1")

function_approximator = LinearApproximator(env, lr=0.001)
actor = kg.SoftmaxPolicy(function_approximator, update_strategy="vanilla")
critic = kg.V(function_approximator, gamma=GAMMA, bootstrap_n=1)
ac = kg.ActorCritic(actor, critic)


def train():
    for episode in tqdm(EPISODES):
        state, _ = env.reset()


def main():
    train()
