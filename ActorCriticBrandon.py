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
from sklearn.preprocessing import scale
import collections
from scipy.stats import sem
import matplotlib.colors as mcolors

GAMMA = 0.99
ALPHAS = [1/4, 1/8, 1/16]
EPSILONS = [0.05, 0.15, 0.25]
EPISODES = 1000
MAX_STEPS = 10000
RUNS = 10