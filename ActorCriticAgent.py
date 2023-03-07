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

GAMMA = 0.99
ALPHAS = [1/4, 1/8, 1/16]
EPSILONS = [0.05, 0.15, 0.25]
EPISODES = 1000
MAX_STEPS = 10000
RUNS = 10
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
# DEVICE = "cpu"


class Actor(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(obs_space, 128)
        self.output = nn.Linear(128, action_space)

    def forward(self, x):
        x = self.layer1(x)
        x = f.relu(x)
        actions = self.output(x)
        action_probs = f.softmax(actions, dim=1)
        return action_probs


class Critic(nn.Module):
    def __init__(self, obs_space):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(obs_space, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = f.relu(x)
        state_value = self.output(x)
        return state_value


def select_action(network, state):
    state = torch.from_numpy(state)
    state = state.float().unsqueeze(0).to(DEVICE)

    action_probs = network(state)
    state = state.detach()

    actions = Categorical(action_probs)
    action = actions.sample()

    return action.item(), actions.log_prob(action)


def initialize(alpha):
    env = gym.make('CartPole-v1')
    env.reset()

    # Initialize network
    actor = Actor(
        env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    critic = Critic(env.observation_space.shape[0]).to(DEVICE)

    # Initialize optimizer
    actor_opt = opt.SGD(actor.parameters(), lr=alpha)
    critic_opt = opt.SGD(critic.parameters(), lr=alpha)

    return env, actor, actor_opt, critic, critic_opt


def train(env, actor, actor_opt, critic, critic_opt):
    rewards = []
    recent_reward = collections.deque(maxlen=100)
    for episode in range(EPISODES):
        state, _ = env.reset()
        terminal = False
        episode_reward = 0
        I = 1

        for step in range(MAX_STEPS):
            action, lp = select_action(actor, state)
            new_state, reward, terminal, _, _ = env.step(action)
            episode_reward += reward

            # get critical value of current state
            state_tensor = torch.from_numpy(state)
            state_tensor = state_tensor.float().unsqueeze(0).to(DEVICE)
            state_val = critic(state_tensor)

            # get critical value of next state
            new_state_tensor = torch.from_numpy(
                new_state).float().unsqueeze(0).to(DEVICE)
            new_state_val = critic(new_state_tensor)

            # if terminal state
            if terminal:
                new_state_val = torch.tensor(
                    [0]).float().unsqueeze(0).to(DEVICE)

            # calculate value function loss with MSE
            val_loss = f.mse_loss(reward + GAMMA * new_state_val, state_val)
            print(f"value loss: {val_loss}")
            val_loss *= I

            # calculate policy loss
            advantage = reward + GAMMA * new_state_val.item() - state_val.item()
            print(f"advatange: {advantage}")
            policy_loss = -lp * advantage
            print(f"policy loss: {policy_loss}")
            policy_loss *= I

            # Backpropagate policy
            actor.zero_grad()
            policy_loss.backward(retain_graph=True)
            actor_opt.step()

            # Backpropagate value
            critic_opt.zero_grad()
            val_loss.backward()
            critic_opt.step()

            if terminal:
                break

            # move into new state, discount I
            state = new_state
            I *= GAMMA

        # append episode_reward
        rewards.append(episode_reward)
        recent_reward.append(episode_reward)
        running_average = np.array(recent_reward).mean()

        # break if agent wins the game
        if running_average > env.spec.reward_threshold:
            print(
                f"Solved at episode {episode} with average score {running_average}")
            break
    return rewards


def plot(rewards):
    sns.set()
    plt.plot(rewards)
    plt.ylabel('Return')
    plt.xlabel('Episodes')
    plt.yscale("log")
    plt.title('Actor Critic')
    plt.show()


def test_run():
    env, actor, actor_opt, critic, critic_opt = initialize(alpha=ALPHAS[2])
    rewards = train(env, actor, actor_opt, critic, critic_opt)
    plot(rewards)


def train_ac():
    x = np.arange(1000)
    colors = [mcolors.TABLEAU_COLORS["tab:blue"],
              mcolors.TABLEAU_COLORS["tab:green"], mcolors.TABLEAU_COLORS["tab:orange"]]
    index = 0
    for alpha in tqdm(ALPHAS):
        average_reward = []
        for seed in range(RUNS):
            print(f"run {seed}")
            env, actor, actor_opt, critic, critic_opt = initialize(alpha)
            rewards = train(env, actor, actor_opt, critic, critic_opt)
            average_reward.append(rewards)

        average_reward = np.mean(average_reward, axis=0)
        max_reward = np.empty(1000)
        max_reward.fill(np.max(average_reward))
        err = sem(average_reward)
        plt.plot(x, average_reward,
                 label=f"alpha = {alpha}", color=colors[index])
        plt.plot(
            x, max_reward, color=colors[index], linestyle="dashed", label=f"y = {int(max_reward[0])}")
        plt.fill_between(
            x, average_reward - err, average_reward + err, color=colors[index], alpha=0.5)

        index += 1

    plt.legend(bbox_to_anchor=(1, 0.5), loc="best")
    plt.title(f"Training Actor Critic")
    plt.ylabel("Return")
    plt.yscale("log")
    plt.xlabel("Episodes")
    plt.show()
