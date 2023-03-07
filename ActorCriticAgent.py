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


GAMMA = 0.99
ALPHA = 1/16
EPISODES = 1000
MAX_STEPS = 10000
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


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


def initialize():
    # Make environment
    env = gym.make('CartPole-v1')

    # Initialize network
    actor = Actor(
        env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    critic = Critic(env.observation_space.shape[0]).to(DEVICE)

    # Initialize optimizer
    actor_opt = opt.SGD(actor.parameters(), lr=0.001)
    critic_opt = opt.SGD(critic.parameters(), lr=0.001)

    return env, actor, actor_opt, critic, critic_opt


def train(env, actor, actor_opt, critic, critic_opt):
    # track scores
    rewards = []

    # track recent scores
    recent_reward = collections.deque(maxlen=100)

    # run episodes
    for episode in tqdm(range(EPISODES)):

        # init variables
        state, _ = env.reset()
        terminal = False
        episode_reward = 0
        I = 1

        # run episode, update online
        for step in range(MAX_STEPS):

            # get action and log probability
            action, lp = select_action(actor, state)

            # step with action
            new_state, reward, terminal, _, _ = env.step(action)

            # update episode score
            episode_reward += reward

            # get state value of current state
            state_tensor = torch.from_numpy(state)
            state_tensor = state_tensor.float().unsqueeze(0).to(DEVICE)
            state_val = critic(state_tensor)

            # get state value of next state
            new_state_tensor = torch.from_numpy(
                new_state).float().unsqueeze(0).to(DEVICE)
            new_state_val = critic(new_state_tensor)

            # if terminal state, next state val is 0
            if terminal:
                new_state_val = torch.tensor(
                    [0]).float().unsqueeze(0).to(DEVICE)

            # calculate value function loss with MSE
            val_loss = f.mse_loss(reward + GAMMA * new_state_val, state_val)
            val_loss *= I

            # calculate policy loss
            advantage = reward + GAMMA * new_state_val.item() - state_val.item()
            policy_loss = -lp * advantage
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

        # append episode episode_reward
        rewards.append(episode_reward)
        recent_reward.append(episode_reward)
        running_average = np.array(recent_reward).mean()

        # early stopping if we meet solved episode_reward goal
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
    plt.title('Actor Critic')
    plt.show()


def main():
    env, actor, actor_opt, critic, critic_opt = initialize()
    rewards = train(env, actor, actor_opt, critic, critic_opt)
    plot(rewards)


# main()
