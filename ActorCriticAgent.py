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


class PolicyNetwork(nn.Module):

    # Takes in observations and outputs actions
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, action_space)

    # forward pass
    def forward(self, x):
        # input states
        x = self.input_layer(x)

        # relu activation
        x = f.relu(x)

        # actions
        actions = self.output_layer(x)

        # get softmax for a probability distribution
        action_probs = f.softmax(actions, dim=1)

        return action_probs


# Using a neural network to learn state value
class StateValueNetwork(nn.Module):

    #Takes in state
    def __init__(self, observation_space):
        super(StateValueNetwork, self).__init__()

        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        # input layer
        x = self.input_layer(x)

        # activiation relu
        x = f.relu(x)

        # get state value
        state_value = self.output_layer(x)

        return state_value


def select_action(network, state):
    state = torch.from_numpy(state)
    state = state.float().unsqueeze(0).to(DEVICE)

    # use network to predict action probabilities
    action_probs = network(state)
    state = state.detach()

    # sample an action using the probability distribution
    m = Categorical(action_probs)
    action = m.sample()

    # return action
    return action.item(), m.log_prob(action)


# Make environment
env = gym.make('CartPole-v1')

# Init network
policy_network = PolicyNetwork(
    env.observation_space.shape[0], env.action_space.n).to(DEVICE)
stateval_network = StateValueNetwork(env.observation_space.shape[0]).to(DEVICE)

# Init optimizer
policy_optimizer = opt.SGD(policy_network.parameters(), lr=0.001)
stateval_optimizer = opt.SGD(stateval_network.parameters(), lr=0.001)


def train():
    # track scores
    scores = []

    # track recent scores
    recent_scores = collections.deque(maxlen=100)

    # run episodes
    for episode in tqdm(range(EPISODES)):

        # init variables
        state, _ = env.reset()
        done = False
        score = 0
        I = 1

        # run episode, update online
        for step in range(MAX_STEPS):

            # get action and log probability
            action, lp = select_action(policy_network, state)

            # step with action
            new_state, reward, done, _, _ = env.step(action)

            # update episode score
            score += reward

            # get state value of current state
            state_tensor = torch.from_numpy(state)
            state_tensor = state_tensor.float().unsqueeze(0).to(DEVICE)
            state_val = stateval_network(state_tensor)

            # get state value of next state
            new_state_tensor = torch.from_numpy(
                new_state).float().unsqueeze(0).to(DEVICE)
            new_state_val = stateval_network(new_state_tensor)

            # if terminal state, next state val is 0
            if done:
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
            policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            policy_optimizer.step()

            # Backpropagate value
            stateval_optimizer.zero_grad()
            val_loss.backward()
            stateval_optimizer.step()

            if done:
                break

            # move into new state, discount I
            state = new_state
            I *= GAMMA

        # append episode score
        scores.append(score)
        recent_scores.append(score)
        running_average = np.array(recent_scores).mean()

        # early stopping if we meet solved score goal
        if running_average > env.spec.reward_threshold:
            print(
                f"Solved at episode {episode} with average score {running_average}")
            break
    return scores


def plot(rewards):
    sns.set()
    plt.plot(rewards)
    plt.ylabel('Return')
    plt.xlabel('Episodes')
    plt.title('Actor Critic')
    plt.show()


def main():
    # pass
    rewards = train()
    plot(rewards)
    # env, actor, actor_opt, critic, critic_opt = initialize()
    # rewards = train(env, actor, actor_opt, critic, critic_opt)
    # print(rewards)


main()
