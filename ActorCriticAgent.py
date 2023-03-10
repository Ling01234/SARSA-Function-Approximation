import numpy as np
import gym
from tqdm import trange
from scipy.special import softmax
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.stats import sem


env = gym.make("CartPole-v1")
num_actions = env.action_space.n
lowerbounds = env.observation_space.low
lowerbounds[1] = -3.5
lowerbounds[3] = -10
upperbounds = env.observation_space.high
upperbounds[1] = 3.5
upperbounds[3] = 10
env.close()


GAMMA = 0.99
ALPHAS = [1/4, 1/8, 1/16]
EPSILONS = [0.05, 0.15, 0.25]
EPISODES = 1000
MAX_STEPS = 10000
RUNS = 10
NUM_BINS = 10

bins = []
for i in range(4):
    bins.append(np.linspace(lowerbounds[i], upperbounds[i], NUM_BINS))


def feature(state):
    state = list(state)
    one_hot = np.zeros(NUM_BINS * 4)
    for i in range(4):
        index = np.maximum(np.digitize(state[i], bins[i]) - 1, 0)
        one_hot[NUM_BINS * i + index] = 1  # one hot encoding

    return one_hot


class Actor():
    def __init__(self, env, alpha):
        self.env = env
        self.alpha = alpha
        self.weights = np.random.uniform(low=-0.001, high=0.001,
                                         size=(NUM_BINS * 4, num_actions))

    def policy(self, state):
        approx = np.dot(feature(state), self.weights)
        probs = softmax(approx)
        action = np.random.choice(num_actions, p=probs)
        return action, probs


class Critic():
    def __init__(self, env, alpha) -> None:
        self.env = env
        self.alpha = alpha
        self.weights = np.random.uniform(low=-0.001, high=0.001,
                                         size=(NUM_BINS * 4))

    def state_value(self, state):
        value = np.dot(feature(state), self.weights)
        return value


def train(actor, critic):
    env = gym.make("CartPole-v1")
    total_rewards = []
    for episode in range(1, EPISODES+1):
        episode_reward = 0
        state, _ = env.reset()
        terminal = False
        while not terminal:
            # choose action with actor policy
            action, probs = actor.policy(state)

            # step in the env
            next_state, reward, terminal, _, _ = env.step(action)
            episode_reward += reward

            # critic update
            loss = reward + GAMMA * \
                critic.state_value(next_state) - critic.state_value(state)
            critic.weights += critic.alpha * loss * feature(state)

            # policy update
            one_hot_action = np.zeros(num_actions)
            one_hot_action[action] = 1
            gradient_log = feature(
                state).reshape(-1, 1) - np.dot(feature(state).reshape(-1, 1), probs.reshape(1, -1))
            actor.weights += actor.alpha * loss * \
                gradient_log * one_hot_action.reshape(1, -1)

            state = next_state

        total_rewards.append(episode_reward)
    env.close()
    return total_rewards, actor, critic


def visualize(actor):
    env = gym.make("CartPole-v1", render_mode="human")
    state, _ = env.reset()
    terminal = False
    r = 0
    while not terminal:
        action, _ = actor.policy(state)
        state, reward, terminal, _, _ = env.step(action)
        r += reward
        env.render()

    env.close()
    print(f"Test reward: {r}")


def train_ac():
    best_alpha = 0
    best_actor = None
    env = gym.make("CartPole-v1")
    x = np.arange(1000)
    colors = [mcolors.TABLEAU_COLORS["tab:blue"],
              mcolors.TABLEAU_COLORS["tab:green"], mcolors.TABLEAU_COLORS["tab:orange"]]

    for index, alpha in enumerate(ALPHAS):
        average_reward = []
        for seed in trange(RUNS):
            env.reset()
            actor = Actor(env, alpha)
            critic = Critic(env, alpha)
            rewards, actor, critic = train(actor, critic)
            average_reward.append(rewards)

        average_reward = np.mean(average_reward, axis=0)
        max_reward = np.empty(1000)
        max_reward.fill(np.max(average_reward))
        err = sem(average_reward)
        plt.plot(x, average_reward,
                 label=f"alpha = {alpha}", color=colors[index])
        plt.plot(x, max_reward, color=colors[index],
                 linestyle="dashed", label=f"y = {int(max_reward[0])}")
        plt.fill_between(x, average_reward - err,
                         average_reward + err, color=colors[index], alpha=0.5)

        if best_alpha < max_reward[0]:
            best_alpha = alpha
            best_actor = actor

    plt.legend(bbox_to_anchor=(1, 0.5), loc="best")
    plt.title(f"Training Actor Critic")
    plt.yscale("log")
    plt.ylabel("Return")
    plt.xlabel("Episode")
    plt.show()

    return best_alpha, actor
