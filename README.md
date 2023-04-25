# Reinforcement Learning with SARSA and Function Approximation
## Authors
- [Ling Fei Zhang](https://github.com/Ling01234)
- [Brandon Ma](https://github.com/brandon840)

This is a Reinforcement learning project on tabular RL and function approximation RL.

## Tabular RL
![Frozen Lake](images/frozen_lake.gif)

In tabular RL, we implemented and compared the performance of SARSA and expected SARSA on the [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) domain from the Gym environment. We analyzed the effect of temperature and learning rate on the performance of each model, and compared them against each other with the best hyperparameter setting.

## Function Approximation RL
![cart-pole](images/cart_pole.gif)

We implemented and compared the performance of Q-learning and actor-critic with linear function approximation on the [cart-pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) domain from the Gym environment. We analyzed the effect of $\epsilon$ and $\alpha$ on the performance of each model, and compared them against each other witht he best hyperparameter setting.

## Running the code
To run the code and see the results, simply run the [Python notebook](Sarsa_Qlearning_AC.ipynb).


