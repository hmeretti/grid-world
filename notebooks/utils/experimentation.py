import itertools

import numpy as np

from grid_world.utils.dacaying_functions import get_linear_decay


# these are some useful functions for experimentation
def get_results(
    base_agent, base_arguments, extra_parameters, training_rounds, episodes
):

    returns = []
    lengths = []

    for _ in range(training_rounds):

        agent = base_agent(**base_arguments, **extra_parameters)

        episode_lengths, episode_returns = agent.train(episodes=episodes)
        returns.append(episode_returns)
        lengths.append(episode_lengths)

    average_returns = np.mean(np.array(returns), axis=0)
    average_lengths = np.mean(np.array(lengths), axis=0)
    return average_returns, average_lengths


def get_exp_results(base_agent, base_arguments, arguments, episodes, training_rounds):
    blown_arguments = [
        {key: values[idx] for idx, key in enumerate(arguments)}
        for values in itertools.product(*[arguments[x] for x in arguments])
    ]
    return {
        f"{cur_args}": get_results(
            base_agent, base_arguments, cur_args, training_rounds, episodes
        )
        for cur_args in blown_arguments
    }


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n
