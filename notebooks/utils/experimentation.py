import itertools

import numpy as np
import pandas as pd

from abstractions import Agent, World
from exploring_agents.training import train_agent


def print_summary(results):
    average_rewards_l10 = {key: (np.mean(results[key][0][-10:])) for key in results}
    average_rewards = {key: (np.mean(results[key][0])) for key in results}

    print("Average reward on rounds")
    for k, v in sorted(average_rewards.items(), key=lambda item: item[1], reverse=True):
        print(f"{k}: {v:.2f}")

    print("\nAverage reward on last 10 episodes")
    for k, v in sorted(
        average_rewards_l10.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"{k}: {v:.2f}")


# these are some useful functions for experimentation
def get_results(
    base_agent: type[Agent],
    world: [World],
    base_arguments: dict[str, any],
    extra_parameters: dict[str, any],
    training_rounds: int,
    episodes: int,
) -> tuple[list[list[float]], list[list[int]]]:
    returns = []
    lengths = []

    for _ in range(training_rounds):

        agent = base_agent(**base_arguments, **extra_parameters)

        episode_lengths, episode_returns = train_agent(
            agent=agent, world=world, episodes=episodes
        )
        returns.append(episode_returns)
        lengths.append(episode_lengths)

    return returns, lengths


def get_exp_results(
    base_agent: type[Agent],
    world: World,
    base_arguments: dict[str, any],
    arguments: dict[str, any],
    episodes: int,
    training_rounds: int,
) -> list[tuple[dict, tuple[list[list[float]], list[list[int]]]]]:
    blown_arguments = [
        {key: values[idx] for idx, key in enumerate(arguments)}
        for values in itertools.product(*[arguments[x] for x in arguments])
    ]
    return [
        (
            cur_args,
            get_results(
                base_agent, world, base_arguments, cur_args, training_rounds, episodes
            ),
        )
        for cur_args in blown_arguments
    ]


def get_summary_df(results, final_episodes=10) -> pd.DataFrame:
    params_keys = list(results[0][0].keys())
    data = [
        [params[key] for key in params_keys]
        + [
            np.mean(np.sum(np.array(values[0]), axis=1)),
            np.std(np.sum(np.array(values[0]), axis=1)),
            np.average(np.array(values[0])[:, -final_episodes:]),
            np.std(np.array(values[0])[:, -final_episodes:]),
            np.min(np.array(values[1])),
        ]
        for (params, values) in results
    ]

    return pd.DataFrame(
        data,
        columns=params_keys
        + [
            "average_round_reward",
            "std_round_reward",
            "average_reward_final_10",
            "std_reward_final_10",
            "shortest_run",
        ],
    ).sort_values(by=["average_round_reward"], ascending=False)


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n
