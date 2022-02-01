from typing import Collection

from grid_world.action import Action
from grid_world.state import State


def returns_from_reward(rewards: Collection[float], gamma: float = 1) -> list[float]:
    cum_reward = 0
    cum_rewards_list = []
    for r in reversed(rewards):
        cum_reward = r + gamma * cum_reward
        cum_rewards_list = [cum_reward] + cum_rewards_list

    return cum_rewards_list


def first_visit_return(
    episode_states: list[State],
    episode_actions: list[Action],
    episode_returns: list[float],
) -> dict[tuple[State, Action], float]:
    fvr = {}
    for i in range(len(episode_returns)):
        if (sa := (episode_states[i], episode_actions[i])) not in fvr:
            fvr[sa] = episode_returns[i]

    return fvr
