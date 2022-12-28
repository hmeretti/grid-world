import sys
import os

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/" + os.path.join(*path.split("/")[:-3]))

from exploring_agents.training import run_episode
from grid_world.visualization.curses_utils import animate_episodes
from notebooks.utils.basics import basic_reward, basic_actions
from notebooks.utils.worlds import small_world_03
from exploring_agents.grid_world_agents import QExplorerAgent


if __name__ == "__main__":
    world = small_world_03
    sleep_time = 1
    rounds = 50
    agent = QExplorerAgent(
        reward_function=basic_reward,
        actions=basic_actions,
        gamma=1,
        alpha=0.7,
        epsilon=0.01,
    )

    states_history = []
    actions_history = []
    for _ in range(rounds):
        states, returns, actions = run_episode(agent, world)
        states_history.append(states)
        actions_history.append(actions)

    animate_episodes(
        world=world,
        states_history=states_history,
        actions_history=actions_history,
        sleep_time=1,
        episodes_to_animate=[10, 49],
    )
