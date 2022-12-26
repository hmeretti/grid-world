import sys
import os

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/" + os.path.join(*path.split("/")[:-3]))

from grid_world.visualization.curses_utils import animate_episodes
from notebooks.utils.basics import basic_reward, basic_actions
from notebooks.utils.worlds import small_world_03
from grid_world.agents.odp_agent import ODPAgent


if __name__ == "__main__":
    world = small_world_03
    sleep_time = 1
    rounds = 2
    show_episodes = [0, 1]
    agent = ODPAgent(
        reward_function=basic_reward,
        actions=basic_actions,
        world_shape=world.grid_shape,
        terminal_coordinates=world.terminal_states_coordinates[0]
    )

    states_history = []
    actions_history = []
    for _ in range(rounds):
        states, returns, actions = agent.run_episode(world)
        states_history.append(states)
        actions_history.append(actions)

    animate_episodes(
        world=world,
        states_history=states_history,
        actions_history=actions_history,
        sleep_time=1,
        episodes_to_animate=show_episodes,
    )
