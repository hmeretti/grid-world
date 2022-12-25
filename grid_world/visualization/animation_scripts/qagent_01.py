import sys
import os

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/"+os.path.join(*path.split('/')[:-3]))

from grid_world.visualization.curses_utils import animate_episodes
from grid_world.agents.q_agent import QAgent
from notebooks.utils.basics import basic_reward, basic_actions
from notebooks.utils.worlds import small_world_03


if __name__ == "__main__":
    world = small_world_03
    sleep_time = 1
    rounds = 50
    agent = QAgent(
        reward_function=basic_reward,
        actions=basic_actions,
        gamma=1,
        alpha=0.7,
        epsilon=0.01,
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
        episodes_to_animate=[5, 49],
    )