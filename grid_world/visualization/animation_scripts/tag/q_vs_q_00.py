import sys
import os

import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
base_path = "/" + os.path.join(*path.split("/")[:-4])
sys.path.append("/" + os.path.join(*path.split("/")[:-4]))

from exploring_agents import QAgent
from exploring_agents.training import run_tag_episode

from grid_world.visualization.curses_utils import animate_tag_episode
from notebooks.utils.worlds import tagging_world_00

if __name__ == "__main__":
    np.random.seed(50)

    world = tagging_world_00
    sleep_time = 1

    prefix = f"{base_path}/persistence/agents/"
    filename1 = "q_agent_1_tagging_world_00"
    filename2 = "q_agent_2_tagging_world_00"

    agent_1 = QAgent.load(f"{prefix}{filename1}")
    agent_2 = QAgent.load(f"{prefix}{filename2}")

    (
        agent_1_states,
        agent_1_returns,
        agent_1_actions,
        agent_2_states,
        agent_2_returns,
        agent_2_actions,
    ) = run_tag_episode(agent_1, agent_2, world, episode_max_length=100)

    animate_tag_episode(
        world=world,
        states_1=agent_1_states,
        actions_1=agent_1_actions,
        states_2=agent_2_states,
        actions_2=agent_2_actions,
        sleep_time=1,
    )
