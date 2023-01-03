import sys
import os

import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
base_path = "/" + os.path.join(*path.split("/")[:-2])
sys.path.append("/" + os.path.join(*path.split("/")[:-2]))

from exploring_agents import QAgent, RandomAgent
from exploring_agents.training import train_tag_agents
from notebooks.utils.basics import basic_reward, basic_actions, basic_running_reward
from notebooks.utils.worlds import tagging_world_01
from grid_world.action import GWorldAction

np.random.seed(50)


if __name__ == "__main__":
    prefix = f"{base_path}/persistence/agents/"
    filename1 = "q_vs_random_agent_1_tagging_world_01"
    filename2 = "random_vs_q_agent_2_tagging_world_01"
    gworld = tagging_world_01

    agent_1 = QAgent(
        reward_function=basic_reward,
        actions=basic_actions + [GWorldAction.wait],
        gamma=1,
        alpha=0.3,
        epsilon=0.01,
    )
    agent_2 = RandomAgent(
        reward_function=basic_running_reward,
        actions=basic_actions,
        gamma=1,
    )
    episode_lengths, agent_1_returns, agent_2_returns = train_tag_agents(
        agent_1=agent_1,
        agent_2=agent_2,
        world=gworld,
        episodes=int(1e4),
        episode_max_length=500,
    )
    agent_1.dump(f"{prefix}{filename1}")
    agent_2.dump(f"{prefix}{filename2}")
