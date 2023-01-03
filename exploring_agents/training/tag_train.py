from abstractions import Agent
from grid_world.action import GWorldAction
from grid_world.grid_world import GridWorld
from grid_world.state import TagState, GWorldState
from utils.returns import returns_from_reward


def run_tag_episode(
    agent_1: Agent,
    agent_2: Agent,
    world: GridWorld,
    initial_state_1: GWorldState = None,
    initial_state_2: GWorldState = None,
    episode_max_length=100,
) -> tuple[
    list[GWorldState],
    list[float],
    list[GWorldAction],
    list[GWorldState],
    list[float],
    list[GWorldAction],
]:
    state_1 = initial_state_1 if initial_state_1 is not None else world.initial_state
    state_2 = initial_state_2 if initial_state_2 is not None else world.initial_state_2

    agent_1_states = [state_1]
    agent_1_rewards = []
    agent_1_actions = []

    agent_2_states = [state_2]
    agent_2_rewards = []
    agent_2_actions = []

    # run through the world while updating q and the policy as we go
    effect_1 = 0
    effect_2 = 0
    t = 0
    while effect_1 != 1 and effect_2 != 1:
        # agent 1 taking action
        state = TagState(state_1.coordinates, state_2.coordinates)
        action_1 = agent_1.select_action(state)
        next_state_1, effect_1 = world.take_action(state_1, action_1)
        if next_state_1 == state_2:
            effect_1 = 1

        next_state = TagState(next_state_1.coordinates, state_2.coordinates)
        reward_1 = agent_1.run_update(state, action_1, effect_1, next_state)

        # agent 2 taking action
        action_2 = agent_2.select_action(state)
        next_state_2, effect_2 = world.take_action(state_2, action_2)
        if t == episode_max_length - 1:
            effect_2 = 1

        final_state = TagState(next_state_1.coordinates, next_state_2.coordinates)
        reward_2 = agent_2.run_update(state, action_2, effect_2, final_state)

        state_1 = next_state_1
        state_2 = next_state_2
        t += 1

        agent_1_states.append(state_1)
        agent_1_rewards.append(reward_1)
        agent_1_actions.append(action_1)

        agent_2_states.append(state_2)
        agent_2_rewards.append(reward_2)
        agent_2_actions.append(action_2)

    agent_1_returns = returns_from_reward(agent_1_rewards, agent_1.gamma)
    agent_2_returns = returns_from_reward(agent_2_rewards, agent_2.gamma)
    agent_1.finalize_episode(agent_1_states, agent_1_returns, agent_1_actions)
    agent_2.finalize_episode(agent_2_states, agent_2_returns, agent_2_actions)

    return (
        agent_1_states,
        agent_1_returns,
        agent_1_actions,
        agent_2_states,
        agent_2_returns,
        agent_2_actions,
    )


def train_tag_agents(
    agent_1: Agent,
    agent_2: Agent,
    world: GridWorld,
    episodes: int = 1000,
    episode_max_length: int = 1000,
) -> tuple[list[int], list[float], list[float]]:
    episode_lengths = []
    agent_1_total_returns = []
    agent_2_total_returns = []
    for episode in range(episodes):
        agent_1_states, agent_1_returns, _, _, agent_2_returns, _ = run_tag_episode(
            agent_1=agent_1,
            agent_2=agent_2,
            world=world,
            episode_max_length=episode_max_length,
        )
        episode_lengths.append(len(agent_1_states))
        agent_1_total_returns.append(agent_1_returns[0])
        agent_2_total_returns.append(agent_2_returns[0])

    return episode_lengths, agent_1_total_returns, agent_2_total_returns
