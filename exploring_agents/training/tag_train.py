from abstractions import Agent
from grid_world.action import GWorldAction
from grid_world.grid_world import GridWorld
from grid_world.state import TagState, GWorldState
from utils.operations import add_tuples
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
    state_1_t0 = initial_state_1 if initial_state_1 is not None else world.initial_state
    state_2_t0 = initial_state_2 if initial_state_2 is not None else world.initial_state_2

    agent_1_states = [state_1_t0]
    agent_1_rewards = []
    agent_1_actions = []

    agent_2_states = [state_2_t0]
    agent_2_rewards = []
    agent_2_actions = []

    # run through the world while updating q and the policy as we go
    effect_1 = 0
    effect_2 = 0
    next_action_1 = None
    t = 0

    # lets predefine this to make life easier
    initial_state_t0 = TagState(state_1_t0.coordinates, state_2_t0.coordinates)
    action_1_t0 = agent_1.select_action(initial_state_t0) if next_action_1 is None else next_action_1
    state_1_t1, effect_1_t0 = world.take_action(state_1_t0, action_1_t0)

    while effect_1 != 1 and effect_2 != 1:
        # agent 1 behaviour is predetermined
        # agent 2 taking action
        intermediate_state_t0 = TagState(state_1_t1.coordinates, state_2_t0.coordinates)
        action_2_t0 = agent_2.select_action(intermediate_state_t0)
        state_2_t1, effect_2_t0 = world.take_action(state_2_t0, action_2_t0)
        initial_state_t1 = TagState(state_1_t1.coordinates, state_2_t1.coordinates)

        # determine agent 1 reward
        # if agent 1 moves to agent 2 position or agent 2 moves to agent 1 position episode ends, 1 gets reward
        if state_1_t1 == state_2_t0 or state_1_t1 == state_2_t1:
            effect_1 = 1
        # if agent 2 survived long enough agent 1 gets punished
        elif t == episode_max_length - 1:
            effect_1 = -1
        else:
            effect_1 = effect_1_t0

        # update agent 1
        reward_1 = agent_1.run_update(initial_state_t0, action_1_t0, effect_1, initial_state_t1)

        # we need to know what state agent 2 will have at t1 in order to update his function
        # for this we need to already determine what agent 1 will do at t1
        action_1_t1 = agent_1.select_action(initial_state_t1)
        state_1_t2, effect_1_t1 = world.take_action(state_1_t1, action_1_t1)
        intermediate_state_t1 = TagState(state_1_t2, state_2_t1)

        # determine agent 2 reward
        # if agent 2 moved to agent 1 position or agent 1 manged to move to agent 2 position, agent 2 gets punished
        if state_1_t1 == state_2_t1 or state_1_t2 == state_2_t1:
            effect_2 = -1
        # if agent 2 survived long enough gets rewarded
        elif t == episode_max_length - 1:
            effect_1 = 1
        else:
            effect_2 = effect_2_t0

        reward_2 = agent_2.run_update(intermediate_state_t0, action_2_t0, effect_2, intermediate_state_t1)

        # store results
        agent_1_states.append(state_1_t1)
        agent_1_rewards.append(reward_1)
        agent_1_actions.append(action_1_t0)

        agent_2_states.append(state_2_t1)
        agent_2_rewards.append(reward_2)
        agent_2_actions.append(action_2_t0)

        # update states and preselected actions for next step
        state_1_t0 = state_1_t1
        state_2_t0 = state_2_t1
        initial_state_t0 = TagState(state_1_t0.coordinates, state_2_t0.coordinates)
        action_1_t0 = action_1_t1
        state_1_t1 = state_1_t2
        t += 1

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
