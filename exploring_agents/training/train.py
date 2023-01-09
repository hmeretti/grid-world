from abstractions import Agent, World
from abstractions.type_vars import StateTypeVar, ActionTypeVar
from utils.returns import returns_from_reward


def run_episode(
    agent: Agent[ActionTypeVar, StateTypeVar], world: World, initial_state: StateTypeVar = None
) -> tuple[list[StateTypeVar], list[float], list[StateTypeVar]]:
    """
    Runs an episode for a generic task. Can be used for the maze problem.

    :param agent: an agent trying to solve the problem
    :param world: the world where this will take place
    :param initial_state: state where the agent starts
    :return: respectively: episode_states, episode_returns, episode_actions
    """
    state = initial_state if initial_state is not None else world.initial_state

    episode_states = [state]
    episode_rewards = []
    episode_actions = []

    # run through the world while updating q and the policy as we go
    effect = 0
    while effect != 1:
        action = agent.select_action(state)
        next_state, effect = world.take_action(state, action)

        reward = agent.run_update(state, action, effect, next_state)

        state = next_state

        episode_states.append(state)
        episode_rewards.append(reward)
        episode_actions.append(action)

    episode_returns = returns_from_reward(episode_rewards, agent.gamma)
    agent.finalize_episode(episode_states, episode_returns, episode_actions)

    return episode_states, episode_returns, episode_actions


def train_agent(
    agent: Agent,
    world: World,
    episodes: int = 100,
) -> tuple[list[int], list[float]]:
    """
    Trains an agent in a generic task. Can be used for the maze problem.

    :param agent: an agent trying to solve the problem
    :param world: the world where this will take place
    :param episodes: how many episodes to run
    :return: respectively: episode_lengths, episode_total_returns
    """
    episode_lengths = []
    episode_total_returns = []
    for episode in range(episodes):
        episode_states, episode_returns, _ = run_episode(agent, world)
        episode_lengths.append(len(episode_states))
        episode_total_returns.append(episode_returns[0])

    return episode_lengths, episode_total_returns
