from abstractions import Action, Agent, State, World
from utils.returns import returns_from_reward


def run_episode(
    agent: Agent, world: World, initial_state: State = None
) -> tuple[list[State], list[float], list[Action]]:
    state = initial_state if initial_state is not None else world.initial_state

    episode_states = []
    episode_rewards = []
    episode_actions = []

    # run through the world while updating q and the policy as we go
    effect = 0
    while effect != 1:
        action = agent.select_action(state)
        next_state, effect = world.take_action(state, action)

        reward = agent.run_update(state, action, effect, next_state)

        episode_states.append(state)
        episode_rewards.append(reward)
        episode_actions.append(action)

        state = next_state

    agent.finalize_episode()

    return episode_states, episode_rewards, episode_actions


def train_agent(
    agent: Agent,
    world: World,
    episodes: int = 100,
) -> tuple[list[int], list[float]]:
    episode_lengths = []
    episode_total_returns = []
    for episode in range(episodes):
        episode_states, episode_rewards, _ = run_episode(agent, world)
        episode_returns = returns_from_reward(episode_rewards, agent.gamma)
        episode_lengths.append(len(episode_states))
        episode_total_returns.append(episode_returns[0])

    return episode_lengths, episode_total_returns
